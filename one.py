from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import torch

from plot_style import apply_readable_plot_style, style_axes, style_colorbar
from pollution_core import (
    DEFAULT_SUBDOMAINS,
    ControllerDynamics,
    GridParams,
    PhysicalParams,
    build_source_map_np,
    enforce_boundary_np,
    enforce_boundary_torch,
    initial_condition,
    mass_centers_by_region,
    max_points_by_region,
    simulate_pde_step_np,
    source_map_torch,
)
from stateful_surrogate import load_checkpoint, rollout_surrogate


apply_readable_plot_style()


LINE_LEGEND_FONT_SIZE = 14
FORCE_CONTROL_FONT_SIZE = 20
AVERAGE_COMPARISON_FONT_SIZE = 20
TRAJECTORY_FONT_SIZE = 20
STATE_PANEL_TITLE_SIZE = 28
STATE_AXIS_LABEL_SIZE = 24
STATE_TICK_LABEL_SIZE = 20
STATE_COLORBAR_LABEL_SIZE = 20
STATE_COLORBAR_TICK_SIZE = 22


def set_legend_font_size(axes, font_size: int) -> None:
    if hasattr(axes, "flat"):
        axis_iter = axes.flat
    elif isinstance(axes, (list, tuple)):
        axis_iter = axes
    else:
        axis_iter = (axes,)
    for axis in axis_iter:
        legend = axis.get_legend()
        if legend is None:
            continue
        for text in legend.get_texts():
            text.set_fontsize(font_size)


def set_tick_font_size(axes, font_size: int) -> None:
    if hasattr(axes, "flat"):
        axis_iter = axes.flat
    elif isinstance(axes, (list, tuple)):
        axis_iter = axes
    else:
        axis_iter = (axes,)
    for axis in axis_iter:
        axis.tick_params(axis="both", which="major", labelsize=font_size)
        axis.tick_params(axis="both", which="minor", labelsize=max(font_size - 2, 8))


MIN_CONTROL_SIGMA_M = 85.0
MIN_CONTROL_U_MAX = 20.0
CONTROL_DELTA_MAX = 5.0
FORCE_DELTA_MAX = 30.0
CONTROL_PLOT_SCALE = 10.0
FIELD_PLOT_SCALE = 76.0
AVERAGE_PLOT_SCALE = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop MPC using the state-aware surrogate.")
    parser.add_argument("--model-path", default="state_aware_pinn_best.pth")
    parser.add_argument("--output-dir", default="mpc_visualization_results_v4")
    parser.add_argument("--prediction-mode", choices=("surrogate", "pde"), default="surrogate")
    parser.add_argument(
        "--controller-mode",
        choices=("mobile", "fixed", "compare"),
        default="mobile",
        help="Run mobile controllers, fixed controllers, or both for comparison.",
    )
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--control-steps", type=int, default=75)
    parser.add_argument("--control-stride", type=int, default=4)
    parser.add_argument("--surrogate-iters", type=int, default=10)
    parser.add_argument("--surrogate-lr", type=float, default=0.18)
    parser.add_argument("--target-mean", type=float, default=0.0001)
    parser.add_argument("--target-max", type=float, default=0.0001)
    parser.add_argument("--min-control-steps", type=int, default=20)
    return parser.parse_args()


def project_to_subdomains(
    positions: np.ndarray,
    subdomains: Tuple[Tuple[float, float, float, float], ...] = DEFAULT_SUBDOMAINS,
    margin: float = 0.02,
) -> np.ndarray:
    projected = positions.copy()
    for ctrl_idx, (x0, x1, y0, y1) in enumerate(subdomains):
        projected[ctrl_idx, 0] = np.clip(projected[ctrl_idx, 0], x0 + margin, x1 - margin)
        projected[ctrl_idx, 1] = np.clip(projected[ctrl_idx, 1], y0 + margin, y1 - margin)
    return projected.astype(np.float32)


def rollout_dynamics(
    dynamics: ControllerDynamics,
    positions: np.ndarray,
    velocities: np.ndarray,
    force_seq: np.ndarray,
    subdomains: Tuple[Tuple[float, float, float, float], ...] = DEFAULT_SUBDOMAINS,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_state = positions.astype(np.float32, copy=True)
    vel_state = velocities.astype(np.float32, copy=True)
    pos_hist = np.zeros((force_seq.shape[0], positions.shape[0], 2), dtype=np.float32)
    vel_hist = np.zeros_like(pos_hist)
    for step_idx in range(force_seq.shape[0]):
        pos_state, vel_state = dynamics.step(pos_state, vel_state, force_seq[step_idx])
        pos_state = enforce_separation_np(pos_state, dynamics.d_min_sep, subdomains=subdomains)
        pos_hist[step_idx] = pos_state
        vel_hist[step_idx] = vel_state
    return pos_hist, vel_hist


def predict_closed_loop_states(
    current_field: np.ndarray,
    positions_seq: np.ndarray,
    controls_seq: np.ndarray,
    control_stride: int,
    grid: GridParams,
    phys: PhysicalParams,
    prediction_mode: str,
    surrogate_model=None,
) -> np.ndarray:
    state = current_field.astype(np.float32, copy=True)
    predicted_states = []

    for step_idx in range(controls_seq.shape[0]):
        source_map = build_source_map_np(positions_seq[step_idx], controls_seq[step_idx], grid, phys)
        if prediction_mode == "surrogate":
            repeated_sources = np.repeat(source_map[None, ...], control_stride, axis=0)
            rollout = rollout_surrogate(surrogate_model, state, repeated_sources, phys=phys)
            state = rollout[-1]
        else:
            for _ in range(control_stride):
                state = simulate_pde_step_np(state, source_map, grid, phys)
        predicted_states.append(state)

    return np.stack(predicted_states, axis=0)


def collision_penalty(positions_seq: np.ndarray, d_min_sep: Optional[float]) -> float:
    if d_min_sep is None:
        return 0.0
    penalty = 0.0
    for step_idx in range(positions_seq.shape[0]):
        for left in range(positions_seq.shape[1]):
            for right in range(left + 1, positions_seq.shape[1]):
                distance = float(np.linalg.norm(positions_seq[step_idx, left] - positions_seq[step_idx, right]))
                if distance < d_min_sep:
                    penalty += 1e4 * (d_min_sep - distance) ** 2
    return penalty


def enforce_separation_np(
    positions: np.ndarray,
    d_min_sep: Optional[float],
    subdomains: Tuple[Tuple[float, float, float, float], ...] = DEFAULT_SUBDOMAINS,
    margin: float = 0.02,
    iterations: int = 8,
) -> np.ndarray:
    if d_min_sep is None:
        return project_to_subdomains(positions, subdomains=subdomains, margin=margin)
    adjusted = project_to_subdomains(positions, subdomains=subdomains, margin=margin)
    for _ in range(iterations):
        changed = False
        for left in range(len(adjusted)):
            for right in range(left + 1, len(adjusted)):
                delta = adjusted[right] - adjusted[left]
                distance = float(np.linalg.norm(delta))
                if distance >= float(d_min_sep):
                    continue
                if distance < 1e-6:
                    direction = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    direction = delta / distance
                correction = 0.5 * (float(d_min_sep) - max(distance, 1e-6)) * direction
                adjusted[left] -= correction
                adjusted[right] += correction
                adjusted = project_to_subdomains(adjusted, subdomains=subdomains, margin=margin)
                changed = True
        if not changed:
            break
    return adjusted.astype(np.float32)


def subdomain_masses(field: np.ndarray, grid: GridParams) -> np.ndarray:
    x = np.linspace(0.0, 1.0, grid.Nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, grid.Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    masses = np.zeros(len(DEFAULT_SUBDOMAINS), dtype=np.float32)
    for idx, (x0, x1, y0, y1) in enumerate(DEFAULT_SUBDOMAINS):
        mask = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        masses[idx] = float(field[mask].sum())
    return masses


def heuristic_force_guess(
    positions: np.ndarray,
    velocities: np.ndarray,
    targets: np.ndarray,
    force_limit: float,
) -> np.ndarray:
    proportional_gain = 24.0
    damping_gain = 4.5
    forces = proportional_gain * (targets - positions) - damping_gain * velocities
    return np.clip(forces, -force_limit, force_limit).astype(np.float32)


def heuristic_control_guess(
    positions: np.ndarray,
    targets: np.ndarray,
    mass_weights: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(targets - positions, axis=1)
    proximity = np.exp(-(distances ** 2) / (2.0 * 0.14 ** 2))
    desired_strength = 0.62 + 1.05 * mass_weights * proximity
    return -np.clip(desired_strength, 0.0, 0.98).astype(np.float32)


def shift_warm_start(sequence: np.ndarray) -> np.ndarray:
    shifted = np.empty_like(sequence)
    shifted[:-1] = sequence[1:]
    shifted[-1] = sequence[-1]
    return shifted


def control_delta_limit_coeff(phys: PhysicalParams) -> float:
    return CONTROL_DELTA_MAX / max(float(phys.u_max), 1e-6)


def project_control_sequence_np(
    previous_controls: np.ndarray,
    controls_seq: np.ndarray,
    phys: PhysicalParams,
) -> np.ndarray:
    delta_limit = control_delta_limit_coeff(phys)
    projected = np.zeros_like(controls_seq, dtype=np.float32)
    previous = previous_controls.astype(np.float32, copy=True)
    for step_idx in range(controls_seq.shape[0]):
        lower = np.maximum(-1.0, previous - delta_limit)
        upper = np.minimum(0.0, previous + delta_limit)
        current = np.clip(controls_seq[step_idx], lower, upper)
        projected[step_idx] = current.astype(np.float32)
        previous = projected[step_idx]
    return projected


def project_force_sequence_np(
    previous_forces: np.ndarray,
    forces_seq: np.ndarray,
    force_limit: float,
) -> np.ndarray:
    projected = np.zeros_like(forces_seq, dtype=np.float32)
    previous = previous_forces.astype(np.float32, copy=True)
    for step_idx in range(forces_seq.shape[0]):
        lower = np.maximum(-force_limit, previous - FORCE_DELTA_MAX)
        upper = np.minimum(force_limit, previous + FORCE_DELTA_MAX)
        current = np.clip(forces_seq[step_idx], lower, upper)
        projected[step_idx] = current.astype(np.float32)
        previous = projected[step_idx]
    return projected


def inverse_sigmoid_from_controls(controls: np.ndarray) -> np.ndarray:
    scaled = np.clip(-controls, 1e-4, 1.0 - 1e-4)
    return np.log(scaled / (1.0 - scaled)).astype(np.float32)


def inverse_tanh_scaled(values: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.clip(values / scale, -0.999, 0.999)
    return np.arctanh(scaled).astype(np.float32)


def delta_logits_from_controls(
    previous_controls: np.ndarray,
    controls_seq: np.ndarray,
    phys: PhysicalParams,
) -> np.ndarray:
    delta_limit = control_delta_limit_coeff(phys)
    previous = previous_controls.astype(np.float32, copy=True)
    deltas = np.zeros_like(controls_seq, dtype=np.float32)
    for step_idx in range(controls_seq.shape[0]):
        deltas[step_idx] = np.clip(controls_seq[step_idx] - previous, -delta_limit, delta_limit)
        previous = controls_seq[step_idx]
    return inverse_tanh_scaled(deltas, delta_limit)


def delta_logits_from_forces(
    previous_forces: np.ndarray,
    forces_seq: np.ndarray,
) -> np.ndarray:
    previous = previous_forces.astype(np.float32, copy=True)
    deltas = np.zeros_like(forces_seq, dtype=np.float32)
    for step_idx in range(forces_seq.shape[0]):
        deltas[step_idx] = np.clip(forces_seq[step_idx] - previous, -FORCE_DELTA_MAX, FORCE_DELTA_MAX)
        previous = forces_seq[step_idx]
    return inverse_tanh_scaled(deltas, FORCE_DELTA_MAX)


def project_to_subdomains_torch(
    positions: torch.Tensor,
    subdomains: Tuple[Tuple[float, float, float, float], ...] = DEFAULT_SUBDOMAINS,
    margin: float = 0.02,
) -> torch.Tensor:
    projected_points = []
    for ctrl_idx, (x0, x1, y0, y1) in enumerate(subdomains):
        clamped_x = torch.clamp(positions[ctrl_idx, 0], min=x0 + margin, max=x1 - margin)
        clamped_y = torch.clamp(positions[ctrl_idx, 1], min=y0 + margin, max=y1 - margin)
        projected_points.append(torch.stack([clamped_x, clamped_y]))
    return torch.stack(projected_points, dim=0)


def enforce_separation_torch(
    positions: torch.Tensor,
    d_min_sep: Optional[float],
    subdomains: Tuple[Tuple[float, float, float, float], ...] = DEFAULT_SUBDOMAINS,
    margin: float = 0.02,
    iterations: int = 8,
) -> torch.Tensor:
    projected = project_to_subdomains_torch(positions, subdomains=subdomains, margin=margin)
    if d_min_sep is None:
        return projected
    safe_distance = torch.tensor(float(d_min_sep), device=projected.device, dtype=projected.dtype)
    for _ in range(iterations):
        changed = False
        for left in range(projected.shape[0]):
            for right in range(left + 1, projected.shape[0]):
                delta = projected[right] - projected[left]
                distance = torch.linalg.norm(delta)
                deficit = torch.clamp(safe_distance - distance, min=0.0)
                if float(deficit.detach().cpu().item()) <= 1e-7:
                    continue
                direction = delta / distance.clamp_min(1e-6)
                correction = 0.5 * deficit * direction
                updated = projected.clone()
                updated[left] = projected[left] - correction
                updated[right] = projected[right] + correction
                projected = project_to_subdomains_torch(updated, subdomains=subdomains, margin=margin)
                changed = True
        if not changed:
            break
    return projected


def rollout_dynamics_torch(
    dynamics: ControllerDynamics,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    force_seq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pos_state = positions
    vel_state = velocities
    pos_hist = []
    vel_hist = []

    for step_idx in range(force_seq.shape[0]):
        applied_force = torch.clamp(force_seq[step_idx], min=-dynamics.force_limit, max=dynamics.force_limit)
        acceleration = dynamics.acceleration_scale * (applied_force / dynamics.mass) - dynamics.damping_scale * vel_state
        vel_state = vel_state + dynamics.dt * acceleration
        if dynamics.velocity_limit is not None:
            vel_state = torch.clamp(vel_state, min=-dynamics.velocity_limit, max=dynamics.velocity_limit)
        pos_state = enforce_separation_torch(pos_state + dynamics.dt * vel_state, dynamics.d_min_sep)
        pos_hist.append(pos_state)
        vel_hist.append(vel_state)

    return torch.stack(pos_hist, dim=0), torch.stack(vel_hist, dim=0)


def collision_penalty_torch(positions_seq: torch.Tensor, d_min_sep: Optional[float]) -> torch.Tensor:
    if d_min_sep is None:
        return torch.tensor(0.0, device=positions_seq.device)
    penalty = torch.tensor(0.0, device=positions_seq.device)
    for step_idx in range(positions_seq.shape[0]):
        for left in range(positions_seq.shape[1]):
            for right in range(left + 1, positions_seq.shape[1]):
                distance = torch.linalg.norm(positions_seq[step_idx, left] - positions_seq[step_idx, right])
                deficit = torch.clamp(torch.tensor(d_min_sep, device=positions_seq.device) - distance, min=0.0)
                penalty = penalty + 1e4 * deficit.square()
    return penalty


def mpc_optimize_slsqp(
    current_field: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    previous_controls: np.ndarray,
    previous_forces: np.ndarray,
    dynamics: ControllerDynamics,
    grid: GridParams,
    phys: PhysicalParams,
    horizon: int,
    max_iter: int,
    control_stride: int,
    prediction_mode: str,
    surrogate_model=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    num_ctrl = phys.num_controllers
    mass_centers = mass_centers_by_region(current_field, grid)
    region_peaks = max_points_by_region(current_field, grid)
    tracking_targets = 0.65 * mass_centers + 0.35 * region_peaks
    region_mass = subdomain_masses(current_field, grid)
    mass_weights = region_mass / max(float(region_mass.max()), 1e-6)

    def flatten(controls_seq: np.ndarray, forces_seq: np.ndarray) -> np.ndarray:
        return np.concatenate([controls_seq.ravel(), forces_seq.ravel()])

    def unflatten(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        controls_size = horizon * num_ctrl
        controls_seq = values[:controls_size].reshape(horizon, num_ctrl)
        forces_seq = values[controls_size:].reshape(horizon, num_ctrl, 2)
        return controls_seq, forces_seq

    def objective(values: np.ndarray) -> float:
        controls_seq, forces_seq = unflatten(values)
        controls_seq = project_control_sequence_np(previous_controls, controls_seq, phys)
        forces_seq = project_force_sequence_np(previous_forces, forces_seq, dynamics.force_limit)
        predicted_positions, predicted_velocities = rollout_dynamics(dynamics, positions, velocities, forces_seq)
        predicted_states = predict_closed_loop_states(
            current_field=current_field,
            positions_seq=predicted_positions,
            controls_seq=controls_seq,
            control_stride=control_stride,
            grid=grid,
            phys=phys,
            prediction_mode=prediction_mode,
            surrogate_model=surrogate_model,
        )

        state_cost = sum(float(np.mean(state ** 2)) for state in predicted_states[:-1])
        terminal_cost = float(np.mean(predicted_states[-1] ** 2))
        control_cost = float(np.mean(np.abs(controls_seq)) + 0.5 * np.mean(controls_seq ** 2))
        force_cost = float(np.mean(forces_seq ** 2))

        target_cost = 0.0
        terminal_target_cost = float(np.mean((predicted_positions[-1] - tracking_targets) ** 2))
        peak_cost = 0.0
        distance_weighted_control_cost = 0.0
        source_overlap_reward = 0.0
        for step_idx in range(horizon):
            dist_sq_to_targets = np.sum((predicted_positions[step_idx] - tracking_targets) ** 2, axis=1)
            target_cost += float(np.mean(dist_sq_to_targets))
            peak_cost += float(np.mean((predicted_positions[step_idx] - region_peaks) ** 2))
            dist_to_targets = np.sqrt(np.maximum(dist_sq_to_targets, 1e-10))
            distance_scale = np.clip(dist_to_targets / 0.18, 0.0, 1.5)
            distance_weighted_control_cost += float(
                np.mean((controls_seq[step_idx] ** 2) * (1.0 + 5.0 * distance_scale) * (0.5 + 0.5 * mass_weights))
            )
            source_map = build_source_map_np(predicted_positions[step_idx], controls_seq[step_idx], grid, phys)
            source_overlap_reward += float(np.mean((-source_map) * predicted_states[step_idx]))

        return (
            65.0 * state_cost
            + 115.0 * terminal_cost
            + 38.0 * target_cost
            + 95.0 * terminal_target_cost
            + 16.0 * peak_cost
            + 0.10 * control_cost
            + 0.45 * distance_weighted_control_cost
            + 0.02 * force_cost
            - 30.0 * source_overlap_reward
        )

    def delta_constraints(values: np.ndarray) -> np.ndarray:
        controls_seq, forces_seq = unflatten(values)
        control_actual = controls_seq * phys.u_max
        previous_control_actual = previous_controls * phys.u_max
        previous_control_seq = np.concatenate([previous_control_actual[None, :], control_actual[:-1]], axis=0)
        control_delta = control_actual - previous_control_seq

        previous_force_seq = np.concatenate([previous_forces[None, :, :], forces_seq[:-1]], axis=0)
        force_delta = forces_seq - previous_force_seq

        return np.concatenate(
            [
                (CONTROL_DELTA_MAX - control_delta).ravel(),
                (CONTROL_DELTA_MAX + control_delta).ravel(),
                (FORCE_DELTA_MAX - force_delta).ravel(),
                (FORCE_DELTA_MAX + force_delta).ravel(),
            ]
        ).astype(np.float64)

    def collision_constraints(values: np.ndarray) -> np.ndarray:
        _, forces_seq = unflatten(values)
        forces_seq = project_force_sequence_np(previous_forces, forces_seq, dynamics.force_limit)
        predicted_positions, _ = rollout_dynamics(dynamics, positions, velocities, forces_seq)
        if dynamics.d_min_sep is None:
            return np.array([1.0], dtype=np.float64)
        margins = []
        for step_idx in range(predicted_positions.shape[0]):
            for left in range(predicted_positions.shape[1]):
                for right in range(left + 1, predicted_positions.shape[1]):
                    distance = float(np.linalg.norm(predicted_positions[step_idx, left] - predicted_positions[step_idx, right]))
                    margins.append(distance - float(dynamics.d_min_sep))
        return np.array(margins, dtype=np.float64)

    heuristic_forces = heuristic_force_guess(positions, velocities, tracking_targets, dynamics.force_limit)
    heuristic_controls = heuristic_control_guess(positions, tracking_targets, mass_weights)
    initial_controls = np.tile(0.6 * previous_controls + 0.4 * heuristic_controls, (horizon, 1)).astype(np.float32)
    initial_forces = np.tile(0.6 * previous_forces[None, :, :] + 0.4 * heuristic_forces[None, :, :], (horizon, 1, 1)).astype(np.float32)
    initial_controls = project_control_sequence_np(previous_controls, initial_controls, phys)
    initial_forces = project_force_sequence_np(previous_forces, initial_forces, dynamics.force_limit)
    initial_values = flatten(initial_controls, initial_forces)

    bounds = [(-1.0, 0.0) for _ in range(horizon * num_ctrl)]
    bounds.extend([(-dynamics.force_limit, dynamics.force_limit) for _ in range(horizon * num_ctrl * 2)])

    result = optimize.minimize(
        objective,
        initial_values,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "ineq", "fun": delta_constraints},
            {"type": "ineq", "fun": collision_constraints},
        ],
        options={"maxiter": max_iter, "disp": False, "ftol": 1e-3, "eps": 1e-3},
    )

    controls_seq, forces_seq = unflatten(result.x)
    controls_seq = project_control_sequence_np(previous_controls, controls_seq, phys)
    forces_seq = project_force_sequence_np(previous_forces, forces_seq, dynamics.force_limit)
    predicted_positions, _ = rollout_dynamics(dynamics, positions, velocities, forces_seq)
    return controls_seq.astype(np.float32), forces_seq.astype(np.float32), predicted_positions, float(objective(result.x))


def mpc_optimize_surrogate(
    current_field: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    previous_controls: np.ndarray,
    previous_forces: np.ndarray,
    dynamics: ControllerDynamics,
    grid: GridParams,
    phys: PhysicalParams,
    horizon: int,
    control_stride: int,
    surrogate_model,
    surrogate_iters: int,
    surrogate_lr: float,
    warm_start_controls: Optional[np.ndarray] = None,
    warm_start_forces: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    device = next(surrogate_model.parameters()).device
    num_ctrl = phys.num_controllers

    mass_centers = mass_centers_by_region(current_field, grid)
    region_peaks = max_points_by_region(current_field, grid)
    tracking_targets = 0.65 * mass_centers + 0.35 * region_peaks
    region_mass = subdomain_masses(current_field, grid)
    mass_weights = region_mass / max(float(region_mass.max()), 1e-6)

    heuristic_forces = heuristic_force_guess(positions, velocities, tracking_targets, dynamics.force_limit)
    heuristic_controls = heuristic_control_guess(positions, tracking_targets, mass_weights)

    if warm_start_controls is not None and warm_start_controls.shape == (horizon, num_ctrl):
        init_controls = np.clip(0.75 * warm_start_controls + 0.25 * heuristic_controls[None, :], -1.0, 0.0)
    else:
        init_controls = np.tile(0.6 * previous_controls + 0.4 * heuristic_controls, (horizon, 1)).astype(np.float32)

    if warm_start_forces is not None and warm_start_forces.shape == (horizon, num_ctrl, 2):
        init_forces = np.clip(0.75 * warm_start_forces + 0.25 * heuristic_forces[None, :, :], -dynamics.force_limit, dynamics.force_limit)
    else:
        init_forces = np.tile(0.6 * previous_forces[None, :, :] + 0.4 * heuristic_forces[None, :, :], (horizon, 1, 1)).astype(np.float32)

    init_controls = project_control_sequence_np(previous_controls, init_controls, phys)
    init_forces = project_force_sequence_np(previous_forces, init_forces, dynamics.force_limit)

    control_delta_logits = torch.nn.Parameter(
        torch.from_numpy(delta_logits_from_controls(previous_controls, init_controls, phys)).to(device=device, dtype=torch.float32)
    )
    force_logits = torch.nn.Parameter(
        torch.from_numpy(delta_logits_from_forces(previous_forces, init_forces)).to(device=device, dtype=torch.float32)
    )

    optimizer = torch.optim.Adam([control_delta_logits, force_logits], lr=surrogate_lr)
    current_field_tensor = torch.from_numpy(current_field).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    initial_positions_tensor = torch.from_numpy(positions).to(device=device, dtype=torch.float32)
    initial_velocities_tensor = torch.from_numpy(velocities).to(device=device, dtype=torch.float32)
    previous_controls_tensor = torch.from_numpy(previous_controls).to(device=device, dtype=torch.float32)
    previous_forces_tensor = torch.from_numpy(previous_forces).to(device=device, dtype=torch.float32)
    tracking_targets_tensor = torch.from_numpy(tracking_targets).to(device=device, dtype=torch.float32)
    region_peaks_tensor = torch.from_numpy(region_peaks).to(device=device, dtype=torch.float32)
    mass_weights_tensor = torch.from_numpy(mass_weights).to(device=device, dtype=torch.float32)
    coord_channels = surrogate_model.coord_channels.expand(horizon, -1, -1, -1)

    best_loss_value = float("inf")
    best_controls = init_controls.copy()
    best_forces = init_forces.copy()
    best_positions = np.tile(positions[None, :, :], (horizon, 1, 1)).astype(np.float32)

    for _ in range(surrogate_iters):
        optimizer.zero_grad(set_to_none=True)

        control_delta_limit = control_delta_limit_coeff(phys)
        controls_steps = []
        previous_control_step = previous_controls_tensor
        for step_idx in range(horizon):
            control_delta = control_delta_limit * torch.tanh(control_delta_logits[step_idx])
            current_control = torch.clamp(previous_control_step + control_delta, min=-1.0, max=0.0)
            controls_steps.append(current_control)
            previous_control_step = current_control
        controls_seq = torch.stack(controls_steps, dim=0)

        force_steps = []
        previous_force_step = previous_forces_tensor
        for step_idx in range(horizon):
            force_delta = FORCE_DELTA_MAX * torch.tanh(force_logits[step_idx])
            current_force = torch.clamp(
                previous_force_step + force_delta,
                min=-dynamics.force_limit,
                max=dynamics.force_limit,
            )
            force_steps.append(current_force)
            previous_force_step = current_force
        forces_seq = torch.stack(force_steps, dim=0)

        predicted_positions, predicted_velocities = rollout_dynamics_torch(
            dynamics,
            initial_positions_tensor,
            initial_velocities_tensor,
            forces_seq,
        )
        source_seq = source_map_torch(predicted_positions, controls_seq, coord_channels, phys)

        state = current_field_tensor
        predicted_states = []
        for step_idx in range(horizon):
            step_source = source_seq[step_idx : step_idx + 1]
            for _ in range(control_stride):
                state = enforce_boundary_torch(surrogate_model(state, step_source), phys.boundary)
            predicted_states.append(state)
        predicted_states_tensor = torch.cat(predicted_states, dim=0)

        state_cost = predicted_states_tensor[:-1].square().mean() if horizon > 1 else torch.tensor(0.0, device=device)
        terminal_cost = predicted_states_tensor[-1].square().mean()
        control_cost = controls_seq.abs().mean() + 0.5 * controls_seq.square().mean()
        force_cost = forces_seq.square().mean()

        target_cost = torch.tensor(0.0, device=device)
        peak_cost = torch.tensor(0.0, device=device)
        distance_weighted_control_cost = torch.tensor(0.0, device=device)
        source_overlap_reward = torch.tensor(0.0, device=device)
        for step_idx in range(horizon):
            dist_sq_to_targets = (predicted_positions[step_idx] - tracking_targets_tensor).square().sum(dim=1)
            target_cost = target_cost + dist_sq_to_targets.mean()
            peak_cost = peak_cost + (predicted_positions[step_idx] - region_peaks_tensor).square().mean()
            dist_to_targets = dist_sq_to_targets.clamp_min(1e-10).sqrt()
            distance_scale = torch.clamp(dist_to_targets / 0.18, min=0.0, max=1.5)
            distance_weighted_control_cost = distance_weighted_control_cost + (
                controls_seq[step_idx].square() * (1.0 + 5.0 * distance_scale) * (0.5 + 0.5 * mass_weights_tensor)
            ).mean()
            source_overlap_reward = source_overlap_reward + ((-source_seq[step_idx]) * predicted_states_tensor[step_idx]).mean()

        terminal_target_cost = (predicted_positions[-1] - tracking_targets_tensor).square().mean()
        total_loss = (
            65.0 * state_cost
            + 115.0 * terminal_cost
            + 38.0 * target_cost
            + 95.0 * terminal_target_cost
            + 16.0 * peak_cost
            + 0.10 * control_cost
            + 0.45 * distance_weighted_control_cost
            + 0.02 * force_cost
            - 30.0 * source_overlap_reward
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([control_delta_logits, force_logits], max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            loss_value = float(total_loss.item())
            if loss_value < best_loss_value:
                best_loss_value = loss_value
                best_controls = controls_seq.detach().cpu().numpy().astype(np.float32)
                best_forces = forces_seq.detach().cpu().numpy().astype(np.float32)
                best_positions = predicted_positions.detach().cpu().numpy().astype(np.float32)

    return best_controls, best_forces, best_positions, best_loss_value


def plot_results(
    output_dir: Path,
    states_record: np.ndarray,
    control_record: np.ndarray,
    force_record: np.ndarray,
    position_record: np.ndarray,
    initial_positions: np.ndarray,
    grid: GridParams,
    phys: PhysicalParams,
    control_stride: int,
    scenario_label: str = "Mobile Actuators",
    solve_time_record: Optional[np.ndarray] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    time_state = np.arange(states_record.shape[0]) * control_stride * grid.dt * phys.time_scale_s
    time_control = np.arange(control_record.shape[0]) * control_stride * grid.dt * phys.time_scale_s
    control_record_scaled = control_record * CONTROL_PLOT_SCALE
    states_record_scaled = states_record * FIELD_PLOT_SCALE

    avg_concentration = states_record.mean(axis=(1, 2)) * AVERAGE_PLOT_SCALE
    plt.figure(figsize=(10, 5))
    plt.plot(time_state, avg_concentration, linewidth=2, color="#1f77b4")
    plt.xlabel("Time")
    plt.ylabel("Average Concentration")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    style_axes(plt.gcf().axes)
    plt.savefig(output_dir / "mpc_2d_average_concentration.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ctrl_idx, axis in enumerate(axes.flatten()):
        axis.step(time_control, control_record_scaled[:, ctrl_idx], where="post", linewidth=1.8)
        axis.set_xlabel("Time")
        axis.set_ylabel(f"u_{ctrl_idx + 1}")
        axis.grid(alpha=0.3)
    plt.tight_layout()
    style_axes(axes)
    plt.savefig(output_dir / "mpc_2d_control_inputs.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    colors = ["#d62728", "#1f77b4", "#ffbf00", "#2ca02c"]
    for ctrl_idx in range(control_record.shape[1]):
        plt.step(
            time_control,
            control_record_scaled[:, ctrl_idx],
            where="post",
            linewidth=1.8,
            color=colors[ctrl_idx],
            label=f"Actuator {ctrl_idx + 1}",
        )
    plt.xlabel("Time", fontsize=FORCE_CONTROL_FONT_SIZE)
    plt.ylabel("Control Input", fontsize=FORCE_CONTROL_FONT_SIZE)
    plt.grid(alpha=0.3)
    plt.legend()
    combined_axes = plt.gcf().axes
    style_axes(combined_axes)
    set_tick_font_size(combined_axes, FORCE_CONTROL_FONT_SIZE)
    set_legend_font_size(combined_axes, FORCE_CONTROL_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_dir / "mpc_2d_control_inputs_combined.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 8))
    for x_split, y_split in ((0.5, None), (None, 0.5)):
        if x_split is not None:
            plt.axvline(x_split, color="gray", linestyle="--", alpha=0.6)
        if y_split is not None:
            plt.axhline(y_split, color="gray", linestyle="--", alpha=0.6)
    for ctrl_idx in range(position_record.shape[1]):
        plt.plot(position_record[:, ctrl_idx, 0], position_record[:, ctrl_idx, 1], linewidth=2, color=colors[ctrl_idx])
        plt.scatter(initial_positions[ctrl_idx, 0], initial_positions[ctrl_idx, 1], marker="s", s=90, color=colors[ctrl_idx])
        plt.scatter(position_record[-1, ctrl_idx, 0], position_record[-1, ctrl_idx, 1], marker="*", s=140, color=colors[ctrl_idx])
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("x", fontsize=TRAJECTORY_FONT_SIZE)
    plt.ylabel("y", fontsize=TRAJECTORY_FONT_SIZE)
    plt.grid(alpha=0.25)
    trajectory_axes = plt.gcf().axes
    style_axes(trajectory_axes)
    set_tick_font_size(trajectory_axes, TRAJECTORY_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_dir / "mpc_2d_controller_trajectories.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ctrl_idx, axis in enumerate(axes.flatten()):
        axis.step(time_control, force_record[:, ctrl_idx, 0], where="post", linewidth=1.6, label="Fx")
        axis.step(time_control, force_record[:, ctrl_idx, 1], where="post", linewidth=1.6, label="Fy")
        axis.set_xlabel("Time", fontsize=FORCE_CONTROL_FONT_SIZE)
        axis.set_ylabel("Force", fontsize=FORCE_CONTROL_FONT_SIZE)
        axis.grid(alpha=0.3)
        axis.legend()
    style_axes(axes)
    set_tick_font_size(axes, FORCE_CONTROL_FONT_SIZE)
    set_legend_font_size(axes, FORCE_CONTROL_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_dir / "mpc_2d_forces.png", dpi=200)
    plt.close()

    if solve_time_record is not None and solve_time_record.size > 0:
        time_steps = np.arange(1, solve_time_record.shape[0] + 1, dtype=np.int32)
        plt.figure(figsize=(10, 4.8))
        plt.plot(time_steps, solve_time_record * 1000.0, linewidth=1.8, color="#9467bd")
        plt.xlabel("Time")
        plt.ylabel("Solve time (ms)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        style_axes(plt.gcf().axes)
        plt.savefig(output_dir / "mpc_2d_solve_time_per_step.png", dpi=200)
        plt.close()

    snapshot_indices = np.linspace(0, states_record.shape[0] - 1, 6, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    vmax = float(states_record_scaled.max())
    for axis, state_idx in zip(axes, snapshot_indices):
        image = axis.imshow(states_record_scaled[state_idx], origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
        axis.set_title(f"t={time_state[state_idx]:.1f}", fontsize=STATE_PANEL_TITLE_SIZE)
        axis.set_xlabel("x", fontsize=STATE_AXIS_LABEL_SIZE, labelpad=2)
        axis.set_ylabel("y", fontsize=STATE_AXIS_LABEL_SIZE, labelpad=2)
        if state_idx == 0:
            axis.scatter(initial_positions[:, 0] * (grid.Nx - 1), initial_positions[:, 1] * (grid.Ny - 1), c="red", s=40)
        else:
            pos_idx = min(state_idx - 1, position_record.shape[0] - 1)
            axis.scatter(
                position_record[pos_idx, :, 0] * (grid.Nx - 1),
                position_record[pos_idx, :, 1] * (grid.Ny - 1),
                c="red",
                s=40,
            )
    fig.subplots_adjust(wspace=0.28, hspace=0.28, left=0.06, right=0.88, bottom=0.07, top=0.93)
    cax = fig.add_axes([0.90, 0.18, 0.015, 0.64])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Concentration")
    style_colorbar(colorbar)
    colorbar.ax.tick_params(labelsize=STATE_COLORBAR_TICK_SIZE)
    colorbar.ax.yaxis.label.set_size(STATE_COLORBAR_LABEL_SIZE)
    style_axes(axes)
    for axis in axes:
        axis.tick_params(axis="both", which="major", labelsize=STATE_TICK_LABEL_SIZE)
        axis.tick_params(axis="both", which="minor", labelsize=max(STATE_TICK_LABEL_SIZE - 2, 8))
    plt.savefig(output_dir / "mpc_2d_state_snapshots.png", dpi=200)
    plt.close()


def load_prediction_context(
    args: argparse.Namespace,
) -> Tuple[GridParams, PhysicalParams, Optional[torch.nn.Module]]:
    grid = GridParams()
    phys = PhysicalParams()
    reference_phys = phys

    surrogate_model = None
    if args.prediction_mode == "surrogate":
        surrogate_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        surrogate_model, loaded_grid, loaded_phys, _, _, _ = load_checkpoint(args.model_path, device=surrogate_device)
        grid = loaded_grid
        phys = loaded_phys
        surrogate_model.eval()
        for parameter in surrogate_model.parameters():
            parameter.requires_grad_(False)
        phys = replace(
            phys,
            sigma_m=max(phys.sigma_m, MIN_CONTROL_SIGMA_M),
            u_max=max(phys.u_max, MIN_CONTROL_U_MAX),
        )
        print(f"Loaded surrogate predictor from {args.model_path} on {surrogate_device}")
        if phys.boundary != reference_phys.boundary or abs(phys.mass_kg - reference_phys.mass_kg) > 1e-6 or abs(phys.damping_per_s - reference_phys.damping_per_s) > 1e-6:
            print(
                "Warning: loaded surrogate checkpoint was trained under different boundary or actuator dynamics. "
                "Retrain pinn.py for a strict comparison setting."
            )
    else:
        print("Using exact PDE as the predictor for MPC.")
        phys = replace(
            phys,
            sigma_m=max(phys.sigma_m, MIN_CONTROL_SIGMA_M),
            u_max=max(phys.u_max, MIN_CONTROL_U_MAX),
        )
    return grid, phys, surrogate_model


def build_controller_dynamics(
    grid: GridParams,
    phys: PhysicalParams,
    control_stride: int,
    mobile_controllers: bool,
) -> ControllerDynamics:
    control_dt = control_stride * grid.dt
    return ControllerDynamics(
        dt=control_dt,
        space_scale_m=phys.space_scale_m,
        time_scale_s=phys.time_scale_s,
        force_limit=100.0 if mobile_controllers else 0.0,
        velocity_limit_m_s=None,
        mass=phys.mass_kg,
        damping=phys.damping_per_s,
        bounds=(0.0, 1.0, 0.0, 1.0),
        d_min_sep=0.10,
    )


def plot_average_concentration_comparison(
    output_dir: Path,
    mobile_result: dict,
    fixed_result: dict,
    control_stride: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = mobile_result["grid"]
    phys = mobile_result["phys"]
    mobile_states = mobile_result["states_record"]
    fixed_states = fixed_result["states_record"]

    mobile_time = np.arange(mobile_states.shape[0]) * control_stride * grid.dt * phys.time_scale_s
    fixed_time = np.arange(fixed_states.shape[0]) * control_stride * grid.dt * phys.time_scale_s
    mobile_avg = mobile_states.mean(axis=(1, 2)) * AVERAGE_PLOT_SCALE
    fixed_avg = fixed_states.mean(axis=(1, 2)) * AVERAGE_PLOT_SCALE

    plt.figure(figsize=(10, 5.5))
    plt.plot(mobile_time, mobile_avg, linewidth=2.2, color="#1f77b4", label="Mobile Actuators")
    plt.plot(fixed_time, fixed_avg, linewidth=2.2, color="#d62728", linestyle="--", label="Fixed Actuators")
    plt.xlabel("Time", fontsize=AVERAGE_COMPARISON_FONT_SIZE)
    plt.ylabel("Average Concentration", fontsize=AVERAGE_COMPARISON_FONT_SIZE)
    plt.grid(alpha=0.3)
    plt.legend()
    average_axes = plt.gcf().axes
    style_axes(average_axes)
    set_tick_font_size(average_axes, AVERAGE_COMPARISON_FONT_SIZE)
    set_legend_font_size(average_axes, AVERAGE_COMPARISON_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_dir / "mpc_2d_average_concentration_comparison.png", dpi=220)
    plt.close()


def plot_average_solve_time_comparison(
    output_dir: Path,
    mobile_result: dict,
    fixed_result: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Mobile Actuators", "Fixed Actuators"]
    values_ms = np.array(
        [
            mobile_result["average_mpc_solve_time_s"] * 1000.0,
            fixed_result["average_mpc_solve_time_s"] * 1000.0,
        ],
        dtype=np.float32,
    )
    colors = ["#1f77b4", "#d62728"]

    plt.figure(figsize=(8.5, 5.5))
    bars = plt.bar(labels, values_ms, color=colors, width=0.72)
    for bar, value in zip(bars, values_ms):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02 * max(float(values_ms.max()), 1.0),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    plt.ylabel("Average solve time per step (ms)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    style_axes(plt.gcf().axes)
    plt.savefig(output_dir / "average_mpc_solve_time_comparison.png", dpi=220)
    plt.close()


def run_closed_loop_case(
    args: argparse.Namespace,
    output_dir: Path,
    scenario_label: str,
    mobile_controllers: bool,
    grid: GridParams,
    phys: PhysicalParams,
    surrogate_model,
) -> dict:
    warm_start_controls = None
    warm_start_forces = None
    dynamics = build_controller_dynamics(grid, phys, args.control_stride, mobile_controllers)

    initial_positions = np.array(
        [
            [0.122, 0.093],
            [0.959, 0.087],
            [0.137, 0.892],
            [0.874, 0.858],
        ],
        dtype=np.float32,
    )
    initial_positions = enforce_separation_np(project_to_subdomains(initial_positions), dynamics.d_min_sep)
    positions = initial_positions.copy()
    velocities = np.zeros_like(positions)
    previous_controls = np.zeros(phys.num_controllers, dtype=np.float32)
    previous_forces = np.zeros((phys.num_controllers, 2), dtype=np.float32)
    field = enforce_boundary_np(initial_condition(grid, phys), phys.boundary)

    states_record: List[np.ndarray] = [field.copy()]
    control_record: List[np.ndarray] = []
    force_record: List[np.ndarray] = []
    position_record: List[np.ndarray] = []
    solve_time_record: List[float] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(field * FIELD_PLOT_SCALE, origin="lower", cmap="viridis")
    style_colorbar(plt.colorbar(label="Concentration"))
    plt.scatter(initial_positions[:, 0] * (grid.Nx - 1), initial_positions[:, 1] * (grid.Ny - 1), c="red", marker="x", s=100)
    plt.tight_layout()
    style_axes(plt.gcf().axes)
    plt.savefig(output_dir / "initial_condition.png", dpi=200)
    plt.close()

    print(
        f"[{scenario_label}] Starting closed loop | mode={args.prediction_mode} | horizon={args.horizon} | "
        f"control_steps={args.control_steps} | control_stride={args.control_stride} | "
        f"target_mean={args.target_mean:.4f} | target_max={args.target_max:.4f}"
    )

    start_time = time.time()
    for step_idx in range(args.control_steps):
        solve_start = time.perf_counter()
        if step_idx == 0:
            applied_controls = np.zeros(phys.num_controllers, dtype=np.float32)
            applied_forces = np.zeros((phys.num_controllers, 2), dtype=np.float32)
            objective_value = 0.0
            solve_elapsed = 0.0
        elif args.prediction_mode == "surrogate":
            controls_seq, forces_seq, predicted_positions, objective_value = mpc_optimize_surrogate(
                current_field=field,
                positions=positions,
                velocities=velocities,
                previous_controls=previous_controls,
                previous_forces=previous_forces,
                dynamics=dynamics,
                grid=grid,
                phys=phys,
                horizon=args.horizon,
                control_stride=args.control_stride,
                surrogate_model=surrogate_model,
                surrogate_iters=args.surrogate_iters,
                surrogate_lr=args.surrogate_lr,
                warm_start_controls=warm_start_controls,
                warm_start_forces=warm_start_forces,
            )
            warm_start_controls = shift_warm_start(controls_seq)
            warm_start_forces = shift_warm_start(forces_seq)
            applied_controls = controls_seq[0]
            applied_forces = forces_seq[0]
            solve_elapsed = time.perf_counter() - solve_start
        else:
            controls_seq, forces_seq, predicted_positions, objective_value = mpc_optimize_slsqp(
                current_field=field,
                positions=positions,
                velocities=velocities,
                previous_controls=previous_controls,
                previous_forces=previous_forces,
                dynamics=dynamics,
                grid=grid,
                phys=phys,
                horizon=args.horizon,
                max_iter=args.max_iter,
                control_stride=args.control_stride,
                prediction_mode=args.prediction_mode,
                surrogate_model=surrogate_model,
            )
            applied_controls = controls_seq[0]
            applied_forces = forces_seq[0]
            solve_elapsed = time.perf_counter() - solve_start
        positions, velocities = dynamics.step(positions, velocities, applied_forces)
        positions = enforce_separation_np(project_to_subdomains(positions), dynamics.d_min_sep)

        source_map = build_source_map_np(positions, applied_controls, grid, phys)
        for _ in range(args.control_stride):
            field = simulate_pde_step_np(field, source_map, grid, phys)

        states_record.append(field.copy())
        control_record.append(applied_controls.copy())
        force_record.append(applied_forces.copy())
        position_record.append(positions.copy())
        solve_time_record.append(float(solve_elapsed))
        previous_controls = applied_controls
        previous_forces = applied_forces

        if step_idx == 0:
            print(
                f"[{scenario_label}] Step {step_idx + 1:03d}/{args.control_steps} | initial zero control | "
                f"mean={field.mean():.5f} | max={field.max():.5f}"
            )
        else:
            print(
                f"[{scenario_label}] Step {step_idx + 1:03d}/{args.control_steps} | objective={objective_value:.4f} | "
                f"mean={field.mean():.5f} | max={field.max():.5f}"
            )

        if (
            step_idx + 1 >= args.min_control_steps
            and field.mean() <= args.target_mean
            and field.max() <= args.target_max
        ):
            print(
                f"[{scenario_label}] Early stop at step {step_idx + 1}: mean={field.mean():.5f}, "
                f"max={field.max():.5f} reached targets."
            )
            break

    elapsed = time.time() - start_time
    solve_time_array = np.array(solve_time_record, dtype=np.float32)
    optimized_solve_times = solve_time_array[1:] if solve_time_array.shape[0] > 1 else solve_time_array
    average_mpc_solve_time_s = float(np.mean(optimized_solve_times)) if optimized_solve_times.size else 0.0
    print(f"[{scenario_label}] Closed-loop rollout finished in {elapsed:.2f}s")
    print(f"[{scenario_label}] Average MPC solve time per step: {average_mpc_solve_time_s * 1000.0:.1f} ms")

    states_array = np.array(states_record)
    controls_array = np.array(control_record)
    forces_array = np.array(force_record)
    positions_array = np.array(position_record)
    plot_results(
        output_dir=output_dir,
        states_record=states_array,
        control_record=controls_array,
        force_record=forces_array,
        position_record=positions_array,
        initial_positions=initial_positions,
        grid=grid,
        phys=phys,
        control_stride=args.control_stride,
        scenario_label=scenario_label,
        solve_time_record=solve_time_array,
    )
    summary = {
        "scenario_label": scenario_label,
        "prediction_mode": args.prediction_mode,
        "controller_mode": "mobile" if mobile_controllers else "fixed",
        "executed_control_steps": int(len(control_record)),
        "elapsed_wall_time_s": float(elapsed),
        "average_mpc_solve_time_s": average_mpc_solve_time_s,
        "average_mpc_solve_time_ms": average_mpc_solve_time_s * 1000.0,
        "final_mean_concentration": float(states_array[-1].mean()),
        "final_max_concentration": float(states_array[-1].max()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[{scenario_label}] Saved MPC plots to {output_dir}")
    return {
        "states_record": states_array,
        "control_record": controls_array,
        "force_record": forces_array,
        "position_record": positions_array,
        "solve_time_record": solve_time_array,
        "initial_positions": initial_positions,
        "grid": grid,
        "phys": phys,
        "elapsed": elapsed,
        "average_mpc_solve_time_s": average_mpc_solve_time_s,
        "scenario_label": scenario_label,
    }


def run_closed_loop() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    grid, phys, surrogate_model = load_prediction_context(args)

    if args.controller_mode == "compare":
        output_dir.mkdir(parents=True, exist_ok=True)
        mobile_result = run_closed_loop_case(
            args=args,
            output_dir=output_dir / "mobile",
            scenario_label="Mobile Actuators",
            mobile_controllers=True,
            grid=grid,
            phys=phys,
            surrogate_model=surrogate_model,
        )
        fixed_result = run_closed_loop_case(
            args=args,
            output_dir=output_dir / "fixed",
            scenario_label="Fixed Actuators",
            mobile_controllers=False,
            grid=grid,
            phys=phys,
            surrogate_model=surrogate_model,
        )
        plot_average_concentration_comparison(output_dir, mobile_result, fixed_result, args.control_stride)
        plot_average_solve_time_comparison(output_dir, mobile_result, fixed_result)
        comparison_summary = {
            "prediction_mode": args.prediction_mode,
            "mobile_average_mpc_solve_time_s": float(mobile_result["average_mpc_solve_time_s"]),
            "fixed_average_mpc_solve_time_s": float(fixed_result["average_mpc_solve_time_s"]),
            "mobile_final_mean_concentration": float(mobile_result["states_record"][-1].mean()),
            "fixed_final_mean_concentration": float(fixed_result["states_record"][-1].mean()),
            "mobile_executed_control_steps": int(mobile_result["control_record"].shape[0]),
            "fixed_executed_control_steps": int(fixed_result["control_record"].shape[0]),
        }
        with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(comparison_summary, handle, ensure_ascii=False, indent=2)
        print(f"Saved comparison plots to {output_dir}")
        return

    scenario_label = "Fixed Actuators" if args.controller_mode == "fixed" else "Mobile Actuators"
    run_closed_loop_case(
        args=args,
        output_dir=output_dir,
        scenario_label=scenario_label,
        mobile_controllers=(args.controller_mode != "fixed"),
        grid=grid,
        phys=phys,
        surrogate_model=surrogate_model,
    )


if __name__ == "__main__":
    run_closed_loop()
