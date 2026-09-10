from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, Tuple
from zipfile import BadZipFile

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from plot_style import apply_readable_plot_style, style_axes, style_colorbar
from rod_cooling_1d.core import (
    ControllerDynamics1D,
    RodCoolingConfig,
    apply_dirichlet,
    average_temperature,
    build_controller_dynamics,
    build_source_map,
    generate_transition_dataset,
    initial_controller_positions,
    initial_state,
    make_spatial_grid,
    paper_centers,
    pde_step,
    project_control_step,
    project_force_step,
)
from rod_cooling_1d.model import device_for_run, load_checkpoint
from rod_cooling_1d.svd_rbf_baseline import (
    SVDRBFBaselineConfig,
    load_or_train_svd_rbf_bundle,
    optimize_svd_rbf_step,
    svd_rbf_predict_rollout,
)

CONTROL_OUTPUT_VERSION = "rod-cooling-1d-control-v3"


apply_readable_plot_style()


def parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run 1-D rod cooling closed-loop MPC.")
    parser.add_argument("--prediction-mode", choices=("pde", "surrogate"), default="surrogate")
    parser.add_argument("--controller-mode", choices=("mobile", "fixed", "svd-rbf", "compare"), default="mobile")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=folder / "outputs" / "train" / "rod_cooling_1d_best.pth",
    )
    parser.add_argument(
        "--svd-rbf-checkpoint",
        type=Path,
        default=folder / "outputs" / "svd_rbf" / "rod_cooling_svd_rbf.pkl",
    )
    parser.add_argument("--output-dir", type=Path, default=folder / "outputs" / "control")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=4)
    parser.add_argument("--control-steps", type=int, default=480)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--svd-rbf-train-trajectories", type=int, default=72)
    parser.add_argument("--svd-rbf-train-steps", type=int, default=240)
    parser.add_argument("--svd-rbf-delay-steps", type=int, default=5)
    parser.add_argument("--svd-rbf-hidden-units", type=int, default=180)
    parser.add_argument("--svd-rbf-gamma", type=float, default=0.25)
    parser.add_argument("--svd-rbf-position-delta-max", type=float, default=0.03)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--build-comparison-only", action="store_true")
    return parser.parse_args()


def predict_with_surrogate(
    model: torch.nn.Module,
    state: np.ndarray,
    source_map: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        source_tensor = torch.from_numpy(source_map).unsqueeze(0).to(device)
        prediction = model(state_tensor, source_tensor)
    return prediction.squeeze(0).cpu().numpy().astype(np.float32)


def control_dt(cfg: RodCoolingConfig) -> float:
    return float(cfg.dt * cfg.control_stride)


def build_case_cache_meta(
    cfg: RodCoolingConfig,
    prediction_mode: str,
    controller_mode: str,
    horizon: int,
    max_iter: int,
    baseline_cfg: Optional[SVDRBFBaselineConfig] = None,
) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "version": CONTROL_OUTPUT_VERSION,
        "prediction_mode": prediction_mode,
        "controller_mode": controller_mode,
        "horizon": int(horizon),
        "max_iter": int(max_iter),
        "config": asdict(cfg),
    }
    if baseline_cfg is not None:
        meta["baseline_config"] = baseline_cfg.to_dict()
    return json.loads(json.dumps(meta))


def try_load_case_cache(
    output_dir: Path,
    expected_meta: Dict[str, object],
) -> Optional[Dict[str, np.ndarray]]:
    meta_path = output_dir / "result_meta.json"
    data_path = output_dir / "result_data.npz"
    if not meta_path.exists() or not data_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as fh:
        actual_meta = json.load(fh)
    if actual_meta != expected_meta:
        return None
    try:
        data = np.load(data_path)
        return {key: data[key].astype(np.float32) for key in data.files}
    except (OSError, ValueError, BadZipFile, EOFError):
        return None


def save_case_cache(
    output_dir: Path,
    result: Dict[str, np.ndarray],
    meta: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "result_data.npz", **result)
    with (output_dir / "result_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def advance_dynamics_with_hold(
    dynamics: ControllerDynamics1D,
    positions: np.ndarray,
    velocities: np.ndarray,
    force: np.ndarray,
    cfg: RodCoolingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_state = positions.astype(np.float32, copy=True)
    vel_state = velocities.astype(np.float32, copy=True)
    for _ in range(cfg.control_stride):
        pos_state, vel_state = dynamics.step(pos_state, vel_state, force)
    return pos_state, vel_state


def rollout_dynamics(
    dynamics: ControllerDynamics1D,
    positions: np.ndarray,
    velocities: np.ndarray,
    cfg: RodCoolingConfig,
    forces_seq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_state = positions.astype(np.float32, copy=True)
    vel_state = velocities.astype(np.float32, copy=True)
    pos_hist = np.zeros((forces_seq.shape[0], positions.shape[0]), dtype=np.float32)
    vel_hist = np.zeros_like(pos_hist)
    for step_idx in range(forces_seq.shape[0]):
        pos_state, vel_state = advance_dynamics_with_hold(dynamics, pos_state, vel_state, forces_seq[step_idx], cfg)
        pos_hist[step_idx] = pos_state
        vel_hist[step_idx] = vel_state
    return pos_hist, vel_hist


def rollout_prediction(
    initial_state_np: np.ndarray,
    controls: np.ndarray,
    positions: np.ndarray,
    grid: np.ndarray,
    cfg: RodCoolingConfig,
    mode: str,
    surrogate_model: Optional[torch.nn.Module],
    device: Optional[torch.device],
) -> np.ndarray:
    states = np.zeros((controls.shape[0] + 1, cfg.grid_points), dtype=np.float32)
    states[0] = apply_dirichlet(initial_state_np)
    current = states[0].copy()
    for step in range(controls.shape[0]):
        for _ in range(cfg.control_stride):
            if mode == "pde":
                current = pde_step(current, positions[step], controls[step], grid, cfg)
            else:
                source_map = build_source_map(grid, positions[step], controls[step], cfg)
                current = predict_with_surrogate(surrogate_model, current, source_map, device)
        states[step + 1] = current
    return states


def build_region_masks(grid: np.ndarray, cfg: RodCoolingConfig) -> list[np.ndarray]:
    edges = np.linspace(cfg.position_min, cfg.position_max, cfg.controller_count + 1)
    masks = []
    for idx in range(cfg.controller_count):
        if idx == cfg.controller_count - 1:
            mask = (grid >= edges[idx]) & (grid <= edges[idx + 1])
        else:
            mask = (grid >= edges[idx]) & (grid < edges[idx + 1])
        masks.append(mask)
    return masks


def compute_tracking_targets(state: np.ndarray, grid: np.ndarray, cfg: RodCoolingConfig) -> Tuple[np.ndarray, np.ndarray]:
    masks = build_region_masks(grid, cfg)
    targets = np.zeros(cfg.controller_count, dtype=np.float32)
    masses = np.zeros(cfg.controller_count, dtype=np.float32)
    for ctrl_idx, mask in enumerate(masks):
        region_values = state[mask]
        region_grid = grid[mask]
        if region_values.size == 0:
            targets[ctrl_idx] = float((ctrl_idx + 0.5) * cfg.rod_length_m / cfg.controller_count)
            continue
        masses[ctrl_idx] = float(np.sum(region_values))
        if masses[ctrl_idx] <= 1e-8:
            targets[ctrl_idx] = float(region_grid.mean())
            continue
        mass_center = float(np.sum(region_grid * region_values) / masses[ctrl_idx])
        peak = float(region_grid[int(np.argmax(region_values))])
        targets[ctrl_idx] = 0.65 * mass_center + 0.35 * peak
    mass_weights = masses / max(float(masses.max()), 1e-6)
    return targets, mass_weights.astype(np.float32)


def compute_dynamic_tracking_targets(states: np.ndarray, grid: np.ndarray, cfg: RodCoolingConfig) -> np.ndarray:
    masks = build_region_masks(grid, cfg)
    targets = np.zeros((states.shape[0], cfg.controller_count), dtype=np.float32)
    for step_idx in range(states.shape[0]):
        state = states[step_idx]
        for ctrl_idx, mask in enumerate(masks):
            region_values = state[mask]
            region_grid = grid[mask]
            if region_values.size == 0:
                targets[step_idx, ctrl_idx] = float((ctrl_idx + 0.5) * cfg.rod_length_m / cfg.controller_count)
                continue
            peak = float(region_grid[int(np.argmax(region_values))])
            if float(np.sum(region_values)) <= 1e-8:
                targets[step_idx, ctrl_idx] = peak
                continue
            mass_center = float(np.sum(region_grid * region_values) / np.sum(region_values))
            targets[step_idx, ctrl_idx] = 0.3 * mass_center + 0.7 * peak
    return targets


def state_activity_scale(state: np.ndarray, reference_rms: float = 0.28) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(state, dtype=np.float32) ** 2)))
    return float(np.clip(rms / max(reference_rms, 1e-6), 0.04, 1.0))


def solver_iteration_budget(state: np.ndarray, max_iter: int, min_iter: int = 3) -> int:
    if max_iter <= min_iter:
        return int(max_iter)
    activity = state_activity_scale(state)
    scaled = float(np.clip(activity ** 0.65, 0.0, 1.0))
    return int(np.clip(np.rint(min_iter + (max_iter - min_iter) * scaled), min_iter, max_iter))


def heuristic_force_guess(
    positions: np.ndarray,
    velocities: np.ndarray,
    targets: np.ndarray,
    activity_scale: float,
    cfg: RodCoolingConfig,
) -> np.ndarray:
    proportional_gain = 120.0
    damping_gain = 12.0
    forces = activity_scale * (proportional_gain * (targets - positions) - damping_gain * velocities)
    return np.clip(forces, -cfg.force_limit, cfg.force_limit).astype(np.float32)


def heuristic_control_guess(
    positions: np.ndarray,
    targets: np.ndarray,
    mass_weights: np.ndarray,
    activity_scale: float,
    cfg: RodCoolingConfig,
) -> np.ndarray:
    distances = np.abs(targets - positions)
    proximity = np.exp(-(distances ** 2) / (2.0 * 0.42 ** 2))
    desired_strength = activity_scale * (2.0 + 5.0 * mass_weights * proximity)
    return -np.clip(desired_strength, 0.0, abs(cfg.control_min) * 0.95).astype(np.float32)


def project_control_sequence(previous_controls: np.ndarray, controls_seq: np.ndarray, cfg: RodCoolingConfig) -> np.ndarray:
    projected = np.zeros_like(controls_seq, dtype=np.float32)
    previous = previous_controls.astype(np.float32, copy=True)
    for step_idx in range(controls_seq.shape[0]):
        projected[step_idx] = project_control_step(controls_seq[step_idx], previous, cfg)
        previous = projected[step_idx]
    return projected


def project_force_sequence(
    previous_forces: np.ndarray,
    forces_seq: np.ndarray,
    cfg: RodCoolingConfig,
    mobile: bool,
) -> np.ndarray:
    projected = np.zeros_like(forces_seq, dtype=np.float32)
    previous = previous_forces.astype(np.float32, copy=True)
    for step_idx in range(forces_seq.shape[0]):
        projected[step_idx] = project_force_step(forces_seq[step_idx], previous, cfg, mobile=mobile)
        previous = projected[step_idx]
    return projected


def build_initial_guess(
    horizon: int,
    cfg: RodCoolingConfig,
    state_now: np.ndarray,
    prev_u: np.ndarray,
    prev_f: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    tracking_targets: np.ndarray,
    mass_weights: np.ndarray,
    last_solution: Optional[Dict[str, np.ndarray]],
    mobile: bool,
) -> np.ndarray:
    activity_scale = state_activity_scale(state_now)
    heuristic_controls = heuristic_control_guess(positions, tracking_targets, mass_weights, activity_scale, cfg)
    heuristic_forces = heuristic_force_guess(positions, velocities, tracking_targets, activity_scale, cfg)

    if last_solution is None:
        control_guess = np.tile(0.15 * prev_u + 0.85 * heuristic_controls, (horizon, 1)).astype(np.float32)
        force_guess = np.tile(0.15 * prev_f + 0.85 * heuristic_forces, (horizon, 1)).astype(np.float32)
    else:
        shifted_controls = np.vstack([last_solution["controls"][1:], last_solution["controls"][-1:]]).astype(np.float32)
        shifted_forces = np.vstack([last_solution["forces"][1:], last_solution["forces"][-1:]]).astype(np.float32)
        heuristic_control_seq = np.tile(heuristic_controls, (horizon, 1)).astype(np.float32)
        heuristic_force_seq = np.tile(heuristic_forces, (horizon, 1)).astype(np.float32)
        control_guess = (0.7 * shifted_controls + 0.3 * heuristic_control_seq).astype(np.float32)
        force_guess = (0.7 * shifted_forces + 0.3 * heuristic_force_seq).astype(np.float32)

    control_guess = project_control_sequence(prev_u, control_guess, cfg)
    if not mobile:
        return control_guess.astype(np.float64).ravel()
    force_guess = project_force_sequence(prev_f, force_guess, cfg, mobile=True)
    return np.concatenate([control_guess.ravel(), force_guess.ravel()]).astype(np.float64)


def unpack_decision_vector(
    vector: np.ndarray,
    horizon: int,
    cfg: RodCoolingConfig,
    mobile: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    control_size = horizon * cfg.controller_count
    controls = vector[:control_size].reshape(horizon, cfg.controller_count)
    if mobile:
        forces = vector[control_size:].reshape(horizon, cfg.controller_count)
    else:
        forces = np.zeros((horizon, cfg.controller_count), dtype=np.float32)
    return controls.astype(np.float32), forces.astype(np.float32)


def delta_constraints(
    vector: np.ndarray,
    horizon: int,
    cfg: RodCoolingConfig,
    prev_u: np.ndarray,
    prev_f: np.ndarray,
    mobile: bool,
) -> np.ndarray:
    controls, forces = unpack_decision_vector(vector, horizon, cfg, mobile)
    constraints = []
    for step_idx in range(horizon):
        base_u = prev_u if step_idx == 0 else controls[step_idx - 1]
        constraints.append(cfg.control_delta_max - np.abs(controls[step_idx] - base_u))
        if mobile:
            base_f = prev_f if step_idx == 0 else forces[step_idx - 1]
            constraints.append(cfg.force_delta_max - np.abs(forces[step_idx] - base_f))
    return np.concatenate([np.atleast_1d(item).astype(np.float32) for item in constraints])


def separation_constraints(
    vector: np.ndarray,
    horizon: int,
    cfg: RodCoolingConfig,
    prev_f: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    dynamics: ControllerDynamics1D,
    mobile: bool,
) -> np.ndarray:
    if not mobile:
        return np.array([1.0], dtype=np.float32)
    _, forces = unpack_decision_vector(vector, horizon, cfg, mobile)
    forces = project_force_sequence(prev_f, forces, cfg, mobile=True)
    predicted_positions, _ = rollout_dynamics(dynamics, positions, velocities, cfg, forces)
    margins = []
    for step_idx in range(predicted_positions.shape[0]):
        for ctrl_idx in range(predicted_positions.shape[1] - 1):
            margins.append(predicted_positions[step_idx, ctrl_idx + 1] - predicted_positions[step_idx, ctrl_idx] - cfg.min_separation)
    return np.asarray(margins, dtype=np.float32) if margins else np.array([1.0], dtype=np.float32)


def objective_from_sequences(
    state_now: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    prev_u: np.ndarray,
    prev_f: np.ndarray,
    controls_seq: np.ndarray,
    forces_seq: np.ndarray,
    tracking_targets: np.ndarray,
    grid: np.ndarray,
    cfg: RodCoolingConfig,
    dynamics: ControllerDynamics1D,
    mode: str,
    surrogate_model: Optional[torch.nn.Module],
    device: Optional[torch.device],
    mobile: bool,
) -> float:
    controls_seq = project_control_sequence(prev_u, controls_seq, cfg)
    forces_seq = project_force_sequence(prev_f, forces_seq, cfg, mobile=mobile)
    predicted_positions, predicted_velocities = rollout_dynamics(dynamics, positions, velocities, cfg, forces_seq)
    predicted_states = rollout_prediction(state_now, controls_seq, predicted_positions, grid, cfg, mode, surrogate_model, device)
    dynamic_targets = compute_dynamic_tracking_targets(predicted_states[1:], grid, cfg)
    state_energy_seq = np.mean(predicted_states[1:] ** 2, axis=1).astype(np.float64)
    activity_weights = state_energy_seq / (state_energy_seq + 4e-3)
    quiet_weights = 1.0 - activity_weights

    state_cost = float(np.mean(predicted_states[1:-1] ** 2)) if predicted_states.shape[0] > 2 else 0.0
    terminal_cost = float(np.mean(predicted_states[-1] ** 2))
    control_energy_seq = np.mean(controls_seq ** 2, axis=1).astype(np.float64)
    force_energy_seq = np.mean(forces_seq ** 2, axis=1).astype(np.float64) if mobile else np.zeros_like(state_energy_seq)
    velocity_energy_seq = np.mean(predicted_velocities ** 2, axis=1).astype(np.float64) if mobile else np.zeros_like(state_energy_seq)
    tracking_error_seq = np.mean((predicted_positions - dynamic_targets) ** 2, axis=1).astype(np.float64)
    control_cost = float(np.mean(np.abs(controls_seq)) + 0.35 * np.mean(control_energy_seq))
    force_cost = float(np.mean(force_energy_seq))
    target_cost = float(np.mean(activity_weights * tracking_error_seq))
    terminal_target_cost = float(activity_weights[-1] * tracking_error_seq[-1]) if tracking_error_seq.size > 0 else 0.0
    source_overlap_reward = 0.0
    for step_idx in range(controls_seq.shape[0]):
        source_map = build_source_map(grid, predicted_positions[step_idx], controls_seq[step_idx], cfg)
        source_overlap_reward += float(activity_weights[step_idx] * np.mean((-source_map) * predicted_states[step_idx + 1]))
    quiescent_control_cost = float(np.mean(quiet_weights * control_energy_seq))
    quiescent_force_cost = float(np.mean(quiet_weights * force_energy_seq))
    quiescent_velocity_cost = float(np.mean(quiet_weights * velocity_energy_seq))
    terminal_control_cost = float(control_energy_seq[-1])
    terminal_force_cost = float(force_energy_seq[-1]) if mobile else 0.0
    terminal_velocity_cost = float(velocity_energy_seq[-1]) if mobile else 0.0
    terminal_quiet_weight = float(quiet_weights[-1]) if quiet_weights.size > 0 else 0.0

    return (
        70.0 * state_cost
        + 105.0 * terminal_cost
        + 34.0 * target_cost
        + 60.0 * terminal_target_cost
        + 0.16 * control_cost
        + 0.006 * force_cost
        + 0.8 * quiescent_control_cost
        + 0.008 * quiescent_force_cost
        + 0.12 * quiescent_velocity_cost
        + terminal_quiet_weight * (2.4 * terminal_control_cost + 0.012 * terminal_force_cost + 0.18 * terminal_velocity_cost)
        - 9.0 * source_overlap_reward
    )


def optimize_mpc_step(
    state_now: np.ndarray,
    prev_u: np.ndarray,
    prev_f: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    grid: np.ndarray,
    cfg: RodCoolingConfig,
    horizon: int,
    max_iter: int,
    mode: str,
    mobile: bool,
    surrogate_model: Optional[torch.nn.Module],
    device: Optional[torch.device],
    last_solution: Optional[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    dynamics = build_controller_dynamics(cfg, mobile=mobile)
    tracking_targets, mass_weights = compute_tracking_targets(state_now, grid, cfg)
    initial_guess = build_initial_guess(
        horizon,
        cfg,
        state_now,
        prev_u,
        prev_f,
        positions,
        velocities,
        tracking_targets,
        mass_weights,
        last_solution,
        mobile,
    )

    bounds = [(cfg.control_min, cfg.control_max)] * (horizon * cfg.controller_count)
    if mobile:
        bounds.extend([(-cfg.force_limit, cfg.force_limit)] * (horizon * cfg.controller_count))

    def objective(vector: np.ndarray) -> float:
        controls, forces = unpack_decision_vector(vector, horizon, cfg, mobile)
        return objective_from_sequences(
            state_now,
            positions,
            velocities,
            prev_u,
            prev_f,
            controls,
            forces,
            tracking_targets,
            grid,
            cfg,
            dynamics,
            mode,
            surrogate_model,
            device,
            mobile,
        )

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: delta_constraints(x, horizon, cfg, prev_u, prev_f, mobile),
        }
    ]
    if mobile:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: separation_constraints(x, horizon, cfg, prev_f, positions, velocities, dynamics, mobile),
            }
        )

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-5, "disp": False},
    )

    controls, forces = unpack_decision_vector(result.x, horizon, cfg, mobile)
    controls = project_control_sequence(prev_u, controls, cfg)
    forces = project_force_sequence(prev_f, forces, cfg, mobile=mobile)
    predicted_positions, predicted_velocities = rollout_dynamics(dynamics, positions, velocities, cfg, forces)
    predicted_states = rollout_prediction(state_now, controls, predicted_positions, grid, cfg, mode, surrogate_model, device)
    return {
        "controls": controls,
        "forces": forces,
        "positions": predicted_positions,
        "velocities": predicted_velocities,
        "states": predicted_states,
        "objective": np.asarray([float(objective(result.x))], dtype=np.float32),
    }


def plot_initial_condition(state0: np.ndarray, controller_positions: np.ndarray, grid: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid, state0 * 100.0, linewidth=2.5, color="tab:blue")
    for pos in controller_positions:
        ax.axvline(float(pos), color="tab:red", linestyle="--", alpha=0.75)
    ax.set_xlabel("z")
    ax.set_ylabel("Temperature")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    style_axes(ax)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_state_snapshots(
    states: np.ndarray,
    positions: np.ndarray,
    grid: np.ndarray,
    cfg: RodCoolingConfig,
    output_path: Path,
    title: str,
) -> None:
    snapshot_indices = np.linspace(0, states.shape[0] - 1, 6, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    y_max = float(np.max(states) * 1.08 + 1e-6)
    axes_flat = axes.flatten()
    for ax, idx in zip(axes_flat, snapshot_indices):
        ax.plot(grid, states[idx] * cfg.temperature_display_scale, linewidth=2.2, color="tab:blue")
        position_idx = min(max(idx - 1, 0), positions.shape[0] - 1)
        for pos in positions[position_idx]:
            ax.axvline(float(pos), color="tab:red", linestyle="--", alpha=0.65)
        ax.set_title(f"t={idx * control_dt(cfg):.3f}s")
        ax.set_ylim(0.0, y_max * cfg.temperature_display_scale)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("z")
    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature")
    fig.tight_layout()
    style_axes(axes)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_average_temperature(states: np.ndarray, cfg: RodCoolingConfig, output_path: Path, title: str) -> None:
    time_axis = np.arange(states.shape[0], dtype=np.float32)
    avg = average_temperature(states)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time_axis, avg * cfg.temperature_display_scale, linewidth=2.2, color="tab:blue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average temperature")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    style_axes(ax)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_average_comparison(
    curves: list[tuple[str, np.ndarray, dict]],
    cfg: RodCoolingConfig,
    output_path: Path,
    title: str = "Average Temperature Comparison",
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for label, states, style in curves:
        time_axis = np.arange(states.shape[0], dtype=np.float32)
        ax.plot(
            time_axis,
            average_temperature(states) * cfg.temperature_display_scale,
            label=label,
            linewidth=2.2,
            **style,
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Average temperature")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    style_axes(ax)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def rollout_svd_with_truth_warmup(
    bundle: Dict[str, object],
    true_states: np.ndarray,
    controls: np.ndarray,
    positions: np.ndarray,
    cfg: RodCoolingConfig,
) -> np.ndarray:
    delay_steps = int(bundle["delay_steps"])
    warmup = min(delay_steps, true_states.shape[0])
    if warmup >= true_states.shape[0]:
        return true_states.copy().astype(np.float32)
    tail_prediction = svd_rbf_predict_rollout(
        bundle,
        true_states[:warmup],
        controls[warmup - 1 :],
        positions[warmup - 1 :],
        cfg,
    )
    full_prediction = np.zeros_like(true_states, dtype=np.float32)
    full_prediction[: warmup - 1] = true_states[: warmup - 1]
    full_prediction[warmup - 1 :] = tail_prediction[: full_prediction.shape[0] - (warmup - 1)]
    return full_prediction


def build_model_vs_truth_error_curves(
    cfg: RodCoolingConfig,
    surrogate_model: torch.nn.Module,
    device: torch.device,
    svd_bundle: Dict[str, object],
    num_trajectories: int = 6,
) -> Dict[str, np.ndarray]:
    dataset = generate_transition_dataset(
        cfg,
        num_trajectories=num_trajectories,
        steps=cfg.control_steps,
        seed=cfg.training_seed + 2000,
        mobile_probability=1.0,
    )
    true_rollouts = dataset["rollouts"]
    controls = dataset["controls"]
    positions = dataset["positions"]
    grid = dataset["grid"]

    pinn_rollouts = []
    svd_rollouts = []
    for traj_idx in range(num_trajectories):
        pinn_rollouts.append(
            rollout_prediction(
                true_rollouts[traj_idx, 0],
                controls[traj_idx],
                positions[traj_idx],
                grid,
                cfg,
                "surrogate",
                surrogate_model,
                device,
            )
        )
        svd_rollouts.append(
            rollout_svd_with_truth_warmup(
                svd_bundle,
                true_rollouts[traj_idx],
                controls[traj_idx],
                positions[traj_idx],
                cfg,
            )
        )

    true_rollouts = np.asarray(true_rollouts, dtype=np.float32)
    pinn_rollouts = np.asarray(pinn_rollouts, dtype=np.float32)
    svd_rollouts = np.asarray(svd_rollouts, dtype=np.float32)

    pinn_avg_abs = np.mean(
        np.abs(average_temperature(pinn_rollouts) - average_temperature(true_rollouts)),
        axis=0,
    ) * cfg.temperature_display_scale
    svd_avg_abs = np.mean(
        np.abs(average_temperature(svd_rollouts) - average_temperature(true_rollouts)),
        axis=0,
    ) * cfg.temperature_display_scale

    pinn_rmse = np.sqrt(np.mean((pinn_rollouts - true_rollouts) ** 2, axis=(0, 2))) * cfg.temperature_display_scale
    svd_rmse = np.sqrt(np.mean((svd_rollouts - true_rollouts) ** 2, axis=(0, 2))) * cfg.temperature_display_scale

    return {
        "pinn_avg_abs": pinn_avg_abs.astype(np.float32),
        "svd_avg_abs": svd_avg_abs.astype(np.float32),
        "pinn_rmse": pinn_rmse.astype(np.float32),
        "svd_rmse": svd_rmse.astype(np.float32),
    }


def plot_pinn_vs_svd_error(
    error_curves: Dict[str, np.ndarray],
    cfg: RodCoolingConfig,
    output_path: Path,
    title: str = "PINN and SVD-RBFNN Error vs PDE Truth",
) -> None:
    aligned_steps = min(
        error_curves["pinn_avg_abs"].shape[0],
        error_curves["svd_avg_abs"].shape[0],
        error_curves["pinn_rmse"].shape[0],
        error_curves["svd_rmse"].shape[0],
    )
    step_axis = np.arange(aligned_steps, dtype=np.int32)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)

    axes[0].plot(
        step_axis,
        error_curves["pinn_avg_abs"][:aligned_steps],
        color="tab:blue",
        linewidth=2.2,
        label="PINN avg-temp abs error",
    )
    axes[0].plot(
        step_axis,
        error_curves["svd_avg_abs"][:aligned_steps],
        color="tab:green",
        linewidth=2.2,
        linestyle="--",
        label="SVD avg-temp abs error",
    )
    axes[0].set_ylabel("Abs error")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        step_axis,
        error_curves["pinn_rmse"][:aligned_steps],
        color="tab:red",
        linewidth=2.0,
        label="PINN state RMSE",
    )
    axes[1].plot(
        step_axis,
        error_curves["svd_rmse"][:aligned_steps],
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
        label="SVD state RMSE",
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("State error")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    style_axes(axes)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_spacetime_surface_3d(
    states: np.ndarray,
    grid: np.ndarray,
    cfg: RodCoolingConfig,
    output_path: Path,
    title: str,
) -> None:
    time_axis = np.arange(states.shape[0], dtype=np.float32) * control_dt(cfg)
    time_mesh, grid_mesh = np.meshgrid(time_axis, grid, indexing="ij")
    temperature = states * cfg.temperature_display_scale

    fig = plt.figure(figsize=(10.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        time_mesh,
        grid_mesh,
        temperature,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        rstride=max(1, states.shape[0] // 160),
        cstride=1,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("z")
    ax.set_zlabel("Temperature")
    style_colorbar(fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.12, label="Temperature"))
    fig.tight_layout()
    style_axes(ax)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_spacetime_surface_3d_comparison(
    mobile_states: np.ndarray,
    fixed_states: np.ndarray,
    grid: np.ndarray,
    cfg: RodCoolingConfig,
    output_path: Path,
    title: str = "Closed-Loop State Surface Comparison",
) -> None:
    step_axis_mobile = np.arange(mobile_states.shape[0], dtype=np.int32)
    step_axis_fixed = np.arange(fixed_states.shape[0], dtype=np.int32)
    time_mesh_mobile, grid_mesh_mobile = np.meshgrid(step_axis_mobile, grid, indexing="ij")
    time_mesh_fixed, grid_mesh_fixed = np.meshgrid(step_axis_fixed, grid, indexing="ij")

    mobile_temperature = mobile_states * cfg.temperature_display_scale
    fixed_temperature = fixed_states * cfg.temperature_display_scale
    vmin = float(min(np.min(mobile_temperature), np.min(fixed_temperature)))
    vmax = float(max(np.max(mobile_temperature), np.max(fixed_temperature)))

    fig = plt.figure(figsize=(16.0, 6.8))
    ax_mobile = fig.add_subplot(121, projection="3d")
    ax_fixed = fig.add_subplot(122, projection="3d")

    surface_mobile = ax_mobile.plot_surface(
        time_mesh_mobile,
        grid_mesh_mobile,
        mobile_temperature,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        rstride=max(1, mobile_states.shape[0] // 160),
        cstride=1,
        vmin=vmin,
        vmax=vmax,
    )
    ax_mobile.set_title("Mobile Actuators")
    ax_mobile.set_xlabel("Time")
    ax_mobile.set_ylabel("p")
    ax_mobile.set_zlabel("Temperature")
    ax_mobile.set_xlim(0, cfg.control_steps)
    ax_mobile.set_xticks(np.linspace(0, cfg.control_steps, 5, dtype=np.int32))

    surface_fixed = ax_fixed.plot_surface(
        time_mesh_fixed,
        grid_mesh_fixed,
        fixed_temperature,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        rstride=max(1, fixed_states.shape[0] // 160),
        cstride=1,
        vmin=vmin,
        vmax=vmax,
    )
    ax_fixed.set_title("Fixed Actuators")
    ax_fixed.set_xlabel("Time")
    ax_fixed.set_ylabel("p")
    ax_fixed.set_zlabel("Temperature")
    ax_fixed.set_xlim(0, cfg.control_steps)
    ax_fixed.set_xticks(np.linspace(0, cfg.control_steps, 5, dtype=np.int32))

    fig.subplots_adjust(left=0.04, right=0.90, bottom=0.06, top=0.90, wspace=0.06)
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.62])
    colorbar = fig.colorbar(surface_mobile, cax=cax)
    colorbar.set_label("Temperature")
    style_colorbar(colorbar)
    style_axes([ax_mobile, ax_fixed])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_controls(controls: np.ndarray, cfg: RodCoolingConfig, output_path: Path, title: str) -> None:
    step_axis = np.arange(1, controls.shape[0] + 1, dtype=np.int32)
    fig, axes = plt.subplots(cfg.controller_count, 1, figsize=(8, 7), sharex=True)
    if cfg.controller_count == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.step(step_axis, controls[:, idx], where="post", linewidth=1.8)
        ax.set_ylabel(f"u_{idx + 1}")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    style_axes(axes)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_positions(positions: np.ndarray, cfg: RodCoolingConfig, output_path: Path, title: str) -> None:
    step_axis = np.arange(1, positions.shape[0] + 1, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx in range(cfg.controller_count):
        ax.plot(step_axis, positions[:, idx], linewidth=2.0, label=f"Actuator {idx + 1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    ax.set_ylim(cfg.position_min - 0.05, cfg.position_max + 0.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    style_axes(ax)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_forces(forces: np.ndarray, cfg: RodCoolingConfig, output_path: Path, title: str) -> None:
    step_axis = np.arange(1, forces.shape[0] + 1, dtype=np.int32)
    fig, axes = plt.subplots(cfg.controller_count, 1, figsize=(8, 7), sharex=True)
    if cfg.controller_count == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.step(step_axis, forces[:, idx], where="post", linewidth=1.8)
        ax.set_ylabel(f"F_{idx + 1}")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    style_axes(axes)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_controls_and_forces(
    controls: np.ndarray,
    forces: np.ndarray,
    cfg: RodCoolingConfig,
    output_path: Path,
) -> None:
    aligned_steps = min(controls.shape[0], forces.shape[0])
    step_axis = np.arange(1, aligned_steps + 1, dtype=np.int32)
    fig, axes = plt.subplots(cfg.controller_count, 2, figsize=(8.94, 3.9), sharex=True)
    if cfg.controller_count == 1:
        axes = np.asarray([axes])

    for idx in range(cfg.controller_count):
        axes[idx, 0].plot(step_axis, controls[:aligned_steps, idx], linewidth=1.7, color="tab:blue")
        axes[idx, 0].set_ylabel(f"Actuator {idx + 1}", fontsize=11, labelpad=22)
        axes[idx, 0].grid(alpha=0.2)

        axes[idx, 1].plot(step_axis, forces[:aligned_steps, idx], linewidth=1.7, color="tab:blue")
        axes[idx, 1].set_ylabel("")
        axes[idx, 1].grid(alpha=0.2)

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    fig.text(0.28, 0.015, "(a) Control input", ha="center", va="bottom", fontsize=12)
    fig.text(0.74, 0.015, "(b) Driving force", ha="center", va="bottom", fontsize=12)
    fig.tight_layout(rect=(0.04, 0.055, 1.0, 1.0))
    style_axes(axes)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_case(
    cfg: RodCoolingConfig,
    grid: np.ndarray,
    output_dir: Path,
    prediction_mode: str,
    controller_mode: str,
    surrogate_model: Optional[torch.nn.Module],
    device: Optional[torch.device],
    horizon: int,
    max_iter: int,
    progress_every: int,
) -> Dict[str, np.ndarray]:
    mobile = controller_mode == "mobile"
    dynamics = build_controller_dynamics(cfg, mobile=mobile)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_now = initial_state(grid, cfg)
    prev_u = np.zeros(cfg.controller_count, dtype=np.float32)
    prev_f = np.zeros(cfg.controller_count, dtype=np.float32)
    prev_p = initial_controller_positions(cfg)
    prev_v = np.zeros(cfg.controller_count, dtype=np.float32)

    states_record = [state_now.copy()]
    controls_record = []
    forces_record = []
    positions_record = []
    velocities_record = []
    objectives = []
    solve_times = []
    last_solution: Optional[Dict[str, np.ndarray]] = None

    for step_idx in range(cfg.control_steps):
        if step_idx == 0:
            applied_controls = np.zeros(cfg.controller_count, dtype=np.float32)
            applied_forces = np.zeros(cfg.controller_count, dtype=np.float32)
            solution_objective = 0.0
            solve_times.append(0.0)
        else:
            solver_max_iter = solver_iteration_budget(state_now, max_iter)
            solve_start = perf_counter()
            solution = optimize_mpc_step(
                state_now=state_now,
                prev_u=prev_u,
                prev_f=prev_f,
                positions=prev_p,
                velocities=prev_v,
                grid=grid,
                cfg=cfg,
                horizon=horizon,
                max_iter=solver_max_iter,
                mode=prediction_mode,
                mobile=mobile,
                surrogate_model=surrogate_model,
                device=device,
                last_solution=last_solution,
            )
            solve_times.append(float(perf_counter() - solve_start))
            applied_controls = solution["controls"][0]
            applied_forces = solution["forces"][0]
            solution_objective = float(solution["objective"][0])
        if mobile:
            positions, velocities = advance_dynamics_with_hold(dynamics, prev_p, prev_v, applied_forces, cfg)
        else:
            positions = prev_p.copy()
            velocities = np.zeros_like(prev_v)

        for _ in range(cfg.control_stride):
            state_now = pde_step(state_now, positions, applied_controls, grid, cfg)

        states_record.append(state_now.copy())
        controls_record.append(applied_controls.copy())
        forces_record.append(applied_forces.copy())
        positions_record.append(positions.copy())
        velocities_record.append(velocities.copy())
        objectives.append(float(solution_objective))

        avg_temp = float(np.mean(state_now))
        max_temp = float(np.max(state_now))
        if progress_every > 0 and ((step_idx + 1) % progress_every == 0 or step_idx == 0):
            print(
                f"[{controller_mode}|{prediction_mode}] step {step_idx + 1}/{cfg.control_steps} | "
                f"avg={avg_temp:.5f} max={max_temp:.5f}"
            )

        prev_u = applied_controls
        prev_f = applied_forces
        prev_p = positions
        prev_v = velocities
        if step_idx > 0:
            last_solution = solution

    states_np = np.asarray(states_record, dtype=np.float32)
    controls_np = np.asarray(controls_record, dtype=np.float32)
    forces_np = np.asarray(forces_record, dtype=np.float32)
    positions_np = np.asarray(positions_record, dtype=np.float32)
    velocities_np = np.asarray(velocities_record, dtype=np.float32)
    solve_times_np = np.asarray(solve_times, dtype=np.float32)

    title_prefix = "Mobile Actuators" if mobile else "Fixed Actuators"
    plot_state_snapshots(states_np, positions_np, grid, cfg, output_dir / "state_snapshots.png", title_prefix)
    plot_spacetime_surface_3d(
        states_np,
        grid,
        cfg,
        output_dir / "state_spacetime_3d.png",
        f"Closed-Loop State Surface ({title_prefix})",
    )
    plot_average_temperature(states_np, cfg, output_dir / "average_temperature.png", f"Average Temperature ({title_prefix})")
    plot_controls(controls_np, cfg, output_dir / "control_inputs.png", f"Control Inputs ({title_prefix})")
    plot_forces(forces_np, cfg, output_dir / "controller_forces.png", f"Actuator Forces ({title_prefix})")
    if mobile:
        plot_controls_and_forces(controls_np, forces_np, cfg, output_dir / "control_inputs_and_forces.png")
    plot_positions(positions_np, cfg, output_dir / "controller_positions.png", f"Actuator Positions ({title_prefix})")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "prediction_mode": prediction_mode,
                "controller_mode": controller_mode,
                "executed_steps": int(controls_np.shape[0]),
                "executed_time_s": float(controls_np.shape[0] * control_dt(cfg)),
                "final_average_temperature": float(average_temperature(states_np)[-1]),
                "final_max_temperature": float(np.max(states_np[-1])),
                "average_mpc_solve_time_s": float(np.mean(solve_times_np)) if solve_times_np.size > 0 else 0.0,
                "median_mpc_solve_time_s": float(np.median(solve_times_np)) if solve_times_np.size > 0 else 0.0,
                "total_mpc_solve_time_s": float(np.sum(solve_times_np)),
                "objectives": objectives,
                "solve_times_s": solve_times,
            },
            fh,
            indent=2,
        )
    result = {
        "states": states_np,
        "controls": controls_np,
        "forces": forces_np,
        "positions": positions_np,
        "velocities": velocities_np,
        "objectives": np.asarray(objectives, dtype=np.float32),
        "solve_times": solve_times_np,
    }
    save_case_cache(
        output_dir,
        result,
        build_case_cache_meta(cfg, prediction_mode, controller_mode, horizon, max_iter),
    )
    return result


def run_svd_rbf_case(
    cfg: RodCoolingConfig,
    grid: np.ndarray,
    output_dir: Path,
    baseline_checkpoint: Path,
    baseline_cfg: SVDRBFBaselineConfig,
    horizon: int,
    max_iter: int,
    progress_every: int,
) -> Dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_or_train_svd_rbf_bundle(cfg, baseline_checkpoint, baseline_cfg=baseline_cfg, verbose=True)
    dynamics = build_controller_dynamics(cfg, mobile=True)

    state_now = initial_state(grid, cfg)
    prev_u = np.zeros(cfg.controller_count, dtype=np.float32)
    prev_f = np.zeros(cfg.controller_count, dtype=np.float32)
    prev_p = initial_controller_positions(cfg)
    prev_v = np.zeros(cfg.controller_count, dtype=np.float32)
    state_history = [state_now.copy() for _ in range(baseline_cfg.delay_steps)]

    states_record = [state_now.copy()]
    controls_record = []
    forces_record = []
    positions_record = []
    velocities_record = []
    objectives = []
    solve_times = []
    last_solution: Optional[Dict[str, np.ndarray]] = None

    for step_idx in range(cfg.control_steps):
        history_np = np.asarray(state_history[-baseline_cfg.delay_steps :], dtype=np.float32)
        if step_idx == 0:
            applied_controls = np.zeros(cfg.controller_count, dtype=np.float32)
            applied_forces = np.zeros(cfg.controller_count, dtype=np.float32)
            applied_position, applied_velocity = advance_dynamics_with_hold(dynamics, prev_p, prev_v, applied_forces, cfg)
            solution_objective = 0.0
            solve_times.append(0.0)
        else:
            solver_max_iter = solver_iteration_budget(state_now, max_iter)
            solve_start = perf_counter()
            solution = optimize_svd_rbf_step(
                state_history=history_np,
                prev_u=prev_u,
                prev_f=prev_f,
                prev_pos=prev_p,
                prev_vel=prev_v,
                grid=grid,
                cfg=cfg,
                bundle=bundle,
                horizon=horizon,
                max_iter=solver_max_iter,
                last_solution=last_solution,
            )
            solve_times.append(float(perf_counter() - solve_start))

            applied_controls = solution["controls"][0]
            applied_forces = solution["forces"][0]
            applied_position, applied_velocity = advance_dynamics_with_hold(dynamics, prev_p, prev_v, applied_forces, cfg)
            solution_objective = float(solution["objective"][0])

        for _ in range(cfg.control_stride):
            state_now = pde_step(state_now, applied_position, applied_controls, grid, cfg)

        states_record.append(state_now.copy())
        controls_record.append(applied_controls.copy())
        forces_record.append(applied_forces.copy())
        positions_record.append(applied_position.copy())
        velocities_record.append(applied_velocity.copy())
        objectives.append(solution_objective)
        state_history.append(state_now.copy())

        avg_temp = float(np.mean(state_now))
        max_temp = float(np.max(state_now))
        if progress_every > 0 and ((step_idx + 1) % progress_every == 0 or step_idx == 0):
            print(
                f"[svd-rbf] step {step_idx + 1}/{cfg.control_steps} | "
                f"avg={avg_temp:.5f} max={max_temp:.5f}"
            )

        prev_u = applied_controls
        prev_f = applied_forces
        prev_p = applied_position
        prev_v = applied_velocity
        if step_idx > 0:
            last_solution = solution

    states_np = np.asarray(states_record, dtype=np.float32)
    controls_np = np.asarray(controls_record, dtype=np.float32)
    forces_np = np.asarray(forces_record, dtype=np.float32)
    positions_np = np.asarray(positions_record, dtype=np.float32)
    velocities_np = np.asarray(velocities_record, dtype=np.float32)
    solve_times_np = np.asarray(solve_times, dtype=np.float32)

    plot_state_snapshots(
        states_np,
        positions_np,
        grid,
        cfg,
        output_dir / "state_snapshots.png",
        "SVD-RBFNN Mobile Actuators",
    )
    plot_spacetime_surface_3d(
        states_np,
        grid,
        cfg,
        output_dir / "state_spacetime_3d.png",
        "Closed-Loop State Surface (SVD-RBFNN Mobile Actuators)",
    )
    plot_average_temperature(
        states_np,
        cfg,
        output_dir / "average_temperature.png",
        "Average Temperature (SVD-RBFNN Mobile Actuators)",
    )
    plot_controls(
        controls_np,
        cfg,
        output_dir / "control_inputs.png",
        "Control Inputs (SVD-RBFNN Mobile Actuators)",
    )
    plot_forces(
        forces_np,
        cfg,
        output_dir / "controller_forces.png",
        "Actuator Forces (SVD-RBFNN Mobile Actuators)",
    )
    plot_positions(
        positions_np,
        cfg,
        output_dir / "controller_positions.png",
        "Actuator Positions (SVD-RBFNN Mobile Actuators)",
    )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "prediction_mode": "svd-rbf",
                "controller_mode": "mobile",
                "executed_steps": int(controls_np.shape[0]),
                "executed_time_s": float(controls_np.shape[0] * control_dt(cfg)),
                "final_average_temperature": float(average_temperature(states_np)[-1]),
                "final_max_temperature": float(np.max(states_np[-1])),
                "average_mpc_solve_time_s": float(np.mean(solve_times_np)) if solve_times_np.size > 0 else 0.0,
                "median_mpc_solve_time_s": float(np.median(solve_times_np)) if solve_times_np.size > 0 else 0.0,
                "total_mpc_solve_time_s": float(np.sum(solve_times_np)),
                "objectives": objectives,
                "solve_times_s": solve_times,
                "svd_rbf_checkpoint": str(baseline_checkpoint),
            },
            fh,
            indent=2,
        )

    result = {
        "states": states_np,
        "controls": controls_np,
        "forces": forces_np,
        "positions": positions_np,
        "velocities": velocities_np,
        "objectives": np.asarray(objectives, dtype=np.float32),
        "solve_times": solve_times_np,
    }
    save_case_cache(
        output_dir,
        result,
        build_case_cache_meta(cfg, "svd-rbf", "mobile", horizon, max_iter, baseline_cfg=baseline_cfg),
    )
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = device_for_run(args.device)
    surrogate_model = None
    runtime_defaults = RodCoolingConfig()
    cfg = RodCoolingConfig(control_steps=args.control_steps, prediction_horizon=args.horizon, max_iter=args.max_iter)
    if args.prediction_mode == "surrogate":
        surrogate_model, checkpoint = load_checkpoint(str(args.checkpoint), device)
        cfg_dict = dict(checkpoint["config"])
        cfg_dict["control_steps"] = args.control_steps
        cfg_dict["prediction_horizon"] = args.horizon
        cfg_dict["max_iter"] = args.max_iter
        cfg_dict["control_stride"] = runtime_defaults.control_stride
        cfg_dict["temperature_display_scale"] = runtime_defaults.temperature_display_scale
        cfg_dict["control_min"] = runtime_defaults.control_min
        cfg_dict["control_max"] = runtime_defaults.control_max
        cfg_dict["control_delta_max"] = runtime_defaults.control_delta_max
        cfg_dict["force_limit"] = runtime_defaults.force_limit
        cfg_dict["force_delta_max"] = runtime_defaults.force_delta_max
        cfg_dict["position_min"] = runtime_defaults.position_min
        cfg_dict["position_max"] = runtime_defaults.position_max
        cfg_dict["min_separation"] = runtime_defaults.min_separation
        cfg_dict["mass_kg"] = runtime_defaults.mass_kg
        cfg_dict["damping_per_s"] = runtime_defaults.damping_per_s
        cfg = RodCoolingConfig(**cfg_dict)
    grid = make_spatial_grid(cfg)
    plot_initial_condition(initial_state(grid, cfg), initial_controller_positions(cfg), grid, args.output_dir / "initial_condition.png")

    baseline_cfg = SVDRBFBaselineConfig(
        delay_steps=args.svd_rbf_delay_steps,
        hidden_units=args.svd_rbf_hidden_units,
        gamma=args.svd_rbf_gamma,
        training_trajectories=args.svd_rbf_train_trajectories,
        training_steps=args.svd_rbf_train_steps,
        position_delta_max=args.svd_rbf_position_delta_max,
    )

    def get_standard_case(case_name: str, output_dir: Path, require_cache: bool = False) -> Dict[str, np.ndarray]:
        if not args.force_rerun:
            cached = try_load_case_cache(
                output_dir,
                build_case_cache_meta(cfg, args.prediction_mode, case_name, args.horizon, args.max_iter),
            )
            if cached is not None:
                print(f"[cache] loaded {case_name} from {output_dir}")
                return cached
        if require_cache:
            raise RuntimeError(f"Missing cached result for {case_name}: {output_dir}")
        return run_case(
            cfg,
            grid,
            output_dir,
            args.prediction_mode,
            case_name,
            surrogate_model,
            device,
            args.horizon,
            args.max_iter,
            args.progress_every,
        )

    def get_svd_case(output_dir: Path, require_cache: bool = False) -> Dict[str, np.ndarray]:
        if not args.force_rerun:
            cached = try_load_case_cache(
                output_dir,
                build_case_cache_meta(cfg, "svd-rbf", "mobile", args.horizon, args.max_iter, baseline_cfg=baseline_cfg),
            )
            if cached is not None:
                print(f"[cache] loaded svd_rbf_mobile from {output_dir}")
                return cached
        if require_cache:
            raise RuntimeError(f"Missing cached result for svd_rbf_mobile: {output_dir}")
        return run_svd_rbf_case(
            cfg,
            grid,
            output_dir,
            args.svd_rbf_checkpoint,
            baseline_cfg,
            args.horizon,
            args.max_iter,
            args.progress_every,
        )

    if args.controller_mode == "compare":
        mobile_result = get_standard_case("mobile", args.output_dir / "mobile", require_cache=args.build_comparison_only)
        fixed_result = get_standard_case("fixed", args.output_dir / "fixed", require_cache=args.build_comparison_only)
        svd_rbf_result = get_svd_case(args.output_dir / "svd_rbf_mobile", require_cache=args.build_comparison_only)
        surrogate_eval_model = surrogate_model
        if surrogate_eval_model is None:
            surrogate_eval_model, _ = load_checkpoint(str(args.checkpoint), device)
        svd_bundle = load_or_train_svd_rbf_bundle(cfg, args.svd_rbf_checkpoint, baseline_cfg=baseline_cfg, verbose=True)
        plot_average_comparison(
            [
                ("PINN-ODE mobile actuators", mobile_result["states"], {}),
                ("Fixed actuators", fixed_result["states"], {"linestyle": "--"}),
            ],
            cfg,
            args.output_dir / "average_temperature_comparison.png",
        )
        plot_spacetime_surface_3d_comparison(
            mobile_result["states"],
            fixed_result["states"],
            grid,
            cfg,
            args.output_dir / "state_spacetime_3d_mobile_fixed_comparison.png",
        )
        model_error_curves = build_model_vs_truth_error_curves(
            cfg,
            surrogate_eval_model,
            device,
            svd_bundle,
        )
        plot_pinn_vs_svd_error(
            model_error_curves,
            cfg,
            args.output_dir / "pinn_vs_svd_error.png",
        )
        plot_controls_and_forces(
            mobile_result["controls"],
            mobile_result["forces"],
            cfg,
            args.output_dir / "control_inputs_and_forces.png",
        )
        with (args.output_dir / "comparison_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "mobile_final_average_temperature": float(average_temperature(mobile_result["states"])[-1]),
                    "fixed_final_average_temperature": float(average_temperature(fixed_result["states"])[-1]),
                    "svd_rbf_final_average_temperature": float(average_temperature(svd_rbf_result["states"])[-1]),
                    "mobile_average_mpc_solve_time_s": float(np.mean(mobile_result["solve_times"])),
                    "fixed_average_mpc_solve_time_s": float(np.mean(fixed_result["solve_times"])),
                    "svd_rbf_average_mpc_solve_time_s": float(np.mean(svd_rbf_result["solve_times"])),
                    "mobile_median_mpc_solve_time_s": float(np.median(mobile_result["solve_times"])),
                    "fixed_median_mpc_solve_time_s": float(np.median(fixed_result["solve_times"])),
                    "svd_rbf_median_mpc_solve_time_s": float(np.median(svd_rbf_result["solve_times"])),
                    "pinn_avg_temp_abs_error_vs_pde": float(np.mean(model_error_curves["pinn_avg_abs"])),
                    "svd_avg_temp_abs_error_vs_pde": float(np.mean(model_error_curves["svd_avg_abs"])),
                    "pinn_state_rmse_vs_pde": float(np.mean(model_error_curves["pinn_rmse"])),
                    "svd_state_rmse_vs_pde": float(np.mean(model_error_curves["svd_rmse"])),
                },
                fh,
                indent=2,
            )
        return

    if args.controller_mode == "svd-rbf":
        get_svd_case(args.output_dir)
        return

    get_standard_case(args.controller_mode, args.output_dir)


if __name__ == "__main__":
    main()
