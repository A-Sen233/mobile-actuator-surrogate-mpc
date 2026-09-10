from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from rod_cooling_1d.core import (
    ControllerDynamics1D,
    RodCoolingConfig,
    apply_dirichlet,
    build_controller_dynamics,
    build_source_map,
    enforce_min_separation_1d,
    initial_controller_positions,
    initial_state,
    make_spatial_grid,
    pde_step,
    project_control_step,
    project_force_step,
    sample_random_trajectory,
    simulate_trajectory,
)


@dataclass(frozen=True)
class SVDRBFBaselineConfig:
    delay_steps: int = 5
    energy_ratio: float = 0.995
    hidden_units: int = 180
    gamma: float = 0.25
    training_trajectories: int = 72
    training_steps: int = 240
    position_delta_max: float = 0.03
    seed: int = 1234

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PaperLikeCostConfig:
    high_state_blend: float = 0.7
    force_gain: float = 120.0
    velocity_gain: float = 12.0
    state_weight: float = 70.0
    terminal_state_weight: float = 105.0
    target_weight: float = 34.0
    terminal_target_weight: float = 60.0
    control_weight: float = 0.16
    force_weight: float = 0.006
    quiet_control_weight: float = 0.8
    quiet_force_weight: float = 0.008
    quiet_velocity_weight: float = 0.12
    terminal_quiet_control_weight: float = 2.4
    terminal_quiet_force_weight: float = 0.012
    terminal_quiet_velocity_weight: float = 0.18
    source_overlap_weight: float = 9.0
    activity_eps: float = 4e-3


class RBFNN:
    def __init__(self, n_hidden: int = 180, gamma: float = 0.25, random_state: int = 42) -> None:
        self.n_hidden = int(n_hidden)
        self.gamma = float(gamma)
        self.random_state = int(random_state)
        self.centers: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

    def _scale_features(self, values: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.scaler_mean = np.mean(values, axis=0)
            self.scaler_std = np.std(values, axis=0)
            self.scaler_std[self.scaler_std < 1e-10] = 1.0
        if self.scaler_mean is None or self.scaler_std is None:
            return values
        return (values - self.scaler_mean) / self.scaler_std

    def _rbf(self, values: np.ndarray, center: np.ndarray) -> np.ndarray:
        return np.exp(-self.gamma * np.linalg.norm(values - center, axis=1) ** 2)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        scaled = self._scale_features(features, fit=True)
        rng = np.random.default_rng(self.random_state)
        hidden = min(self.n_hidden, scaled.shape[0])
        indices = rng.choice(scaled.shape[0], hidden, replace=False)
        self.centers = scaled[indices]

        hidden_matrix = np.zeros((scaled.shape[0], hidden), dtype=np.float64)
        for idx in range(hidden):
            hidden_matrix[:, idx] = self._rbf(scaled, self.centers[idx])
        hidden_with_bias = np.column_stack([hidden_matrix, np.ones(scaled.shape[0])])
        solution = np.linalg.lstsq(hidden_with_bias, targets, rcond=None)[0]
        self.bias = solution[-1]
        self.weights = solution[:-1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.centers is None or self.weights is None or self.bias is None:
            raise RuntimeError("RBFNN must be fitted before predict().")
        scaled = self._scale_features(features, fit=False)
        hidden_matrix = np.zeros((scaled.shape[0], self.centers.shape[0]), dtype=np.float64)
        for idx in range(self.centers.shape[0]):
            hidden_matrix[:, idx] = self._rbf(scaled, self.centers[idx])
        return hidden_matrix @ self.weights + self.bias


def baseline_signature(cfg: RodCoolingConfig) -> Dict[str, float]:
    initial_positions = initial_controller_positions(cfg)
    return {
        "rod_length_m": cfg.rod_length_m,
        "diffusion": cfg.diffusion,
        "reaction": cfg.reaction,
        "controller_count": cfg.controller_count,
        "controller_sigma_m": cfg.controller_sigma_m,
        "initial_width_m": cfg.initial_width_m,
        "initial_amplitude": cfg.initial_amplitude,
        "grid_points": cfg.grid_points,
        "dt": cfg.dt,
        "control_min": cfg.control_min,
        "control_max": cfg.control_max,
        "control_delta_max": cfg.control_delta_max,
        "force_limit": cfg.force_limit,
        "force_delta_max": cfg.force_delta_max,
        "position_min": cfg.position_min,
        "position_max": cfg.position_max,
        "min_separation": cfg.min_separation,
        "mass_kg": cfg.mass_kg,
        "damping_per_s": cfg.damping_per_s,
        "initial_position_1": float(initial_positions[0]) if initial_positions.size > 0 else 0.0,
        "initial_position_2": float(initial_positions[1]) if initial_positions.size > 1 else 0.0,
        "initial_position_3": float(initial_positions[2]) if initial_positions.size > 2 else 0.0,
    }


def perform_svd(data_matrix: np.ndarray, energy_ratio: float) -> Tuple[StandardScaler, TruncatedSVD]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_matrix)
    tsvd_full = TruncatedSVD(n_components=min(data_matrix.shape))
    tsvd_full.fit(scaled)
    cumulative_energy = np.cumsum(tsvd_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative_energy >= energy_ratio) + 1)
    tsvd = TruncatedSVD(n_components=n_components)
    tsvd.fit(scaled)
    return scaler, tsvd


def build_position_features(positions: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    z_min, z_max = float(np.min(grid)), float(np.max(grid))
    span = max(z_max - z_min, 1e-12)
    pos_norm = (np.asarray(positions, dtype=np.float32) - z_min) / span
    width_norm = np.full_like(pos_norm, sigma / span, dtype=np.float32)
    return np.concatenate([pos_norm, width_norm]).astype(np.float32)


def region_masks(grid: np.ndarray, cfg: RodCoolingConfig) -> List[np.ndarray]:
    edges = np.linspace(cfg.position_min, cfg.position_max, cfg.controller_count + 1, dtype=np.float32)
    masks = []
    for idx in range(cfg.controller_count):
        if idx == cfg.controller_count - 1:
            mask = (grid >= edges[idx]) & (grid <= edges[idx + 1])
        else:
            mask = (grid >= edges[idx]) & (grid < edges[idx + 1])
        masks.append(mask)
    return masks


def state_activity_scale(state: np.ndarray, reference_rms: float = 0.28) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(state, dtype=np.float32) ** 2)))
    return float(np.clip(rms / max(reference_rms, 1e-6), 0.04, 1.0))


def rollout_dynamics(
    dynamics: ControllerDynamics1D,
    positions: np.ndarray,
    velocities: np.ndarray,
    forces_seq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_state = positions.astype(np.float32, copy=True)
    vel_state = velocities.astype(np.float32, copy=True)
    pos_hist = np.zeros((forces_seq.shape[0], positions.shape[0]), dtype=np.float32)
    vel_hist = np.zeros_like(pos_hist)
    for step_idx in range(forces_seq.shape[0]):
        pos_state, vel_state = dynamics.step(pos_state, vel_state, forces_seq[step_idx])
        pos_hist[step_idx] = pos_state
        vel_hist[step_idx] = vel_state
    return pos_hist, vel_hist


def sample_force_driven_rollout(
    cfg: RodCoolingConfig,
    baseline_cfg: SVDRBFBaselineConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    rollout = sample_random_trajectory(cfg, steps=baseline_cfg.training_steps, rng=rng, mobile=True)
    simulation = simulate_trajectory(cfg, rollout["controls"], rollout["positions"])
    return {
        "controls": rollout["controls"],
        "forces": rollout["forces"],
        "positions": rollout["positions"],
        "velocities": rollout["velocities"],
        "states": simulation["states"],
    }


def build_training_matrices(
    cfg: RodCoolingConfig,
    baseline_cfg: SVDRBFBaselineConfig,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, TruncatedSVD, np.ndarray]:
    rng = np.random.default_rng(baseline_cfg.seed)
    rollouts = [
        sample_force_driven_rollout(cfg, baseline_cfg, rng)
        for _ in range(baseline_cfg.training_trajectories)
    ]
    all_states = np.concatenate([rollout["states"] for rollout in rollouts], axis=0)
    scaler, tsvd = perform_svd(all_states, baseline_cfg.energy_ratio)
    feature_list = []
    target_list = []
    grid = make_spatial_grid(cfg)

    for rollout in rollouts:
        states = rollout["states"]
        controls = rollout["controls"]
        positions = rollout["positions"]
        coeffs = tsvd.transform(scaler.transform(states))
        for step_idx in range(baseline_cfg.delay_steps - 1, controls.shape[0]):
            history = coeffs[step_idx - baseline_cfg.delay_steps + 1 : step_idx + 1]
            feature = np.hstack(
                [
                    history.flatten(),
                    controls[step_idx],
                    build_position_features(positions[step_idx], grid, cfg.controller_sigma_m),
                ]
            )
            feature_list.append(feature.astype(np.float32))
            target_list.append(coeffs[step_idx + 1].astype(np.float32))

    return (
        np.asarray(feature_list, dtype=np.float32),
        np.asarray(target_list, dtype=np.float32),
        scaler,
        tsvd,
        grid,
    )


def heuristic_force_guess(
    positions: np.ndarray,
    velocities: np.ndarray,
    targets: np.ndarray,
    cfg: RodCoolingConfig,
    paper_cfg: PaperLikeCostConfig,
    activity_scale: float,
) -> np.ndarray:
    forces = activity_scale * (paper_cfg.force_gain * (targets - positions) - paper_cfg.velocity_gain * velocities)
    return np.clip(forces, -cfg.force_limit, cfg.force_limit).astype(np.float32)


def train_svd_rbf_bundle(
    cfg: RodCoolingConfig,
    baseline_cfg: SVDRBFBaselineConfig,
    checkpoint_path: Path,
    verbose: bool = True,
) -> Dict[str, object]:
    features, targets, scaler, tsvd, grid = build_training_matrices(cfg, baseline_cfg)
    split = max(1, int(0.9 * features.shape[0]))
    model = RBFNN(
        n_hidden=min(baseline_cfg.hidden_units, split),
        gamma=baseline_cfg.gamma,
        random_state=baseline_cfg.seed,
    )
    model.fit(features[:split], targets[:split])
    train_mse = float(np.mean((model.predict(features[:split]) - targets[:split]) ** 2))
    val_mse = float(np.mean((model.predict(features[split:]) - targets[split:]) ** 2)) if split < features.shape[0] else train_mse

    bundle: Dict[str, object] = {
        "rbf_model": model,
        "svd_model": tsvd,
        "scaler": scaler,
        "grid": grid,
        "delay_steps": baseline_cfg.delay_steps,
        "cfg_signature": baseline_signature(cfg),
        "baseline_config": baseline_cfg.to_dict(),
        "train_mse": train_mse,
        "val_mse": val_mse,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, checkpoint_path)
    if verbose:
        print(
            f"[svd-rbf] trained baseline | train_mse={train_mse:.6e} | "
            f"val_mse={val_mse:.6e} | saved={checkpoint_path}"
        )
    return bundle


def load_or_train_svd_rbf_bundle(
    cfg: RodCoolingConfig,
    checkpoint_path: Path,
    baseline_cfg: Optional[SVDRBFBaselineConfig] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    baseline_cfg = baseline_cfg or SVDRBFBaselineConfig()
    if checkpoint_path.exists():
        bundle = joblib.load(checkpoint_path)
        if bundle.get("cfg_signature") == baseline_signature(cfg):
            if verbose:
                print(f"[svd-rbf] loaded baseline checkpoint: {checkpoint_path}")
            return bundle
        if verbose:
            print("[svd-rbf] checkpoint config mismatch, retraining baseline...")
    return train_svd_rbf_bundle(cfg, baseline_cfg, checkpoint_path, verbose=verbose)


def state_to_coeff(state: np.ndarray, bundle: Dict[str, object]) -> np.ndarray:
    scaler: StandardScaler = bundle["scaler"]  # type: ignore[assignment]
    tsvd: TruncatedSVD = bundle["svd_model"]  # type: ignore[assignment]
    return tsvd.transform(scaler.transform(np.asarray(state, dtype=np.float32).reshape(1, -1)))[0].astype(np.float32)


def coeff_to_state(coeffs: np.ndarray, bundle: Dict[str, object], grid_points: int) -> np.ndarray:
    scaler: StandardScaler = bundle["scaler"]  # type: ignore[assignment]
    tsvd: TruncatedSVD = bundle["svd_model"]  # type: ignore[assignment]
    scaled = tsvd.inverse_transform(coeffs.reshape(1, -1))[0]
    full_state = np.zeros(grid_points, dtype=np.float32)
    full_state[: min(grid_points, scaled.shape[0])] = scaled[:grid_points]
    restored = scaler.inverse_transform(full_state.reshape(1, -1))[0]
    return apply_dirichlet(np.maximum(restored, 0.0).astype(np.float32))


def svd_rbf_predict_rollout(
    bundle: Dict[str, object],
    state_history: np.ndarray,
    controls_seq: np.ndarray,
    positions_seq: np.ndarray,
    cfg: RodCoolingConfig,
) -> np.ndarray:
    model: RBFNN = bundle["rbf_model"]  # type: ignore[assignment]
    grid: np.ndarray = bundle["grid"]  # type: ignore[assignment]
    delay_steps = int(bundle["delay_steps"])

    coeff_history = [state_to_coeff(state, bundle) for state in state_history[-delay_steps:]]
    current_state = state_history[-1].astype(np.float32)
    predicted_states = [current_state.copy()]

    for step_idx in range(controls_seq.shape[0]):
        history = np.asarray(coeff_history[-delay_steps:], dtype=np.float32).flatten()
        feature = np.hstack(
            [
                history,
                controls_seq[step_idx].astype(np.float32),
                build_position_features(positions_seq[step_idx], grid, cfg.controller_sigma_m),
            ]
        ).reshape(1, -1)
        next_coeff = model.predict(feature)[0].astype(np.float32)
        next_state = coeff_to_state(next_coeff, bundle, cfg.grid_points)
        coeff_history.append(next_coeff)
        predicted_states.append(next_state)

    return np.asarray(predicted_states, dtype=np.float32)


def peak_targets_from_state(state: np.ndarray, grid: np.ndarray, cfg: RodCoolingConfig) -> np.ndarray:
    targets = np.zeros(cfg.controller_count, dtype=np.float32)
    for idx, mask in enumerate(region_masks(grid, cfg)):
        region_values = state[mask]
        region_grid = grid[mask]
        if region_values.size == 0:
            targets[idx] = float((idx + 0.5) * cfg.rod_length_m / cfg.controller_count)
        else:
            targets[idx] = float(region_grid[int(np.argmax(region_values))])
    return targets


def compute_dynamic_tracking_targets(states: np.ndarray, grid: np.ndarray, cfg: RodCoolingConfig) -> np.ndarray:
    targets = np.zeros((states.shape[0], cfg.controller_count), dtype=np.float32)
    for step_idx, state in enumerate(states):
        for ctrl_idx, mask in enumerate(region_masks(grid, cfg)):
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


def build_initial_guess(
    horizon: int,
    cfg: RodCoolingConfig,
    prev_u: np.ndarray,
    prev_f: np.ndarray,
    prev_pos: np.ndarray,
    prev_vel: np.ndarray,
    state_now: np.ndarray,
    grid: np.ndarray,
    last_solution: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    paper_cfg = PaperLikeCostConfig()
    activity_scale = state_activity_scale(state_now)
    targets = peak_targets_from_state(state_now, grid, cfg)
    heuristic_u = np.full(
        cfg.controller_count,
        -abs(cfg.control_min) * (0.15 + 0.45 * activity_scale),
        dtype=np.float32,
    )
    heuristic_f = heuristic_force_guess(prev_pos, prev_vel, targets, cfg, paper_cfg, activity_scale)

    if last_solution is None:
        control_guess = np.tile(0.2 * prev_u + 0.8 * heuristic_u, (horizon, 1)).astype(np.float32)
        force_guess = np.tile(0.15 * prev_f + 0.85 * heuristic_f, (horizon, 1)).astype(np.float32)
    else:
        shifted_controls = np.vstack([last_solution["controls"][1:], last_solution["controls"][-1:]]).astype(np.float32)
        shifted_forces = np.vstack([last_solution["forces"][1:], last_solution["forces"][-1:]]).astype(np.float32)
        heuristic_control_seq = np.tile(heuristic_u, (horizon, 1)).astype(np.float32)
        heuristic_force_seq = np.tile(heuristic_f, (horizon, 1)).astype(np.float32)
        control_guess = (0.7 * shifted_controls + 0.3 * heuristic_control_seq).astype(np.float32)
        force_guess = (0.7 * shifted_forces + 0.3 * heuristic_force_seq).astype(np.float32)

    projected_controls = np.zeros_like(control_guess, dtype=np.float32)
    projected_forces = np.zeros_like(force_guess, dtype=np.float32)
    last_u = prev_u.astype(np.float32, copy=True)
    last_f = prev_f.astype(np.float32, copy=True)
    for idx in range(horizon):
        projected_controls[idx] = project_control_step(control_guess[idx], last_u, cfg)
        projected_forces[idx] = project_force_step(force_guess[idx], last_f, cfg, mobile=True)
        last_u = projected_controls[idx]
        last_f = projected_forces[idx]
    return np.concatenate([projected_controls.ravel(), projected_forces.ravel()]).astype(np.float64)


def unpack_vector(vector: np.ndarray, horizon: int, cfg: RodCoolingConfig) -> Tuple[np.ndarray, np.ndarray]:
    control_size = horizon * cfg.controller_count
    controls = vector[:control_size].reshape(horizon, cfg.controller_count)
    forces = vector[control_size:].reshape(horizon, cfg.controller_count)
    return controls.astype(np.float32), forces.astype(np.float32)


def delta_constraints(
    vector: np.ndarray,
    horizon: int,
    cfg: RodCoolingConfig,
    prev_u: np.ndarray,
    prev_f: np.ndarray,
) -> np.ndarray:
    controls, forces = unpack_vector(vector, horizon, cfg)
    constraints = []
    for step_idx in range(horizon):
        base_u = prev_u if step_idx == 0 else controls[step_idx - 1]
        constraints.append(cfg.control_delta_max - np.abs(controls[step_idx] - base_u))
        base_f = prev_f if step_idx == 0 else forces[step_idx - 1]
        constraints.append(cfg.force_delta_max - np.abs(forces[step_idx] - base_f))
    return np.concatenate([np.atleast_1d(item).astype(np.float32) for item in constraints])


def separation_constraints(
    vector: np.ndarray,
    horizon: int,
    cfg: RodCoolingConfig,
    prev_f: np.ndarray,
    prev_pos: np.ndarray,
    prev_vel: np.ndarray,
    dynamics: ControllerDynamics1D,
) -> np.ndarray:
    _, forces = unpack_vector(vector, horizon, cfg)
    projected = np.zeros_like(forces, dtype=np.float32)
    previous_f = prev_f.astype(np.float32, copy=True)
    for step_idx in range(forces.shape[0]):
        projected[step_idx] = project_force_step(forces[step_idx], previous_f, cfg, mobile=True)
        previous_f = projected[step_idx]
    predicted_positions, _ = rollout_dynamics(dynamics, prev_pos, prev_vel, projected)
    margins = []
    for step_idx in range(predicted_positions.shape[0]):
        for ctrl_idx in range(predicted_positions.shape[1] - 1):
            margins.append(predicted_positions[step_idx, ctrl_idx + 1] - predicted_positions[step_idx, ctrl_idx] - cfg.min_separation)
    return np.asarray(margins, dtype=np.float32) if margins else np.array([1.0], dtype=np.float32)


def paper_like_objective(
    state_history: np.ndarray,
    prev_u: np.ndarray,
    prev_f: np.ndarray,
    prev_pos: np.ndarray,
    prev_vel: np.ndarray,
    controls_seq: np.ndarray,
    forces_seq: np.ndarray,
    grid: np.ndarray,
    cfg: RodCoolingConfig,
    bundle: Dict[str, object],
    paper_cfg: PaperLikeCostConfig,
    dynamics: ControllerDynamics1D,
) -> float:
    projected_controls = np.zeros_like(controls_seq, dtype=np.float32)
    previous_u = prev_u.astype(np.float32, copy=True)
    for step_idx in range(controls_seq.shape[0]):
        projected_controls[step_idx] = project_control_step(controls_seq[step_idx], previous_u, cfg)
        previous_u = projected_controls[step_idx]
    projected_forces = np.zeros_like(forces_seq, dtype=np.float32)
    previous_f = prev_f.astype(np.float32, copy=True)
    for step_idx in range(forces_seq.shape[0]):
        projected_forces[step_idx] = project_force_step(forces_seq[step_idx], previous_f, cfg, mobile=True)
        previous_f = projected_forces[step_idx]

    projected_positions, projected_velocities = rollout_dynamics(dynamics, prev_pos, prev_vel, projected_forces)

    predicted_states = svd_rbf_predict_rollout(bundle, state_history, projected_controls, projected_positions, cfg)
    dynamic_targets = compute_dynamic_tracking_targets(predicted_states[1:], grid, cfg)
    state_energy_seq = np.mean(predicted_states[1:] ** 2, axis=1).astype(np.float64)
    activity_weights = state_energy_seq / (state_energy_seq + paper_cfg.activity_eps)
    quiet_weights = 1.0 - activity_weights
    state_cost = float(np.mean(predicted_states[1:-1] ** 2)) if predicted_states.shape[0] > 2 else 0.0
    terminal_cost = float(np.mean(predicted_states[-1] ** 2))
    control_energy_seq = np.mean(projected_controls ** 2, axis=1).astype(np.float64)
    force_energy_seq = np.mean(projected_forces ** 2, axis=1).astype(np.float64)
    velocity_energy_seq = np.mean(projected_velocities ** 2, axis=1).astype(np.float64)
    tracking_error_seq = np.mean((projected_positions - dynamic_targets) ** 2, axis=1).astype(np.float64)
    control_cost = float(np.mean(np.abs(projected_controls)) + 0.35 * np.mean(control_energy_seq))
    force_cost = float(np.mean(force_energy_seq))
    target_cost = float(np.mean(activity_weights * tracking_error_seq))
    terminal_target_cost = float(activity_weights[-1] * tracking_error_seq[-1]) if tracking_error_seq.size > 0 else 0.0
    source_overlap_reward = 0.0
    for step_idx in range(projected_controls.shape[0]):
        source_map = build_source_map(grid, projected_positions[step_idx], projected_controls[step_idx], cfg)
        source_overlap_reward += float(activity_weights[step_idx] * np.mean((-source_map) * predicted_states[step_idx + 1]))
    quiescent_control_cost = float(np.mean(quiet_weights * control_energy_seq))
    quiescent_force_cost = float(np.mean(quiet_weights * force_energy_seq))
    quiescent_velocity_cost = float(np.mean(quiet_weights * velocity_energy_seq))
    terminal_control_cost = float(control_energy_seq[-1])
    terminal_force_cost = float(force_energy_seq[-1])
    terminal_velocity_cost = float(velocity_energy_seq[-1])
    terminal_quiet_weight = float(quiet_weights[-1]) if quiet_weights.size > 0 else 0.0

    return float(
        paper_cfg.state_weight * state_cost
        + paper_cfg.terminal_state_weight * terminal_cost
        + paper_cfg.target_weight * target_cost
        + paper_cfg.terminal_target_weight * terminal_target_cost
        + paper_cfg.control_weight * control_cost
        + paper_cfg.force_weight * force_cost
        + paper_cfg.quiet_control_weight * quiescent_control_cost
        + paper_cfg.quiet_force_weight * quiescent_force_cost
        + paper_cfg.quiet_velocity_weight * quiescent_velocity_cost
        + terminal_quiet_weight
        * (
            paper_cfg.terminal_quiet_control_weight * terminal_control_cost
            + paper_cfg.terminal_quiet_force_weight * terminal_force_cost
            + paper_cfg.terminal_quiet_velocity_weight * terminal_velocity_cost
        )
        - paper_cfg.source_overlap_weight * source_overlap_reward
    )


def optimize_svd_rbf_step(
    state_history: np.ndarray,
    prev_u: np.ndarray,
    prev_f: np.ndarray,
    prev_pos: np.ndarray,
    prev_vel: np.ndarray,
    grid: np.ndarray,
    cfg: RodCoolingConfig,
    bundle: Dict[str, object],
    horizon: int,
    max_iter: int,
    last_solution: Optional[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    paper_cfg = PaperLikeCostConfig()
    dynamics = build_controller_dynamics(cfg, mobile=True)
    initial_guess = build_initial_guess(
        horizon,
        cfg,
        prev_u,
        prev_f,
        prev_pos,
        prev_vel,
        state_history[-1],
        grid,
        last_solution,
    )
    bounds = [(cfg.control_min, cfg.control_max)] * (horizon * cfg.controller_count)
    bounds.extend([(-cfg.force_limit, cfg.force_limit)] * (horizon * cfg.controller_count))

    def objective(vector: np.ndarray) -> float:
        controls, forces = unpack_vector(vector, horizon, cfg)
        return paper_like_objective(
            state_history,
            prev_u,
            prev_f,
            prev_pos,
            prev_vel,
            controls,
            forces,
            grid,
            cfg,
            bundle,
            paper_cfg,
            dynamics,
        )

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: delta_constraints(x, horizon, cfg, prev_u, prev_f),
        },
        {
            "type": "ineq",
            "fun": lambda x: separation_constraints(x, horizon, cfg, prev_f, prev_pos, prev_vel, dynamics),
        },
    ]

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-5, "disp": False},
    )

    raw_controls, raw_forces = unpack_vector(result.x, horizon, cfg)
    controls = np.zeros_like(raw_controls, dtype=np.float32)
    forces = np.zeros_like(raw_forces, dtype=np.float32)
    previous_u = prev_u.astype(np.float32, copy=True)
    previous_f = prev_f.astype(np.float32, copy=True)
    for step_idx in range(raw_controls.shape[0]):
        controls[step_idx] = project_control_step(raw_controls[step_idx], previous_u, cfg)
        forces[step_idx] = project_force_step(raw_forces[step_idx], previous_f, cfg, mobile=True)
        previous_u = controls[step_idx]
        previous_f = forces[step_idx]
    positions, velocities = rollout_dynamics(dynamics, prev_pos, prev_vel, forces)
    predicted_states = svd_rbf_predict_rollout(bundle, state_history, controls, positions, cfg)
    return {
        "controls": controls,
        "forces": forces,
        "positions": positions,
        "velocities": velocities,
        "states": predicted_states,
        "objective": np.asarray([float(objective(result.x))], dtype=np.float32),
    }
