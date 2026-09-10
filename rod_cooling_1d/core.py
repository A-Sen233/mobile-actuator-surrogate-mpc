from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class RodCoolingConfig:
    rod_length_m: float = 3.0
    diffusion: float = 0.5
    reaction: float = -1.2
    controller_count: int = 3
    controller_sigma_m: float = 0.15
    initial_constant: float = 0.26
    initial_width_m: float = 0.3
    initial_amplitude: float = 0.5
    temperature_display_scale: float = 100.0
    grid_points: int = 50
    dt: float = 0.002
    control_steps: int = 480
    control_stride: int = 1
    prediction_horizon: int = 10
    max_iter: int = 12
    control_min: float = -10.0
    control_max: float = 0.0
    control_delta_max: float = 3.0
    force_limit: float = 100.0
    force_delta_max: float = 30.0
    position_min: float = 0.0
    position_max: float = 3.0
    min_separation: float = 0.6
    initial_controller_offset_m: float = 0.2
    mass_kg: float = 4.0
    damping_per_s: float = 0.05
    velocity_limit_m_s: Optional[float] = None
    training_trajectories: int = 240
    evaluation_trajectories: int = 6
    training_seed: int = 1234

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def make_config(**overrides: float) -> RodCoolingConfig:
    return RodCoolingConfig(**overrides)


def make_spatial_grid(cfg: RodCoolingConfig) -> np.ndarray:
    return np.linspace(0.0, cfg.rod_length_m, cfg.grid_points, dtype=np.float32)


def paper_centers(cfg: RodCoolingConfig) -> np.ndarray:
    centers = np.arange(cfg.controller_count, dtype=np.float32) + 0.5
    return np.clip(centers, cfg.position_min, cfg.position_max)


def initial_controller_positions(cfg: RodCoolingConfig) -> np.ndarray:
    if cfg.controller_count == 3 and abs(cfg.rod_length_m - 3.0) < 1e-6:
        reference_positions = np.array([0.1, 1.1, 2.9], dtype=np.float32)
        return enforce_min_separation_1d(reference_positions, cfg)
    centers = paper_centers(cfg)
    signs = np.ones(cfg.controller_count, dtype=np.float32)
    signs[::2] = -1.0
    shifted = centers + signs * cfg.initial_controller_offset_m
    return enforce_min_separation_1d(shifted, cfg)


def apply_dirichlet(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).copy()
    state[0] = 0.0
    state[-1] = 0.0
    return state


def initial_state(grid: np.ndarray, cfg: RodCoolingConfig) -> np.ndarray:
    field = np.full_like(grid, fill_value=cfg.initial_constant, dtype=np.float32)
    return apply_dirichlet(field)


def truncated_gaussian_kernel(grid: np.ndarray, center: float, sigma: float) -> np.ndarray:
    kernel = np.zeros_like(grid, dtype=np.float32)
    mask = np.abs(grid - center) <= sigma
    kernel[mask] = np.exp(-((grid[mask] - center) ** 2) / (2.0 * sigma ** 2))
    return kernel


def build_source_map(grid: np.ndarray, positions: np.ndarray, controls: np.ndarray, cfg: RodCoolingConfig) -> np.ndarray:
    source = np.zeros_like(grid, dtype=np.float32)
    for ctrl_idx in range(cfg.controller_count):
        source += controls[ctrl_idx] * truncated_gaussian_kernel(
            grid,
            float(positions[ctrl_idx]),
            cfg.controller_sigma_m,
        )
    return source


def pde_rhs(state: np.ndarray, source_map: np.ndarray, dz: float, cfg: RodCoolingConfig) -> np.ndarray:
    rhs = np.zeros_like(state, dtype=np.float32)
    lap = (state[:-2] - 2.0 * state[1:-1] + state[2:]) / (dz ** 2)
    rhs[1:-1] = cfg.diffusion * lap + cfg.reaction * state[1:-1] + source_map[1:-1]
    return rhs


def pde_step(state: np.ndarray, positions: np.ndarray, controls: np.ndarray, grid: np.ndarray, cfg: RodCoolingConfig) -> np.ndarray:
    dz = float(grid[1] - grid[0])
    source_map = build_source_map(grid, positions, controls, cfg)
    next_state = state + cfg.dt * pde_rhs(state, source_map, dz, cfg)
    next_state = np.maximum(next_state, 0.0)
    return apply_dirichlet(next_state.astype(np.float32))


def enforce_min_separation_1d(positions: np.ndarray, cfg: RodCoolingConfig) -> np.ndarray:
    projected = np.clip(np.asarray(positions, dtype=np.float32), cfg.position_min, cfg.position_max)
    if projected.size <= 1:
        return projected
    order = np.argsort(projected)
    sorted_pos = projected[order].copy()
    usable_span = cfg.position_max - cfg.position_min
    required_span = (len(sorted_pos) - 1) * cfg.min_separation
    spacing = cfg.min_separation if required_span <= usable_span else usable_span / max(len(sorted_pos) - 1, 1)
    sorted_pos[0] = np.clip(sorted_pos[0], cfg.position_min, cfg.position_max - (len(sorted_pos) - 1) * spacing)
    for idx in range(1, len(sorted_pos)):
        sorted_pos[idx] = max(sorted_pos[idx], sorted_pos[idx - 1] + spacing)
    if sorted_pos[-1] > cfg.position_max:
        sorted_pos -= sorted_pos[-1] - cfg.position_max
    if sorted_pos[0] < cfg.position_min:
        sorted_pos += cfg.position_min - sorted_pos[0]
    for idx in range(1, len(sorted_pos)):
        sorted_pos[idx] = max(sorted_pos[idx], sorted_pos[idx - 1] + spacing)
    sorted_pos = np.clip(sorted_pos, cfg.position_min, cfg.position_max)
    restored = projected.copy()
    restored[order] = sorted_pos
    return restored


class ControllerDynamics1D:
    def __init__(
        self,
        dt: float,
        force_limit: float,
        mass: float,
        damping: float,
        position_bounds: Tuple[float, float],
        d_min_sep: Optional[float],
        velocity_limit_m_s: Optional[float] = None,
    ) -> None:
        self.dt = float(dt)
        self.force_limit = float(force_limit)
        self.mass = float(mass)
        self.damping = float(damping)
        self.position_bounds = position_bounds
        self.d_min_sep = d_min_sep
        self.velocity_limit_m_s = velocity_limit_m_s

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        forces: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        clipped_forces = np.clip(forces, -self.force_limit, self.force_limit)
        acceleration = clipped_forces / self.mass - self.damping * velocities
        vel_next = velocities + self.dt * acceleration
        if self.velocity_limit_m_s is not None:
            vel_next = np.clip(vel_next, -self.velocity_limit_m_s, self.velocity_limit_m_s)
        pos_next = positions + self.dt * vel_next
        pos_next = np.clip(pos_next, self.position_bounds[0], self.position_bounds[1])
        if self.d_min_sep is not None:
            cfg_proxy = RodCoolingConfig(
                position_min=self.position_bounds[0],
                position_max=self.position_bounds[1],
                min_separation=float(self.d_min_sep),
            )
            pos_next = enforce_min_separation_1d(pos_next, cfg_proxy)
        return pos_next.astype(np.float32), vel_next.astype(np.float32)

    def rollout(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        forces_seq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_state = positions.astype(np.float32, copy=True)
        vel_state = velocities.astype(np.float32, copy=True)
        pos_hist = np.zeros((forces_seq.shape[0], positions.shape[0]), dtype=np.float32)
        vel_hist = np.zeros_like(pos_hist)
        for step_idx in range(forces_seq.shape[0]):
            pos_state, vel_state = self.step(pos_state, vel_state, forces_seq[step_idx])
            pos_hist[step_idx] = pos_state
            vel_hist[step_idx] = vel_state
        return pos_hist, vel_hist


def build_controller_dynamics(cfg: RodCoolingConfig, mobile: bool = True) -> ControllerDynamics1D:
    return ControllerDynamics1D(
        dt=cfg.dt,
        force_limit=cfg.force_limit if mobile else 0.0,
        mass=cfg.mass_kg,
        damping=cfg.damping_per_s,
        position_bounds=(cfg.position_min, cfg.position_max),
        d_min_sep=cfg.min_separation if mobile else cfg.min_separation,
        velocity_limit_m_s=cfg.velocity_limit_m_s,
    )


def project_control_step(candidate: np.ndarray, previous: np.ndarray, cfg: RodCoolingConfig) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=np.float32)
    delta = np.clip(candidate - previous, -cfg.control_delta_max, cfg.control_delta_max)
    projected = np.clip(previous + delta, cfg.control_min, cfg.control_max)
    return np.minimum(projected, 0.0).astype(np.float32)


def project_force_step(candidate: np.ndarray, previous: np.ndarray, cfg: RodCoolingConfig, mobile: bool = True) -> np.ndarray:
    if not mobile:
        return np.zeros_like(previous, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    delta = np.clip(candidate - previous, -cfg.force_delta_max, cfg.force_delta_max)
    projected = np.clip(previous + delta, -cfg.force_limit, cfg.force_limit)
    return projected.astype(np.float32)


def sample_random_trajectory(
    cfg: RodCoolingConfig,
    steps: int,
    rng: np.random.Generator,
    mobile: bool = True,
) -> Dict[str, np.ndarray]:
    controls = np.zeros((steps, cfg.controller_count), dtype=np.float32)
    forces = np.zeros((steps, cfg.controller_count), dtype=np.float32)
    positions = np.zeros((steps, cfg.controller_count), dtype=np.float32)
    velocities = np.zeros((steps, cfg.controller_count), dtype=np.float32)

    dynamics = build_controller_dynamics(cfg, mobile=mobile)
    prev_u = np.zeros(cfg.controller_count, dtype=np.float32)
    prev_f = np.zeros(cfg.controller_count, dtype=np.float32)
    prev_p = initial_controller_positions(cfg)
    prev_v = np.zeros(cfg.controller_count, dtype=np.float32)

    for step in range(steps):
        control_noise = rng.uniform(-cfg.control_delta_max, cfg.control_delta_max, size=cfg.controller_count)
        raw_u = prev_u + control_noise.astype(np.float32)
        next_u = project_control_step(raw_u, prev_u, cfg)

        force_noise = rng.uniform(-cfg.force_delta_max, cfg.force_delta_max, size=cfg.controller_count)
        raw_f = prev_f + force_noise.astype(np.float32)
        next_f = project_force_step(raw_f, prev_f, cfg, mobile=mobile)

        if mobile:
            next_p, next_v = dynamics.step(prev_p, prev_v, next_f)
        else:
            next_p = prev_p.copy()
            next_v = np.zeros_like(prev_v)

        controls[step] = next_u
        forces[step] = next_f
        positions[step] = next_p
        velocities[step] = next_v

        prev_u = next_u
        prev_f = next_f
        prev_p = next_p
        prev_v = next_v

    return {
        "controls": controls,
        "forces": forces,
        "positions": positions,
        "velocities": velocities,
    }


def simulate_trajectory(
    cfg: RodCoolingConfig,
    controls: np.ndarray,
    positions: np.ndarray,
    initial: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    grid = make_spatial_grid(cfg)
    steps = controls.shape[0]
    states = np.zeros((steps + 1, cfg.grid_points), dtype=np.float32)
    sources = np.zeros((steps, cfg.grid_points), dtype=np.float32)
    states[0] = initial_state(grid, cfg) if initial is None else apply_dirichlet(initial)
    for step in range(steps):
        sources[step] = build_source_map(grid, positions[step], controls[step], cfg)
        states[step + 1] = pde_step(states[step], positions[step], controls[step], grid, cfg)
    return {"grid": grid, "states": states, "sources": sources, "controls": controls, "positions": positions}


def generate_transition_dataset(
    cfg: RodCoolingConfig,
    num_trajectories: int,
    steps: int,
    seed: Optional[int] = None,
    mobile_probability: float = 0.8,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.training_seed if seed is None else seed)
    all_states = []
    all_targets = []
    all_sources = []
    all_controls = []
    all_forces = []
    all_positions = []
    all_velocities = []
    all_rollouts = []
    for _ in range(num_trajectories):
        is_mobile = bool(rng.random() < mobile_probability)
        rollout = sample_random_trajectory(cfg, steps=steps, rng=rng, mobile=is_mobile)
        simulation = simulate_trajectory(cfg, rollout["controls"], rollout["positions"])
        all_states.append(simulation["states"][:-1])
        all_targets.append(simulation["states"][1:])
        all_sources.append(simulation["sources"])
        all_controls.append(simulation["controls"])
        all_forces.append(rollout["forces"])
        all_positions.append(simulation["positions"])
        all_velocities.append(rollout["velocities"])
        all_rollouts.append(simulation["states"])
    return {
        "grid": make_spatial_grid(cfg),
        "state_inputs": np.asarray(all_states, dtype=np.float32),
        "state_targets": np.asarray(all_targets, dtype=np.float32),
        "source_inputs": np.asarray(all_sources, dtype=np.float32),
        "controls": np.asarray(all_controls, dtype=np.float32),
        "forces": np.asarray(all_forces, dtype=np.float32),
        "positions": np.asarray(all_positions, dtype=np.float32),
        "velocities": np.asarray(all_velocities, dtype=np.float32),
        "rollouts": np.asarray(all_rollouts, dtype=np.float32),
    }


def average_temperature(states: np.ndarray) -> np.ndarray:
    return np.mean(states, axis=-1).astype(np.float32)
