from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in environments without torch
    torch = None


EPS = 1e-8
DEFAULT_SUBDOMAINS: Tuple[Tuple[float, float, float, float], ...] = (
    (0.0, 0.5, 0.0, 0.5),
    (0.5, 1.0, 0.0, 0.5),
    (0.0, 0.5, 0.5, 1.0),
    (0.5, 1.0, 0.5, 1.0),
)


@dataclass(frozen=True)
class PhysicalParams:
    space_scale_m: float = 1000.0
    time_scale_s: float = 240.0
    alpha: float = 0.5
    sigma_m: float = 65.0
    u_max: float = 15.0
    decay: float = -0.1
    num_controllers: int = 4
    initial_width: float = 0.1
    boundary: str = "dirichlet"
    mass_kg: float = 20.0
    damping_per_s: float = 0.3

    @property
    def diffusion(self) -> float:
        return self.alpha * self.time_scale_s / (self.space_scale_m ** 2)

    @property
    def sigma(self) -> float:
        return self.sigma_m / self.space_scale_m

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data["diffusion"] = self.diffusion
        data["sigma"] = self.sigma
        return data


@dataclass(frozen=True)
class GridParams:
    Nx: int = 50
    Ny: int = 50
    Nt: int = 300

    @property
    def dx(self) -> float:
        return 1.0 / (self.Nx - 1)

    @property
    def dy(self) -> float:
        return 1.0 / (self.Ny - 1)

    @property
    def dt(self) -> float:
        return 1.0 / (self.Nt - 1)

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def grid_arrays(grid: GridParams) -> Dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, grid.Nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, grid.Ny, dtype=np.float32)
    t = np.linspace(0.0, 1.0, grid.Nt, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return {"x": x, "y": y, "t": t, "X": X.astype(np.float32), "Y": Y.astype(np.float32)}


def coordinate_channels(grid: GridParams, device: Optional[torch.device] = None) -> torch.Tensor:
    if torch is None:
        raise ImportError("torch is required for coordinate_channels().")
    arrays = grid_arrays(grid)
    coords = np.stack([arrays["X"], arrays["Y"]], axis=0)
    tensor = torch.from_numpy(coords).float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values.astype(np.float32), (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def generate_control_trajectory(
    grid: GridParams,
    seed: Optional[int] = None,
    subdomains: Tuple[Tuple[float, float, float, float], ...] = DEFAULT_SUBDOMAINS,
    phys: Optional[PhysicalParams] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    phys = phys or PhysicalParams()
    rng = np.random.default_rng(seed)
    positions = np.zeros((grid.Nt, len(subdomains), 2), dtype=np.float32)
    controls = np.zeros((grid.Nt, len(subdomains)), dtype=np.float32)
    control_step = float(rng.uniform(0.008, 0.05))
    smooth_window = int(rng.integers(5, 30))
    force_limit = 6.0
    force_smooth = float(rng.uniform(0.80, 0.94))
    force_noise = float(rng.uniform(0.35, 1.20))
    accel_scale = (phys.time_scale_s ** 2) / phys.space_scale_m
    damping_scale = phys.damping_per_s * phys.time_scale_s

    for ctrl_idx, (x0, x1, y0, y1) in enumerate(subdomains):
        pos_track = np.zeros((grid.Nt, 2), dtype=np.float32)
        vel_track = np.zeros((grid.Nt, 2), dtype=np.float32)
        force_track = np.zeros((grid.Nt, 2), dtype=np.float32)
        u_track = np.zeros(grid.Nt, dtype=np.float32)

        pos_track[0, 0] = float(rng.uniform(x0 + 0.05, x1 - 0.05))
        pos_track[0, 1] = float(rng.uniform(y0 + 0.05, y1 - 0.05))
        u_track[0] = float(rng.uniform(-0.95, -0.05))

        for step in range(1, grid.Nt):
            force_track[step] = np.clip(
                force_smooth * force_track[step - 1] + rng.normal(0.0, force_noise, size=2),
                -force_limit,
                force_limit,
            )
            acceleration = accel_scale * (force_track[step] / phys.mass_kg) - damping_scale * vel_track[step - 1]
            vel_track[step] = vel_track[step - 1] + grid.dt * acceleration
            pos_track[step] = pos_track[step - 1] + grid.dt * vel_track[step]
            pos_track[step, 0] = np.clip(pos_track[step, 0], x0 + 0.02, x1 - 0.02)
            pos_track[step, 1] = np.clip(pos_track[step, 1], y0 + 0.02, y1 - 0.02)
            if pos_track[step, 0] <= x0 + 0.02 or pos_track[step, 0] >= x1 - 0.02:
                vel_track[step, 0] = 0.0
            if pos_track[step, 1] <= y0 + 0.02 or pos_track[step, 1] >= y1 - 0.02:
                vel_track[step, 1] = 0.0
            u_track[step] = np.clip(u_track[step - 1] + rng.normal(0.0, control_step), -1.0, 0.0)

        positions[:, ctrl_idx, 0] = smooth_series(pos_track[:, 0], smooth_window)
        positions[:, ctrl_idx, 1] = smooth_series(pos_track[:, 1], smooth_window)
        controls[:, ctrl_idx] = smooth_series(u_track, smooth_window)

        positions[:, ctrl_idx, 0] = np.clip(positions[:, ctrl_idx, 0], x0 + 0.02, x1 - 0.02)
        positions[:, ctrl_idx, 1] = np.clip(positions[:, ctrl_idx, 1], y0 + 0.02, y1 - 0.02)
        controls[:, ctrl_idx] = np.clip(controls[:, ctrl_idx], -1.0, 0.0)

    return positions, controls


def initial_condition(grid: GridParams, phys: PhysicalParams) -> np.ndarray:
    arrays = grid_arrays(grid)
    X, Y = arrays["X"], arrays["Y"]
    field = np.zeros((grid.Ny, grid.Nx), dtype=np.float32)
    centers = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
    for cx, cy in centers:
        field += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * phys.initial_width ** 2)).astype(np.float32)
    return field / float(len(centers))


def gaussian_map_np(X: np.ndarray, Y: np.ndarray, center_x: float, center_y: float, sigma: float) -> np.ndarray:
    return np.exp(-((X - center_x) ** 2 + (Y - center_y) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


def build_source_map_np(
    positions: np.ndarray,
    controls: np.ndarray,
    grid: GridParams,
    phys: PhysicalParams,
) -> np.ndarray:
    arrays = grid_arrays(grid)
    X, Y = arrays["X"], arrays["Y"]
    source = np.zeros((grid.Ny, grid.Nx), dtype=np.float32)
    for ctrl_idx in range(phys.num_controllers):
        source += controls[ctrl_idx] * phys.u_max * gaussian_map_np(
            X,
            Y,
            float(positions[ctrl_idx, 0]),
            float(positions[ctrl_idx, 1]),
            phys.sigma,
        )
    return source


def laplacian_neumann_np(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    padded = np.pad(field, ((1, 1), (1, 1)), mode="edge")
    u_xx = (padded[1:-1, 2:] - 2.0 * padded[1:-1, 1:-1] + padded[1:-1, :-2]) / (dx ** 2)
    u_yy = (padded[2:, 1:-1] - 2.0 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]) / (dy ** 2)
    return (u_xx + u_yy).astype(np.float32)


def laplacian_dirichlet_np(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    padded = np.pad(field, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
    u_xx = (padded[1:-1, 2:] - 2.0 * padded[1:-1, 1:-1] + padded[1:-1, :-2]) / (dx ** 2)
    u_yy = (padded[2:, 1:-1] - 2.0 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]) / (dy ** 2)
    return (u_xx + u_yy).astype(np.float32)


def laplacian_np(field: np.ndarray, dx: float, dy: float, boundary: str) -> np.ndarray:
    if boundary == "dirichlet":
        return laplacian_dirichlet_np(field, dx, dy)
    return laplacian_neumann_np(field, dx, dy)


def enforce_boundary_np(field: np.ndarray, boundary: str) -> np.ndarray:
    if boundary != "dirichlet":
        return field.astype(np.float32, copy=False)
    constrained = field.astype(np.float32, copy=True)
    constrained[0, :] = 0.0
    constrained[-1, :] = 0.0
    constrained[:, 0] = 0.0
    constrained[:, -1] = 0.0
    return constrained


def simulate_pde_step_np(
    current_field: np.ndarray,
    source_map: np.ndarray,
    grid: GridParams,
    phys: PhysicalParams,
) -> np.ndarray:
    lap = laplacian_np(current_field, grid.dx, grid.dy, phys.boundary)
    next_field = current_field + grid.dt * (phys.diffusion * lap + phys.decay * current_field + source_map)
    return enforce_boundary_np(np.clip(next_field, 0.0, None).astype(np.float32), phys.boundary)


def rollout_pde(
    positions: np.ndarray,
    controls: np.ndarray,
    grid: GridParams,
    phys: PhysicalParams,
) -> Tuple[np.ndarray, np.ndarray]:
    fields = np.zeros((grid.Nt, grid.Ny, grid.Nx), dtype=np.float32)
    source_maps = np.zeros((grid.Nt - 1, grid.Ny, grid.Nx), dtype=np.float32)
    fields[0] = enforce_boundary_np(initial_condition(grid, phys), phys.boundary)

    for step in range(grid.Nt - 1):
        source_maps[step] = build_source_map_np(positions[step], controls[step], grid, phys)
        fields[step + 1] = simulate_pde_step_np(fields[step], source_maps[step], grid, phys)

    return fields, source_maps


def build_transition_dataset(
    num_trajectories: int,
    samples_per_trajectory: int,
    grid: GridParams,
    phys: PhysicalParams,
    start_seed: int = 0,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(start_seed + 77)
    current_fields = []
    source_maps = []
    next_fields = []

    for traj_idx in range(num_trajectories):
        traj_seed = start_seed + traj_idx
        positions, controls = generate_control_trajectory(grid, seed=traj_seed, phys=phys)
        fields, sources = rollout_pde(positions, controls, grid, phys)
        sample_count = min(samples_per_trajectory, grid.Nt - 1)
        transition_idx = rng.choice(grid.Nt - 1, size=sample_count, replace=False)
        transition_idx.sort()

        current_fields.append(fields[transition_idx])
        source_maps.append(sources[transition_idx])
        next_fields.append(fields[transition_idx + 1])

    current_array = np.concatenate(current_fields, axis=0)
    source_array = np.concatenate(source_maps, axis=0)
    next_array = np.concatenate(next_fields, axis=0)

    return {
        "current": current_array[:, None, :, :].astype(np.float32),
        "source": source_array[:, None, :, :].astype(np.float32),
        "next": next_array[:, None, :, :].astype(np.float32),
    }


def mass_centers_by_region(
    field: np.ndarray,
    grid: GridParams,
    subdomains: Tuple[Tuple[float, float, float, float], ...] = DEFAULT_SUBDOMAINS,
) -> np.ndarray:
    arrays = grid_arrays(grid)
    X, Y = arrays["X"], arrays["Y"]
    centers = np.zeros((len(subdomains), 2), dtype=np.float32)

    for idx, (x0, x1, y0, y1) in enumerate(subdomains):
        mask = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        region = np.where(mask, field, 0.0)
        total = float(region.sum())
        if total < EPS:
            centers[idx, 0] = 0.5 * (x0 + x1)
            centers[idx, 1] = 0.5 * (y0 + y1)
            continue
        centers[idx, 0] = float((X * region).sum() / total)
        centers[idx, 1] = float((Y * region).sum() / total)
    return centers


def max_points_by_region(
    field: np.ndarray,
    grid: GridParams,
    subdomains: Tuple[Tuple[float, float, float, float], ...] = DEFAULT_SUBDOMAINS,
    threshold: float = 0.01,
) -> np.ndarray:
    arrays = grid_arrays(grid)
    X, Y = arrays["X"], arrays["Y"]
    targets = np.zeros((len(subdomains), 2), dtype=np.float32)

    for idx, (x0, x1, y0, y1) in enumerate(subdomains):
        mask = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        region = np.where(mask, field, -np.inf)
        if float(np.max(region)) <= threshold:
            targets[idx] = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
            continue
        max_index = np.unravel_index(np.argmax(region), region.shape)
        targets[idx, 0] = X[max_index]
        targets[idx, 1] = Y[max_index]
    return targets


class ControllerDynamics:
    def __init__(
        self,
        dt: float,
        space_scale_m: float = 1000.0,
        time_scale_s: float = 240.0,
        force_limit: float = 3.0,
        velocity_limit_m_s: Optional[float] = None,
        mass: float = 20.0,
        damping: float = 0.3,
        bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        d_min_sep: Optional[float] = 0.1,
    ) -> None:
        self.dt = dt
        self.space_scale_m = space_scale_m
        self.time_scale_s = time_scale_s
        self.force_limit = force_limit
        self.velocity_limit_m_s = velocity_limit_m_s
        self.mass = mass
        self.damping = damping
        self.bounds = bounds
        self.d_min_sep = d_min_sep
        self.acceleration_scale = (self.time_scale_s ** 2) / self.space_scale_m
        self.damping_scale = self.damping * self.time_scale_s
        self.velocity_limit = None
        if self.velocity_limit_m_s is not None:
            self.velocity_limit = self.velocity_limit_m_s * self.time_scale_s / self.space_scale_m

    def step(self, positions: np.ndarray, velocities: np.ndarray, forces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        clipped_forces = np.clip(forces, -self.force_limit, self.force_limit)
        acceleration = self.acceleration_scale * (clipped_forces / self.mass) - self.damping_scale * velocities
        vel_next = velocities + self.dt * acceleration
        if self.velocity_limit is not None:
            vel_next = np.clip(vel_next, -self.velocity_limit, self.velocity_limit)
        pos_next = positions + self.dt * vel_next
        pos_next = self._clip_to_bounds(pos_next)
        if self.d_min_sep is not None:
            pos_next = self._enforce_separation(pos_next)
        return pos_next.astype(np.float32), vel_next.astype(np.float32)

    def rollout(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        forces_seq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_state = positions.astype(np.float32, copy=True)
        vel_state = velocities.astype(np.float32, copy=True)
        pos_hist = np.zeros((forces_seq.shape[0], positions.shape[0], 2), dtype=np.float32)
        vel_hist = np.zeros_like(pos_hist)
        for step_idx in range(forces_seq.shape[0]):
            pos_state, vel_state = self.step(pos_state, vel_state, forces_seq[step_idx])
            pos_hist[step_idx] = pos_state
            vel_hist[step_idx] = vel_state
        return pos_hist, vel_hist

    def _clip_to_bounds(self, positions: np.ndarray) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.bounds
        positions = positions.copy()
        positions[:, 0] = np.clip(positions[:, 0], x_min, x_max)
        positions[:, 1] = np.clip(positions[:, 1], y_min, y_max)
        return positions

    def _enforce_separation(self, positions: np.ndarray) -> np.ndarray:
        positions = positions.copy()
        for _ in range(3):
            for left in range(len(positions)):
                for right in range(left + 1, len(positions)):
                    delta = positions[right] - positions[left]
                    distance = float(np.linalg.norm(delta))
                    if distance < EPS or distance >= float(self.d_min_sep):
                        continue
                    correction = 0.5 * (float(self.d_min_sep) - distance) * delta / distance
                    positions[left] -= correction
                    positions[right] += correction
        return self._clip_to_bounds(positions)


def laplacian_neumann_torch(field: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    if torch is None:
        raise ImportError("torch is required for laplacian_neumann_torch().")
    padded = torch.nn.functional.pad(field, (1, 1, 1, 1), mode="replicate")
    u_xx = (padded[:, :, 1:-1, 2:] - 2.0 * padded[:, :, 1:-1, 1:-1] + padded[:, :, 1:-1, :-2]) / (dx ** 2)
    u_yy = (padded[:, :, 2:, 1:-1] - 2.0 * padded[:, :, 1:-1, 1:-1] + padded[:, :, :-2, 1:-1]) / (dy ** 2)
    return u_xx + u_yy


def laplacian_dirichlet_torch(field: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    if torch is None:
        raise ImportError("torch is required for laplacian_dirichlet_torch().")
    padded = torch.nn.functional.pad(field, (1, 1, 1, 1), mode="constant", value=0.0)
    u_xx = (padded[:, :, 1:-1, 2:] - 2.0 * padded[:, :, 1:-1, 1:-1] + padded[:, :, 1:-1, :-2]) / (dx ** 2)
    u_yy = (padded[:, :, 2:, 1:-1] - 2.0 * padded[:, :, 1:-1, 1:-1] + padded[:, :, :-2, 1:-1]) / (dy ** 2)
    return u_xx + u_yy


def laplacian_torch(field: torch.Tensor, dx: float, dy: float, boundary: str) -> torch.Tensor:
    if boundary == "dirichlet":
        return laplacian_dirichlet_torch(field, dx, dy)
    return laplacian_neumann_torch(field, dx, dy)


def enforce_boundary_torch(field: torch.Tensor, boundary: str) -> torch.Tensor:
    if boundary != "dirichlet":
        return field
    constrained = field.clone()
    constrained[:, :, 0, :] = 0.0
    constrained[:, :, -1, :] = 0.0
    constrained[:, :, :, 0] = 0.0
    constrained[:, :, :, -1] = 0.0
    return constrained


def source_map_torch(
    positions: torch.Tensor,
    controls: torch.Tensor,
    coord_channels_tensor: torch.Tensor,
    phys: PhysicalParams,
) -> torch.Tensor:
    if torch is None:
        raise ImportError("torch is required for source_map_torch().")
    x_grid = coord_channels_tensor[:, 0:1]
    y_grid = coord_channels_tensor[:, 1:2]
    source = torch.zeros_like(x_grid)
    for ctrl_idx in range(phys.num_controllers):
        pos_x = positions[:, ctrl_idx, 0].view(-1, 1, 1, 1)
        pos_y = positions[:, ctrl_idx, 1].view(-1, 1, 1, 1)
        amplitude = controls[:, ctrl_idx].view(-1, 1, 1, 1) * phys.u_max
        gaussian = torch.exp(-((x_grid - pos_x) ** 2 + (y_grid - pos_y) ** 2) / (2.0 * phys.sigma ** 2))
        source = source + amplitude * gaussian
    return source


def rollout_transition_baseline(
    initial_field_tensor: torch.Tensor,
    source_seq_tensor: torch.Tensor,
    grid: GridParams,
    phys: PhysicalParams,
) -> torch.Tensor:
    if torch is None:
        raise ImportError("torch is required for rollout_transition_baseline().")
    state = initial_field_tensor
    outputs = []
    for step_idx in range(source_seq_tensor.shape[1]):
        lap = laplacian_torch(state, grid.dx, grid.dy, phys.boundary)
        state = torch.clamp(
            state + grid.dt * (phys.diffusion * lap + phys.decay * state + source_seq_tensor[:, step_idx]),
            min=0.0,
        )
        state = enforce_boundary_torch(state, phys.boundary)
        outputs.append(state)
    return torch.stack(outputs, dim=1)
