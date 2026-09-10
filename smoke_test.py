from __future__ import annotations

import numpy as np
import torch

from pollution_core import (
    GridParams,
    PhysicalParams,
    build_source_map_np,
    initial_condition,
    simulate_pde_step_np,
)
from rod_cooling_1d.core import (
    RodCoolingConfig,
    build_source_map as build_source_map_1d,
    initial_controller_positions,
    initial_state as initial_state_1d,
    make_spatial_grid,
    pde_step as pde_step_1d,
)
from rod_cooling_1d.model import StateAwareSurrogate1D
from stateful_surrogate import StateAwareSurrogate


def check_two_dimensional_path() -> None:
    grid = GridParams(Nx=12, Ny=12, Nt=30)
    phys = PhysicalParams()
    positions = np.array(
        [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
        dtype=np.float32,
    )
    controls = np.zeros(phys.num_controllers, dtype=np.float32)
    state = initial_condition(grid, phys)
    source = build_source_map_np(positions, controls, grid, phys)
    next_state = simulate_pde_step_np(state, source, grid, phys)

    assert state.shape == (grid.Ny, grid.Nx)
    assert source.shape == state.shape
    assert next_state.shape == state.shape
    assert np.allclose(next_state[[0, -1], :], 0.0)
    assert np.allclose(next_state[:, [0, -1]], 0.0)

    model = StateAwareSurrogate(grid)
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(state)[None, None, ...],
            torch.from_numpy(source)[None, None, ...],
        )
    assert tuple(prediction.shape) == (1, 1, grid.Ny, grid.Nx)


def check_one_dimensional_path() -> None:
    cfg = RodCoolingConfig(grid_points=20, control_steps=2)
    grid = make_spatial_grid(cfg)
    positions = initial_controller_positions(cfg)
    controls = np.zeros(cfg.controller_count, dtype=np.float32)
    state = initial_state_1d(grid, cfg)
    source = build_source_map_1d(grid, positions, controls, cfg)
    next_state = pde_step_1d(state, positions, controls, grid, cfg)

    assert state.shape == (cfg.grid_points,)
    assert source.shape == state.shape
    assert next_state.shape == state.shape
    assert next_state[0] == 0.0 and next_state[-1] == 0.0

    model = StateAwareSurrogate1D()
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(state)[None, ...],
            torch.from_numpy(source)[None, ...],
        )
    assert tuple(prediction.shape) == (1, cfg.grid_points)
    assert prediction[0, 0].item() == 0.0 and prediction[0, -1].item() == 0.0


def main() -> None:
    check_two_dimensional_path()
    check_one_dimensional_path()
    print("Smoke tests passed for the 2-D and 1-D simulation paths.")


if __name__ == "__main__":
    main()
