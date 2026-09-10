from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from pollution_core import (
    GridParams,
    PhysicalParams,
    coordinate_channels,
    enforce_boundary_torch,
    laplacian_torch,
)


@dataclass(frozen=True)
class ModelConfig:
    hidden_channels: int = 48
    num_blocks: int = 6
    dropout: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class LossWeights:
    data: float = 1.0
    physics: float = 0.5
    boundary: float = 0.1
    mass: float = 0.1

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class TransitionDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray]) -> None:
        self.current = torch.from_numpy(arrays["current"]).float()
        self.source = torch.from_numpy(arrays["source"]).float()
        self.next = torch.from_numpy(arrays["next"]).float()

    def __len__(self) -> int:
        return self.current.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.current[index], self.source[index], self.next[index]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.dropout(hidden)
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return inputs + hidden


class StateAwareSurrogate(nn.Module):
    def __init__(self, grid: GridParams, model_config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.grid = grid
        self.model_config = model_config or ModelConfig()
        coords = coordinate_channels(grid).unsqueeze(0)
        self.register_buffer("coord_channels", coords)

        hidden = self.model_config.hidden_channels
        self.input_proj = nn.Conv2d(4, hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden, dropout=self.model_config.dropout) for _ in range(self.model_config.num_blocks)
        )
        self.output_norm = nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden)
        self.output_proj = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, current_field: torch.Tensor, source_map: torch.Tensor) -> torch.Tensor:
        coords = self.coord_channels.expand(current_field.shape[0], -1, -1, -1)
        hidden = torch.cat([current_field, source_map, coords], dim=1)
        hidden = self.input_proj(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        delta = self.output_proj(F.silu(self.output_norm(hidden)))
        return torch.relu(current_field + delta)


def boundary_flux_loss(field: torch.Tensor) -> torch.Tensor:
    left = field[:, :, :, 0] - field[:, :, :, 1]
    right = field[:, :, :, -1] - field[:, :, :, -2]
    bottom = field[:, :, 0, :] - field[:, :, 1, :]
    top = field[:, :, -1, :] - field[:, :, -2, :]
    return left.square().mean() + right.square().mean() + bottom.square().mean() + top.square().mean()


def boundary_value_loss(field: torch.Tensor) -> torch.Tensor:
    return (
        field[:, :, 0, :].square().mean()
        + field[:, :, -1, :].square().mean()
        + field[:, :, :, 0].square().mean()
        + field[:, :, :, -1].square().mean()
    )


def compute_losses(
    model: StateAwareSurrogate,
    current_field: torch.Tensor,
    source_map: torch.Tensor,
    next_field: torch.Tensor,
    grid: GridParams,
    phys: PhysicalParams,
    weights: Optional[LossWeights] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    loss_weights = weights or LossWeights()
    prediction = enforce_boundary_torch(model(current_field, source_map), phys.boundary)

    data_loss = F.mse_loss(prediction, next_field)

    pde_target = torch.clamp(
        current_field
        + grid.dt * (phys.diffusion * laplacian_torch(current_field, grid.dx, grid.dy, phys.boundary) + phys.decay * current_field + source_map),
        min=0.0,
    )
    pde_target = enforce_boundary_torch(pde_target, phys.boundary)
    physics_loss = F.mse_loss(prediction, pde_target)
    boundary_loss = boundary_value_loss(prediction) if phys.boundary == "dirichlet" else boundary_flux_loss(prediction)
    mass_loss = F.mse_loss(prediction.mean(dim=(-1, -2)), next_field.mean(dim=(-1, -2)))

    total_loss = (
        loss_weights.data * data_loss
        + loss_weights.physics * physics_loss
        + loss_weights.boundary * boundary_loss
        + loss_weights.mass * mass_loss
    )

    return total_loss, {
        "data": data_loss,
        "physics": physics_loss,
        "boundary": boundary_loss,
        "mass": mass_loss,
    }, prediction


def save_checkpoint(
    path: str,
    model: StateAwareSurrogate,
    grid: GridParams,
    phys: PhysicalParams,
    model_config: ModelConfig,
    loss_weights: LossWeights,
    training_meta: Dict[str, float],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "grid_params": grid.to_dict(),
        "physical_params": phys.to_dict(),
        "model_config": model_config.to_dict(),
        "loss_weights": loss_weights.to_dict(),
        "training_meta": training_meta,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> Tuple[StateAwareSurrogate, GridParams, PhysicalParams, ModelConfig, LossWeights, Dict[str, float]]:
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=target_device, weights_only=False)

    grid = GridParams(**checkpoint["grid_params"])
    physical_kwargs = {key: value for key, value in checkpoint["physical_params"].items() if key in PhysicalParams.__dataclass_fields__}
    phys = PhysicalParams(**physical_kwargs)
    model_config = ModelConfig(**checkpoint["model_config"])
    loss_weights = LossWeights(**checkpoint.get("loss_weights", LossWeights().to_dict()))

    model = StateAwareSurrogate(grid=grid, model_config=model_config).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, grid, phys, model_config, loss_weights, checkpoint.get("training_meta", {})


def rollout_surrogate(
    model: StateAwareSurrogate,
    initial_field: np.ndarray,
    source_maps: np.ndarray,
    phys: Optional[PhysicalParams] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    target_device = device or next(model.parameters()).device
    rollout_phys = phys or PhysicalParams()
    state = torch.from_numpy(initial_field[None, None]).float().to(target_device)
    state = enforce_boundary_torch(state, rollout_phys.boundary)
    outputs = [state.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)]

    with torch.no_grad():
        for step in range(source_maps.shape[0]):
            source_tensor = torch.from_numpy(source_maps[step : step + 1, None]).float().to(target_device)
            state = enforce_boundary_torch(model(state, source_tensor), rollout_phys.boundary)
            outputs.append(state.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32))

    return np.stack(outputs, axis=0)
