from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class SurrogateConfig:
    channels: int = 32
    kernel_size: int = 5

    def to_dict(self) -> Dict[str, int]:
        return {"channels": self.channels, "kernel_size": self.kernel_size}


class StateAwareSurrogate1D(nn.Module):
    def __init__(self, channels: int = 32, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(channels, channels * 2, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(channels, 1, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, current_state: torch.Tensor, source_map: torch.Tensor) -> torch.Tensor:
        if current_state.ndim != 2 or source_map.ndim != 2:
            raise ValueError("current_state and source_map must have shape [batch, grid_points].")
        features = torch.stack([current_state, source_map], dim=1)
        delta = self.net(features).squeeze(1)
        prediction = torch.relu(current_state + delta)
        if prediction.shape[1] <= 2:
            return torch.zeros_like(prediction)
        interior = prediction[:, 1:-1]
        zeros = torch.zeros(
            prediction.shape[0],
            1,
            dtype=prediction.dtype,
            device=prediction.device,
        )
        return torch.cat([zeros, interior, zeros], dim=1)


def device_for_run(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: str, device: torch.device) -> Tuple[StateAwareSurrogate1D, Dict]:
    checkpoint = torch.load(path, map_location=device)
    model_cfg = checkpoint.get("model_config", {})
    model = StateAwareSurrogate1D(
        channels=int(model_cfg.get("channels", 32)),
        kernel_size=int(model_cfg.get("kernel_size", 5)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
