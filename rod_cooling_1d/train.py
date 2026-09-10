from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from plot_style import apply_readable_plot_style, style_axes
from rod_cooling_1d.core import RodCoolingConfig, generate_transition_dataset
from rod_cooling_1d.model import StateAwareSurrogate1D, SurrogateConfig, device_for_run


apply_readable_plot_style()


def parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    default_output = folder / "outputs" / "train"
    parser = argparse.ArgumentParser(description="Train the 1-D rod cooling state-aware surrogate.")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--num-trajectories", type=int, default=240)
    parser.add_argument("--steps-per-traj", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(dataset: dict, batch_size: int) -> tuple[DataLoader, DataLoader]:
    states = dataset["state_inputs"]
    sources = dataset["source_inputs"]
    targets = dataset["state_targets"]
    split = max(1, int(0.9 * states.shape[0]))
    train_tensors = (
        torch.from_numpy(states[:split].reshape(-1, states.shape[-1])),
        torch.from_numpy(sources[:split].reshape(-1, sources.shape[-1])),
        torch.from_numpy(targets[:split].reshape(-1, targets.shape[-1])),
    )
    val_tensors = (
        torch.from_numpy(states[split:].reshape(-1, states.shape[-1])),
        torch.from_numpy(sources[split:].reshape(-1, sources.shape[-1])),
        torch.from_numpy(targets[split:].reshape(-1, targets.shape[-1])),
    )
    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for states, sources, targets in loader:
            states = states.to(device)
            sources = sources.to(device)
            targets = targets.to(device)
            predictions = model(states, sources)
            losses.append(float(criterion(predictions, targets).item()))
    return float(np.mean(losses)) if losses else 0.0


def plot_losses(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    style_axes(ax)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = device_for_run(args.device)

    cfg = RodCoolingConfig(
        training_trajectories=args.num_trajectories,
        control_steps=args.steps_per_traj,
        training_seed=args.seed,
    )
    model_cfg = SurrogateConfig(channels=args.channels, kernel_size=args.kernel_size)

    dataset = generate_transition_dataset(
        cfg,
        num_trajectories=args.num_trajectories,
        steps=args.steps_per_traj,
        seed=args.seed,
    )
    train_loader, val_loader = build_loaders(dataset, args.batch_size)

    model = StateAwareSurrogate1D(
        channels=model_cfg.channels,
        kernel_size=model_cfg.kernel_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_path = args.output_dir / "rod_cooling_1d_best.pth"
    final_path = args.output_dir / "rod_cooling_1d_final.pth"

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for states, sources, targets in train_loader:
            states = states.to(device)
            sources = sources.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(states, sources)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss = evaluate(model, val_loader, device, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        checkpoint = {
            "config": cfg.to_dict(),
            "model_config": model_cfg.to_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "model_state": model.state_dict(),
        }
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(checkpoint, best_path)
        print(f"Epoch {epoch + 1:03d}/{args.epochs:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

    torch.save(
        {
            "config": cfg.to_dict(),
            "model_config": model_cfg.to_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "model_state": model.state_dict(),
        },
        final_path,
    )

    plot_losses(train_losses, val_losses, args.output_dir / "training_loss.png")
    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": cfg.to_dict(), "model_config": model_cfg.to_dict()}, fh, indent=2)
    print(f"Best checkpoint: {best_path}")
    print(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
