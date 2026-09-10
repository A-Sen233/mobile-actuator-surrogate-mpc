from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from plot_style import apply_readable_plot_style, style_axes
from pollution_core import GridParams, PhysicalParams, build_transition_dataset, seed_everything
from stateful_surrogate import (
    LossWeights,
    ModelConfig,
    StateAwareSurrogate,
    TransitionDataset,
    compute_losses,
    save_checkpoint,
)


apply_readable_plot_style()


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str = "pinn_generalization_results_v2"
    best_model_path: str = "state_aware_pinn_best.pth"
    final_model_path: str = "state_aware_pinn_final.pth"
    epochs: int = 80
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    train_trajectories: int = 160
    val_trajectories: int = 24
    samples_per_trajectory: int = 24
    seed: int = 42
    patience: int = 15


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the state-aware surrogate.")
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--best-model-path", default=TrainConfig.best_model_path)
    parser.add_argument("--final-model-path", default=TrainConfig.final_model_path)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--train-trajectories", type=int, default=TrainConfig.train_trajectories)
    parser.add_argument("--val-trajectories", type=int, default=TrainConfig.val_trajectories)
    parser.add_argument("--samples-per-trajectory", type=int, default=TrainConfig.samples_per_trajectory)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    args = parser.parse_args()
    return TrainConfig(
        output_dir=args.output_dir,
        best_model_path=args.best_model_path,
        final_model_path=args.final_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_trajectories=args.train_trajectories,
        val_trajectories=args.val_trajectories,
        samples_per_trajectory=args.samples_per_trajectory,
        seed=args.seed,
        patience=args.patience,
    )


def build_dataloader(arrays: Dict[str, torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TransitionDataset(arrays)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(
    model: StateAwareSurrogate,
    dataloader: DataLoader,
    device: torch.device,
    grid: GridParams,
    phys: PhysicalParams,
    loss_weights: LossWeights,
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "data": 0.0, "physics": 0.0, "boundary": 0.0, "mass": 0.0}
    total_batches = 0

    with torch.no_grad():
        for current_field, source_map, next_field in dataloader:
            current_field = current_field.to(device)
            source_map = source_map.to(device)
            next_field = next_field.to(device)

            total_loss, losses, _ = compute_losses(model, current_field, source_map, next_field, grid, phys, loss_weights)
            totals["loss"] += float(total_loss.item())
            for key in ("data", "physics", "boundary", "mass"):
                totals[key] += float(losses[key].item())
            total_batches += 1

    if total_batches == 0:
        return totals
    return {key: value / total_batches for key, value in totals.items()}


def plot_history(history: List[Dict[str, float]], output_path: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    train_total = [entry["train_loss"] for entry in history]
    val_total = [entry["val_loss"] for entry in history]
    train_data = [entry["train_data"] for entry in history]
    val_data = [entry["val_data"] for entry in history]
    train_physics = [entry["train_physics"] for entry in history]
    val_physics = [entry["val_physics"] for entry in history]
    train_boundary = [entry["train_boundary"] for entry in history]
    val_boundary = [entry["val_boundary"] for entry in history]

    plt.figure(figsize=(12, 8))
    plt.semilogy(epochs, train_total, label="Train Total", linewidth=2)
    plt.semilogy(epochs, val_total, label="Val Total", linewidth=2)
    plt.semilogy(epochs, train_data, label="Train Data", alpha=0.8)
    plt.semilogy(epochs, val_data, label="Val Data", alpha=0.8)
    plt.semilogy(epochs, train_physics, label="Train Physics", alpha=0.8)
    plt.semilogy(epochs, val_physics, label="Val Physics", alpha=0.8)
    plt.semilogy(epochs, train_boundary, label="Train Boundary", alpha=0.6)
    plt.semilogy(epochs, val_boundary, label="Val Boundary", alpha=0.6)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    style_axes(plt.gcf().axes)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = GridParams()
    phys = PhysicalParams()
    model_config = ModelConfig()
    loss_weights = LossWeights()

    print(f"Using device: {device}")
    print("Building transition datasets...")
    train_arrays = build_transition_dataset(
        num_trajectories=config.train_trajectories,
        samples_per_trajectory=config.samples_per_trajectory,
        grid=grid,
        phys=phys,
        start_seed=config.seed,
    )
    val_arrays = build_transition_dataset(
        num_trajectories=config.val_trajectories,
        samples_per_trajectory=config.samples_per_trajectory,
        grid=grid,
        phys=phys,
        start_seed=config.seed + config.train_trajectories + 1000,
    )

    train_loader = build_dataloader(train_arrays, batch_size=config.batch_size, shuffle=True)
    val_loader = build_dataloader(val_arrays, batch_size=config.batch_size, shuffle=False)

    model = StateAwareSurrogate(grid=grid, model_config=model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    print(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, "
        f"Batch size: {config.batch_size}, Epochs: {config.epochs}"
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = {"loss": 0.0, "data": 0.0, "physics": 0.0, "boundary": 0.0, "mass": 0.0}
        batches = 0

        for current_field, source_map, next_field in train_loader:
            current_field = current_field.to(device)
            source_map = source_map.to(device)
            next_field = next_field.to(device)

            optimizer.zero_grad(set_to_none=True)
            total_loss, losses, _ = compute_losses(model, current_field, source_map, next_field, grid, phys, loss_weights)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running["loss"] += float(total_loss.item())
            for key in ("data", "physics", "boundary", "mass"):
                running[key] += float(losses[key].item())
            batches += 1

        scheduler.step()
        train_metrics = {key: value / max(batches, 1) for key, value in running.items()}
        val_metrics = evaluate(model, val_loader, device, grid, phys, loss_weights)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_data": train_metrics["data"],
                "train_physics": train_metrics["physics"],
                "train_boundary": train_metrics["boundary"],
                "train_mass": train_metrics["mass"],
                "val_loss": val_metrics["loss"],
                "val_data": val_metrics["data"],
                "val_physics": val_metrics["physics"],
                "val_boundary": val_metrics["boundary"],
                "val_mass": val_metrics["mass"],
            }
        )

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d} | lr={current_lr:.2e} | "
            f"train={train_metrics['loss']:.4e} | val={val_metrics['loss']:.4e} | "
            f"val_data={val_metrics['data']:.4e} | val_physics={val_metrics['physics']:.4e}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                config.best_model_path,
                model,
                grid,
                phys,
                model_config,
                loss_weights,
                {
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                },
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping at epoch {epoch} after {config.patience} epochs without improvement.")
            break

    save_checkpoint(
        config.final_model_path,
        model,
        grid,
        phys,
        model_config,
        loss_weights,
        {
            "best_val_loss": best_val_loss,
            "epoch": history[-1]["epoch"] if history else 0,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        },
    )

    plot_history(history, output_dir / "training_loss.png")
    print(f"Best model saved to {config.best_model_path}")
    print(f"Final model saved to {config.final_model_path}")
    print(f"Training curves saved to {output_dir / 'training_loss.png'}")


if __name__ == "__main__":
    main()
