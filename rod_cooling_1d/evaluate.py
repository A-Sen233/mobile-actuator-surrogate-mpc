from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from plot_style import apply_readable_plot_style, style_axes
from rod_cooling_1d.core import RodCoolingConfig, average_temperature, generate_transition_dataset
from rod_cooling_1d.model import device_for_run, load_checkpoint


apply_readable_plot_style()

TRAJECTORY_PANEL_TITLE_SIZE = 32
TRAJECTORY_ROW_LABEL_SIZE = 28
TRAJECTORY_AXIS_LABEL_SIZE = 28
TRAJECTORY_TICK_LABEL_SIZE = 20


def parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate the 1-D rod cooling surrogate.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=folder / "outputs" / "train" / "rod_cooling_1d_best.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=folder / "outputs" / "eval")
    parser.add_argument("--num-trajectories", type=int, default=6)
    parser.add_argument("--steps-per-traj", type=int, default=240)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def rollout_surrogate(
    model: torch.nn.Module,
    initial_state_np: np.ndarray,
    sources_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    states = np.zeros((sources_np.shape[0] + 1, initial_state_np.shape[0]), dtype=np.float32)
    states[0] = initial_state_np
    current = torch.from_numpy(initial_state_np).unsqueeze(0).to(device)
    with torch.no_grad():
        for step in range(sources_np.shape[0]):
            source = torch.from_numpy(sources_np[step]).unsqueeze(0).to(device)
            prediction = model(current, source)
            states[step + 1] = prediction.squeeze(0).cpu().numpy()
            current = prediction
    return states


def plot_trajectory_comparison(
    grid: np.ndarray,
    true_states: np.ndarray,
    pred_states: np.ndarray,
    output_path: Path,
    trajectory_index: int,
    dt: float,
) -> None:
    snapshot_indices = np.array([0, true_states.shape[0] // 2, true_states.shape[0] - 1], dtype=int)
    fig, axes = plt.subplots(3, len(snapshot_indices), figsize=(18, 8), sharex=True)
    global_max = float(max(np.max(true_states), np.max(pred_states)))
    global_err = float(np.max(np.abs(true_states - pred_states)))
    for col, idx in enumerate(snapshot_indices):
        axes[0, col].plot(grid, true_states[idx], color="tab:blue", linewidth=2)
        axes[1, col].plot(grid, pred_states[idx], color="tab:orange", linewidth=2)
        axes[2, col].plot(grid, np.abs(true_states[idx] - pred_states[idx]), color="tab:red", linewidth=2)
        axes[0, col].set_title(f"t={idx * dt:.3f}", fontsize=TRAJECTORY_PANEL_TITLE_SIZE)
        axes[0, col].set_ylim(0.0, global_max * 1.05 + 1e-6)
        axes[1, col].set_ylim(0.0, global_max * 1.05 + 1e-6)
        axes[2, col].set_ylim(0.0, max(global_err * 1.05, 1e-4))
    for row, label in enumerate(("True", "Pred", "Abs Err")):
        axes[row, 0].set_ylabel(label, fontsize=TRAJECTORY_ROW_LABEL_SIZE, labelpad=12)
    for ax in axes[-1]:
        ax.set_xlabel("z", fontsize=TRAJECTORY_AXIS_LABEL_SIZE, labelpad=8)
        ax.grid(alpha=0.2)
    style_axes(axes)
    for axis in axes.flat:
        axis.yaxis.set_label_position("left")
        axis.yaxis.tick_left()
        axis.tick_params(axis="both", which="major", labelsize=TRAJECTORY_TICK_LABEL_SIZE)
        axis.tick_params(axis="both", which="minor", labelsize=max(TRAJECTORY_TICK_LABEL_SIZE - 2, 8))
        axis.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_average_curve(true_avg: np.ndarray, pred_avg: np.ndarray, dt: float, output_path: Path) -> None:
    time_axis = np.arange(true_avg.shape[0], dtype=np.float32) * dt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time_axis, true_avg, label="True PDE", linewidth=2)
    ax.plot(time_axis, pred_avg, label="Surrogate", linewidth=2, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average temperature")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    style_axes(ax)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = device_for_run(args.device)
    model, checkpoint = load_checkpoint(str(args.checkpoint), device)
    cfg = RodCoolingConfig(**checkpoint["config"])

    dataset = generate_transition_dataset(
        cfg,
        num_trajectories=args.num_trajectories,
        steps=args.steps_per_traj,
        seed=cfg.training_seed + 1000,
    )
    grid = dataset["grid"]
    true_rollouts = dataset["rollouts"]
    sources = dataset["source_inputs"]

    predicted_rollouts = []
    maes = []
    rmses = []
    for traj_idx in range(args.num_trajectories):
        pred_states = rollout_surrogate(model, true_rollouts[traj_idx, 0], sources[traj_idx], device)
        predicted_rollouts.append(pred_states)
        error = pred_states - true_rollouts[traj_idx]
        maes.append(float(np.mean(np.abs(error))))
        rmses.append(float(np.sqrt(np.mean(error ** 2))))
        plot_trajectory_comparison(
            grid,
            true_rollouts[traj_idx],
            pred_states,
            args.output_dir / f"traj_{traj_idx}_field_comparison.png",
            trajectory_index=traj_idx,
            dt=cfg.dt,
        )

    predicted_rollouts_np = np.asarray(predicted_rollouts, dtype=np.float32)
    true_avg = average_temperature(true_rollouts).mean(axis=0)
    pred_avg = average_temperature(predicted_rollouts_np).mean(axis=0)
    plot_average_curve(true_avg, pred_avg, cfg.dt, args.output_dir / "average_temperature_comparison.png")

    metrics = {
        "mean_mae": float(np.mean(maes)),
        "mean_rmse": float(np.mean(rmses)),
        "per_trajectory_mae": maes,
        "per_trajectory_rmse": rmses,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
