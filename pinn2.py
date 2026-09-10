from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_style import apply_readable_plot_style, style_axes, style_colorbar
from pollution_core import generate_control_trajectory, rollout_pde
from stateful_surrogate import load_checkpoint, rollout_surrogate


apply_readable_plot_style()

FIELD_PANEL_TITLE_SIZE = 16
FIELD_ROW_LABEL_SIZE = 20
FIELD_COLORBAR_TICK_SIZE = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the state-aware surrogate.")
    parser.add_argument("--model-path", default="state_aware_pinn_best.pth")
    parser.add_argument("--output-dir", default="pinn_generalization_test_results_v2")
    parser.add_argument("--num-test-trajectories", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=10000)
    return parser.parse_args()


def calculate_errors(u_true: np.ndarray, u_pred: np.ndarray) -> Dict[str, np.ndarray]:
    mae = float(np.mean(np.abs(u_true - u_pred)))
    rmse = float(np.sqrt(np.mean((u_true - u_pred) ** 2)))
    max_ae = float(np.max(np.abs(u_true - u_pred)))

    mask = u_true > 0.05
    if np.any(mask):
        rel_errors = np.abs(u_true[mask] - u_pred[mask]) / np.maximum(u_true[mask], 1e-6) * 100.0
        rel_error = float(np.mean(rel_errors))
        rel_error_median = float(np.median(rel_errors))
    else:
        rel_error = 0.0
        rel_error_median = 0.0

    mae_per_t = np.mean(np.abs(u_true - u_pred), axis=(1, 2))
    rmse_per_t = np.sqrt(np.mean((u_true - u_pred) ** 2, axis=(1, 2)))
    rel_error_per_t = np.zeros(u_true.shape[0], dtype=np.float32)
    rel_error_median_per_t = np.zeros(u_true.shape[0], dtype=np.float32)

    for time_idx in range(u_true.shape[0]):
        mask_t = u_true[time_idx] > 0.05
        if not np.any(mask_t):
            continue
        rel_errors_t = (
            np.abs(u_true[time_idx][mask_t] - u_pred[time_idx][mask_t])
            / np.maximum(u_true[time_idx][mask_t], 1e-6)
            * 100.0
        )
        rel_error_per_t[time_idx] = float(np.mean(rel_errors_t))
        rel_error_median_per_t[time_idx] = float(np.median(rel_errors_t))

    return {
        "mae": mae,
        "rmse": rmse,
        "max_ae": max_ae,
        "rel_error": rel_error,
        "rel_error_median": rel_error_median,
        "mae_per_t": mae_per_t,
        "rmse_per_t": rmse_per_t,
        "rel_error_per_t": rel_error_per_t,
        "rel_error_median_per_t": rel_error_median_per_t,
    }


def plot_comparison(
    u_true: np.ndarray,
    u_pred: np.ndarray,
    errors: Dict[str, np.ndarray],
    output_dir: Path,
    traj_idx: int,
    time_axis_seconds: np.ndarray,
) -> None:
    time_indices = [0, len(u_true) // 2, len(u_true) - 1]
    vmin = 0.0
    vmax = max(float(u_true.max()), float(u_pred.max()))

    fig, axes = plt.subplots(3, len(time_indices), figsize=(9.2, 8.2))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    for column, time_idx in enumerate(time_indices):
        title = f"t={time_axis_seconds[time_idx]:.1f}s"
        im_true = axes[0, column].imshow(u_true[time_idx], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, column].set_title(title, fontsize=FIELD_PANEL_TITLE_SIZE)
        axes[0, column].set_ylabel("True")
        axes[0, column].axis("off")

        im_pred = axes[1, column].imshow(u_pred[time_idx], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, column].set_ylabel("Pred")
        axes[1, column].axis("off")

        error_map = np.abs(u_true[time_idx] - u_pred[time_idx])
        im_err = axes[2, column].imshow(error_map, origin="lower", cmap="magma")
        axes[2, column].set_ylabel("Abs Err")
        axes[2, column].axis("off")

    fig.subplots_adjust(wspace=0.04, hspace=0.16, left=0.09, right=0.78, bottom=0.04, top=0.90)
    for label, ypos in (("True", 0.775), ("Pred", 0.485), ("Abs Err", 0.195)):
        fig.text(0.035, ypos, label, va="center", ha="center", rotation=90, fontsize=FIELD_ROW_LABEL_SIZE)
    colorbar_specs = (
        (im_true, [0.82, 0.675, 0.020, 0.20]),
        (im_pred, [0.82, 0.385, 0.020, 0.20]),
        (im_err, [0.82, 0.095, 0.020, 0.20]),
    )
    for image, bounds in colorbar_specs:
        colorbar = fig.colorbar(image, cax=fig.add_axes(bounds))
        style_colorbar(colorbar)
        colorbar.ax.tick_params(labelsize=FIELD_COLORBAR_TICK_SIZE)
    style_axes(axes)
    plt.savefig(output_dir / f"traj_{traj_idx}_field_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(time_axis_seconds, errors["mae_per_t"], label="MAE", color="#1f77b4", linewidth=2)
    ax1.plot(time_axis_seconds, errors["rmse_per_t"], label="RMSE", color="#2ca02c", linewidth=2, linestyle="--")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Absolute Error")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        time_axis_seconds,
        errors["rel_error_median_per_t"],
        label="Median Relative Error",
        color="#ff7f0e",
        linewidth=2,
    )
    ax2.set_ylabel("Median Relative Error (%)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout()
    style_axes([ax1, ax2])
    plt.savefig(output_dir / f"traj_{traj_idx}_error_time.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_overall_statistics(
    all_errors: List[Dict[str, np.ndarray]],
    output_dir: Path,
    time_axis_seconds: np.ndarray,
) -> None:
    maes = np.array([entry["mae"] for entry in all_errors], dtype=np.float32)
    rmses = np.array([entry["rmse"] for entry in all_errors], dtype=np.float32)
    rel_medians = np.array([entry["rel_error_median"] for entry in all_errors], dtype=np.float32)
    max_aes = np.array([entry["max_ae"] for entry in all_errors], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(maes, bins=10, alpha=0.7, color="#1f77b4", edgecolor="black")
    axes[0, 0].set_xlabel("MAE")

    axes[0, 1].hist(rmses, bins=10, alpha=0.7, color="#2ca02c", edgecolor="black")
    axes[0, 1].set_xlabel("RMSE")

    axes[1, 0].hist(rel_medians, bins=10, alpha=0.7, color="#ff7f0e", edgecolor="black")
    axes[1, 0].set_xlabel("Median Relative Error (%)")

    axes[1, 1].hist(max_aes, bins=10, alpha=0.7, color="#d62728", edgecolor="black")
    axes[1, 1].set_xlabel("Max Absolute Error")

    for axis in axes.flat:
        axis.grid(alpha=0.25)

    plt.tight_layout()
    style_axes(axes)
    plt.savefig(output_dir / "overall_error_statistics.png", dpi=300, bbox_inches="tight")
    plt.close()

    avg_mae = np.mean([entry["mae_per_t"] for entry in all_errors], axis=0)
    avg_rmse = np.mean([entry["rmse_per_t"] for entry in all_errors], axis=0)
    avg_rel = np.mean([entry["rel_error_median_per_t"] for entry in all_errors], axis=0)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(time_axis_seconds, avg_mae, label="Average MAE", color="#1f77b4", linewidth=2)
    ax1.plot(time_axis_seconds, avg_rmse, label="Average RMSE", color="#2ca02c", linewidth=2, linestyle="--")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Average Absolute Error")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(time_axis_seconds, avg_rel, label="Average Median Relative Error", color="#ff7f0e", linewidth=2)
    ax2.set_ylabel("Average Median Relative Error (%)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout()
    style_axes([ax1, ax2])
    plt.savefig(output_dir / "average_error_time.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, grid, phys, _, _, _ = load_checkpoint(args.model_path, device=device)
    time_axis_seconds = np.linspace(0.0, phys.time_scale_s, grid.Nt, dtype=np.float32)

    print(f"Loaded surrogate from {args.model_path}")
    print(f"Running {args.num_test_trajectories} unseen trajectories on {device}...")

    all_errors: List[Dict[str, np.ndarray]] = []
    for traj_idx in range(args.num_test_trajectories):
        seed = args.start_seed + traj_idx
        positions, controls = generate_control_trajectory(grid, seed=seed, phys=phys)
        u_true, source_maps = rollout_pde(positions, controls, grid, phys)
        u_pred = rollout_surrogate(model, u_true[0], source_maps, phys=phys, device=device)

        errors = calculate_errors(u_true, u_pred)
        all_errors.append(errors)

        print(
            f"Trajectory {traj_idx + 1:02d}/{args.num_test_trajectories} | seed={seed} | "
            f"MAE={errors['mae']:.4f} | RMSE={errors['rmse']:.4f} | "
            f"Median Rel={errors['rel_error_median']:.2f}% | Max AE={errors['max_ae']:.4f}"
        )

        if traj_idx < 5:
            plot_comparison(u_true, u_pred, errors, output_dir, traj_idx, time_axis_seconds)

    plot_overall_statistics(all_errors, output_dir, time_axis_seconds)
    np.save(output_dir / "all_test_errors.npy", np.array(all_errors, dtype=object), allow_pickle=True)

    print("-" * 72)
    print(f"Average MAE: {np.mean([entry['mae'] for entry in all_errors]):.4f}")
    print(f"Average RMSE: {np.mean([entry['rmse'] for entry in all_errors]):.4f}")
    print(f"Average Median Relative Error: {np.mean([entry['rel_error_median'] for entry in all_errors]):.2f}%")
    print(f"Average Max Absolute Error: {np.mean([entry['max_ae'] for entry in all_errors]):.4f}")
    print(f"Saved plots and metrics to {output_dir}")


if __name__ == "__main__":
    main()
