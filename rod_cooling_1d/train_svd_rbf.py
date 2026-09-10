from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from rod_cooling_1d.core import RodCoolingConfig
from rod_cooling_1d.svd_rbf_baseline import SVDRBFBaselineConfig, train_svd_rbf_bundle


def parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the 1-D SVD-RBFNN-style baseline model.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=folder / "outputs" / "svd_rbf" / "rod_cooling_svd_rbf.pkl",
    )
    parser.add_argument("--num-trajectories", type=int, default=72)
    parser.add_argument("--steps-per-traj", type=int, default=240)
    parser.add_argument("--delay-steps", type=int, default=5)
    parser.add_argument("--hidden-units", type=int, default=180)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--energy-ratio", type=float, default=0.995)
    parser.add_argument("--position-delta-max", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RodCoolingConfig()
    baseline_cfg = SVDRBFBaselineConfig(
        delay_steps=args.delay_steps,
        energy_ratio=args.energy_ratio,
        hidden_units=args.hidden_units,
        gamma=args.gamma,
        training_trajectories=args.num_trajectories,
        training_steps=args.steps_per_traj,
        position_delta_max=args.position_delta_max,
        seed=args.seed,
    )
    bundle = train_svd_rbf_bundle(cfg, baseline_cfg, args.output_path, verbose=True)
    config_path = args.output_path.with_suffix(".json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "cfg_signature": bundle["cfg_signature"],
                "baseline_config": bundle["baseline_config"],
                "train_mse": bundle["train_mse"],
                "val_mse": bundle["val_mse"],
            },
            fh,
            indent=2,
        )
    print(f"Saved baseline config summary to: {config_path}")


if __name__ == "__main__":
    main()
