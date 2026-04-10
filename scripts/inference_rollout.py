#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emulator.data import GCMDatasetFull
from emulator.models import EmulatorBase, EmulatorEnsemble, PerturbationModel


def parse_args():
    default_data_root = Path(
        os.environ.get("EMULATOR_DATA_ROOT", str((REPO_ROOT.parent / "data").resolve()))
    )
    default_stats_path = Path(
        os.environ.get(
            "EMULATOR_STATS_PATH",
            str(REPO_ROOT / "artifacts" / "stats" / "stats_full.pt"),
        )
    )
    default_cache_dir = Path(
        os.environ.get("EMULATOR_CACHE_DIR", str(REPO_ROOT / "outputs" / "cache"))
    )

    parser = argparse.ArgumentParser(description="Autoregressive ensemble inference")
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--stats-path", type=Path, default=default_stats_path)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "checkpoints" / "emulator_stage2_best.pth",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "inference" / "forecast_output.pt",
    )
    parser.add_argument("--val-runs", nargs="+", default=["run16"])
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--lead-time-days", type=int, default=10)
    parser.add_argument("--ensemble-members", type=int, default=48)
    return parser.parse_args()


def load_state_dict(path, device):
    state_dict = torch.load(path, map_location=device, weights_only=True)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value for key, value in state_dict.items()
        }
    return state_dict


def main():
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not args.stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {args.stats_path}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = EmulatorBase(img_size=(64, 128)).to(device)
    model_p = PerturbationModel(in_channels=28, out_channels=28).to(device)
    model_q = PerturbationModel(in_channels=42, out_channels=28).to(device)
    model = EmulatorEnsemble(base, model_p, model_q).to(device)
    model.load_state_dict(load_state_dict(args.checkpoint_path, device))
    model.eval()

    val_ds = GCMDatasetFull(
        data_root=args.data_root,
        stats_path=args.stats_path,
        runs=args.val_runs,
        cache_dir=args.cache_dir,
    )

    input_seq, _ = val_ds[args.seed_index]
    x_init = input_seq.unsqueeze(0).to(device)

    forecasts = []
    with torch.no_grad():
        pred = model(x_init, y=None, num_samples=args.ensemble_members)
        forecasts.append(pred.cpu())

        prev_day = x_init[:, 14:, :, :].repeat(args.ensemble_members, 1, 1, 1)
        current_input = torch.cat([prev_day, pred], dim=1)

        for _ in range(1, args.lead_time_days):
            pred = model(current_input, y=None, num_samples=1)
            forecasts.append(pred.cpu())
            prev_day = current_input[:, 14:, :, :]
            current_input = torch.cat([prev_day, pred], dim=1)

    all_forecasts = torch.stack(forecasts, dim=0)

    stats = torch.load(args.stats_path, map_location="cpu", weights_only=True)
    mean = stats["mean"].view(1, 1, -1, 1, 1)
    std = stats["std"].view(1, 1, -1, 1, 1)
    physical_forecasts = (all_forecasts * (std + 1e-6)) + mean

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(physical_forecasts, args.output_path)
    print(f"Saved forecast rollout to {args.output_path}")


if __name__ == "__main__":
    main()
