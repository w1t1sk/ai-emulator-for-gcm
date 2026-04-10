#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emulator.evaluation import (
    CHANNEL_NAMES,
    CHANNEL_TITLES,
    CHANNEL_UNITS,
    RawRunReader,
    build_inference_model,
    default_checkpoint_path,
    default_data_root,
    default_stats_path,
    ensemble_size_or_default,
    forecast_members_physical,
    load_stats,
    roll_forward_input,
    tensor_from_pair,
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save forecast-vs-truth panel plots for selected lead times"
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root(REPO_ROOT))
    parser.add_argument("--stats-path", type=Path, default=default_stats_path(REPO_ROOT))
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=default_checkpoint_path(REPO_ROOT),
    )
    parser.add_argument("--verify-run", default="run16")
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument("--lead-times", nargs="+", type=int, default=[3, 5, 7, 9, 11])
    parser.add_argument("--ensemble-members", type=int, default=4)
    parser.add_argument(
        "--use-data-parallel",
        action="store_true",
        help="Wrap inference in torch.nn.DataParallel when multiple GPUs are visible",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "evaluation" / "forecast_panels",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "evaluation" / "forecast_panels",
    )
    return parser.parse_args()


def repo_relative_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), REPO_ROOT.resolve())


def save_panel(
    output_path: Path,
    variable_name: str,
    unit: str,
    lead_time: int,
    members: np.ndarray,
    truth: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
):
    ensemble_mean = members.mean(axis=0)
    vmin = min(
        truth.min(),
        ensemble_mean.min(),
        members.min(),
    )
    vmax = max(
        truth.max(),
        ensemble_mean.max(),
        members.max(),
    )

    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 9),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )
    fig.suptitle(f"{CHANNEL_TITLES[variable_name]} | Lead {lead_time} days", fontsize=16)

    panels = [
        ("Ground Truth", truth),
        ("Ensemble Mean", ensemble_mean),
        ("Member 1", members[0]),
        ("Member 2", members[1]),
        ("Member 3", members[2]),
        ("Member 4", members[3]),
    ]

    # Contour levels for smooth rendering
    levels = np.linspace(vmin, vmax, 64)

    image = None
    for ax, (title, data) in zip(axes.ravel(), panels):
        data_cyclic, lon_cyclic = add_cyclic_point(data, coord=longitudes)
        image = ax.contourf(
            lon_cyclic, latitudes, data_cyclic,
            levels=levels, cmap="turbo",
            extend="both",
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=0.8, color="k")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
        ax.set_global()
        ax.set_title(title)

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(unit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not args.stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {args.stats_path}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_members = ensemble_size_or_default(args.ensemble_members)
    if ensemble_members < 4:
        raise ValueError("--ensemble-members must be at least 4 for the requested panel layout")

    log(f"[panels] device={device}")
    log(f"[panels] verify_run={args.verify_run} lead_times={args.lead_times}")
    reader = RawRunReader(args.data_root, args.verify_run)
    latitudes = reader.latitudes
    longitudes = reader.longitudes
    max_lead = max(args.lead_times)
    available_truth_days = reader.available_truth_days(args.start_day)
    if max_lead > available_truth_days:
        raise ValueError(
            f"Lead time {max_lead} exceeds available truth horizon {available_truth_days} "
            f"for {args.verify_run} starting from day {args.start_day}"
        )

    log("[panels] loading stats and checkpoint")
    mean, std = load_stats(args.stats_path, device="cpu")
    model = build_inference_model(
        args.checkpoint_path,
        device=device,
        use_data_parallel=args.use_data_parallel,
    )

    log("[panels] preparing seed state")
    seed_pair = reader.get_input_pair(args.start_day)
    current_input = tensor_from_pair(seed_pair, mean, std, device).repeat(
        ensemble_members, 1, 1, 1
    )

    captured = {}
    with torch.no_grad():
        for lead_day in range(1, max_lead + 1):
            pred_norm, pred_phys = forecast_members_physical(model, current_input, mean, std)
            if lead_day in args.lead_times:
                truth = reader.get_day(args.start_day + 1 + lead_day)
                captured[lead_day] = {
                    "members": pred_phys[:4].copy(),
                    "truth": truth.copy(),
                    "mean": pred_phys.mean(axis=0).copy(),
                }
                log(f"[panels] captured lead day {lead_day}")
            current_input = roll_forward_input(current_input, pred_norm)

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for lead_day, payload in captured.items():
        log(f"[panels] rendering plots for lead day {lead_day}")
        np.savez_compressed(
            data_dir / f"lead_{lead_day:03d}.npz",
            forecast_members=payload["members"].astype(np.float32),
            ensemble_mean=payload["mean"].astype(np.float32),
            truth=payload["truth"].astype(np.float32),
            channel_names=np.array(CHANNEL_NAMES),
        )

        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            save_panel(
                args.results_dir / f"lead_{lead_day:03d}" / f"{channel_name}.png",
                variable_name=channel_name,
                unit=CHANNEL_UNITS[channel_name],
                lead_time=lead_day,
                members=payload["members"][:, channel_idx, :, :],
                truth=payload["truth"][channel_idx],
                latitudes=latitudes,
                longitudes=longitudes,
            )

    metadata = {
        "verify_run": args.verify_run,
        "start_day": args.start_day,
        "lead_times": list(args.lead_times),
        "ensemble_members": ensemble_members,
        "checkpoint_path": repo_relative_path(args.checkpoint_path),
        "stats_path": repo_relative_path(args.stats_path),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    log(f"[panels] saved panel data under {args.output_dir}")
    log(f"[panels] saved panel plots under {args.results_dir}")


if __name__ == "__main__":
    main()
