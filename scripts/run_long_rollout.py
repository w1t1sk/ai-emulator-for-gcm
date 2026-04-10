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
    anomaly_rms_to_climatology,
    build_inference_model,
    cast_save_dtype,
    default_checkpoint_path,
    default_climatology_path,
    default_data_root,
    default_stats_path,
    ensemble_size_or_default,
    forecast_members_physical,
    latitude_weights,
    load_stats,
    roll_forward_input,
    save_wide_metrics_csv,
    tensor_from_pair,
    weighted_acc,
    weighted_field_std,
    weighted_mean,
    weighted_rmse,
    weighted_spread,
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a long free forecast rollout and compute overlap skill metrics"
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root(REPO_ROOT))
    parser.add_argument("--stats-path", type=Path, default=default_stats_path(REPO_ROOT))
    parser.add_argument(
        "--climatology-path",
        type=Path,
        default=default_climatology_path(REPO_ROOT),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=default_checkpoint_path(REPO_ROOT),
    )
    parser.add_argument("--seed-run", default="run16")
    parser.add_argument(
        "--truth-run",
        default="run16",
        help="Run used for ACC/RMSE diagnostics",
    )
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument("--num-days", type=int, default=2000)
    parser.add_argument("--ensemble-members", type=int, default=4)
    parser.add_argument("--chunk-days", type=int, default=365)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print rollout progress every N forecast days",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Storage dtype for saved forecast chunks",
    )
    parser.add_argument(
        "--save-members",
        action="store_true",
        help="Save full ensemble members in addition to the ensemble mean",
    )
    parser.add_argument(
        "--use-data-parallel",
        action="store_true",
        help="Wrap inference in torch.nn.DataParallel when multiple GPUs are visible",
    )
    parser.add_argument(
        "--acc-window-start",
        type=int,
        default=0,
        help="Start of the short-range ACC lead-time window",
    )
    parser.add_argument(
        "--acc-window-end",
        type=int,
        default=30,
        help="End of the short-range ACC lead-time window",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "evaluation" / "long_rollout",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "evaluation" / "long_rollout",
    )
    return parser.parse_args()


def plot_metric_series(
    output_path: Path,
    y_values: np.ndarray,
    title: str,
    y_label: str,
    x_values: np.ndarray,
    x_label: str,
    horizontal_line: float | None = None,
    x_limits: tuple[float, float] | None = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, linewidth=1.6)
    if horizontal_line is not None:
        plt.axhline(horizontal_line, color="gray", linestyle=":", linewidth=1.2)
    if x_limits is not None:
        plt.xlim(*x_limits)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_acc_views(
    output_root: Path,
    y_values: np.ndarray,
    title: str,
    x_values: np.ndarray,
    short_start: int,
    short_end: int,
):
    output_root.mkdir(parents=True, exist_ok=True)

    plot_metric_series(
        output_root / "full_horizon.png",
        y_values=y_values,
        title=title,
        y_label="ACC",
        x_values=x_values,
        x_label="Lead Time (Days)",
        horizontal_line=0.0,
    )

    short_mask = (x_values >= short_start) & (x_values <= short_end)
    if not np.any(short_mask):
        raise ValueError(
            f"No ACC samples found in requested short-range window {short_start}-{short_end}."
        )

    plot_metric_series(
        output_root / f"lead_days_{short_start:02d}_{short_end:02d}.png",
        y_values=y_values[short_mask],
        title=title,
        y_label="ACC",
        x_values=x_values[short_mask],
        x_label="Lead Time (Days)",
        horizontal_line=0.6,
        x_limits=(short_start, short_end),
    )


def save_chunk(
    path: Path,
    forecast_mean: np.ndarray,
    lead_days: np.ndarray,
    channel_names: list[str],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    save_dtype: str,
    forecast_members: np.ndarray | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "forecast_mean": cast_save_dtype(forecast_mean, save_dtype),
        "lead_days": lead_days.astype(np.int32),
        "channel_names": np.array(channel_names),
        "latitudes": latitudes.astype(np.float32),
        "longitudes": longitudes.astype(np.float32),
    }
    if forecast_members is not None:
        payload["forecast_members"] = cast_save_dtype(forecast_members, save_dtype)
    np.savez_compressed(path, **payload)


def save_metadata(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def repo_relative_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), REPO_ROOT.resolve())


def main():
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not args.stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {args.stats_path}")
    if not args.climatology_path.exists():
        raise FileNotFoundError(
            f"Climatology file not found: {args.climatology_path}. "
            "Run scripts/compute_climatology.py first."
        )
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    if args.chunk_days <= 0:
        raise ValueError("--chunk-days must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.start_day < 0:
        raise ValueError("--start-day must be non-negative")
    if args.acc_window_start < 0:
        raise ValueError("--acc-window-start must be at least 0")
    if args.acc_window_end < args.acc_window_start:
        raise ValueError("--acc-window-end must be >= --acc-window-start")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_members = ensemble_size_or_default(args.ensemble_members)
    truth_run = args.truth_run

    log(f"[rollout] device={device}")
    log(f"[rollout] loading run readers for seed={args.seed_run} truth={truth_run}")

    seed_reader = RawRunReader(args.data_root, args.seed_run)
    truth_reader = RawRunReader(args.data_root, truth_run)

    log(f"[rollout] loading climatology from {args.climatology_path}")
    climatology = torch.load(args.climatology_path, map_location="cpu", weights_only=False)
    climatology_map = climatology["mean_map"].numpy()
    latitudes = (
        climatology["latitudes"].numpy()
        if climatology.get("latitudes") is not None
        else seed_reader.latitudes
    )
    longitudes = (
        climatology["longitudes"].numpy()
        if climatology.get("longitudes") is not None
        else seed_reader.longitudes
    )
    weights = latitude_weights(latitudes)

    log(f"[rollout] loading stats from {args.stats_path}")
    mean, std = load_stats(args.stats_path, device="cpu")
    log(f"[rollout] loading checkpoint from {args.checkpoint_path}")
    model = build_inference_model(
        args.checkpoint_path,
        device=device,
        use_data_parallel=args.use_data_parallel,
    )

    log("[rollout] preparing seed state")
    seed_pair = seed_reader.get_input_pair(args.start_day)
    current_input = tensor_from_pair(seed_pair, mean, std, device).repeat(
        ensemble_members, 1, 1, 1
    )

    total_days = args.num_days
    if total_days <= 0:
        raise ValueError("--num-days must be positive")
    truth_overlap_days = min(total_days, truth_reader.available_truth_days(args.start_day))

    outputs_dir = args.output_dir
    results_dir = args.results_dir
    chunks_dir = outputs_dir / "chunks"
    diagnostics_dir = outputs_dir / "diagnostics"
    overlap_plot_dir = results_dir / "truth_overlap"
    drift_plot_dir = results_dir / "free_run"

    for path in [chunks_dir, diagnostics_dir, overlap_plot_dir, drift_plot_dir]:
        path.mkdir(parents=True, exist_ok=True)

    daily_anomaly_rms = np.zeros((total_days, len(CHANNEL_NAMES)), dtype=np.float32)
    daily_field_std = np.zeros_like(daily_anomaly_rms)
    daily_bias_to_clim = np.zeros_like(daily_anomaly_rms)

    overlap_acc = np.full((total_days, len(CHANNEL_NAMES)), np.nan, dtype=np.float32)
    overlap_rmse = np.full_like(overlap_acc, np.nan)
    overlap_spread = np.full_like(overlap_acc, np.nan)

    chunk_mean = np.empty(
        (args.chunk_days, len(CHANNEL_NAMES), *seed_reader.get_day(0).shape[1:]),
        dtype=np.float32,
    )
    chunk_members = (
        np.empty(
            (
                args.chunk_days,
                ensemble_members,
                len(CHANNEL_NAMES),
                *seed_reader.get_day(0).shape[1:],
            ),
            dtype=np.float32,
        )
        if args.save_members
        else None
    )
    chunk_start_day = 1

    print(
        f"Starting free rollout for {total_days} days "
        f"from {args.seed_run} day {args.start_day} using {ensemble_members} members."
    )
    if truth_overlap_days > 0:
        log(
            f"Truth overlap available for {truth_overlap_days} days "
            f"against {truth_run}."
        )
    log(
        f"[rollout] progress updates every {args.progress_every} days; "
        f"chunk save every {args.chunk_days} days"
    )

    for lead_day in range(1, total_days + 1):
        pred_norm, pred_phys = forecast_members_physical(model, current_input, mean, std)
        ensemble_mean = pred_phys.mean(axis=0)

        chunk_idx = (lead_day - 1) % args.chunk_days
        chunk_mean[chunk_idx] = ensemble_mean
        if chunk_members is not None:
            chunk_members[chunk_idx] = pred_phys

        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            clim_field = climatology_map[channel_idx]
            forecast_field = ensemble_mean[channel_idx]
            daily_anomaly_rms[lead_day - 1, channel_idx] = anomaly_rms_to_climatology(
                forecast_field,
                clim_field,
                weights,
            )
            daily_field_std[lead_day - 1, channel_idx] = weighted_field_std(
                forecast_field,
                weights,
            )
            daily_bias_to_clim[lead_day - 1, channel_idx] = weighted_mean(
                forecast_field - clim_field,
                weights,
            )

        if lead_day <= truth_overlap_days:
            truth_field = truth_reader.get_day(args.start_day + 1 + lead_day)
            for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
                overlap_acc[lead_day - 1, channel_idx] = weighted_acc(
                    ensemble_mean[channel_idx],
                    truth_field[channel_idx],
                    climatology_map[channel_idx],
                    weights,
                )
                overlap_rmse[lead_day - 1, channel_idx] = weighted_rmse(
                    ensemble_mean[channel_idx],
                    truth_field[channel_idx],
                    weights,
                )
                overlap_spread[lead_day - 1, channel_idx] = weighted_spread(
                    pred_phys[:, channel_idx, :, :],
                    weights,
                )

        current_input = roll_forward_input(current_input, pred_norm)

        chunk_full = (lead_day % args.chunk_days == 0) or (lead_day == total_days)
        if chunk_full:
            chunk_end_day = lead_day
            valid_length = chunk_idx + 1
            chunk_path = chunks_dir / (
                f"forecast_days_{chunk_start_day:07d}_{chunk_end_day:07d}.npz"
            )
            save_chunk(
                chunk_path,
                forecast_mean=chunk_mean[:valid_length],
                forecast_members=None if chunk_members is None else chunk_members[:valid_length],
                lead_days=np.arange(chunk_start_day, chunk_end_day + 1),
                channel_names=CHANNEL_NAMES,
                latitudes=latitudes,
                longitudes=longitudes,
                save_dtype=args.save_dtype,
            )
            log(f"[rollout] saved chunk {chunk_start_day}-{chunk_end_day} to {chunk_path}")
            chunk_start_day = lead_day + 1

        if (
            lead_day == 1
            or lead_day % args.progress_every == 0
            or lead_day == total_days
        ):
            log(f"[rollout] completed forecast day {lead_day}/{total_days}")

    save_wide_metrics_csv(
        diagnostics_dir / "free_run_daily_metrics.csv",
        lead_values=np.arange(1, total_days + 1),
        metric_names=["anomaly_rms_to_climatology", "field_std", "bias_to_climatology"],
        values={
            "anomaly_rms_to_climatology": daily_anomaly_rms,
            "field_std": daily_field_std,
            "bias_to_climatology": daily_bias_to_clim,
        },
        x_label="forecast_day",
    )

    if truth_overlap_days > 0:
        save_wide_metrics_csv(
            diagnostics_dir / f"truth_overlap_metrics_{truth_run}.csv",
            lead_values=np.arange(1, total_days + 1),
            metric_names=["acc", "rmse", "spread"],
            values={
                "acc": overlap_acc,
                "rmse": overlap_rmse,
                "spread": overlap_spread,
            },
            x_label="lead_day",
        )

    drift_x = np.arange(1, total_days + 1)
    drift_x_label = "Forecast Day"
    overlap_x = np.arange(1, total_days + 1)
    overlap_x_label = "Lead Time (Days)"

    for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
        title = CHANNEL_TITLES[channel_name]
        unit = CHANNEL_UNITS[channel_name]

        plot_metric_series(
            drift_plot_dir / "anomaly_rms_to_climatology" / f"{channel_name}.png",
            daily_anomaly_rms[:, channel_idx],
            title=f"{title}: Anomaly RMS to Climatology",
            y_label=unit,
            x_values=drift_x,
            x_label=drift_x_label,
        )
        plot_metric_series(
            drift_plot_dir / "field_std" / f"{channel_name}.png",
            daily_field_std[:, channel_idx],
            title=f"{title}: Spatial Standard Deviation",
            y_label=unit,
            x_values=drift_x,
            x_label=drift_x_label,
        )
        plot_metric_series(
            drift_plot_dir / "bias_to_climatology" / f"{channel_name}.png",
            daily_bias_to_clim[:, channel_idx],
            title=f"{title}: Mean Bias to Climatology",
            y_label=unit,
            x_values=drift_x,
            x_label=drift_x_label,
            horizontal_line=0.0,
        )

        if truth_overlap_days > 0:
            plot_acc_views(
                overlap_plot_dir / "acc" / channel_name,
                overlap_acc[:, channel_idx],
                title=f"{title}: ACC",
                x_values=overlap_x,
                short_start=args.acc_window_start,
                short_end=args.acc_window_end,
            )

    if truth_overlap_days > 0:
        overall_mean_acc = np.nanmean(overlap_acc, axis=1)
        plot_acc_views(
            overlap_plot_dir / "acc" / "overall_mean_acc",
            overall_mean_acc,
            title="Overall Mean ACC",
            x_values=overlap_x,
            short_start=args.acc_window_start,
            short_end=args.acc_window_end,
        )

    save_metadata(
        outputs_dir / "metadata.json",
        {
            "seed_run": args.seed_run,
            "truth_run": truth_run,
            "start_day": args.start_day,
            "total_days": total_days,
            "truth_overlap_days": int(truth_overlap_days),
            "ensemble_members": ensemble_members,
            "chunk_days": args.chunk_days,
            "save_dtype": args.save_dtype,
            "save_members": args.save_members,
            "checkpoint_path": repo_relative_path(args.checkpoint_path),
            "stats_path": repo_relative_path(args.stats_path),
            "climatology_path": repo_relative_path(args.climatology_path),
            "latitudes": latitudes.tolist(),
            "longitudes": longitudes.tolist(),
            "channel_names": CHANNEL_NAMES,
        },
    )

    log(f"[rollout] saved rollout outputs under {outputs_dir}")
    log(f"[rollout] saved plots under {results_dir}")


if __name__ == "__main__":
    main()
