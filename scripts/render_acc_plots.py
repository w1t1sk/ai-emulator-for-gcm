#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emulator.evaluation import CHANNEL_NAMES, CHANNEL_TITLES
from run_long_rollout import plot_acc_views


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render ACC plots from an existing truth-overlap diagnostics CSV"
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
        help="Path to truth_overlap_metrics_<run>.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to the long_rollout/truth_overlap results directory",
    )
    parser.add_argument("--acc-window-start", type=int, default=0)
    parser.add_argument("--acc-window-end", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {args.metrics_csv}")
    if args.acc_window_start < 0:
        raise ValueError("--acc-window-start must be at least 0")
    if args.acc_window_end < args.acc_window_start:
        raise ValueError("--acc-window-end must be >= --acc-window-start")

    data = np.genfromtxt(args.metrics_csv, delimiter=",", names=True, dtype=np.float64)
    lead_days = np.asarray(data["lead_day"], dtype=np.int32)
    acc_root = args.results_dir / "acc"

    stacked = []
    for channel_name in CHANNEL_NAMES:
        values = np.asarray(data[f"acc__{channel_name}"], dtype=np.float64)
        stacked.append(values)
        plot_acc_views(
            acc_root / channel_name,
            y_values=values,
            title=f"{CHANNEL_TITLES[channel_name]}: ACC",
            x_values=lead_days,
            short_start=args.acc_window_start,
            short_end=args.acc_window_end,
        )

    overall_mean_acc = np.nanmean(np.stack(stacked, axis=1), axis=1)
    plot_acc_views(
        acc_root / "overall_mean_acc",
        y_values=overall_mean_acc,
        title="Overall Mean ACC",
        x_values=lead_days,
        short_start=args.acc_window_start,
        short_end=args.acc_window_end,
    )


if __name__ == "__main__":
    main()
