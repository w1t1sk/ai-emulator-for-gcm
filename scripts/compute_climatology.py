#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emulator.evaluation import (
    CHANNEL_NAMES,
    default_climatology_path,
    default_data_root,
    sorted_year_files,
    stack_dataset_channels,
)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute multi-run spatial climatology")
    parser.add_argument("--data-root", type=Path, default=default_data_root(REPO_ROOT))
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["run16"],
        help="Runs used to estimate climatology",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=default_climatology_path(REPO_ROOT),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    sum_field = None
    sumsq_field = None
    latitudes = None
    longitudes = None
    total_days = 0

    log(f"[climatology] data_root={args.data_root}")
    log(f"[climatology] runs={', '.join(args.runs)}")

    for run in args.runs:
        run_path = args.data_root / run
        if not run_path.exists():
            raise FileNotFoundError(f"Missing run directory: {run_path}")

        year_files = sorted_year_files(run_path)
        if not year_files:
            raise FileNotFoundError(f"No year*.nc files found in {run_path}")

        log(f"[climatology] processing {run} with {len(year_files)} yearly files")

        for year_idx, year_path in enumerate(year_files, start=1):
            with xr.open_dataset(year_path) as ds:
                batch = stack_dataset_channels(ds).astype(np.float64)
                if latitudes is None and "lat" in ds.coords:
                    latitudes = np.asarray(ds["lat"].values, dtype=np.float32)
                if longitudes is None and "lon" in ds.coords:
                    longitudes = np.asarray(ds["lon"].values, dtype=np.float32)

            if sum_field is None:
                sum_field = np.zeros(batch.shape[1:], dtype=np.float64)
                sumsq_field = np.zeros(batch.shape[1:], dtype=np.float64)

            sum_field += batch.sum(axis=0)
            sumsq_field += np.square(batch).sum(axis=0)
            total_days += batch.shape[0]
            log(
                f"[climatology] {run} year {year_idx}/{len(year_files)} complete "
                f"({total_days} days accumulated)"
            )

    if total_days == 0 or sum_field is None or sumsq_field is None:
        raise RuntimeError("No data was accumulated for climatology.")

    mean_map = sum_field / total_days
    variance_map = np.maximum((sumsq_field / total_days) - np.square(mean_map), 0.0)
    std_map = np.sqrt(variance_map)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean_map": torch.from_numpy(mean_map).float(),
            "std_map": torch.from_numpy(std_map).float(),
            "channel_names": CHANNEL_NAMES,
            "runs": list(args.runs),
            "num_days": total_days,
            "latitudes": None if latitudes is None else torch.from_numpy(latitudes),
            "longitudes": None if longitudes is None else torch.from_numpy(longitudes),
        },
        args.save_path,
    )
    log(f"[climatology] saved to {args.save_path}")


if __name__ == "__main__":
    main()
