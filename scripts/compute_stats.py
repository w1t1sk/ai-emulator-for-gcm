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

from emulator.constants import DEFAULT_TRAIN_RUNS, DEFAULT_YEARS, VARS_2D, VARS_3D


def parse_args():
    default_data_root = Path(
        os.environ.get("EMULATOR_DATA_ROOT", str((REPO_ROOT.parent / "data").resolve()))
    )
    parser = argparse.ArgumentParser(description="Compute normalization statistics")
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "stats" / "stats_full.pt",
    )
    parser.add_argument("--train-runs", nargs="+", default=DEFAULT_TRAIN_RUNS)
    parser.add_argument("--years", nargs="+", default=DEFAULT_YEARS)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    n_channels = len(VARS_2D) + (len(VARS_3D) * 3)
    sum_x = np.zeros(n_channels, dtype=np.float64)
    sum_sq_x = np.zeros(n_channels, dtype=np.float64)
    total_pixels = 0

    for run in args.train_runs:
        for year_file in args.years:
            path = args.data_root / run / year_file
            if not path.exists():
                raise FileNotFoundError(f"Missing NetCDF file: {path}")

            with xr.open_dataset(path) as ds:
                data_2d = [ds[var_name].values[:, np.newaxis, :, :] for var_name in VARS_2D]
                data_3d = [ds[var_name].values for var_name in VARS_3D]

            batch_2d = np.concatenate(data_2d, axis=1)
            batch_3d = np.concatenate(data_3d, axis=1)
            batch = np.concatenate([batch_2d, batch_3d], axis=1).astype(np.float64)

            sum_x += np.sum(batch, axis=(0, 2, 3))
            sum_sq_x += np.sum(batch ** 2, axis=(0, 2, 3))
            total_pixels += batch.shape[0] * batch.shape[2] * batch.shape[3]

    mean = sum_x / total_pixels
    variance = (sum_sq_x / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"mean": torch.from_numpy(mean).float(), "std": torch.from_numpy(std).float()},
        args.save_path,
    )
    print(f"Saved stats to {args.save_path}")


if __name__ == "__main__":
    main()
