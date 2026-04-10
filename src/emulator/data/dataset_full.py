from pathlib import Path
import glob
import os

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from torch.utils.data import Dataset

from emulator.constants import SAMPLES_PER_FILE, VARS_2D, VARS_3D


class GCMDatasetFull(Dataset):
    """Dataset that loads two consecutive timesteps and predicts the third."""

    def __init__(self, data_root, stats_path, runs, cache_dir=None):
        self.data_root = Path(data_root)
        self.stats_path = Path(stats_path)
        self.runs = list(runs)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.data_root
        self.samples_per_file = SAMPLES_PER_FILE
        self.year_data = []

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"dataset_cache_{self.runs[0]}_{self.runs[-1]}.pt"
        cache_path = self.cache_dir / cache_name

        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0 and not cache_path.exists():
                print(f"Rank 0: building cache {cache_path} ...")
                self._build_and_save_data(cache_path)
            dist.barrier()
            self.year_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            if not cache_path.exists():
                self._build_and_save_data(cache_path)
            self.year_data = torch.load(cache_path, map_location="cpu", weights_only=False)

    def _build_and_save_data(self, cache_path):
        stats = torch.load(self.stats_path, map_location="cpu", weights_only=True)
        mean = stats["mean"].view(-1, 1, 1)
        std = stats["std"].view(-1, 1, 1)

        temp_data = []
        for run in self.runs:
            run_path = self.data_root / run
            if not run_path.exists():
                raise FileNotFoundError(f"Missing run directory: {run_path}")

            nc_files = sorted(
                glob.glob(os.path.join(run_path, "year*.nc")),
                key=lambda path: int(Path(path).stem.replace("year", "")),
            )

            for path in nc_files:
                with xr.open_dataset(path) as ds:
                    vars_2d = [
                        ds[var_name].values[:, np.newaxis, ...].astype(np.float32)
                        for var_name in VARS_2D
                    ]
                    vars_3d = [
                        ds[var_name].values.astype(np.float32) for var_name in VARS_3D
                    ]

                single_year = np.concatenate(vars_2d + vars_3d, axis=1)
                year_tensor = torch.from_numpy(single_year)
                year_tensor = (year_tensor - mean) / (std + 1e-6)
                temp_data.append(year_tensor)

        torch.save(temp_data, cache_path)

    def __len__(self):
        return len(self.year_data) * self.samples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        day_idx = idx % self.samples_per_file
        tensor = self.year_data[file_idx][day_idx : day_idx + 3]

        input_tensor = torch.cat([tensor[0], tensor[1]], dim=0)
        target_tensor = tensor[2]
        return input_tensor, target_tensor
