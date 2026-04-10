from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from emulator.constants import DEFAULT_ENSEMBLE_SIZE, GRID_SHAPE, VARS_2D, VARS_3D
from emulator.models import EmulatorBase, EmulatorEnsemble, PerturbationModel


CHANNEL_NAMES = [
    "surface_air_temperature",
    "surface_air_pressure",
    "air_temperature_l1",
    "air_temperature_l2",
    "air_temperature_l3",
    "specific_humidity_l1",
    "specific_humidity_l2",
    "specific_humidity_l3",
    "eastward_wind_l1",
    "eastward_wind_l2",
    "eastward_wind_l3",
    "northward_wind_l1",
    "northward_wind_l2",
    "northward_wind_l3",
]

CHANNEL_TITLES = {
    "surface_air_temperature": "Surface Air Temperature",
    "surface_air_pressure": "Surface Air Pressure",
    "air_temperature_l1": "Air Temperature (260 hPa)",
    "air_temperature_l2": "Air Temperature (500 hPa)",
    "air_temperature_l3": "Air Temperature (860 hPa)",
    "specific_humidity_l1": "Specific Humidity (260 hPa)",
    "specific_humidity_l2": "Specific Humidity (500 hPa)",
    "specific_humidity_l3": "Specific Humidity (860 hPa)",
    "eastward_wind_l1": "Eastward Wind (260 hPa)",
    "eastward_wind_l2": "Eastward Wind (500 hPa)",
    "eastward_wind_l3": "Eastward Wind (860 hPa)",
    "northward_wind_l1": "Northward Wind (260 hPa)",
    "northward_wind_l2": "Northward Wind (500 hPa)",
    "northward_wind_l3": "Northward Wind (860 hPa)",
}

CHANNEL_UNITS = {
    "surface_air_temperature": "K",
    "surface_air_pressure": "Pa",
    "air_temperature_l1": "K",
    "air_temperature_l2": "K",
    "air_temperature_l3": "K",
    "specific_humidity_l1": "kg kg$^{-1}$",
    "specific_humidity_l2": "kg kg$^{-1}$",
    "specific_humidity_l3": "kg kg$^{-1}$",
    "eastward_wind_l1": "m s$^{-1}$",
    "eastward_wind_l2": "m s$^{-1}$",
    "eastward_wind_l3": "m s$^{-1}$",
    "northward_wind_l1": "m s$^{-1}$",
    "northward_wind_l2": "m s$^{-1}$",
    "northward_wind_l3": "m s$^{-1}$",
}

def first_existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_data_root(repo_root: Path) -> Path:
    env_value = os.environ.get("EMULATOR_DATA_ROOT")
    if env_value:
        return Path(env_value)

    local_default = first_existing_path(
        repo_root.parent / "data",
        repo_root / "artifacts" / "data",
    )
    if local_default is not None:
        return local_default
    return repo_root.parent / "data"


def default_stats_path(repo_root: Path) -> Path:
    env_value = os.environ.get("EMULATOR_STATS_PATH")
    if env_value:
        return Path(env_value)
    return repo_root / "artifacts" / "stats" / "stats_full.pt"


def default_climatology_path(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "stats" / "climatology_run16.pt"


def artifact_checkpoint_path(repo_root: Path) -> Path:
    fallback = first_existing_path(
        repo_root / "artifacts" / "checkpoints" / "emulator_stage2_best_resumed.pth",
        repo_root / "artifacts" / "checkpoints" / "emulator_stage2_best.pth",
    )
    if fallback is not None:
        return fallback
    return repo_root / "artifacts" / "checkpoints" / "emulator_stage2_best_resumed.pth"


def resolve_repo_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def latest_completed_finetune_run(repo_root: Path) -> dict | None:
    runs_root = repo_root / "outputs" / "training" / "finetune_spread"
    if not runs_root.exists():
        return None

    latest_candidate = None
    latest_key = None

    for summary_path in runs_root.glob("*/metadata/run_summary.json"):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        status = payload.get("status", {})
        if not isinstance(status, dict) or not status.get("completed", False):
            continue

        paths = payload.get("paths", {})
        if not isinstance(paths, dict):
            continue
        checkpoint_paths = paths.get("checkpoint_paths", {})
        if not isinstance(checkpoint_paths, dict):
            continue

        best_total = checkpoint_paths.get("best_total")
        if not best_total:
            continue

        checkpoint_path = resolve_repo_path(best_total, repo_root)
        if not checkpoint_path.exists():
            continue

        try:
            modified_time = summary_path.stat().st_mtime
        except OSError:
            modified_time = 0.0

        run_dir = summary_path.parents[1]
        candidate = {
            "run_dir": run_dir,
            "summary_path": summary_path,
            "summary": payload,
            "best_total_path": checkpoint_path,
        }
        candidate_key = (modified_time, str(run_dir))

        if latest_key is None or candidate_key > latest_key:
            latest_key = candidate_key
            latest_candidate = candidate

    return latest_candidate


def latest_completed_finetune_best_total_path(repo_root: Path) -> Path | None:
    latest_run = latest_completed_finetune_run(repo_root)
    if latest_run is None:
        return None
    return latest_run["best_total_path"]


def default_checkpoint_path(repo_root: Path) -> Path:
    env_value = os.environ.get("EMULATOR_STAGE2_CKPT")
    if env_value:
        return Path(env_value)

    latest_finetune = latest_completed_finetune_best_total_path(repo_root)
    if latest_finetune is not None:
        return latest_finetune

    return artifact_checkpoint_path(repo_root)


def sanitize_label(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return sanitized or "evaluation"


def checkpoint_label(checkpoint_path: Path, repo_root: Path) -> str:
    checkpoint_path = resolve_repo_path(checkpoint_path, repo_root)

    finetune_root = repo_root / "outputs" / "training" / "finetune_spread"
    try:
        relative = checkpoint_path.relative_to(finetune_root)
    except ValueError:
        relative = None

    if relative is not None:
        parts = relative.parts
        if len(parts) >= 3 and parts[1] == "checkpoints":
            run_name = parts[0]
            checkpoint_name = Path(parts[-1]).stem
            return sanitize_label(f"finetune_spread__{run_name}__{checkpoint_name}")

    artifacts_root = repo_root / "artifacts" / "checkpoints"
    try:
        relative = checkpoint_path.relative_to(artifacts_root)
    except ValueError:
        relative = None

    if relative is not None:
        return sanitize_label(f"artifacts__{Path(relative).stem}")

    return sanitize_label(checkpoint_path.stem)


def load_state_dict(path: Path, device: torch.device) -> dict:
    state_dict = torch.load(path, map_location=device, weights_only=True)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value for key, value in state_dict.items()
        }
    return state_dict


def build_inference_model(
    checkpoint_path: Path,
    device: torch.device,
    use_data_parallel: bool = False,
):
    base = EmulatorBase(img_size=GRID_SHAPE).to(device)
    model_p = PerturbationModel(in_channels=28, out_channels=28).to(device)
    model_q = PerturbationModel(in_channels=42, out_channels=28).to(device)
    model = EmulatorEnsemble(base, model_p, model_q).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path, device))
    model.eval()

    if use_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def load_stats(stats_path: Path, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    stats = torch.load(stats_path, map_location=device, weights_only=True)
    return stats["mean"].float(), stats["std"].float()


def normalize_physical_tensor(
    tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (tensor - mean.view(-1, 1, 1)) / (std.view(-1, 1, 1) + 1e-6)


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if tensor.ndim == 5:
        view_shape = (1, 1, -1, 1, 1)
    elif tensor.ndim == 4:
        view_shape = (1, -1, 1, 1)
    elif tensor.ndim == 3:
        view_shape = (-1, 1, 1)
    else:
        raise ValueError(f"Unsupported tensor rank for denormalization: {tensor.ndim}")
    return (tensor * (std.view(*view_shape) + 1e-6)) + mean.view(*view_shape)


def channel_index(name: str) -> int:
    return CHANNEL_NAMES.index(name)


def stack_dataset_channels(ds: xr.Dataset) -> np.ndarray:
    vars_2d = [ds[var_name].values[:, np.newaxis, ...].astype(np.float32) for var_name in VARS_2D]
    vars_3d = [ds[var_name].values.astype(np.float32) for var_name in VARS_3D]
    return np.concatenate(vars_2d + vars_3d, axis=1)


def sorted_year_files(run_path: Path) -> list[Path]:
    return sorted(
        run_path.glob("year*.nc"),
        key=lambda path: int(path.stem.replace("year", "")),
    )


class RawRunReader:
    """Read one run lazily, caching a single yearly file in memory."""

    def __init__(self, data_root: Path | str, run: str):
        self.data_root = Path(data_root)
        self.run = run
        self.run_path = self.data_root / run
        if not self.run_path.exists():
            raise FileNotFoundError(f"Missing run directory: {self.run_path}")

        self.year_files = sorted_year_files(self.run_path)
        if not self.year_files:
            raise FileNotFoundError(f"No year*.nc files found in {self.run_path}")

        self.days_per_year: list[int] = []
        self.latitudes: np.ndarray | None = None
        self.longitudes: np.ndarray | None = None
        for path in self.year_files:
            with xr.open_dataset(path) as ds:
                self.days_per_year.append(int(ds.sizes["time"]))
                if self.latitudes is None and "lat" in ds.coords:
                    self.latitudes = np.asarray(ds["lat"].values, dtype=np.float32)
                if self.longitudes is None and "lon" in ds.coords:
                    self.longitudes = np.asarray(ds["lon"].values, dtype=np.float32)

        if self.latitudes is None:
            self.latitudes = np.linspace(90.0, -90.0, GRID_SHAPE[0], dtype=np.float32)
        if self.longitudes is None:
            self.longitudes = np.linspace(0.0, 360.0, GRID_SHAPE[1], endpoint=False, dtype=np.float32)

        self.offsets = np.cumsum([0] + self.days_per_year)
        self.total_days = int(self.offsets[-1])

        self._cache_year_idx: int | None = None
        self._cache_year_data: np.ndarray | None = None

    def _get_year_and_offset(self, day_index: int) -> tuple[int, int]:
        if day_index < 0 or day_index >= self.total_days:
            raise IndexError(
                f"day_index={day_index} is out of bounds for run {self.run} "
                f"with {self.total_days} days"
            )
        year_idx = int(np.searchsorted(self.offsets[1:], day_index, side="right"))
        within_year = day_index - int(self.offsets[year_idx])
        return year_idx, within_year

    def _load_year(self, year_idx: int) -> np.ndarray:
        if self._cache_year_idx != year_idx or self._cache_year_data is None:
            with xr.open_dataset(self.year_files[year_idx]) as ds:
                self._cache_year_data = stack_dataset_channels(ds)
            self._cache_year_idx = year_idx
        return self._cache_year_data

    def get_day(self, day_index: int) -> np.ndarray:
        year_idx, within_year = self._get_year_and_offset(day_index)
        year_data = self._load_year(year_idx)
        return year_data[within_year].copy()

    def get_input_pair(self, start_day: int) -> np.ndarray:
        return self.get_sequence(start_day, 2)

    def get_sequence(self, start_day: int, num_days: int) -> np.ndarray:
        if start_day < 0:
            raise ValueError("start_day must be non-negative")
        if num_days <= 0:
            raise ValueError("num_days must be positive")
        if start_day + num_days > self.total_days:
            raise ValueError(
                f"Requested days [{start_day}, {start_day + num_days}) exceed "
                f"available range for run {self.run}"
            )

        out = np.empty((num_days, len(CHANNEL_NAMES), *GRID_SHAPE), dtype=np.float32)
        for offset in range(num_days):
            out[offset] = self.get_day(start_day + offset)
        return out

    def available_truth_days(self, start_day: int) -> int:
        return max(self.total_days - start_day - 2, 0)


def latitude_weights(latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes)).astype(np.float64)
    return weights[:, None]


def weighted_mean(field: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(field * weights) / np.sum(weights))


def weighted_rmse(forecast: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
    return math.sqrt(float(np.sum(((forecast - truth) ** 2) * weights) / np.sum(weights)))


def weighted_acc(
    forecast: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    weights: np.ndarray,
) -> float:
    forecast_anom = forecast - climatology
    truth_anom = truth - climatology
    numerator = np.sum(weights * forecast_anom * truth_anom)
    denominator = math.sqrt(
        float(np.sum(weights * (forecast_anom ** 2)) * np.sum(weights * (truth_anom ** 2)))
    )
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def weighted_field_std(field: np.ndarray, weights: np.ndarray) -> float:
    mean_value = weighted_mean(field, weights)
    variance = float(np.sum(weights * ((field - mean_value) ** 2)) / np.sum(weights))
    return math.sqrt(max(variance, 0.0))


def weighted_spread(members: np.ndarray, weights: np.ndarray) -> float:
    spread_field = np.std(members, axis=0)
    return weighted_mean(spread_field, weights)


def anomaly_rms_to_climatology(
    forecast: np.ndarray,
    climatology: np.ndarray,
    weights: np.ndarray,
) -> float:
    return weighted_rmse(forecast, climatology, weights)


def tensor_from_pair(
    pair_physical: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    pair_tensor = torch.from_numpy(pair_physical)
    pair_norm = normalize_physical_tensor(pair_tensor, mean.cpu(), std.cpu())
    return pair_norm.reshape(1, -1, *GRID_SHAPE).to(device)


def roll_forward_input(current_input: torch.Tensor, pred_norm: torch.Tensor) -> torch.Tensor:
    return torch.cat([current_input[:, 14:, :, :], pred_norm], dim=1)


def forecast_members_physical(
    model,
    current_input: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray]:
    with torch.no_grad():
        pred_norm = model(current_input, y=None, num_samples=1)
    pred_phys = denormalize_tensor(pred_norm.detach().cpu(), mean.cpu(), std.cpu()).numpy()
    return pred_norm, pred_phys.astype(np.float32, copy=False)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_wide_metrics_csv(
    path: Path,
    lead_values: np.ndarray,
    metric_names: list[str],
    values: dict[str, np.ndarray],
    x_label: str,
) -> None:
    ensure_parent(path)
    header = [x_label]
    for metric_name in metric_names:
        for channel_name in CHANNEL_NAMES:
            header.append(f"{metric_name}__{channel_name}")

    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for idx, lead_value in enumerate(lead_values):
            row = [str(int(lead_value))]
            for metric_name in metric_names:
                row.extend(str(float(values[metric_name][idx, ch])) for ch in range(len(CHANNEL_NAMES)))
            fh.write(",".join(row) + "\n")


def cast_save_dtype(array: np.ndarray, save_dtype: str) -> np.ndarray:
    if save_dtype == "float16":
        finite = np.isfinite(array)
        if finite.any():
            finfo = np.finfo(np.float16)
            max_value = float(np.max(array[finite]))
            min_value = float(np.min(array[finite]))
            if max_value > finfo.max or min_value < finfo.min:
                raise ValueError(
                    "float16 output would overflow physical forecast values; "
                    "use --save-dtype float32 instead"
                )
        return array.astype(np.float16)
    if save_dtype == "float32":
        return array.astype(np.float32)
    raise ValueError(f"Unsupported save dtype: {save_dtype}")


def years_to_days(num_years: int) -> int:
    return num_years * 365


def ensemble_size_or_default(value: int | None) -> int:
    return value if value is not None else DEFAULT_ENSEMBLE_SIZE
