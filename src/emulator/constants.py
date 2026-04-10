"""Shared constants for the aquaplanet emulator workflow."""

GRID_SHAPE = (64, 128)
INPUT_CHANNELS = 28
TARGET_CHANNELS = 14
SAMPLES_PER_FILE = 363
DEFAULT_ENSEMBLE_SIZE = 4

VARS_2D = [
    "surface_air_temperature",
    "surface_air_pressure",
]

VARS_3D = [
    "air_temperature",
    "specific_humidity",
    "eastward_wind",
    "northward_wind",
]

DEFAULT_TRAIN_RUNS = [f"run{i}" for i in range(1, 14)]
DEFAULT_VAL_RUNS = [f"run{i}" for i in range(14, 17)]
DEFAULT_YEARS = [f"year{i}.nc" for i in range(1, 21)]
