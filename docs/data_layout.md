# Data Layout

The training scripts expect raw GCM output in the following layout:

```text
data/
├── run1/
│   ├── year1.nc
│   ├── year2.nc
│   └── ...
├── run2/
│   └── ...
└── run16/
    └── year20.nc
```

## Splits Used In Training

- Training runs: `run1` to `run13`
- Validation runs: `run14` to `run16`
- Years per run: `year1.nc` to `year20.nc`

## Variables Expected In Each NetCDF File

2D variables:

- `surface_air_temperature`
- `surface_air_pressure`

3D variables:

- `air_temperature`
- `specific_humidity`
- `eastward_wind`
- `northward_wind`

The current dataset class builds 28 input channels from two consecutive timesteps and predicts 14 target channels at the next timestep.

## Dataset Cache

The cleaned code writes preprocessed cache files to a configurable cache directory instead of the raw data directory.

By default the scripts use:

```text
outputs/cache/
```

That keeps the raw data location read-only if needed.
