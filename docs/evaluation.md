# Evaluation Workflow

This repo now includes a compact post-training evaluation path built around three scripts:

- `scripts/compute_climatology.py`
- `scripts/run_long_rollout.py`
- `scripts/plot_forecast_panels.py`

For day-to-day use on this machine, the easiest entrypoint is:

```bash
conda activate emulator
cd /path/to/ai-emulator
bash scripts/run_evaluation_local.sh
```

That wrapper runs the three steps below in sequence and writes a terminal log under `outputs/evaluation/local_logs/`.
When a completed spread-fine-tune run exists, it automatically evaluates the latest `best_total` checkpoint and writes the outputs into a checkpoint-labeled folder under `outputs/evaluation/` and `results/evaluation/`.

## 1. Climatology

Use `run16` to estimate the spatial climatology:

```bash
python scripts/compute_climatology.py \
  --data-root "$EMULATOR_DATA_ROOT" \
  --runs run16
```

This writes:

- `artifacts/stats/climatology_run16.pt`

The saved file contains:

- `mean_map`: `[14, 64, 128]`
- `std_map`: `[14, 64, 128]`
- latitude / longitude coordinates
- channel names and source runs

## 2. Long Rollout

Run a free forecast for 2000 days and save daily data in compressed chunks:

```bash
python scripts/run_long_rollout.py \
  --data-root "$EMULATOR_DATA_ROOT" \
  --seed-run run16 \
  --truth-run run16 \
  --num-days 2000 \
  --ensemble-members 4 \
  --chunk-days 365
```

By default this saves the daily ensemble mean for all 14 variables. If you also want the full ensemble members written into the chunk files, add:

```bash
--save-members
```

Chunk files are saved in `float32` by default. That is intentional, because physical fields such as surface pressure overflow `float16`.

This default setup is now fully truth-verified over the requested horizon:

- `run16` is used for the 2000-day forecast and verification
- `run16` is also used to build the climatology reference
- ACC, RMSE, and spread are computed day by day against `run16`
- climatology-relative drift diagnostics are also saved day by day

`run16` contains `year1.nc` through `year20.nc`, so it provides 7,300 daily states in total. With the first two days used as model input, the verified forecast horizon is 7,298 days, which easily covers the requested 2,000-day evaluation.

Main outputs:

- `outputs/evaluation/<evaluation-tag>/long_rollout/chunks/forecast_days_*.npz`
- `outputs/evaluation/<evaluation-tag>/long_rollout/diagnostics/free_run_daily_metrics.csv`
- `outputs/evaluation/<evaluation-tag>/long_rollout/diagnostics/truth_overlap_metrics_<run>.csv`
- `results/evaluation/<evaluation-tag>/long_rollout/free_run/`
- `results/evaluation/<evaluation-tag>/long_rollout/truth_overlap/`

ACC plots for every variable are saved under:

- `results/evaluation/<evaluation-tag>/long_rollout/truth_overlap/acc/<variable>/full_horizon.png`
- `results/evaluation/<evaluation-tag>/long_rollout/truth_overlap/acc/<variable>/lead_days_00_30.png`
- `results/evaluation/<evaluation-tag>/long_rollout/truth_overlap/acc/overall_mean_acc/full_horizon.png`
- `results/evaluation/<evaluation-tag>/long_rollout/truth_overlap/acc/overall_mean_acc/lead_days_00_30.png`

## 3. Forecast Panel Plots

Save one figure per variable and lead time, with:

- 4 ensemble members
- 1 ensemble mean
- 1 ground truth field

Example:

```bash
python scripts/plot_forecast_panels.py \
  --data-root "$EMULATOR_DATA_ROOT" \
  --verify-run run16 \
  --lead-times 3 5 7 9 11 \
  --ensemble-members 4
```

Outputs:

- `outputs/evaluation/<evaluation-tag>/forecast_panels/data/lead_*.npz`
- `results/evaluation/<evaluation-tag>/forecast_panels/lead_*/<variable>.png`
