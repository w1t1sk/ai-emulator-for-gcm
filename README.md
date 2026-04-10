# Emulator Aquaplanet Training Repo

This folder is a cleaned, GitHub-ready export of the main training pipeline used for the circular-padding aquaplanet emulator.

It keeps the core workflow only:

- Stage 1 deterministic pretraining
- Stage 2 probabilistic joint training
- Shared model code with circular padding
- Full-dataset statistics generation
- Autoregressive inference rollout
- Climatology computation, long-rollout diagnostics, and forecast panel plotting
- Lightweight artifacts needed for reproducibility

It intentionally leaves out the exploratory notebooks, plot-generation scripts, SLURM stdout files, and the 168 GB raw dataset.

## Layout

```text
ai-emulator/
├── artifacts/
│   ├── checkpoints/        # copied stage-1 and stage-2 weights
│   ├── data/               # documentation only; raw data is not copied
│   ├── logs/               # copied training CSV logs
│   └── stats/              # copied normalization stats
├── docs/
├── outputs/                # new runs write here
│   ├── evaluation/         # rollout arrays, diagnostics, and local logs
│   └── training/           # structured directories for new training/fine-tune runs
├── results/                # placeholder for plots and figures
├── scripts/
└── src/emulator/
```

## What Was Copied

- `finetune_circular_padding/train_stage1.py`
- `finetune_circular_padding/train_stage2.py`
- `finetune_circular_padding/train_ctd-stage-2.py` logic, folded into a resume option
- `finetune_circular_padding/emulator_model.py`
- `finetune_circular_padding/swin_layers.py`
- `final/dataset_full.py`
- `final/get_stats_full.py`
- `final_2/inference.py`
- relevant stage checkpoints, stats, and CSV loss logs

See [docs/source_mapping.md](docs/source_mapping.md) for the exact mapping from old files to the cleaned structure.

## Raw Data

The raw NetCDF data was not copied into this folder. The scripts expect the original run/year layout documented in [docs/data_layout.md](docs/data_layout.md).

For local use on this machine, you can point the scripts at the existing dataset:

```bash
export EMULATOR_DATA_ROOT=/path/to/data
```

## Environment Setup

The cleaned repo now includes the actual working conda environment exported from this machine, renamed to `emulator` for repository consistency:

- `environment.yml`: full exact export of the `emulator` environment
- `environment.from-history.yml`: lighter recreate-from-history spec, now supplemented with the pip-installed packages from the original environment
- `requirements.txt`: lightweight fallback, not the main reproducibility path

For the closest match to the environment used in the original runs:

```bash
conda env create -f environment.yml
conda activate emulator
```

If the environment already exists and you want to sync it to the exported spec:

```bash
conda env update -n emulator -f environment.yml --prune
conda activate emulator
```

If you want a cleaner base environment instead of the full lock-style export:

```bash
conda env create -f environment.from-history.yml
conda activate emulator
```

More detail is in [docs/environment.md](docs/environment.md).

## Quick Start

Compute stats if needed:

```bash
python scripts/compute_stats.py --data-root "$EMULATOR_DATA_ROOT"
```

Run stage 1:

```bash
python -m torch.distributed.run --nproc_per_node=2 scripts/train_stage1.py \
  --data-root "$EMULATOR_DATA_ROOT"
```

Run stage 2 from the stage-1 checkpoint:

```bash
python -m torch.distributed.run --nproc_per_node=4 scripts/train_stage2.py \
  --data-root "$EMULATOR_DATA_ROOT"
```

Resume stage 2 from an existing stage-2 checkpoint:

```bash
python -m torch.distributed.run --nproc_per_node=4 scripts/train_stage2.py \
  --data-root "$EMULATOR_DATA_ROOT" \
  --resume-from artifacts/checkpoints/emulator_stage2_best.pth \
  --resume-epoch 23 \
  --save-name emulator_stage2_best_resumed.pth
```

Run inference:

```bash
python scripts/inference_rollout.py \
  --data-root "$EMULATOR_DATA_ROOT" \
  --checkpoint-path artifacts/checkpoints/emulator_stage2_best.pth
```

Run the KL-heavier spread fine-tuning on the `gpu_prio` partition with `cn3` excluded:

```bash
bash scripts/run_finetune_spread.sh
```

You can also submit it directly with `sbatch scripts/run_finetune_spread.sh`, but the `bash ...` entrypoint is cleaner because it places the Slurm stdout/stderr inside the run-specific `slurm/` directory automatically.

By default that launcher resumes from `artifacts/checkpoints/emulator_stage2_best_resumed.pth`, raises the KL weight from `1e-4` to `1e-2`, and writes each run into its own labeled directory:

```text
outputs/training/finetune_spread/<run-name>/
├── checkpoints/
│   ├── emulator_stage2_spread_best_total.pth
│   ├── emulator_stage2_spread_best_crps.pth
│   └── emulator_stage2_spread_last.pth
├── logs/finetune_losses.csv
├── metadata/run_summary.json
└── slurm/slurm_<jobid>.out|err
```

Useful overrides:

- `EMULATOR_KL_WEIGHT`: set the KL penalty for spread tuning
- `EMULATOR_FT_LR`: fine-tuning learning rate
- `EMULATOR_FT_RUN_NAME`: custom label for the run directory
- `EMULATOR_FT_OUTPUT_ROOT`: change the base output tree
- `EMULATOR_FT_OUTPUT_DIR`: force an explicit run directory
- `EMULATOR_STAGE2_CKPT`: choose a different stage-2 checkpoint to resume from
- `EMULATOR_SLURM_PARTITION`: override the default Slurm partition
- `EMULATOR_SLURM_EXCLUDE`: override the default excluded node list

## Evaluation Workflow

The cleaned repo now includes a compact evaluation pipeline for:

- climatology estimation from `run16`
- a 2000-day free rollout with daily chunked forecast saves
- ACC, RMSE, and spread diagnostics against `run16`
- forecast-vs-truth panel plots at selected lead times

Run it directly from the terminal on this machine:

```bash
conda activate emulator
cd /path/to/ai-emulator
bash scripts/run_evaluation_local.sh
```

By default, that launcher now does two useful things automatically:

- if a completed spread-fine-tune run exists, it uses that run's `best_total` checkpoint
- it writes rollout arrays, diagnostics, panel data, plots, and logs into a checkpoint-labeled evaluation folder instead of overwriting the earlier baseline outputs

If no completed fine-tune run exists yet, the wrapper exits instead of silently falling back to the old baseline checkpoint.

That launcher uses:

- data: `../data` by default, or `EMULATOR_DATA_ROOT` if set
- checkpoint: latest completed `outputs/training/finetune_spread/.../checkpoints/emulator_stage2_spread_best_total.pth`
- stats: `artifacts/stats/stats_full.pt`
- climatology: `artifacts/stats/climatology_run16.pt`
- verification run: `run16`
- horizon: `2000` days

Example evaluation layout after fine-tuning:

```text
outputs/evaluation/finetune_spread__<run-name>__emulator_stage2_spread_best_total/
├── forecast_panels/
├── local_logs/
└── long_rollout/

results/evaluation/finetune_spread__<run-name>__emulator_stage2_spread_best_total/
├── forecast_panels/
└── long_rollout/
```

You can still override the checkpoint manually with `EMULATOR_STAGE2_CKPT=/path/to/checkpoint.pth bash scripts/run_evaluation_local.sh`.

By default, this workflow now uses `run16` for climatology and verification, which keeps the evaluation self-contained inside the held-out run. On this dataset, `run16` spans 20 years of daily data, so a 2000-day rollout stays fully inside the available truth window.

More detail is in [docs/evaluation.md](docs/evaluation.md).
