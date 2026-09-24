# Emulator Aquaplanet Training Repo

This repository contains the main training and evaluation pipeline for the aquaplanet emulator.

The workflow has two training stages:

1. deterministic pretraining
2. probabilistic joint training

It also includes spread fine-tuning, autoregressive rollout, climatology generation, forecast diagnostics, and plotting tools.

The repository is intentionally kept small. Raw data, exploratory notebooks, old plotting scripts, and Slurm output files are not included.

## Repository structure

```text
ai-emulator/
├── artifacts/
│   ├── checkpoints/        # saved model checkpoints
│   ├── data/               # data-layout documentation
│   ├── logs/               # training loss logs
│   └── stats/              # normalization and climatology files
├── docs/
├── outputs/
│   ├── evaluation/         # rollout arrays, diagnostics, and evaluation logs
│   └── training/           # new training and fine-tuning runs
├── results/                # generated plots and figures
├── scripts/
└── src/
    └── emulator/
```

## Raw data

The raw NetCDF dataset is not stored in this repository.

Point the scripts to the dataset with:

```bash
export EMULATOR_DATA_ROOT=/path/to/data
```

The expected run and year layout is described in:

```text
docs/data_layout.md
```

## Environment setup

The recommended environment is provided in:

```text
environment.yml
```

Create it with:

```bash
conda env create -f environment.yml
conda activate emulator
```

If the environment already exists:

```bash
conda env update -n emulator -f environment.yml --prune
conda activate emulator
```

A smaller environment specification is also available:

```bash
conda env create -f environment.from-history.yml
conda activate emulator
```

`requirements.txt` is included as a lightweight fallback.

More environment details are available in:

```text
docs/environment.md
```

## Compute dataset statistics

Before training, generate the normalization statistics if they are not already available:

```bash
python scripts/compute_stats.py \
    --data-root "$EMULATOR_DATA_ROOT"
```

The resulting statistics are used by the training and inference scripts.

## Stage 1 training

Stage 1 performs deterministic pretraining.

For a two-GPU run:

```bash
python -m torch.distributed.run \
    --nproc_per_node=2 \
    scripts/train_stage1.py \
    --data-root "$EMULATOR_DATA_ROOT"
```

The model code is shared across both training stages and includes circular padding for the longitudinal dimension.

## Stage 2 training

Stage 2 starts from the stage-1 model and trains the probabilistic objective.

For a four-GPU run:

```bash
python -m torch.distributed.run \
    --nproc_per_node=4 \
    scripts/train_stage2.py \
    --data-root "$EMULATOR_DATA_ROOT"
```

To resume an existing stage-2 run:

```bash
python -m torch.distributed.run \
    --nproc_per_node=4 \
    scripts/train_stage2.py \
    --data-root "$EMULATOR_DATA_ROOT" \
    --resume-from artifacts/checkpoints/emulator_stage2_best.pth \
    --resume-epoch 23 \
    --save-name emulator_stage2_best_resumed.pth
```

## Autoregressive inference

Run an autoregressive rollout with:

```bash
python scripts/inference_rollout.py \
    --data-root "$EMULATOR_DATA_ROOT" \
    --checkpoint-path artifacts/checkpoints/emulator_stage2_best.pth
```

The rollout feeds the model's own predictions back into the next forecast step.

## Spread fine-tuning

A separate fine-tuning launcher is included for increasing ensemble spread using a stronger KL penalty.

Run it with:

```bash
bash scripts/run_finetune_spread.sh
```

The launcher uses the `gpu_prio` Slurm partition by default and excludes `cn3`.

It resumes from:

```text
artifacts/checkpoints/emulator_stage2_best_resumed.pth
```

The default KL weight is increased from:

```text
1e-4
```

to:

```text
1e-2
```

Each fine-tuning run gets its own output directory:

```text
outputs/training/finetune_spread/<run-name>/
├── checkpoints/
│   ├── emulator_stage2_spread_best_total.pth
│   ├── emulator_stage2_spread_best_crps.pth
│   └── emulator_stage2_spread_last.pth
├── logs/
│   └── finetune_losses.csv
├── metadata/
│   └── run_summary.json
└── slurm/
    └── slurm_<jobid>.out
```

Useful environment-variable overrides include:

```text
EMULATOR_KL_WEIGHT
EMULATOR_FT_LR
EMULATOR_FT_RUN_NAME
EMULATOR_FT_OUTPUT_ROOT
EMULATOR_FT_OUTPUT_DIR
EMULATOR_STAGE2_CKPT
EMULATOR_SLURM_PARTITION
EMULATOR_SLURM_EXCLUDE
```

For example:

```bash
EMULATOR_KL_WEIGHT=0.005 \
EMULATOR_FT_RUN_NAME=test_run \
bash scripts/run_finetune_spread.sh
```

## Evaluation

The evaluation pipeline includes:

- climatology calculation
- long free-running forecasts
- ACC
- RMSE
- spread diagnostics
- forecast and truth comparison panels

Run the default evaluation with:

```bash
conda activate emulator
cd /path/to/ai-emulator
bash scripts/run_evaluation_local.sh
```

By default, the script looks for the latest completed spread fine-tuning run and uses its `best_total` checkpoint.

If no completed fine-tuned checkpoint is available, the launcher exits rather than falling back to an older checkpoint.

The default evaluation uses:

```text
Data root:
../data
or EMULATOR_DATA_ROOT if set

Statistics:
artifacts/stats/stats_full.pt

Climatology:
artifacts/stats/climatology_run16.pt

Verification run:
run16

Rollout length:
2000 days
```

You can choose a checkpoint manually with:

```bash
EMULATOR_STAGE2_CKPT=/path/to/checkpoint.pth \
bash scripts/run_evaluation_local.sh
```

A typical evaluation output is stored under:

```text
outputs/evaluation/finetune_spread__<run-name>__emulator_stage2_spread_best_total/
├── forecast_panels/
├── local_logs/
└── long_rollout/
```

Plots are written to:

```text
results/evaluation/finetune_spread__<run-name>__emulator_stage2_spread_best_total/
├── forecast_panels/
└── long_rollout/
```

More details are available in:

```text
docs/evaluation.md
```

## Typical workflow

A full run looks like this:

```bash
export EMULATOR_DATA_ROOT=/path/to/data

conda activate emulator

python scripts/compute_stats.py \
    --data-root "$EMULATOR_DATA_ROOT"

python -m torch.distributed.run \
    --nproc_per_node=2 \
    scripts/train_stage1.py \
    --data-root "$EMULATOR_DATA_ROOT"

python -m torch.distributed.run \
    --nproc_per_node=4 \
    scripts/train_stage2.py \
    --data-root "$EMULATOR_DATA_ROOT"

python scripts/inference_rollout.py \
    --data-root "$EMULATOR_DATA_ROOT" \
    --checkpoint-path artifacts/checkpoints/emulator_stage2_best.pth
```

For the spread fine-tuning and long-rollout evaluation:

```bash
bash scripts/run_finetune_spread.sh

bash scripts/run_evaluation_local.sh
```

## Main components

| Component | Purpose |
|---|---|
| Stage 1 | Deterministic pretraining |
| Stage 2 | Probabilistic joint training |
| Spread fine-tuning | Increase forecast spread with a stronger KL term |
| Inference rollout | Run the emulator autoregressively |
| Climatology | Build the baseline used for forecast evaluation |
| Diagnostics | Compute ACC, RMSE, and spread |
| Forecast panels | Compare predicted and true fields at selected lead times |
