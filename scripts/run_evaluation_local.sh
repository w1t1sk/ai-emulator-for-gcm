#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${EMULATOR_DATA_ROOT:-$REPO_ROOT/../data}
STATS_PATH=${EMULATOR_STATS_PATH:-$REPO_ROOT/artifacts/stats/stats_full.pt}
CLIM_PATH=${EMULATOR_CLIM_PATH:-$REPO_ROOT/artifacts/stats/climatology_run16.pt}
NUM_DAYS=${EMULATOR_NUM_DAYS:-2000}
ENSEMBLE_MEMBERS=${EMULATOR_ENSEMBLE_MEMBERS:-4}
CHUNK_DAYS=${EMULATOR_CHUNK_DAYS:-365}

ENV_NAME=${EMULATOR_ENV_NAME:-emulator}
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  PYTHON_CMD=(python -u)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "$ENV_NAME" python -u)
else
  echo "Conda not found and environment '$ENV_NAME' is not active." >&2
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp}
export EMULATOR_EVAL_REPO_ROOT="$REPO_ROOT"

if [[ -n "${EMULATOR_STAGE2_CKPT:-}" ]]; then
  CHECKPOINT_PATH=${EMULATOR_STAGE2_CKPT}
else
  CHECKPOINT_PATH=$("${PYTHON_CMD[@]}" -c 'from pathlib import Path; import os; from emulator.evaluation import latest_completed_finetune_best_total_path; checkpoint = latest_completed_finetune_best_total_path(Path(os.environ["EMULATOR_EVAL_REPO_ROOT"])); print("" if checkpoint is None else checkpoint)')
  if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "No completed fine-tune best-total checkpoint found yet." >&2
    echo "Wait for fine-tuning to finish, or set EMULATOR_STAGE2_CKPT manually." >&2
    exit 1
  fi
fi

export EMULATOR_EVAL_CKPT="$CHECKPOINT_PATH"
if [[ -n "${EMULATOR_EVAL_TAG:-}" ]]; then
  EVAL_TAG=${EMULATOR_EVAL_TAG}
else
  EVAL_TAG=$("${PYTHON_CMD[@]}" -c 'from pathlib import Path; import os; from emulator.evaluation import checkpoint_label; print(checkpoint_label(Path(os.environ["EMULATOR_EVAL_CKPT"]), Path(os.environ["EMULATOR_EVAL_REPO_ROOT"])))')
fi

EVAL_OUTPUT_ROOT=${EMULATOR_EVAL_OUTPUT_ROOT:-$REPO_ROOT/outputs/evaluation/$EVAL_TAG}
EVAL_RESULTS_ROOT=${EMULATOR_EVAL_RESULTS_ROOT:-$REPO_ROOT/results/evaluation/$EVAL_TAG}
OUTPUT_DIR=${EMULATOR_LONG_OUTPUT_DIR:-$EVAL_OUTPUT_ROOT/long_rollout}
RESULTS_DIR=${EMULATOR_LONG_RESULTS_DIR:-$EVAL_RESULTS_ROOT/long_rollout}
PANEL_OUTPUT_DIR=${EMULATOR_PANEL_OUTPUT_DIR:-$EVAL_OUTPUT_ROOT/forecast_panels}
PANEL_RESULTS_DIR=${EMULATOR_PANEL_RESULTS_DIR:-$EVAL_RESULTS_ROOT/forecast_panels}
LOG_DIR=${EMULATOR_LOCAL_LOG_DIR:-$EVAL_OUTPUT_ROOT/local_logs}
LOG_FILE=${LOG_DIR}/evaluation_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$RESULTS_DIR" "$PANEL_OUTPUT_DIR" "$PANEL_RESULTS_DIR"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Data root not found: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$STATS_PATH" ]]; then
  echo "Stats file not found: $STATS_PATH" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

echo "Logging to $LOG_FILE"

{
  echo "[$(date)] Starting local evaluation"
  echo "REPO_ROOT=$REPO_ROOT"
  echo "EVAL_TAG=$EVAL_TAG"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "STATS_PATH=$STATS_PATH"
  echo "CLIM_PATH=$CLIM_PATH"
  echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
  echo "NUM_DAYS=$NUM_DAYS"
  echo "ENSEMBLE_MEMBERS=$ENSEMBLE_MEMBERS"
  echo "EVAL_OUTPUT_ROOT=$EVAL_OUTPUT_ROOT"
  echo "EVAL_RESULTS_ROOT=$EVAL_RESULTS_ROOT"
  echo "PYTHON_CMD=${PYTHON_CMD[*]}"

  echo "[$(date)] Step 1/3: compute climatology"
  "${PYTHON_CMD[@]}" "$REPO_ROOT/scripts/compute_climatology.py" \
    --data-root "$DATA_ROOT" \
    --runs run16 \
    --save-path "$CLIM_PATH"

  echo "[$(date)] Step 2/3: run ${NUM_DAYS}-day rollout"
  "${PYTHON_CMD[@]}" "$REPO_ROOT/scripts/run_long_rollout.py" \
    --data-root "$DATA_ROOT" \
    --stats-path "$STATS_PATH" \
    --climatology-path "$CLIM_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --seed-run run16 \
    --truth-run run16 \
    --num-days "$NUM_DAYS" \
    --ensemble-members "$ENSEMBLE_MEMBERS" \
    --chunk-days "$CHUNK_DAYS" \
    --progress-every "${EMULATOR_PROGRESS_EVERY:-10}" \
    --output-dir "$OUTPUT_DIR" \
    --results-dir "$RESULTS_DIR"

  echo "[$(date)] Step 3/3: render forecast panels"
  "${PYTHON_CMD[@]}" "$REPO_ROOT/scripts/plot_forecast_panels.py" \
    --data-root "$DATA_ROOT" \
    --stats-path "$STATS_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --verify-run run16 \
    --lead-times 3 5 7 9 11 \
    --ensemble-members "$ENSEMBLE_MEMBERS" \
    --output-dir "$PANEL_OUTPUT_DIR" \
    --results-dir "$PANEL_RESULTS_DIR"

  echo "[$(date)] Local evaluation finished"
} 2>&1 | tee "$LOG_FILE"
