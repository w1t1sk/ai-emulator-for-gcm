#!/bin/bash
#SBATCH --job-name=spr
#SBATCH --partition=gpu_prio
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=cn3
#SBATCH --output=outputs/training/finetune_spread/slurm_%j.out
#SBATCH --error=outputs/training/finetune_spread/slurm_%j.err

set -euo pipefail

# Fine-tune the emulator with higher KL weight for more ensemble spread.
# Run with:  bash scripts/run_finetune_spread.sh
# You can also use: sbatch scripts/run_finetune_spread.sh
# The bash entrypoint is preferred because it wires Slurm stdout/stderr into the per-run folder.

if [[ -n "${EMULATOR_REPO_ROOT:-}" ]]; then
  REPO_ROOT=$EMULATOR_REPO_ROOT
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/run_finetune_spread.sh" ]]; then
  REPO_ROOT=$SLURM_SUBMIT_DIR
else
  REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
export EMULATOR_REPO_ROOT="$REPO_ROOT"
SBATCH_PARTITION=${EMULATOR_SLURM_PARTITION:-gpu_prio}
SBATCH_GPUS=${EMULATOR_SLURM_GPUS:-2}
SBATCH_EXCLUDE=${EMULATOR_SLURM_EXCLUDE:-cn3}

DATA_ROOT=${EMULATOR_DATA_ROOT:-$REPO_ROOT/../data}
STATS_PATH=${EMULATOR_STATS_PATH:-$REPO_ROOT/artifacts/stats/stats_full.pt}
CHECKPOINT=${EMULATOR_STAGE2_CKPT:-$REPO_ROOT/artifacts/checkpoints/emulator_stage2_best_resumed.pth}
CACHE_DIR=${EMULATOR_CACHE_DIR:-$REPO_ROOT/outputs/cache}

# Tuning parameters
KL_WEIGHT=${EMULATOR_KL_WEIGHT:-1e-2}
LR=${EMULATOR_FT_LR:-1e-6}
EPOCHS=${EMULATOR_FT_EPOCHS:-10}
BATCH_SIZE=${EMULATOR_FT_BATCH_SIZE:-8}
ENSEMBLE_SIZE=${EMULATOR_FT_ENS_SIZE:-4}

OUTPUT_ROOT=${EMULATOR_FT_OUTPUT_ROOT:-$REPO_ROOT/outputs/training/finetune_spread}
RUN_NAME=${EMULATOR_FT_RUN_NAME:-kl_${KL_WEIGHT}_lr_${LR}_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${EMULATOR_FT_OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}
SLURM_DIR=$RUN_DIR/slurm
BEST_TOTAL_CKPT=$RUN_DIR/checkpoints/emulator_stage2_spread_best_total.pth
BEST_CRPS_CKPT=$RUN_DIR/checkpoints/emulator_stage2_spread_best_crps.pth
LAST_CKPT=$RUN_DIR/checkpoints/emulator_stage2_spread_last.pth

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "$SLURM_DIR"
  export EMULATOR_SLURM_PARTITION="$SBATCH_PARTITION"
  export EMULATOR_SLURM_GPUS="$SBATCH_GPUS"
  export EMULATOR_SLURM_EXCLUDE="$SBATCH_EXCLUDE"
  export EMULATOR_DATA_ROOT="$DATA_ROOT"
  export EMULATOR_STATS_PATH="$STATS_PATH"
  export EMULATOR_STAGE2_CKPT="$CHECKPOINT"
  export EMULATOR_CACHE_DIR="$CACHE_DIR"
  export EMULATOR_REPO_ROOT="$REPO_ROOT"
  export EMULATOR_KL_WEIGHT="$KL_WEIGHT"
  export EMULATOR_FT_LR="$LR"
  export EMULATOR_FT_EPOCHS="$EPOCHS"
  export EMULATOR_FT_BATCH_SIZE="$BATCH_SIZE"
  export EMULATOR_FT_ENS_SIZE="$ENSEMBLE_SIZE"
  export EMULATOR_FT_OUTPUT_ROOT="$OUTPUT_ROOT"
  export EMULATOR_FT_RUN_NAME="$RUN_NAME"
  export EMULATOR_FT_OUTPUT_DIR="$RUN_DIR"
  echo "Submitting fine-tuning job to partition ${SBATCH_PARTITION} with ${SBATCH_GPUS} GPUs..."
  exec sbatch \
    --chdir="$REPO_ROOT" \
    --partition="$SBATCH_PARTITION" \
    --gpus-per-node="$SBATCH_GPUS" \
    --exclude="$SBATCH_EXCLUDE" \
    --output="$SLURM_DIR/slurm_%j.out" \
    --error="$SLURM_DIR/slurm_%j.err" \
    "$REPO_ROOT/scripts/run_finetune_spread.sh" \
    "$@"
fi

ENV_NAME=${EMULATOR_ENV_NAME:-emulator}
NUM_GPUS=$SBATCH_GPUS

mkdir -p "$RUN_DIR" "$SLURM_DIR"

echo "[$(date)] Starting spread fine-tuning on $(hostname)"
echo "  Partition:  $SBATCH_PARTITION"
echo "  Exclude:    $SBATCH_EXCLUDE"
echo "  Checkpoint: $CHECKPOINT"
echo "  KL weight:  $KL_WEIGHT (was 1e-4)"
echo "  LR:         $LR"
echo "  Epochs:     $EPOCHS"
echo "  GPUs:       $NUM_GPUS"
echo "  Run dir:    $RUN_DIR"
echo "  Slurm log:  $SLURM_DIR"

# Activate the requested environment only when none is already active.
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  "$REPO_ROOT/scripts/finetune_spread.py" \
  --data-root "$DATA_ROOT" \
  --stats-path "$STATS_PATH" \
  --cache-dir "$CACHE_DIR" \
  --resume-from "$CHECKPOINT" \
  --output-dir "$RUN_DIR" \
  --kl-weight "$KL_WEIGHT" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --ensemble-size "$ENSEMBLE_SIZE"

echo ""
echo "[$(date)] Fine-tuning complete."
echo "  Best total checkpoint: $BEST_TOTAL_CKPT"
echo "  Best CRPS checkpoint:  $BEST_CRPS_CKPT"
echo "  Final checkpoint:      $LAST_CKPT"
echo ""
echo "To evaluate the best-total checkpoint:"
echo "  EMULATOR_STAGE2_CKPT=$BEST_TOTAL_CKPT bash scripts/run_evaluation_local.sh"
