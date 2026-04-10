# Artifacts

This directory stores lightweight artifacts copied from the original working tree.

- `checkpoints/`: stage-1 and stage-2 model weights
- `logs/`: CSV training logs from the main training stages
- `stats/`: normalization statistics used by the dataset loader
- `data/`: documentation only; the raw dataset is intentionally not copied

The raw dataset remains outside this cleaned repo because it is too large for version control.
