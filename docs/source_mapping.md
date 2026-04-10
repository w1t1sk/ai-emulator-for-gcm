# Source Mapping

This cleaned repo was assembled from the original working directories below.

| Original location | Cleaned location | Notes |
| --- | --- | --- |
| `finetune_circular_padding/train_stage1.py` | `scripts/train_stage1.py` | Converted to CLI-driven script with relative imports |
| `finetune_circular_padding/train_stage2.py` | `scripts/train_stage2.py` | Main stage-2 training path |
| `finetune_circular_padding/train_ctd-stage-2.py` | `scripts/train_stage2.py` | Resume logic folded into `--resume-from` and `--resume-epoch` |
| `final/dataset_full.py` and `final_2/dataset_full.py` | `src/emulator/data/dataset_full.py` | Same dataset logic, now with configurable cache directory |
| `final/get_stats_full.py` and `final_2/get_stats_full.py` | `scripts/compute_stats.py` | Paths made configurable |
| `finetune_circular_padding/emulator_model.py` | `src/emulator/models/emulator_model.py` | Circular-padding model retained |
| `finetune_circular_padding/swin_layers.py` | `src/emulator/models/swin_layers.py` | Shared Swin blocks |
| `final_2/inference.py` | `scripts/inference_rollout.py` | Cleaned inference entry point |

Files intentionally omitted from the cleaned repo:

- exploratory notebooks
- plot-generation scripts
- alternate experimental training folders
- the raw `data/` directory
