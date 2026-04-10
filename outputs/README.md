# Outputs

New training runs, caches, raw rollout chunks, and evaluation CSV products should be written here.

Suggested training layout:

- `outputs/training/finetune_spread/<run-name>/checkpoints/`: best-total, best-CRPS, and final checkpoints
- `outputs/training/finetune_spread/<run-name>/logs/`: per-epoch CSV logs
- `outputs/training/finetune_spread/<run-name>/metadata/`: JSON run manifest with hyperparameters and save paths
- `outputs/training/finetune_spread/<run-name>/slurm/`: batch stdout/stderr captured per job

Suggested evaluation layout:

- `outputs/evaluation/<evaluation-tag>/long_rollout/chunks/`: compressed daily forecast chunks
- `outputs/evaluation/<evaluation-tag>/long_rollout/diagnostics/`: ACC, RMSE, spread, and free-run drift CSV files
- `outputs/evaluation/<evaluation-tag>/forecast_panels/data/`: saved arrays for selected lead-time panels
- `outputs/evaluation/<evaluation-tag>/local_logs/`: terminal logs for the full evaluation wrapper

Matching rendered plots live under:

- `results/evaluation/<evaluation-tag>/long_rollout/truth_overlap/acc/<variable>/`: ACC plots for each variable in full-horizon and lead-days-00-30 views
- `results/evaluation/<evaluation-tag>/long_rollout/truth_overlap/acc/overall_mean_acc/`: overall mean ACC in full-horizon and lead-days-00-30 views
- `results/evaluation/<evaluation-tag>/forecast_panels/lead_*/`: forecast-vs-truth panels

This directory is ignored by Git except for this README.
