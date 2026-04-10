#!/usr/bin/env python3
"""Fine-tune the stage-2 emulator model with a higher KL weight to increase
ensemble spread.  Resumes from the existing best checkpoint and saves a new one.

Launch with torchrun:
    torchrun --nproc_per_node=2 scripts/finetune_spread.py [OPTIONS]
"""

import argparse
import csv
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emulator.constants import DEFAULT_TRAIN_RUNS, DEFAULT_VAL_RUNS, GRID_SHAPE
from emulator.data import GCMDatasetFull
from emulator.models import EmulatorBase, EmulatorEnsemble, PerturbationModel
from emulator.training import Stage2Loss


def parse_args():
    default_data_root = Path(
        os.environ.get("EMULATOR_DATA_ROOT", str((REPO_ROOT.parent / "data").resolve()))
    )
    default_stats_path = Path(
        os.environ.get(
            "EMULATOR_STATS_PATH",
            str(REPO_ROOT / "artifacts" / "stats" / "stats_full.pt"),
        )
    )
    default_cache_dir = Path(
        os.environ.get("EMULATOR_CACHE_DIR", str(REPO_ROOT / "outputs" / "cache"))
    )
    default_output_root = REPO_ROOT / "outputs" / "training" / "finetune_spread"

    parser = argparse.ArgumentParser(
        description="Fine-tune stage-2 with higher KL weight for more ensemble spread"
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--stats-path", type=Path, default=default_stats_path)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=REPO_ROOT / "artifacts" / "checkpoints" / "emulator_stage2_best_resumed.pth",
        help="Stage-2 checkpoint to fine-tune from",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Base directory used when creating an organized run directory",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run label. Defaults to a timestamped KL/LR-based name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit run directory. Overrides --output-root/--run-name when set.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional explicit CSV log path. Defaults to <run-dir>/logs/finetune_losses.csv.",
    )
    parser.add_argument(
        "--save-name",
        default="emulator_stage2_spread_best_total.pth",
        help="Filename for the best validation-total-loss checkpoint",
    )
    parser.add_argument(
        "--best-crps-name",
        default="emulator_stage2_spread_best_crps.pth",
        help="Filename for the best validation-CRPS checkpoint",
    )
    parser.add_argument(
        "--last-name",
        default="emulator_stage2_spread_last.pth",
        help="Filename for the final checkpoint saved after the last epoch",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=1e-2,
        help="KL weight (default 1e-2, was 1e-4 in original training)",
    )
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate (lower than stage-2 since fine-tuning)")
    parser.add_argument("--eta-min", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--train-runs", nargs="+", default=DEFAULT_TRAIN_RUNS)
    parser.add_argument("--val-runs", nargs="+", default=DEFAULT_VAL_RUNS)
    parser.add_argument("--img-height", type=int, default=GRID_SHAPE[0])
    parser.add_argument("--img-width", type=int, default=GRID_SHAPE[1])
    return parser.parse_args()


def load_state_dict(path, device):
    state_dict = torch.load(path, map_location=device, weights_only=True)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value for key, value in state_dict.items()
        }
    return state_dict


def format_float_token(value):
    return f"{value:g}".replace("+", "")


def repo_relative_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), REPO_ROOT.resolve())


def resolve_run_layout(args):
    if args.output_dir is not None:
        output_dir = args.output_dir
        run_name = output_dir.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or (
            f"kl_{format_float_token(args.kl_weight)}"
            f"__lr_{format_float_token(args.lr)}"
            f"__{timestamp}"
        )
        output_dir = args.output_root / run_name

    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    metadata_dir = output_dir / "metadata"
    log_path = args.log_path or (logs_dir / "finetune_losses.csv")

    return {
        "run_name": run_name,
        "output_dir": output_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "metadata_dir": metadata_dir,
        "log_path": log_path,
    }


def build_run_summary(args, layout):
    env = os.environ
    checkpoint_paths = {
        "best_total": repo_relative_path(layout["checkpoints_dir"] / args.save_name),
        "best_crps": repo_relative_path(layout["checkpoints_dir"] / args.best_crps_name),
        "last": repo_relative_path(layout["checkpoints_dir"] / args.last_name),
    }
    return {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "run_name": layout["run_name"],
        "paths": {
            "output_dir": repo_relative_path(layout["output_dir"]),
            "checkpoints_dir": repo_relative_path(layout["checkpoints_dir"]),
            "logs_dir": repo_relative_path(layout["logs_dir"]),
            "metadata_dir": repo_relative_path(layout["metadata_dir"]),
            "log_path": repo_relative_path(layout["log_path"]),
            "resume_from": repo_relative_path(args.resume_from),
            "checkpoint_paths": checkpoint_paths,
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "ensemble_size": args.ensemble_size,
            "kl_weight": args.kl_weight,
            "lr": args.lr,
            "eta_min": args.eta_min,
            "weight_decay": args.weight_decay,
            "train_runs": list(args.train_runs),
            "val_runs": list(args.val_runs),
            "img_height": args.img_height,
            "img_width": args.img_width,
        },
        "data": {
            "data_root": repo_relative_path(args.data_root),
            "stats_path": repo_relative_path(args.stats_path),
            "cache_dir": repo_relative_path(args.cache_dir),
        },
        "slurm": {
            "job_id": env.get("SLURM_JOB_ID"),
            "job_name": env.get("SLURM_JOB_NAME"),
            "partition": env.get("SLURM_JOB_PARTITION"),
            "node_list": env.get("SLURM_JOB_NODELIST"),
        },
        "status": {
            "completed": False,
            "last_finished_epoch": 0,
            "best_val_total_loss": None,
            "best_val_crps": None,
        },
    }


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    args = parse_args()
    layout = resolve_run_layout(args)
    summary_path = layout["metadata_dir"] / "run_summary.json"
    best_total_path = layout["checkpoints_dir"] / args.save_name
    best_crps_path = layout["checkpoints_dir"] / args.best_crps_name
    last_checkpoint_path = layout["checkpoints_dir"] / args.last_name

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not args.stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {args.stats_path}")
    if not args.resume_from.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True

    if dist.get_rank() == 0:
        layout["checkpoints_dir"].mkdir(parents=True, exist_ok=True)
        layout["logs_dir"].mkdir(parents=True, exist_ok=True)
        layout["metadata_dir"].mkdir(parents=True, exist_ok=True)
        layout["log_path"].parent.mkdir(parents=True, exist_ok=True)
        run_summary = build_run_summary(args, layout)
        write_json(summary_path, run_summary)
        with open(layout["log_path"], "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "train_total_loss",
                    "train_crps",
                    "train_kl",
                    "val_total_loss",
                    "val_crps",
                    "val_kl",
                ]
            )

    train_ds = GCMDatasetFull(
        data_root=args.data_root,
        stats_path=args.stats_path,
        runs=args.train_runs,
        cache_dir=args.cache_dir,
    )
    train_sampler = DistributedSampler(train_ds)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_ds = GCMDatasetFull(
        data_root=args.data_root,
        stats_path=args.stats_path,
        runs=args.val_runs,
        cache_dir=args.cache_dir,
    )
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Build model and load existing checkpoint
    base = EmulatorBase(img_size=(args.img_height, args.img_width)).to(device)
    model_p = PerturbationModel(in_channels=28, out_channels=28).to(device)
    model_q = PerturbationModel(in_channels=42, out_channels=28).to(device)
    model = EmulatorEnsemble(base, model_p, model_q).to(device)
    model.load_state_dict(load_state_dict(args.resume_from, device))

    if dist.get_rank() == 0:
        print(f"Loaded checkpoint from {args.resume_from}")
        print(f"Fine-tuning with KL weight = {args.kl_weight} (was 1e-4)")
        print(f"Learning rate = {args.lr}, epochs = {args.epochs}")
        print(f"Run directory = {layout['output_dir']}")

    model = DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False
    )

    criterion = Stage2Loss(
        ensemble_size=args.ensemble_size,
        kl_weight=args.kl_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.eta_min,
    )

    best_val_loss = float("inf")
    best_val_crps = float("inf")

    if dist.get_rank() == 0:
        print("Starting spread fine-tuning...")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_crps = 0.0
        train_kl = 0.0

        for step, (inputs, targets) in enumerate(train_dl):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds_flat, mu_p, logvar_p, mu_q, logvar_q = model(
                inputs,
                y=targets,
                num_samples=args.ensemble_size,
            )
            loss, crps, kl = criterion(
                preds_flat,
                mu_p,
                logvar_p,
                mu_q,
                logvar_q,
                targets,
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_crps += crps.item()
            train_kl += kl.item()

            if dist.get_rank() == 0 and step % 50 == 0:
                print(
                    f"Epoch {epoch + 1} | Batch {step}/{len(train_dl)} | "
                    f"Total: {loss.item():.4f} | CRPS: {crps.item():.4f} | KL: {kl.item():.4f}"
                )

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_crps = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for inputs, targets in val_dl:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds_flat, mu_p, logvar_p, mu_q, logvar_q = model(
                    inputs,
                    y=targets,
                    num_samples=args.ensemble_size,
                )
                loss, crps, kl = criterion(
                    preds_flat,
                    mu_p,
                    logvar_p,
                    mu_q,
                    logvar_q,
                    targets,
                )
                val_loss += loss.item()
                val_crps += crps.item()
                val_kl += kl.item()

        metrics = torch.tensor(
            [train_loss, train_crps, train_kl, val_loss, val_crps, val_kl],
            device=device,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            world_size = dist.get_world_size()
            avg_t_loss = metrics[0].item() / (world_size * len(train_dl))
            avg_t_crps = metrics[1].item() / (world_size * len(train_dl))
            avg_t_kl = metrics[2].item() / (world_size * len(train_dl))
            avg_v_loss = metrics[3].item() / (world_size * len(val_dl))
            avg_v_crps = metrics[4].item() / (world_size * len(val_dl))
            avg_v_kl = metrics[5].item() / (world_size * len(val_dl))
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1:02d} | LR: {current_lr:.7f} | "
                f"Train: {avg_t_loss:.4f} (CRPS {avg_t_crps:.4f}, KL {avg_t_kl:.4f}) | "
                f"Val: {avg_v_loss:.4f} (CRPS {avg_v_crps:.4f}, KL {avg_v_kl:.4f})"
            )

            with open(layout["log_path"], "a", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        epoch + 1,
                        avg_t_loss,
                        avg_t_crps,
                        avg_t_kl,
                        avg_v_loss,
                        avg_v_crps,
                        avg_v_kl,
                    ]
                )

            if avg_v_loss < best_val_loss:
                best_val_loss = avg_v_loss
                torch.save(model.module.state_dict(), best_total_path)
                print(f"Saved new best total-loss checkpoint to {best_total_path}")

            if avg_v_crps < best_val_crps:
                best_val_crps = avg_v_crps
                torch.save(model.module.state_dict(), best_crps_path)
                print(f"Saved new best CRPS checkpoint to {best_crps_path}")

            run_summary["status"]["last_finished_epoch"] = epoch + 1
            run_summary["status"]["best_val_total_loss"] = best_val_loss
            run_summary["status"]["best_val_crps"] = best_val_crps
            run_summary["status"]["latest_metrics"] = {
                "train_total_loss": avg_t_loss,
                "train_crps": avg_t_crps,
                "train_kl": avg_t_kl,
                "val_total_loss": avg_v_loss,
                "val_crps": avg_v_crps,
                "val_kl": avg_v_kl,
                "lr": current_lr,
            }
            write_json(summary_path, run_summary)

    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), last_checkpoint_path)
        run_summary["status"]["completed"] = True
        write_json(summary_path, run_summary)
        print(f"Saved final checkpoint to {last_checkpoint_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
