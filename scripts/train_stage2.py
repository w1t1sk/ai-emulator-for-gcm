#!/usr/bin/env python3

import argparse
import csv
import os
import sys
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
    default_output_dir = REPO_ROOT / "outputs" / "stage2"

    parser = argparse.ArgumentParser(description="Stage-2 probabilistic joint training")
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--stats-path", type=Path, default=default_stats_path)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir)
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "checkpoints" / "emulator_stage1_base_best.pth",
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-epoch", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=default_output_dir / "training_losses_stage2.csv",
    )
    parser.add_argument("--save-name", default="emulator_stage2_best.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=4.5e-6)
    parser.add_argument("--eta-min", type=float, default=1e-7)
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


def main():
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not args.stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {args.stats_path}")
    if args.resume_from is None and not args.stage1_checkpoint.exists():
        raise FileNotFoundError(
            f"Stage-1 checkpoint not found: {args.stage1_checkpoint}"
        )
    if args.resume_from is not None and not args.resume_from.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_from}")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True

    if dist.get_rank() == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume_from is None or not args.log_path.exists():
            with open(args.log_path, "w", newline="") as handle:
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

    base = EmulatorBase(img_size=(args.img_height, args.img_width)).to(device)
    model_p = PerturbationModel(in_channels=28, out_channels=28).to(device)
    model_q = PerturbationModel(in_channels=42, out_channels=28).to(device)
    model = EmulatorEnsemble(base, model_p, model_q).to(device)

    if args.resume_from is not None:
        model.load_state_dict(load_state_dict(args.resume_from, device))
        start_epoch = args.resume_epoch
        if dist.get_rank() == 0:
            print(f"Loaded stage-2 checkpoint from {args.resume_from}")
    else:
        base.load_state_dict(load_state_dict(args.stage1_checkpoint, device))
        start_epoch = 0
        if dist.get_rank() == 0:
            print(f"Loaded stage-1 checkpoint from {args.stage1_checkpoint}")

    model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

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

    for _ in range(start_epoch):
        scheduler.step()

    best_val_crps = float("inf")

    if dist.get_rank() == 0:
        print("Starting stage-2 probabilistic joint training...")

    for epoch in range(start_epoch, args.epochs):
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
                f"Epoch {epoch + 1:02d} | LR: {current_lr:.6f} | "
                f"Train: {avg_t_loss:.4f} (CRPS {avg_t_crps:.4f}, KL {avg_t_kl:.4f}) | "
                f"Val: {avg_v_loss:.4f} (CRPS {avg_v_crps:.4f}, KL {avg_v_kl:.4f})"
            )

            with open(args.log_path, "a", newline="") as handle:
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

            if avg_v_crps < best_val_crps:
                best_val_crps = avg_v_crps
                save_path = args.output_dir / args.save_name
                torch.save(model.module.state_dict(), save_path)
                print(f"Saved new best stage-2 checkpoint to {save_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
