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
from emulator.models import EmulatorBase
from emulator.training import LatitudeWeightedL1


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
    default_output_dir = REPO_ROOT / "outputs" / "stage1"

    parser = argparse.ArgumentParser(description="Stage-1 deterministic training")
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--stats-path", type=Path, default=default_stats_path)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=default_output_dir / "training_losses_stage1.csv",
    )
    parser.add_argument("--save-name", default="emulator_stage1_base_best.pth")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--train-runs", nargs="+", default=DEFAULT_TRAIN_RUNS)
    parser.add_argument("--val-runs", nargs="+", default=DEFAULT_VAL_RUNS)
    parser.add_argument("--img-height", type=int, default=GRID_SHAPE[0])
    parser.add_argument("--img-width", type=int, default=GRID_SHAPE[1])
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not args.stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {args.stats_path}")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True

    if dist.get_rank() == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_loss_latl1", "val_loss_latl1"])

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

    model = EmulatorBase(img_size=(args.img_height, args.img_width)).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    criterion = LatitudeWeightedL1(h=args.img_height).to(device)
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

    if dist.get_rank() == 0:
        print("Starting stage-1 deterministic pretraining...")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        for step, (inputs, targets) in enumerate(train_dl):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(inputs)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if dist.get_rank() == 0 and step % 100 == 0:
                print(
                    f"Epoch {epoch + 1} | Batch {step}/{len(train_dl)} | "
                    f"Train Lat-L1: {loss.item():.4f}"
                )

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dl:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                pred = model(inputs)
                loss = criterion(pred, targets)
                val_loss += loss.item()

        metrics = torch.tensor([train_loss, val_loss], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            world_size = dist.get_world_size()
            avg_train_loss = metrics[0].item() / (world_size * len(train_dl))
            avg_val_loss = metrics[1].item() / (world_size * len(val_dl))
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1:02d} | LR: {current_lr:.6f} | "
                f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}"
            )

            with open(args.log_path, "a", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = args.output_dir / args.save_name
                torch.save(model.module.state_dict(), save_path)
                print(f"Saved new best stage-1 checkpoint to {save_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
