"""Train the text-map detector on build_cache.py's output.

Usage:
    python ml/textseg/train.py --cache <cache dir> --out <run dir>          # a real run (GPU)
    python ml/textseg/train.py --cache ml/textseg/cache --out ml/train/runs/smoke \
        --smoke                                                             # CPU, minutes

--smoke proves the code path on a handful of pages before a GPU run.
Resumable from <out>/last.pt; --max-hours stops cleanly before a session cap.

Validation uses hand-reviewed pages only, drawn with --val-seed so runs share
them. Its pixel F1 is a training curve; ml/eval/score_detect.py decides.

An EMA of the weights (--ema) is validated each epoch. best.pt holds the EMA at
the best val F1 and is the checkpoint to export; final.pt holds the last epoch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from dataset import TextMapDataset, load_records
from loss import text_map_loss
from model import DEFAULT_BACKBONE, TextMapNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_seed(worker: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker)
    np.random.seed((seed + worker) % 2**32)


def split(records: list[dict], val_pages: int, seed: int) -> tuple[list[dict], list[dict]]:
    reviewed = sorted((r for r in records if r["kind"] == "page" and r["reviewed"]), key=lambda r: r["post"])
    val = random.Random(seed).sample(reviewed, min(val_pages, len(reviewed)))
    held = {r["post"] for r in val}
    return [r for r in records if r["post"] not in held], val


@torch.no_grad()
def validate(net: TextMapNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    net.eval()
    tp = fp = fn = 0.0
    losses = []
    for images, target, mask in loader:
        images, target, mask = images.to(device), target.to(device), mask.to(device)
        logits = net(images)
        losses.append(text_map_loss(logits, target, mask)["loss"].item())
        pred = (torch.sigmoid(logits) > 0.5).float()
        tp += (pred * target * mask).sum().item()
        fp += (pred * (1 - target) * mask).sum().item()
        fn += ((1 - pred) * target * mask).sum().item()
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"val_loss": float(np.mean(losses)), "val_p": precision, "val_r": recall, "val_f1": f1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--crop", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-pages", type=int, default=25)
    parser.add_argument("--val-seed", type=int, default=0,
                        help="draws the val pages; keep it fixed across seeds so runs share a val set")
    parser.add_argument("--ema", type=float, default=0.999, help="EMA decay per optimizer step")
    parser.add_argument("--max-hours", type=float, default=8.0,
                        help="stop cleanly after the epoch that crosses this (Kaggle caps at 9)")
    parser.add_argument("--smoke", action="store_true",
                        help="CPU, random-init backbone, 2 epochs of a few steps: proves the path, nothing else")
    args = parser.parse_args()

    if args.smoke:
        args.epochs, args.batch, args.crop, args.workers, args.val_pages = 2, 2, 384, 0, 2

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    records = load_records(args.cache)
    train_records, val_records = split(records, args.val_pages, args.val_seed)
    if args.smoke:
        train_records = train_records[:8]
    print(f"{len(train_records)} train records ({sum(r['kind'] == 'negative' for r in train_records)} negative), "
          f"{len(val_records)} val pages (hand-reviewed), device {device}", flush=True)

    train_loader = DataLoader(
        TextMapDataset(args.cache, train_records, crop=args.crop, train=True),
        batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
        pin_memory=device.type == "cuda", worker_init_fn=worker_seed, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        TextMapDataset(args.cache, val_records, train=False),
        batch_size=2, shuffle=False, num_workers=args.workers,
    )

    net = TextMapNet(args.backbone, pretrained=not args.smoke).to(device)
    # use_buffers: BatchNorm's running statistics are averaged with the weights.
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup = int(args.warmup_epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch, best_f1 = 0, -1.0
    last_path, best_path, final_path = args.out / "last.pt", args.out / "best.pt", args.out / "final.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_f1 = state["epoch"] + 1, state["best_f1"]
        print(f"resumed from {last_path} at epoch {start_epoch}", flush=True)

    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1),
                                        encoding="utf-8")
    log_path = args.out / "results.csv"
    started = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        net.train()
        epoch_started = time.perf_counter()
        sums = {"loss": 0.0, "bce": 0.0, "dice": 0.0}
        for step, (images, target, mask) in enumerate(train_loader):
            if args.smoke and step >= 3:
                break
            images = images.to(device, non_blocking=True)
            target, mask = target.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = net(images)
            losses = text_map_loss(logits, target, mask)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update_parameters(net)
            for key in sums:
                sums[key] += float(losses[key].detach())
        steps = min(steps_per_epoch, 3) if args.smoke else steps_per_epoch
        row = {"epoch": epoch, **{k: v / max(1, steps) for k, v in sums.items()},
               **validate(ema.module, val_loader, device), "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - epoch_started}

        state = {"model": net.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best_f1": max(best_f1, row["val_f1"]), "backbone": args.backbone}
        weights = {"model": ema.module.state_dict(), "backbone": args.backbone, "epoch": epoch}
        if row["val_f1"] > best_f1:
            best_f1 = row["val_f1"]
            torch.save(weights, best_path)
        torch.save(weights, final_path)
        torch.save(state, last_path)

        with log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
              flush=True)

        if (time.perf_counter() - started) / 3600 > args.max_hours:
            print(f"stopping after epoch {epoch}: --max-hours {args.max_hours} reached; rerun to resume",
                  flush=True)
            break

    print(f"best val f1 {best_f1:.4f} -> {best_path}; EMA weights at the last epoch -> {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
