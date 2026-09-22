#!/usr/bin/env python3
"""Train FieldLens runs 1-3."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fieldlens.constants import NUM_CLASSES  # noqa: E402
from fieldlens.dataset import FieldLensDataset, collate_fieldlens  # noqa: E402
from fieldlens.losses import build_loss  # noqa: E402
from fieldlens.metrics import iou_from_confusion, logits_to_pred_class, update_agri_confusion  # noqa: E402
from fieldlens.models import build_model, count_parameters  # noqa: E402
from fieldlens.paths import REPO_ROOT, repo_path  # noqa: E402
from fieldlens.profile import (  # noqa: E402
    add_profile_arg,
    apply_profile_to_run_cfg,
    load_profile,
    profile_ckpt_dir,
    profile_run_dir,
)


def make_grad_scaler(enabled: bool):
    """GradScaler across torch versions (torch.cuda.amp.GradScaler is deprecated)."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # torch < 2.3


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_data_paths(cfg: dict) -> dict:
    data = cfg["data"]
    return {
        "subset_root": repo_path(data["subset_root"]),
        "split_csv": repo_path(data["split_csv"]),
        "stats_json": repo_path(data["stats_json"]),
        "channels": data["channels"],
        "batch_size": data["batch_size"],
        "num_workers": data["num_workers"],
        "pin_memory": data["pin_memory"],
    }


def make_loader(ds, cfg_data: dict, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg_data["batch_size"],
        shuffle=shuffle,
        num_workers=cfg_data["num_workers"],
        pin_memory=cfg_data["pin_memory"],
        collate_fn=collate_fieldlens,
    )


def poly_lr(base: float, epoch: int, epochs: int, warmup: int) -> float:
    if epoch < warmup:
        return base * float(epoch + 1) / float(max(warmup, 1))
    t = (epoch - warmup) / float(max(epochs - warmup, 1))
    return base * (1.0 - t) ** 0.9


def set_lrs(optimizer: torch.optim.Optimizer, enc_lr: float, other_lr: float) -> None:
    optimizer.param_groups[0]["lr"] = enc_lr
    optimizer.param_groups[1]["lr"] = other_lr


@torch.no_grad()
def validate(model, loader, device, head: str, threshold: float, loss_fn, amp: bool) -> dict[str, float]:
    """One val pass. Returns losses, pixel accuracy, standard mIoU, modified mIoU."""
    model.eval()
    conf_mod = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    conf_std = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    loss_sum = 0.0
    batches = 0
    correct = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device)
        with autocast(device, amp):
            logits = model(images)
            if head == "softmax":
                loss = loss_fn(logits, batch["singlelabel"].to(device))
            else:
                loss = loss_fn(
                    logits,
                    batch["multilabel"].to(device),
                    batch["valid"].to(device),
                )
        loss_sum += float(loss.item())
        batches += 1
        logits_np = logits.float().cpu().numpy()
        for b in range(images.size(0)):
            pred = logits_to_pred_class(logits_np[b], head=head, threshold=threshold)
            gt_multi = batch["multilabel"][b].numpy()
            valid = batch["valid"][b].numpy().astype(bool)
            single = batch["singlelabel"][b].numpy()
            update_agri_confusion(conf_mod, pred, gt_multi, valid)
            # Standard single label confusion on valid pixels
            pv = pred[valid]
            sv = single[valid]
            for c in range(NUM_CLASSES):
                mask = sv == c
                if not mask.any():
                    continue
                conf_std[c] += np.bincount(pv[mask], minlength=NUM_CLASSES)
            correct += int(np.count_nonzero(pv == sv))
            total += int(valid.sum())
    _, mod_miou = iou_from_confusion(conf_mod)
    _, std_miou = iou_from_confusion(conf_std)
    acc = float(correct / total) if total else 0.0
    return {
        "val_loss": loss_sum / max(batches, 1),
        "val_pixel_accuracy": acc,
        "val_miou": float(std_miou),
        "val_modified_miou": float(mod_miou),
    }


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, cfg, accum, amp):
    model.train()
    running = 0.0
    steps = 0
    pending = 0
    correct = 0
    total = 0
    optimizer.zero_grad(set_to_none=True)
    head = cfg["model"]["head"]
    threshold = float(cfg.get("threshold", 0.5))

    def step() -> None:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        images = batch["image"].to(device)
        try:
            with autocast(device, amp):
                logits = model(images)
                if head == "softmax":
                    loss = loss_fn(logits, batch["singlelabel"].to(device))
                else:
                    loss = loss_fn(
                        logits,
                        batch["multilabel"].to(device),
                        batch["valid"].to(device),
                    )
                loss = loss / accum
            scaler.scale(loss).backward()
            pending += 1
            if pending == accum:
                step()
                pending = 0
            running += loss.item() * accum
            steps += 1
            with torch.no_grad():
                logits_np = logits.float().cpu().numpy()
                for b in range(images.size(0)):
                    pred = logits_to_pred_class(logits_np[b], head=head, threshold=threshold)
                    valid = batch["valid"][b].numpy().astype(bool)
                    single = batch["singlelabel"][b].numpy()
                    correct += int(np.count_nonzero(pred[valid] == single[valid]))
                    total += int(valid.sum())
        except torch.cuda.OutOfMemoryError:
            print(
                "CUDA out of memory. Try a smaller batch_size and raise "
                "optim.accum_steps so the effective batch stays similar."
            )
            raise SystemExit(2) from None
    if pending:
        # Trailing micro-batches when the epoch length is not a multiple of accum.
        step()
    return {
        "train_loss": running / max(steps, 1),
        "train_pixel_accuracy": float(correct / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    add_profile_arg(parser)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Also keep a per-epoch checkpoint (last.pt and best.pt are always written)",
    )
    args = parser.parse_args()

    profile = load_profile(args.profile)
    cfg = apply_profile_to_run_cfg(load_cfg(Path(args.config)), profile)
    print(f"profile={args.profile}")
    set_seed(cfg["seed"])
    data_paths = resolve_data_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["optim"]["amp"]) and device.type == "cuda"

    train_ds = FieldLensDataset(
        subset_root=data_paths["subset_root"],
        split_csv=data_paths["split_csv"],
        our_split="train",
        stats_json=data_paths["stats_json"],
        channels=data_paths["channels"],
        augment=True,
        seed=cfg["seed"],
    )
    val_ds = FieldLensDataset(
        subset_root=data_paths["subset_root"],
        split_csv=data_paths["split_csv"],
        our_split="val",
        stats_json=data_paths["stats_json"],
        channels=data_paths["channels"],
        augment=False,
        seed=cfg["seed"],
    )

    if args.smoke:
        train_ds = Subset(train_ds, list(range(min(20, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(8, len(val_ds)))))
        cfg["optim"]["epochs"] = 2

    train_loader = make_loader(train_ds, data_paths, shuffle=True)
    val_loader = make_loader(val_ds, data_paths, shuffle=False)

    model = build_model(cfg).to(device)
    info = count_parameters(model)
    print(f"params={info.parameters} size_mb={info.size_mb:.2f} device={device}")

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": cfg["optim"]["encoder_lr"]},
            {"params": other_params, "lr": cfg["optim"]["other_lr"]},
        ],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    scaler = make_grad_scaler(amp)
    loss_fn = build_loss(cfg).to(device)

    run_name = cfg["run_name"]
    ckpt_dir = profile_ckpt_dir(args.profile, run_name)
    run_dir = profile_run_dir(args.profile, run_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_csv = run_dir / "log.csv"
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    start_epoch = 0
    best_miou = -1.0
    if args.resume and (ckpt_dir / "last.pt").is_file():
        ckpt = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", -1.0)
        print(f"Resumed from epoch {start_epoch}")

    if args.benchmark:
        model.train()
        times = []
        peak = 0.0
        it = iter(train_loader)
        for step in range(100):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            images = batch["image"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with autocast(device, amp):
                logits = model(images)
                if cfg["model"]["head"] == "softmax":
                    loss = loss_fn(logits, batch["singlelabel"].to(device))
                else:
                    loss = loss_fn(
                        logits, batch["multilabel"].to(device), batch["valid"].to(device)
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
                peak = max(peak, torch.cuda.max_memory_allocated() / 1024**2)
            times.append(time.perf_counter() - t0)
        sec = float(np.mean(times))
        steps_per_epoch = max(len(train_loader), 1)
        print(f"sec_per_step={sec:.4f}")
        print(f"est_sec_per_epoch={sec * steps_per_epoch:.1f}")
        print(f"est_sec_full_run={sec * steps_per_epoch * cfg['optim']['epochs']:.1f}")
        print(f"peak_vram_mb={peak:.1f}")
        return

    if not log_csv.exists():
        with log_csv.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_pixel_accuracy",
                    "val_pixel_accuracy",
                    "val_miou",
                    "val_modified_miou",
                    "lr_encoder",
                    "lr_other",
                    "epoch_sec",
                    "peak_vram_mb",
                ]
            )

    epochs = cfg["optim"]["epochs"]
    accum = int(cfg["optim"].get("accum_steps", 1))
    threshold = float(cfg.get("threshold", 0.5))

    for epoch in range(start_epoch, epochs):
        enc_lr = poly_lr(cfg["optim"]["encoder_lr"], epoch, epochs, cfg["optim"]["warmup_epochs"])
        other_lr = poly_lr(cfg["optim"]["other_lr"], epoch, epochs, cfg["optim"]["warmup_epochs"])
        set_lrs(optimizer, enc_lr, other_lr)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device, cfg, accum, amp
        )
        val_stats = validate(
            model, val_loader, device, cfg["model"]["head"], threshold, loss_fn, amp
        )
        epoch_sec = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0

        with log_csv.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{train_stats['train_loss']:.6f}",
                    f"{val_stats['val_loss']:.6f}",
                    f"{train_stats['train_pixel_accuracy']:.6f}",
                    f"{val_stats['val_pixel_accuracy']:.6f}",
                    f"{val_stats['val_miou']:.6f}",
                    f"{val_stats['val_modified_miou']:.6f}",
                    f"{enc_lr:.8f}",
                    f"{other_lr:.8f}",
                    f"{epoch_sec:.2f}",
                    f"{peak:.1f}",
                ]
            )
        writer.add_scalar("loss/train", train_stats["train_loss"], epoch)
        writer.add_scalar("loss/val", val_stats["val_loss"], epoch)
        writer.add_scalar("metrics/val_pixel_accuracy", val_stats["val_pixel_accuracy"], epoch)
        writer.add_scalar("metrics/val_miou", val_stats["val_miou"], epoch)
        writer.add_scalar("metrics/val_modified_miou", val_stats["val_modified_miou"], epoch)
        print(
            f"epoch={epoch} train_loss={train_stats['train_loss']:.4f} "
            f"val_loss={val_stats['val_loss']:.4f} "
            f"train_acc={train_stats['train_pixel_accuracy']:.4f} "
            f"val_acc={val_stats['val_pixel_accuracy']:.4f} "
            f"val_miou={val_stats['val_miou']:.4f} "
            f"val_mod_miou={val_stats['val_modified_miou']:.4f} "
            f"sec={epoch_sec:.1f} peak_vram_mb={peak:.1f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
            "cfg": cfg,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if args.save_every_epoch:
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
        if val_stats["val_modified_miou"] > best_miou:
            best_miou = val_stats["val_modified_miou"]
            ckpt["best_miou"] = best_miou
            torch.save(ckpt, ckpt_dir / "best.pt")

    writer.close()


if __name__ == "__main__":
    main()
