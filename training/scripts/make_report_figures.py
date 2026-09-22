#!/usr/bin/env python3
"""Build report PNGs under notebooks/figures/ for the active profile."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fieldlens.constants import ANOMALY_CLASSES, CLASS_COLORS_RGB  # noqa: E402
from fieldlens.paths import REPO_ROOT  # noqa: E402
from fieldlens.transforms_drone import apply_drone_condition  # noqa: E402

ANOMALY = list(ANOMALY_CLASSES)
COLORS = {c: "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS_RGB[c]) for c in ANOMALY}


def load_valid(base: Path, tid: str) -> np.ndarray:
    b = np.array(Image.open(base / "boundaries" / f"{tid}.png").convert("L")) > 0
    m = np.array(Image.open(base / "masks" / f"{tid}.png").convert("L")) > 0
    return b & m


def gt_overlay(base: Path, tid: str, valid: np.ndarray) -> np.ndarray:
    h, w = valid.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for c in ANOMALY:
        p = base / "labels" / c / f"{tid}.png"
        if not p.is_file():
            continue
        lab = np.array(Image.open(p).convert("L")) > 0
        out[lab & valid] = (*CLASS_COLORS_RGB[c], 200)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="pilot_v2")
    args = parser.parse_args()
    profile = args.profile
    data = REPO_ROOT / "data" / "fieldlens" / profile
    stats_path = data / "stats.json"
    split_path = data / "split.csv"
    subset = data / "tiles"
    out = REPO_ROOT / "notebooks" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    print("PROFILE", profile)

    rows = list(csv.DictReader(split_path.open())) if split_path.is_file() else []
    sample = [r for r in rows if r["our_split"] == "test"][:4]

    # 1 sample tiles
    if sample:
        fig, axes = plt.subplots(len(sample), 4, figsize=(12, 3 * len(sample)))
        if len(sample) == 1:
            axes = np.array([axes])
        for i, r in enumerate(sample):
            base = subset / r["source_split"]
            tid = r["tile_id"]
            rgb = np.array(Image.open(base / "images" / "rgb" / f"{tid}.jpg").convert("RGB"))
            nir_p = base / "images" / "nir" / f"{tid}.jpg"
            if not nir_p.is_file():
                nir_p = base / "images" / "nir" / f"{tid}.png"
            nir = np.array(Image.open(nir_p).convert("L"))
            valid = load_valid(base, tid)
            gt = gt_overlay(base, tid, valid)
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title("RGB")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(nir, cmap="gray")
            axes[i, 1].set_title("NIR")
            axes[i, 1].axis("off")
            axes[i, 2].imshow(rgb)
            axes[i, 2].imshow(gt)
            axes[i, 2].set_title("Ground truth")
            axes[i, 2].axis("off")
            axes[i, 3].imshow(valid, cmap="gray")
            axes[i, 3].set_title("Valid mask")
            axes[i, 3].axis("off")
        fig.suptitle(f"Dataset samples ({profile})")
        fig.tight_layout()
        fig.savefig(out / "sample_tiles.png", dpi=140)
        plt.close(fig)
        print("wrote sample_tiles.png")

    # 2 drone conditions
    eval_cfg = yaml.safe_load((REPO_ROOT / "training" / "configs" / "eval.yaml").read_text())
    if sample:
        r = sample[0]
        base = subset / r["source_split"]
        tid = r["tile_id"]
        rgb = np.array(
            Image.open(base / "images" / "rgb" / f"{tid}.jpg").convert("RGB"), dtype=np.float32
        ) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1)
        effects = [
            ("motion_blur", "blur"),
            ("downscale", "low resolution"),
            ("brightness", "brightness"),
            ("noise", "noise"),
        ]
        fig, axes = plt.subplots(len(effects), 5, figsize=(12, 2.4 * len(effects)))
        for row, (effect, title) in enumerate(effects):
            for sev in range(5):
                y = apply_drone_condition(x.clone(), sev, eval_cfg, effect=effect) if sev else x
                img = y.permute(1, 2, 0).numpy().clip(0, 1)
                axes[row, sev].imshow(img)
                axes[row, sev].set_title(f"{title} s{sev}" if row == 0 else f"s{sev}")
                axes[row, sev].axis("off")
            axes[row, 0].set_ylabel(title)
        fig.suptitle("Drone capture simulation by effect and severity")
        fig.tight_layout()
        fig.savefig(out / "drone_conditions.png", dpi=140)
        plt.close(fig)
        print("wrote drone_conditions.png")

    # 3 class distribution
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        pc = stats["per_class"]
        fig, ax = plt.subplots(figsize=(10, 4))
        xpos = np.arange(len(ANOMALY))
        w = 0.25
        for i, sp in enumerate(["train", "val", "test"]):
            vals = [pc[sp][c]["tile_count"] for c in ANOMALY]
            ax.bar(xpos + i * w, vals, w, label=sp, color=["#0d9488", "#2563eb", "#ca8a04"][i])
        ax.set_xticks(xpos + w)
        ax.set_xticklabels(ANOMALY, rotation=45, ha="right")
        ax.set_ylabel("Tiles with class")
        ax.set_title(f"Class distribution by split ({profile})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "class_distribution.png", dpi=140)
        plt.close(fig)
        print("wrote class_distribution.png")

    def load_log(run: str):
        p = REPO_ROOT / "runs" / profile / run / "log.csv"
        return list(csv.DictReader(p.open())) if p.is_file() else []

    def load_metrics(run: str):
        p = REPO_ROOT / "runs" / profile / run / "eval" / "metrics.json"
        return json.loads(p.read_text()) if p.is_file() else None

    runs = ["run1", "run2", "run3"]
    run_colors = {"run1": "#0d9488", "run2": "#2563eb", "run3": "#ca8a04"}

    # 4 5 6 curves
    for metric, title, fname in [
        ("train_loss", "Training loss", "loss_curves.png"),
        ("val_loss", "Validation loss", "val_loss_curves.png"),
        ("val_pixel_accuracy", "Validation pixel accuracy", "accuracy_curves.png"),
        ("train_pixel_accuracy", "Training pixel accuracy", "train_accuracy_curves.png"),
        ("val_modified_miou", "Validation modified mIoU", "miou_curves.png"),
        ("val_miou", "Validation mIoU", "val_miou_curves.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        any_line = False
        for run in runs:
            rows = load_log(run)
            if not rows:
                continue
            key = metric if metric in rows[0] else ("val_miou" if metric == "val_modified_miou" else None)
            if key is None or key not in rows[0]:
                continue
            ep = [int(float(r["epoch"])) for r in rows]
            vals = [float(r[key]) for r in rows]
            ax.plot(ep, vals, label=run, color=run_colors[run])
            any_line = True
        if any_line:
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / fname, dpi=140)
            print("wrote", fname)
        plt.close(fig)

    # Combined loss/acc/miou requested names
    # also write training_curves.png as loss overview
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, metric, title in zip(
        axes,
        ["train_loss", "val_pixel_accuracy", "val_modified_miou"],
        ["Train loss", "Val pixel accuracy", "Val modified mIoU"],
    ):
        for run in runs:
            rows = load_log(run)
            if not rows:
                continue
            key = metric if metric in rows[0] else ("val_miou" if "val_miou" in rows[0] else None)
            if not key:
                continue
            ax.plot(
                [int(float(r["epoch"])) for r in rows],
                [float(r[key]) for r in rows],
                label=run,
                color=run_colors[run],
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=140)
    plt.close(fig)
    print("wrote training_curves.png")

    # 7 confusion
    for run in runs:
        src = REPO_ROOT / "runs" / profile / run / "eval" / "confusion.png"
        if src.is_file():
            Image.open(src).save(out / f"confusion_{run}.png")
            print("copied", f"confusion_{run}.png")

    # 8 per class IoU F1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    xpos = np.arange(len(ANOMALY))
    w = 0.25
    for i, run in enumerate(runs):
        m = load_metrics(run)
        if not m:
            continue
        iou = [m.get("per_class_iou_modified", {}).get(c, 0.0) for c in ANOMALY]
        f1 = [m.get("multilabel", {}).get("f1", {}).get(c, 0.0) for c in ANOMALY]
        axes[0].bar(xpos + i * w, iou, w, label=run, color=run_colors[run])
        axes[1].bar(xpos + i * w, f1, w, label=run, color=run_colors[run])
    for ax, title in zip(axes, ["Per class IoU (modified)", "Per class F1 (multilabel)"]):
        ax.set_xticks(xpos + w)
        ax.set_xticklabels(ANOMALY, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "per_class_iou_f1.png", dpi=140)
    plt.close(fig)
    print("wrote per_class_iou_f1.png")

    # 9 PR curves
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, run in zip(axes, runs):
        prp = REPO_ROOT / "runs" / profile / run / "eval" / "pr_curves.json"
        if not prp.is_file():
            ax.set_title(f"{run} missing")
            continue
        pr = json.loads(prp.read_text())
        for c in ANOMALY:
            curve = pr.get(c) or {}
            if curve.get("recall"):
                ax.plot(curve["recall"], curve["precision"], label=c, color=COLORS[c], linewidth=1)
        ax.set_title(f"{run} PR curves")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=140)
    plt.close(fig)
    print("wrote pr_curves.png")

    # 10 robustness
    fig, ax = plt.subplots(figsize=(6, 4))
    any_rob = False
    for run in runs:
        rp = REPO_ROOT / "runs" / profile / run / "eval" / "robustness.json"
        if not rp.is_file():
            continue
        pts = json.loads(rp.read_text())
        ax.plot(
            [p["severity"] for p in pts],
            [p["modified_miou"] for p in pts],
            marker="o",
            label=run,
            color=run_colors[run],
        )
        any_rob = True
    ax.set_xlabel("Severity")
    ax.set_ylabel("Modified mIoU")
    ax.set_title("Robustness")
    if any_rob:
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "robustness.png", dpi=140)
        print("wrote robustness.png")
    else:
        print("no robustness.json yet")
    plt.close(fig)

    # 11 efficiency table
    rows_eff = []
    for run in runs:
        ep = REPO_ROOT / "runs" / profile / run / "eval" / "efficiency.json"
        if ep.is_file():
            e = json.loads(ep.read_text())
            rows_eff.append(
                [
                    run,
                    e.get("parameters"),
                    round(e.get("size_mb", 0), 2),
                    round(e.get("mean_infer_sec_gpu", 0), 4),
                    round(e.get("mean_infer_sec_cpu", 0), 4),
                ]
            )
    if rows_eff:
        fig, ax = plt.subplots(figsize=(8, 2 + 0.4 * len(rows_eff)))
        ax.axis("off")
        table = ax.table(
            cellText=rows_eff,
            colLabels=["run", "params", "size_mb", "infer_s_gpu", "infer_s_cpu"],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        ax.set_title("Efficiency")
        fig.tight_layout()
        fig.savefig(out / "efficiency_table.png", dpi=140)
        plt.close(fig)
        print("wrote efficiency_table.png")

    # 12 sample predictions from gallery
    gdir = REPO_ROOT / "dashboard" / "public" / "data" / "gallery"
    index = gdir / "index.json"
    if index.is_file():
        tiles = (json.loads(index.read_text()).get("tiles") or [])[:6]
        if tiles:
            fig, axes = plt.subplots(len(tiles), 5, figsize=(12, 2.2 * len(tiles)))
            if len(tiles) == 1:
                axes = np.array([axes])
            for i, t in enumerate(tiles):
                tid = t.get("id") or t.get("tile_id")
                folder = gdir / tid

                def show(ax, path, title):
                    if Path(path).is_file():
                        ax.imshow(Image.open(path))
                    ax.set_title(title)
                    ax.axis("off")

                show(axes[i, 0], folder / "rgb.webp", "RGB")
                show(axes[i, 1], folder / "gt.webp", "GT")
                show(axes[i, 2], folder / "pred_run1.webp", "Run 1")
                show(axes[i, 3], folder / "pred_run2.webp", "Run 2")
                show(axes[i, 4], folder / "pred_run3.webp", "Run 3")
            fig.suptitle("Sample predictions")
            fig.tight_layout()
            fig.savefig(out / "sample_predictions.png", dpi=140)
            plt.close(fig)
            print("wrote sample_predictions.png")

    # 13 overlap examples
    overlap_tiles = []
    for r in rows:
        if r["our_split"] != "test":
            continue
        base = subset / r["source_split"]
        tid = r["tile_id"]
        valid = load_valid(base, tid)
        labs = []
        for c in ANOMALY:
            p = base / "labels" / c / f"{tid}.png"
            lab = (np.array(Image.open(p).convert("L")) > 0) if p.is_file() else np.zeros_like(valid)
            labs.append(lab)
        stack = np.stack(labs, 0)
        if ((stack.sum(0) >= 2) & valid).any():
            overlap_tiles.append((r, valid, stack))
        if len(overlap_tiles) >= 4:
            break
    if overlap_tiles:
        fig, axes = plt.subplots(len(overlap_tiles), 3, figsize=(9, 2.5 * len(overlap_tiles)))
        if len(overlap_tiles) == 1:
            axes = np.array([axes])
        for i, (r, valid, stack) in enumerate(overlap_tiles):
            base = subset / r["source_split"]
            tid = r["tile_id"]
            rgb = np.array(Image.open(base / "images" / "rgb" / f"{tid}.jpg").convert("RGB"))
            ov = (stack.sum(0) >= 2) & valid
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title("RGB")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(gt_overlay(base, tid, valid))
            axes[i, 1].set_title("All labels")
            axes[i, 1].axis("off")
            axes[i, 2].imshow(rgb)
            mask = np.zeros((*valid.shape, 4), dtype=np.uint8)
            mask[ov] = (255, 0, 0, 180)
            axes[i, 2].imshow(mask)
            axes[i, 2].set_title("Overlap pixels")
            axes[i, 2].axis("off")
        fig.suptitle("Pixels with two or more anomaly labels")
        fig.tight_layout()
        fig.savefig(out / "overlap_examples.png", dpi=140)
        plt.close(fig)
        print("wrote overlap_examples.png")
    else:
        print("no overlap tiles found on disk for this profile yet")

    print("Done. Figures in", out)


if __name__ == "__main__":
    main()
