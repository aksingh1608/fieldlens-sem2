#!/usr/bin/env python3
"""Evaluate a FieldLens checkpoint on our test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fieldlens.constants import ANOMALY_CLASSES, CLASS_NAMES, NUM_ANOMALY_CLASSES, NUM_CLASSES  # noqa: E402
from fieldlens.dataset import FieldLensDataset, collate_fieldlens  # noqa: E402
from fieldlens.metrics import (  # noqa: E402
    binary_counts,
    iou_from_confusion,
    logits_to_multilabel,
    logits_to_pred_class,
    metrics_from_counts,
    multilabel_binary_metrics,
    overlap_mask,
    tile_positive,
    update_agri_confusion,
)
from fieldlens.models import build_model, count_parameters  # noqa: E402
from fieldlens.paths import CONFIG_DIR, REPO_ROOT, repo_path  # noqa: E402
from fieldlens.profile import (  # noqa: E402
    add_profile_arg,
    apply_profile_to_run_cfg,
    load_profile,
    profile_run_dir,
)
from fieldlens.transforms_drone import apply_drone_condition  # noqa: E402


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def denorm_for_corruption(image: torch.Tensor, channels: str, nir_mean: float, nir_std: float) -> torch.Tensor:
    """Approx undo ImageNet/NIR norm to [0,1] for applying drone corruptions, then re-norm."""
    from fieldlens.constants import IMAGENET_MEAN, IMAGENET_STD

    x = image.clone()
    for c in range(3):
        x[c] = x[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    if channels == "rgb_nir" and x.shape[0] == 4:
        x[3] = x[3] * nir_std + nir_mean
    return x.clamp(0, 1)


def renorm(image01: torch.Tensor, channels: str, nir_mean: float, nir_std: float) -> torch.Tensor:
    from fieldlens.constants import IMAGENET_MEAN, IMAGENET_STD

    x = image01.clone()
    for c in range(3):
        x[c] = (x[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    if channels == "rgb_nir" and x.shape[0] == 4:
        x[3] = (x[3] - nir_mean) / nir_std
    return x


@torch.no_grad()
def run_eval(
    model,
    loader,
    device,
    head: str,
    threshold: float,
    tile_frac: float,
    tile_score_mode: str,
    severity: int,
    eval_cfg: dict,
    channels: str,
    nir_mean: float,
    nir_std: float,
) -> dict:
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    # Streaming accumulators: holding every tile's prediction would cost ~8 MB
    # per 512x512 tile and blows up on a full test split.
    pixel_counts = np.zeros((NUM_ANOMALY_CLASSES, 3), dtype=np.float64)
    overlap_sums = {k: np.zeros(NUM_ANOMALY_CLASSES) for k in ("iou", "precision", "recall", "f1")}
    overlap_tiles = 0
    tile_rows = []

    gt_tile = {c: [] for c in ANOMALY_CLASSES}
    score_tile = {c: [] for c in ANOMALY_CLASSES}
    pred_tile = {c: [] for c in ANOMALY_CLASSES}

    for batch in loader:
        images = batch["image"].to(device)
        if severity > 0:
            corrupted = []
            for b in range(images.size(0)):
                x01 = denorm_for_corruption(images[b].cpu(), channels, nir_mean, nir_std)
                x01 = apply_drone_condition(x01, severity, eval_cfg)
                corrupted.append(renorm(x01, channels, nir_mean, nir_std))
            images = torch.stack(corrupted, dim=0).to(device)

        logits = model(images)
        for b in range(images.size(0)):
            logit = logits[b].float().cpu().numpy()
            gt = batch["multilabel"][b].numpy()
            valid = batch["valid"][b].numpy()
            pred = logits_to_pred_class(logit, head, threshold)
            pred_m = logits_to_multilabel(logit, head, threshold)
            update_agri_confusion(conf, pred, gt, valid)
            pixel_counts += binary_counts(pred_m, gt, valid)

            # overlap-only pixels (two or more ground-truth classes)
            ov = overlap_mask(gt, valid)
            if ov.any():
                m = multilabel_binary_metrics(pred_m, gt, ov)
                for k, acc in overlap_sums.items():
                    acc += np.asarray(m[k])
                overlap_tiles += 1

            # tile metrics
            if head == "softmax":
                probs = None
            else:
                probs = 1.0 / (1.0 + np.exp(-logit))

            row = {"tile_id": batch["tile_id"][b]}
            vcount = max(int(valid.sum()), 1)
            for i, c in enumerate(ANOMALY_CLASSES):
                gt_pos = tile_positive(gt[i] > 0.5, valid, tile_frac)
                pred_pos = tile_positive(pred_m[i] > 0.5, valid, tile_frac)
                covered = float((pred_m[i][valid] > 0.5).sum()) / vcount
                if tile_score_mode == "mean_probability" and probs is not None:
                    score = float(probs[i][valid].mean()) if valid.any() else 0.0
                else:
                    score = covered
                gt_tile[c].append(1 if gt_pos else 0)
                score_tile[c].append(score)
                pred_tile[c].append(1 if pred_pos else 0)
                row[f"gt_{c}"] = int(gt_pos)
                row[f"pred_{c}"] = int(pred_pos)
                row[f"score_{c}"] = score
            tile_rows.append(row)

    iou_c, miou = iou_from_confusion(conf)
    total = float(conf.sum())
    pixel_accuracy = float(np.diag(conf).sum() / total) if total > 0 else 0.0
    ml = metrics_from_counts(pixel_counts)
    overlap_mean = {
        k: list(v / overlap_tiles) if overlap_tiles else [0.0] * NUM_ANOMALY_CLASSES
        for k, v in overlap_sums.items()
    }

    # tile-level PR
    tile_metrics = {}
    pr_curves = {}
    for c in ANOMALY_CLASSES:
        y = np.array(gt_tile[c])
        s = np.array(score_tile[c])
        if y.sum() == 0:
            tile_metrics[c] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0}
            pr_curves[c] = {"precision": [], "recall": [], "thresholds": []}
            continue
        # Binary decisions come from the tile rule; sklearn PR uses the raw score.
        pred_pos = np.array(pred_tile[c])
        tp = float(((pred_pos == 1) & (y == 1)).sum())
        fp = float(((pred_pos == 1) & (y == 0)).sum())
        fn = float(((pred_pos == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        ap = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else 0.0
        p_curve, r_curve, thr = precision_recall_curve(y, s)
        tile_metrics[c] = {"precision": prec, "recall": rec, "f1": f1, "ap": ap}
        pr_curves[c] = {
            "precision": p_curve.tolist(),
            "recall": r_curve.tolist(),
            "thresholds": thr.tolist(),
        }

    return {
        "modified_miou": miou,
        "pixel_accuracy": pixel_accuracy,
        "per_class_iou_modified": {CLASS_NAMES[i]: float(iou_c[i]) for i in range(NUM_CLASSES)},
        "confusion": conf.tolist(),
        "multilabel": {k: {ANOMALY_CLASSES[i]: float(v[i]) for i in range(NUM_ANOMALY_CLASSES)} for k, v in ml.items()},
        "overlap_multilabel": {
            k: {ANOMALY_CLASSES[i]: float(v[i]) for i in range(NUM_ANOMALY_CLASSES)}
            for k, v in overlap_mean.items()
        },
        "tile_metrics": tile_metrics,
        "pr_curves": pr_curves,
        "tile_rows": tile_rows,
        "tile_score_mode": tile_score_mode,
    }


def plot_confusion(conf: list[list[int]], out: Path) -> None:
    arr = np.array(conf, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=90)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("second index (see docs/evaluation.md)")
    ax.set_ylabel("first index")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def timed_inference(model, loader, device, n: int, warmup: int) -> float:
    model.eval()
    times = []
    count = 0
    with torch.no_grad():
        for batch in loader:
            for b in range(batch["image"].size(0)):
                x = batch["image"][b : b + 1].to(device)
                if count < warmup:
                    _ = model(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    count += 1
                    continue
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                count += 1
                if len(times) >= n:
                    return float(np.mean(times))
    return float(np.mean(times)) if times else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="run config yaml")
    add_profile_arg(parser)
    parser.add_argument("--eval-config", default=str(CONFIG_DIR / "eval.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--skip-efficiency", action="store_true")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    cfg = apply_profile_to_run_cfg(load_yaml(Path(args.config)), profile)
    print(f"profile={args.profile}")
    eval_cfg = load_yaml(Path(args.eval_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = cfg["data"]
    stats_path = repo_path(data["stats_json"])
    ds = FieldLensDataset(
        subset_root=repo_path(data["subset_root"]),
        split_csv=repo_path(data["split_csv"]),
        our_split="test",
        stats_json=stats_path,
        channels=data["channels"],
        augment=False,
    )
    loader = DataLoader(
        ds,
        batch_size=data["batch_size"],
        shuffle=False,
        num_workers=data["num_workers"],
        pin_memory=data["pin_memory"],
        collate_fn=collate_fieldlens,
    )

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    nir_mean = float(stats.get("nir_train", {}).get("mean", 0.0))
    nir_std = float(stats.get("nir_train", {}).get("std", 1.0)) or 1.0

    threshold = float(cfg.get("threshold", eval_cfg.get("threshold", 0.5)))
    out_dir = profile_run_dir(args.profile, cfg["run_name"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_eval(
        model,
        loader,
        device,
        head=cfg["model"]["head"],
        threshold=threshold,
        tile_frac=float(eval_cfg["tile_positive_fraction"]),
        tile_score_mode=str(eval_cfg.get("tile_score", "covered_fraction")),
        severity=0,
        eval_cfg=eval_cfg,
        channels=data["channels"],
        nir_mean=nir_mean,
        nir_std=nir_std,
    )
    plot_confusion(result["confusion"], out_dir / "confusion.png")
    with (out_dir / "metrics.json").open("w") as f:
        json.dump({k: v for k, v in result.items() if k not in ("tile_rows", "pr_curves")}, f, indent=2)
    with (out_dir / "pr_curves.json").open("w") as f:
        json.dump(result["pr_curves"], f)
    with (out_dir / "tile_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result["tile_rows"][0].keys()) if result["tile_rows"] else ["tile_id"])
        w.writeheader()
        w.writerows(result["tile_rows"])

    if args.robustness:
        rob_points = []
        for sev in eval_cfg["robustness"]["severities"]:
            r = run_eval(
                model,
                loader,
                device,
                head=cfg["model"]["head"],
                threshold=threshold,
                tile_frac=float(eval_cfg["tile_positive_fraction"]),
                tile_score_mode=str(eval_cfg.get("tile_score", "covered_fraction")),
                severity=int(sev),
                eval_cfg=eval_cfg,
                channels=data["channels"],
                nir_mean=nir_mean,
                nir_std=nir_std,
            )
            rob_points.append({"severity": int(sev), "modified_miou": r["modified_miou"]})
            print(f"severity={sev} miou={r['modified_miou']:.4f}")
        (out_dir / "robustness.json").write_text(json.dumps(rob_points, indent=2))
        fig, ax = plt.subplots()
        ax.plot([p["severity"] for p in rob_points], [p["modified_miou"] for p in rob_points], marker="o")
        ax.set_xlabel("severity")
        ax.set_ylabel("modified mIoU")
        ax.set_title("Robustness (pilot)")
        fig.savefig(out_dir / "robustness.png", dpi=120)
        plt.close(fig)

    if not args.skip_efficiency:
        info = count_parameters(model)
        gpu_ms = timed_inference(model, loader, device, eval_cfg["efficiency"]["timed_tiles"], eval_cfg["efficiency"]["warmup"])
        cpu_model = build_model(cfg)
        cpu_model.load_state_dict(ckpt["model"])
        cpu_model.to("cpu")
        cpu_ms = timed_inference(cpu_model, loader, torch.device("cpu"), eval_cfg["efficiency"]["timed_tiles"], eval_cfg["efficiency"]["warmup"])
        peak = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        eff = {
            "parameters": info.parameters,
            "size_mb": info.size_mb,
            "mean_infer_sec_gpu": gpu_ms,
            "mean_infer_sec_cpu": cpu_ms,
            "peak_vram_mb": peak,
        }
        (out_dir / "efficiency.json").write_text(json.dumps(eff, indent=2))
        print(eff)

    print(f"Wrote {out_dir}")
    print(f"modified_miou={result['modified_miou']:.4f}")
    print(f"pixel_accuracy={result['pixel_accuracy']:.4f}")


if __name__ == "__main__":
    main()
