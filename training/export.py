#!/usr/bin/env python3
"""Export metrics and (optionally) gallery tiles for the dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fieldlens.constants import (  # noqa: E402
    ANOMALY_CLASSES,
    CLASS_COLORS_RGB,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from fieldlens.metrics import logits_to_multilabel, logits_to_pred_class  # noqa: E402
from fieldlens.models import build_model  # noqa: E402
from fieldlens.paths import CONFIG_DIR, REPO_ROOT, repo_path  # noqa: E402
from fieldlens.profile import (  # noqa: E402
    add_profile_arg,
    apply_profile_to_run_cfg,
    classes_export_payload,
    load_profile,
    merge_data_with_profile,
    profile_ckpt_dir,
    profile_run_dir,
)


def load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def load_log_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def pick_gallery(split_csv: Path, subset_root: Path, n: int = 60, seed: int = 42) -> list[dict]:
    """Choose ~60 test tiles covering every class >=3 times, rare classes, >=10 overlaps."""
    rows = []
    with split_csv.open() as f:
        for r in csv.DictReader(f):
            if r["our_split"] == "test":
                rows.append(r)
    rng = np.random.default_rng(seed)

    def paths(tid: str, src: str):
        base = subset_root / src
        return base, tid

    info = []
    for r in rows:
        base, tid = paths(r["tile_id"], r["source_split"])
        valid = np.array(Image.open(base / "boundaries" / f"{tid}.png").convert("L")) > 0
        valid &= np.array(Image.open(base / "masks" / f"{tid}.png").convert("L")) > 0
        present = []
        labs = []
        for c in ANOMALY_CLASSES:
            p = base / "labels" / c / f"{tid}.png"
            lab = (np.array(Image.open(p).convert("L")) > 0) if p.is_file() else np.zeros_like(valid)
            labs.append(lab)
            if (lab & valid).any():
                present.append(c)
        stack = np.stack(labs, 0) if labs else np.zeros((0,) + valid.shape)
        overlap = int(((stack.sum(0) >= 2) & valid).sum()) if len(labs) else 0
        info.append(
            {
                "tile_id": tid,
                "source_split": r["source_split"],
                "classes": present,
                "overlap": overlap,
            }
        )

    selected: list[dict] = []
    selected_ids: set[str] = set()
    class_counts = defaultdict(int)

    overlap_tiles = [t for t in info if t["overlap"] > 0]
    rng.shuffle(overlap_tiles)
    for t in overlap_tiles:
        if sum(1 for s in selected if s["overlap"] > 0) >= 10:
            break
        selected.append(t)
        selected_ids.add(t["tile_id"])
        for c in t["classes"]:
            class_counts[c] += 1

    for c in ANOMALY_CLASSES:
        while class_counts[c] < 3:
            candidates = [
                t for t in info if c in t["classes"] and t["tile_id"] not in selected_ids
            ]
            if not candidates:
                break
            candidates.sort(key=lambda t: (len(t["classes"]), -t["overlap"]))
            t = candidates[0]
            selected.append(t)
            selected_ids.add(t["tile_id"])
            for cc in t["classes"]:
                class_counts[cc] += 1

    remaining = [t for t in info if t["tile_id"] not in selected_ids]
    rng.shuffle(remaining)
    for t in remaining:
        if len(selected) >= n:
            break
        selected.append(t)
        selected_ids.add(t["tile_id"])
    return selected[:n]


def colorize_multilabel(masks: np.ndarray, valid: np.ndarray) -> Image.Image:
    """masks: (C,H,W) bool. Later classes overwrite earlier where they overlap."""
    h, w = valid.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for i, c in enumerate(ANOMALY_CLASSES):
        color = CLASS_COLORS_RGB[c]
        m = masks[i] & valid
        out[m] = (*color, 200)
    return Image.fromarray(out, "RGBA")


def load_tile_tensors(
    subset_root: Path,
    source_split: str,
    tid: str,
    channels: str,
    nir_mean: float,
    nir_std: float,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    base = subset_root / source_split
    rgb = np.array(Image.open(base / "images" / "rgb" / f"{tid}.jpg").convert("RGB"), dtype=np.float32) / 255.0
    valid = np.array(Image.open(base / "boundaries" / f"{tid}.png").convert("L")) > 0
    valid &= np.array(Image.open(base / "masks" / f"{tid}.png").convert("L")) > 0
    gt = []
    for c in ANOMALY_CLASSES:
        p = base / "labels" / c / f"{tid}.png"
        lab = (np.array(Image.open(p).convert("L")) > 0) if p.is_file() else np.zeros(valid.shape, dtype=bool)
        gt.append(lab)
    gt_arr = np.stack(gt, 0)

    x = torch.from_numpy(rgb).permute(2, 0, 1).clone()
    for c in range(3):
        x[c] = (x[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    if channels == "rgb_nir":
        nir_path = base / "images" / "nir" / f"{tid}.jpg"
        if not nir_path.is_file():
            nir_path = base / "images" / "nir" / f"{tid}.png"
        nir = np.array(Image.open(nir_path).convert("L"), dtype=np.float32) / 255.0
        nir_t = torch.from_numpy((nir - nir_mean) / (nir_std or 1.0)).unsqueeze(0)
        x = torch.cat([x, nir_t], dim=0)
    return x, valid, gt_arr


def multilabel_iou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """Mean per class IoU over anomaly channels on valid pixels."""
    ious = []
    v = valid.astype(bool)
    for i in range(len(ANOMALY_CLASSES)):
        p = pred[i][v] > 0.5
        g = gt[i][v] > 0.5
        inter = float(np.count_nonzero(p & g))
        union = float(np.count_nonzero(p | g))
        ious.append(inter / union if union > 0 else 1.0 if inter == 0 else 0.0)
    return float(np.mean(ious)) if ious else 0.0


def export_tile_images(
    tile: dict,
    subset_root: Path,
    out_dir: Path,
) -> dict:
    tid = tile["tile_id"]
    src = tile["source_split"]
    base = subset_root / src
    dest = out_dir / tid
    dest.mkdir(parents=True, exist_ok=True)

    rgb = Image.open(base / "images" / "rgb" / f"{tid}.jpg").convert("RGB")
    nir_path = base / "images" / "nir" / f"{tid}.jpg"
    if not nir_path.is_file():
        nir_path = base / "images" / "nir" / f"{tid}.png"
    nir = Image.open(nir_path).convert("L")
    rgb.save(dest / "rgb.webp", "WEBP", quality=85)
    nir.convert("RGB").save(dest / "nir.webp", "WEBP", quality=85)

    valid = np.array(Image.open(base / "boundaries" / f"{tid}.png").convert("L")) > 0
    valid &= np.array(Image.open(base / "masks" / f"{tid}.png").convert("L")) > 0
    vcount = max(int(valid.sum()), 1)
    coverage = {}
    alerts = []
    combined = np.zeros((*valid.shape, 4), dtype=np.uint8)
    gt_masks = []
    for c in ANOMALY_CLASSES:
        p = base / "labels" / c / f"{tid}.png"
        lab = (np.array(Image.open(p).convert("L")) > 0) if p.is_file() else np.zeros(valid.shape, dtype=bool)
        gt_masks.append(lab)
        frac = float((lab & valid).sum()) / vcount
        coverage[c] = frac
        mask_img = np.zeros((*valid.shape, 4), dtype=np.uint8)
        color = CLASS_COLORS_RGB[c]
        mask_img[lab & valid] = (*color, 200)
        Image.fromarray(mask_img, "RGBA").save(dest / f"gt_{c}.png")
        combined[lab & valid] = (*color, 200)
        if frac >= 0.01:
            alerts.append({"class": c, "coverage_pct": round(100 * frac, 2)})

    Image.fromarray(combined, "RGBA").save(dest / "gt.webp", "WEBP", quality=90)
    (dest / "tile.json").write_text(
        json.dumps({"tile_id": tid, "coverage_gt": coverage, "alerts_gt": alerts}, indent=2)
    )

    rel = f"gallery/{tid}"
    top_alert = None
    if alerts:
        best = max(alerts, key=lambda a: a["coverage_pct"])
        top_alert = {
            "class_id": best["class"],
            "class_label": best["class"].replace("_", " "),
            "coverage_percent": best["coverage_pct"],
        }
    return {
        "id": tid,
        "label": tid,
        "path": rel,
        "source_split": src,
        "rgb_url": f"/data/{rel}/rgb.webp",
        "nir_url": f"/data/{rel}/nir.webp",
        "gt_url": f"/data/{rel}/gt.webp",
        "predictions": {"run1": None, "run2": None, "run3": None},
        "alert": top_alert,
    }


@torch.no_grad()
def write_run_predictions(
    tiles_meta: list[dict],
    subset_root: Path,
    gdir: Path,
    profile_name: str,
    runs: list[str],
    device: torch.device,
) -> list[dict]:
    """Write pred_<run>.webp for each gallery tile and fill tile_compare IoUs."""
    stats = load_json(repo_path(f"data/fieldlens/{profile_name}/stats.json")) or {}
    nir_mean = float(stats.get("nir_train", {}).get("mean", 0.0))
    nir_std = float(stats.get("nir_train", {}).get("std", 1.0)) or 1.0

    models = {}
    cfgs = {}
    for run in runs:
        cfg_path = CONFIG_DIR / f"{run}.yaml"
        profile = load_profile(profile_name)
        cfg = apply_profile_to_run_cfg(load_yaml(cfg_path), profile)
        ckpt_path = profile_ckpt_dir(profile_name, run) / "best.pt"
        if not ckpt_path.is_file():
            print(f"skip preds for {run}: missing {ckpt_path}")
            continue
        model = build_model(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        models[run] = model
        cfgs[run] = cfg

    tile_compare = []
    for meta in tiles_meta:
        tid = meta["id"]
        src = meta["source_split"]
        dest = gdir / tid
        compare_row = {"tile_id": tid}
        preds_urls = {}
        for run in runs:
            if run not in models:
                preds_urls[run] = None
                compare_row[f"iou_{run}"] = None
                continue
            cfg = cfgs[run]
            channels = cfg["data"]["channels"]
            head = cfg["model"]["head"]
            threshold = float(cfg.get("threshold", 0.5))
            x, valid, gt = load_tile_tensors(
                subset_root, src, tid, channels, nir_mean, nir_std
            )
            logits = models[run](x.unsqueeze(0).to(device))[0].float().cpu().numpy()
            pred_m = logits_to_multilabel(logits, head, threshold)
            overlay = colorize_multilabel(pred_m > 0.5, valid)
            out_name = f"pred_{run}.webp"
            overlay.save(dest / out_name, "WEBP", quality=90)
            preds_urls[run] = f"/data/gallery/{tid}/{out_name}"
            compare_row[f"iou_{run}"] = multilabel_iou(pred_m, gt.astype(np.float32), valid)
        meta["predictions"] = {
            "run1": preds_urls.get("run1"),
            "run2": preds_urls.get("run2"),
            "run3": preds_urls.get("run3"),
        }
        tile_compare.append(compare_row)
    return tile_compare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_arg(parser)
    parser.add_argument(
        "--allow-images",
        action="store_true",
        help="Required to write dataset tile images. Read terms first; AK must approve.",
    )
    parser.add_argument("--runs", nargs="+", default=["run1", "run2", "run3"])
    args = parser.parse_args()
    profile = load_profile(args.profile)
    data_cfg = merge_data_with_profile(profile)
    print(f"profile={args.profile}")

    print("=" * 72)
    print("AGRICULTURE-VISION TERMS (plain words)")
    print("=" * 72)
    print(
        "From the Agriculture-Vision Workshop Dataset Terms "
        "(https://www.agriculture-vision.com/dataset-terms):\n"
        "- Non-commercial academic research only.\n"
        "- Do not sell, license, transfer, or redistribute the dataset to third parties "
        "(except research colleagues who also accept the terms).\n"
        "- Do not use it for commercial products or services.\n"
        "- Do not create derivative works from the dataset other than a challenge Submission.\n"
        "- Cite the Agriculture-Vision paper in publications.\n"
        "- Challenge-era terms also said to delete copies after the challenge period.\n\n"
        "Showing tiles on a public website is redistribution to the public. "
        "That conflicts with the no-redistribute rule unless IntelinAir gives permission.\n"
        "Export will NOT write gallery images unless you pass --allow-images after you say yes."
    )
    print("=" * 72)

    out_root = REPO_ROOT / "dashboard" / "public" / "data"
    out_root.mkdir(parents=True, exist_ok=True)

    classes_payload = classes_export_payload()
    (out_root / "classes.json").write_text(json.dumps(classes_payload, indent=2))
    print(f"Wrote {out_root / 'classes.json'}")

    site = {
        "enable_try_page": False,
        "profile": args.profile,
        "pilot_disclaimer": (
            "FieldLens is a research pilot. Results are trends from small runs "
            "on a data subset, not benchmark numbers."
        ),
    }
    (out_root / "site.json").write_text(json.dumps(site, indent=2))
    print(f"Wrote {out_root / 'site.json'}")

    stats = load_json(repo_path(data_cfg["stats_json"]))
    tiles = (stats or {}).get("tiles_per_split") or {}
    results = {
        "pilot_disclaimer": site["pilot_disclaimer"],
        "profile": args.profile,
        "status": "pending_runs" if not stats else "partial",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Agriculture-Vision 2021 supervised (FieldLens subset)",
            "subset_tiles": sum(int(v) for v in tiles.values()) if tiles else None,
            "train_tiles": tiles.get("train"),
            "val_tiles": tiles.get("val"),
            "test_tiles": tiles.get("test"),
        },
        "dataset_stats": stats,
        "classes": classes_payload,
        "runs": {},
        "tile_compare": None,
    }

    for run in args.runs:
        eval_dir = profile_run_dir(args.profile, run) / "eval"
        metrics = load_json(eval_dir / "metrics.json")
        rob = load_json(eval_dir / "robustness.json")
        eff = load_json(eval_dir / "efficiency.json")
        pr = load_json(eval_dir / "pr_curves.json")
        log = load_log_csv(profile_run_dir(args.profile, run) / "log.csv")
        if metrics is None and not log:
            results["runs"][run] = None
            continue
        results["runs"][run] = {
            "metrics": metrics,
            "robustness": rob,
            "efficiency": eff,
            "pr_curves": pr,
            "training_log": log,
        }

    gallery_index = {"tiles": [], "note": "images not exported yet"}
    if args.allow_images:
        subset_root = repo_path(data_cfg["subset_root"])
        split_csv = repo_path(data_cfg["split_csv"])
        selected = pick_gallery(split_csv, subset_root)
        gdir = out_root / "gallery"
        gdir.mkdir(parents=True, exist_ok=True)
        tiles_meta = []
        for t in selected:
            meta = export_tile_images(t, subset_root, gdir)
            tiles_meta.append(meta)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tile_compare = write_run_predictions(
            tiles_meta, subset_root, gdir, args.profile, list(args.runs), device
        )
        results["tile_compare"] = tile_compare
        gallery_index = {"tiles": tiles_meta, "note": "exported with --allow-images and run overlays"}
        print(f"Exported {len(tiles_meta)} gallery tiles with prediction overlays")
    else:
        print("Skipped gallery images (pass --allow-images only after AK approval).")

    if all(results["runs"].get(r) and results["runs"][r].get("metrics") for r in args.runs):
        results["status"] = "complete" if results.get("tile_compare") else "partial"

    (out_root / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_root / 'results.json'}")

    (out_root / "gallery" / "index.json").parent.mkdir(parents=True, exist_ok=True)
    (out_root / "gallery" / "index.json").write_text(json.dumps(gallery_index, indent=2))


if __name__ == "__main__":
    main()
