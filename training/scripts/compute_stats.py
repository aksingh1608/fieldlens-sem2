#!/usr/bin/env python3
"""Compute FieldLens subset stats -> data/fieldlens/stats.json."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fieldlens.constants import ANOMALY_CLASSES  # noqa: E402
from fieldlens.paths import CONFIG_DIR, REPO_ROOT as ROOT, repo_path  # noqa: E402
from fieldlens.profile import add_profile_arg, load_profile, merge_data_with_profile  # noqa: E402


def load_rows(split_csv: Path) -> list[dict[str, str]]:
    with split_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def read_bin(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DIR / "data.yaml"))
    add_profile_arg(parser)
    parser.add_argument("--limit", type=int, default=0, help="Optional tile cap for smoke tests")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    cfg = merge_data_with_profile(profile)
    print(f"profile={args.profile}")
    subset_root = repo_path(cfg["subset_root"])
    split_csv = repo_path(cfg["split_csv"])
    out_path = repo_path(cfg["stats_json"])

    rows = load_rows(split_csv)
    if args.limit:
        rows = rows[: args.limit]

    tiles_per_split: dict[str, int] = defaultdict(int)
    fields_per_split: dict[str, set[str]] = defaultdict(set)
    per_class: dict[str, dict[str, dict[str, int]]] = {
        s: {c: {"valid_pixels": 0, "tile_count": 0} for c in ANOMALY_CLASSES}
        for s in ("train", "val", "test")
    }
    overlap_pixels = {"train": 0, "val": 0, "test": 0}
    nir_sum = 0.0
    nir_sq = 0.0
    nir_n = 0

    for i, row in enumerate(rows, 1):
        our = row["our_split"]
        src = row["source_split"]
        tid = row["tile_id"]
        fid = row["field_id"]
        tiles_per_split[our] += 1
        fields_per_split[our].add(fid)
        base = subset_root / src
        valid = read_bin(base / "boundaries" / f"{tid}.png") & read_bin(
            base / "masks" / f"{tid}.png"
        )
        labels = []
        for c in ANOMALY_CLASSES:
            p = base / "labels" / c / f"{tid}.png"
            lab = read_bin(p) if p.is_file() else np.zeros_like(valid)
            labels.append(lab)
            vp = int((lab & valid).sum())
            per_class[our][c]["valid_pixels"] += vp
            if vp > 0:
                per_class[our][c]["tile_count"] += 1
        stack = np.stack(labels, axis=0)
        overlap_pixels[our] += int(((stack.sum(axis=0) >= 2) & valid).sum())

        if our == "train":
            nir_path = base / "images" / "nir" / f"{tid}.jpg"
            if not nir_path.is_file():
                nir_path = base / "images" / "nir" / f"{tid}.png"
            nir = np.array(Image.open(nir_path).convert("L"), dtype=np.float64) / 255.0
            nir_v = nir[valid]
            nir_sum += float(nir_v.sum())
            nir_sq += float((nir_v**2).sum())
            nir_n += int(nir_v.size)

        if i % 50 == 0:
            print(f"processed {i}/{len(rows)}")

    nir_mean = nir_sum / nir_n if nir_n else 0.0
    nir_var = (nir_sq / nir_n - nir_mean**2) if nir_n else 0.0
    nir_std = float(np.sqrt(max(nir_var, 0.0)))

    stats = {
        "tiles_per_split": dict(tiles_per_split),
        "fields_per_split": {k: len(v) for k, v in fields_per_split.items()},
        "per_class": per_class,
        "overlap_valid_pixels": overlap_pixels,
        "nir_train": {"mean": nir_mean, "std": nir_std, "n_pixels": nir_n},
        "pilot_disclaimer": (
            "FieldLens is a research pilot. Results are trends from small runs "
            "on a data subset, not benchmark numbers."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")
    print(f"tiles_per_split={dict(tiles_per_split)}")
    print(
        "fields_per_split=",
        {k: len(v) for k, v in fields_per_split.items()},
    )
    print(f"overlap_valid_pixels={overlap_pixels}")
    for sp in ("train", "val", "test"):
        print(f"per_class tile_count [{sp}]:")
        for c in ANOMALY_CLASSES:
            tc = per_class[sp][c]["tile_count"]
            print(f"  {c}: tiles={tc} pixels={per_class[sp][c]['valid_pixels']}")
            if tc < 10:
                print(
                    f"  WARNING: class {c} has only {tc} tiles in {sp} "
                    f"(fewer than 10)"
                )


if __name__ == "__main__":
    main()
