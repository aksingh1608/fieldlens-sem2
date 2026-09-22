#!/usr/bin/env python3
"""Save a 6-tile preview grid: RGB, NIR, colored GT, valid mask."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fieldlens.constants import ANOMALY_CLASSES, CLASS_COLORS_RGB  # noqa: E402
from fieldlens.paths import REPO_ROOT as ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--out", default=str(ROOT / "runs" / "preview" / "preview_grid.png"))
    args = parser.parse_args()

    split_csv = ROOT / "data" / "fieldlens" / "split.csv"
    subset = ROOT / "data" / "fieldlens" / "tiles"
    if not split_csv.is_file():
        raise SystemExit(f"missing {split_csv}")

    rows = []
    with split_csv.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
            if len(rows) >= args.n:
                break

    cell_w, cell_h = 256, 256
    cols = 4
    grid = Image.new("RGB", (cols * cell_w, len(rows) * cell_h), (20, 20, 20))

    for i, r in enumerate(rows):
        tid, src = r["tile_id"], r["source_split"]
        base = subset / src
        rgb = Image.open(base / "images" / "rgb" / f"{tid}.jpg").convert("RGB").resize((cell_w, cell_h))
        nir_p = base / "images" / "nir" / f"{tid}.jpg"
        if not nir_p.is_file():
            nir_p = base / "images" / "nir" / f"{tid}.png"
        nir = Image.open(nir_p).convert("L").resize((cell_w, cell_h)).convert("RGB")
        boundary = np.array(Image.open(base / "boundaries" / f"{tid}.png").convert("L").resize((cell_w, cell_h), Image.NEAREST)) > 0
        mask = np.array(Image.open(base / "masks" / f"{tid}.png").convert("L").resize((cell_w, cell_h), Image.NEAREST)) > 0
        valid = boundary & mask
        gt = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        for c in ANOMALY_CLASSES:
            p = base / "labels" / c / f"{tid}.png"
            if not p.is_file():
                continue
            lab = np.array(Image.open(p).convert("L").resize((cell_w, cell_h), Image.NEAREST)) > 0
            color = np.array(CLASS_COLORS_RGB[c], dtype=np.uint8)
            gt[lab & valid] = color
        valid_img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        valid_img[valid] = (220, 220, 220)

        y0 = i * cell_h
        for j, im in enumerate([rgb, nir, Image.fromarray(gt), Image.fromarray(valid_img)]):
            grid.paste(im, (j * cell_w, y0))
            draw = ImageDraw.Draw(grid)
            draw.text((j * cell_w + 4, y0 + 4), ["RGB", "NIR", "GT", "valid"][j], fill=(255, 255, 0))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
