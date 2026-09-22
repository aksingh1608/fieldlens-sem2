"""Agriculture-Vision tile dataset for FieldLens."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fieldlens.constants import (
    ANOMALY_CLASSES,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_ANOMALY_CLASSES,
)


def field_id_from_tile_id(tile_id: str) -> str:
    if "_" not in tile_id:
        raise ValueError(f"tile_id missing underscore field prefix: {tile_id}")
    return tile_id.split("_", 1)[0]


def load_split_rows(split_csv: Path, our_split: str | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with split_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if our_split is None or row["our_split"] == our_split:
                rows.append(row)
    return rows


def _read_gray(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return arr


def _read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_nir(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return arr


class FieldLensDataset(Dataset):
    """
    Lazy tile loader. Never loads the full dataset into RAM.

    Returns dict:
      image: float CHW (3 for rgb, 4 for rgb_nir), normalized
      multilabel: float (8,H,W)
      singlelabel: long (H,W) with IGNORE_INDEX for invalid
      valid: bool (H,W)
      tile_id: str
    """

    def __init__(
        self,
        subset_root: str | Path,
        split_csv: str | Path,
        our_split: str,
        stats_json: str | Path | None = None,
        channels: str = "rgb",
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.subset_root = Path(subset_root)
        self.rows = load_split_rows(Path(split_csv), our_split)
        self.channels = channels
        self.augment = augment
        self.seed = int(seed)
        # Seeded lazily, per process: DataLoader workers are forked with a copy of
        # this object, so one eagerly-seeded generator would hand every worker the
        # same augmentation stream and repeat it every epoch.
        self._rng: np.random.Generator | None = None
        self._rng_pid: int | None = None

        self.nir_mean = 0.0
        self.nir_std = 1.0
        self.class_pixel_counts: list[int] | None = None
        if stats_json and Path(stats_json).is_file():
            stats = json.loads(Path(stats_json).read_text())
            nir = stats.get("nir_train", {})
            self.nir_mean = float(nir.get("mean", 0.0))
            self.nir_std = float(nir.get("std", 1.0)) or 1.0
            # Rarest class rule uses train pixel frequency over anomaly classes.
            train_counts = stats.get("per_class", {}).get("train", {})
            counts = []
            for c in ANOMALY_CLASSES:
                counts.append(int(train_counts.get(c, {}).get("valid_pixels", 0)))
            self.class_pixel_counts = counts

        if not self.rows:
            raise FileNotFoundError(
                f"No rows for our_split={our_split} in {split_csv}. "
                "Run download_subset.py first."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _worker_rng(self) -> np.random.Generator:
        """Augmentation RNG for the current process.

        In a worker, torch seeds each worker differently per epoch and derives
        those seeds from the global seed, so augmentations stay reproducible
        without repeating across workers or epochs. In the main process the
        generator persists and keeps advancing.
        """
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            info = torch.utils.data.get_worker_info()
            seed = int(torch.initial_seed()) % (2**32) if info is not None else self.seed
            self._rng = np.random.default_rng(seed)
            self._rng_pid = pid
        return self._rng

    def _tile_paths(self, tile_id: str, source_split: str) -> dict[str, Path]:
        # Local subset layout mirrors archive split folders under subset_root/source_split/
        base = self.subset_root / source_split
        return {
            "rgb": base / "images" / "rgb" / f"{tile_id}.jpg",
            "nir": base / "images" / "nir" / f"{tile_id}.jpg",
            "boundary": base / "boundaries" / f"{tile_id}.png",
            "mask": base / "masks" / f"{tile_id}.png",
            "labels": {
                c: base / "labels" / c / f"{tile_id}.png" for c in ANOMALY_CLASSES
            },
        }

    def _build_singlelabel(
        self, multilabel: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        """
        singlelabel rule:
        - invalid -> 255
        - no anomaly -> 0 background
        - one anomaly -> that class id (1..8)
        - two or more -> rarest by train pixel frequency from stats.json
        """
        h, w = valid.shape
        out = np.full((h, w), IGNORE_INDEX, dtype=np.int64)
        present = multilabel > 0.5  # (8,H,W)
        n = present.sum(axis=0)
        out[valid & (n == 0)] = 0
        single = valid & (n == 1)
        if single.any():
            cls_idx = present[:, single].argmax(axis=0)  # 0..7
            out[single] = cls_idx + 1
        multi = valid & (n >= 2)
        if multi.any():
            counts = self.class_pixel_counts
            if counts is None:
                # Fallback alphabetical rarest among present: lowest channel index
                # if stats missing (should not happen in real runs).
                order = list(range(NUM_ANOMALY_CLASSES))
            else:
                order = sorted(range(NUM_ANOMALY_CLASSES), key=lambda i: (counts[i], i))
            rank = np.empty(NUM_ANOMALY_CLASSES, dtype=np.int64)
            for position, class_idx in enumerate(order):
                rank[class_idx] = position
            # Lowest rank among present classes wins; absent classes score worse
            # than any real rank.
            scored = np.where(present, rank[:, None, None], NUM_ANOMALY_CLASSES)
            chosen = scored.argmin(axis=0)
            out[multi] = chosen[multi] + 1
        return out

    def _augment(
        self,
        image: np.ndarray,
        multilabel: np.ndarray,
        valid: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # image HWC, multilabel CHW, valid HW
        if rng.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            multilabel = np.flip(multilabel, axis=2).copy()
            valid = np.flip(valid, axis=1).copy()
        if rng.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            multilabel = np.flip(multilabel, axis=1).copy()
            valid = np.flip(valid, axis=0).copy()
        k = int(rng.integers(0, 4))
        if k:
            image = np.rot90(image, k, axes=(0, 1)).copy()
            multilabel = np.rot90(multilabel, k, axes=(1, 2)).copy()
            valid = np.rot90(valid, k, axes=(0, 1)).copy()
        return image, multilabel, valid

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        tile_id = row["tile_id"]
        source_split = row["source_split"]
        paths = self._tile_paths(tile_id, source_split)

        rgb = _read_rgb(paths["rgb"])
        nir = _read_nir(paths["nir"])
        boundary = _read_gray(paths["boundary"]) > 0
        mask = _read_gray(paths["mask"]) > 0
        valid = boundary & mask

        multilabel = np.zeros((NUM_ANOMALY_CLASSES, rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        for i, c in enumerate(ANOMALY_CLASSES):
            lab_path = paths["labels"][c]
            if lab_path.is_file():
                multilabel[i] = (_read_gray(lab_path) > 0).astype(np.float32)

        if self.channels == "rgb_nir":
            image = np.concatenate([rgb.astype(np.float32), nir[..., None].astype(np.float32)], axis=2)
        else:
            image = rgb.astype(np.float32)

        if self.augment:
            image, multilabel, valid = self._augment(
                image, multilabel, valid, self._worker_rng()
            )

        # Normalize to [0,1] then standardize.
        image = image / 255.0
        if self.channels == "rgb_nir":
            rgb_f = image[..., :3]
            nir_f = image[..., 3]
            for c in range(3):
                rgb_f[..., c] = (rgb_f[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            nir_f = (nir_f - self.nir_mean) / self.nir_std
            image = np.concatenate([rgb_f, nir_f[..., None]], axis=2)
        else:
            for c in range(3):
                image[..., c] = (image[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        singlelabel = self._build_singlelabel(multilabel, valid)

        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        return {
            "image": image_t,
            "multilabel": torch.from_numpy(np.ascontiguousarray(multilabel)).float(),
            "singlelabel": torch.from_numpy(singlelabel).long(),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)).bool(),
            "tile_id": tile_id,
        }


def collate_fieldlens(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "multilabel": torch.stack([b["multilabel"] for b in batch], dim=0),
        "singlelabel": torch.stack([b["singlelabel"] for b in batch], dim=0),
        "valid": torch.stack([b["valid"] for b in batch], dim=0),
        "tile_id": [b["tile_id"] for b in batch],
    }
