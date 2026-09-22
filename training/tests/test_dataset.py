"""Dataset helper tests that do not need real tiles on disk."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fieldlens.constants import IGNORE_INDEX  # noqa: E402
from fieldlens.dataset import FieldLensDataset  # noqa: E402


def test_singlelabel_rules():
    ds = FieldLensDataset.__new__(FieldLensDataset)
    ds.class_pixel_counts = [100, 50, 10, 200, 5, 80, 90, 1]  # class index 7 rarest, then 4

    multilabel = np.zeros((8, 4, 4), dtype=np.float32)
    valid = np.zeros((4, 4), dtype=bool)
    valid[0:3, 0:3] = True

    # background
    # single class 0 at (0,0)
    multilabel[0, 0, 0] = 1
    # overlap class 4 and 7 at (1,1) -> rarest is 7 (count 1)
    multilabel[4, 1, 1] = 1
    multilabel[7, 1, 1] = 1
    # invalid stays 255 at (3,3)

    out = FieldLensDataset._build_singlelabel(ds, multilabel, valid)
    assert out[0, 0] == 1  # anomaly 0 -> id 1
    assert out[1, 1] == 8  # rarest anomaly index 7 -> id 8
    assert out[2, 2] == 0  # valid empty -> background
    assert out[3, 3] == IGNORE_INDEX


def test_singlelabel_overlap_picks_rarest_of_three():
    ds = FieldLensDataset.__new__(FieldLensDataset)
    ds.class_pixel_counts = [100, 50, 10, 200, 5, 80, 90, 1]

    multilabel = np.zeros((8, 2, 2), dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    # three overlapping classes; counts say index 4 (5 px) is rarest of these
    multilabel[0, 0, 0] = 1
    multilabel[2, 0, 0] = 1
    multilabel[4, 0, 0] = 1
    # two overlapping classes where the rarer one is the higher index
    multilabel[3, 1, 1] = 1
    multilabel[5, 1, 1] = 1

    out = FieldLensDataset._build_singlelabel(ds, multilabel, valid)
    assert out[0, 0] == 5  # anomaly index 4 -> id 5
    assert out[1, 1] == 6  # anomaly index 5 (80 px) beats index 3 (200 px)


def test_singlelabel_overlap_without_stats_uses_lowest_index():
    ds = FieldLensDataset.__new__(FieldLensDataset)
    ds.class_pixel_counts = None

    multilabel = np.zeros((8, 1, 1), dtype=np.float32)
    multilabel[2, 0, 0] = 1
    multilabel[6, 0, 0] = 1
    valid = np.ones((1, 1), dtype=bool)

    out = FieldLensDataset._build_singlelabel(ds, multilabel, valid)
    assert out[0, 0] == 3  # lowest channel index 2 -> id 3


def test_augment_alignment():
    rng = np.random.default_rng(0)
    ds = FieldLensDataset.__new__(FieldLensDataset)
    ds.augment = True
    image = np.arange(3 * 8 * 8, dtype=np.float32).reshape(8, 8, 3)
    multilabel = np.zeros((8, 8, 8), dtype=np.float32)
    multilabel[0, 1, 2] = 1
    valid = np.zeros((8, 8), dtype=bool)
    valid[1, 2] = True
    img2, ml2, v2 = FieldLensDataset._augment(
        ds, image.copy(), multilabel.copy(), valid.copy(), rng
    )
    # Find the single True in valid; multilabel channel 0 must match
    ys, xs = np.where(v2)
    assert len(ys) == 1
    assert ml2[0, ys[0], xs[0]] == 1
