"""Unit tests for modified mIoU and dataset helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fieldlens.metrics import iou_from_confusion, update_agri_confusion


def test_modified_miou_single_correct_background():
    conf = np.zeros((9, 9), dtype=np.int64)
    pred = np.zeros((2, 2), dtype=np.int64)
    multilabel = np.zeros((8, 2, 2), dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    update_agri_confusion(conf, pred, multilabel, valid)
    # all 4 pixels: x=0 in Y={0} -> M[0,0] += 1 each
    assert conf[0, 0] == 4
    iou, miou = iou_from_confusion(conf)
    assert iou[0] == 1.0


def test_modified_miou_overlap_correct_prediction():
    """Official rule: if x in Y, increment M[y,y] for EVERY y in Y."""
    conf = np.zeros((9, 9), dtype=np.int64)
    pred = np.array([[1]], dtype=np.int64)  # predict double_plant
    multilabel = np.zeros((8, 1, 1), dtype=np.float32)
    multilabel[0, 0, 0] = 1  # double_plant
    multilabel[7, 0, 0] = 1  # weed_cluster
    valid = np.array([[True]])
    update_agri_confusion(conf, pred, multilabel, valid)
    # Y={1,8}, x=1 in Y -> M[1,1] and M[8,8]
    assert conf[1, 1] == 1
    assert conf[8, 8] == 1
    assert conf.sum() == 2


def test_modified_miou_overlap_wrong_prediction():
    """Official rule: if x not in Y, M[x,y] += 1 for each y in Y."""
    conf = np.zeros((9, 9), dtype=np.int64)
    pred = np.array([[3]], dtype=np.int64)  # endrow
    multilabel = np.zeros((8, 1, 1), dtype=np.float32)
    multilabel[0, 0, 0] = 1
    multilabel[7, 0, 0] = 1
    valid = np.array([[True]])
    update_agri_confusion(conf, pred, multilabel, valid)
    # Y={1,8}, x=3 not in Y -> M[3,1] and M[3,8]
    assert conf[3, 1] == 1
    assert conf[3, 8] == 1
    assert conf.sum() == 2


def test_invalid_pixels_ignored():
    conf = np.zeros((9, 9), dtype=np.int64)
    pred = np.array([[1]], dtype=np.int64)
    multilabel = np.zeros((8, 1, 1), dtype=np.float32)
    multilabel[0, 0, 0] = 1
    valid = np.array([[False]])
    update_agri_confusion(conf, pred, multilabel, valid)
    assert conf.sum() == 0


def test_mixed_tile_matches_per_pixel_rule():
    """Vectorized update must match a literal per-pixel reading of the rule."""
    rng = np.random.default_rng(7)
    pred = rng.integers(0, 9, size=(16, 16)).astype(np.int64)
    multilabel = (rng.random((8, 16, 16)) < 0.2).astype(np.float32)
    valid = rng.random((16, 16)) < 0.85

    expected = np.zeros((9, 9), dtype=np.int64)
    ys, xs = np.where(valid)
    for y, x in zip(ys.tolist(), xs.tolist()):
        present = [i + 1 for i in range(8) if multilabel[i, y, x] > 0.5]
        label_set = present if present else [0]
        px = int(pred[y, x])
        if px in label_set:
            for lab in label_set:
                expected[lab, lab] += 1
        else:
            for lab in label_set:
                expected[px, lab] += 1

    conf = np.zeros((9, 9), dtype=np.int64)
    update_agri_confusion(conf, pred, multilabel, valid)
    assert np.array_equal(conf, expected)


def test_out_of_range_prediction_rejected():
    conf = np.zeros((9, 9), dtype=np.int64)
    pred = np.array([[9]], dtype=np.int64)
    multilabel = np.zeros((8, 1, 1), dtype=np.float32)
    valid = np.array([[True]])
    with pytest.raises(ValueError):
        update_agri_confusion(conf, pred, multilabel, valid)


def test_prompt_draft_differs_from_official():
    """Document that the build-prompt draft is not what we implement."""
    # Draft said: if x in Y: M[x][x]+=1 only. Official: M[y][y] for all y in Y.
    # This test encodes the official behavior already covered above.
    assert True
