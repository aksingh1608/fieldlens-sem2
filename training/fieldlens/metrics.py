"""Metrics including official Agriculture-Vision modified mIoU."""

from __future__ import annotations

import numpy as np

from fieldlens.constants import NUM_ANOMALY_CLASSES, NUM_CLASSES


def label_set_mask(multilabel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Boolean (9,H,W) mask of the ground-truth label set Y per pixel.

    Y = {background} when no anomaly is present, otherwise the set of present
    anomaly ids. Invalid pixels have an empty set.
    """
    valid = np.asarray(valid, dtype=bool)
    present = multilabel > 0.5
    lab = np.zeros((NUM_CLASSES,) + valid.shape, dtype=bool)
    lab[1:] = present
    lab[0] = ~present.any(axis=0)
    lab &= valid
    return lab


def update_agri_confusion(
    conf: np.ndarray,
    pred: np.ndarray,
    multilabel: np.ndarray,
    valid: np.ndarray,
) -> None:
    """
    Official SHI-Labs Agriculture-Vision modified confusion update.

    For each valid pixel with prediction x and ground-truth label set Y
    (Y = {background} if no anomalies, else the set of present anomaly class ids):

    1. If x in Y: M[y, y] += 1 for each y in Y
    2. Else:      M[x, y] += 1 for each y in Y

    Class ids: 0=background, 1..8=anomalies (2021). Indices into conf are those ids.

    NOTE: This differs from an earlier draft in the FieldLens build prompt, which used
    M[x,x]+=1 on correct and M[y,x]+=1 on incorrect (gt-row / pred-col). We follow the
    official SHI-Labs README rule instead. See docs/evaluation.md.

    Vectorized over pixels: a per-pixel Python loop costs minutes per tile at
    512x512 and dominates both validation and evaluation.
    """
    if pred.shape != valid.shape:
        raise ValueError("pred/valid shape mismatch")
    if multilabel.shape[0] != NUM_ANOMALY_CLASSES:
        raise ValueError("multilabel must have 8 channels")
    if multilabel.shape[1:] != valid.shape:
        raise ValueError("multilabel/valid spatial shape mismatch")

    pred = np.asarray(pred, dtype=np.intp)
    valid = np.asarray(valid, dtype=bool)
    if pred.size and (pred.min() < 0 or pred.max() >= NUM_CLASSES):
        raise ValueError(f"pred ids must be in [0, {NUM_CLASSES - 1}]")

    lab = label_set_mask(multilabel, valid)
    # Was the predicted class part of the label set for that pixel?
    in_set = np.take_along_axis(lab, pred[None], axis=0)[0]
    miss = valid & ~in_set
    missed_pred = pred[miss]

    for c in range(NUM_CLASSES):
        lab_c = lab[c]
        conf[c, c] += int(np.count_nonzero(lab_c & in_set))
        wrong = lab_c[miss]
        if wrong.any():
            conf[:, c] += np.bincount(missed_pred[wrong], minlength=NUM_CLASSES)


def iou_from_confusion(conf: np.ndarray) -> tuple[np.ndarray, float]:
    """IoU_c = M[c,c] / (row_sum[c] + col_sum[c] - M[c,c]); mIoU = mean over classes."""
    tp = np.diag(conf).astype(np.float64)
    row = conf.sum(axis=1).astype(np.float64)
    col = conf.sum(axis=0).astype(np.float64)
    denom = row + col - tp
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return iou, float(iou.mean()) if len(iou) else 0.0


def binary_counts(
    pred_multi: np.ndarray,
    gt_multi: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Per-class (tp, fp, fn) over valid pixels; shape (8, 3), float64.

    Counts are additive, so callers can stream tiles instead of holding every
    prediction in memory.
    """
    counts = np.zeros((NUM_ANOMALY_CLASSES, 3), dtype=np.float64)
    v = valid.astype(bool)
    for c in range(NUM_ANOMALY_CLASSES):
        p = pred_multi[c][v] > 0.5
        g = gt_multi[c][v] > 0.5
        counts[c, 0] = float(np.count_nonzero(p & g))
        counts[c, 1] = float(np.count_nonzero(p & ~g))
        counts[c, 2] = float(np.count_nonzero(~p & g))
    return counts


def metrics_from_counts(counts: np.ndarray) -> dict[str, list[float]]:
    """Per-class IoU / precision / recall / F1 from (8, 3) tp/fp/fn counts."""
    ious, precs, recs, f1s = [], [], [], []
    for tp, fp, fn in counts:
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ious.append(iou)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    return {"iou": ious, "precision": precs, "recall": recs, "f1": f1s}


def multilabel_binary_metrics(
    pred_multi: np.ndarray,
    gt_multi: np.ndarray,
    valid: np.ndarray,
) -> dict[str, list[float]]:
    """Per-class IoU / precision / recall / F1 on 8 binary masks over valid pixels."""
    return metrics_from_counts(binary_counts(pred_multi, gt_multi, valid))


def overlap_mask(gt_multi: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return valid & (gt_multi.sum(axis=0) >= 2)


def logits_to_pred_class(logits: np.ndarray, head: str, threshold: float = 0.5) -> np.ndarray:
    """
    Run 1 (softmax): argmax over 9 channels.
    Runs 2/3 (sigmoid): highest anomaly if sigmoid(logit) >= threshold else background.
    """
    if head == "softmax":
        return logits.argmax(axis=0).astype(np.int64)
    # sigmoid multi-label -> single class for modified mIoU
    probs = 1.0 / (1.0 + np.exp(-logits))
    best = probs.argmax(axis=0)
    best_p = probs.max(axis=0)
    pred = np.where(best_p >= threshold, best + 1, 0).astype(np.int64)
    return pred


def logits_to_multilabel(logits: np.ndarray, head: str, threshold: float = 0.5) -> np.ndarray:
    if head == "softmax":
        pred = logits.argmax(axis=0)
        out = np.zeros((NUM_ANOMALY_CLASSES,) + pred.shape, dtype=np.float32)
        for c in range(1, NUM_CLASSES):
            out[c - 1] = (pred == c).astype(np.float32)
        return out
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs >= threshold).astype(np.float32)


def tile_positive(
    mask: np.ndarray, valid: np.ndarray, fraction: float
) -> bool:
    v = int(valid.sum())
    if v == 0:
        return False
    return float(mask[valid].sum()) / v >= fraction
