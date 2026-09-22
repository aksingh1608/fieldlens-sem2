"""Loss tests (requires torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fieldlens.losses import BCEDiceLoss, build_loss  # noqa: E402


def _batch(h: int = 6, w: int = 5):
    logits = torch.zeros(2, 8, h, w)
    target = torch.zeros(2, 8, h, w)
    target[:, 0, :2, :2] = 1.0
    valid = torch.ones(2, h, w, dtype=torch.bool)
    return logits, target, valid


@pytest.mark.parametrize("shape", [(6, 5), (8, 8), (17, 3)])
def test_bce_dice_runs_for_any_spatial_shape(shape):
    """pos_weight broadcasts from the trailing axis, so it must not be a flat (C,)."""
    logits, target, valid = _batch(*shape)
    loss = BCEDiceLoss(pos_weight=torch.ones(8))
    value = loss(logits, target, valid)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_bce_dice_without_pos_weight():
    logits, target, valid = _batch()
    value = BCEDiceLoss()(logits, target, valid)
    assert torch.isfinite(value)


def test_pos_weight_applies_to_the_class_axis():
    """Up-weighting class 0 must change the loss; up-weighting an absent class must not."""
    logits, target, valid = _batch()
    base = BCEDiceLoss()(logits, target, valid)

    w_present = torch.ones(8)
    w_present[0] = 50.0  # class 0 is the only one with positives
    weighted = BCEDiceLoss(pos_weight=w_present)(logits, target, valid)
    assert weighted > base

    w_absent = torch.ones(8)
    w_absent[5] = 50.0  # no positives for class 5
    unchanged = BCEDiceLoss(pos_weight=w_absent)(logits, target, valid)
    assert torch.isclose(unchanged, base)


def test_invalid_pixels_do_not_change_bce():
    logits, target, valid = _batch()
    loss = BCEDiceLoss()
    reference = loss(logits, target, valid)

    noisy = logits.clone()
    noisy[:, :, -1, :] = 25.0  # nonsense in a region we mark invalid
    partial = valid.clone()
    partial[:, -1, :] = False

    masked = loss(noisy, target, partial)
    same_rows_masked = loss(logits, target, partial)
    assert torch.isclose(masked, same_rows_masked)
    assert not torch.isclose(masked, reference)


def test_build_loss_wires_pos_weight():
    cfg = {"loss": {"type": "bce_dice", "use_pos_weight": True}}
    loss = build_loss(cfg, pos_weight=torch.arange(1, 9, dtype=torch.float32))
    assert loss.pos_weight.shape == (1, 8, 1, 1)

    cfg_off = {"loss": {"type": "bce_dice", "use_pos_weight": False}}
    assert build_loss(cfg_off, pos_weight=torch.ones(8)).pos_weight is None


def test_build_loss_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_loss({"loss": {"type": "nope"}})
