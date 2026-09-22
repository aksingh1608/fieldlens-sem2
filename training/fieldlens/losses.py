"""Losses for FieldLens runs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fieldlens.constants import IGNORE_INDEX, NUM_ANOMALY_CLASSES


class SoftmaxCELoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, ignore_index=self.ignore_index)


class BCEDiceLoss(nn.Module):
    """Per-class BCE + soft Dice, computed only over valid pixels."""

    def __init__(self, pos_weight: torch.Tensor | None = None, dice_weight: float = 1.0) -> None:
        super().__init__()
        # pos_weight broadcasts against (B,C,H,W) from the trailing dimension, so a
        # flat (C,) tensor would line up with width instead of the class axis.
        if pos_weight is None:
            weight = None
        else:
            weight = torch.as_tensor(pos_weight, dtype=torch.float32).reshape(
                1, NUM_ANOMALY_CLASSES, 1, 1
            )
        self.register_buffer("pos_weight", weight)
        self.dice_weight = dice_weight

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        # logits/target: (B,C,H,W), valid: (B,H,W)
        valid_f = valid.unsqueeze(1).float()
        pos_weight = self.pos_weight
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
            pos_weight=None if pos_weight is None else pos_weight.to(logits.dtype),
        )
        bce = (bce * valid_f).sum() / valid_f.sum().clamp_min(1.0)

        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (probs * target * valid_f).sum(dim=dims)
        denom = (probs * valid_f).sum(dim=dims) + (target * valid_f).sum(dim=dims)
        dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
        return bce + self.dice_weight * dice.mean()


def build_loss(run_cfg: dict, pos_weight: torch.Tensor | None = None) -> nn.Module:
    loss_cfg = run_cfg["loss"]
    if loss_cfg["type"] == "cross_entropy":
        return SoftmaxCELoss(ignore_index=loss_cfg.get("ignore_index", IGNORE_INDEX))
    if loss_cfg["type"] == "bce_dice":
        pw = pos_weight if loss_cfg.get("use_pos_weight", False) else None
        return BCEDiceLoss(pos_weight=pw)
    raise ValueError(f"unknown loss type: {loss_cfg['type']}")
