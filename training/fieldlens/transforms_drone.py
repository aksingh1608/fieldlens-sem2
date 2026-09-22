"""Drone condition transforms for robustness testing only (never used in training)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy_hwc(img: torch.Tensor) -> np.ndarray:
    """CHW float tensor -> HWC float32 numpy in roughly [0,1] display space if normalized."""
    x = img.detach().cpu().float()
    if x.ndim != 3:
        raise ValueError(f"expected CHW, got {tuple(x.shape)}")
    return x.permute(1, 2, 0).numpy()


def _from_numpy_hwc(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()


def motion_blur(image: torch.Tensor, severity: int, ksizes: list[int]) -> torch.Tensor:
    k = int(ksizes[severity])
    if k <= 1:
        return image
    if k % 2 == 0:
        k += 1
    arr = _to_numpy_hwc(image)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    out = cv2.filter2D(arr, -1, kernel)
    if out.ndim == 2:
        out = out[..., None]
    return _from_numpy_hwc(out)


def downscale_upscale(image: torch.Tensor, severity: int, factors: list[float]) -> torch.Tensor:
    f = float(factors[severity])
    if f >= 1.0:
        return image
    c, h, w = image.shape
    nh, nw = max(1, int(h * f)), max(1, int(w * f))
    x = image.unsqueeze(0)
    small = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    back = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return back.squeeze(0)


def brightness_shift(image: torch.Tensor, severity: int, deltas: list[float]) -> torch.Tensor:
    d = float(deltas[severity])
    if d == 0.0:
        return image
    return (image + d).clamp(0.0, 1.0)


def gaussian_noise(
    image: torch.Tensor, severity: int, stds: list[float], generator: torch.Generator | None = None
) -> torch.Tensor:
    std = float(stds[severity])
    if std <= 0.0:
        return image
    noise = torch.randn(image.shape, dtype=image.dtype, device=image.device, generator=generator) * std
    return (image + noise).clamp(0.0, 1.0)


def apply_drone_condition(
    image: torch.Tensor,
    severity: int,
    eval_cfg: dict[str, Any],
    effect: str = "all",
) -> torch.Tensor:
    """
    Apply robustness effects to an image tensor in approximately [0,1] CHW space.
    Same parameters are used for RGB and NIR by the caller.
    """
    if severity <= 0:
        return image
    rob = eval_cfg.get("robustness", eval_cfg)
    x = image
    if effect in ("all", "motion_blur"):
        x = motion_blur(x, severity, rob["motion_blur_ksizes"])
    if effect in ("all", "downscale"):
        x = downscale_upscale(x, severity, rob["downscale_factors"])
    if effect in ("all", "brightness"):
        x = brightness_shift(x, severity, rob["brightness_deltas"])
    if effect in ("all", "noise"):
        x = gaussian_noise(x, severity, rob["gaussian_noise_stds"])
    return x
