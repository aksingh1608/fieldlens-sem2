"""Shared constants for Agriculture-Vision 2021 (supervised challenge subset).

Class ids and colours are loaded from training/configs/data.yaml (single source of truth).
"""

from __future__ import annotations

from fieldlens.profile import anomaly_ids_from_data, class_colors_from_data, load_data_config

_DATA = load_data_config()
ANOMALY_CLASSES: tuple[str, ...] = anomaly_ids_from_data(_DATA)
CLASS_NAMES: tuple[str, ...] = ("background",) + ANOMALY_CLASSES
NUM_ANOMALY_CLASSES = len(ANOMALY_CLASSES)
NUM_CLASSES = len(CLASS_NAMES)

IGNORE_INDEX = 255

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = class_colors_from_data(_DATA)

FORBIDDEN_LEGACY_CLASS_IDS: tuple[str, ...] = (
    "cloud_shadow",
    "standing_water",
    "weed_strip",
)
