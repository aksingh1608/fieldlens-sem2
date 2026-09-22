"""Load shared data.yaml and scale profiles. No if-scale branching outside overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fieldlens.paths import CONFIG_DIR, REPO_ROOT, repo_path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def load_profile(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / "profiles" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown profile {name!r}: missing {path}")
    cfg = load_yaml(path)
    if cfg.get("profile") != name:
        raise ValueError(f"profile file {path} must set profile: {name}")
    return cfg


def load_data_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "data.yaml")


def merge_data_with_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Shared data.yaml plus profile subset sizes and profile-scoped paths."""
    data = load_data_config()
    name = profile["profile"]
    for key in ("n_train", "n_val", "n_test", "strict_tile_cap", "seed", "max_tiles_per_field"):
        if key in profile:
            data[key] = profile[key]
    # Keep pilot and full data trees apart
    data["data_root"] = f"data/fieldlens/{name}"
    data["subset_root"] = f"data/fieldlens/{name}/tiles"
    data["split_csv"] = f"data/fieldlens/{name}/split.csv"
    data["stats_json"] = f"data/fieldlens/{name}/stats.json"
    data["profile"] = name
    return data


def apply_profile_to_run_cfg(run_cfg: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Overlay profile train/model/loader fields onto a run yaml dict (copy)."""
    out = yaml.safe_load(yaml.dump(run_cfg))
    data_paths = merge_data_with_profile(profile)
    out.setdefault("data", {})
    out.setdefault("model", {})
    out.setdefault("optim", {})
    out["profile"] = profile["profile"]
    out["data"]["subset_root"] = data_paths["subset_root"]
    out["data"]["split_csv"] = data_paths["split_csv"]
    out["data"]["stats_json"] = data_paths["stats_json"]
    out["data"]["batch_size"] = profile["batch_size"]
    out["data"]["num_workers"] = profile["num_workers"]
    out["data"]["pin_memory"] = profile.get("pin_memory", True)
    out["model"]["encoder"] = profile["encoder"]
    out["optim"]["epochs"] = profile["epochs"]
    out["optim"]["accum_steps"] = profile["accum_steps"]
    out["optim"]["encoder_lr"] = profile["encoder_lr"]
    out["optim"]["other_lr"] = profile["other_lr"]
    out["optim"]["weight_decay"] = profile["weight_decay"]
    out["optim"]["warmup_epochs"] = profile["warmup_epochs"]
    out["optim"]["amp"] = profile.get("amp", True)
    return out


def profile_run_dir(profile_name: str, run_name: str) -> Path:
    return REPO_ROOT / "runs" / profile_name / run_name


def profile_ckpt_dir(profile_name: str, run_name: str) -> Path:
    return REPO_ROOT / "checkpoints" / profile_name / run_name


def add_profile_arg(parser) -> None:
    parser.add_argument(
        "--profile",
        default="pilot",
        help="Scale profile under configs/profiles/ (default: pilot)",
    )


def rgb_to_hex(rgb: list[int] | tuple[int, ...]) -> str:
    r, g, b = (int(x) for x in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def classes_export_payload(data_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Payload written to dashboard/public/data/classes.json."""
    data = data_cfg or load_data_config()
    anomalies = []
    for item in data["classes"]:
        anomalies.append(
            {
                "id": item["id"],
                "label": item["label"],
                "color": rgb_to_hex(item["color_rgb"]),
                "color_rgb": list(item["color_rgb"]),
            }
        )
    bg = data["background"]
    return {
        "source": "training/configs/data.yaml",
        "background": {
            "id": bg["id"],
            "label": bg["label"],
            "color": rgb_to_hex(bg["color_rgb"]),
            "color_rgb": list(bg["color_rgb"]),
        },
        "anomalies": anomalies,
        "anomaly_ids": [a["id"] for a in anomalies],
    }


def anomaly_ids_from_data(data_cfg: dict[str, Any] | None = None) -> tuple[str, ...]:
    data = data_cfg or load_data_config()
    return tuple(item["id"] for item in data["classes"])


def class_colors_from_data(data_cfg: dict[str, Any] | None = None) -> dict[str, tuple[int, int, int]]:
    data = data_cfg or load_data_config()
    out: dict[str, tuple[int, int, int]] = {}
    bg = data["background"]
    out[bg["id"]] = tuple(int(x) for x in bg["color_rgb"])  # type: ignore[assignment]
    for item in data["classes"]:
        out[item["id"]] = tuple(int(x) for x in item["color_rgb"])  # type: ignore[assignment]
    return out
