"""Profile loading and path separation for pilot vs full."""

from __future__ import annotations

from fieldlens.profile import (
    apply_profile_to_run_cfg,
    load_profile,
    merge_data_with_profile,
    profile_ckpt_dir,
    profile_run_dir,
)


def test_pilot_profile_sizes():
    p = load_profile("pilot")
    assert p["n_train"] == 1500
    assert p["n_val"] == 400
    assert p["n_test"] == 800
    assert p["encoder"] == "nvidia/mit-b0"
    assert p["epochs"] == 10


def test_full_profile_is_marked_placeholder():
    p = load_profile("full")
    assert p["n_train"] == 56944
    assert "PLACEHOLDER" in str(p["encoder"])
    assert p["epochs"] == 100


def test_profile_paths_include_profile_name():
    assert "pilot" in str(profile_run_dir("pilot", "run1"))
    assert "full" in str(profile_ckpt_dir("full", "run2"))
    data = merge_data_with_profile(load_profile("pilot"))
    assert data["subset_root"].endswith("pilot/tiles")
    assert data["split_csv"].endswith("pilot/split.csv")


def test_apply_profile_overlays_run_config():
    run = {
        "run_name": "run1",
        "seed": 42,
        "data": {"channels": "rgb", "batch_size": 99},
        "model": {"head": "softmax", "encoder": "old"},
        "optim": {"epochs": 1, "accum_steps": 1, "encoder_lr": 1, "other_lr": 1, "weight_decay": 0.01, "warmup_epochs": 1, "amp": True},
    }
    out = apply_profile_to_run_cfg(run, load_profile("pilot"))
    assert out["optim"]["epochs"] == 10
    assert out["data"]["batch_size"] == 4
    assert out["model"]["encoder"] == "nvidia/mit-b0"
    assert out["profile"] == "pilot"
