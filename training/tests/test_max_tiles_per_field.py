"""Tests for max_tiles_per_field sampling."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_subset import (  # noqa: E402
    assign_fields,
    field_id_from_name,
    limit_tiles_per_field,
    max_tiles_per_field_cfg,
)
from fieldlens.profile import load_profile, merge_data_with_profile  # noqa: E402


def _fake_tiles(n_fields: int, tiles_per_field: int) -> list[str]:
    out = []
    for f in range(n_fields):
        fid = f"F{f:03d}"
        for i in range(tiles_per_field):
            out.append(f"{fid}_{i:04d}-0-0-0")
    return out


def test_limit_tiles_per_field_cap():
    tiles = _fake_tiles(5, 20)
    limited = limit_tiles_per_field(tiles, 10, seed=42)
    counts = {}
    for t in limited:
        fid = field_id_from_name(t)
        counts[fid] = counts.get(fid, 0) + 1
    assert all(v <= 10 for v in counts.values())
    assert len(limited) == 50


def test_assign_fields_respects_per_field_cap():
    tiles = _fake_tiles(40, 50)
    first, second = assign_fields(
        tiles,
        n_first=100,
        n_second=50,
        seed=42,
        max_per_field_first=10,
        max_per_field_second=10,
    )
    for group in (first, second):
        counts = {}
        for t in group:
            fid = field_id_from_name(t)
            counts[fid] = counts.get(fid, 0) + 1
        assert counts
        assert max(counts.values()) <= 10


def test_assign_fields_no_field_overlap():
    tiles = _fake_tiles(30, 20)
    first, second = assign_fields(
        tiles, 80, 40, seed=0, max_per_field_first=10, max_per_field_second=10
    )
    a = {field_id_from_name(t) for t in first}
    b = {field_id_from_name(t) for t in second}
    assert a.isdisjoint(b)


def test_pilot_v2_profile_has_caps():
    p = load_profile("pilot_v2")
    assert p["n_train"] == 1500
    assert p["n_val"] == 400
    assert p["n_test"] == 800
    assert p["max_tiles_per_field"]["train"] == 10
    assert p["max_tiles_per_field"]["val"] == 10
    assert p["max_tiles_per_field"]["test"] == 5
    data = merge_data_with_profile(p)
    assert data["max_tiles_per_field"]["train"] == 10
    assert "pilot_v2" in data["subset_root"]


def test_max_tiles_cfg_shapes():
    assert max_tiles_per_field_cfg({}) == {"train": None, "val": None, "test": None}
    assert max_tiles_per_field_cfg({"max_tiles_per_field": 7})["train"] == 7
    d = max_tiles_per_field_cfg({"max_tiles_per_field": {"train": 10, "val": 10, "test": 5}})
    assert d == {"train": 10, "val": 10, "test": 5}
