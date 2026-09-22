"""Class list sync: data.yaml, Python constants, and dashboard classes.json must match."""

from __future__ import annotations

import json
from pathlib import Path

from fieldlens.constants import ANOMALY_CLASSES, FORBIDDEN_LEGACY_CLASS_IDS
from fieldlens.profile import anomaly_ids_from_data, classes_export_payload, load_data_config
from fieldlens.paths import REPO_ROOT


EXPECTED_2021 = (
    "double_plant",
    "drydown",
    "endrow",
    "nutrient_deficiency",
    "planter_skip",
    "water",
    "waterway",
    "weed_cluster",
)


def test_data_yaml_has_2021_classes_only():
    ids = anomaly_ids_from_data(load_data_config())
    assert ids == EXPECTED_2021


def test_python_constants_match_data_yaml():
    assert ANOMALY_CLASSES == anomaly_ids_from_data()


def test_classes_json_matches_data_yaml_when_present():
    path = REPO_ROOT / "dashboard" / "public" / "data" / "classes.json"
    assert path.is_file(), "classes.json missing; run export or write from profile.classes_export_payload"
    disk = json.loads(path.read_text())
    expected = classes_export_payload()
    assert disk["anomaly_ids"] == list(expected["anomaly_ids"])
    assert [a["id"] for a in disk["anomalies"]] == list(expected["anomaly_ids"])
    assert disk["background"]["id"] == "background"


def test_dashboard_source_has_no_legacy_class_ids():
    dash = REPO_ROOT / "dashboard"
    bad_hits: list[str] = []
    for path in dash.rglob("*"):
        if path.suffix not in {".ts", ".tsx", ".json", ".md"}:
            continue
        if ".next" in path.parts or "node_modules" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for bad in FORBIDDEN_LEGACY_CLASS_IDS:
            if bad in text or bad.replace("_", " ") in text.lower():
                # allow mention only inside FORBIDDEN_LEGACY_CLASS_IDS definition
                if path.name == "constants.ts" and "FORBIDDEN_LEGACY_CLASS_IDS" in text:
                    continue
                bad_hits.append(f"{path}:{bad}")
    assert not bad_hits, "legacy 2020 class names still present:\n" + "\n".join(bad_hits)


def test_no_hardcoded_anomaly_array_in_dashboard_constants():
    constants = (REPO_ROOT / "dashboard" / "lib" / "constants.ts").read_text()
    assert "ANOMALY_CLASSES =" not in constants
    assert "standing_water" not in constants or "FORBIDDEN" in constants
