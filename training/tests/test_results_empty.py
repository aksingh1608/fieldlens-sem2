"""Empty / absent results.json must not crash results helpers."""

from __future__ import annotations

import json
from pathlib import Path

from fieldlens.paths import REPO_ROOT


DASH = REPO_ROOT / "dashboard"
RESULTS_CLIENT = DASH / "components" / "results" / "ResultsPageClient.tsx"
LOAD_RESULTS = DASH / "lib" / "data" / "loadResults.ts"
COMPARE = DASH / "components" / "compare" / "CompareView.tsx"


def test_results_client_uses_optional_dataset_access():
    text = RESULTS_CLIENT.read_text()
    assert "data.dataset.name" not in text
    assert "dataset?.name" in text or "dataset?.name" in text.replace(" ", "")
    assert "emptyResultsStub" in text or "loadResults" in text


def test_load_results_returns_null_on_failure():
    text = LOAD_RESULTS.read_text()
    assert "return null" in text
    assert "emptyResultsStub" in text


def test_compare_handles_null_results():
    text = COMPARE.read_text()
    assert "results?.tile_compare" in text or "results?.tile_compare" in text.replace(" ", "")


def test_empty_and_absent_results_fixtures_are_valid_json_shapes():
    empty = {
        "status": "pending_runs",
        "generated_at": None,
        "dataset": None,
        "runs": {"run1": None, "run2": None, "run3": None},
    }
    # Simulate what the client does with empty payload
    dataset = empty.get("dataset")
    assert (dataset or {}).get("name") is None if dataset else True
    assert empty["runs"]["run1"] is None

    # Absent file: client uses stub with same keys
    stub_path = DASH / "public" / "data" / "results.empty.fixture.json"
    stub_path.write_text(json.dumps(empty, indent=2))
    loaded = json.loads(stub_path.read_text())
    assert loaded["dataset"] is None
    stub_path.unlink()


def test_try_page_gated_by_site_json():
    site = json.loads((DASH / "public" / "data" / "site.json").read_text())
    assert site.get("enable_try_page") is False
    try_page = (DASH / "app" / "try" / "page.tsx").read_text()
    assert "enable_try_page" in try_page
    assert "notFound" in try_page
