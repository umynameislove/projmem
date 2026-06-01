"""Checks for committed synthetic pattern fixture metadata."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE_MANIFEST = Path("tests/fixtures/patterns/known_pattern_fixtures.json")
EXPECTED_FIXTURES = {
    "config_failure_ground_truth",
    "dataset_failure_ground_truth",
    "recurring_failure_ground_truth",
    "temporal_drift_ground_truth",
    "anomaly_ground_truth",
    "pattern_cli_ground_truth",
}


def test_pattern_fixture_manifest_is_complete_and_privacy_safe() -> None:
    """Fixture evidence should remain committed and claim-safe."""

    payload = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    raw_json = json.dumps(payload, sort_keys=True).casefold()
    fixture_ids = {str(item["fixture_id"]) for item in payload["fixtures"]}

    assert payload["schema_version"] == "pattern-fixture-manifest-v1"
    assert fixture_ids == EXPECTED_FIXTURES
    assert payload["privacy_policy"] == {
        "raw_failure_text": False,
        "raw_config_values": False,
        "raw_artifact_paths": False,
        "network": False,
    }
    assert "private" not in raw_json
    assert "/users/" not in raw_json
    assert "/home/" not in raw_json
    assert "caused_by" not in raw_json
    assert "true root cause" not in raw_json
    assert all(Path(item["test_file"]).exists() for item in payload["fixtures"])
