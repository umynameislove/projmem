"""Unit coverage for conflict and shared-path helper edge cases."""

from __future__ import annotations

from pathlib import Path

from pmem.services import conflict_detection as detection
from pmem.services import conflict_resolution as resolution
from pmem.services import shared_paths

HASH = "sha256:" + "a" * 64


def test_conflict_detection_helper_edge_cases(tmp_path) -> None:
    """Private helpers should stay deterministic for malformed optional data."""

    assert detection._entity_records({"not": "a-list"}) == ()
    assert detection._semantic_run_id({}) == ""
    assert detection._normalize_command(None) == ""
    assert detection._normalize_command("  python   train.py  ") == "python train.py"
    assert detection._timestamp_date(None) == ""
    assert detection._timestamp_date("bad") == ""
    assert detection._timestamp_date("2026-05-22T10:20:30Z") == "2026-05-22"
    assert detection._sorted_metric_values(None) == ""
    assert detection._sorted_metric_values({"b": 2, "a": 1, "nested": {"x": 1}}) == "a=1,b=2"
    assert detection._baseline_run_id({"metadata": []}) is None
    assert detection._baseline_run_id({"metadata": {"baseline_run_id": "run_1"}}) == "run_1"
    assert detection._normalize_artifact_key(None) is None
    assert detection._normalize_artifact_key("bad\\path") is None
    assert detection._normalize_artifact_key("../bad") is None
    assert detection._normalize_artifact_key("outputs/./model.bin") == "outputs/model.bin"
    assert detection._artifact_hash({"hash": HASH}) == HASH
    assert detection._artifact_hash({"sha256": "bad"}) is None
    assert detection._entity_type_from_field("artifact_index[0].run_id") == "artifact_index"
    assert detection._entity_type_from_field("other") == "bundle"
    assert detection._display_path(tmp_path, Path("/tmp/outside.json")) == "outside.json"


def test_conflict_resolution_validation_helpers() -> None:
    """Accept explicit blanks but reject bad conflict-resolution shapes."""

    assert resolution._validate_optional_hash(None, "before_hash") is None
    assert resolution._validate_optional_hash(" ", "before_hash") is None
    assert resolution._validate_optional_hash(HASH, "before_hash") == HASH
    assert resolution._validate_action("KEEP-LOCAL") == "keep-local"


def test_shared_path_helper_edge_cases(tmp_path) -> None:
    """shared-path helper behavior should not expose absolute paths by default."""

    assert shared_paths._display_path(tmp_path, tmp_path / "shared") == "shared"
    assert shared_paths._display_path(tmp_path, Path("/tmp/external-share")) == (
        "<external:external-share>"
    )
    assert shared_paths._has_control_character("ok") is False
    assert shared_paths._has_control_character("bad\x00") is True
    assert shared_paths._normalize_alias(None, tmp_path / "shared") == "shared"
    assert shared_paths._validate_mode("READ") == "read"
