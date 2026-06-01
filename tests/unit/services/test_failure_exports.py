"""failure export service tests for privacy-safe failure listing/export."""

from __future__ import annotations

import json
import sys

import pytest

from pmem.errors import PmemSecurityError
from pmem.services.failure_exports import (
    FAILURE_EXPORT_SCHEMA_VERSION,
    export_failure_records,
    failure_export_payload,
    list_failure_records,
)
from pmem.services.failure_logging import log_failure
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command


def test_failure_export_payload_handles_empty_project(tmp_path) -> None:
    """An initialized project with no failures should still have a stable contract."""

    init_project(tmp_path, project_name="demo")

    payload = failure_export_payload(tmp_path, generated_at="2026-01-01T00:00:00Z")

    assert payload == {
        "schema_version": FAILURE_EXPORT_SCHEMA_VERSION,
        "generated_at": "2026-01-01T00:00:00Z",
        "privacy_mode": "redacted",
        "include_text": False,
        "record_count": 0,
        "records": [],
    }


def test_list_failure_records_excludes_raw_text_by_default(tmp_path) -> None:
    """Default records should not leak failure free text into analysis substrates."""

    _create_failure(tmp_path)

    records = list_failure_records(tmp_path)

    assert len(records) == 1
    record = records[0]
    assert record["text_included"] is False
    assert record["error_type"] == "MetricRegression"
    assert record["tags"] == ["data_quality"]
    assert "description" not in record
    assert "root_cause" not in record
    assert "lesson" not in record


def test_list_failure_records_can_include_text_when_caller_confirms_policy(tmp_path) -> None:
    """Service supports explicit text inclusion; CLI owns confirmation UX."""

    _create_failure(tmp_path)

    records = list_failure_records(tmp_path, include_text=True)

    assert records[0]["text_included"] is True
    assert records[0]["description"] == "SECRET training accuracy dropped."
    assert records[0]["root_cause"] == "Bad data split"
    assert records[0]["lesson"] == "Audit split seed"


def test_export_failure_records_writes_canonical_json(tmp_path) -> None:
    """Export should write reviewable JSON outside `.pmem`."""

    _create_failure(tmp_path)

    result = export_failure_records(tmp_path, output_path="exports/failures.json")
    payload = json.loads((tmp_path / "exports" / "failures.json").read_text(encoding="utf-8"))

    assert result.display_path == "exports/failures.json"
    assert payload["schema_version"] == FAILURE_EXPORT_SCHEMA_VERSION
    assert payload["record_count"] == 1
    assert payload["include_text"] is False
    assert "description" not in payload["records"][0]


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../failures.json",
        ".PMEM/failures.json",
        "/tmp/failures.json",
        "bad\\path.json",
        "bad\x00path.json",
    ],
)
def test_export_failure_records_rejects_unsafe_paths(tmp_path, unsafe_path) -> None:
    """Failure export paths must not cross project or `.pmem` boundaries."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemSecurityError):
        export_failure_records(tmp_path, output_path=unsafe_path)


def _create_failure(tmp_path) -> None:
    init_project(tmp_path, project_name="demo")
    run_result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="MetricRegression",
        description="SECRET training accuracy dropped.",
        root_cause="Bad data split",
        lesson="Audit split seed",
        severity="high",
        tags=("data quality",),
        source="user_confirmed",
    )
