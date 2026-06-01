"""incremental graph build conservative graph incremental build tests."""

from __future__ import annotations

import json
import stat
import sys

import pytest

from pmem.errors import PmemNotFoundError
from pmem.graph import incremental
from pmem.graph.incremental import (
    BUILD_MODE_FULL,
    BUILD_MODE_INCREMENTAL_FALLBACK_FULL,
    BUILD_MODE_INCREMENTAL_NOOP,
    build_graph_full,
    build_graph_incremental,
    compute_graph_source_fingerprint,
)
from pmem.graph.persistence import default_graph_artifact_path, write_graph_document
from pmem.graph.schema import GRAPH_SCHEMA_VERSION
from pmem.services.failure_logging import log_failure
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command


def test_source_fingerprint_is_deterministic_and_counts_tables(tmp_path) -> None:
    init_project(tmp_path, project_name="fingerprint-demo")
    run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    first = compute_graph_source_fingerprint(tmp_path / ".pmem" / "pmem.db")
    second = compute_graph_source_fingerprint(tmp_path / ".pmem" / "pmem.db")

    assert first.value == second.value
    assert first.table_counts == second.table_counts
    assert first.to_dict()["value"] == first.value
    assert first.table_counts["projects"] == 1
    assert first.table_counts["runs"] == 1


def test_source_fingerprint_payload_hashes_structured_project_vocabulary() -> None:
    for column, value in {
        "metrics_json": '{"RAW_SENTINEL_metric": 0.94}',
        "tags_json": '["RAW_SENTINEL_tag"]',
        "error_type": "RAW_SENTINEL_error",
    }.items():
        canonical = incremental._canonical_value("runs", column, value)  # noqa: SLF001

        assert isinstance(canonical, dict)
        assert set(canonical) == {"sha256", "stored"}
        assert canonical["stored"] is False
        assert "RAW_SENTINEL" not in json.dumps(canonical, sort_keys=True)


def test_source_fingerprint_rejects_missing_database(tmp_path) -> None:
    missing = tmp_path / ".pmem" / "missing.db"

    with pytest.raises(PmemNotFoundError):
        compute_graph_source_fingerprint(missing)


def test_source_fingerprint_changes_when_run_or_failure_changes(tmp_path) -> None:
    init_project(tmp_path, project_name="fingerprint-change-demo")
    first = compute_graph_source_fingerprint(tmp_path / ".pmem" / "pmem.db")
    run_result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    second = compute_graph_source_fingerprint(tmp_path / ".pmem" / "pmem.db")
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="ObservedFailure",
        description="RAW_SENTINEL failure text",
    )
    third = compute_graph_source_fingerprint(tmp_path / ".pmem" / "pmem.db")

    assert first.value != second.value
    assert second.value != third.value
    assert third.table_counts["failures"] == 1


def test_full_build_metadata_is_privacy_safe_and_json_serializable(tmp_path) -> None:
    init_project(tmp_path, project_name="full-build-demo")
    run_result = run_command(tmp_path, [sys.executable, "-c", "print('RAW_SENTINEL')"])
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="PrivacyRegression",
        description="RAW_SENTINEL failure text",
        root_cause="RAW_SENTINEL root cause",
        lesson="RAW_SENTINEL lesson",
    )

    result = build_graph_full(tmp_path)
    metadata_json = json.dumps(result.document.metadata, sort_keys=True)

    assert result.mode == BUILD_MODE_FULL
    assert result.should_persist is True
    assert result.document.metadata["database_mutation"] is False
    assert result.document.metadata["source_db"] == ".pmem/pmem.db"
    assert result.document.metadata["source_fingerprint"].startswith("sha256:")
    assert "RAW_SENTINEL" not in metadata_json
    assert str(tmp_path) not in metadata_json
    json.dumps(result.to_dict(), sort_keys=True)


def test_incremental_missing_artifact_falls_back_to_full(tmp_path) -> None:
    init_project(tmp_path, project_name="missing-artifact-demo")

    result = build_graph_incremental(tmp_path)

    assert result.mode == BUILD_MODE_INCREMENTAL_FALLBACK_FULL
    assert result.should_persist is True
    assert result.graph_changed is True
    assert result.previous_fingerprint is None
    assert "missing" in result.warnings[0]


def test_incremental_noop_reuses_existing_graph_without_rewrite(tmp_path) -> None:
    init_project(tmp_path, project_name="noop-demo")
    run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    full = build_graph_full(tmp_path)
    graph_path = default_graph_artifact_path(tmp_path)
    write_graph_document(full.document, graph_path, project_root=tmp_path)
    original_payload = graph_path.read_text(encoding="utf-8")

    result = build_graph_incremental(tmp_path)

    assert result.mode == BUILD_MODE_INCREMENTAL_NOOP
    assert result.should_persist is False
    assert result.source_changed is False
    assert result.graph_changed is False
    assert result.document.to_dict() == full.document.to_dict()
    assert graph_path.read_text(encoding="utf-8") == original_payload


def test_incremental_changed_source_uses_safe_full_rebuild(tmp_path) -> None:
    init_project(tmp_path, project_name="changed-source-demo")
    run_result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    full = build_graph_full(tmp_path)
    graph_path = default_graph_artifact_path(tmp_path)
    write_graph_document(full.document, graph_path, project_root=tmp_path)
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="ChangedSource",
        description="Changed source text",
    )

    result = build_graph_incremental(tmp_path)

    assert result.mode == BUILD_MODE_INCREMENTAL_FALLBACK_FULL
    assert result.should_persist is True
    assert result.previous_fingerprint == full.current_fingerprint
    assert result.current_fingerprint != full.current_fingerprint
    assert "full rebuild required" in result.warnings[0]


def test_incremental_unreadable_or_mismatched_artifact_falls_back_safely(tmp_path) -> None:
    init_project(tmp_path, project_name="corrupt-demo")
    graph_path = default_graph_artifact_path(tmp_path)
    graph_path.write_text("{not-json", encoding="utf-8")
    graph_path.chmod(0o600)

    corrupt_result = build_graph_incremental(tmp_path)

    assert corrupt_result.mode == BUILD_MODE_INCREMENTAL_FALLBACK_FULL
    assert "unreadable" in corrupt_result.warnings[0]

    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "graph-schema-v999",
                "method": "test",
                "nodes": [],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    graph_path.chmod(0o600)
    mismatch_result = build_graph_incremental(tmp_path)

    assert mismatch_result.mode == BUILD_MODE_INCREMENTAL_FALLBACK_FULL
    assert mismatch_result.document.schema_version == GRAPH_SCHEMA_VERSION


def test_incremental_preserves_private_file_mode_when_service_persists(tmp_path) -> None:
    init_project(tmp_path, project_name="mode-demo")
    full = build_graph_full(tmp_path)
    graph_path = default_graph_artifact_path(tmp_path)
    write_graph_document(full.document, graph_path, project_root=tmp_path)

    assert stat.S_IMODE(graph_path.stat().st_mode) == 0o600
