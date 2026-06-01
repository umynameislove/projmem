"""incremental graph build graph incremental update integration tests."""

from __future__ import annotations

import json
import stat
import sys

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.graph.incremental import BUILD_MODE_INCREMENTAL_FALLBACK_FULL, build_graph_full
from pmem.graph.persistence import default_graph_artifact_path, read_graph_document
from pmem.services.failure_logging import log_failure
from pmem.services.graph_operations import build_graph_artifact
from pmem.services.note_logging import add_note
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command

runner = CliRunner()


def test_empty_project_full_then_incremental_noop_is_graceful(tmp_path) -> None:
    init_project(tmp_path, project_name="d48-empty")

    full = build_graph_artifact(tmp_path)
    incremental = build_graph_artifact(tmp_path, incremental=True)

    assert full["mode"] == "full"
    assert incremental["mode"] == "incremental_noop"
    assert incremental["source_changed"] is False
    assert incremental["graph_changed"] is False
    assert full["counts"] == incremental["counts"]
    assert stat.S_IMODE(default_graph_artifact_path(tmp_path).stat().st_mode) == 0o600


def test_incremental_changed_data_matches_fresh_full_rebuild(tmp_path) -> None:
    init_project(tmp_path, project_name="d48-changed", primary_metric="accuracy")
    run_result = _seed_private_project(tmp_path)
    initial = build_graph_artifact(tmp_path)
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="incremental graph buildChanged",
        description="RAW_SENTINEL changed failure",
    )
    add_note(tmp_path, content="RAW_SENTINEL changed note", run_id=run_result.record.run_id)

    incremental = build_graph_artifact(tmp_path, incremental=True)
    fresh = build_graph_full(tmp_path)
    persisted = read_graph_document(default_graph_artifact_path(tmp_path))

    assert incremental["mode"] == BUILD_MODE_INCREMENTAL_FALLBACK_FULL
    assert incremental["source_changed"] is True
    assert incremental["graph_changed"] is True
    assert incremental["previous_source_fingerprint"] == initial["source_fingerprint"]
    assert incremental["counts"] == fresh.document.counts
    assert persisted.counts == fresh.document.counts


def test_corrupted_artifact_rebuilds_safely_without_raw_text(tmp_path) -> None:
    init_project(tmp_path, project_name="d48-corrupt", primary_metric="accuracy")
    _seed_private_project(tmp_path)
    graph_path = default_graph_artifact_path(tmp_path)
    graph_path.write_text("{not-json", encoding="utf-8")
    graph_path.chmod(0o600)

    result = build_graph_artifact(tmp_path, incremental=True)
    persisted_json = graph_path.read_text(encoding="utf-8")

    assert result["mode"] == BUILD_MODE_INCREMENTAL_FALLBACK_FULL
    assert "unreadable" in result["warnings"][0]
    assert stat.S_IMODE(graph_path.stat().st_mode) == 0o600
    assert "RAW_SENTINEL" not in persisted_json
    assert "artifact.txt" not in persisted_json
    assert str(tmp_path) not in persisted_json


def test_cli_incremental_json_and_status_include_freshness_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    init_project(tmp_path, project_name="d48-cli", primary_metric="accuracy")
    _seed_private_project(tmp_path)

    first = runner.invoke(app, ["graph", "build", "--json"])
    second = runner.invoke(app, ["graph", "build", "--incremental", "--json"])
    status = runner.invoke(app, ["graph", "status", "--json"])
    first_payload = json.loads(first.stdout)
    second_payload = json.loads(second.stdout)
    status_payload = json.loads(status.stdout)

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert status.exit_code == 0
    assert first_payload["mode"] == "full"
    assert second_payload["mode"] == "incremental_noop"
    assert second_payload["persisted"] is False
    assert second_payload["source_fingerprint"] == first_payload["source_fingerprint"]
    assert status_payload["build_mode"] == "full"
    assert status_payload["source_fingerprint_prefix"].startswith("sha256:")
    assert "RAW_SENTINEL" not in json.dumps(second_payload, sort_keys=True)


def _seed_private_project(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 0.94}), encoding="utf-8")
    (tmp_path / "artifact.txt").write_text("artifact-data", encoding="utf-8")
    run_result = run_command(
        tmp_path,
        [sys.executable, "-c", "print('RAW_SENTINEL')"],
        metrics_path="metrics.json",
        artifact_paths=("artifact.txt",),
    )
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="incremental graph buildPrivacy",
        description="RAW_SENTINEL failure",
        root_cause="RAW_SENTINEL cause",
        lesson="RAW_SENTINEL lesson",
    )
    add_note(tmp_path, content="RAW_SENTINEL note", run_id=run_result.record.run_id)
    return run_result
