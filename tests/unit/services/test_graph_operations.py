"""graph CLI graph operation service tests."""

from __future__ import annotations

import json
import stat
import sys

import pytest

from pmem.errors import PmemSecurityError, PmemValidationError
from pmem.graph.persistence import default_graph_artifact_path
from pmem.graph.schema import EdgeType, NodeType, run_node_id
from pmem.services import graph_operations
from pmem.services.failure_logging import log_failure
from pmem.services.graph_operations import (
    build_graph_artifact,
    export_graph_artifact,
    graph_lineage_payload,
    graph_query_payload,
    graph_status_payload,
)
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command


def test_build_status_and_query_are_metadata_only(tmp_path) -> None:
    init_project(tmp_path, project_name="graph-service-demo")
    run_result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    build = build_graph_artifact(tmp_path)
    status = graph_status_payload(tmp_path)
    query = graph_query_payload(tmp_path, node_id=run_node_id(run_result.record.run_id))
    raw_json = json.dumps(query, sort_keys=True)

    assert build["ok"] is True
    assert build["graph_path"] == ".pmem/graph.json"
    assert build["counts"]["nodes"] >= 1
    assert status["exists"] is True
    assert status["counts"]["nodes"] == build["counts"]["nodes"]
    assert query["found"] is True
    assert query["node"] == {
        "id": run_node_id(run_result.record.run_id),
        "type": NodeType.RUN.value,
    }
    assert query["neighbor_count"] >= 1
    assert "python -c" not in raw_json
    assert "stdout" not in raw_json.lower()
    assert stat.S_IMODE(default_graph_artifact_path(tmp_path).stat().st_mode) == 0o600


def test_status_is_graceful_when_graph_artifact_is_missing(tmp_path) -> None:
    init_project(tmp_path, project_name="graph-status-demo")

    status = graph_status_payload(tmp_path)

    assert status["exists"] is False
    assert "pmem graph build" in status["message"]


def test_query_supports_path_and_sanitized_subgraph(tmp_path) -> None:
    init_project(tmp_path, project_name="graph-query-demo")
    run_result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    build_graph_artifact(tmp_path)

    payload = graph_query_payload(
        tmp_path,
        node_id=run_node_id(run_result.record.run_id),
        edge_type=EdgeType.BELONGS_TO.value,
        depth=1,
        path_to=run_node_id(run_result.record.run_id),
    )
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["path"]["found"] is True
    assert payload["subgraph"]["counts"]["nodes"] >= 1
    assert "attributes" not in raw_json
    assert "python -c" not in raw_json


def test_graph_lineage_payload_is_metadata_only(tmp_path) -> None:
    init_project(tmp_path, project_name="graph-lineage-demo")
    run_result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="LineagePrivacy",
        description="PRIVATE lineage text",
    )
    build_graph_artifact(tmp_path)

    payload = graph_lineage_payload(tmp_path, run_id=run_result.record.run_id)
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == "graph-lineage-result-v1"
    assert payload["database_mutation"] is False
    assert payload["lineage"]["counts"]["hops"] >= 1
    assert "PRIVATE lineage text" not in raw_json
    assert "python -c" not in raw_json


def test_graph_export_requires_confirm_and_writes_private_file(tmp_path) -> None:
    init_project(tmp_path, project_name="graph-export-demo")
    run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    build_graph_artifact(tmp_path)

    with pytest.raises(PmemValidationError):
        export_graph_artifact(tmp_path, output_path="graph-export.json", confirm=False)

    result = export_graph_artifact(tmp_path, output_path="exports/graph.json", confirm=True)
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))

    assert result.display_path == "exports/graph.json"
    assert payload["schema_version"] == "graph-export-result-v1"
    assert payload["graph"]["counts"]["nodes"] >= 1
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    assert list(result.output_path.parent.glob(".*.tmp")) == []


def test_graph_export_uses_atomic_private_replace(monkeypatch, tmp_path) -> None:
    init_project(tmp_path, project_name="graph-export-atomic-demo")
    run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    build_graph_artifact(tmp_path)
    replaced_paths: list[tuple[str, str]] = []
    original_replace = graph_operations.os.replace

    def capture_replace(source, target) -> None:
        replaced_paths.append((str(source), str(target)))
        original_replace(source, target)

    monkeypatch.setattr(graph_operations.os, "replace", capture_replace)

    result = export_graph_artifact(tmp_path, output_path="exports/graph.json", confirm=True)

    assert replaced_paths
    temp_path, final_path = replaced_paths[0]
    assert "/." in temp_path
    assert temp_path.endswith(".tmp")
    assert final_path == str(result.output_path)
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../graph.json",
        ".pmem/graph-export.json",
        "/tmp/graph.json",
        "bad\x00graph.json",
    ],
)
def test_graph_export_rejects_unsafe_paths(tmp_path, unsafe_path: str) -> None:
    init_project(tmp_path, project_name="graph-export-safety-demo")
    run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    build_graph_artifact(tmp_path)

    with pytest.raises(PmemSecurityError):
        export_graph_artifact(tmp_path, output_path=unsafe_path, confirm=True)


def test_graph_export_rejects_blank_directory_and_symlink_targets(tmp_path) -> None:
    init_project(tmp_path, project_name="graph-export-target-demo")
    run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    build_graph_artifact(tmp_path)
    (tmp_path / "already-dir").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "parent-link").symlink_to(tmp_path / "outside", target_is_directory=True)
    (tmp_path / "target.json").write_text("{}", encoding="utf-8")
    (tmp_path / "target-link.json").symlink_to(tmp_path / "target.json")

    with pytest.raises(PmemValidationError):
        export_graph_artifact(tmp_path, output_path=" ", confirm=True)
    with pytest.raises(PmemSecurityError):
        export_graph_artifact(tmp_path, output_path="already-dir", confirm=True)
    with pytest.raises(PmemSecurityError):
        export_graph_artifact(tmp_path, output_path="target-link.json", confirm=True)
    with pytest.raises(PmemSecurityError):
        export_graph_artifact(tmp_path, output_path="parent-link/graph.json", confirm=True)


def test_graph_export_rejects_symlink_parent_inside_project(tmp_path) -> None:
    init_project(tmp_path, project_name="graph-export-parent-demo")
    run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    build_graph_artifact(tmp_path)
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)

    with pytest.raises(PmemSecurityError):
        export_graph_artifact(tmp_path, output_path="link/graph.json", confirm=True)
