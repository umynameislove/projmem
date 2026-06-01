"""graph engine graph engine and persistence integration tests."""

from __future__ import annotations

import json
import stat
import sys

from pmem.graph.engine import GraphEngine
from pmem.graph.ingestion import build_graph_from_project
from pmem.graph.persistence import (
    default_graph_artifact_path,
    read_graph_document,
    write_graph_document,
)
from pmem.graph.schema import EdgeType, NodeType
from pmem.services.decision_logging import log_decision
from pmem.services.failure_logging import log_failure
from pmem.services.note_logging import add_note
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command
from pmem.services.tracking import track_path


def test_engine_persists_private_graph_artifact_from_real_workflow(tmp_path) -> None:
    """graph data should round-trip through graph engine and graph.json."""

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"lr": 0.1}), encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 0.93}), encoding="utf-8")
    (tmp_path / "artifact.txt").write_text("artifact-data", encoding="utf-8")
    init_project(tmp_path, project_name="demo", primary_metric="accuracy")
    track_path(tmp_path, "README.md")
    run_result = run_command(
        tmp_path,
        [sys.executable, "-c", "print('ok')"],
        config_path="config.json",
        metrics_path="metrics.json",
        artifact_paths=("artifact.txt",),
    )
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="MetricRegression",
        description="PRIVATE failure description",
        root_cause="PRIVATE root cause",
        lesson="PRIVATE lesson",
        severity="high",
        tags=("data_quality",),
    )
    log_decision(
        tmp_path,
        description="PRIVATE decision description",
        rationale="PRIVATE decision rationale",
        experiment_id=run_result.record.experiment_id,
    )
    add_note(
        tmp_path,
        content="PRIVATE note content",
        run_id=run_result.record.run_id,
        tags=("follow_up",),
    )

    ingested = build_graph_from_project(tmp_path)
    engine_document = GraphEngine.from_document(ingested).to_document()
    graph_path = default_graph_artifact_path(tmp_path)
    write_graph_document(engine_document, graph_path, project_root=tmp_path)
    persisted = read_graph_document(graph_path)
    persisted_json = graph_path.read_text(encoding="utf-8")

    assert persisted.counts == engine_document.counts
    assert len(persisted.nodes) == len(engine_document.nodes)
    assert len(persisted.edges) == len(engine_document.edges)
    assert stat.S_IMODE(graph_path.stat().st_mode) == 0o600
    assert persisted.counts["node_types"][NodeType.FAILURE.value] == 1
    assert persisted.counts["edge_types"][EdgeType.OBSERVED_IN.value] == 1
    assert all(node.provenance for node in persisted.nodes)
    assert all(edge.provenance for edge in persisted.edges)
    assert "PRIVATE" not in persisted_json
    assert "artifact.txt" not in persisted_json
    assert "README.md" not in persisted_json
