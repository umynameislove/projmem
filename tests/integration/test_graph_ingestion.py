"""graph ingestion integration tests."""

from __future__ import annotations

import json
import sys

from pmem.graph.ingestion import build_graph_from_project
from pmem.graph.schema import EdgeClass, EdgeType, NodeType
from pmem.services.decision_logging import log_decision
from pmem.services.failure_logging import log_failure
from pmem.services.note_logging import add_note
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command
from pmem.services.tracking import track_path


def test_ingestion_builds_direct_graph_from_real_pmem_workflow(tmp_path) -> None:
    """Real SQLite rows should become privacy-safe graph nodes and direct edges."""

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps({"lr": 0.1, "batch_size": 16}),
        encoding="utf-8",
    )
    (tmp_path / "artifact.txt").write_text("artifact-data", encoding="utf-8")

    init_project(
        tmp_path,
        project_name="demo",
        primary_metric="accuracy",
        metric_direction="max",
        target_value=0.9,
    )
    track_path(tmp_path, "README.md")
    write_metrics = (
        "from pathlib import Path; import json; "
        "Path('metrics.json').write_text("
        "json.dumps({'accuracy': 0.91, 'loss': 0.12, 'note': 'not numeric', 'flag': True}), "
        "encoding='utf-8'); "
        "print('ok')"
    )
    run_result = run_command(
        tmp_path,
        [sys.executable, "-c", write_metrics],
        seed="42",
        config_path="config.json",
        metrics_path="metrics.json",
        artifact_paths=("artifact.txt",),
    )
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="MetricRegression",
        description="SECRET failure description",
        root_cause="SECRET root cause",
        lesson="SECRET lesson",
        severity="high",
        tags=("data_quality",),
    )
    log_decision(
        tmp_path,
        description="SECRET decision description",
        rationale="SECRET decision rationale",
        experiment_id=run_result.record.experiment_id,
    )
    add_note(
        tmp_path,
        content="SECRET note content",
        run_id=run_result.record.run_id,
        tags=("follow_up",),
    )

    payload = build_graph_from_project(tmp_path).to_dict()
    node_types = payload["counts"]["node_types"]
    edge_types = payload["counts"]["edge_types"]
    raw_json = json.dumps(payload, sort_keys=True)

    assert node_types[NodeType.PROJECT.value] == 1
    assert node_types[NodeType.EXPERIMENT.value] == 1
    assert node_types[NodeType.RUN.value] == 1
    assert node_types[NodeType.CONFIG.value] == 1
    assert node_types[NodeType.METRIC.value] == 2
    assert node_types[NodeType.ARTIFACT.value] == 1
    assert node_types[NodeType.FAILURE.value] == 1
    assert node_types[NodeType.DECISION.value] == 1
    assert node_types[NodeType.NOTE.value] == 1
    assert node_types[NodeType.CODE_MODULE.value] == 1
    assert NodeType.DATASET.value not in node_types

    assert edge_types[EdgeType.BELONGS_TO.value] == 1
    assert edge_types[EdgeType.USES_CONFIG.value] == 1
    assert edge_types[EdgeType.PRODUCES_METRIC.value] == 2
    assert edge_types[EdgeType.PRODUCES_ARTIFACT.value] == 1
    assert edge_types[EdgeType.OBSERVED_IN.value] == 1
    assert edge_types[EdgeType.DECISION_IN_PROJECT.value] == 1
    assert edge_types[EdgeType.DECISION_IN_EXPERIMENT.value] == 1
    assert edge_types[EdgeType.NOTE_ON.value] == 1
    assert edge_types[EdgeType.NOTE_IN_EXPERIMENT.value] == 1
    assert edge_types[EdgeType.TRACKS_CODE.value] == 1
    assert EdgeType.SUPPORTS.value not in edge_types
    assert EdgeType.CONTRADICTS.value not in edge_types
    assert EdgeType.TRAINED_ON.value not in edge_types

    assert payload["metadata"]["database_mutation"] is False
    assert payload["metadata"]["artifact_persistence"] is False
    assert payload["skipped_counts"]["metric_not_numeric"] == 2
    assert "SECRET" not in raw_json
    assert "artifact.txt" not in raw_json
    assert "README.md" not in raw_json
    assert "python -c" not in raw_json
    assert "CAUSED_BY" not in raw_json


def test_every_node_and_edge_has_provenance(tmp_path) -> None:
    """graph ingestion must not create graph entities without source evidence."""

    init_project(tmp_path, project_name="demo")
    run_result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="ConfigError",
        description="Failure text must not be exposed.",
    )

    payload = build_graph_from_project(tmp_path).to_dict()

    for node in payload["nodes"]:
        assert node["provenance"]
        for item in node["provenance"]:
            assert {"source_table", "source_pk", "source_field", "creation_rule"} <= set(item)
    for edge in payload["edges"]:
        assert edge["provenance"]
        assert edge["edge_class"] in {
            EdgeClass.DIRECT.value,
            EdgeClass.CONDITIONAL_DIRECT.value,
        }


def test_empty_project_graph_is_graceful(tmp_path) -> None:
    """Initialized-but-empty project should build without crashing."""

    init_project(tmp_path, project_name="demo")

    payload = build_graph_from_project(tmp_path).to_dict()

    assert payload["counts"]["nodes"] == 1
    assert payload["counts"]["edges"] == 0
    assert payload["counts"]["node_types"] == {NodeType.PROJECT.value: 1}
    assert payload["warnings"] == []
