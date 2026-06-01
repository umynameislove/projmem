"""graph query and lineage graph query and lineage integration tests."""

from __future__ import annotations

import json
import sys

from pmem.graph.engine import GraphEngine
from pmem.graph.ingestion import build_graph_from_project
from pmem.graph.lineage import GraphLineageService
from pmem.graph.query import GraphQueryService
from pmem.graph.schema import EdgeType, NodeType, run_node_id
from pmem.services.decision_logging import log_decision
from pmem.services.failure_logging import log_failure
from pmem.services.note_logging import add_note
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command
from pmem.services.tracking import track_path


def test_query_and_lineage_from_real_pmem_workflow(tmp_path) -> None:
    """Query graph data without raw text or mutation."""

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"lr": 0.1}), encoding="utf-8")
    (tmp_path / "artifact.txt").write_text("artifact-data", encoding="utf-8")
    init_project(tmp_path, project_name="demo", primary_metric="accuracy")
    track_path(tmp_path, "README.md")
    write_metrics = (
        "from pathlib import Path; import json; "
        "Path('metrics.json').write_text(json.dumps({'accuracy': 0.93}), encoding='utf-8'); "
        "print('ok')"
    )
    run_result = run_command(
        tmp_path,
        [sys.executable, "-c", write_metrics],
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

    engine = GraphEngine.from_document(build_graph_from_project(tmp_path))
    before = engine.to_document().to_dict()
    query = GraphQueryService(engine)
    lineage_service = GraphLineageService(query)
    run_node = run_node_id(run_result.record.run_id)

    outgoing = query.get_neighbors(run_node, direction="out")
    failures = query.get_neighbors(run_node, edge_type=EdgeType.OBSERVED_IN, direction="in")
    subgraph = query.get_subgraph(run_node, depth=1)
    lineage = lineage_service.trace_run_lineage(run_result.record.run_id)
    payload = json.dumps(
        {
            "outgoing": [item.to_dict() for item in outgoing],
            "failures": [item.to_dict() for item in failures],
            "subgraph": subgraph.to_dict(),
            "lineage": lineage.to_dict(),
        },
        sort_keys=True,
    )
    lineage_edge_types = {hop.edge_type for hop in lineage.hops if hop.edge_type is not None}
    subgraph_payload = json.dumps(subgraph.to_dict(), sort_keys=True)

    assert {item.edge_type for item in outgoing} >= {
        EdgeType.BELONGS_TO.value,
        EdgeType.USES_CONFIG.value,
        EdgeType.PRODUCES_METRIC.value,
        EdgeType.PRODUCES_ARTIFACT.value,
    }
    assert [item.node_type for item in failures] == [NodeType.FAILURE.value]
    assert subgraph.nodes
    assert subgraph.edges
    assert {
        EdgeType.BELONGS_TO.value,
        EdgeType.USES_CONFIG.value,
        EdgeType.PRODUCES_METRIC.value,
        EdgeType.PRODUCES_ARTIFACT.value,
        EdgeType.OBSERVED_IN.value,
        EdgeType.NOTE_ON.value,
    } <= lineage_edge_types
    assert EdgeType.SUPPORTS.value not in lineage_edge_types
    assert EdgeType.CONTRADICTS.value not in lineage_edge_types
    assert all(hop.provenance for hop in lineage.hops)
    assert "PRIVATE" not in payload
    assert "PRIVATE" not in subgraph_payload
    assert "artifact.txt" not in payload
    assert "artifact.txt" not in subgraph_payload
    assert "README.md" not in payload
    assert "README.md" not in subgraph_payload
    assert "python -c" not in payload
    assert "CAUSED_BY" not in payload
    assert engine.to_document().to_dict() == before
