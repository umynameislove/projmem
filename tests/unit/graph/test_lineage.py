"""lineage graph lineage unit tests."""

from __future__ import annotations

import json

import pytest

from pmem.errors import PmemValidationError
from pmem.graph.engine import GraphEngine
from pmem.graph.ingestion import GraphDocument, GraphEdge, GraphNode
from pmem.graph.lineage import GraphLineageService
from pmem.graph.provenance import provenance
from pmem.graph.query import GraphQueryService
from pmem.graph.schema import (
    GRAPH_SCHEMA_VERSION,
    EdgeClass,
    EdgeType,
    NodeType,
    artifact_node_id,
    code_module_node_id,
    config_node_id,
    decision_node_id,
    edge_id,
    experiment_node_id,
    failure_node_id,
    metric_node_id,
    note_node_id,
    project_node_id,
    run_node_id,
)


def test_trace_run_lineage_returns_ordered_direct_context() -> None:
    service = _lineage_service(_full_lineage_document())

    lineage = service.trace_run_lineage("run_1")
    payload = lineage.to_dict()
    edge_types = [hop.edge_type for hop in lineage.hops]

    assert payload["run_node_id"] == run_node_id("run_1")
    assert edge_types == [
        None,
        EdgeType.BELONGS_TO.value,
        None,
        EdgeType.TRACKS_CODE.value,
        EdgeType.USES_CONFIG.value,
        EdgeType.PRODUCES_METRIC.value,
        EdgeType.PRODUCES_ARTIFACT.value,
        EdgeType.OBSERVED_IN.value,
        EdgeType.NOTE_ON.value,
        EdgeType.NOTE_IN_EXPERIMENT.value,
        EdgeType.DECISION_IN_EXPERIMENT.value,
        EdgeType.DECISION_IN_PROJECT.value,
    ]
    assert all(hop.provenance for hop in lineage.hops)
    assert lineage.warnings == ()


def test_trace_run_lineage_handles_sparse_graph_gracefully() -> None:
    project = _node(project_node_id("project_1"), NodeType.PROJECT)
    experiment = _node(
        experiment_node_id("exp_1"),
        NodeType.EXPERIMENT,
        {"project_id": project.node_id},
    )
    run = _node(run_node_id("run_1"), NodeType.RUN)
    document = _document(
        nodes=(project, experiment, run),
        edges=(_edge(EdgeType.BELONGS_TO, run.node_id, experiment.node_id),),
    )
    service = _lineage_service(document)

    lineage = service.trace_run_lineage(run.node_id)

    assert [hop.edge_type for hop in lineage.hops] == [None, EdgeType.BELONGS_TO.value, None]
    assert "No config node is linked to this run." in lineage.warnings
    assert "No metric nodes are linked to this run." in lineage.warnings
    assert "No failure nodes are linked to this run." in lineage.warnings


def test_trace_run_lineage_handles_run_without_experiment_context() -> None:
    run = _node(run_node_id("run_1"), NodeType.RUN)
    service = _lineage_service(_document(nodes=(run,), edges=()))

    lineage = service.trace_run_lineage("run_1")

    assert [hop.edge_type for hop in lineage.hops] == [None]
    assert "No config node is linked to this run." in lineage.warnings
    assert all(hop.direction != "attribute" for hop in lineage.hops)


def test_trace_run_lineage_warns_when_experiment_project_context_is_missing() -> None:
    experiment = _node(experiment_node_id("exp_1"), NodeType.EXPERIMENT)
    run = _node(run_node_id("run_1"), NodeType.RUN)
    document = _document(
        nodes=(experiment, run),
        edges=(_edge(EdgeType.BELONGS_TO, run.node_id, experiment.node_id),),
    )
    service = _lineage_service(document)

    lineage = service.trace_run_lineage("run_1")

    assert "Experiment context does not include a project_id attribute." in lineage.warnings
    assert all(hop.direction != "attribute" for hop in lineage.hops)


def test_trace_run_lineage_warns_when_experiment_project_node_is_missing() -> None:
    experiment = _node(
        experiment_node_id("exp_1"),
        NodeType.EXPERIMENT,
        {"project_id": project_node_id("missing_project")},
    )
    run = _node(run_node_id("run_1"), NodeType.RUN)
    document = _document(
        nodes=(experiment, run),
        edges=(_edge(EdgeType.BELONGS_TO, run.node_id, experiment.node_id),),
    )
    service = _lineage_service(document)

    lineage = service.trace_run_lineage("run_1")

    assert "Experiment project context node was not found." in lineage.warnings
    assert all(hop.direction != "attribute" for hop in lineage.hops)


def test_trace_run_lineage_handles_empty_graph() -> None:
    service = _lineage_service(_document(nodes=(), edges=()))

    lineage = service.trace_run_lineage("run_1")

    assert lineage.hops == ()
    assert lineage.warnings == ("Run node was not found.",)


def test_trace_run_lineage_unknown_and_invalid_run_id() -> None:
    service = _lineage_service(_full_lineage_document())

    missing = service.trace_run_lineage("missing")

    assert missing.hops == ()
    assert missing.warnings == ("Run node was not found.",)
    with pytest.raises(PmemValidationError):
        service.trace_run_lineage(" ")


def test_lineage_output_excludes_raw_text_and_causal_edge_names() -> None:
    service = _lineage_service(_full_lineage_document())

    payload = json.dumps(service.trace_run_lineage("run_1").to_dict(), sort_keys=True)

    assert "PRIVATE" not in payload
    assert "CAUSED_BY" not in payload
    assert "root_cause" not in payload
    assert "SUPPORTS" not in payload
    assert "CONTRADICTS" not in payload


def test_lineage_is_deterministic() -> None:
    service = _lineage_service(_full_lineage_document())

    first = service.trace_run_lineage("run_1").to_dict()
    second = service.trace_run_lineage(run_node_id("run_1")).to_dict()

    assert first == second


def _lineage_service(document: GraphDocument) -> GraphLineageService:
    return GraphLineageService(GraphQueryService(GraphEngine.from_document(document)))


def _full_lineage_document() -> GraphDocument:
    project = _node(project_node_id("project_1"), NodeType.PROJECT)
    experiment = _node(
        experiment_node_id("exp_1"),
        NodeType.EXPERIMENT,
        {"project_id": project.node_id},
    )
    run = _node(run_node_id("run_1"), NodeType.RUN, {"raw_text_included": False})
    config = _node(config_node_id("hash_1"), NodeType.CONFIG)
    metric = _node(metric_node_id("run_1", "accuracy"), NodeType.METRIC)
    artifact = _node(artifact_node_id("run_1", "artifacts/model.bin"), NodeType.ARTIFACT)
    failure = _node(failure_node_id("fail_1"), NodeType.FAILURE, {"raw_text_included": False})
    note = _node(note_node_id("note_1"), NodeType.NOTE, {"raw_text_included": False})
    decision = _node(
        decision_node_id("dec_1"),
        NodeType.DECISION,
        {"has_rationale": True, "raw_text_included": False},
    )
    code = _node(code_module_node_id("project_1", "src/train.py"), NodeType.CODE_MODULE)
    edges = (
        _edge(EdgeType.BELONGS_TO, run.node_id, experiment.node_id),
        _edge(
            EdgeType.USES_CONFIG,
            run.node_id,
            config.node_id,
            edge_class=EdgeClass.CONDITIONAL_DIRECT,
        ),
        _edge(
            EdgeType.PRODUCES_METRIC,
            run.node_id,
            metric.node_id,
            edge_class=EdgeClass.CONDITIONAL_DIRECT,
        ),
        _edge(
            EdgeType.PRODUCES_ARTIFACT,
            run.node_id,
            artifact.node_id,
            edge_class=EdgeClass.CONDITIONAL_DIRECT,
        ),
        _edge(EdgeType.OBSERVED_IN, failure.node_id, run.node_id),
        _edge(
            EdgeType.NOTE_ON,
            note.node_id,
            run.node_id,
            edge_class=EdgeClass.CONDITIONAL_DIRECT,
        ),
        _edge(
            EdgeType.NOTE_IN_EXPERIMENT,
            note.node_id,
            experiment.node_id,
            edge_class=EdgeClass.CONDITIONAL_DIRECT,
        ),
        _edge(
            EdgeType.DECISION_IN_EXPERIMENT,
            decision.node_id,
            experiment.node_id,
            edge_class=EdgeClass.CONDITIONAL_DIRECT,
        ),
        _edge(EdgeType.DECISION_IN_PROJECT, decision.node_id, project.node_id),
        _edge(
            EdgeType.TRACKS_CODE,
            project.node_id,
            code.node_id,
            edge_class=EdgeClass.CONDITIONAL_DIRECT,
        ),
    )
    return _document(
        nodes=(project, experiment, run, config, metric, artifact, failure, note, decision, code),
        edges=edges,
    )


def _document(
    *,
    nodes: tuple[GraphNode, ...],
    edges: tuple[GraphEdge, ...],
) -> GraphDocument:
    return GraphDocument(
        schema_version=GRAPH_SCHEMA_VERSION,
        method="unit-lineage",
        nodes=nodes,
        edges=edges,
        counts={"nodes": len(nodes), "edges": len(edges), "node_types": {}, "edge_types": {}},
        warnings=(),
        skipped_counts={},
        metadata={"database_mutation": False},
    )


def _node(
    node_id: str,
    node_type: NodeType,
    attributes: dict[str, object] | None = None,
) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        attributes=attributes or {},
        provenance=(
            provenance(
                source_table="unit_nodes",
                source_pk=node_id,
                source_field="id",
                creation_rule="unit test node",
            ),
        ),
    )


def _edge(
    edge_type: EdgeType,
    source: str,
    target: str,
    *,
    edge_class: EdgeClass = EdgeClass.DIRECT,
) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id(edge_type, source, target),
        edge_type=edge_type,
        source=source,
        target=target,
        edge_class=edge_class,
        attributes={"direct_lineage_test": True},
        provenance=(
            provenance(
                source_table="unit_edges",
                source_pk=f"{source}->{target}",
                source_field="id",
                creation_rule="unit test edge",
            ),
        ),
    )
