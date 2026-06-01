"""NetworkX graph engine unit tests."""

from __future__ import annotations

from typing import Any, cast

import pytest

from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.graph.engine import GraphEngine
from pmem.graph.ingestion import GraphDocument, GraphEdge, GraphNode
from pmem.graph.provenance import provenance
from pmem.graph.schema import (
    GRAPH_SCHEMA_VERSION,
    EdgeClass,
    EdgeType,
    NodeType,
    edge_id,
    experiment_node_id,
    project_node_id,
    run_node_id,
)


def test_empty_graph_document_round_trips_through_engine() -> None:
    document = _document(nodes=(), edges=())

    engine = GraphEngine.from_document(document)
    result = engine.to_document()

    assert result.to_dict() == document.to_dict()
    assert engine.counts()["nodes"] == 0
    assert engine.counts()["edges"] == 0


def test_style_document_preserves_counts_and_order() -> None:
    document = _sample_document()

    engine = GraphEngine.from_document(document)
    first = engine.to_document().to_dict()
    second = engine.to_document().to_dict()

    assert first == second
    assert first == document.to_dict()
    assert engine.counts()["node_types"] == {
        NodeType.EXPERIMENT.value: 1,
        NodeType.PROJECT.value: 1,
        NodeType.RUN.value: 1,
    }
    assert engine.counts()["edge_types"] == {EdgeType.BELONGS_TO.value: 1}


def test_invalid_schema_version_is_rejected_by_engine() -> None:
    document = _document(nodes=(), edges=())
    bad_document = GraphDocument(
        schema_version="graph-schema-v999",
        method=document.method,
        nodes=document.nodes,
        edges=document.edges,
        counts=document.counts,
        warnings=document.warnings,
        skipped_counts=document.skipped_counts,
        metadata=document.metadata,
    )

    with pytest.raises(PmemValidationError):
        GraphEngine.from_document(bad_document)


def test_add_update_and_delete_node() -> None:
    engine = GraphEngine.from_document(_document(nodes=(), edges=()))
    node = _node(project_node_id("project_1"), NodeType.PROJECT, {"name": "demo"})

    engine.add_node(node)
    engine.update_node(node.node_id, {"status": "active"})

    updated = engine.get_node(node.node_id)
    assert updated is not None
    assert updated.attributes == {"name": "demo", "status": "active"}

    engine.delete_node(node.node_id)
    assert engine.get_node(node.node_id) is None
    assert engine.counts()["nodes"] == 0


def test_add_node_replaces_existing_node() -> None:
    node_id = project_node_id("project_1")
    first = _node(node_id, NodeType.PROJECT, {"name": "old"})
    second = _node(node_id, NodeType.PROJECT, {"name": "new", "status": "active"})
    engine = GraphEngine.from_document(_document(nodes=(first,), edges=()))

    engine.add_node(second)

    result = engine.get_node(node_id)
    assert result is not None
    assert result.attributes == {"name": "new", "status": "active"}
    assert engine.counts()["nodes"] == 1


def test_missing_node_operations_raise() -> None:
    engine = GraphEngine.from_document(_document(nodes=(), edges=()))

    assert engine.get_node("missing") is None
    with pytest.raises(PmemNotFoundError):
        engine.update_node("missing", {"status": "active"})
    with pytest.raises(PmemNotFoundError):
        engine.delete_node("missing")


def test_update_node_rejects_non_mapping_attributes() -> None:
    node = _node(project_node_id("project_1"), NodeType.PROJECT)
    engine = GraphEngine.from_document(_document(nodes=(node,), edges=()))

    with pytest.raises(PmemValidationError):
        engine.update_node(node.node_id, cast(Any, ["bad"]))


def test_add_and_delete_edge() -> None:
    project = _node(project_node_id("project_1"), NodeType.PROJECT)
    run = _node(run_node_id("run_1"), NodeType.RUN)
    edge = _edge(EdgeType.BELONGS_TO, run.node_id, project.node_id)
    engine = GraphEngine.from_document(_document(nodes=(project, run), edges=()))

    engine.add_edge(edge)
    assert engine.get_edge(edge.edge_id) == edge

    engine.delete_edge(edge.edge_id)
    assert engine.get_edge(edge.edge_id) is None
    assert engine.counts()["edges"] == 0


def test_delete_node_rebuilds_edge_index() -> None:
    project = _node(project_node_id("project_1"), NodeType.PROJECT)
    run = _node(run_node_id("run_1"), NodeType.RUN)
    edge = _edge(EdgeType.BELONGS_TO, run.node_id, project.node_id)
    engine = GraphEngine.from_document(_document(nodes=(project, run), edges=(edge,)))

    engine.delete_node(run.node_id)

    assert engine.get_edge(edge.edge_id) is None
    assert engine.counts()["edges"] == 0


def test_add_edge_replaces_existing_edge_id() -> None:
    project = _node(project_node_id("project_1"), NodeType.PROJECT)
    run = _node(run_node_id("run_1"), NodeType.RUN)
    first = _edge(EdgeType.BELONGS_TO, run.node_id, project.node_id)
    second = GraphEdge(
        edge_id=first.edge_id,
        edge_type=first.edge_type,
        source=first.source,
        target=first.target,
        edge_class=first.edge_class,
        attributes={"direct_ingestion": True, "version": 2},
        provenance=first.provenance,
    )
    engine = GraphEngine.from_document(_document(nodes=(project, run), edges=()))

    engine.add_edge(first)
    engine.add_edge(second)

    assert engine.counts()["edges"] == 1
    assert engine.get_edge(first.edge_id) == second


def test_delete_missing_edge_raises() -> None:
    engine = GraphEngine.from_document(_document(nodes=(), edges=()))

    assert engine.get_edge("missing") is None
    with pytest.raises(PmemNotFoundError):
        engine.delete_edge("missing")


def test_parallel_edges_with_different_ids_do_not_overwrite() -> None:
    source = _node(run_node_id("run_1"), NodeType.RUN)
    target = _node(experiment_node_id("exp_1"), NodeType.EXPERIMENT)
    first = _edge(EdgeType.BELONGS_TO, source.node_id, target.node_id)
    second = _edge(
        EdgeType.OBSERVED_IN,
        source.node_id,
        target.node_id,
        edge_class=EdgeClass.CONDITIONAL_DIRECT,
    )
    engine = GraphEngine.from_document(_document(nodes=(source, target), edges=()))

    engine.add_edge(first)
    engine.add_edge(second)

    assert engine.counts()["edges"] == 2
    assert engine.get_edge(first.edge_id) == first
    assert engine.get_edge(second.edge_id) == second


def test_missing_source_or_target_edge_is_rejected() -> None:
    source = _node(run_node_id("run_1"), NodeType.RUN)
    edge = _edge(EdgeType.BELONGS_TO, source.node_id, experiment_node_id("missing"))
    engine = GraphEngine.from_document(_document(nodes=(source,), edges=()))

    with pytest.raises(PmemNotFoundError):
        engine.add_edge(edge)


def test_non_d44_edge_classes_are_rejected() -> None:
    source = _node(run_node_id("run_1"), NodeType.RUN)
    target = _node(experiment_node_id("exp_1"), NodeType.EXPERIMENT)
    engine = GraphEngine.from_document(_document(nodes=(source, target), edges=()))

    for edge_class in (EdgeClass.DERIVED, EdgeClass.OPTIONAL, EdgeClass.DEFERRED):
        edge = _edge(EdgeType.SUPPORTS, source.node_id, target.node_id, edge_class=edge_class)
        with pytest.raises(PmemValidationError):
            engine.add_edge(edge)


def test_nodes_and_edges_require_provenance() -> None:
    engine = GraphEngine.from_document(_document(nodes=(), edges=()))
    node = GraphNode(
        node_id=project_node_id("project_1"),
        node_type=NodeType.PROJECT,
        attributes={},
        provenance=(),
    )

    with pytest.raises(PmemValidationError):
        engine.add_node(node)

    source = _node(run_node_id("run_1"), NodeType.RUN)
    target = _node(experiment_node_id("exp_1"), NodeType.EXPERIMENT)
    edge = GraphEdge(
        edge_id=edge_id(EdgeType.BELONGS_TO, source.node_id, target.node_id),
        edge_type=EdgeType.BELONGS_TO,
        source=source.node_id,
        target=target.node_id,
        edge_class=EdgeClass.DIRECT,
        attributes={},
        provenance=(),
    )
    engine.add_node(source)
    engine.add_node(target)
    with pytest.raises(PmemValidationError):
        engine.add_edge(edge)


def test_invalid_node_and_edge_objects_are_rejected() -> None:
    engine = GraphEngine.from_document(_document(nodes=(), edges=()))

    with pytest.raises(PmemValidationError):
        engine.add_node(cast(Any, object()))
    with pytest.raises(PmemValidationError):
        engine.add_edge(cast(Any, object()))


def test_empty_node_or_edge_ids_are_rejected() -> None:
    node = GraphNode(
        node_id="",
        node_type=NodeType.PROJECT,
        attributes={},
        provenance=(
            provenance(
                source_table="unit_nodes",
                source_pk="project_1",
                source_field="id",
                creation_rule="unit test node",
            ),
        ),
    )
    edge = GraphEdge(
        edge_id="",
        edge_type=EdgeType.BELONGS_TO,
        source="",
        target=experiment_node_id("exp_1"),
        edge_class=EdgeClass.DIRECT,
        attributes={},
        provenance=(
            provenance(
                source_table="unit_edges",
                source_pk="edge_1",
                source_field="id",
                creation_rule="unit test edge",
            ),
        ),
    )
    engine = GraphEngine.from_document(_document(nodes=(), edges=()))

    with pytest.raises(PmemValidationError):
        engine.add_node(node)
    with pytest.raises(PmemValidationError):
        engine.add_edge(edge)


def test_malformed_engine_provenance_is_rejected() -> None:
    node = _node(project_node_id("project_1"), NodeType.PROJECT)
    engine = GraphEngine.from_document(_document(nodes=(node,), edges=()))
    engine._graph.nodes[node.node_id]["provenance"] = []  # noqa: SLF001

    with pytest.raises(PmemValidationError):
        engine.get_node(node.node_id)


def test_malformed_engine_provenance_item_is_rejected() -> None:
    node = _node(project_node_id("project_1"), NodeType.PROJECT)
    engine = GraphEngine.from_document(_document(nodes=(node,), edges=()))
    engine._graph.nodes[node.node_id]["provenance"] = [object()]  # noqa: SLF001

    with pytest.raises(PmemValidationError):
        engine.get_node(node.node_id)


def test_malformed_engine_provenance_missing_source_evidence_is_rejected() -> None:
    node = _node(project_node_id("project_1"), NodeType.PROJECT)
    engine = GraphEngine.from_document(_document(nodes=(node,), edges=()))
    engine._graph.nodes[node.node_id]["provenance"] = [  # noqa: SLF001
        {
            "source_table": "unit_nodes",
            "source_pk": "",
            "source_field": "id",
            "creation_rule": "unit test node",
        }
    ]

    with pytest.raises(PmemValidationError):
        engine.get_node(node.node_id)


def test_to_document_raises_when_node_lookup_is_corrupted(monkeypatch) -> None:
    document = _sample_document()
    engine = GraphEngine.from_document(document)
    monkeypatch.setattr(engine, "get_node", lambda _node_id: None)

    with pytest.raises(PmemValidationError):
        engine.to_document()


def test_to_document_raises_on_corrupted_edge_data() -> None:
    document = _sample_document()
    engine = GraphEngine.from_document(document)
    edge = document.edges[0]
    location = engine._edge_location(edge.edge_id)  # noqa: SLF001
    assert location is not None
    source, target, key = location
    del engine._graph.edges[source, target, key]["edge_id"]  # noqa: SLF001

    with pytest.raises(PmemValidationError):
        engine.to_document()


def test_to_document_raises_on_inconsistent_edge_index() -> None:
    document = _sample_document()
    engine = GraphEngine.from_document(document)
    edge = document.edges[0]
    engine._edge_index[edge.edge_id] = ("wrong", "wrong", edge.edge_id)  # noqa: SLF001

    with pytest.raises(PmemValidationError):
        engine.to_document()


def test_provenance_is_preserved_for_nodes_and_edges() -> None:
    document = _sample_document()

    result = GraphEngine.from_document(document).to_document()

    for node in result.nodes:
        assert node.provenance
        assert (
            node.provenance
            == document.nodes[
                [item.node_id for item in document.nodes].index(node.node_id)
            ].provenance
        )
    for edge in result.edges:
        assert edge.provenance
        assert (
            edge.provenance
            == document.edges[
                [item.edge_id for item in document.edges].index(edge.edge_id)
            ].provenance
        )


def _sample_document() -> GraphDocument:
    project = _node(project_node_id("project_1"), NodeType.PROJECT)
    experiment = _node(experiment_node_id("exp_1"), NodeType.EXPERIMENT)
    run = _node(run_node_id("run_1"), NodeType.RUN)
    edge = _edge(EdgeType.BELONGS_TO, run.node_id, experiment.node_id)
    return _document(nodes=(run, project, experiment), edges=(edge,))


def _document(
    *,
    nodes: tuple[GraphNode, ...],
    edges: tuple[GraphEdge, ...],
) -> GraphDocument:
    node_types: dict[str, int] = {}
    edge_types: dict[str, int] = {}
    for node in nodes:
        node_types[node.node_type.value] = node_types.get(node.node_type.value, 0) + 1
    for edge in edges:
        edge_types[edge.edge_type.value] = edge_types.get(edge.edge_type.value, 0) + 1
    return GraphDocument(
        schema_version=GRAPH_SCHEMA_VERSION,
        method="sqlite-direct-ingestion-v1",
        nodes=nodes,
        edges=edges,
        counts={
            "nodes": len(nodes),
            "edges": len(edges),
            "node_types": dict(sorted(node_types.items())),
            "edge_types": dict(sorted(edge_types.items())),
        },
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
        attributes={"direct_ingestion": True},
        provenance=(
            provenance(
                source_table="unit_edges",
                source_pk=f"{source}->{target}",
                source_field="id",
                creation_rule="unit test edge",
            ),
        ),
    )
