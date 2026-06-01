"""graph query graph query API unit tests."""

from __future__ import annotations

import time

import pytest

from pmem.errors import PmemValidationError
from pmem.graph.engine import GraphEngine
from pmem.graph.ingestion import GraphDocument, GraphEdge, GraphNode
from pmem.graph.provenance import provenance
from pmem.graph.query import GraphQueryService
from pmem.graph.schema import (
    GRAPH_SCHEMA_VERSION,
    EdgeClass,
    EdgeType,
    NodeType,
    config_node_id,
    decision_node_id,
    edge_id,
    experiment_node_id,
    failure_node_id,
    project_node_id,
    run_node_id,
)


def test_neighbors_support_out_in_both_and_edge_filter() -> None:
    service = GraphQueryService(GraphEngine.from_document(_query_document()))
    run = run_node_id("run_1")
    experiment = experiment_node_id("exp_1")

    outgoing = service.get_neighbors(run, direction="out")
    incoming = service.get_neighbors(experiment, direction="in")
    both = service.get_neighbors(experiment, direction="both")
    project_incoming = service.get_neighbors(project_node_id("project_1"), direction="in")
    filtered = service.get_neighbors(run, edge_type=EdgeType.USES_CONFIG, direction="out")

    assert [item.node_id for item in outgoing] == [config_node_id("hash_1"), experiment]
    assert [item.node_id for item in incoming] == [run]
    assert [item.node_id for item in both] == [run]
    assert [item.node_id for item in project_incoming] == [decision_node_id("dec_1")]
    assert [item.edge_type for item in filtered] == [EdgeType.USES_CONFIG.value]
    assert all(item.provenance for item in outgoing)


def test_unknown_node_returns_empty_neighbors() -> None:
    service = GraphQueryService(GraphEngine.from_document(_query_document()))

    assert service.get_neighbors("run:missing") == ()


def test_nodes_by_type_and_path_to_dict_are_stable() -> None:
    service = GraphQueryService(GraphEngine.from_document(_query_document()))
    run = run_node_id("run_1")
    config = config_node_id("hash_1")

    assert [node.node_id for node in service.nodes_by_type(NodeType.RUN)] == [run]
    assert [node.node_id for node in service.nodes_by_type("run")] == [run]
    assert service.get_path(run, config).to_dict()["found"] is True
    with pytest.raises(PmemValidationError):
        service.nodes_by_type("unknown")


def test_invalid_direction_and_edge_type_are_rejected() -> None:
    service = GraphQueryService(GraphEngine.from_document(_query_document()))

    with pytest.raises(PmemValidationError):
        service.get_neighbors(run_node_id("run_1"), direction="sideways")
    with pytest.raises(PmemValidationError):
        service.get_neighbors(run_node_id("run_1"), edge_type="CAUSED_BY")


def test_path_search_is_deterministic_and_respects_max_depth() -> None:
    service = GraphQueryService(GraphEngine.from_document(_query_document()))
    source = failure_node_id("fail_1")
    target = config_node_id("hash_1")

    first = service.get_path(source, target)
    second = service.get_path(source, target)

    assert first == second
    assert first.found is True
    assert first.node_ids == (source, run_node_id("run_1"), target)
    assert len(first.edge_ids) == 2
    assert service.get_path(source, target, max_depth=1).found is False


def test_path_handles_no_path_unknown_node_same_node_and_bad_depth() -> None:
    service = GraphQueryService(GraphEngine.from_document(_query_document()))
    run = run_node_id("run_1")

    assert service.get_path(run, "project:missing").found is False
    assert service.get_path("run:missing", project_node_id("project_1")).found is False
    same = service.get_path(run, run)
    assert same.found is True
    assert same.node_ids == (run,)
    assert same.edge_ids == ()
    with pytest.raises(PmemValidationError):
        service.get_path(run, project_node_id("project_1"), max_depth=-1)


def test_subgraph_depth_zero_and_one_are_deterministic() -> None:
    service = GraphQueryService(GraphEngine.from_document(_query_document()))
    root = experiment_node_id("exp_1")

    depth_zero = service.get_subgraph(root, depth=0)
    first = service.get_subgraph(root, depth=1)
    second = service.get_subgraph(root, depth=1)

    assert [node.node_id for node in depth_zero.nodes] == [root]
    assert depth_zero.edges == ()
    assert first.to_dict() == second.to_dict()
    assert [node.node_id for node in first.nodes] == [
        root,
        run_node_id("run_1"),
    ]
    assert {edge.edge_type for edge in first.edges} == {EdgeType.BELONGS_TO}


def test_subgraph_filter_and_unknown_root() -> None:
    service = GraphQueryService(GraphEngine.from_document(_query_document()))
    root = run_node_id("run_1")

    filtered = service.get_subgraph(root, depth=1, edge_type=EdgeType.USES_CONFIG)
    missing = service.get_subgraph("run:missing", depth=1)

    assert [node.node_id for node in filtered.nodes] == [config_node_id("hash_1"), root]
    assert [edge.edge_type for edge in filtered.edges] == [EdgeType.USES_CONFIG]
    assert missing.nodes == ()
    assert missing.edges == ()
    assert missing.warnings == ("Root node was not found.",)
    with pytest.raises(PmemValidationError):
        service.get_subgraph(root, depth=-1)
    with pytest.raises(PmemValidationError):
        service.get_subgraph(root, edge_type="CAUSED_BY")


def test_query_does_not_mutate_graph_engine() -> None:
    engine = GraphEngine.from_document(_query_document())
    before = engine.to_document().to_dict()
    service = GraphQueryService(engine)

    service.get_neighbors(run_node_id("run_1"))
    service.get_path(run_node_id("run_1"), project_node_id("project_1"))
    service.get_subgraph(run_node_id("run_1"), depth=2)

    assert engine.to_document().to_dict() == before


def test_parallel_edges_are_returned_without_overwrite() -> None:
    source = _node(run_node_id("run_1"), NodeType.RUN)
    target = _node(experiment_node_id("exp_1"), NodeType.EXPERIMENT)
    first = _edge(EdgeType.BELONGS_TO, source.node_id, target.node_id)
    second = _edge(
        EdgeType.USES_CONFIG,
        source.node_id,
        target.node_id,
        edge_class=EdgeClass.CONDITIONAL_DIRECT,
    )
    service = GraphQueryService(
        GraphEngine.from_document(_document(nodes=(source, target), edges=(first, second)))
    )

    neighbors = service.get_neighbors(source.node_id, direction="out")

    assert [item.edge_id for item in neighbors] == sorted([first.edge_id, second.edge_id])


def test_path_uses_first_deterministic_parallel_edge_to_same_target() -> None:
    source = _node(run_node_id("run_1"), NodeType.RUN)
    target = _node(experiment_node_id("exp_1"), NodeType.EXPERIMENT)
    first = _edge(EdgeType.BELONGS_TO, source.node_id, target.node_id)
    second = _edge(
        EdgeType.USES_CONFIG,
        source.node_id,
        target.node_id,
        edge_class=EdgeClass.CONDITIONAL_DIRECT,
    )
    service = GraphQueryService(
        GraphEngine.from_document(_document(nodes=(source, target), edges=(first, second)))
    )

    path = service.get_path(source.node_id, target.node_id)

    assert path.found is True
    assert path.edge_ids == (min(first.edge_id, second.edge_id),)


def test_query_performance_on_one_thousand_nodes_is_bounded() -> None:
    experiment = _node(experiment_node_id("exp_1"), NodeType.EXPERIMENT)
    run_nodes = tuple(_node(run_node_id(f"run_{index}"), NodeType.RUN) for index in range(999))
    edges = tuple(
        _edge(EdgeType.BELONGS_TO, node.node_id, experiment.node_id) for node in run_nodes
    )
    service = GraphQueryService(
        GraphEngine.from_document(_document(nodes=(experiment, *run_nodes), edges=edges))
    )

    start = time.perf_counter()
    neighbors = service.get_neighbors(experiment.node_id, direction="in")
    path = service.get_path(run_nodes[0].node_id, run_nodes[-1].node_id)
    subgraph = service.get_subgraph(experiment.node_id, depth=1)
    elapsed = time.perf_counter() - start

    assert len(neighbors) == 999
    assert path.found is True
    assert len(subgraph.nodes) == 1000
    assert elapsed < 0.5


def _query_document() -> GraphDocument:
    project = _node(project_node_id("project_1"), NodeType.PROJECT)
    experiment = _node(
        experiment_node_id("exp_1"),
        NodeType.EXPERIMENT,
        {"project_id": project.node_id},
    )
    run = _node(run_node_id("run_1"), NodeType.RUN)
    config = _node(config_node_id("hash_1"), NodeType.CONFIG)
    decision = _node(decision_node_id("dec_1"), NodeType.DECISION)
    failure = _node(failure_node_id("fail_1"), NodeType.FAILURE)
    edges = (
        _edge(EdgeType.BELONGS_TO, run.node_id, experiment.node_id),
        _edge(
            EdgeType.USES_CONFIG,
            run.node_id,
            config.node_id,
            edge_class=EdgeClass.CONDITIONAL_DIRECT,
        ),
        _edge(EdgeType.DECISION_IN_PROJECT, decision.node_id, project.node_id),
        _edge(EdgeType.OBSERVED_IN, failure.node_id, run.node_id),
    )
    return _document(nodes=(run, config, project, experiment, decision, failure), edges=edges)


def _document(
    *,
    nodes: tuple[GraphNode, ...],
    edges: tuple[GraphEdge, ...],
) -> GraphDocument:
    return GraphDocument(
        schema_version=GRAPH_SCHEMA_VERSION,
        method="unit-query",
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
        attributes={"direct_query_test": True},
        provenance=(
            provenance(
                source_table="unit_edges",
                source_pk=f"{source}->{target}",
                source_field="id",
                creation_rule="unit test edge",
            ),
        ),
    )
