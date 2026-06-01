"""graph ingestion unit tests."""

from __future__ import annotations

import json
import sys

import pytest

from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.graph.ingestion import (
    GraphDocument,
    GraphNode,
    _GraphIngestion,
    build_graph_from_database,
    build_graph_from_project,
)
from pmem.graph.provenance import provenance
from pmem.graph.schema import (
    EdgeClass,
    EdgeType,
    NodeType,
    decision_node_id,
    failure_node_id,
    project_node_id,
)
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command


def test_graph_document_to_dict_is_deterministic() -> None:
    """GraphDocument should be stable before graph engine persistence exists."""

    document = GraphDocument(
        schema_version="graph-schema-v1",
        method="test",
        nodes=(),
        edges=(),
        counts={"nodes": 0, "edges": 0, "node_types": {}, "edge_types": {}},
        warnings=("No project rows found; graph is empty.",),
        skipped_counts={},
        metadata={"database_mutation": False},
    )

    assert document.to_dict() == document.to_dict()
    assert document.to_dict()["metadata"]["database_mutation"] is False


def test_build_graph_handles_initialized_sparse_project(tmp_path) -> None:
    """graph ingestion should not crash on sparse projects with no runs."""

    init_project(tmp_path, project_name="demo", primary_metric="accuracy")

    document = build_graph_from_project(tmp_path)
    payload = document.to_dict()

    assert payload["counts"]["node_types"] == {NodeType.PROJECT.value: 1}
    assert payload["counts"]["edges"] == 0
    assert payload["metadata"]["database_mutation"] is False


def test_build_graph_from_database_rejects_missing_database(tmp_path) -> None:
    """graph ingestion should fail closed instead of creating a missing DB implicitly."""

    missing_db = tmp_path / "missing.db"

    with pytest.raises(PmemNotFoundError):
        build_graph_from_database(missing_db)

    assert not missing_db.exists()


def test_run_without_config_skips_config_edge_gracefully(tmp_path) -> None:
    """Null config_hash should skip Config nodes and USES_CONFIG edges."""

    init_project(tmp_path, project_name="demo")
    run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    document = build_graph_from_project(tmp_path)
    payload = document.to_dict()

    assert NodeType.CONFIG.value not in payload["counts"]["node_types"]
    assert EdgeType.USES_CONFIG.value not in payload["counts"]["edge_types"]
    assert payload["skipped_counts"]["config_missing"] == 1


def test_metric_ingestion_filters_non_numeric_values(tmp_path) -> None:
    """Metric nodes should use the graph schema finite numeric policy."""

    init_project(tmp_path, project_name="demo", primary_metric="accuracy")
    metrics = {
        "accuracy": 0.91,
        "loss": 0.1,
        "is_good": True,
        "label": "ok",
        "missing": None,
    }
    write_metrics = (
        "from pathlib import Path; import json; "
        f"Path('metrics.json').write_text({json.dumps(json.dumps(metrics))}, encoding='utf-8')"
    )
    run_command(
        tmp_path,
        [sys.executable, "-c", write_metrics],
        metrics_path="metrics.json",
    )

    document = build_graph_from_project(tmp_path)
    payload = document.to_dict()
    metric_nodes = [node for node in payload["nodes"] if node["type"] == NodeType.METRIC.value]

    assert [node["attributes"]["metric_name"] for node in metric_nodes] == [
        "accuracy",
        "loss",
    ]
    assert payload["counts"]["edge_types"][EdgeType.PRODUCES_METRIC.value] == 2
    assert payload["skipped_counts"]["metric_not_numeric"] == 3


def test_ingestion_output_is_deterministic(tmp_path) -> None:
    """Same DB should produce the same in-memory document twice."""

    init_project(tmp_path, project_name="demo")
    run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    first = build_graph_from_project(tmp_path).to_dict()
    second = build_graph_from_project(tmp_path).to_dict()

    assert first == second


def test_non_direct_edge_classes_are_not_created(tmp_path) -> None:
    """Graph ingestion must drop derived/optional/deferred edges."""

    ingestion, connection = _ingestion_for_initialized_project(tmp_path)
    source = failure_node_id("failure_1")
    target = decision_node_id("decision_1")
    edge_provenance = provenance(
        source_table="failures",
        source_pk="failure_1",
        source_field="id",
        creation_rule="unit test non-graph ingestion edge",
    )

    try:
        for edge_class in (EdgeClass.DERIVED, EdgeClass.OPTIONAL, EdgeClass.DEFERRED):
            ingestion._put_edge(EdgeType.SUPPORTS, source, target, edge_class, edge_provenance)

        assert ingestion._edges == {}
        assert ingestion._skipped["edge_class_derived"] == 1
        assert ingestion._skipped["edge_class_optional"] == 1
        assert ingestion._skipped["edge_class_deferred"] == 1
    finally:
        connection.close()


def test_put_node_rejects_empty_provenance(tmp_path) -> None:
    """A graph node without provenance is invalid evidence."""

    ingestion, connection = _ingestion_for_initialized_project(tmp_path)
    node = GraphNode(
        node_id=project_node_id("project_1"),
        node_type=NodeType.PROJECT,
        attributes={},
        provenance=(),
    )

    try:
        with pytest.raises(PmemValidationError):
            ingestion._put_node(node)
    finally:
        connection.close()


def _ingestion_for_initialized_project(tmp_path):
    init_project(tmp_path, project_name="demo")
    connection = connect_database(project_database_path(tmp_path))
    ingestion = _GraphIngestion(connection)
    return ingestion, connection
