"""graph engine graph persistence unit tests."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, cast

import pytest

from pmem.errors import (
    PmemNotFoundError,
    PmemPersistenceError,
    PmemSecurityError,
    PmemValidationError,
)
from pmem.graph.ingestion import GraphDocument, GraphEdge, GraphNode
from pmem.graph.persistence import (
    default_graph_artifact_path,
    read_graph_document,
    write_graph_document,
)
from pmem.graph.provenance import provenance
from pmem.graph.schema import (
    GRAPH_SCHEMA_VERSION,
    EdgeClass,
    EdgeType,
    NodeType,
    edge_id,
    experiment_node_id,
    run_node_id,
)


def test_default_graph_artifact_path_is_project_private() -> None:
    assert default_graph_artifact_path(Path("/project")) == Path("/project/.pmem/graph.json")


def test_write_creates_private_graph_file_and_round_trips(tmp_path) -> None:
    (tmp_path / ".pmem").mkdir()
    path = default_graph_artifact_path(tmp_path)
    document = _sample_document()

    write_graph_document(document, path, project_root=tmp_path)
    result = read_graph_document(path)

    assert result.to_dict() == document.to_dict()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list((tmp_path / ".pmem").glob(".*.tmp")) == []


def test_write_allows_default_shape_without_project_root(tmp_path) -> None:
    (tmp_path / ".pmem").mkdir()
    path = default_graph_artifact_path(tmp_path)

    write_graph_document(_sample_document(), path)

    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_write_rejects_custom_path_without_project_root(tmp_path) -> None:
    with pytest.raises(PmemSecurityError):
        write_graph_document(_sample_document(), tmp_path / "graph.json")


def test_write_rejects_missing_graph_directory(tmp_path) -> None:
    with pytest.raises(PmemNotFoundError):
        write_graph_document(_sample_document(), default_graph_artifact_path(tmp_path))


def test_write_rejects_non_document_object(tmp_path) -> None:
    (tmp_path / ".pmem").mkdir()

    with pytest.raises(PmemValidationError):
        write_graph_document(cast(Any, {}), default_graph_artifact_path(tmp_path))


def test_write_rejects_invalid_document_schema(tmp_path) -> None:
    (tmp_path / ".pmem").mkdir()
    document = _sample_document()
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
        write_graph_document(bad_document, default_graph_artifact_path(tmp_path))


def test_write_converts_os_errors_to_safe_persistence_error(tmp_path) -> None:
    (tmp_path / ".pmem").mkdir()
    target_dir = default_graph_artifact_path(tmp_path)
    target_dir.mkdir()

    with pytest.raises(PmemPersistenceError):
        write_graph_document(_sample_document(), target_dir)


def test_missing_graph_artifact_is_rejected(tmp_path) -> None:
    with pytest.raises(PmemNotFoundError):
        read_graph_document(tmp_path / "missing.json")


def test_read_os_error_is_safe(tmp_path) -> None:
    graph_dir = tmp_path / "graph.json"
    graph_dir.mkdir()

    with pytest.raises(PmemPersistenceError):
        read_graph_document(graph_dir)


def test_invalid_json_is_rejected(tmp_path) -> None:
    path = tmp_path / "graph.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(PmemValidationError):
        read_graph_document(path)


def test_invalid_schema_version_is_rejected(tmp_path) -> None:
    path = tmp_path / "graph.json"
    payload = _sample_document().to_dict()
    payload["schema_version"] = "graph-schema-v999"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PmemValidationError):
        read_graph_document(path)


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        (lambda payload: [], PmemValidationError),
        (lambda payload: {**payload, "nodes": {}}, PmemValidationError),
        (lambda payload: {**payload, "counts": []}, PmemValidationError),
        (lambda payload: {**payload, "metadata": []}, PmemValidationError),
        (lambda payload: {**payload, "skipped_counts": []}, PmemValidationError),
        (lambda payload: {**payload, "skipped_counts": {"bad": True}}, PmemValidationError),
        (lambda payload: {**payload, "warnings": [1]}, PmemValidationError),
        (lambda payload: {**payload, "nodes": [1]}, PmemValidationError),
        (
            lambda payload: {
                **payload,
                "nodes": [{**payload["nodes"][0], "attributes": []}],
            },
            PmemValidationError,
        ),
        (
            lambda payload: {**payload, "nodes": [{**payload["nodes"][0], "type": "bad"}]},
            PmemValidationError,
        ),
        (lambda payload: {**payload, "edges": [1]}, PmemValidationError),
        (
            lambda payload: {
                **payload,
                "edges": [{**payload["edges"][0], "attributes": []}],
            },
            PmemValidationError,
        ),
        (
            lambda payload: {**payload, "edges": [{**payload["edges"][0], "type": "bad"}]},
            PmemValidationError,
        ),
        (
            lambda payload: {
                **payload,
                "nodes": [{**payload["nodes"][0], "provenance": {}}],
            },
            PmemValidationError,
        ),
        (
            lambda payload: {
                **payload,
                "nodes": [{**payload["nodes"][0], "provenance": [1]}],
            },
            PmemValidationError,
        ),
        (
            lambda payload: {
                **payload,
                "nodes": [
                    {
                        **payload["nodes"][0],
                        "provenance": [
                            {
                                "source_table": "runs",
                                "source_pk": "run_1",
                                "source_field": "",
                                "creation_rule": "bad",
                            }
                        ],
                    }
                ],
            },
            PmemValidationError,
        ),
    ],
)
def test_malformed_payload_shapes_are_rejected(tmp_path, mutator, expected_error) -> None:
    path = tmp_path / "graph.json"
    payload = mutator(_sample_document().to_dict())
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(expected_error):
        read_graph_document(path)


def test_missing_provenance_is_rejected(tmp_path) -> None:
    path = tmp_path / "graph.json"
    payload = _sample_document().to_dict()
    payload["nodes"][0]["provenance"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PmemValidationError):
        read_graph_document(path)


def test_malformed_edge_is_rejected(tmp_path) -> None:
    path = tmp_path / "graph.json"
    payload = _sample_document().to_dict()
    del payload["edges"][0]["target"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PmemValidationError):
        read_graph_document(path)


def test_edge_with_missing_endpoint_is_rejected(tmp_path) -> None:
    path = tmp_path / "graph.json"
    payload = _sample_document().to_dict()
    payload["edges"][0]["target"] = experiment_node_id("missing")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PmemValidationError):
        read_graph_document(path)


def test_document_with_non_node_or_edge_objects_is_rejected(tmp_path) -> None:
    (tmp_path / ".pmem").mkdir()
    document = _sample_document()
    bad_node_document = GraphDocument(
        schema_version=document.schema_version,
        method=document.method,
        nodes=(cast(Any, object()),),
        edges=(),
        counts=document.counts,
        warnings=document.warnings,
        skipped_counts=document.skipped_counts,
        metadata=document.metadata,
    )
    bad_edge_document = GraphDocument(
        schema_version=document.schema_version,
        method=document.method,
        nodes=document.nodes,
        edges=(cast(Any, object()),),
        counts=document.counts,
        warnings=document.warnings,
        skipped_counts=document.skipped_counts,
        metadata=document.metadata,
    )

    with pytest.raises(PmemValidationError):
        write_graph_document(bad_node_document, default_graph_artifact_path(tmp_path))
    with pytest.raises(PmemValidationError):
        write_graph_document(bad_edge_document, default_graph_artifact_path(tmp_path))


def test_document_with_empty_node_or_edge_evidence_is_rejected(tmp_path) -> None:
    (tmp_path / ".pmem").mkdir()
    document = _sample_document()
    bad_node = GraphNode(
        node_id="",
        node_type=NodeType.RUN,
        attributes={},
        provenance=document.nodes[0].provenance,
    )
    bad_edge = GraphEdge(
        edge_id="",
        edge_type=EdgeType.BELONGS_TO,
        source=document.edges[0].source,
        target=document.edges[0].target,
        edge_class=EdgeClass.DIRECT,
        attributes={},
        provenance=document.edges[0].provenance,
    )

    with pytest.raises(PmemValidationError):
        write_graph_document(
            GraphDocument(
                schema_version=document.schema_version,
                method=document.method,
                nodes=(bad_node,),
                edges=(),
                counts=document.counts,
                warnings=document.warnings,
                skipped_counts=document.skipped_counts,
                metadata=document.metadata,
            ),
            default_graph_artifact_path(tmp_path),
        )
    with pytest.raises(PmemValidationError):
        write_graph_document(
            GraphDocument(
                schema_version=document.schema_version,
                method=document.method,
                nodes=document.nodes,
                edges=(bad_edge,),
                counts=document.counts,
                warnings=document.warnings,
                skipped_counts=document.skipped_counts,
                metadata=document.metadata,
            ),
            default_graph_artifact_path(tmp_path),
        )


def test_custom_path_outside_graph_artifact_is_rejected(tmp_path) -> None:
    (tmp_path / ".pmem").mkdir()
    document = _sample_document()

    with pytest.raises(PmemSecurityError):
        write_graph_document(document, tmp_path / "graph.json", project_root=tmp_path)


def test_traversal_graph_artifact_path_is_rejected(tmp_path) -> None:
    unsafe = tmp_path / ".pmem" / ".." / ".pmem" / "graph.json"

    with pytest.raises(PmemSecurityError):
        write_graph_document(_sample_document(), unsafe)


def test_symlink_graph_artifact_path_is_rejected(tmp_path) -> None:
    (tmp_path / ".pmem").mkdir()
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    graph_path = default_graph_artifact_path(tmp_path)
    graph_path.symlink_to(target)

    with pytest.raises(PmemSecurityError):
        write_graph_document(_sample_document(), graph_path, project_root=tmp_path)


def test_persisted_json_is_deterministic_and_has_no_raw_text_sentinel(tmp_path) -> None:
    (tmp_path / ".pmem").mkdir()
    path = default_graph_artifact_path(tmp_path)
    document = _sample_document()

    write_graph_document(document, path, project_root=tmp_path)
    first = path.read_text(encoding="utf-8")
    write_graph_document(document, path, project_root=tmp_path)
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert "PRIVATE_FAILURE_TEXT" not in first


def _sample_document() -> GraphDocument:
    run = GraphNode(
        node_id=run_node_id("run_1"),
        node_type=NodeType.RUN,
        attributes={"status": "success", "raw_text_included": False},
        provenance=(
            provenance(
                source_table="runs",
                source_pk="run_1",
                source_field="run_id",
                creation_rule="unit test run",
            ),
        ),
    )
    experiment = GraphNode(
        node_id=experiment_node_id("exp_1"),
        node_type=NodeType.EXPERIMENT,
        attributes={"status": "active"},
        provenance=(
            provenance(
                source_table="experiments",
                source_pk="exp_1",
                source_field="id",
                creation_rule="unit test experiment",
            ),
        ),
    )
    belongs_to = GraphEdge(
        edge_id=edge_id(EdgeType.BELONGS_TO, run.node_id, experiment.node_id),
        edge_type=EdgeType.BELONGS_TO,
        source=run.node_id,
        target=experiment.node_id,
        edge_class=EdgeClass.DIRECT,
        attributes={"direct_ingestion": True},
        provenance=(
            provenance(
                source_table="runs",
                source_pk="run_1",
                source_field="experiment_id",
                creation_rule="unit test edge",
            ),
        ),
    )
    return GraphDocument(
        schema_version=GRAPH_SCHEMA_VERSION,
        method="sqlite-direct-ingestion-v1",
        nodes=(run, experiment),
        edges=(belongs_to,),
        counts={
            "nodes": 2,
            "edges": 1,
            "node_types": {
                NodeType.EXPERIMENT.value: 1,
                NodeType.RUN.value: 1,
            },
            "edge_types": {EdgeType.BELONGS_TO.value: 1},
        },
        warnings=(),
        skipped_counts={},
        metadata={"database_mutation": False},
    )
