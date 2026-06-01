"""recurring failure detection unit tests."""

from __future__ import annotations

import json

import pytest

from pmem.errors import PmemValidationError
from pmem.graph.ingestion import GraphDocument, GraphEdge, GraphNode
from pmem.graph.provenance import provenance
from pmem.graph.schema import (
    GRAPH_SCHEMA_VERSION,
    EdgeClass,
    EdgeType,
    NodeType,
    edge_id,
    experiment_node_id,
    failure_node_id,
    run_node_id,
)
from pmem.patterns.recurring_failures import (
    RECURRING_FAILURE_SCHEMA_VERSION,
    recurring_failure_report_from_inputs,
)
from pmem.services.failure_clustering import FAILURE_CLUSTER_SCHEMA_VERSION
from pmem.services.failure_embeddings import FAILURE_EMBEDDING_SCHEMA_VERSION

NOW = "2026-05-30T00:00:00Z"


def test_recurring_failures_groups_five_same_tag_failures() -> None:
    """Keep the seed cluster and emit a non-causal recurring candidate."""

    failure_ids = tuple(f"failure_same_{index}" for index in range(5))
    payload = recurring_failure_report_from_inputs(
        embedding_payload=_embedding_payload(failure_ids, same_vector=True),
        cluster_payload=_cluster_payload([list(failure_ids)]),
        graph_document=_graph_document(failure_ids, tag="timeout"),
        generated_at=NOW,
    )

    assert payload["schema_version"] == RECURRING_FAILURE_SCHEMA_VERSION
    assert payload["record_count"] == 5
    assert payload["cluster_count"] == 1
    assert payload["recurring_cluster_count"] == 1
    cluster = payload["clusters"][0]
    assert cluster["recurring"] is True
    assert cluster["failure_ids"] == list(failure_ids)
    assert cluster["run_ids"] == [run_node_id(f"run_{index}") for index in range(5)]
    assert cluster["claim"] == "recurring_failure_candidate_not_root_cause"
    assert "caused" not in json.dumps(payload, sort_keys=True).casefold()
    assert payload["algorithm"]["optional_nlp_dependency_required"] is False
    assert payload["algorithm"]["derived_graph_edges"] is False


def test_recurring_failures_keeps_unrelated_failures_as_singletons() -> None:
    """Unrelated failures should remain separate singleton clusters."""

    failure_ids = tuple(f"failure_unrelated_{index}" for index in range(5))
    payload = recurring_failure_report_from_inputs(
        embedding_payload=_embedding_payload(failure_ids, same_vector=False),
        cluster_payload=_cluster_payload([[failure_id] for failure_id in failure_ids]),
        graph_document=_graph_document(failure_ids, tag_prefix="tag"),
        recurrence_threshold=0.95,
        generated_at=NOW,
    )

    assert payload["cluster_count"] == 5
    assert payload["recurring_cluster_count"] == 0
    assert all(cluster["recurring"] is False for cluster in payload["clusters"])
    assert all(
        cluster["claim"] == "single_failure_no_recurrence_claim" for cluster in payload["clusters"]
    )


def test_recurring_failures_uses_graph_similarity_to_link_contextual_matches() -> None:
    """Graph context can link nearby failures even when seed clusters differ."""

    failure_ids = ("failure_a", "failure_b")
    payload = recurring_failure_report_from_inputs(
        embedding_payload=_embedding_payload(failure_ids, same_vector=False),
        cluster_payload=_cluster_payload([["failure_a"], ["failure_b"]]),
        graph_document=_graph_document(failure_ids, tag="shared_context"),
        recurrence_threshold=0.2,
        graph_weight=1.0,
        generated_at=NOW,
    )

    assert payload["cluster_count"] == 1
    assert payload["recurring_cluster_count"] == 1
    assert payload["clusters"][0]["score"]["mean_graph_similarity"] > 0.0


def test_recurring_failures_validates_inputs() -> None:
    """Malformed upstream artifacts should fail closed."""

    with pytest.raises(PmemValidationError, match="failure-embedding-v1"):
        recurring_failure_report_from_inputs(
            embedding_payload={"schema_version": "bad", "records": []},
            cluster_payload=_cluster_payload([]),
            graph_document=_graph_document(()),
        )
    with pytest.raises(PmemValidationError, match="failure-cluster-v1"):
        recurring_failure_report_from_inputs(
            embedding_payload=_embedding_payload((), same_vector=False),
            cluster_payload={"schema_version": "bad", "clusters": []},
            graph_document=_graph_document(()),
        )
    with pytest.raises(PmemValidationError, match="recurrence_threshold"):
        recurring_failure_report_from_inputs(
            embedding_payload=_embedding_payload((), same_vector=False),
            cluster_payload=_cluster_payload([]),
            graph_document=_graph_document(()),
            recurrence_threshold=2.0,
        )


def test_recurring_failures_handles_empty_input() -> None:
    """Empty projects should produce a clear no-record warning."""

    payload = recurring_failure_report_from_inputs(
        embedding_payload=_embedding_payload((), same_vector=False),
        cluster_payload=_cluster_payload([]),
        graph_document=_graph_document(()),
        generated_at=NOW,
    )

    assert payload["record_count"] == 0
    assert payload["clusters"] == []
    assert any("No confirmed failure" in warning for warning in payload["warnings"])


def _embedding_payload(failure_ids: tuple[str, ...], *, same_vector: bool) -> dict:
    records = []
    for index, failure_id in enumerate(failure_ids):
        vector = [1.0, 0.0, 0.0] if same_vector else _basis_vector(index)
        records.append(
            {
                "failure_id": failure_id,
                "run_id": f"run_{index}",
                "error_type": "Timeout" if same_vector else f"Error{index}",
                "severity": "high",
                "source": "user_confirmed",
                "tags": ["timeout"] if same_vector else [f"tag_{index}"],
                "created_at": NOW,
                "vector": vector,
                "vector_norm": 1.0,
                "source_fields": ("structured",),
                "text_included": False,
            }
        )
    return {
        "schema_version": FAILURE_EMBEDDING_SCHEMA_VERSION,
        "generated_at": NOW,
        "method": "local_hashing_tf_l2",
        "dimension": 3,
        "privacy_mode": "structured_only",
        "include_text": False,
        "record_count": len(records),
        "records": records,
        "privacy_flags": [],
        "algorithm": {"raw_text_in_output": False},
    }


def _cluster_payload(clusters: list[list[str]]) -> dict:
    return {
        "schema_version": FAILURE_CLUSTER_SCHEMA_VERSION,
        "generated_at": NOW,
        "embedding_schema_version": FAILURE_EMBEDDING_SCHEMA_VERSION,
        "embedding_method": "local_hashing_tf_l2",
        "dimension": 3,
        "similarity_threshold": 0.42,
        "privacy_mode": "structured_only",
        "include_text": False,
        "record_count": sum(len(cluster) for cluster in clusters),
        "cluster_count": len(clusters),
        "clusters": [
            {
                "cluster_id": f"cluster_{index:03d}",
                "size": len(cluster),
                "failure_ids": cluster,
                "prototype_failure_id": cluster[0] if cluster else "",
            }
            for index, cluster in enumerate(clusters, start=1)
        ],
        "points": [],
        "projection": {"raw_text_in_output": False},
        "privacy_flags": [],
    }


def _graph_document(
    failure_ids: tuple[str, ...],
    *,
    tag: str | None = None,
    tag_prefix: str | None = None,
) -> GraphDocument:
    nodes: list[GraphNode] = [
        GraphNode(
            node_id=experiment_node_id("exp"),
            node_type=NodeType.EXPERIMENT,
            attributes={"project_id": "project:p1"},
            provenance=(_prov("experiments", "exp"),),
        )
    ]
    edges: list[GraphEdge] = []
    for index, failure_id in enumerate(failure_ids):
        run_id = run_node_id(f"run_{index}")
        node_id = failure_node_id(failure_id)
        nodes.append(
            GraphNode(
                node_id=run_id,
                node_type=NodeType.RUN,
                attributes={
                    "experiment_id": experiment_node_id("exp"),
                    "status": "failed",
                    "exit_code": 1,
                    "has_config": True,
                },
                provenance=(_prov("runs", f"run_{index}"),),
            )
        )
        nodes.append(
            GraphNode(
                node_id=node_id,
                node_type=NodeType.FAILURE,
                attributes={
                    "run_id": run_id,
                    "error_type": "Timeout" if tag else f"Error{index}",
                    "severity": "high",
                    "source": "user_confirmed",
                    "tags": [tag or f"{tag_prefix}_{index}"],
                    "raw_text_included": False,
                },
                provenance=(_prov("failures", failure_id),),
            )
        )
        edges.append(
            GraphEdge(
                edge_id=edge_id(EdgeType.OBSERVED_IN, node_id, run_id),
                edge_type=EdgeType.OBSERVED_IN,
                source=node_id,
                target=run_id,
                edge_class=EdgeClass.DIRECT,
                attributes={},
                provenance=(_prov("failures", failure_id),),
            )
        )
    return GraphDocument(
        schema_version=GRAPH_SCHEMA_VERSION,
        method="test",
        nodes=tuple(nodes),
        edges=tuple(edges),
        counts={},
        warnings=(),
        skipped_counts={},
        metadata={},
    )


def _basis_vector(index: int) -> list[float]:
    vector = [0.0, 0.0, 0.0]
    vector[index % len(vector)] = 1.0
    return vector


def _prov(table: str, pk: str):
    return provenance(source_table=table, source_pk=pk, source_field="id", creation_rule="test")
