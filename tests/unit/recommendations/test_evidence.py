"""recommendation evidence recommendation evidence-linking unit tests."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.graph.ingestion import GraphDocument, GraphNode, build_graph_from_project
from pmem.graph.provenance import provenance
from pmem.graph.schema import NodeType, failure_node_id, metric_node_id, run_node_id
from pmem.recommendations import (
    EvidenceItem,
    EvidenceSource,
    Recommendation,
    RecommendationConfidence,
    RecommendationType,
    link_recommendation_evidence_from_document,
)
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_init import init_project

NOW = datetime(2026, 5, 31, 0, 0, tzinfo=timezone.utc)
NOW_TEXT = "2026-05-31T00:00:00Z"


def test_link_recommendation_evidence_verifies_graph_and_sqlite(tmp_path) -> None:
    """recommendation evidence should link only evidence that exists in both graph and SQLite."""

    ids = _seed_project(tmp_path)
    document = build_graph_from_project(tmp_path)
    recommendation = _recommendation(
        supporting_evidence=[_evidence(ids["run_node"], NodeType.RUN)],
        opposing_evidence=[_evidence(ids["metric_node"], NodeType.METRIC)],
        related_failures=[_evidence(ids["failure_node"], NodeType.FAILURE)],
    )

    links = link_recommendation_evidence_from_document(
        project_database_path(tmp_path),
        document,
        recommendation,
    )
    payload = links.to_dict()
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == "recommendation-evidence-link-v1"
    assert payload["counts"] == {
        "supporting_evidence": 1,
        "opposing_evidence": 1,
        "related_failures": 1,
        "total_evidence": 3,
    }
    assert payload["database_mutation"] is False
    assert payload["raw_text_in_output"] is False
    assert payload["derived_graph_edges"] is False
    assert payload["supporting_evidence"][0]["sqlite_verified"] is True
    assert payload["related_failures"][0]["sqlite_sources"][0]["source_table"] == "failures"
    assert "PRIVATE" not in raw_json
    assert "python train.py" not in raw_json
    assert "attributes" not in raw_json


def test_link_recommendation_evidence_rejects_missing_graph_node(tmp_path) -> None:
    """Fabricated-looking node ids should fail if absent from the graph."""

    _seed_project(tmp_path)
    document = build_graph_from_project(tmp_path)
    recommendation = _recommendation(
        supporting_evidence=[_evidence("run:missing", NodeType.RUN)],
    )

    with pytest.raises(PmemValidationError, match="not found in graph"):
        link_recommendation_evidence_from_document(
            project_database_path(tmp_path),
            document,
            recommendation,
        )


def test_link_recommendation_evidence_rejects_missing_database(tmp_path) -> None:
    """recommendation evidence should fail closed when the SQLite evidence source is absent."""

    ids = _seed_project(tmp_path)
    document = build_graph_from_project(tmp_path)
    missing_database = tmp_path / ".pmem" / "missing.db"
    recommendation = _recommendation(
        supporting_evidence=[_evidence(ids["run_node"], NodeType.RUN)],
    )

    with pytest.raises(PmemNotFoundError, match="Project database was not found"):
        link_recommendation_evidence_from_document(
            missing_database,
            document,
            recommendation,
        )
    assert not missing_database.exists()


def test_link_recommendation_evidence_rejects_missing_sqlite_row(tmp_path) -> None:
    """A graph node alone is insufficient if its provenance row is absent."""

    _seed_project(tmp_path)
    document = _document_with_extra_node(
        build_graph_from_project(tmp_path),
        GraphNode(
            node_id=run_node_id("ghost"),
            node_type=NodeType.RUN,
            attributes={},
            provenance=(
                provenance(
                    source_table="runs",
                    source_pk="ghost",
                    source_field="run_id",
                    creation_rule="unit-test fake run",
                ),
            ),
        ),
    )
    recommendation = _recommendation(
        supporting_evidence=[_evidence(run_node_id("ghost"), NodeType.RUN)],
    )

    with pytest.raises(PmemValidationError, match="not verified in SQLite"):
        link_recommendation_evidence_from_document(
            project_database_path(tmp_path),
            document,
            recommendation,
        )


def test_link_recommendation_evidence_rejects_unsupported_provenance_table(tmp_path) -> None:
    """Evidence provenance must come from an allowlisted canonical table."""

    _seed_project(tmp_path)
    document = _document_with_extra_node(
        build_graph_from_project(tmp_path),
        GraphNode(
            node_id=run_node_id("unsafe"),
            node_type=NodeType.RUN,
            attributes={},
            provenance=(
                provenance(
                    source_table="unsupported_table",
                    source_pk="unsafe",
                    source_field="id",
                    creation_rule="unsupported table",
                ),
            ),
        ),
    )
    recommendation = _recommendation(
        supporting_evidence=[_evidence(run_node_id("unsafe"), NodeType.RUN)],
    )

    with pytest.raises(PmemValidationError, match="unsupported provenance table"):
        link_recommendation_evidence_from_document(
            project_database_path(tmp_path),
            document,
            recommendation,
        )


def test_related_failures_must_have_observed_in_graph_edge(tmp_path) -> None:
    """Related failure evidence should be tied to an observed run in the graph."""

    ids = _seed_project(tmp_path)
    document = build_graph_from_project(tmp_path)
    document_without_failure_edges = replace(
        document,
        edges=tuple(edge for edge in document.edges if edge.edge_type.value != "OBSERVED_IN"),
    )
    recommendation = _recommendation(
        supporting_evidence=[_evidence(ids["run_node"], NodeType.RUN)],
        related_failures=[_evidence(ids["failure_node"], NodeType.FAILURE)],
    )

    with pytest.raises(PmemValidationError, match="observed in a run"):
        link_recommendation_evidence_from_document(
            project_database_path(tmp_path),
            document_without_failure_edges,
            recommendation,
        )


def test_duplicate_evidence_in_one_bucket_is_rejected(tmp_path) -> None:
    """Repeated evidence weakens auditability and should fail closed."""

    ids = _seed_project(tmp_path)
    document = build_graph_from_project(tmp_path)
    run_evidence = _evidence(ids["run_node"], NodeType.RUN)
    recommendation = _recommendation(
        supporting_evidence=[run_evidence, run_evidence],
    )

    with pytest.raises(PmemValidationError, match="duplicate evidence"):
        link_recommendation_evidence_from_document(
            project_database_path(tmp_path),
            document,
            recommendation,
        )


def _seed_project(tmp_path) -> dict[str, str]:
    init_result = init_project(
        tmp_path,
        project_name="recommendation-evidence",
        primary_metric="accuracy",
    )
    connection = connect_database(project_database_path(tmp_path))
    try:
        ExperimentRepository(connection).create(
            experiment_id="exp_d58",
            project_id=init_result.project_id,
            name="d58",
            created_at=NOW_TEXT,
            updated_at=NOW_TEXT,
        )
        RunRepository(connection).create(
            run_id="run_d58",
            experiment_id="exp_d58",
            command="python train.py",
            cwd=".",
            exit_code=0,
            status="success",
            metrics={"accuracy": 0.91},
            timestamp=NOW_TEXT,
        )
        FailureRepository(connection).create(
            failure_id="failure_d58",
            run_id="run_d58",
            error_type="MetricRegression",
            description="PRIVATE failure description",
            root_cause="PRIVATE root cause",
            lesson="PRIVATE lesson",
            severity="high",
            tags=["metric"],
            source="user_confirmed",
            created_at=NOW_TEXT,
        )
    finally:
        connection.close()
    return {
        "run_node": run_node_id("run_d58"),
        "metric_node": metric_node_id("run_d58", "accuracy"),
        "failure_node": failure_node_id("failure_d58"),
    }


def _recommendation(
    *,
    supporting_evidence: list[EvidenceItem],
    opposing_evidence: list[EvidenceItem] | None = None,
    related_failures: list[EvidenceItem] | None = None,
) -> Recommendation:
    return Recommendation(
        recommendation_id="rec_d58",
        type=RecommendationType.VERIFY,
        title="Verify candidate",
        description="Review project-local evidence before acting.",
        supporting_evidence=supporting_evidence,
        opposing_evidence=opposing_evidence or [],
        related_failures=related_failures or [],
        confidence=RecommendationConfidence.LOW,
        suggested_action="Inspect linked graph and SQLite evidence.",
        generated_at=NOW,
    )


def _evidence(entity_id: str, entity_type: NodeType) -> EvidenceItem:
    return EvidenceItem(
        entity_id=entity_id,
        entity_type=entity_type,
        source=EvidenceSource.GRAPH,
        summary="metadata-only evidence summary",
    )


def _document_with_extra_node(document: GraphDocument, node: GraphNode) -> GraphDocument:
    return replace(
        document,
        nodes=tuple(sorted((*document.nodes, node), key=lambda item: item.node_id)),
    )
