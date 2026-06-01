"""recommendation evidence recommendation evidence-linking integration tests."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pytest

from pmem.errors import PmemValidationError
from pmem.graph.schema import NodeType, failure_node_id, run_node_id
from pmem.recommendations import (
    EvidenceItem,
    EvidenceSource,
    Recommendation,
    RecommendationConfidence,
    RecommendationType,
    link_recommendation_evidence,
)
from pmem.services.failure_logging import log_failure
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command

NOW = datetime(2026, 5, 31, 0, 0, tzinfo=timezone.utc)


def test_links_recommendation_evidence_to_real_graph_and_sqlite_entities(tmp_path) -> None:
    """Verify recommendation evidence without exposing raw private text."""

    init_project(tmp_path, project_name="d58-integration", primary_metric="accuracy")
    (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 0.88}), encoding="utf-8")
    run_result = run_command(
        tmp_path,
        [sys.executable, "-c", "print('PRIVATE command output')"],
        metrics_path="metrics.json",
    )
    failure = log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="MetricRegression",
        description="PRIVATE failure text",
        root_cause="PRIVATE root cause",
        lesson="PRIVATE lesson",
        severity="high",
    )
    recommendation = _recommendation(
        run_id=run_result.record.run_id,
        failure_id=failure.id,
    )

    links = link_recommendation_evidence(tmp_path, recommendation)
    payload = links.to_dict()
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == "recommendation-evidence-link-v1"
    assert payload["recommendation_id"] == "rec_d58_integration"
    assert payload["supporting_evidence"][0]["entity_id"] == run_node_id(run_result.record.run_id)
    assert payload["related_failures"][0]["entity_id"] == failure_node_id(failure.id)
    assert payload["supporting_evidence"][0]["sqlite_sources"][0]["source_table"] == "runs"
    assert payload["related_failures"][0]["sqlite_sources"][0]["source_table"] == "failures"
    assert payload["database_mutation"] is False
    assert payload["raw_text_in_output"] is False
    assert "PRIVATE" not in raw_json
    assert "command output" not in raw_json
    assert "attributes" not in raw_json


def test_rejects_recommendation_with_fabricated_graph_entity(tmp_path) -> None:
    """Reject ids that only look valid but are absent from graph/SQLite."""

    init_project(tmp_path, project_name="d58-missing")
    recommendation = _recommendation(run_id="missing", failure_id="missing_failure")

    with pytest.raises(PmemValidationError, match="not found in graph"):
        link_recommendation_evidence(tmp_path, recommendation)


def _recommendation(*, run_id: str, failure_id: str) -> Recommendation:
    return Recommendation(
        recommendation_id="rec_d58_integration",
        type=RecommendationType.INVESTIGATE,
        title="Investigate observed failure",
        description="Review verified project-local evidence before taking action.",
        supporting_evidence=[
            EvidenceItem(
                entity_id=run_node_id(run_id),
                entity_type=NodeType.RUN,
                source=EvidenceSource.GRAPH,
                summary="linked run metadata",
            )
        ],
        opposing_evidence=[],
        related_failures=[
            EvidenceItem(
                entity_id=failure_node_id(failure_id),
                entity_type=NodeType.FAILURE,
                source=EvidenceSource.FAILURE_RECORD,
                summary="linked failure metadata",
            )
        ],
        confidence=RecommendationConfidence.LOW,
        suggested_action="Inspect the linked run and failure evidence.",
        generated_at=NOW,
    )
