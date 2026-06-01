"""recommendation model recommendation data model tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pmem.graph.schema import NodeType
from pmem.recommendations import (
    EvidenceItem,
    EvidenceSource,
    Recommendation,
    RecommendationConfidence,
    RecommendationType,
)

NOW = datetime(2026, 5, 31, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("recommendation_type", tuple(RecommendationType))
def test_recommendation_validates_all_d57_types(recommendation_type: RecommendationType) -> None:
    """Each recommendation type should validate with real-looking evidence."""

    recommendation = _recommendation(recommendation_type)

    assert recommendation.type is recommendation_type
    assert recommendation.supporting_evidence[0].entity_id == "run:run_1"
    assert recommendation.confidence is RecommendationConfidence.MEDIUM
    assert recommendation.model_dump(mode="json")["type"] == recommendation_type.value


def test_recommendation_rejects_fabricated_or_mismatched_evidence_ids() -> None:
    """Evidence must use typed graph entity ids, not arbitrary fabricated labels."""

    with pytest.raises(ValidationError, match="does not match entity_type"):
        EvidenceItem(
            entity_id="fabricated-run-id",
            entity_type=NodeType.RUN,
            source=EvidenceSource.GRAPH,
            summary="fake id should be rejected",
        )
    with pytest.raises(ValidationError, match="does not match entity_type"):
        EvidenceItem(
            entity_id="failure:failure_1",
            entity_type=NodeType.RUN,
            source=EvidenceSource.GRAPH,
            summary="wrong type prefix should be rejected",
        )
    with pytest.raises(ValidationError, match="not an edge id"):
        EvidenceItem(
            entity_id="OBSERVED_IN::failure:failure_1::run:run_1",
            entity_type=NodeType.FAILURE,
            source=EvidenceSource.GRAPH,
            summary="edge ids are not entity ids",
        )


def test_recommendation_requires_supporting_evidence_and_timezone() -> None:
    """Prevent unsupported or non-reproducible recommendation records."""

    with pytest.raises(ValidationError, match="at least 1 item"):
        _recommendation(RecommendationType.TRY_NEXT, supporting_evidence=[])
    with pytest.raises(ValidationError, match="timezone"):
        _recommendation(RecommendationType.TRY_NEXT, generated_at=datetime(2026, 5, 31))


def test_related_failures_must_reference_failure_entities() -> None:
    """The related_failures field should not accept generic non-failure evidence."""

    with pytest.raises(ValidationError, match="related_failures"):
        _recommendation(
            RecommendationType.INVESTIGATE,
            related_failures=[
                EvidenceItem(
                    entity_id="run:run_1",
                    entity_type=NodeType.RUN,
                    source=EvidenceSource.GRAPH,
                    summary="wrong related failure type",
                )
            ],
        )

    recommendation = _recommendation(
        RecommendationType.INVESTIGATE,
        related_failures=[
            EvidenceItem(
                entity_id="failure:failure_1",
                entity_type=NodeType.FAILURE,
                source=EvidenceSource.FAILURE_RECORD,
                summary="confirmed failure evidence",
            )
        ],
    )

    assert recommendation.related_failures[0].entity_type is NodeType.FAILURE


def test_recommendation_rejects_blank_control_and_extra_fields() -> None:
    """User-facing recommendation text should be non-blank and schema-locked."""

    with pytest.raises(ValidationError, match="cannot be blank"):
        _recommendation(RecommendationType.AVOID, title=" ")
    with pytest.raises(ValidationError, match="unsafe characters"):
        _recommendation(RecommendationType.VERIFY, suggested_action="review\nnow")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Recommendation.model_validate(
            {
                "recommendation_id": "rec_extra",
                "type": RecommendationType.PROMOTE,
                "title": "Promote run",
                "description": "Candidate with supporting evidence.",
                "supporting_evidence": [_run_evidence()],
                "confidence": RecommendationConfidence.LOW,
                "suggested_action": "Review supporting run evidence.",
                "generated_at": NOW,
                "unsupported": "nope",
            }
        )


def test_confidence_levels_are_locked() -> None:
    """Confidence must be one of the scoped recommendation model labels."""

    with pytest.raises(ValidationError, match="low|medium|high"):
        Recommendation.model_validate(
            {
                "recommendation_id": "rec_bad_confidence",
                "type": RecommendationType.TRY_NEXT,
                "title": "Try next",
                "description": "Candidate with supporting evidence.",
                "supporting_evidence": [_run_evidence()],
                "confidence": "certain",
                "suggested_action": "Review supporting evidence.",
                "generated_at": NOW,
            }
        )

    assert [item.value for item in RecommendationConfidence] == ["low", "medium", "high"]


def _recommendation(
    recommendation_type: RecommendationType,
    *,
    supporting_evidence: list[EvidenceItem] | None = None,
    related_failures: list[EvidenceItem] | None = None,
    confidence: RecommendationConfidence = RecommendationConfidence.MEDIUM,
    generated_at: datetime = NOW,
    title: str = "Review candidate",
    suggested_action: str = "Inspect the supporting evidence before acting.",
) -> Recommendation:
    return Recommendation(
        recommendation_id=f"rec_{recommendation_type.value}",
        type=recommendation_type,
        title=title,
        description="Project-local recommendation candidate with scoped evidence.",
        supporting_evidence=supporting_evidence
        if supporting_evidence is not None
        else [_run_evidence()],
        opposing_evidence=[],
        related_failures=related_failures or [],
        confidence=confidence,
        suggested_action=suggested_action,
        generated_at=generated_at,
    )


def _run_evidence() -> EvidenceItem:
    return EvidenceItem(
        entity_id="run:run_1",
        entity_type=NodeType.RUN,
        source=EvidenceSource.GRAPH,
        summary="run evidence exists in graph",
    )
