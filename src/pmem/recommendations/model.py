"""recommendation model recommendation data model.

These models are schema contracts, not a generator. They require typed graph
entity ids and scoped confidence labels so later recommendation evidence workflow work cannot emit
recommendations with fabricated-looking evidence or overconfident wording.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pmem.domain.common import PmemStrEnum
from pmem.graph.schema import NodeType


class RecommendationType(PmemStrEnum):
    """recommendation model locked recommendation types."""

    TRY_NEXT = "try_next"
    AVOID = "avoid"
    VERIFY = "verify"
    PROMOTE = "promote"
    INVESTIGATE = "investigate"


class RecommendationConfidence(PmemStrEnum):
    """Scoped confidence labels for project-local recommendation evidence."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvidenceSource(PmemStrEnum):
    """Where a recommendation evidence item came from."""

    GRAPH = "graph"
    PATTERN = "pattern"
    RUN_METRIC = "run_metric"
    FAILURE_RECORD = "failure_record"
    HUMAN_REVIEW = "human_review"


class EvidenceItem(BaseModel):
    """One evidence link used by a recommendation.

    The model validates the entity id shape. Evidence linking is responsible for
    proving that the id exists in SQLite/graph state before generated
    recommendations are emitted.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    entity_type: NodeType
    source: EvidenceSource
    summary: str

    _NODE_ID_PREFIX_BY_TYPE: ClassVar[dict[NodeType, str]] = {
        NodeType.PROJECT: "project:",
        NodeType.EXPERIMENT: "experiment:",
        NodeType.RUN: "run:",
        NodeType.CONFIG: "config:",
        NodeType.DATASET: "dataset:",
        NodeType.METRIC: "metric:",
        NodeType.ARTIFACT: "artifact:",
        NodeType.FAILURE: "failure:",
        NodeType.DECISION: "decision:",
        NodeType.NOTE: "note:",
        NodeType.CODE_MODULE: "code:",
    }

    @field_validator("entity_id", "summary")
    @classmethod
    def text_must_be_non_blank_and_safe(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            msg = "recommendation evidence text fields cannot be blank"
            raise ValueError(msg)
        if any(ord(char) < 32 for char in cleaned):
            msg = "recommendation evidence text fields contain unsafe characters"
            raise ValueError(msg)
        if "::" in cleaned:
            msg = "recommendation evidence entity_id must be a graph node id, not an edge id"
            raise ValueError(msg)
        return cleaned

    @model_validator(mode="after")
    def entity_id_must_match_type(self) -> EvidenceItem:
        prefix = self._NODE_ID_PREFIX_BY_TYPE[self.entity_type]
        if not self.entity_id.startswith(prefix):
            msg = "recommendation evidence entity_id does not match entity_type"
            raise ValueError(msg)
        return self


class Recommendation(BaseModel):
    """recommendation model recommendation contract shared by recommendation evidence workflow."""

    model_config = ConfigDict(extra="forbid")

    recommendation_id: str
    type: RecommendationType
    title: str
    description: str
    supporting_evidence: list[EvidenceItem] = Field(min_length=1)
    opposing_evidence: list[EvidenceItem] = Field(default_factory=list)
    related_failures: list[EvidenceItem] = Field(default_factory=list)
    confidence: RecommendationConfidence
    suggested_action: str
    generated_at: datetime

    @field_validator("recommendation_id", "title", "description", "suggested_action")
    @classmethod
    def required_text_must_be_non_blank_and_safe(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            msg = "recommendation text fields cannot be blank"
            raise ValueError(msg)
        if any(ord(char) < 32 for char in cleaned):
            msg = "recommendation text fields contain unsafe characters"
            raise ValueError(msg)
        return cleaned

    @field_validator("generated_at")
    @classmethod
    def generated_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            msg = "recommendation generated_at must include a timezone"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def related_failures_must_be_failure_entities(self) -> Recommendation:
        if any(item.entity_type is not NodeType.FAILURE for item in self.related_failures):
            msg = "recommendation related_failures must reference failure entities"
            raise ValueError(msg)
        return self
