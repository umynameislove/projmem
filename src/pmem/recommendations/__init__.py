"""Recommendation contracts and generator APIs.

The model module defines data models. The evidence module links recommendation
evidence against real graph and SQLite entities. The generator module produces
conservative candidates that higher-level surfaces can render safely.
"""

from pmem.recommendations.evidence import (
    LinkedEvidence,
    RecommendationEvidenceLinks,
    link_recommendation_evidence,
    link_recommendation_evidence_from_document,
)
from pmem.recommendations.generator import generate_recommendations
from pmem.recommendations.model import (
    EvidenceItem,
    EvidenceSource,
    Recommendation,
    RecommendationConfidence,
    RecommendationType,
)

__all__ = [
    "EvidenceItem",
    "EvidenceSource",
    "LinkedEvidence",
    "Recommendation",
    "RecommendationConfidence",
    "RecommendationEvidenceLinks",
    "RecommendationType",
    "generate_recommendations",
    "link_recommendation_evidence",
    "link_recommendation_evidence_from_document",
]
