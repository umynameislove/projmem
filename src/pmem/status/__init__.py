"""Public ``pmem status`` contract and pure next-action policy.

The schema/model layer (STS-001) and the pure, deterministic next-action policy
(STS-003) live here. Both are dependency-light: service assembly (STS-002) and
the CLI (STS-004/005) live elsewhere and are not imported from this package.
"""

from __future__ import annotations

from pmem.status.model import (
    STATUS_SCHEMA_VERSION,
    GraphState,
    RecommendationMode,
    StatusBaseline,
    StatusBestRun,
    StatusCounts,
    StatusGraph,
    StatusMetric,
    StatusNextAction,
    StatusPayload,
    StatusProject,
    StatusRecommendations,
    StatusWarning,
    TargetStatus,
    WarningSeverity,
    WarningSource,
)
from pmem.status.next_action import select_next_action

__all__ = [
    "STATUS_SCHEMA_VERSION",
    "GraphState",
    "RecommendationMode",
    "StatusBaseline",
    "StatusBestRun",
    "StatusCounts",
    "StatusGraph",
    "StatusMetric",
    "StatusNextAction",
    "StatusPayload",
    "StatusProject",
    "StatusRecommendations",
    "StatusWarning",
    "TargetStatus",
    "WarningSeverity",
    "WarningSource",
    "select_next_action",
]
