"""Public ``pmem status`` payload contract (``status-v1``).

Only the schema/model layer lives here (STS-001). Service assembly, next-action
rules, and CLI are separate tasks and must not be imported from this package.
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
]
