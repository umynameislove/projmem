"""Shared domain vocabulary for local-memory entities.

The DOCX deliberately separates technical status, target evaluation, failure
candidates, and confirmed failures. These enums make that language explicit in
code so later services do not invent competing strings.
"""

from __future__ import annotations

from enum import Enum


class PmemStrEnum(str, Enum):
    """String enum with stable JSON-friendly values."""

    def __str__(self) -> str:
        return self.value


class ProjectStatus(PmemStrEnum):
    """Lifecycle state for a local project memory root."""

    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class ExperimentStatus(PmemStrEnum):
    """Lifecycle state for an experiment/approach inside a project."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class RunStatus(PmemStrEnum):
    """Technical command status captured immediately after `pmem run`."""

    UNKNOWN = "unknown"
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    TIMEOUT = "timeout"


class MetricDirection(PmemStrEnum):
    """Direction used to compare a metric against target and baseline."""

    MAX = "max"
    MIN = "min"


class FailureSeverity(PmemStrEnum):
    """Severity levels from the failure taxonomy spec."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FailureSource(PmemStrEnum):
    """How a confirmed failure entered the system."""

    USER_CONFIRMED = "user_confirmed"
    AUTO_TECHNICAL = "auto_technical"
    PROMOTED_CANDIDATE = "promoted_candidate"


class FailureCandidateKind(PmemStrEnum):
    """Rule-based warning types that may later become confirmed failures."""

    TECHNICAL_FAILURE = "technical_failure"
    TARGET_MISS = "target_miss"
    BASELINE_REGRESSION = "baseline_regression"
    CONSTRAINT_FAILURE = "constraint_failure"
    DATA_SEMANTIC = "data_semantic"
    RESEARCH_HYPOTHESIS = "research_hypothesis"


class EvaluationFlag(PmemStrEnum):
    """Flags stored in `runs.evaluation_json` after run evaluation."""

    TARGET_PASSED = "target_passed"
    TARGET_MISSED = "target_missed"
    BASELINE_REGRESSED = "baseline_regressed"
    CONSTRAINT_VIOLATED = "constraint_violated"
    MISSING_PRIMARY_METRIC = "missing_primary_metric"
    TECHNICAL_FAILED = "technical_failed"
