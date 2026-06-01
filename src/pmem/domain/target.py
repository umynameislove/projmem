"""Target, evaluation, and failure-candidate models.

These models are the code version of the DOCX's target/failure semantics:
`pmem` needs objective, metric direction, target, baseline, constraints, and
criteria before it can produce meaningful warnings.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pmem.domain.common import (
    EvaluationFlag,
    FailureCandidateKind,
    FailureSeverity,
    MetricDirection,
    RunStatus,
)
from pmem.domain.failure_taxonomy import normalize_tag


class BaselineReference(BaseModel):
    """Human or DB reference to the current baseline comparison point."""

    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    experiment_id: str | None = None
    run_id: str | None = None

    @field_validator("metric_value")
    @classmethod
    def metric_value_must_be_finite(cls, value: float | None) -> float | None:
        """Reject NaN/inf baselines because comparisons would be meaningless."""

        if value is not None and not math.isfinite(value):
            msg = "baseline metric_value must be finite"
            raise ValueError(msg)
        return value


class TargetSpec(BaseModel):
    """Project or experiment target stored in `target_json`."""

    model_config = ConfigDict(extra="forbid")

    target_value: float | None = None
    baseline: BaselineReference | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    owner_notes: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("target_value")
    @classmethod
    def target_value_must_be_finite(cls, value: float | None) -> float | None:
        """A target can be absent, but if present it must be comparable."""

        if value is not None and not math.isfinite(value):
            msg = "target_value must be finite"
            raise ValueError(msg)
        return value


class FailureCriterion(BaseModel):
    """Simple rule text used to create warnings/failure candidates."""

    model_config = ConfigDict(extra="forbid")

    expression: str
    candidate_kind: FailureCandidateKind | None = None
    tag: str | None = None
    severity: FailureSeverity = FailureSeverity.MEDIUM
    description: str | None = None

    @field_validator("expression")
    @classmethod
    def expression_must_not_be_blank(cls, value: str) -> str:
        """Rules are stored for review, so blank criteria are not useful."""

        cleaned = value.strip()
        if not cleaned:
            msg = "failure criterion expression cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("tag")
    @classmethod
    def tag_must_be_normalized(cls, value: str | None) -> str | None:
        """Normalize suggested tags while still allowing user-defined tags."""

        return normalize_tag(value) if value is not None else None


class RunEvaluation(BaseModel):
    """Rule-based evaluation stored in `runs.evaluation_json`."""

    model_config = ConfigDict(extra="forbid")

    technical_status: RunStatus = RunStatus.UNKNOWN
    primary_metric: str | None = None
    primary_metric_value: float | None = None
    target_value: float | None = None
    baseline_value: float | None = None
    flags: list[EvaluationFlag] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("primary_metric")
    @classmethod
    def primary_metric_must_not_be_blank(cls, value: str | None) -> str | None:
        """Metric names are used as JSON keys and should not be empty."""

        cleaned = value.strip() if value is not None else None
        if cleaned == "":
            msg = "primary_metric cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("primary_metric_value", "target_value", "baseline_value")
    @classmethod
    def numeric_values_must_be_finite(cls, value: float | None) -> float | None:
        """Reject NaN/inf values before summary comparisons."""

        if value is not None and not math.isfinite(value):
            msg = "evaluation numeric values must be finite"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def metric_value_requires_metric_name(self) -> RunEvaluation:
        """A metric value without the metric name cannot be interpreted later."""

        if self.primary_metric_value is not None and not self.primary_metric:
            msg = "primary_metric is required when primary_metric_value is set"
            raise ValueError(msg)
        return self


class FailureCandidate(BaseModel):
    """Unconfirmed failure warning stored in `runs.failure_candidates_json`."""

    model_config = ConfigDict(extra="forbid")

    kind: FailureCandidateKind
    evidence: str
    suggested_tag: str | None = None
    severity: FailureSeverity = FailureSeverity.MEDIUM
    source_rule: str | None = None
    resolved: bool = False

    @field_validator("evidence")
    @classmethod
    def evidence_must_not_be_blank(cls, value: str) -> str:
        """Candidates need evidence so the user can confirm or dismiss them."""

        cleaned = value.strip()
        if not cleaned:
            msg = "failure candidate evidence cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("suggested_tag")
    @classmethod
    def suggested_tag_must_be_normalized(cls, value: str | None) -> str | None:
        """Normalize candidate tags before they appear in summary/status."""

        return normalize_tag(value) if value is not None else None


class ProjectTarget(BaseModel):
    """Convenience model combining project-level target columns.

    In SQLite, these fields are split between typed columns and JSON columns.
    The domain model keeps them together for validation at `pmem init`.
    """

    model_config = ConfigDict(extra="forbid")

    current_objective: str | None = None
    primary_metric: str | None = None
    metric_direction: MetricDirection | None = None
    target: TargetSpec = Field(default_factory=TargetSpec)
    failure_criteria: list[FailureCriterion] = Field(default_factory=list)

    @field_validator("current_objective", "primary_metric")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        """Optional text may be absent, but present values must carry meaning."""

        cleaned = value.strip() if value is not None else None
        if cleaned == "":
            msg = "target text fields cannot be blank"
            raise ValueError(msg)
        return cleaned

    @model_validator(mode="after")
    def target_requires_metric_context(self) -> ProjectTarget:
        """A numeric target needs both metric name and direction."""

        if self.target.target_value is not None:
            if not self.primary_metric:
                msg = "primary_metric is required when target_value is set"
                raise ValueError(msg)
            if self.metric_direction is None:
                msg = "metric_direction is required when target_value is set"
                raise ValueError(msg)
        return self
