"""Project entity model.

Project is the top-level memory root. It stores both general identity and the
target context required by the new DOCX semantics: current objective, primary
metric, metric direction, target JSON, and failure criteria JSON.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pmem.domain.common import MetricDirection, ProjectStatus
from pmem.domain.target import FailureCriterion, TargetSpec


class Project(BaseModel):
    """A local `.pmem/` project memory root."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    goal: str | None = None
    current_objective: str | None = None
    status: ProjectStatus = ProjectStatus.ACTIVE
    primary_metric: str | None = None
    metric_direction: MetricDirection | None = None
    target: TargetSpec = Field(default_factory=TargetSpec)
    failure_criteria: list[FailureCriterion] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "name")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        """Required project identifiers must be non-empty strings."""

        cleaned = value.strip()
        if not cleaned:
            msg = "project id/name cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("goal", "current_objective", "primary_metric")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        """Optional text may be omitted, but present values must be meaningful."""

        cleaned = value.strip() if value is not None else None
        if cleaned == "":
            msg = "project text fields cannot be blank"
            raise ValueError(msg)
        return cleaned

    @model_validator(mode="after")
    def target_requires_metric_context(self) -> Project:
        """A project-level numeric target needs metric name and direction."""

        if self.target.target_value is not None:
            if not self.primary_metric:
                msg = "primary_metric is required when target_value is set"
                raise ValueError(msg)
            if self.metric_direction is None:
                msg = "metric_direction is required when target_value is set"
                raise ValueError(msg)
        return self
