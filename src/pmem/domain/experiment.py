"""Experiment entity model.

Experiments group runs into approaches/hypotheses. They also carry baseline
state and optional target overrides, which the summary engine will need when a
specific experiment has different success criteria than the project default.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pmem.domain.common import ExperimentStatus
from pmem.domain.target import TargetSpec


class Experiment(BaseModel):
    """An approach or hypothesis within a project."""

    model_config = ConfigDict(extra="forbid")

    id: str
    project_id: str
    name: str
    hypothesis: str | None = None
    status: ExperimentStatus = ExperimentStatus.ACTIVE
    is_baseline: bool = False
    primary_metric: str | None = None
    target: TargetSpec | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "project_id", "name")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        """Required experiment identifiers must be non-empty strings."""

        cleaned = value.strip()
        if not cleaned:
            msg = "experiment id/project_id/name cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("hypothesis", "primary_metric")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        """Optional text fields may be absent but not blank."""

        cleaned = value.strip() if value is not None else None
        if cleaned == "":
            msg = "experiment text fields cannot be blank"
            raise ValueError(msg)
        return cleaned
