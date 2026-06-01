"""Confirmed failure entity model.

Failures are first-class memory records. The new DOCX semantics require a clear
source field so summary can distinguish user-confirmed research failures from
hard technical failures and promoted candidates.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pmem.domain.common import FailureSeverity, FailureSource
from pmem.domain.failure_taxonomy import normalize_tags


class Failure(BaseModel):
    """A confirmed failure linked to a run."""

    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    error_type: str
    description: str
    root_cause: str | None = None
    lesson: str | None = None
    severity: FailureSeverity = FailureSeverity.MEDIUM
    tags: list[str] = Field(default_factory=list)
    source: FailureSource = FailureSource.USER_CONFIRMED
    created_at: datetime | None = None

    @field_validator("id", "run_id", "error_type", "description")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        """Required failure fields must be non-empty strings."""

        cleaned = value.strip()
        if not cleaned:
            msg = "failure id/run_id/error_type/description cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("root_cause", "lesson")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        """Optional text may be omitted, but present values must be meaningful."""

        cleaned = value.strip() if value is not None else None
        if cleaned == "":
            msg = "failure optional text fields cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("tags", mode="before")
    @classmethod
    def tags_must_be_normalized(cls, value: list[str] | tuple[str, ...] | None) -> list[str]:
        """Normalize tags to snake_case and remove duplicates."""

        return normalize_tags(list(value or []))
