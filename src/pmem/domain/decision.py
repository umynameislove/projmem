"""Decision entity model.

Decisions record durable project choices and rationale. They can attach to a
project overall or to one experiment, matching the audited schema fix.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Decision(BaseModel):
    """A durable decision and rationale entry."""

    model_config = ConfigDict(extra="forbid")

    id: str
    project_id: str
    description: str
    experiment_id: str | None = None
    rationale: str | None = None
    related_experiments: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    author: str | None = None

    @field_validator("id", "project_id", "description")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        """Required decision fields must be non-empty strings."""

        cleaned = value.strip()
        if not cleaned:
            msg = "decision id/project_id/description cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("experiment_id", "rationale", "author")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        """Optional text fields may be absent but not blank."""

        cleaned = value.strip() if value is not None else None
        if cleaned == "":
            msg = "decision optional text fields cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("related_experiments", mode="before")
    @classmethod
    def related_experiments_must_not_be_blank(
        cls,
        value: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        """Optional related experiment ids must be meaningful when provided."""

        cleaned_values: list[str] = []
        for experiment_id in value or []:
            cleaned = str(experiment_id).strip()
            if not cleaned:
                msg = "decision related experiment ids cannot be blank"
                raise ValueError(msg)
            cleaned_values.append(cleaned)
        return cleaned_values
