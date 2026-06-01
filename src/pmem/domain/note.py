"""Note entity model.

Notes hold lightweight project memory, especially open questions. They may link
to a project, experiment, and run without forcing every note into one context.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pmem.domain.failure_taxonomy import normalize_tags


class Note(BaseModel):
    """A note or question captured during project work."""

    model_config = ConfigDict(extra="forbid")

    id: str
    project_id: str
    content: str
    experiment_id: str | None = None
    run_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False
    created_at: datetime | None = None

    @field_validator("id", "project_id", "content")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        """Required note fields must be non-empty strings."""

        cleaned = value.strip()
        if not cleaned:
            msg = "note id/project_id/content cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("experiment_id", "run_id")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        """Optional ids may be absent but not blank."""

        cleaned = value.strip() if value is not None else None
        if cleaned == "":
            msg = "note optional ids cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("tags", mode="before")
    @classmethod
    def tags_must_be_normalized(cls, value: list[str] | tuple[str, ...] | None) -> list[str]:
        """Normalize tags used for questions and memory filtering."""

        return normalize_tags(list(value or []))
