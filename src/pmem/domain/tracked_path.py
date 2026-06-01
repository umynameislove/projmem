"""Tracked path entity model.

Tracked paths let `pmem track` connect files/directories to later runs. Hashes
must be SHA-256 so reproducibility metadata artifact/config lineage remains consistent.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from pmem.domain.failure_taxonomy import normalize_tag


class TrackedPath(BaseModel):
    """A file or directory tracked as run context."""

    model_config = ConfigDict(extra="forbid")

    id: str
    project_id: str
    path: str
    hash: str
    tag: str | None = None
    size_bytes: int | None = None
    last_checked: datetime | None = None
    created_at: datetime | None = None

    @field_validator("id", "project_id", "path", "hash")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        """Required tracked-path fields must be non-empty strings."""

        cleaned = value.strip()
        if not cleaned:
            msg = "tracked path id/project_id/path/hash cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("tag")
    @classmethod
    def tag_must_be_normalized(cls, value: str | None) -> str | None:
        """Normalize optional tracking tags."""

        return normalize_tag(value) if value is not None else None

    @field_validator("size_bytes")
    @classmethod
    def size_bytes_must_not_be_negative(cls, value: int | None) -> int | None:
        """File sizes can be unknown but cannot be negative."""

        if value is not None and value < 0:
            msg = "size_bytes cannot be negative"
            raise ValueError(msg)
        return value
