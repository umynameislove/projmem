"""Domain contract for portability and failure-analysis import dry-run reports."""

from __future__ import annotations

from dataclasses import dataclass

EXPORT_FORMAT_VERSION = "1.0"
BUNDLE_SCHEMA_VERSION = "schema-v1"

ENTITY_KEYS: tuple[str, ...] = (
    "projects",
    "experiments",
    "runs",
    "failures",
    "decisions",
    "notes",
    "tracked_paths",
)

ENTITY_ID_FIELDS: dict[str, str] = {
    "projects": "id",
    "experiments": "id",
    "runs": "run_id",
    "failures": "id",
    "decisions": "id",
    "notes": "id",
    "tracked_paths": "id",
}

FREE_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "runs": ("command", "stdout_preview", "stderr_preview"),
    "failures": ("description", "root_cause", "lesson"),
    "decisions": ("description", "rationale"),
    "notes": ("content",),
}


@dataclass(frozen=True)
class ImportValidationIssue:
    """One actionable import dry-run validation issue."""

    code: str
    message: str
    field: str | None = None


@dataclass(frozen=True)
class PrivacyReviewItem:
    """A privacy review summary for bundle fields that may contain sensitive text."""

    field: str
    count: int
    message: str


@dataclass(frozen=True)
class ConflictPreviewItem:
    """A non-mutating import conflict preview item."""

    conflict_type: str
    entity_type: str
    entity_id: str
    message: str


@dataclass(frozen=True)
class ImportDryRunReport:
    """Result of validating an import bundle without mutating SQLite."""

    ok: bool
    dry_run: bool
    bundle_path: str
    export_format_version: str | None
    schema_version: str | None
    entity_counts: dict[str, int]
    errors: tuple[ImportValidationIssue, ...]
    warnings: tuple[ImportValidationIssue, ...]
    privacy_review: tuple[PrivacyReviewItem, ...]
    conflicts: tuple[ConflictPreviewItem, ...]
    database_mutation: bool = False
