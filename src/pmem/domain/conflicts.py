"""Domain contracts for portability and failure-analysis conflict detection and resolution."""

from __future__ import annotations

from dataclasses import dataclass

SAFE_RESOLUTION_ACTIONS = frozenset({"skip", "keep-local", "manual-required", "duplicate"})
DESTRUCTIVE_RESOLUTION_ACTIONS = frozenset({"take-imported", "overwrite"})
RESOLUTION_ACTIONS = SAFE_RESOLUTION_ACTIONS | DESTRUCTIVE_RESOLUTION_ACTIONS


@dataclass(frozen=True)
class ConflictItem:
    """One privacy-preserving conflict finding."""

    conflict_id: str
    conflict_type: str
    entity_type: str
    entity_id: str
    severity: str
    message: str
    local_hash: str | None = None
    incoming_hash: str | None = None
    action_required: str = "manual-review"


@dataclass(frozen=True)
class ConflictCheckReport:
    """Result of checking a bundle before any merge or destructive write."""

    ok: bool
    bundle_path: str
    validation_ok: bool
    conflict_count: int
    conflicts: tuple[ConflictItem, ...]
    validation_errors: tuple[dict[str, str | None], ...]
    database_mutation: bool = False


@dataclass(frozen=True)
class ConflictResolutionResult:
    """Audit result for a non-destructive conflict resolution decision."""

    ok: bool
    conflict_id: str
    action: str
    event_id: str
    event_type: str
    before_hash: str | None
    after_hash: str | None
    database_mutation: str
    destructive_confirmed: bool
    hash_evidence_complete: bool
