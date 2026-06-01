"""portability and failure-analysis non-destructive conflict resolution audit service."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.domain.conflicts import (
    DESTRUCTIVE_RESOLUTION_ACTIONS,
    RESOLUTION_ACTIONS,
    ConflictResolutionResult,
)
from pmem.errors import PmemValidationError
from pmem.repositories.portability import AuditEventRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.database import ensure_database
from pmem.services.project_context import require_project_context

HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def record_conflict_resolution(
    project_root: str | Path,
    *,
    conflict_id: str,
    action: str,
    before_hash: str | None = None,
    after_hash: str | None = None,
    confirm: bool = False,
) -> ConflictResolutionResult:
    """Record one operator resolution decision without overwriting local data."""

    clean_conflict_id = _validate_conflict_id(conflict_id)
    clean_action = _validate_action(action)
    before = _validate_optional_hash(before_hash, "before_hash")
    after = _validate_optional_hash(after_hash, "after_hash")
    destructive = clean_action in DESTRUCTIVE_RESOLUTION_ACTIONS
    if destructive and not confirm:
        raise PmemValidationError(
            "Destructive-looking resolution actions require --confirm and still only "
            "record audit intent."
        )

    context = require_project_context(project_root)
    ensure_database(context.root)
    timestamp = _utc_now_iso()
    event_id = f"audit_{uuid.uuid4().hex}"
    metadata = {
        "action": clean_action,
        "conflict_id": clean_conflict_id,
        "destructive_confirmed": destructive and confirm,
        "database_mutation": "audit_event_only",
        "canonical_data_mutation": False,
        "hash_evidence_complete": before is not None and after is not None,
    }
    connection = connect_database(project_database_path(context.root))
    try:
        connection.execute("BEGIN")
        event = AuditEventRepository(connection).insert(
            event_id=event_id,
            event_type="conflict.resolution_recorded",
            entity_type="conflict",
            entity_id=clean_conflict_id,
            before_hash=before,
            after_hash=after,
            actor="local",
            timestamp=timestamp,
            metadata=metadata,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        connection.close()
        raise
    finally:
        if connection:
            connection.close()

    return ConflictResolutionResult(
        ok=True,
        conflict_id=clean_conflict_id,
        action=clean_action,
        event_id=event.id,
        event_type=event.event_type,
        before_hash=before,
        after_hash=after,
        database_mutation="audit_event_only",
        destructive_confirmed=destructive and confirm,
        hash_evidence_complete=before is not None and after is not None,
    )


def conflict_resolution_result_json(result: ConflictResolutionResult) -> dict[str, Any]:
    """Return stable JSON for `pmem resolve --json`."""

    return {
        "ok": result.ok,
        "conflict_id": result.conflict_id,
        "action": result.action,
        "event_id": result.event_id,
        "event_type": result.event_type,
        "before_hash": result.before_hash,
        "after_hash": result.after_hash,
        "database_mutation": result.database_mutation,
        "destructive_confirmed": result.destructive_confirmed,
        "hash_evidence_complete": result.hash_evidence_complete,
    }


def _validate_conflict_id(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise PmemValidationError("Conflict id cannot be blank.")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", cleaned):
        raise PmemValidationError("Conflict id contains unsupported characters.")
    return cleaned


def _validate_action(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned not in RESOLUTION_ACTIONS:
        allowed = ", ".join(sorted(RESOLUTION_ACTIONS))
        raise PmemValidationError(f"Resolution action must be one of: {allowed}.")
    return cleaned


def _validate_optional_hash(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if HASH_PATTERN.fullmatch(cleaned) is None:
        raise PmemValidationError(f"{field_name} must use sha256:<64 lowercase hex chars>.")
    return cleaned


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
