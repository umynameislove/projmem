"""portability and failure-analysis shared memory path registration and status checks."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.errors import PmemConflictError, PmemSecurityError, PmemValidationError
from pmem.repositories.portability import SharedPathRecord, SharedPathRepository
from pmem.repositories.sqlite import PMEM_DIRNAME, connect_database, project_database_path
from pmem.services.database import ensure_database
from pmem.services.project_context import require_project_context

SHARED_PATH_MODES = frozenset({"read", "write", "read_write"})


@dataclass(frozen=True)
class SharedPathStatus:
    """User-safe status for one registered shared memory path."""

    id: str
    alias: str
    mode: str
    path_display: str
    status: str
    readable: bool
    writable: bool
    message: str
    last_checked_at: str | None


@dataclass(frozen=True)
class SharedPathRegistrationResult:
    """Result of registering one shared path."""

    record: SharedPathRecord
    status: SharedPathStatus


def register_shared_path(
    project_root: str | Path,
    user_path: str | Path,
    *,
    alias: str | None = None,
    mode: str = "read_write",
) -> SharedPathRegistrationResult:
    """Register a user-approved local shared memory path."""

    context = require_project_context(project_root)
    ensure_database(context.root)
    clean_mode = _validate_mode(mode)
    resolved_path = _resolve_shared_path(context.root, user_path)
    clean_alias = _normalize_alias(alias, resolved_path)
    created_at = _utc_now_iso()
    connection = connect_database(project_database_path(context.root))
    try:
        repository = SharedPathRepository(connection)
        if repository.get_by_alias(clean_alias) is not None:
            raise PmemConflictError("Shared path alias already exists.")
        if repository.get_by_path(resolved_path.as_posix()) is not None:
            raise PmemConflictError("Shared path is already registered.")
        record = repository.create(
            shared_path_id=f"share_{uuid.uuid4().hex}",
            alias=clean_alias,
            path=resolved_path.as_posix(),
            mode=clean_mode,
            policy={
                "server": False,
                "daemon": False,
                "artifact_policy": "metadata_only",
                "overwrite_default": "never",
            },
            created_at=created_at,
        )
    finally:
        connection.close()

    status = _status_from_record(context.root, record, checked_at=None)
    return SharedPathRegistrationResult(record=record, status=status)


def list_shared_path_statuses(project_root: str | Path) -> tuple[SharedPathStatus, ...]:
    """Validate all registered shared paths and update last_checked_at."""

    context = require_project_context(project_root)
    ensure_database(context.root)
    checked_at = _utc_now_iso()
    connection = connect_database(project_database_path(context.root))
    try:
        repository = SharedPathRepository(connection)
        records = repository.list_all()
        statuses = tuple(
            _status_from_record(context.root, record, checked_at=checked_at) for record in records
        )
        for record in records:
            repository.update_last_checked(record.id, checked_at)
        connection.commit()
        return statuses
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def shared_path_registration_json(result: SharedPathRegistrationResult) -> dict[str, Any]:
    """Return stable JSON without leaking absolute private paths."""

    return {
        "ok": True,
        "id": result.record.id,
        "alias": result.record.alias,
        "mode": result.record.mode,
        "path_display": result.status.path_display,
        "status": result.status.status,
        "readable": result.status.readable,
        "writable": result.status.writable,
        "message": result.status.message,
        "database_mutation": "shared_paths_insert",
    }


def shared_path_statuses_json(statuses: tuple[SharedPathStatus, ...]) -> dict[str, Any]:
    """Return stable JSON for `pmem share status`."""

    return {
        "ok": True,
        "count": len(statuses),
        "database_mutation": "shared_paths_last_checked_update",
        "shared_paths": [
            {
                "id": status.id,
                "alias": status.alias,
                "mode": status.mode,
                "path_display": status.path_display,
                "status": status.status,
                "readable": status.readable,
                "writable": status.writable,
                "message": status.message,
                "last_checked_at": status.last_checked_at,
            }
            for status in statuses
        ],
    }


def _resolve_shared_path(project_root: Path, user_path: str | Path) -> Path:
    raw_text = str(user_path).strip()
    if not raw_text:
        raise PmemValidationError("Shared path cannot be blank.")
    if _has_control_character(raw_text) or "\\" in raw_text:
        raise PmemSecurityError("Shared path contains unsafe characters.")
    raw_path = Path(raw_text)
    if any(part in {".."} for part in raw_path.parts):
        raise PmemSecurityError("Shared path cannot contain traversal segments.")
    if any(part.casefold() == PMEM_DIRNAME.casefold() for part in raw_path.parts):
        raise PmemSecurityError("Shared path cannot point inside .pmem.")

    root = project_root.resolve()
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    resolved = candidate.resolve(strict=False)
    if any(part.casefold() == PMEM_DIRNAME.casefold() for part in resolved.parts):
        raise PmemSecurityError("Shared path cannot point inside .pmem.")
    if not raw_path.is_absolute() and root != resolved and root not in resolved.parents:
        raise PmemSecurityError("Relative shared path must stay inside the project.")
    _reject_symlink_parts(candidate)
    if not resolved.exists():
        raise PmemValidationError("Shared path must exist before registration.")
    if resolved.is_symlink():
        raise PmemSecurityError("Shared path cannot be a symlink.")
    if not resolved.is_dir():
        raise PmemSecurityError("Shared path must be a directory.")
    return resolved


def _reject_symlink_parts(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path()
    for part in path.parts:
        if part in {"", path.anchor, "."}:
            continue
        current = current / part
        if current.exists() and current.is_symlink():
            raise PmemSecurityError("Shared path cannot contain symlinks.")


def _normalize_alias(alias: str | None, resolved_path: Path) -> str:
    raw = alias if alias is not None else (resolved_path.name or "shared")
    cleaned = raw.strip()
    if not cleaned:
        raise PmemValidationError("Shared path alias cannot be blank.")
    if _has_control_character(cleaned) or "/" in cleaned or "\\" in cleaned:
        raise PmemSecurityError("Shared path alias contains unsafe characters.")
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", cleaned) is None:
        raise PmemValidationError(
            "Shared path alias must use letters, numbers, dot, dash, or underscore."
        )
    return cleaned


def _validate_mode(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned not in SHARED_PATH_MODES:
        raise PmemValidationError("Shared path mode must be read, write, or read_write.")
    return cleaned


def _status_from_record(
    project_root: Path,
    record: SharedPathRecord,
    *,
    checked_at: str | None,
) -> SharedPathStatus:
    path = Path(record.path)
    exists = path.exists()
    is_dir = path.is_dir()
    readable = exists and os.access(path, os.R_OK)
    writable = exists and os.access(path, os.W_OK)
    expected_writable = record.mode in {"write", "read_write"}
    expected_readable = record.mode in {"read", "read_write"}
    ok = (
        exists
        and is_dir
        and (not expected_readable or readable)
        and (not expected_writable or writable)
    )
    message = "Shared path is available." if ok else "Shared path is not currently usable."
    return SharedPathStatus(
        id=record.id,
        alias=record.alias,
        mode=record.mode,
        path_display=_display_path(project_root, path),
        status="ok" if ok else "invalid",
        readable=readable,
        writable=writable,
        message=message,
        last_checked_at=checked_at or record.last_checked_at,
    )


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return f"<external:{path.name or 'shared-path'}>"


def _has_control_character(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
