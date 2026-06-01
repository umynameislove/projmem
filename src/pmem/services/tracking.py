"""`pmem track` service workflow.

file tracking tracks regular files only. Directory tracking is intentionally rejected until
recursive policy, performance limits, and snapshot semantics are specified.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pmem.errors import PmemNotFoundError, PmemSecurityError, PmemValidationError
from pmem.repositories.projects import ProjectRepository
from pmem.repositories.sqlite import PMEM_DIRNAME, connect_database, project_database_path
from pmem.repositories.tracked_paths import TrackedPathRecord, TrackedPathRepository
from pmem.services.config import project_config_path, read_project_config
from pmem.services.database import ensure_database
from pmem.utils.hashing import compute_file_hash

MAX_TRACKED_PATH_LENGTH = 512


@dataclass(frozen=True)
class TrackPathResult:
    """User-facing result of `pmem track <path>`."""

    already_tracked: bool
    updated: bool
    path: str
    sha256: str
    size_bytes: int | None
    project_id: str


@dataclass(frozen=True)
class ValidatedTrackPath:
    """Safe normalized file path ready for hashing and persistence."""

    absolute_path: Path
    relative_path: str
    size_bytes: int


def track_path(
    project_root: str | Path,
    user_path: str | Path,
    *,
    update: bool = False,
) -> TrackPathResult:
    """Track one regular file in an initialized project."""

    root = Path(project_root)
    config_path = project_config_path(root)
    db_path = project_database_path(root)
    if not config_path.exists() or not db_path.exists():
        raise PmemNotFoundError("projmem is not initialized. Run `pmem init` first.")

    ensure_database(root)
    config = read_project_config(config_path)
    validated_path = validate_track_path(root, user_path)

    connection = connect_database(db_path)
    try:
        project = ProjectRepository(connection).get_by_id(config.project_id)
        if project is None:
            raise PmemNotFoundError("projmem is not initialized. Run `pmem init` first.")

        repository = TrackedPathRepository(connection)
        existing = repository.get_by_project_and_path(project.id, validated_path.relative_path)
        if existing is not None and not update:
            return _result_from_record(existing, already_tracked=True)

        try:
            sha256 = compute_file_hash(validated_path.absolute_path)
        except OSError as exc:
            raise PmemValidationError("Tracked path could not be read.") from exc

        timestamp = _utc_now_iso()
        if existing is not None:
            record = repository.update_hash(
                tracked_path_id=existing.id,
                sha256=sha256,
                size_bytes=validated_path.size_bytes,
                last_checked=timestamp,
            )
            return _result_from_record(record, already_tracked=True, updated=True)

        record = repository.add(
            tracked_path_id=f"track_{uuid.uuid4().hex}",
            project_id=project.id,
            path=validated_path.relative_path,
            sha256=sha256,
            size_bytes=validated_path.size_bytes,
            last_checked=timestamp,
            created_at=timestamp,
        )
        return _result_from_record(record, already_tracked=False)
    finally:
        connection.close()


def validate_track_path(project_root: str | Path, user_path: str | Path) -> ValidatedTrackPath:
    """Validate and normalize a file tracking tracked file path."""

    raw_text = str(user_path)
    if not raw_text.strip():
        raise PmemValidationError("Track path cannot be blank.")
    if len(raw_text) > MAX_TRACKED_PATH_LENGTH:
        raise PmemValidationError("Track path is too long.")
    if any(ord(character) < 32 for character in raw_text):
        raise PmemValidationError("Track path contains unsupported control characters.")

    raw_path = Path(raw_text)
    if raw_path.is_absolute():
        raise PmemSecurityError("Track path must be relative to the project root.")
    if _is_pmem_internal_path(raw_path):
        raise PmemSecurityError("projmem internal files cannot be tracked.")

    root = Path(project_root).resolve()
    candidate = root / raw_path
    if _has_symlink_component(root, raw_path):
        raise PmemSecurityError("Symlink tracking is not supported.")

    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise PmemSecurityError("Track path must stay inside the project root.") from exc

    if _is_pmem_internal_path(relative):
        raise PmemSecurityError("projmem internal files cannot be tracked.")
    if not candidate.exists():
        raise PmemNotFoundError("Tracked path does not exist.")
    if candidate.is_dir():
        raise PmemValidationError("Directory tracking is not supported.")
    if not candidate.is_file():
        raise PmemValidationError("Only regular files can be tracked.")

    try:
        size_bytes = candidate.stat().st_size
    except OSError as exc:
        raise PmemValidationError("Tracked path could not be read.") from exc

    return ValidatedTrackPath(
        absolute_path=candidate,
        relative_path=relative.as_posix(),
        size_bytes=size_bytes,
    )


def _has_symlink_component(project_root: Path, raw_path: Path) -> bool:
    """Return whether any user-provided path component is a symlink."""

    current = project_root
    for part in raw_path.parts:
        if part in ("", "."):
            continue
        current = current / part
        if current.is_symlink():
            return True
    return False


def _is_pmem_internal_path(path: Path) -> bool:
    """Return whether a path targets `.pmem/`, including case variants."""

    return bool(path.parts and path.parts[0].casefold() == PMEM_DIRNAME.casefold())


def _result_from_record(
    record: TrackedPathRecord,
    *,
    already_tracked: bool,
    updated: bool = False,
) -> TrackPathResult:
    """Create a service result from a repository record."""

    return TrackPathResult(
        already_tracked=already_tracked,
        updated=updated,
        path=record.path,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        project_id=record.project_id,
    )


def _utc_now_iso() -> str:
    """Return compact UTC ISO timestamp for file tracking tracking records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
