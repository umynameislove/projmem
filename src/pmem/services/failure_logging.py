"""`pmem log-failure` service workflow."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from pmem.domain.common import FailureSeverity, FailureSource
from pmem.domain.failure import Failure
from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.repositories.failures import FailureRecord, FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_context import require_project_context


def log_failure(
    project_root: str | Path,
    *,
    run_id: str,
    error_type: str,
    description: str,
    root_cause: str | None = None,
    lesson: str | None = None,
    severity: str = "medium",
    tags: tuple[str, ...] = (),
    source: str = "user_confirmed",
) -> FailureRecord:
    """Validate and persist one confirmed failure for an existing run."""

    context = require_project_context(project_root)
    created_at = _utc_now_iso()
    parsed_severity = _parse_severity(severity)
    parsed_source = _parse_source(source)
    try:
        failure = Failure(
            id=f"fail_{uuid.uuid4().hex}",
            run_id=run_id,
            error_type=error_type,
            description=description,
            root_cause=root_cause,
            lesson=lesson,
            severity=parsed_severity,
            tags=list(tags),
            source=parsed_source,
            created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00")),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise PmemValidationError(_failure_validation_message(str(exc))) from exc

    connection = connect_database(project_database_path(context.root))
    try:
        run = RunRepository(connection).get_by_id(failure.run_id)
        if run is None:
            raise PmemNotFoundError("Run was not found.")
        return FailureRepository(connection).create(
            failure_id=failure.id,
            run_id=failure.run_id,
            error_type=failure.error_type,
            description=failure.description,
            root_cause=failure.root_cause,
            lesson=failure.lesson,
            severity=failure.severity.value,
            tags=failure.tags,
            source=failure.source.value,
            created_at=created_at,
        )
    finally:
        connection.close()


def _failure_validation_message(raw_message: str) -> str:
    """Return a concise public validation message."""

    if "severity" in raw_message:
        return "Failure severity must be critical, high, medium, or low."
    if "source" in raw_message:
        return "Failure source must be user_confirmed, auto_technical, or promoted_candidate."
    if "tags" in raw_message:
        return "Failure tags cannot be blank."
    if "cannot be blank" in raw_message:
        return "Failure text fields cannot be blank."
    return "Failure input is invalid."


def _parse_severity(value: str) -> FailureSeverity:
    """Parse a CLI/service severity string into the locked failure taxonomy enum."""

    try:
        return FailureSeverity(value.strip().lower())
    except ValueError as exc:
        raise PmemValidationError(
            "Failure severity must be critical, high, medium, or low."
        ) from exc


def _parse_source(value: str) -> FailureSource:
    """Parse a CLI/service source string into the locked failure taxonomy enum."""

    try:
        return FailureSource(value.strip().lower())
    except ValueError as exc:
        raise PmemValidationError(
            "Failure source must be user_confirmed, auto_technical, or promoted_candidate."
        ) from exc


def _utc_now_iso() -> str:
    """Return compact UTC ISO timestamp for memory records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
