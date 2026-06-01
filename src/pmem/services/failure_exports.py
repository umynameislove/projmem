"""Privacy-safe failure listing and export substrate for failure-analysis layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from pmem.errors import PmemSecurityError, PmemValidationError
from pmem.repositories.failures import FailureRecord, FailureRepository
from pmem.repositories.sqlite import PMEM_DIRNAME, connect_database, project_database_path
from pmem.services.project_context import require_project_context

FAILURE_EXPORT_SCHEMA_VERSION = "failure-export-v1"


@dataclass(frozen=True)
class FailureExportResult:
    """Result of writing one failure export JSON file."""

    output_path: Path
    display_path: str
    payload: dict[str, Any]


def failure_export_payload(
    project_root: str | Path,
    *,
    include_text: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic-shape failure export payload."""

    records = list_failure_records(project_root, include_text=include_text)
    privacy_mode = "explicit_text" if include_text else "redacted"
    return {
        "schema_version": FAILURE_EXPORT_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "privacy_mode": privacy_mode,
        "include_text": include_text,
        "record_count": len(records),
        "records": records,
    }


def list_failure_records(
    project_root: str | Path,
    *,
    include_text: bool = False,
) -> list[dict[str, Any]]:
    """List confirmed failures without exposing raw free text by default."""

    context = require_project_context(project_root)
    connection = connect_database(project_database_path(context.root))
    try:
        failures = FailureRepository(connection).list_for_project(context.project.id)
    finally:
        connection.close()
    return [_failure_record_payload(record, include_text=include_text) for record in failures]


def export_failure_records(
    project_root: str | Path,
    *,
    output_path: str | Path,
    include_text: bool = False,
) -> FailureExportResult:
    """Write a failure export JSON file with project-safe path checks."""

    context = require_project_context(project_root)
    output = _resolve_failure_export_path(context.root, output_path)
    payload = failure_export_payload(context.root, include_text=include_text)
    try:
        output.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
        output.chmod(0o600)
    except OSError as exc:
        raise PmemValidationError("Failure export file could not be written.") from exc
    return FailureExportResult(
        output_path=output,
        display_path=output.resolve(strict=False).relative_to(context.root.resolve()).as_posix(),
        payload=payload,
    )


def _failure_record_payload(record: FailureRecord, *, include_text: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": record.id,
        "run_id": record.run_id,
        "error_type": record.error_type,
        "severity": record.severity,
        "source": record.source,
        "tags": _tags(record.tags_json),
        "created_at": record.created_at,
        "text_included": include_text,
    }
    if include_text:
        payload["description"] = record.description
        payload["root_cause"] = record.root_cause
        payload["lesson"] = record.lesson
    return payload


def _tags(raw_tags: str) -> list[str]:
    value = json.loads(raw_tags)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _resolve_failure_export_path(project_root: Path, user_path: str | Path) -> Path:
    raw_text = str(user_path).strip()
    if not raw_text:
        raise PmemValidationError("Failure export path cannot be blank.")
    if "\\" in raw_text or "\x00" in raw_text or any(ord(char) < 32 for char in raw_text):
        raise PmemSecurityError("Failure export path contains unsafe characters.")
    raw_path = Path(raw_text)
    if raw_path.is_absolute() or PureWindowsPath(raw_text).is_absolute():
        raise PmemSecurityError("Failure export path must be project-relative.")
    if any(part == ".." for part in raw_path.parts):
        raise PmemSecurityError("Failure export path cannot contain traversal segments.")
    if any(part.casefold() == PMEM_DIRNAME.casefold() for part in raw_path.parts):
        raise PmemSecurityError("Failure export path cannot point inside .pmem.")

    root = project_root.resolve()
    output = root / raw_path
    parent = output.parent.resolve(strict=False)
    if root != parent and root not in parent.parents:
        raise PmemSecurityError("Failure export path must stay inside the project.")
    _reject_symlink_parts(root, raw_path.parent)
    if output.exists() and output.is_dir():
        raise PmemSecurityError("Failure export path must point to a file, not a directory.")
    if output.exists() and output.is_symlink():
        raise PmemSecurityError("Failure export path cannot be a symlink.")
    parent.mkdir(parents=True, exist_ok=True)
    return parent / output.name


def _reject_symlink_parts(project_root: Path, relative_parent: Path) -> None:
    current = project_root
    for part in relative_parent.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.exists() and current.is_symlink():
            raise PmemSecurityError("Failure export path cannot contain symlinks.")


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
