"""portability and failure-analysis import apply service with pending/quarantine semantics."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.domain.import_bundle import ENTITY_KEYS, ImportDryRunReport
from pmem.errors import PmemPersistenceError, PmemSecurityError, PmemValidationError
from pmem.repositories.portability import ImportJobRecord, ImportJobRepository
from pmem.repositories.sqlite import connect_database, execute, project_database_path
from pmem.services.database import ensure_database
from pmem.services.import_dry_run import dry_run_import_bundle, import_dry_run_report_json
from pmem.services.project_context import require_project_context
from pmem.utils.hashing import compute_file_hash

PORTABILITY_TABLES: tuple[str, ...] = (
    "export_packages",
    "import_jobs",
    "shared_paths",
    "audit_events",
)


@dataclass(frozen=True)
class ImportApplyResult:
    """User-facing result for one pending/quarantined import apply."""

    job: ImportJobRecord
    dry_run_report: ImportDryRunReport
    source_hash: str
    row_counts_before: dict[str, int]
    row_counts_after: dict[str, int]
    integrity_check: str
    foreign_key_check: list[dict[str, Any]]


def apply_import_bundle(
    project_root: str | Path,
    bundle_path: str | Path,
    *,
    confirm: bool,
) -> ImportApplyResult:
    """Validate and quarantine an import bundle without overwriting trusted rows."""

    if not confirm:
        raise PmemValidationError("Import apply requires --confirm after reviewing dry-run output.")

    context = require_project_context(project_root)
    report = dry_run_import_bundle(context.root, bundle_path)
    if not report.ok:
        raise PmemValidationError(
            "Import bundle failed dry-run validation. Run `pmem import --dry-run` for details."
        )

    ensure_database(context.root)
    resolved_bundle = _resolve_bundle_for_hash(context.root, bundle_path)
    source_hash = f"sha256:{compute_file_hash(resolved_bundle)}"
    connection = connect_database(project_database_path(context.root))
    created_at = _utc_now_iso()
    job_id = f"import_{uuid.uuid4().hex}"
    before = _row_counts(connection)
    report_payload = import_dry_run_report_json(report)

    try:
        connection.execute("BEGIN")
        job = ImportJobRepository(connection).insert_pending(
            job_id=job_id,
            source_hash=source_hash,
            conflict_count=len(report.conflicts),
            report=report_payload,
            created_at=created_at,
            provenance_source=_provenance_source(report_payload),
        )
        _insert_audit_event(
            connection,
            event_id=f"audit_{uuid.uuid4().hex}",
            import_job_id=job.id,
            source_hash=source_hash,
            timestamp=created_at,
            conflict_count=len(report.conflicts),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        connection.close()
        raise

    try:
        after = _row_counts(connection)
        integrity_check = _integrity_check(connection)
        foreign_key_check = _foreign_key_check(connection)
    finally:
        connection.close()

    return ImportApplyResult(
        job=job,
        dry_run_report=report,
        source_hash=source_hash,
        row_counts_before=before,
        row_counts_after=after,
        integrity_check=integrity_check,
        foreign_key_check=foreign_key_check,
    )


def import_apply_result_json(result: ImportApplyResult) -> dict[str, Any]:
    """Return stable machine-readable output for `pmem import --apply --json`."""

    return {
        "ok": True,
        "status": result.job.status,
        "import_job_id": result.job.id,
        "source_hash": result.source_hash,
        "conflict_count": result.job.conflict_count,
        "database_mutation": "quarantine_pending_import_job",
        "row_counts_before": result.row_counts_before,
        "row_counts_after": result.row_counts_after,
        "integrity_check": result.integrity_check,
        "foreign_key_check": result.foreign_key_check,
    }


def _resolve_bundle_for_hash(project_root: Path, bundle_path: str | Path) -> Path:
    raw_text = str(bundle_path).strip()
    if not raw_text:
        raise PmemValidationError("Bundle path cannot be blank.")
    raw_path = Path(raw_text)
    if raw_path.is_absolute():
        raise PmemSecurityError("Bundle path must be project-relative.")
    if any(part.lower() == ".pmem" for part in raw_path.parts):
        raise PmemSecurityError("Bundle path cannot point inside .pmem.")

    root = project_root.resolve()
    resolved = (root / raw_path).resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise PmemSecurityError("Bundle path must stay inside the project.")
    if resolved.is_dir() or not resolved.is_file():
        raise PmemSecurityError("Bundle path must point to a regular file.")
    return resolved


def _row_counts(connection: Any) -> dict[str, int]:
    tables = tuple(ENTITY_KEYS) + PORTABILITY_TABLES
    return {table: _count_rows(connection, table) for table in tables}


def _count_rows(connection: Any, table: str) -> int:
    row = execute(connection, f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    if row is None:  # pragma: no cover - COUNT(*) always returns one row.
        raise PmemPersistenceError()
    return int(row["count"])


def _insert_audit_event(
    connection: Any,
    *,
    event_id: str,
    import_job_id: str,
    source_hash: str,
    timestamp: str,
    conflict_count: int,
) -> None:
    metadata_json = json.dumps(
        {
            "status": "pending",
            "conflict_count": conflict_count,
            "database_mutation": "quarantine_pending_import_job",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    execute(
        connection,
        """
        INSERT INTO audit_events(
            id, event_type, entity_type, entity_id, before_hash, after_hash,
            actor, timestamp, metadata_json
        )
        VALUES (?, 'import.apply_quarantined', 'import_jobs', ?, NULL, ?, 'local', ?, ?)
        """,
        (event_id, import_job_id, source_hash, timestamp, metadata_json),
    )


def _integrity_check(connection: Any) -> str:
    row = execute(connection, "PRAGMA integrity_check").fetchone()
    return str(row[0]) if row is not None else "unknown"


def _foreign_key_check(connection: Any) -> list[dict[str, Any]]:
    rows = execute(connection, "PRAGMA foreign_key_check").fetchall()
    return [dict(row) for row in rows]


def _provenance_source(report_payload: dict[str, Any]) -> str | None:
    version = report_payload.get("export_format_version")
    schema = report_payload.get("schema_version")
    if isinstance(version, str) and isinstance(schema, str):
        return f"export_bundle:{version}:{schema}"
    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
