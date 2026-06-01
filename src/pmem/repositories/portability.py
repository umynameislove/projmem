"""Repositories for portability and failure-analysis portability metadata tables."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from pmem.repositories.sqlite import execute, query_one


@dataclass(frozen=True)
class ExportPackageRecord:
    """SQLite representation of one export package audit row."""

    id: str
    version: str
    manifest_hash: str
    payload_hash: str
    artifact_count: int
    path: str
    status: str
    created_at: str


@dataclass(frozen=True)
class ImportJobRecord:
    """SQLite representation of one pending/quarantined import job."""

    id: str
    source_hash: str
    status: str
    conflict_count: int
    report_json: str
    created_at: str
    applied_at: str | None
    provenance_source: str | None


@dataclass(frozen=True)
class SharedPathRecord:
    """SQLite representation of one explicit shared memory path."""

    id: str
    alias: str
    path: str
    mode: str
    policy_json: str
    last_checked_at: str | None
    created_at: str


@dataclass(frozen=True)
class AuditEventRecord:
    """SQLite representation of one append-only audit event."""

    id: str
    event_type: str
    entity_type: str | None
    entity_id: str | None
    before_hash: str | None
    after_hash: str | None
    actor: str | None
    timestamp: str
    metadata_json: str


class ExportPackageRepository:
    """Read and write `export_packages` rows with parameterized SQL."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def create_or_get(
        self,
        *,
        package_id: str,
        version: str,
        scope: dict[str, Any],
        manifest_hash: str,
        payload_hash: str,
        artifact_count: int,
        path: str,
        created_at: str,
        status: str = "written",
    ) -> ExportPackageRecord:
        """Insert one export package row, or return the existing manifest row."""

        existing = self.get_by_manifest_hash(manifest_hash)
        if existing is not None:
            return existing

        scope_json = json.dumps(scope, sort_keys=True, separators=(",", ":"))
        execute(
            self._connection,
            """
            INSERT INTO export_packages(
                id, version, scope_json, manifest_hash, payload_hash, artifact_count,
                path, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                package_id,
                version,
                scope_json,
                manifest_hash,
                payload_hash,
                artifact_count,
                path,
                status,
                created_at,
            ),
        )
        self._connection.commit()
        created = self.get_by_manifest_hash(manifest_hash)
        if created is None:  # pragma: no cover - guarded by SQLite insert success.
            raise RuntimeError("export package insert did not return a row")
        return created

    def get_by_manifest_hash(self, manifest_hash: str) -> ExportPackageRecord | None:
        """Return an export package row by manifest hash."""

        row = query_one(
            self._connection,
            """
            SELECT id, version, manifest_hash, payload_hash, artifact_count,
                   path, status, created_at
            FROM export_packages
            WHERE manifest_hash = ?
            """,
            (manifest_hash,),
        )
        return _export_package_from_row(row) if row is not None else None


class ImportJobRepository:
    """Write import job rows while the service owns the transaction boundary."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def insert_pending(
        self,
        *,
        job_id: str,
        source_hash: str,
        conflict_count: int,
        report: dict[str, Any],
        created_at: str,
        provenance_source: str | None,
    ) -> ImportJobRecord:
        """Insert a pending import job without committing the caller transaction."""

        report_json = json.dumps(report, sort_keys=True, separators=(",", ":"))
        execute(
            self._connection,
            """
            INSERT INTO import_jobs(
                id, source_hash, status, conflict_count, report_json, created_at,
                applied_at, provenance_source
            )
            VALUES (?, ?, 'pending', ?, ?, ?, NULL, ?)
            """,
            (
                job_id,
                source_hash,
                conflict_count,
                report_json,
                created_at,
                provenance_source,
            ),
        )
        return ImportJobRecord(
            id=job_id,
            source_hash=source_hash,
            status="pending",
            conflict_count=conflict_count,
            report_json=report_json,
            created_at=created_at,
            applied_at=None,
            provenance_source=provenance_source,
        )

    def has_source_hash(self, source_hash: str) -> bool:
        """Return whether a bundle file hash already has an import job."""

        row = query_one(
            self._connection,
            "SELECT 1 FROM import_jobs WHERE source_hash = ? LIMIT 1",
            (source_hash,),
        )
        return row is not None


class SharedPathRepository:
    """Read and write `shared_paths` rows with parameterized SQL."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def create(
        self,
        *,
        shared_path_id: str,
        alias: str,
        path: str,
        mode: str,
        policy: dict[str, Any],
        created_at: str,
    ) -> SharedPathRecord:
        """Insert one shared path row."""

        policy_json = json.dumps(policy, sort_keys=True, separators=(",", ":"))
        execute(
            self._connection,
            """
            INSERT INTO shared_paths(
                id, alias, path, mode, policy_json, last_checked_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, NULL, ?)
            """,
            (shared_path_id, alias, path, mode, policy_json, created_at),
        )
        self._connection.commit()
        row = self.get_by_alias(alias)
        if row is None:  # pragma: no cover - guarded by SQLite insert success.
            raise RuntimeError("shared path insert did not return a row")
        return row

    def get_by_alias(self, alias: str) -> SharedPathRecord | None:
        """Return a shared path by alias."""

        row = query_one(
            self._connection,
            """
            SELECT id, alias, path, mode, policy_json, last_checked_at, created_at
            FROM shared_paths
            WHERE alias = ?
            """,
            (alias,),
        )
        return _shared_path_from_row(row) if row is not None else None

    def get_by_path(self, path: str) -> SharedPathRecord | None:
        """Return a shared path by stored normalized path."""

        row = query_one(
            self._connection,
            """
            SELECT id, alias, path, mode, policy_json, last_checked_at, created_at
            FROM shared_paths
            WHERE path = ?
            """,
            (path,),
        )
        return _shared_path_from_row(row) if row is not None else None

    def list_all(self) -> tuple[SharedPathRecord, ...]:
        """Return all shared paths in deterministic alias order."""

        rows = execute(
            self._connection,
            """
            SELECT id, alias, path, mode, policy_json, last_checked_at, created_at
            FROM shared_paths
            ORDER BY alias ASC
            """,
        ).fetchall()
        return tuple(_shared_path_from_row(row) for row in rows)

    def update_last_checked(self, shared_path_id: str, checked_at: str) -> None:
        """Record the last validation timestamp for one shared path."""

        execute(
            self._connection,
            "UPDATE shared_paths SET last_checked_at = ? WHERE id = ?",
            (checked_at, shared_path_id),
        )


class AuditEventRepository:
    """Append audit events while the service owns policy decisions."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def insert(
        self,
        *,
        event_id: str,
        event_type: str,
        entity_type: str | None,
        entity_id: str | None,
        before_hash: str | None,
        after_hash: str | None,
        actor: str | None,
        timestamp: str,
        metadata: dict[str, Any],
    ) -> AuditEventRecord:
        """Insert one append-only audit event without mutating canonical records."""

        metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        execute(
            self._connection,
            """
            INSERT INTO audit_events(
                id, event_type, entity_type, entity_id, before_hash, after_hash,
                actor, timestamp, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                entity_type,
                entity_id,
                before_hash,
                after_hash,
                actor,
                timestamp,
                metadata_json,
            ),
        )
        return AuditEventRecord(
            id=event_id,
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            before_hash=before_hash,
            after_hash=after_hash,
            actor=actor,
            timestamp=timestamp,
            metadata_json=metadata_json,
        )


def _export_package_from_row(row: sqlite3.Row) -> ExportPackageRecord:
    return ExportPackageRecord(
        id=str(row["id"]),
        version=str(row["version"]),
        manifest_hash=str(row["manifest_hash"]),
        payload_hash=str(row["payload_hash"]),
        artifact_count=int(row["artifact_count"]),
        path=str(row["path"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
    )


def _shared_path_from_row(row: sqlite3.Row) -> SharedPathRecord:
    return SharedPathRecord(
        id=str(row["id"]),
        alias=str(row["alias"]),
        path=str(row["path"]),
        mode=str(row["mode"]),
        policy_json=str(row["policy_json"]),
        last_checked_at=str(row["last_checked_at"]) if row["last_checked_at"] is not None else None,
        created_at=str(row["created_at"]),
    )
