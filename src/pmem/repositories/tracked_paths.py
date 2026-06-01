"""Tracked-path table repository.

The repository only persists normalized path records. Filesystem validation,
hash calculation, and `.pmem/` protection belong to the tracking service.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from pmem.errors import PmemPersistenceError
from pmem.repositories.sqlite import execute, query_one


@dataclass(frozen=True)
class TrackedPathRecord:
    """SQLite representation of one tracked file."""

    id: str
    project_id: str
    path: str
    sha256: str
    tag: str | None
    size_bytes: int | None
    last_checked: str
    created_at: str


class TrackedPathRepository:
    """Read and write tracked-path records with parameterized queries."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def add(
        self,
        *,
        tracked_path_id: str,
        project_id: str,
        path: str,
        sha256: str,
        size_bytes: int,
        last_checked: str,
        created_at: str,
        tag: str | None = None,
    ) -> TrackedPathRecord:
        """Insert one tracked-path record and commit it atomically."""

        try:
            self._connection.execute("BEGIN")
            execute(
                self._connection,
                """
                INSERT INTO tracked_paths(
                    id, project_id, path, tag, hash, size_bytes, last_checked, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tracked_path_id,
                    project_id,
                    path,
                    tag,
                    sha256,
                    size_bytes,
                    last_checked,
                    created_at,
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

        return TrackedPathRecord(
            id=tracked_path_id,
            project_id=project_id,
            path=path,
            sha256=sha256,
            tag=tag,
            size_bytes=size_bytes,
            last_checked=last_checked,
            created_at=created_at,
        )

    def get_by_project_and_path(self, project_id: str, path: str) -> TrackedPathRecord | None:
        """Return a tracked file by project and normalized path."""

        row = query_one(
            self._connection,
            """
            SELECT id, project_id, path, tag, hash, size_bytes, last_checked, created_at
            FROM tracked_paths
            WHERE project_id = ? AND path = ?
            """,
            (project_id, path),
        )
        return _tracked_path_from_row(row) if row is not None else None

    def update_hash(
        self,
        *,
        tracked_path_id: str,
        sha256: str,
        size_bytes: int,
        last_checked: str,
    ) -> TrackedPathRecord:
        """Refresh hash and size for an existing tracked path."""

        try:
            self._connection.execute("BEGIN")
            execute(
                self._connection,
                """
                UPDATE tracked_paths
                SET hash = ?, size_bytes = ?, last_checked = ?
                WHERE id = ?
                """,
                (sha256, size_bytes, last_checked, tracked_path_id),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

        row = query_one(
            self._connection,
            """
            SELECT id, project_id, path, tag, hash, size_bytes, last_checked, created_at
            FROM tracked_paths
            WHERE id = ?
            """,
            (tracked_path_id,),
        )
        if row is None:
            raise PmemPersistenceError()
        return _tracked_path_from_row(row)

    def list_for_project(self, project_id: str) -> tuple[TrackedPathRecord, ...]:
        """Return tracked files for one project in deterministic order."""

        rows = execute(
            self._connection,
            """
            SELECT id, project_id, path, tag, hash, size_bytes, last_checked, created_at
            FROM tracked_paths
            WHERE project_id = ?
            ORDER BY path
            """,
            (project_id,),
        ).fetchall()
        return tuple(_tracked_path_from_row(row) for row in rows)


def _tracked_path_from_row(row: sqlite3.Row) -> TrackedPathRecord:
    """Convert a SQLite row to the typed repository record."""

    return TrackedPathRecord(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        path=str(row["path"]),
        sha256=str(row["hash"]),
        tag=str(row["tag"]) if row["tag"] is not None else None,
        size_bytes=int(row["size_bytes"]) if row["size_bytes"] is not None else None,
        last_checked=str(row["last_checked"]),
        created_at=str(row["created_at"]),
    )
