"""Failure table repository.

This module owns persistence for confirmed failures only. It serializes tag
metadata, uses parameterized queries, and leaves taxonomy validation to the
domain/service layer.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from pmem.repositories.sqlite import execute, query_one


@dataclass(frozen=True)
class FailureRecord:
    """SQLite representation of one confirmed failure row."""

    id: str
    run_id: str
    error_type: str
    description: str
    root_cause: str | None
    lesson: str | None
    severity: str
    tags_json: str
    source: str
    created_at: str


class FailureRepository:
    """Read and write confirmed failures through parameterized SQLite queries."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def create(
        self,
        *,
        failure_id: str,
        run_id: str,
        error_type: str,
        description: str,
        root_cause: str | None,
        lesson: str | None,
        severity: str,
        tags: list[str],
        source: str,
        created_at: str,
    ) -> FailureRecord:
        """Insert one confirmed failure row and commit it atomically."""

        tags_json = json.dumps(tags, sort_keys=True, separators=(",", ":"))
        try:
            self._connection.execute("BEGIN")
            execute(
                self._connection,
                """
                INSERT INTO failures(
                    id, run_id, error_type, description, root_cause, lesson,
                    severity, tags_json, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    failure_id,
                    run_id,
                    error_type,
                    description,
                    root_cause,
                    lesson,
                    severity,
                    tags_json,
                    source,
                    created_at,
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

        return FailureRecord(
            id=failure_id,
            run_id=run_id,
            error_type=error_type,
            description=description,
            root_cause=root_cause,
            lesson=lesson,
            severity=severity,
            tags_json=tags_json,
            source=source,
            created_at=created_at,
        )

    def get_by_id(self, failure_id: str) -> FailureRecord | None:
        """Return a confirmed failure by id."""

        row = query_one(
            self._connection,
            """
            SELECT id, run_id, error_type, description, root_cause, lesson,
                   severity, tags_json, source, created_at
            FROM failures
            WHERE id = ?
            """,
            (failure_id,),
        )
        return _failure_from_row(row) if row is not None else None

    def list_for_run(self, run_id: str) -> tuple[FailureRecord, ...]:
        """Return failures for one run in creation order."""

        rows = execute(
            self._connection,
            """
            SELECT id, run_id, error_type, description, root_cause, lesson,
                   severity, tags_json, source, created_at
            FROM failures
            WHERE run_id = ?
            ORDER BY created_at, id
            """,
            (run_id,),
        ).fetchall()
        return tuple(_failure_from_row(row) for row in rows)

    def list_for_project(self, project_id: str) -> tuple[FailureRecord, ...]:
        """Return failures for one project in newest-first order."""

        rows = execute(
            self._connection,
            """
            SELECT failures.id, failures.run_id, failures.error_type,
                   failures.description, failures.root_cause, failures.lesson,
                   failures.severity, failures.tags_json, failures.source,
                   failures.created_at
            FROM failures
            JOIN runs ON runs.run_id = failures.run_id
            JOIN experiments ON experiments.id = runs.experiment_id
            WHERE experiments.project_id = ?
            ORDER BY failures.created_at DESC, failures.id DESC
            """,
            (project_id,),
        ).fetchall()
        return tuple(_failure_from_row(row) for row in rows)


def _failure_from_row(row: sqlite3.Row) -> FailureRecord:
    """Convert a SQLite row to the typed repository record."""

    return FailureRecord(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        error_type=str(row["error_type"]),
        description=str(row["description"]),
        root_cause=str(row["root_cause"]) if row["root_cause"] is not None else None,
        lesson=str(row["lesson"]) if row["lesson"] is not None else None,
        severity=str(row["severity"]),
        tags_json=str(row["tags_json"]),
        source=str(row["source"]),
        created_at=str(row["created_at"]),
    )
