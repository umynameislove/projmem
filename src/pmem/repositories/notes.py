"""Note table repository.

Notes are lightweight project memory records. Repository code stores JSON
fields deterministically and keeps context validation in the service layer.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from pmem.repositories.sqlite import execute, query_one


@dataclass(frozen=True)
class NoteRecord:
    """SQLite representation of one note row."""

    id: str
    project_id: str
    experiment_id: str | None
    run_id: str | None
    content: str
    tags_json: str
    context_json: str
    resolved: bool
    created_at: str


class NoteRepository:
    """Read and write notes through parameterized SQLite queries."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def create(
        self,
        *,
        note_id: str,
        project_id: str,
        content: str,
        tags: list[str],
        context: dict[str, Any],
        resolved: bool,
        created_at: str,
        experiment_id: str | None = None,
        run_id: str | None = None,
    ) -> NoteRecord:
        """Insert one note row and commit it atomically."""

        tags_json = json.dumps(tags, sort_keys=True, separators=(",", ":"))
        context_json = json.dumps(context, sort_keys=True, separators=(",", ":"))
        try:
            self._connection.execute("BEGIN")
            execute(
                self._connection,
                """
                INSERT INTO notes(
                    id, project_id, experiment_id, run_id, content, tags_json,
                    context_json, resolved, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note_id,
                    project_id,
                    experiment_id,
                    run_id,
                    content,
                    tags_json,
                    context_json,
                    1 if resolved else 0,
                    created_at,
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

        return NoteRecord(
            id=note_id,
            project_id=project_id,
            experiment_id=experiment_id,
            run_id=run_id,
            content=content,
            tags_json=tags_json,
            context_json=context_json,
            resolved=resolved,
            created_at=created_at,
        )

    def get_by_id(self, note_id: str) -> NoteRecord | None:
        """Return a note by id."""

        row = query_one(
            self._connection,
            """
            SELECT id, project_id, experiment_id, run_id, content, tags_json,
                   context_json, resolved, created_at
            FROM notes
            WHERE id = ?
            """,
            (note_id,),
        )
        return _note_from_row(row) if row is not None else None

    def list_for_project(self, project_id: str) -> tuple[NoteRecord, ...]:
        """Return project notes in newest-first order."""

        rows = execute(
            self._connection,
            """
            SELECT id, project_id, experiment_id, run_id, content, tags_json,
                   context_json, resolved, created_at
            FROM notes
            WHERE project_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (project_id,),
        ).fetchall()
        return tuple(_note_from_row(row) for row in rows)


def _note_from_row(row: sqlite3.Row) -> NoteRecord:
    """Convert a SQLite row to the typed repository record."""

    return NoteRecord(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        experiment_id=str(row["experiment_id"]) if row["experiment_id"] is not None else None,
        run_id=str(row["run_id"]) if row["run_id"] is not None else None,
        content=str(row["content"]),
        tags_json=str(row["tags_json"]),
        context_json=str(row["context_json"]),
        resolved=bool(row["resolved"]),
        created_at=str(row["created_at"]),
    )
