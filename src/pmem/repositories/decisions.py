"""Decision table repository.

Decisions are durable project choices and rationale. This repository handles
only SQLite persistence; service code validates project/experiment context.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from pmem.repositories.sqlite import execute, query_one


@dataclass(frozen=True)
class DecisionRecord:
    """SQLite representation of one decision row."""

    id: str
    project_id: str
    experiment_id: str | None
    description: str
    rationale: str | None
    related_experiments_json: str
    created_at: str
    author: str | None


class DecisionRepository:
    """Read and write decisions through parameterized SQLite queries."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def create(
        self,
        *,
        decision_id: str,
        project_id: str,
        description: str,
        created_at: str,
        experiment_id: str | None = None,
        rationale: str | None = None,
        related_experiments: list[str] | None = None,
        author: str | None = None,
    ) -> DecisionRecord:
        """Insert one decision row and commit it atomically."""

        related_json = json.dumps(
            related_experiments or [],
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            self._connection.execute("BEGIN")
            execute(
                self._connection,
                """
                INSERT INTO decisions(
                    id, project_id, experiment_id, description, rationale,
                    related_experiments_json, created_at, author
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    project_id,
                    experiment_id,
                    description,
                    rationale,
                    related_json,
                    created_at,
                    author,
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

        return DecisionRecord(
            id=decision_id,
            project_id=project_id,
            experiment_id=experiment_id,
            description=description,
            rationale=rationale,
            related_experiments_json=related_json,
            created_at=created_at,
            author=author,
        )

    def get_by_id(self, decision_id: str) -> DecisionRecord | None:
        """Return a decision by id."""

        row = query_one(
            self._connection,
            """
            SELECT id, project_id, experiment_id, description, rationale,
                   related_experiments_json, created_at, author
            FROM decisions
            WHERE id = ?
            """,
            (decision_id,),
        )
        return _decision_from_row(row) if row is not None else None

    def list_for_project(self, project_id: str) -> tuple[DecisionRecord, ...]:
        """Return project decisions in newest-first order."""

        rows = execute(
            self._connection,
            """
            SELECT id, project_id, experiment_id, description, rationale,
                   related_experiments_json, created_at, author
            FROM decisions
            WHERE project_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (project_id,),
        ).fetchall()
        return tuple(_decision_from_row(row) for row in rows)


def _decision_from_row(row: sqlite3.Row) -> DecisionRecord:
    """Convert a SQLite row to the typed repository record."""

    return DecisionRecord(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        experiment_id=str(row["experiment_id"]) if row["experiment_id"] is not None else None,
        description=str(row["description"]),
        rationale=str(row["rationale"]) if row["rationale"] is not None else None,
        related_experiments_json=str(row["related_experiments_json"]),
        created_at=str(row["created_at"]),
        author=str(row["author"]) if row["author"] is not None else None,
    )
