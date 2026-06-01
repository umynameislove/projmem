"""Project table repository.

This module owns persistence for the `projects` table only. It does not create
`.pmem/`, decide init policy, or render CLI output; those responsibilities stay
in service and CLI layers.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from pmem.repositories.sqlite import execute, query_one


@dataclass(frozen=True)
class ProjectRecord:
    """SQLite representation of one project memory root."""

    id: str
    name: str
    goal: str | None
    current_objective: str | None
    primary_metric: str | None
    metric_direction: str | None
    target_json: str
    created_at: str
    updated_at: str


class ProjectRepository:
    """Read and write project records through parameterized SQLite queries."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def create(
        self,
        *,
        project_id: str,
        name: str,
        created_at: str,
        updated_at: str,
        goal: str | None = None,
        current_objective: str | None = None,
        primary_metric: str | None = None,
        metric_direction: str | None = None,
        target: dict[str, Any] | None = None,
        failure_criteria: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProjectRecord:
        """Insert one project row and commit it atomically."""

        target_json = json.dumps(target or {}, sort_keys=True, separators=(",", ":"))
        failure_criteria_json = json.dumps(
            failure_criteria or [],
            sort_keys=True,
            separators=(",", ":"),
        )
        metadata_json = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))
        try:
            self._connection.execute("BEGIN")
            execute(
                self._connection,
                """
                INSERT INTO projects(
                    id, name, goal, current_objective, primary_metric, metric_direction,
                    target_json, failure_criteria_json, created_at, updated_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    name,
                    goal,
                    current_objective,
                    primary_metric,
                    metric_direction,
                    target_json,
                    failure_criteria_json,
                    created_at,
                    updated_at,
                    metadata_json,
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

        return ProjectRecord(
            id=project_id,
            name=name,
            goal=goal,
            current_objective=current_objective,
            primary_metric=primary_metric,
            metric_direction=metric_direction,
            target_json=target_json,
            created_at=created_at,
            updated_at=updated_at,
        )

    def get_by_id(self, project_id: str) -> ProjectRecord | None:
        """Return a project by stable id."""

        row = query_one(
            self._connection,
            """
            SELECT id, name, goal, current_objective, primary_metric, metric_direction,
                   target_json, created_at, updated_at
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        )
        return _project_from_row(row) if row is not None else None

    def get_by_name(self, name: str) -> ProjectRecord | None:
        """Return a project by unique local name."""

        row = query_one(
            self._connection,
            """
            SELECT id, name, goal, current_objective, primary_metric, metric_direction,
                   target_json, created_at, updated_at
            FROM projects
            WHERE name = ?
            """,
            (name,),
        )
        return _project_from_row(row) if row is not None else None

    def list_projects(self) -> tuple[ProjectRecord, ...]:
        """Return all project rows in deterministic order."""

        rows = execute(
            self._connection,
            """
            SELECT id, name, goal, current_objective, primary_metric, metric_direction,
                   target_json, created_at, updated_at
            FROM projects
            ORDER BY created_at, id
            """,
        ).fetchall()
        return tuple(_project_from_row(row) for row in rows)


def _project_from_row(row: sqlite3.Row) -> ProjectRecord:
    """Convert a SQLite row to the typed repository record."""

    return ProjectRecord(
        id=str(row["id"]),
        name=str(row["name"]),
        goal=str(row["goal"]) if row["goal"] is not None else None,
        current_objective=(
            str(row["current_objective"]) if row["current_objective"] is not None else None
        ),
        primary_metric=str(row["primary_metric"]) if row["primary_metric"] is not None else None,
        metric_direction=(
            str(row["metric_direction"]) if row["metric_direction"] is not None else None
        ),
        target_json=str(row["target_json"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
