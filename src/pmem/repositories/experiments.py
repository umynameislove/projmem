"""Experiment table repository.

The repository owns only SQLite persistence for `experiments`. It does not
decide when a default experiment should exist; the run service calls the small
primitive here when run capture needs a storage parent for captured runs.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from pmem.errors import PmemPersistenceError
from pmem.repositories.sqlite import execute, query_one


@dataclass(frozen=True)
class ExperimentRecord:
    """SQLite representation of one experiment row."""

    id: str
    project_id: str
    name: str
    hypothesis: str | None
    status: str
    is_baseline: bool
    primary_metric: str | None
    target_json: str | None
    created_at: str
    updated_at: str
    metadata_json: str


class ExperimentRepository:
    """Read and write experiment rows through parameterized SQLite queries."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def create(
        self,
        *,
        experiment_id: str,
        project_id: str,
        name: str,
        created_at: str,
        updated_at: str,
        hypothesis: str | None = None,
        status: str = "active",
        is_baseline: bool = False,
        primary_metric: str | None = None,
        target: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        """Insert one experiment row and commit it atomically."""

        target_json = (
            json.dumps(target, sort_keys=True, separators=(",", ":"))
            if target is not None
            else None
        )
        metadata_json = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))
        try:
            self._connection.execute("BEGIN")
            execute(
                self._connection,
                """
                INSERT INTO experiments(
                    id, project_id, name, hypothesis, status, is_baseline,
                    primary_metric, target_json, created_at, updated_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    project_id,
                    name,
                    hypothesis,
                    status,
                    1 if is_baseline else 0,
                    primary_metric,
                    target_json,
                    created_at,
                    updated_at,
                    metadata_json,
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

        return ExperimentRecord(
            id=experiment_id,
            project_id=project_id,
            name=name,
            hypothesis=hypothesis,
            status=status,
            is_baseline=is_baseline,
            primary_metric=primary_metric,
            target_json=target_json,
            created_at=created_at,
            updated_at=updated_at,
            metadata_json=metadata_json,
        )

    def get_by_id(self, experiment_id: str) -> ExperimentRecord | None:
        """Return an experiment by stable id."""

        row = query_one(
            self._connection,
            """
            SELECT id, project_id, name, hypothesis, status, is_baseline,
                   primary_metric, target_json, created_at, updated_at, metadata_json
            FROM experiments
            WHERE id = ?
            """,
            (experiment_id,),
        )
        return _experiment_from_row(row) if row is not None else None

    def get_by_project_and_name(
        self,
        project_id: str,
        name: str,
    ) -> ExperimentRecord | None:
        """Return an experiment by project-local name."""

        row = query_one(
            self._connection,
            """
            SELECT id, project_id, name, hypothesis, status, is_baseline,
                   primary_metric, target_json, created_at, updated_at, metadata_json
            FROM experiments
            WHERE project_id = ? AND name = ?
            """,
            (project_id, name),
        )
        return _experiment_from_row(row) if row is not None else None

    def list_for_project(self, project_id: str) -> tuple[ExperimentRecord, ...]:
        """Return all experiments for one project in deterministic order."""

        rows = execute(
            self._connection,
            """
            SELECT id, project_id, name, hypothesis, status, is_baseline,
                   primary_metric, target_json, created_at, updated_at, metadata_json
            FROM experiments
            WHERE project_id = ?
            ORDER BY created_at, id
            """,
            (project_id,),
        ).fetchall()
        return tuple(_experiment_from_row(row) for row in rows)

    def get_or_create_default(
        self,
        *,
        project_id: str,
        timestamp: str,
    ) -> ExperimentRecord:
        """Return the project default experiment, creating it once if missing."""

        existing = self.get_by_project_and_name(project_id, "default")
        if existing is not None:
            return existing

        try:
            return self.create(
                experiment_id=f"exp_default_{project_id}",
                project_id=project_id,
                name="default",
                created_at=timestamp,
                updated_at=timestamp,
                metadata={"created_by": "pmem run"},
            )
        except PmemPersistenceError:
            # If another process created the default row between read and insert,
            # return the now-existing row instead of surfacing a duplicate error.
            existing_after_conflict = self.get_by_project_and_name(project_id, "default")
            if existing_after_conflict is not None:
                return existing_after_conflict
            raise

    def update_metadata(
        self,
        *,
        experiment_id: str,
        metadata: dict[str, Any],
        updated_at: str,
    ) -> ExperimentRecord:
        """Replace experiment metadata JSON and return the updated row."""

        metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        try:
            self._connection.execute("BEGIN")
            execute(
                self._connection,
                """
                UPDATE experiments
                SET metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (metadata_json, updated_at, experiment_id),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

        updated = self.get_by_id(experiment_id)
        if updated is None:
            raise PmemPersistenceError()
        return updated


def _experiment_from_row(row: sqlite3.Row) -> ExperimentRecord:
    """Convert a SQLite row to the typed repository record."""

    return ExperimentRecord(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        name=str(row["name"]),
        hypothesis=str(row["hypothesis"]) if row["hypothesis"] is not None else None,
        status=str(row["status"]),
        is_baseline=bool(row["is_baseline"]),
        primary_metric=str(row["primary_metric"]) if row["primary_metric"] is not None else None,
        target_json=str(row["target_json"]) if row["target_json"] is not None else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        metadata_json=str(row["metadata_json"]),
    )
