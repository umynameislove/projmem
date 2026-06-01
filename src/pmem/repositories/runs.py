"""Run table repository.

run capture stores command execution evidence in `runs`. This repository keeps SQL,
JSON serialization, row mapping, and transaction rollback local to persistence
code so the run service can focus on filesystem and subprocess policy.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from pmem.errors import PmemPersistenceError
from pmem.repositories.sqlite import execute, query_one


@dataclass(frozen=True)
class RunRecord:
    """SQLite representation of one captured run."""

    run_id: str
    experiment_id: str
    name: str | None
    command: str
    cwd: str
    exit_code: int | None
    status: str
    duration_sec: float | None
    seed: str | None
    stdout_path: str | None
    stderr_path: str | None
    stdout_preview: str | None
    stderr_preview: str | None
    env_json: str
    config_json: str
    config_hash: str | None
    metrics_json: str
    artifacts_json: str
    git_json: str
    evaluation_json: str
    failure_candidates_json: str
    timestamp: str


class RunRepository:
    """Read and write run records with parameterized SQLite queries."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def create(
        self,
        *,
        run_id: str,
        experiment_id: str,
        command: str,
        cwd: str,
        status: str,
        timestamp: str,
        name: str | None = None,
        exit_code: int | None = None,
        duration_sec: float | None = None,
        seed: str | None = None,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
        stdout_preview: str | None = None,
        stderr_preview: str | None = None,
        env: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        config_hash: str | None = None,
        metrics: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        git: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
        failure_candidates: list[dict[str, Any]] | None = None,
    ) -> RunRecord:
        """Insert one run row and commit it atomically."""

        env_json = _stable_json_object(env)
        config_json = _stable_json_object(config)
        metrics_json = _stable_json_object(metrics)
        artifacts_json = _stable_json_array(artifacts)
        git_json = _stable_json_object(git)
        evaluation_json = _stable_json_object(evaluation)
        failure_candidates_json = _stable_json_array(failure_candidates)

        try:
            self._connection.execute("BEGIN")
            execute(
                self._connection,
                """
                INSERT INTO runs(
                    run_id, experiment_id, name, command, cwd, exit_code, status,
                    duration_sec, seed, stdout_path, stderr_path, stdout_preview,
                    stderr_preview, env_json, config_json, config_hash, metrics_json,
                    artifacts_json, git_json, evaluation_json, failure_candidates_json,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_id,
                    name,
                    command,
                    cwd,
                    exit_code,
                    status,
                    duration_sec,
                    seed,
                    stdout_path,
                    stderr_path,
                    stdout_preview,
                    stderr_preview,
                    env_json,
                    config_json,
                    config_hash,
                    metrics_json,
                    artifacts_json,
                    git_json,
                    evaluation_json,
                    failure_candidates_json,
                    timestamp,
                ),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

        return RunRecord(
            run_id=run_id,
            experiment_id=experiment_id,
            name=name,
            command=command,
            cwd=cwd,
            exit_code=exit_code,
            status=status,
            duration_sec=duration_sec,
            seed=seed,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout_preview=stdout_preview,
            stderr_preview=stderr_preview,
            env_json=env_json,
            config_json=config_json,
            config_hash=config_hash,
            metrics_json=metrics_json,
            artifacts_json=artifacts_json,
            git_json=git_json,
            evaluation_json=evaluation_json,
            failure_candidates_json=failure_candidates_json,
            timestamp=timestamp,
        )

    def get_by_id(self, run_id: str) -> RunRecord | None:
        """Return a run by stable id."""

        row = query_one(
            self._connection,
            """
            SELECT run_id, experiment_id, name, command, cwd, exit_code, status,
                   duration_sec, seed, stdout_path, stderr_path, stdout_preview,
                   stderr_preview, env_json, config_json, config_hash, metrics_json,
                   artifacts_json, git_json, evaluation_json, failure_candidates_json,
                   timestamp
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        )
        return _run_from_row(row) if row is not None else None

    def list_for_experiment(self, experiment_id: str) -> tuple[RunRecord, ...]:
        """Return runs for one experiment in newest-first order."""

        rows = execute(
            self._connection,
            """
            SELECT run_id, experiment_id, name, command, cwd, exit_code, status,
                   duration_sec, seed, stdout_path, stderr_path, stdout_preview,
                   stderr_preview, env_json, config_json, config_hash, metrics_json,
                   artifacts_json, git_json, evaluation_json, failure_candidates_json,
                   timestamp
            FROM runs
            WHERE experiment_id = ?
            ORDER BY timestamp DESC, run_id DESC
            """,
            (experiment_id,),
        ).fetchall()
        return tuple(_run_from_row(row) for row in rows)

    def list_for_project(self, project_id: str) -> tuple[RunRecord, ...]:
        """Return runs for one project in newest-first order."""

        rows = execute(
            self._connection,
            """
            SELECT runs.run_id, runs.experiment_id, runs.name, runs.command, runs.cwd,
                   runs.exit_code, runs.status, runs.duration_sec, runs.seed,
                   runs.stdout_path, runs.stderr_path, runs.stdout_preview,
                   runs.stderr_preview, runs.env_json, runs.config_json,
                   runs.config_hash, runs.metrics_json, runs.artifacts_json,
                   runs.git_json, runs.evaluation_json, runs.failure_candidates_json,
                   runs.timestamp
            FROM runs
            JOIN experiments ON experiments.id = runs.experiment_id
            WHERE experiments.project_id = ?
            ORDER BY runs.timestamp DESC, runs.run_id DESC
            """,
            (project_id,),
        ).fetchall()
        return tuple(_run_from_row(row) for row in rows)

    def update_evaluation(self, *, run_id: str, evaluation: dict[str, Any]) -> RunRecord:
        """Replace run evaluation JSON and return the updated row."""

        evaluation_json = _stable_json_object(evaluation)
        try:
            self._connection.execute("BEGIN")
            execute(
                self._connection,
                """
                UPDATE runs
                SET evaluation_json = ?
                WHERE run_id = ?
                """,
                (evaluation_json, run_id),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

        updated = self.get_by_id(run_id)
        if updated is None:
            raise PmemPersistenceError()
        return updated


def _stable_json_object(value: dict[str, Any] | None) -> str:
    """Return compact deterministic JSON object text."""

    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def _stable_json_array(value: list[dict[str, Any]] | None) -> str:
    """Return compact deterministic JSON array text."""

    return json.dumps(value or [], sort_keys=True, separators=(",", ":"))


def _run_from_row(row: sqlite3.Row) -> RunRecord:
    """Convert a SQLite row to the typed repository record."""

    return RunRecord(
        run_id=str(row["run_id"]),
        experiment_id=str(row["experiment_id"]),
        name=str(row["name"]) if row["name"] is not None else None,
        command=str(row["command"]),
        cwd=str(row["cwd"]),
        exit_code=int(row["exit_code"]) if row["exit_code"] is not None else None,
        status=str(row["status"]),
        duration_sec=float(row["duration_sec"]) if row["duration_sec"] is not None else None,
        seed=str(row["seed"]) if row["seed"] is not None else None,
        stdout_path=str(row["stdout_path"]) if row["stdout_path"] is not None else None,
        stderr_path=str(row["stderr_path"]) if row["stderr_path"] is not None else None,
        stdout_preview=(str(row["stdout_preview"]) if row["stdout_preview"] is not None else None),
        stderr_preview=(str(row["stderr_preview"]) if row["stderr_preview"] is not None else None),
        env_json=str(row["env_json"]),
        config_json=str(row["config_json"]),
        config_hash=str(row["config_hash"]) if row["config_hash"] is not None else None,
        metrics_json=str(row["metrics_json"]),
        artifacts_json=str(row["artifacts_json"]),
        git_json=str(row["git_json"]),
        evaluation_json=str(row["evaluation_json"]),
        failure_candidates_json=str(row["failure_candidates_json"]),
        timestamp=str(row["timestamp"]),
    )
