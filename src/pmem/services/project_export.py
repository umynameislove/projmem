"""project export project export service.

The export is intentionally local and deterministic: it reads schema-v1 SQLite
rows for the current project, parses JSON metadata into JSON objects, and never
reads artifact file contents.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.repositories.decisions import DecisionRecord, DecisionRepository
from pmem.repositories.experiments import ExperimentRecord, ExperimentRepository
from pmem.repositories.failures import FailureRecord, FailureRepository
from pmem.repositories.notes import NoteRecord, NoteRepository
from pmem.repositories.projects import ProjectRecord
from pmem.repositories.runs import RunRecord, RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.repositories.tracked_paths import TrackedPathRecord, TrackedPathRepository
from pmem.services.project_context import require_project_context

EXPORT_SCHEMA_VERSION = "schema-v1"


def export_project(project_root: str | Path) -> dict[str, Any]:
    """Return the project export JSON export payload for the initialized project."""

    context = require_project_context(project_root)
    connection = connect_database(project_database_path(context.root))
    try:
        project_id = context.project.id
        experiments = ExperimentRepository(connection).list_for_project(project_id)
        runs = RunRepository(connection).list_for_project(project_id)
        tracked_paths = TrackedPathRepository(connection).list_for_project(project_id)
        failures = FailureRepository(connection).list_for_project(project_id)
        decisions = DecisionRepository(connection).list_for_project(project_id)
        notes = NoteRepository(connection).list_for_project(project_id)
    finally:
        connection.close()

    return {
        "export_at": _utc_now(),
        "schema_version": EXPORT_SCHEMA_VERSION,
        "project_id": context.project.id,
        "entities": {
            "projects": [_project_payload(context.project)],
            "experiments": [_experiment_payload(record) for record in experiments],
            "runs": [_run_payload(record) for record in runs],
            "failures": [_failure_payload(record) for record in failures],
            "decisions": [_decision_payload(record) for record in decisions],
            "notes": [_note_payload(record) for record in notes],
            "tracked_paths": [_tracked_path_payload(record) for record in tracked_paths],
        },
    }


def _project_payload(record: ProjectRecord) -> dict[str, Any]:
    payload = _record_payload(record)
    payload["target"] = _json_object(record.target_json)
    del payload["target_json"]
    return payload


def _experiment_payload(record: ExperimentRecord) -> dict[str, Any]:
    payload = _record_payload(record)
    payload["target"] = _json_object(record.target_json) if record.target_json is not None else None
    payload["metadata"] = _json_object(record.metadata_json)
    del payload["target_json"]
    del payload["metadata_json"]
    return payload


def _run_payload(record: RunRecord) -> dict[str, Any]:
    payload = _record_payload(record)
    payload["env"] = _json_object(record.env_json)
    payload["config"] = _json_object(record.config_json)
    payload["metrics"] = _json_object(record.metrics_json)
    payload["artifacts"] = _json_array(record.artifacts_json)
    payload["git"] = _json_object(record.git_json)
    payload["evaluation"] = _json_object(record.evaluation_json)
    payload["failure_candidates"] = _json_array(record.failure_candidates_json)
    for key in (
        "env_json",
        "config_json",
        "metrics_json",
        "artifacts_json",
        "git_json",
        "evaluation_json",
        "failure_candidates_json",
    ):
        del payload[key]
    return payload


def _failure_payload(record: FailureRecord) -> dict[str, Any]:
    payload = _record_payload(record)
    payload["tags"] = _json_array(record.tags_json)
    del payload["tags_json"]
    return payload


def _decision_payload(record: DecisionRecord) -> dict[str, Any]:
    payload = _record_payload(record)
    payload["related_experiments"] = _json_array(record.related_experiments_json)
    del payload["related_experiments_json"]
    return payload


def _note_payload(record: NoteRecord) -> dict[str, Any]:
    payload = _record_payload(record)
    payload["tags"] = _json_array(record.tags_json)
    payload["context"] = _json_object(record.context_json)
    del payload["tags_json"]
    del payload["context_json"]
    return payload


def _tracked_path_payload(record: TrackedPathRecord) -> dict[str, Any]:
    return _record_payload(record)


def _record_payload(record: Any) -> dict[str, Any]:
    return dict(asdict(record))


def _json_object(raw_json: str) -> dict[str, Any]:
    value = json.loads(raw_json)
    return value if isinstance(value, dict) else {}


def _json_array(raw_json: str) -> list[Any]:
    value = json.loads(raw_json)
    return value if isinstance(value, list) else []


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
