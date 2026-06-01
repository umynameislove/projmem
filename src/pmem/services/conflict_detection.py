"""portability and failure-analysis conflict detection for import/export bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import posixpath
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.domain.conflicts import ConflictCheckReport, ConflictItem
from pmem.domain.import_bundle import ENTITY_ID_FIELDS, ENTITY_KEYS
from pmem.repositories.portability import ImportJobRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.import_dry_run import (
    _read_bundle_json,
    _resolve_bundle_path,
    dry_run_import_bundle,
)
from pmem.services.project_context import require_project_context
from pmem.services.project_export import export_project
from pmem.utils.hashing import compute_file_hash


def check_bundle_conflicts(
    project_root: str | Path, bundle_path: str | Path
) -> ConflictCheckReport:
    """Detect portability and failure-analysis conflicts without mutating the project database."""

    context = require_project_context(project_root)
    dry_run = dry_run_import_bundle(context.root, bundle_path)
    resolved_bundle = _resolve_bundle_path(context.root, bundle_path)
    display_path = _display_path(context.root, resolved_bundle)
    payload = _read_bundle_json(resolved_bundle)
    validation_errors = tuple(_issue_payload(issue) for issue in dry_run.errors)
    conflicts: dict[str, ConflictItem] = {}

    if isinstance(payload, dict):
        entities = payload.get("entities")
        artifact_index = payload.get("artifact_index")
        if isinstance(entities, dict):
            local_entities = export_project(context.root)["entities"]
            _collect_entity_conflicts(conflicts, local_entities, entities)
            _collect_semantic_duplicates(conflicts, local_entities, entities)
            _collect_stale_baselines(conflicts, local_entities, entities)
            if isinstance(artifact_index, list):
                _collect_artifact_conflicts(conflicts, local_entities, entities, artifact_index)
        _collect_schema_conflicts(conflicts, validation_errors)
        _collect_missing_dependency_conflicts(conflicts, validation_errors)
        _collect_already_applied_conflict(conflicts, context.root, resolved_bundle)

    ordered = tuple(sorted(conflicts.values(), key=lambda item: item.conflict_id))
    return ConflictCheckReport(
        ok=not validation_errors,
        bundle_path=display_path,
        validation_ok=not validation_errors,
        conflict_count=len(ordered),
        conflicts=ordered,
        validation_errors=validation_errors,
        database_mutation=False,
    )


def conflict_check_report_json(report: ConflictCheckReport) -> dict[str, Any]:
    """Return stable machine-readable JSON for `pmem conflict-check --json`."""

    return {
        "ok": report.ok,
        "bundle_path": report.bundle_path,
        "validation_ok": report.validation_ok,
        "conflict_count": report.conflict_count,
        "database_mutation": report.database_mutation,
        "validation_errors": list(report.validation_errors),
        "conflicts": [
            {
                "conflict_id": item.conflict_id,
                "conflict_type": item.conflict_type,
                "entity_type": item.entity_type,
                "entity_id": item.entity_id,
                "severity": item.severity,
                "message": item.message,
                "local_hash": item.local_hash,
                "incoming_hash": item.incoming_hash,
                "action_required": item.action_required,
            }
            for item in report.conflicts
        ],
    }


def _collect_entity_conflicts(
    conflicts: dict[str, ConflictItem],
    local_entities: dict[str, Any],
    incoming_entities: dict[str, Any],
) -> None:
    for entity_type in ENTITY_KEYS:
        id_field = ENTITY_ID_FIELDS[entity_type]
        local_by_id = _records_by_id(local_entities.get(entity_type), id_field)
        incoming_by_id = _records_by_id(incoming_entities.get(entity_type), id_field)
        for entity_id, incoming in incoming_by_id.items():
            local = local_by_id.get(entity_id)
            if local is None:
                continue
            local_hash = _record_hash(local)
            incoming_hash = _record_hash(incoming)
            if local_hash == incoming_hash:
                _add_conflict(
                    conflicts,
                    "same_id_same_hash",
                    entity_type,
                    entity_id,
                    "info",
                    (
                        "Local record has the same id and identical content hash; "
                        "default action is skip."
                    ),
                    local_hash=local_hash,
                    incoming_hash=incoming_hash,
                    action_required="skip",
                )
            else:
                _add_conflict(
                    conflicts,
                    "same_id_different_hash",
                    entity_type,
                    entity_id,
                    "high",
                    "Local record has the same id but a different content hash.",
                    local_hash=local_hash,
                    incoming_hash=incoming_hash,
                )
                _add_conflict(
                    conflicts,
                    "unsafe_overwrite_risk",
                    entity_type,
                    entity_id,
                    "high",
                    (
                        "Importing this record into canonical data would risk "
                        "overwriting local evidence."
                    ),
                    local_hash=local_hash,
                    incoming_hash=incoming_hash,
                )


def _collect_semantic_duplicates(
    conflicts: dict[str, ConflictItem],
    local_entities: dict[str, Any],
    incoming_entities: dict[str, Any],
) -> None:
    local_by_semantic: dict[str, dict[str, Any]] = {}
    for run in _entity_records(local_entities.get("runs")):
        semantic_id = _semantic_run_id(run)
        if semantic_id:
            local_by_semantic.setdefault(semantic_id, run)

    for incoming in _entity_records(incoming_entities.get("runs")):
        semantic_id = _semantic_run_id(incoming)
        if not semantic_id:
            continue
        local = local_by_semantic.get(semantic_id)
        if local is None:
            continue
        incoming_id = str(incoming.get("run_id", ""))
        local_id = str(local.get("run_id", ""))
        if incoming_id == local_id:
            continue
        _add_conflict(
            conflicts,
            "semantic_duplicate",
            "runs",
            incoming_id,
            "medium",
            "Run appears semantically duplicate by command/date/metrics comparison.",
            local_hash=_record_hash(local),
            incoming_hash=_record_hash(incoming),
        )


def _collect_stale_baselines(
    conflicts: dict[str, ConflictItem],
    local_entities: dict[str, Any],
    incoming_entities: dict[str, Any],
) -> None:
    local_experiments = _records_by_id(local_entities.get("experiments"), "id")
    incoming_experiments = _records_by_id(incoming_entities.get("experiments"), "id")
    for experiment_id, incoming in incoming_experiments.items():
        local = local_experiments.get(experiment_id)
        if local is None:
            continue
        local_baseline = _baseline_run_id(local)
        incoming_baseline = _baseline_run_id(incoming)
        if not local_baseline or not incoming_baseline or local_baseline == incoming_baseline:
            continue
        if _timestamp_key(str(incoming.get("updated_at", ""))) < _timestamp_key(
            str(local.get("updated_at", ""))
        ):
            _add_conflict(
                conflicts,
                "stale_baseline",
                "experiments",
                experiment_id,
                "medium",
                "Imported baseline metadata is older than the local experiment baseline.",
                local_hash=_record_hash(local),
                incoming_hash=_record_hash(incoming),
            )


def _collect_artifact_conflicts(
    conflicts: dict[str, ConflictItem],
    local_entities: dict[str, Any],
    incoming_entities: dict[str, Any],
    artifact_index: list[Any],
) -> None:
    local_artifacts = _artifact_map_from_runs(_entity_records(local_entities.get("runs")))
    incoming_artifacts = _artifact_map_from_index(artifact_index)
    for path_key, incoming in incoming_artifacts.items():
        local = local_artifacts.get(path_key)
        if local is None:
            continue
        incoming_hash = _artifact_hash(incoming)
        local_hash = _artifact_hash(local)
        conflict_type = (
            "artifact_hash_mismatch" if local_hash != incoming_hash else "artifact_path_collision"
        )
        severity = "high" if conflict_type == "artifact_hash_mismatch" else "info"
        _add_conflict(
            conflicts,
            conflict_type,
            "artifact_index",
            str(incoming.get("path", path_key)),
            severity,
            "Artifact path collides with a local artifact; review hashes before sharing.",
            local_hash=local_hash,
            incoming_hash=incoming_hash,
            action_required="manual-review" if severity == "high" else "skip",
        )

    indexed_paths = set(incoming_artifacts)
    for run in _entity_records(incoming_entities.get("runs")):
        for artifact in _entity_records(run.get("artifacts")):
            path_key = _normalize_artifact_key(artifact.get("path"))
            if path_key and path_key not in indexed_paths:
                _add_conflict(
                    conflicts,
                    "missing_artifact",
                    "artifact_index",
                    str(artifact.get("path", path_key)),
                    "medium",
                    "Run artifact metadata is missing from artifact_index.",
                    incoming_hash=_artifact_hash(artifact),
                )

    for artifact in _entity_records(artifact_index):
        if artifact.get("content_encoding") == "base64" and not isinstance(
            artifact.get("content_base64"), str
        ):
            _add_conflict(
                conflicts,
                "missing_artifact",
                "artifact_index",
                str(artifact.get("path", "unknown")),
                "medium",
                "Artifact declares inline bytes but content_base64 is missing.",
                incoming_hash=_artifact_hash(artifact),
            )


def _collect_schema_conflicts(
    conflicts: dict[str, ConflictItem],
    validation_errors: tuple[dict[str, str | None], ...],
) -> None:
    mapping = {
        "unsupported_export_format_version": "schema_version_mismatch",
        "unsupported_schema_version": "schema_version_mismatch",
    }
    for error in validation_errors:
        code = error.get("code")
        conflict_type = mapping.get(str(code))
        if conflict_type is None:
            continue
        _add_conflict(
            conflicts,
            conflict_type,
            "bundle",
            str(error.get("field") or code),
            "high",
            "Bundle version is not compatible with this projmem importer.",
        )


def _collect_missing_dependency_conflicts(
    conflicts: dict[str, ConflictItem],
    validation_errors: tuple[dict[str, str | None], ...],
) -> None:
    for error in validation_errors:
        if error.get("code") != "missing_dependency":
            continue
        field = str(error.get("field") or "unknown")
        _add_conflict(
            conflicts,
            "missing_dependency",
            _entity_type_from_field(field),
            field,
            "high",
            "Bundle references data that is absent from the bundle and local database.",
            action_required="reject",
        )


def _collect_already_applied_conflict(
    conflicts: dict[str, ConflictItem],
    project_root: Path,
    bundle_path: Path,
) -> None:
    source_hash = f"sha256:{compute_file_hash(bundle_path)}"
    connection = connect_database(project_database_path(project_root))
    try:
        already_applied = ImportJobRepository(connection).has_source_hash(source_hash)
    finally:
        connection.close()
    if already_applied:
        _add_conflict(
            conflicts,
            "already_applied_package",
            "import_jobs",
            source_hash,
            "medium",
            "This exact bundle file hash already has an import job.",
            incoming_hash=source_hash,
            action_required="skip",
        )


def _add_conflict(
    conflicts: dict[str, ConflictItem],
    conflict_type: str,
    entity_type: str,
    entity_id: str,
    severity: str,
    message: str,
    *,
    local_hash: str | None = None,
    incoming_hash: str | None = None,
    action_required: str = "manual-review",
) -> None:
    conflict_id = _conflict_id(conflict_type, entity_type, entity_id, local_hash, incoming_hash)
    conflicts.setdefault(
        conflict_id,
        ConflictItem(
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            entity_type=entity_type,
            entity_id=entity_id,
            severity=severity,
            message=message,
            local_hash=local_hash,
            incoming_hash=incoming_hash,
            action_required=action_required,
        ),
    )


def _records_by_id(value: Any, id_field: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in _entity_records(value):
        entity_id = record.get(id_field)
        if isinstance(entity_id, str) and entity_id.strip():
            records[entity_id] = record
    return records


def _entity_records(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _record_hash(record: dict[str, Any]) -> str:
    candidate = copy.deepcopy(record)
    candidate["content_hash"] = None
    encoded = json.dumps(
        candidate,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _semantic_run_id(run: dict[str, Any]) -> str:
    command = _normalize_command(run.get("command"))
    date = _timestamp_date(run.get("timestamp"))
    metrics = _sorted_metric_values(run.get("metrics"))
    raw = f"run:{command}:{date}:{metrics}"
    if raw == "run:::":
        return ""
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _normalize_command(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.strip())


def _timestamp_date(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw[:10] if len(raw) >= 10 else ""
    return parsed.astimezone(timezone.utc).date().isoformat()


def _sorted_metric_values(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    parts = []
    for key in sorted(value):
        metric = value[key]
        if isinstance(metric, str | int | float | bool) or metric is None:
            parts.append(f"{key}={metric}")
    return ",".join(parts)


def _baseline_run_id(experiment: dict[str, Any]) -> str | None:
    metadata = experiment.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("baseline_run_id")
    return value if isinstance(value, str) and value.strip() else None


def _timestamp_key(value: str) -> str:
    return _timestamp_date(value) or value


def _artifact_map_from_runs(runs: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for run in runs:
        for artifact in _entity_records(run.get("artifacts")):
            key = _normalize_artifact_key(artifact.get("path"))
            if key:
                artifacts.setdefault(key, artifact)
    return artifacts


def _artifact_map_from_index(index: list[Any]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for item in _entity_records(index):
        key = _normalize_artifact_key(item.get("path"))
        if key:
            artifacts.setdefault(key, item)
    return artifacts


def _normalize_artifact_key(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        return None
    parts = tuple(part for part in value.split("/") if part not in {"", "."})
    if not parts or ".." in parts:
        return None
    return posixpath.normpath("/".join(parts)).casefold()


def _artifact_hash(artifact: dict[str, Any]) -> str | None:
    value = artifact.get("sha256")
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return f"sha256:{value}"
    value = artifact.get("hash")
    return value if isinstance(value, str) and value.startswith("sha256:") else None


def _entity_type_from_field(field: str) -> str:
    if field.startswith("entities."):
        parts = field.split(".")
        return parts[1] if len(parts) > 1 else "entities"
    if field.startswith("artifact_index"):
        return "artifact_index"
    return "bundle"


def _conflict_id(
    conflict_type: str,
    entity_type: str,
    entity_id: str,
    local_hash: str | None,
    incoming_hash: str | None,
) -> str:
    raw = json.dumps(
        {
            "conflict_type": conflict_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "local_hash": local_hash,
            "incoming_hash": incoming_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"conflict_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _issue_payload(issue: Any) -> dict[str, str | None]:
    return {
        "code": issue.code,
        "field": issue.field,
        "message": issue.message,
    }


def _display_path(project_root: Path, resolved_path: Path) -> str:
    try:
        return resolved_path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.name
