"""portability and failure-analysis import dry-run validation service."""

from __future__ import annotations

import copy
import hashlib
import json
import posixpath
import re
from pathlib import Path, PureWindowsPath
from typing import Any

from pmem.domain.import_bundle import (
    BUNDLE_SCHEMA_VERSION,
    ENTITY_ID_FIELDS,
    ENTITY_KEYS,
    EXPORT_FORMAT_VERSION,
    FREE_TEXT_FIELDS,
    ConflictPreviewItem,
    ImportDryRunReport,
    ImportValidationIssue,
    PrivacyReviewItem,
)
from pmem.errors import PmemNotFoundError, PmemSecurityError, PmemValidationError
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_context import require_project_context

MAX_BUNDLE_BYTES = 25 * 1024 * 1024
HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ARTIFACT_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PROVENANCE_ALLOWED_KEYS = frozenset(
    {
        "tool",
        "tool_version",
        "source",
        "project_name",
        "git_commit_hash",
        "git_dirty",
    }
)


def dry_run_import_bundle(project_root: str | Path, bundle_path: str | Path) -> ImportDryRunReport:
    """Validate a bundle and return a report without writing to the project DB."""

    context = require_project_context(project_root)
    resolved_bundle = _resolve_bundle_path(context.root, bundle_path)
    display_path = _display_path(context.root, resolved_bundle)
    payload = _read_bundle_json(resolved_bundle)
    local_ids = _load_local_ids(project_database_path(context.root))

    errors: list[ImportValidationIssue] = []
    warnings: list[ImportValidationIssue] = []
    privacy_review: list[PrivacyReviewItem] = []
    conflicts: list[ConflictPreviewItem] = []
    entity_counts = {key: 0 for key in ENTITY_KEYS}
    export_format_version: str | None = None
    schema_version: str | None = None

    if not isinstance(payload, dict):
        errors.append(
            ImportValidationIssue(
                code="invalid_bundle_shape",
                field="$",
                message=(
                    "Bundle must be a JSON object with manifest, entities, "
                    "artifact_index, privacy_flags, and provenance."
                ),
            )
        )
        return _report(
            display_path,
            export_format_version,
            schema_version,
            entity_counts,
            errors,
            warnings,
            privacy_review,
            conflicts,
        )

    _validate_top_level_keys(payload, errors)
    manifest = payload.get("manifest")
    entities = payload.get("entities")
    artifact_index = payload.get("artifact_index")
    privacy_flags = payload.get("privacy_flags")
    provenance = payload.get("provenance")

    if isinstance(manifest, dict):
        export_format_version = _string_or_none(manifest.get("export_format_version"))
        schema_version = _string_or_none(manifest.get("schema_version"))
        _validate_manifest(manifest, payload, errors)
    else:
        errors.append(
            ImportValidationIssue(
                code="invalid_manifest",
                field="manifest",
                message="Bundle manifest must be an object.",
            )
        )

    entity_id_sets: dict[str, set[str]] = {key: set() for key in ENTITY_KEYS}
    if isinstance(entities, dict):
        entity_counts = _validate_entities(entities, entity_id_sets, errors)
        _validate_entity_counts(manifest, entity_counts, errors)
        _validate_foreign_keys(entities, entity_id_sets, local_ids, errors)
        privacy_review.extend(_collect_free_text_review(entities))
        conflicts.extend(_collect_conflict_preview(entities, local_ids))
    else:
        errors.append(
            ImportValidationIssue(
                code="invalid_entities",
                field="entities",
                message=(
                    "Bundle entities must be an object containing all local-memory entity arrays."
                ),
            )
        )

    if not isinstance(artifact_index, list):
        errors.append(
            ImportValidationIssue(
                code="invalid_artifact_index",
                field="artifact_index",
                message="artifact_index must be an array, even when empty.",
            )
        )
    else:
        _validate_artifact_index(artifact_index, manifest, entity_id_sets, local_ids, errors)
        privacy_review.extend(_collect_artifact_review(artifact_index))

    if not isinstance(privacy_flags, list):
        errors.append(
            ImportValidationIssue(
                code="invalid_privacy_flags",
                field="privacy_flags",
                message="privacy_flags must be an array, even when empty.",
            )
        )
    else:
        privacy_review.append(
            PrivacyReviewItem(
                field="privacy_flags",
                count=len(privacy_flags),
                message="Review bundle privacy flags before sharing or applying imported memory.",
            )
        )

    if isinstance(provenance, dict):
        _validate_provenance(provenance, errors)
    else:
        errors.append(
            ImportValidationIssue(
                code="invalid_provenance",
                field="provenance",
                message="provenance must be an object with safe local export metadata.",
            )
        )

    if export_format_version != EXPORT_FORMAT_VERSION:
        errors.append(
            ImportValidationIssue(
                code="unsupported_export_format_version",
                field="manifest.export_format_version",
                message=f"Unsupported export_format_version. Expected {EXPORT_FORMAT_VERSION}.",
            )
        )

    if schema_version != BUNDLE_SCHEMA_VERSION:
        errors.append(
            ImportValidationIssue(
                code="unsupported_schema_version",
                field="manifest.schema_version",
                message=f"Unsupported schema_version. Expected {BUNDLE_SCHEMA_VERSION}.",
            )
        )

    return _report(
        display_path,
        export_format_version,
        schema_version,
        entity_counts,
        errors,
        warnings,
        privacy_review,
        conflicts,
    )


def import_dry_run_report_json(report: ImportDryRunReport) -> dict[str, Any]:
    """Return the stable CLI JSON payload for an import dry-run report."""

    return {
        "ok": report.ok,
        "dry_run": report.dry_run,
        "bundle_path": report.bundle_path,
        "export_format_version": report.export_format_version,
        "schema_version": report.schema_version,
        "entity_counts": report.entity_counts,
        "errors": [_issue_payload(issue) for issue in report.errors],
        "warnings": [_issue_payload(issue) for issue in report.warnings],
        "privacy_review": [
            {
                "field": item.field,
                "count": item.count,
                "message": item.message,
            }
            for item in report.privacy_review
        ],
        "conflicts": [
            {
                "conflict_type": item.conflict_type,
                "entity_type": item.entity_type,
                "entity_id": item.entity_id,
                "message": item.message,
            }
            for item in report.conflicts
        ],
        "database_mutation": report.database_mutation,
    }


def _resolve_bundle_path(project_root: Path, user_path: str | Path) -> Path:
    raw_text = str(user_path).strip()
    if not raw_text:
        raise PmemValidationError("Bundle path cannot be blank.")

    if "\x00" in raw_text or any(ord(c) < 32 for c in raw_text):
        raise PmemSecurityError("Bundle path contains unsafe control characters.")
    raw_path = Path(raw_text)
    if raw_path.is_absolute():
        raise PmemSecurityError("Bundle path must be project-relative.")

    if any(part.lower() == ".pmem" for part in raw_path.parts):
        raise PmemSecurityError("Bundle path cannot point inside .pmem.")

    root = project_root.resolve()
    unresolved = root / raw_path
    resolved = unresolved.resolve(strict=False)
    if root != resolved and root not in resolved.parents:
        raise PmemSecurityError("Bundle path must stay inside the project.")

    _reject_symlink_parts(root, raw_path)

    if not resolved.exists():
        raise PmemNotFoundError("Bundle file was not found.")
    if resolved.is_dir():
        raise PmemSecurityError("Bundle path must point to a file, not a directory.")
    if not resolved.is_file():
        raise PmemSecurityError("Bundle path must point to a regular file.")

    return resolved


def _reject_symlink_parts(root: Path, raw_path: Path) -> None:
    current = root
    for part in raw_path.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.exists() and current.is_symlink():
            raise PmemSecurityError("Bundle path cannot contain symlinks.")


def _display_path(project_root: Path, resolved_path: Path) -> str:
    try:
        return resolved_path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.name


def _read_bundle_json(path: Path) -> Any:
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise PmemValidationError("Bundle file is too large for dry-run validation.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise PmemValidationError("Bundle file must be UTF-8 JSON.") from exc
    except json.JSONDecodeError:
        return {"__invalid_json__": True}


def _validate_top_level_keys(
    payload: dict[str, Any],
    errors: list[ImportValidationIssue],
) -> None:
    required = {"manifest", "entities", "artifact_index", "privacy_flags", "provenance"}
    actual = set(payload)
    for key in sorted(required - actual):
        errors.append(
            ImportValidationIssue(
                code="missing_top_level_key",
                field=key,
                message=f"Bundle is missing required top-level key `{key}`.",
            )
        )
    for key in sorted(actual - required):
        if key == "__invalid_json__":
            errors.append(
                ImportValidationIssue(
                    code="invalid_json",
                    field="$",
                    message="Invalid JSON bundle. Re-export the bundle and try dry-run again.",
                )
            )
        else:
            errors.append(
                ImportValidationIssue(
                    code="unknown_top_level_key",
                    field=key,
                    message=f"Unknown top-level key `{key}` is not allowed in export bundle v1.",
                )
            )


def _validate_manifest(
    manifest: dict[str, Any],
    payload: dict[str, Any],
    errors: list[ImportValidationIssue],
) -> None:
    required = {
        "export_format_version",
        "schema_version",
        "generated_at",
        "freeze_timestamp",
        "project_id",
        "entity_counts",
        "artifact_count",
        "canonical_json",
        "manifest_hash",
        "payload_hash",
    }
    for key in sorted(required - set(manifest)):
        errors.append(
            ImportValidationIssue(
                code="missing_manifest_key",
                field=f"manifest.{key}",
                message=f"Manifest is missing `{key}`.",
            )
        )

    manifest_hash = manifest.get("manifest_hash")
    payload_hash = manifest.get("payload_hash")
    if not _is_hash_string(manifest_hash):
        errors.append(
            ImportValidationIssue(
                code="invalid_manifest_hash",
                field="manifest.manifest_hash",
                message="manifest_hash must use sha256:<64 lowercase hex chars>.",
            )
        )
    elif manifest_hash != _manifest_hash(manifest):
        errors.append(
            ImportValidationIssue(
                code="manifest_hash_mismatch",
                field="manifest.manifest_hash",
                message="manifest_hash mismatch. The bundle manifest may have been edited.",
            )
        )

    if not _is_hash_string(payload_hash):
        errors.append(
            ImportValidationIssue(
                code="invalid_payload_hash",
                field="manifest.payload_hash",
                message="payload_hash must use sha256:<64 lowercase hex chars>.",
            )
        )
    elif payload_hash != _payload_hash(payload):
        errors.append(
            ImportValidationIssue(
                code="payload_hash_mismatch",
                field="manifest.payload_hash",
                message="payload_hash mismatch. The bundle payload may have been edited.",
            )
        )


def _validate_entities(
    entities: dict[str, Any],
    entity_id_sets: dict[str, set[str]],
    errors: list[ImportValidationIssue],
) -> dict[str, int]:
    counts = {key: 0 for key in ENTITY_KEYS}
    actual_keys = set(entities)

    for key in sorted(set(ENTITY_KEYS) - actual_keys):
        errors.append(
            ImportValidationIssue(
                code="missing_entity_array",
                field=f"entities.{key}",
                message=f"entities.{key} must be present as an array.",
            )
        )

    for key in sorted(actual_keys - set(ENTITY_KEYS)):
        errors.append(
            ImportValidationIssue(
                code="unknown_entity_array",
                field=f"entities.{key}",
                message=f"Unknown entity array `{key}` is not allowed in export bundle v1.",
            )
        )

    for entity_type in ENTITY_KEYS:
        value = entities.get(entity_type)
        if not isinstance(value, list):
            errors.append(
                ImportValidationIssue(
                    code="invalid_entity_array",
                    field=f"entities.{entity_type}",
                    message=f"entities.{entity_type} must be an array.",
                )
            )
            continue
        counts[entity_type] = len(value)
        id_field = ENTITY_ID_FIELDS[entity_type]
        for index, item in enumerate(value):
            field_prefix = f"entities.{entity_type}[{index}]"
            if not isinstance(item, dict):
                errors.append(
                    ImportValidationIssue(
                        code="invalid_entity",
                        field=field_prefix,
                        message=f"{field_prefix} must be an object.",
                    )
                )
                continue
            entity_id = item.get(id_field)
            if not _is_nonblank_string(entity_id):
                errors.append(
                    ImportValidationIssue(
                        code="missing_entity_id",
                        field=f"{field_prefix}.{id_field}",
                        message=f"{field_prefix}.{id_field} must be a non-empty string.",
                    )
                )
                continue
            if entity_id in entity_id_sets[entity_type]:
                errors.append(
                    ImportValidationIssue(
                        code="duplicate_entity_id",
                        field=f"{field_prefix}.{id_field}",
                        message=f"Duplicate {entity_type} id `{entity_id}` appears in this bundle.",
                    )
                )
            entity_id_sets[entity_type].add(str(entity_id))
    return counts


def _validate_entity_counts(
    manifest: Any,
    entity_counts: dict[str, int],
    errors: list[ImportValidationIssue],
) -> None:
    if not isinstance(manifest, dict):
        return
    expected = manifest.get("entity_counts")
    if not isinstance(expected, dict):
        errors.append(
            ImportValidationIssue(
                code="invalid_entity_counts",
                field="manifest.entity_counts",
                message="manifest.entity_counts must be an object matching entities array lengths.",
            )
        )
        return
    for entity_type, actual_count in entity_counts.items():
        expected_count = expected.get(entity_type)
        if expected_count != actual_count:
            errors.append(
                ImportValidationIssue(
                    code="entity_count_mismatch",
                    field=f"manifest.entity_counts.{entity_type}",
                    message=(
                        f"Entity count mismatch for {entity_type}: manifest says "
                        f"{expected_count}, bundle has {actual_count}."
                    ),
                )
            )


def _validate_artifact_index(
    artifact_index: list[Any],
    manifest: Any,
    bundle_ids: dict[str, set[str]],
    local_ids: dict[str, set[str]],
    errors: list[ImportValidationIssue],
) -> None:
    """Validate metadata-only artifact index entries from an import bundle."""

    if isinstance(manifest, dict):
        artifact_count = manifest.get("artifact_count")
        if not isinstance(artifact_count, int) or isinstance(artifact_count, bool):
            errors.append(
                ImportValidationIssue(
                    code="invalid_artifact_count",
                    field="manifest.artifact_count",
                    message="manifest.artifact_count must be a non-negative integer.",
                )
            )
        elif artifact_count < 0:
            errors.append(
                ImportValidationIssue(
                    code="invalid_artifact_count",
                    field="manifest.artifact_count",
                    message="manifest.artifact_count must be a non-negative integer.",
                )
            )
        elif artifact_count != len(artifact_index):
            errors.append(
                ImportValidationIssue(
                    code="artifact_count_mismatch",
                    field="manifest.artifact_count",
                    message=(
                        "manifest.artifact_count must match the number of artifact_index entries."
                    ),
                )
            )

    normalized_paths: set[str] = set()
    for index, item in enumerate(artifact_index):
        field_prefix = f"artifact_index[{index}]"
        if not isinstance(item, dict):
            errors.append(
                ImportValidationIssue(
                    code="invalid_artifact_entry",
                    field=field_prefix,
                    message=f"{field_prefix} must be an object.",
                )
            )
            continue

        normalized_path = _validate_artifact_path(
            item.get("path"),
            field_prefix,
            normalized_paths,
            errors,
        )
        _validate_artifact_hash(item, field_prefix, errors)
        _validate_artifact_size(item.get("size_bytes"), field_prefix, errors)
        _require_artifact_run_reference(
            index,
            item.get("run_id"),
            bundle_ids["runs"],
            local_ids["runs"],
            errors,
        )

        if normalized_path is not None:
            normalized_paths.add(normalized_path.casefold())


def _validate_artifact_path(
    value: Any,
    field_prefix: str,
    normalized_paths: set[str],
    errors: list[ImportValidationIssue],
) -> str | None:
    if not _is_nonblank_string(value):
        errors.append(
            ImportValidationIssue(
                code="invalid_artifact_path",
                field=f"{field_prefix}.path",
                message="Artifact path must be a non-empty POSIX relative path.",
            )
        )
        return None

    path_text = str(value)
    if "\\" in path_text:
        errors.append(
            ImportValidationIssue(
                code="invalid_artifact_path",
                field=f"{field_prefix}.path",
                message="Artifact path must use POSIX `/` separators, not backslashes.",
            )
        )
        return None
    if _has_control_character(path_text):
        errors.append(
            ImportValidationIssue(
                code="invalid_artifact_path",
                field=f"{field_prefix}.path",
                message="Artifact path cannot contain null bytes or control characters.",
            )
        )
        return None
    if path_text.startswith("/") or PureWindowsPath(path_text).is_absolute():
        errors.append(
            ImportValidationIssue(
                code="invalid_artifact_path",
                field=f"{field_prefix}.path",
                message="Artifact path must be project-relative, not absolute.",
            )
        )
        return None

    parts = tuple(part for part in path_text.split("/") if part not in {"", "."})
    if not parts:
        errors.append(
            ImportValidationIssue(
                code="invalid_artifact_path",
                field=f"{field_prefix}.path",
                message="Artifact path must name a file.",
            )
        )
        return None
    if ".." in parts:
        errors.append(
            ImportValidationIssue(
                code="invalid_artifact_path",
                field=f"{field_prefix}.path",
                message="Artifact path cannot contain path traversal segments.",
            )
        )
        return None
    if any(part.lower() == ".pmem" for part in parts):
        errors.append(
            ImportValidationIssue(
                code="invalid_artifact_path",
                field=f"{field_prefix}.path",
                message="Artifact path cannot point inside .pmem.",
            )
        )
        return None

    normalized = posixpath.normpath("/".join(parts))
    normalized_key = normalized.casefold()
    if normalized_key in normalized_paths:
        errors.append(
            ImportValidationIssue(
                code="duplicate_artifact_path",
                field=f"{field_prefix}.path",
                message="Artifact path duplicates another artifact after normalization.",
            )
        )
        return None
    return normalized


def _validate_artifact_hash(
    item: dict[str, Any],
    field_prefix: str,
    errors: list[ImportValidationIssue],
) -> None:
    algorithm = item.get("hash_algorithm", "sha256")
    if algorithm != "sha256":
        errors.append(
            ImportValidationIssue(
                code="unsupported_artifact_hash_algorithm",
                field=f"{field_prefix}.hash_algorithm",
                message="Artifact hash_algorithm must be sha256.",
            )
        )

    sha256 = item.get("sha256")
    if not isinstance(sha256, str):
        errors.append(
            ImportValidationIssue(
                code="missing_artifact_hash",
                field=f"{field_prefix}.sha256",
                message="Artifact sha256 is required.",
            )
        )
    elif ARTIFACT_SHA256_PATTERN.fullmatch(sha256) is None:
        errors.append(
            ImportValidationIssue(
                code="invalid_artifact_hash",
                field=f"{field_prefix}.sha256",
                message="Artifact sha256 must be 64 lowercase hexadecimal characters.",
            )
        )


def _validate_artifact_size(
    value: Any,
    field_prefix: str,
    errors: list[ImportValidationIssue],
) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(
            ImportValidationIssue(
                code="invalid_artifact_size",
                field=f"{field_prefix}.size_bytes",
                message="Artifact size_bytes must be a non-negative integer.",
            )
        )


def _require_artifact_run_reference(
    index: int,
    value: Any,
    bundle_runs: set[str],
    local_runs: set[str],
    errors: list[ImportValidationIssue],
) -> None:
    if not _is_nonblank_string(value):
        errors.append(
            ImportValidationIssue(
                code="missing_reference",
                field=f"artifact_index[{index}].run_id",
                message="artifact_index run_id must reference an existing run.",
            )
        )
        return

    value_text = str(value)
    if value_text not in bundle_runs and value_text not in local_runs:
        errors.append(
            ImportValidationIssue(
                code="missing_dependency",
                field=f"artifact_index[{index}].run_id",
                message=(
                    f"artifact_index[{index}].run_id references `{value_text}`, "
                    "which is not present in the bundle or local database."
                ),
            )
        )


def _validate_foreign_keys(
    entities: dict[str, Any],
    bundle_ids: dict[str, set[str]],
    local_ids: dict[str, set[str]],
    errors: list[ImportValidationIssue],
) -> None:
    for index, experiment in _iter_entity_objects(entities, "experiments"):
        _require_reference(
            "experiments",
            index,
            "project_id",
            experiment.get("project_id"),
            bundle_ids["projects"],
            local_ids["projects"],
            errors,
        )

    for index, run in _iter_entity_objects(entities, "runs"):
        _require_reference(
            "runs",
            index,
            "experiment_id",
            run.get("experiment_id"),
            bundle_ids["experiments"],
            local_ids["experiments"],
            errors,
        )

    for index, failure in _iter_entity_objects(entities, "failures"):
        _require_reference(
            "failures",
            index,
            "run_id",
            failure.get("run_id"),
            bundle_ids["runs"],
            local_ids["runs"],
            errors,
        )

    for index, decision in _iter_entity_objects(entities, "decisions"):
        _require_reference(
            "decisions",
            index,
            "project_id",
            decision.get("project_id"),
            bundle_ids["projects"],
            local_ids["projects"],
            errors,
        )
        _require_optional_reference(
            "decisions",
            index,
            "experiment_id",
            decision.get("experiment_id"),
            bundle_ids["experiments"],
            local_ids["experiments"],
            errors,
        )
        related = decision.get("related_experiments", [])
        if not isinstance(related, list):
            errors.append(
                ImportValidationIssue(
                    code="invalid_related_experiments",
                    field=f"entities.decisions[{index}].related_experiments",
                    message="related_experiments must be an array of experiment ids.",
                )
            )
        else:
            for related_index, experiment_id in enumerate(related):
                _require_reference(
                    "decisions",
                    index,
                    f"related_experiments[{related_index}]",
                    experiment_id,
                    bundle_ids["experiments"],
                    local_ids["experiments"],
                    errors,
                )

    for index, note in _iter_entity_objects(entities, "notes"):
        _require_reference(
            "notes",
            index,
            "project_id",
            note.get("project_id"),
            bundle_ids["projects"],
            local_ids["projects"],
            errors,
        )
        _require_optional_reference(
            "notes",
            index,
            "experiment_id",
            note.get("experiment_id"),
            bundle_ids["experiments"],
            local_ids["experiments"],
            errors,
        )
        _require_optional_reference(
            "notes",
            index,
            "run_id",
            note.get("run_id"),
            bundle_ids["runs"],
            local_ids["runs"],
            errors,
        )

    for index, tracked_path in _iter_entity_objects(entities, "tracked_paths"):
        _require_reference(
            "tracked_paths",
            index,
            "project_id",
            tracked_path.get("project_id"),
            bundle_ids["projects"],
            local_ids["projects"],
            errors,
        )


def _validate_provenance(
    provenance: dict[str, Any],
    errors: list[ImportValidationIssue],
) -> None:
    """Validate v1 provenance without accepting private paths or remote metadata."""

    actual_keys = set(provenance)
    for key in sorted(actual_keys - PROVENANCE_ALLOWED_KEYS):
        errors.append(
            ImportValidationIssue(
                code="unknown_provenance_key",
                field=f"provenance.{key}",
                message="Unknown provenance keys are not allowed in export bundle v1.",
            )
        )

    if provenance.get("tool") != "projmem":
        errors.append(
            ImportValidationIssue(
                code="invalid_provenance_tool",
                field="provenance.tool",
                message="provenance.tool must be `projmem`.",
            )
        )
    if not _is_nonblank_string(provenance.get("tool_version")):
        errors.append(
            ImportValidationIssue(
                code="invalid_provenance_tool_version",
                field="provenance.tool_version",
                message="provenance.tool_version must be a non-empty string.",
            )
        )
    if not _is_nonblank_string(provenance.get("source")):
        errors.append(
            ImportValidationIssue(
                code="invalid_provenance_source",
                field="provenance.source",
                message="provenance.source must be a non-empty string.",
            )
        )

    project_name = provenance.get("project_name")
    if project_name is not None and not isinstance(project_name, str):
        errors.append(
            ImportValidationIssue(
                code="invalid_provenance_project_name",
                field="provenance.project_name",
                message="provenance.project_name must be a string or null.",
            )
        )

    git_commit_hash = provenance.get("git_commit_hash")
    if git_commit_hash is not None and not isinstance(git_commit_hash, str):
        errors.append(
            ImportValidationIssue(
                code="invalid_provenance_git_commit_hash",
                field="provenance.git_commit_hash",
                message="provenance.git_commit_hash must be a string or null.",
            )
        )

    git_dirty = provenance.get("git_dirty")
    if git_dirty is not None and not isinstance(git_dirty, bool):
        errors.append(
            ImportValidationIssue(
                code="invalid_provenance_git_dirty",
                field="provenance.git_dirty",
                message="provenance.git_dirty must be a boolean or null.",
            )
        )


def _require_reference(
    entity_type: str,
    index: int,
    field: str,
    value: Any,
    bundle_targets: set[str],
    local_targets: set[str],
    errors: list[ImportValidationIssue],
) -> None:
    if not _is_nonblank_string(value):
        errors.append(
            ImportValidationIssue(
                code="missing_reference",
                field=f"entities.{entity_type}[{index}].{field}",
                message=f"{entity_type}[{index}].{field} must reference an existing record.",
            )
        )
        return
    value_text = str(value)
    if value_text not in bundle_targets and value_text not in local_targets:
        errors.append(
            ImportValidationIssue(
                code="missing_dependency",
                field=f"entities.{entity_type}[{index}].{field}",
                message=(
                    f"{entity_type}[{index}].{field} references `{value_text}`, "
                    "which is not present in the bundle or local database."
                ),
            )
        )


def _require_optional_reference(
    entity_type: str,
    index: int,
    field: str,
    value: Any,
    bundle_targets: set[str],
    local_targets: set[str],
    errors: list[ImportValidationIssue],
) -> None:
    if value is None:
        return
    _require_reference(entity_type, index, field, value, bundle_targets, local_targets, errors)


def _collect_free_text_review(entities: dict[str, Any]) -> list[PrivacyReviewItem]:
    items: list[PrivacyReviewItem] = []
    for entity_type, fields in FREE_TEXT_FIELDS.items():
        records = entities.get(entity_type, [])
        if not isinstance(records, list):
            continue
        for field in fields:
            count = sum(
                1
                for record in records
                if isinstance(record, dict) and _is_nonblank_string(record.get(field))
            )
            if count:
                items.append(
                    PrivacyReviewItem(
                        field=f"{entity_type}.{field}",
                        count=count,
                        message=(
                            "Free-text field may contain secrets, private data, "
                            "or research-sensitive context."
                        ),
                    )
                )
    return items


def _collect_artifact_review(artifact_index: list[Any]) -> list[PrivacyReviewItem]:
    if not artifact_index:
        return []
    return [
        PrivacyReviewItem(
            field="artifact_index",
            count=len(artifact_index),
            message=(
                "Artifact metadata can expose local file names, dataset names, "
                "or project structure."
            ),
        )
    ]


def _collect_conflict_preview(
    entities: dict[str, Any],
    local_ids: dict[str, set[str]],
) -> list[ConflictPreviewItem]:
    conflicts: list[ConflictPreviewItem] = []
    for entity_type in ENTITY_KEYS:
        id_field = ENTITY_ID_FIELDS[entity_type]
        records = entities.get(entity_type, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            entity_id = record.get(id_field)
            if _is_nonblank_string(entity_id) and str(entity_id) in local_ids[entity_type]:
                conflicts.append(
                    ConflictPreviewItem(
                        conflict_type="same_id_present",
                        entity_type=entity_type,
                        entity_id=str(entity_id),
                        message=(
                            "Local database already has this id. Import dry-run only previews "
                            "conflicts; apply/resolution belongs to later portability work."
                        ),
                    )
                )
    return conflicts


def _iter_entity_objects(
    entities: dict[str, Any], entity_type: str
) -> tuple[tuple[int, dict[str, Any]], ...]:
    records = entities.get(entity_type, [])
    if not isinstance(records, list):
        return ()
    return tuple((index, item) for index, item in enumerate(records) if isinstance(item, dict))


def _load_local_ids(db_path: Path) -> dict[str, set[str]]:
    queries = {
        "projects": "SELECT id FROM projects",
        "experiments": "SELECT id FROM experiments",
        "runs": "SELECT run_id FROM runs",
        "failures": "SELECT id FROM failures",
        "decisions": "SELECT id FROM decisions",
        "notes": "SELECT id FROM notes",
        "tracked_paths": "SELECT id FROM tracked_paths",
    }
    connection = connect_database(db_path)
    try:
        return {
            entity_type: {str(row[0]) for row in connection.execute(sql).fetchall()}
            for entity_type, sql in queries.items()
        }
    finally:
        connection.close()


def _manifest_hash(manifest: dict[str, Any]) -> str:
    candidate = copy.deepcopy(manifest)
    candidate["manifest_hash"] = None
    return _sha256_tag(candidate)


def _payload_hash(payload: dict[str, Any]) -> str:
    candidate = copy.deepcopy(payload)
    manifest = candidate.get("manifest")
    if isinstance(manifest, dict):
        manifest["manifest_hash"] = None
        manifest["payload_hash"] = None
    return _sha256_tag(candidate)


def _sha256_tag(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _is_hash_string(value: Any) -> bool:
    return isinstance(value, str) and HASH_PATTERN.fullmatch(value) is not None


def _is_nonblank_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_control_character(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _report(
    bundle_path: str,
    export_format_version: str | None,
    schema_version: str | None,
    entity_counts: dict[str, int],
    errors: list[ImportValidationIssue],
    warnings: list[ImportValidationIssue],
    privacy_review: list[PrivacyReviewItem],
    conflicts: list[ConflictPreviewItem],
) -> ImportDryRunReport:
    return ImportDryRunReport(
        ok=not errors,
        dry_run=True,
        bundle_path=bundle_path,
        export_format_version=export_format_version,
        schema_version=schema_version,
        entity_counts=entity_counts,
        errors=tuple(errors),
        warnings=tuple(warnings),
        privacy_review=tuple(privacy_review),
        conflicts=tuple(conflicts),
        database_mutation=False,
    )


def _issue_payload(issue: ImportValidationIssue) -> dict[str, str | None]:
    return {
        "code": issue.code,
        "field": issue.field,
        "message": issue.message,
    }
