"""portability and failure-analysis export-bundle service."""

from __future__ import annotations

import base64
import copy
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from pmem import __version__
from pmem.domain.import_bundle import ENTITY_KEYS, EXPORT_FORMAT_VERSION, FREE_TEXT_FIELDS
from pmem.errors import PmemNotFoundError, PmemSecurityError, PmemValidationError
from pmem.repositories.portability import ExportPackageRecord, ExportPackageRepository
from pmem.repositories.sqlite import PMEM_DIRNAME, connect_database, project_database_path
from pmem.services.database import ensure_database
from pmem.services.project_export import EXPORT_SCHEMA_VERSION, export_project
from pmem.utils.hashing import compute_text_hash

CANONICAL_JSON_RULE = {
    "encoding": "utf-8",
    "sort_keys": True,
    "separators": [",", ":"],
    "line_ending": "lf",
}
SUPPORTED_SCOPE = "project"
SUPPORTED_REDACT_FIELDS = frozenset(
    f"{entity_type}.{field}" for entity_type, fields in FREE_TEXT_FIELDS.items() for field in fields
)


@dataclass(frozen=True)
class ExportBundleResult:
    """User-facing result of writing one export bundle."""

    output_path: Path
    display_path: str
    manifest_hash: str
    payload_hash: str
    artifact_count: int
    export_package: ExportPackageRecord
    bundle: dict[str, Any]


def export_bundle(
    project_root: str | Path,
    output_path: str | Path,
    *,
    scope: str = SUPPORTED_SCOPE,
    include_artifacts: bool = False,
    redact_fields: tuple[str, ...] = (),
    freeze_timestamp: str | None = None,
) -> ExportBundleResult:
    """Build and write a deterministic portability and failure-analysis bundle."""

    root = Path(project_root)
    ensure_database(root)
    output = _resolve_output_path(root, output_path)
    generated_at = _generated_at(freeze_timestamp)
    bundle = build_export_bundle(
        root,
        scope=scope,
        include_artifacts=include_artifacts,
        redact_fields=redact_fields,
        generated_at=generated_at,
        freeze_timestamp=freeze_timestamp is not None,
    )

    _write_bundle_file(output, bundle)
    connection = connect_database(project_database_path(root))
    try:
        export_package = ExportPackageRepository(connection).create_or_get(
            package_id=f"export_{uuid.uuid4().hex}",
            version=EXPORT_FORMAT_VERSION,
            scope={"scope": scope},
            manifest_hash=str(bundle["manifest"]["manifest_hash"]),
            payload_hash=str(bundle["manifest"]["payload_hash"]),
            artifact_count=len(bundle["artifact_index"]),
            path=_display_path(root, output),
            created_at=generated_at,
        )
    finally:
        connection.close()

    return ExportBundleResult(
        output_path=output,
        display_path=_display_path(root, output),
        manifest_hash=str(bundle["manifest"]["manifest_hash"]),
        payload_hash=str(bundle["manifest"]["payload_hash"]),
        artifact_count=len(bundle["artifact_index"]),
        export_package=export_package,
        bundle=bundle,
    )


def build_export_bundle(
    project_root: str | Path,
    *,
    scope: str = SUPPORTED_SCOPE,
    include_artifacts: bool = False,
    redact_fields: tuple[str, ...] = (),
    generated_at: str,
    freeze_timestamp: bool,
) -> dict[str, Any]:
    """Return the export bundle payload without writing files."""

    if scope != SUPPORTED_SCOPE:
        raise PmemValidationError("Only project scope is supported for export-bundle.")
    redaction_fields = _normalize_redact_fields(redact_fields)
    project_root_path = Path(project_root)
    export_payload = export_project(project_root_path)
    entities = _prepare_entities(copy.deepcopy(export_payload["entities"]))
    redacted_counts = _apply_redactions(entities, redaction_fields)
    artifact_index = _artifact_index(
        project_root_path,
        entities["runs"],
        include_artifacts=include_artifacts,
    )
    privacy_flags = _privacy_flags(entities, artifact_index, redacted_counts)
    provenance = _provenance(entities)
    manifest = {
        "export_format_version": EXPORT_FORMAT_VERSION,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "freeze_timestamp": freeze_timestamp,
        "project_id": export_payload["project_id"],
        "entity_counts": {entity_type: len(entities[entity_type]) for entity_type in ENTITY_KEYS},
        "artifact_count": len(artifact_index),
        "canonical_json": CANONICAL_JSON_RULE,
        "manifest_hash": None,
        "payload_hash": None,
    }
    bundle = {
        "manifest": manifest,
        "entities": entities,
        "artifact_index": artifact_index,
        "privacy_flags": privacy_flags,
        "provenance": provenance,
    }
    _refresh_hashes(bundle)
    return bundle


def export_bundle_result_json(result: ExportBundleResult) -> dict[str, Any]:
    """Return stable machine-readable output for `pmem export-bundle --json`."""

    return {
        "ok": True,
        "bundle_path": result.display_path,
        "export_format_version": EXPORT_FORMAT_VERSION,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "manifest_hash": result.manifest_hash,
        "payload_hash": result.payload_hash,
        "artifact_count": result.artifact_count,
        "export_package_id": result.export_package.id,
    }


def _prepare_entities(entities: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    prepared: dict[str, list[dict[str, Any]]] = {}
    for entity_type in ENTITY_KEYS:
        records = entities.get(entity_type, [])
        prepared[entity_type] = [dict(record) for record in records if isinstance(record, dict)]

    for run in prepared["runs"]:
        git = run.get("git")
        if isinstance(git, dict):
            run["git"] = _safe_git_metadata(git)
            commit = git.get("commit")
            run["git_commit_hash"] = commit if isinstance(commit, str) else None
        else:
            run["git"] = {}
            run["git_commit_hash"] = None

    return {
        "projects": sorted(prepared["projects"], key=lambda item: _sort_key(item, "id")),
        "experiments": sorted(
            prepared["experiments"],
            key=lambda item: (
                _sort_key(item, "project_id"),
                _sort_key(item, "created_at"),
                _sort_key(item, "id"),
            ),
        ),
        "runs": sorted(
            prepared["runs"],
            key=lambda item: (
                _sort_key(item, "experiment_id"),
                _sort_key(item, "timestamp"),
                _sort_key(item, "run_id"),
            ),
        ),
        "failures": sorted(
            prepared["failures"],
            key=lambda item: (
                _sort_key(item, "run_id"),
                _sort_key(item, "created_at"),
                _sort_key(item, "id"),
            ),
        ),
        "decisions": sorted(
            prepared["decisions"],
            key=lambda item: (
                _sort_key(item, "project_id"),
                _sort_key(item, "created_at"),
                _sort_key(item, "id"),
            ),
        ),
        "notes": sorted(
            prepared["notes"],
            key=lambda item: (
                _sort_key(item, "project_id"),
                _sort_key(item, "created_at"),
                _sort_key(item, "id"),
            ),
        ),
        "tracked_paths": sorted(
            prepared["tracked_paths"],
            key=lambda item: (
                _sort_key(item, "project_id"),
                _sort_key(item, "path"),
                _sort_key(item, "id"),
            ),
        ),
    }


def _artifact_index(
    project_root: Path,
    runs: list[dict[str, Any]],
    *,
    include_artifacts: bool,
) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for run in runs:
        artifacts = run.get("artifacts", [])
        if not isinstance(artifacts, list):
            continue
        run_id = run.get("run_id")
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            path = artifact.get("path")
            sha256 = artifact.get("sha256")
            size_bytes = artifact.get("size_bytes")
            if not isinstance(path, str) or not isinstance(sha256, str):
                continue
            entry: dict[str, Any] = {
                "path": path,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "run_id": run_id,
                "hash_algorithm": "sha256",
            }
            if include_artifacts:
                artifact_path = _resolve_artifact_path(project_root, path)
                entry["content_encoding"] = "base64"
                entry["content_base64"] = base64.b64encode(artifact_path.read_bytes()).decode(
                    "ascii"
                )
            index.append(entry)
    return sorted(index, key=lambda item: str(item["path"]).casefold())


def _apply_redactions(
    entities: dict[str, list[dict[str, Any]]],
    redaction_fields: tuple[str, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for field_path in redaction_fields:
        entity_type, field = field_path.split(".", 1)
        count = 0
        for record in entities[entity_type]:
            if record.get(field) is not None:
                record[field] = "[REDACTED]"
                count += 1
        counts[field_path] = count
    return counts


def _privacy_flags(
    entities: dict[str, list[dict[str, Any]]],
    artifact_index: list[dict[str, Any]],
    redacted_counts: dict[str, int],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for entity_type, fields in FREE_TEXT_FIELDS.items():
        records = entities.get(entity_type, [])
        for field in fields:
            count = sum(
                1
                for record in records
                if isinstance(record.get(field), str) and bool(str(record[field]).strip())
            )
            if count:
                flags.append(
                    {
                        "code": "free_text_present",
                        "severity": "warning",
                        "field": f"{entity_type}.{field}",
                        "count": count,
                        "message": "Free-text memory may contain sensitive information.",
                    }
                )
    for field_path, count in redacted_counts.items():
        flags.append(
            {
                "code": "redacted_field",
                "severity": "info",
                "field": field_path,
                "count": count,
                "message": "Field was redacted before bundle hashing and writing.",
            }
        )
    if artifact_index:
        flags.append(
            {
                "code": "artifact_metadata_present",
                "severity": "warning",
                "field": "artifact_index",
                "count": len(artifact_index),
                "message": "Artifact metadata can expose filenames, dataset names, or structure.",
            }
        )
    return sorted(flags, key=lambda item: (str(item["field"]), str(item["code"])))


def _provenance(entities: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    project = entities["projects"][0] if entities["projects"] else {}
    return {
        "tool": "projmem",
        "tool_version": __version__,
        "source": "local-export",
        "project_name": project.get("name"),
        "git_commit_hash": None,
        "git_dirty": None,
    }


def _safe_git_metadata(git: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in git.items()
        if "remote" not in key.casefold() and "url" not in key.casefold()
    }


def _normalize_redact_fields(raw_fields: tuple[str, ...]) -> tuple[str, ...]:
    fields: list[str] = []
    for raw_field in raw_fields:
        for candidate in raw_field.split(","):
            cleaned = candidate.strip()
            if cleaned:
                fields.append(cleaned)
    unknown = sorted(field for field in fields if field not in SUPPORTED_REDACT_FIELDS)
    if unknown:
        raise PmemValidationError(f"Unsupported redact field: {unknown[0]}.")
    return tuple(dict.fromkeys(fields))


def _resolve_output_path(project_root: Path, user_path: str | Path) -> Path:
    raw_text = str(user_path).strip()
    if not raw_text:
        raise PmemValidationError("Bundle output path cannot be blank.")
    raw_path = Path(raw_text)
    if any(part.casefold() == PMEM_DIRNAME.casefold() for part in raw_path.parts):
        raise PmemSecurityError("Bundle output path cannot point inside .pmem.")
    output = raw_path if raw_path.is_absolute() else project_root / raw_path
    output_parent = output.parent.resolve(strict=False)
    if output.exists() and output.is_dir():
        raise PmemSecurityError("Bundle output path must point to a file, not a directory.")
    if output.exists() and output.is_symlink():
        raise PmemSecurityError("Bundle output path cannot be a symlink.")
    output_parent.mkdir(parents=True, exist_ok=True)
    return output_parent / output.name


def _resolve_artifact_path(project_root: Path, artifact_path: str) -> Path:
    if "\\" in artifact_path or artifact_path.startswith("/"):
        raise PmemSecurityError("Artifact path in export bundle must be project-relative.")
    if PureWindowsPath(artifact_path).is_absolute():
        raise PmemSecurityError("Artifact path in export bundle must be project-relative.")
    if "\x00" in artifact_path or any(ord(c) < 32 for c in artifact_path):
        raise PmemSecurityError(
            "Artifact path in export bundle contains unsafe control characters."
        )
    raw_path = Path(artifact_path)
    if any(part.casefold() == PMEM_DIRNAME.casefold() or part == ".." for part in raw_path.parts):
        raise PmemSecurityError("Artifact path in export bundle is unsafe.")
    _reject_artifact_symlink_parts(project_root, raw_path)
    resolved = (project_root / raw_path).resolve(strict=False)
    root = project_root.resolve()
    if root != resolved and root not in resolved.parents:
        raise PmemSecurityError("Artifact path in export bundle must stay inside project.")
    if not resolved.exists():
        raise PmemNotFoundError("Artifact file referenced by run metadata was not found.")
    if resolved.is_symlink() or not resolved.is_file():
        raise PmemSecurityError("Artifact file must be a regular file.")
    return resolved


def _reject_artifact_symlink_parts(project_root: Path, raw_path: Path) -> None:
    current = project_root
    for part in raw_path.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.exists() and current.is_symlink():
            raise PmemSecurityError("Artifact path in export bundle cannot contain symlinks.")


def _write_bundle_file(path: Path, bundle: dict[str, Any]) -> None:
    try:
        path.write_text(_canonical_json(bundle) + "\n", encoding="utf-8")
        path.chmod(0o600)
    except OSError as exc:
        raise PmemValidationError("Export bundle could not be written.") from exc


def _refresh_hashes(bundle: dict[str, Any]) -> None:
    bundle["manifest"]["manifest_hash"] = None
    bundle["manifest"]["payload_hash"] = None
    bundle["manifest"]["payload_hash"] = _hash_tag(_canonical_json(bundle))
    manifest = copy.deepcopy(bundle["manifest"])
    manifest["manifest_hash"] = None
    bundle["manifest"]["manifest_hash"] = _hash_tag(_canonical_json(manifest))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_tag(canonical_json: str) -> str:
    return f"sha256:{compute_text_hash(canonical_json)}"


def _generated_at(freeze_timestamp: str | None) -> str:
    if freeze_timestamp is None:
        return _utc_now_iso()
    raw = freeze_timestamp.strip()
    if not raw:
        raise PmemValidationError("Freeze timestamp cannot be blank.")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PmemValidationError("Freeze timestamp must be ISO-8601 UTC.") from exc
    if parsed.tzinfo is None:
        raise PmemValidationError("Freeze timestamp must include UTC timezone.")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _sort_key(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    return str(value) if value is not None else ""
