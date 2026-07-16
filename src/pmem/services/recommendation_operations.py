"""recommendation CLI privacy-safe recommendation CLI operations."""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from pmem.errors import (
    PmemNotFoundError,
    PmemPersistenceError,
    PmemSecurityError,
    PmemValidationError,
)
from pmem.graph.ingestion import build_graph_from_database_readonly
from pmem.graph.privacy import GRAPH_JSON_FILE_MODE
from pmem.recommendations import (
    Recommendation,
    generate_recommendations,
    link_recommendation_evidence_from_document,
)
from pmem.repositories.sqlite import (
    PMEM_DIRNAME,
    connect_database_readonly,
    execute,
    project_database_path,
)
from pmem.services.project_context import require_project_context_readonly

RECOMMENDATION_LIST_RESULT_VERSION = "recommendation-list-result-v1"
RECOMMENDATION_DETAIL_RESULT_VERSION = "recommendation-detail-result-v1"
RECOMMENDATION_EXPORT_RESULT_VERSION = "recommendation-export-result-v1"


@dataclass(frozen=True, slots=True)
class RecommendationExportResult:
    """Result of exporting privacy-safe recommendation candidates."""

    output_path: Path
    display_path: str
    payload: dict[str, Any]


def recommendation_list_payload(
    project_root: str | Path,
    *,
    max_recommendations: int = 5,
) -> dict[str, Any]:
    """Return recommendation CLI recommendation candidates with evidence-link warnings."""

    limit = _clean_limit(max_recommendations)
    context = require_project_context_readonly(project_root)
    recommendations = generate_recommendations(context.root, max_recommendations=limit)
    basis_counts = _basis_counts(context.root, context.project.id)
    payload_items = _recommendation_payloads(context.root, recommendations)
    warnings = _payload_warnings(
        payload_items,
        basis_counts,
        _data_quality_warnings(
            context.root,
            context.project.id,
            primary_metric=context.project.primary_metric,
            metric_direction=context.project.metric_direction or "max",
        ),
    )
    return {
        "schema_version": RECOMMENDATION_LIST_RESULT_VERSION,
        "recommendation_count": len(payload_items),
        "recommendations": payload_items,
        "basis_counts": basis_counts,
        "scope_message": _scope_message(basis_counts),
        "warnings": warnings,
        "privacy_mode": "metadata_only",
        "database_mutation": False,
        "raw_text_in_output": False,
        "derived_graph_edges": False,
    }


def recommendation_detail_payload(
    project_root: str | Path,
    *,
    recommendation_id: str,
    max_recommendations: int = 5,
) -> dict[str, Any]:
    """Return one generated recommendation by stable recommendation id."""

    cleaned_id = recommendation_id.strip()
    if not cleaned_id:
        raise PmemValidationError("Recommendation id cannot be blank.")
    payload = recommendation_list_payload(
        project_root,
        max_recommendations=max_recommendations,
    )
    for item in payload["recommendations"]:
        if isinstance(item, dict) and item.get("recommendation_id") == cleaned_id:
            return {
                "schema_version": RECOMMENDATION_DETAIL_RESULT_VERSION,
                "recommendation": item,
                "basis_counts": payload["basis_counts"],
                "scope_message": payload["scope_message"],
                "warnings": payload["warnings"],
                "privacy_mode": payload["privacy_mode"],
                "database_mutation": False,
                "raw_text_in_output": False,
                "derived_graph_edges": False,
            }
    raise PmemNotFoundError("Recommendation candidate was not found.")


def export_recommendations(
    project_root: str | Path,
    *,
    output_path: str | Path,
    max_recommendations: int = 5,
) -> RecommendationExportResult:
    """Write recommendation candidates to a private project-local JSON file."""

    context = require_project_context_readonly(project_root)
    output = _resolve_recommendation_export_path(context.root, output_path)
    payload = recommendation_list_payload(context.root, max_recommendations=max_recommendations)
    export_payload = {
        "schema_version": RECOMMENDATION_EXPORT_RESULT_VERSION,
        "recommendations": payload,
    }
    _write_private_json(output, export_payload)
    return RecommendationExportResult(
        output_path=output,
        display_path=_display_path(context.root, output),
        payload={
            "schema_version": RECOMMENDATION_EXPORT_RESULT_VERSION,
            "ok": True,
            "output_path": _display_path(context.root, output),
            "recommendation_count": payload["recommendation_count"],
            "basis_counts": payload["basis_counts"],
            "privacy_mode": payload["privacy_mode"],
            "file_mode": f"0o{(output.stat().st_mode & 0o777):03o}",
            "database_mutation": False,
            "raw_text_in_output": False,
            "derived_graph_edges": False,
        },
    )


def _recommendation_payloads(
    project_root: Path,
    recommendations: tuple[Recommendation, ...],
) -> list[dict[str, Any]]:
    if not recommendations:
        return []
    db_path = project_database_path(project_root)
    document = build_graph_from_database_readonly(db_path)
    payloads: list[dict[str, Any]] = []
    for recommendation in recommendations:
        links = link_recommendation_evidence_from_document(db_path, document, recommendation)
        payload = recommendation.model_dump(mode="json")
        payload["evidence_counts"] = links.to_dict()["counts"]
        payload["evidence_link_warnings"] = list(links.warnings)
        payloads.append(payload)
    return sorted(payloads, key=lambda item: str(item["recommendation_id"]))


def _basis_counts(project_root: Path, project_id: str) -> dict[str, int]:
    connection = connect_database_readonly(project_database_path(project_root))
    try:
        experiments = execute(
            connection,
            "SELECT COUNT(*) AS count FROM experiments WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        runs = execute(
            connection,
            """
            SELECT COUNT(*) AS count
            FROM runs
            JOIN experiments ON experiments.id = runs.experiment_id
            WHERE experiments.project_id = ?
            """,
            (project_id,),
        ).fetchone()
        failures = execute(
            connection,
            """
            SELECT COUNT(*) AS count
            FROM failures
            JOIN runs ON runs.run_id = failures.run_id
            JOIN experiments ON experiments.id = runs.experiment_id
            WHERE experiments.project_id = ?
            """,
            (project_id,),
        ).fetchone()
    finally:
        connection.close()
    return {
        "experiments": int(experiments["count"]) if experiments is not None else 0,
        "runs": int(runs["count"]) if runs is not None else 0,
        "failures": int(failures["count"]) if failures is not None else 0,
    }


def _payload_warnings(
    recommendations: list[dict[str, Any]],
    basis_counts: dict[str, int],
    quality_warnings: list[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    if not recommendations:
        warnings.append(
            "Insufficient project evidence for recommendation candidates. "
            "Set a primary metric, capture multiple runs, and log confirmed failures."
        )
    for item in recommendations:
        item_warnings = item.get("evidence_link_warnings")
        if isinstance(item_warnings, list):
            warnings.extend(str(warning) for warning in item_warnings)
    if basis_counts.get("experiments", 0) < 3:
        warnings.append("Recommendation scope is limited by fewer than 3 experiments.")
    warnings.extend(quality_warnings or [])
    return sorted(set(warnings))


def _scope_message(basis_counts: dict[str, int]) -> str:
    return (
        f"Based on {basis_counts.get('experiments', 0)} experiments, "
        f"{basis_counts.get('runs', 0)} runs, and "
        f"{basis_counts.get('failures', 0)} confirmed failures. "
        "Recommendation candidates require human review."
    )


def _data_quality_warnings(
    project_root: Path,
    project_id: str,
    *,
    primary_metric: str | None,
    metric_direction: str,
) -> list[str]:
    rows, failure_run_ids = _load_run_quality_rows(project_root, project_id)
    warnings: list[str] = []
    config_dataset_runs = 0
    artifact_dataset_runs = 0
    failed_metric_runs = 0
    failure_labeled_metric_runs: list[tuple[str, float]] = []
    successful_metric_values: list[float] = []

    for row in rows:
        config = _safe_json_object(str(row["config_json"]))
        artifacts = _safe_json_array(str(row["artifacts_json"]))
        metrics = _safe_json_object(str(row["metrics_json"]))
        run_id = str(row["run_id"])
        status = str(row["status"])
        metric_value = _metric_value(metrics, primary_metric)
        has_failure = run_id in failure_run_ids

        if _contains_key(config, "dataset_id"):
            config_dataset_runs += 1
        if any(isinstance(item, dict) and "dataset_id" in item for item in artifacts):
            artifact_dataset_runs += 1
        if metric_value is not None and status != "success":
            failed_metric_runs += 1
        if metric_value is not None and status == "success" and not has_failure:
            successful_metric_values.append(metric_value)
        if metric_value is not None and has_failure:
            failure_labeled_metric_runs.append((run_id, metric_value))

    if config_dataset_runs and artifact_dataset_runs == 0:
        warnings.append(
            "Dataset ids appear in run config metadata but not artifact metadata; "
            "dataset-failure screening may remain insufficient."
        )
    if failed_metric_runs:
        warnings.append(
            f"{failed_metric_runs} failed run(s) include primary metric values; "
            "review stale metric risk before acting."
        )

    best_metric = _best_metric(successful_metric_values, metric_direction)
    strong_failure_labels = sum(
        1
        for _run_id, value in failure_labeled_metric_runs
        if _is_strong_metric(value, best_metric=best_metric, metric_direction=metric_direction)
    )
    if strong_failure_labels:
        warnings.append(
            f"{strong_failure_labels} failure-labeled run(s) have strong primary metrics; "
            "review possible mislabels before trusting avoid candidates."
        )
    return warnings


def _load_run_quality_rows(project_root: Path, project_id: str) -> tuple[list[Any], set[str]]:
    connection = connect_database_readonly(project_database_path(project_root))
    try:
        rows = execute(
            connection,
            """
            SELECT runs.run_id, runs.status, runs.config_json, runs.metrics_json,
                   runs.artifacts_json
            FROM runs
            JOIN experiments ON experiments.id = runs.experiment_id
            WHERE experiments.project_id = ?
            ORDER BY runs.timestamp, runs.run_id
            """,
            (project_id,),
        ).fetchall()
        failure_rows = execute(
            connection,
            """
            SELECT failures.run_id
            FROM failures
            JOIN runs ON runs.run_id = failures.run_id
            JOIN experiments ON experiments.id = runs.experiment_id
            WHERE experiments.project_id = ?
            ORDER BY failures.created_at, failures.id
            """,
            (project_id,),
        ).fetchall()
    finally:
        connection.close()
    return list(rows), {str(row["run_id"]) for row in failure_rows}


def _safe_json_object(raw_json: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_json_array(raw_json: str) -> list[Any]:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def _metric_value(metrics: dict[str, Any], primary_metric: str | None) -> float | None:
    if not primary_metric:
        return None
    value = metrics.get(primary_metric)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value_float = float(value)
    return value_float if math.isfinite(value_float) else None


def _best_metric(values: list[float], metric_direction: str) -> float | None:
    if not values:
        return None
    return min(values) if metric_direction == "min" else max(values)


def _is_strong_metric(
    value: float,
    *,
    best_metric: float | None,
    metric_direction: str,
) -> bool:
    if best_metric is None:
        return False
    tolerance = max(abs(best_metric) * 0.10, 0.05)
    if metric_direction == "min":
        return value <= best_metric + tolerance
    return value >= best_metric - tolerance


def _clean_limit(value: int) -> int:
    if value < 1:
        raise PmemValidationError("Recommendation limit must be at least 1.")
    if value > 50:
        raise PmemValidationError("Recommendation limit must be 50 or less.")
    return value


def _resolve_recommendation_export_path(project_root: Path, user_path: str | Path) -> Path:
    raw_text = str(user_path).strip()
    if not raw_text:
        raise PmemValidationError("Recommendation export path cannot be blank.")
    if "\\" in raw_text or "\x00" in raw_text or any(ord(char) < 32 for char in raw_text):
        raise PmemSecurityError("Recommendation export path contains unsafe characters.")
    raw_path = Path(raw_text)
    if raw_path.is_absolute() or PureWindowsPath(raw_text).is_absolute():
        raise PmemSecurityError("Recommendation export path must be project-relative.")
    if any(part == ".." for part in raw_path.parts):
        raise PmemSecurityError("Recommendation export path cannot contain traversal segments.")
    if any(part.casefold() == PMEM_DIRNAME.casefold() for part in raw_path.parts):
        raise PmemSecurityError("Recommendation export path cannot point inside .pmem.")

    root = project_root.resolve()
    output = root / raw_path
    parent = output.parent.resolve(strict=False)
    if root != parent and root not in parent.parents:
        raise PmemSecurityError("Recommendation export path must stay inside the project.")
    _reject_symlink_parts(root, raw_path.parent)
    if output.exists() and output.is_dir():
        raise PmemSecurityError("Recommendation export path must point to a file.")
    if output.exists() and output.is_symlink():
        raise PmemSecurityError("Recommendation export path cannot be a symlink.")
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_parts(root, raw_path.parent)
    if parent.is_symlink():
        raise PmemSecurityError("Recommendation export path cannot contain symlinks.")
    return parent / output.name


def _write_private_json(output: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temp_path = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, GRAPH_JSON_FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output)
        os.chmod(output, GRAPH_JSON_FILE_MODE)
    except OSError as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise PmemPersistenceError("Recommendation export file could not be written.") from exc


def _reject_symlink_parts(project_root: Path, relative_parent: Path) -> None:
    current = project_root
    for part in relative_parent.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.exists() and current.is_symlink():
            raise PmemSecurityError("Recommendation export path cannot contain symlinks.")


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name
