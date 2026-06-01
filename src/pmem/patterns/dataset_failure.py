"""Dataset-failure and metric-distribution correlation mining.

Dataset-failure correlation is a conservative screening layer. It only uses
explicit dataset metadata from run artifact records, confirmed failure ids, and
finite numeric metrics. It does not infer dataset identity from artifact paths
and does not emit raw failure text, command text, or artifact paths.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.errors import PmemValidationError
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_context import require_project_context
from pmem.utils.hashing import compute_text_hash

DATASET_FAILURE_CORRELATION_SCHEMA_VERSION = "dataset-failure-correlation-v1"
DATASET_FAILURE_CORRELATION_METHOD = "fisher_dataset_failure_metric_anomaly_v1"
DEFAULT_MIN_TOTAL_RUNS = 10
DEFAULT_MIN_DATASET_RUNS = 2
DEFAULT_MAX_RESULTS = 25
DEFAULT_MAX_EVIDENCE_IDS = 20

_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_SENSITIVE_TOKENS = {
    "api",
    "auth",
    "credential",
    "credentials",
    "key",
    "password",
    "private",
    "secret",
    "token",
}
_SCALAR_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Explicit dataset metadata attached to one run artifact."""

    dataset_id: str
    version: str = "unknown"


@dataclass(frozen=True, slots=True)
class DatasetRunOutcome:
    """One run's explicit dataset identities, metrics, and failure outcome."""

    run_id: str
    datasets: tuple[DatasetIdentity, ...]
    metrics: dict[str, float]
    has_failure: bool
    failure_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _DatasetFeature:
    dataset_id: str
    dataset_id_redacted: bool
    version: str
    version_redacted: bool
    feature_id: str


def dataset_failure_correlation_payload(
    project_root: str | Path,
    *,
    min_total_runs: int = DEFAULT_MIN_TOTAL_RUNS,
    min_dataset_runs: int = DEFAULT_MIN_DATASET_RUNS,
    max_results: int = DEFAULT_MAX_RESULTS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic dataset-failure correlation dataset/failure correlation report."""

    _validate_parameters(
        min_total_runs=min_total_runs,
        min_dataset_runs=min_dataset_runs,
        max_results=max_results,
    )
    context = require_project_context(project_root)
    outcomes, skipped_counts = _load_project_outcomes(context.root, context.project.id)
    return dataset_failure_correlation_from_outcomes(
        outcomes,
        min_total_runs=min_total_runs,
        min_dataset_runs=min_dataset_runs,
        max_results=max_results,
        generated_at=generated_at,
        skipped_counts=skipped_counts,
    )


def dataset_failure_correlation_from_outcomes(
    outcomes: tuple[DatasetRunOutcome, ...],
    *,
    min_total_runs: int = DEFAULT_MIN_TOTAL_RUNS,
    min_dataset_runs: int = DEFAULT_MIN_DATASET_RUNS,
    max_results: int = DEFAULT_MAX_RESULTS,
    generated_at: str | None = None,
    skipped_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a dataset-failure correlation report from preloaded dataset/run outcomes."""

    _validate_parameters(
        min_total_runs=min_total_runs,
        min_dataset_runs=min_dataset_runs,
        max_results=max_results,
    )
    ordered_outcomes = tuple(sorted(outcomes, key=lambda item: item.run_id))
    total_runs = len(ordered_outcomes)
    dataset_runs = sum(1 for item in ordered_outcomes if item.datasets)
    failure_runs = sum(1 for item in ordered_outcomes if item.has_failure)
    non_failure_runs = total_runs - failure_runs
    metric_run_count = sum(1 for item in ordered_outcomes if item.metrics)
    skipped = {
        "invalid_dataset_metadata": 0,
        "non_numeric_metrics": 0,
        "sensitive_or_unsafe_dataset_labels": 0,
        **(skipped_counts or {}),
    }
    warnings = _dataset_warnings(
        total_runs=total_runs,
        dataset_runs=dataset_runs,
        failure_runs=failure_runs,
        non_failure_runs=non_failure_runs,
        metric_run_count=metric_run_count,
        min_total_runs=min_total_runs,
    )

    candidates: list[dict[str, Any]] = []
    if not warnings:
        feature_index: dict[str, tuple[_DatasetFeature, set[str]]] = {}
        for outcome in ordered_outcomes:
            for dataset in outcome.datasets:
                feature = _dataset_feature(dataset)
                if feature.dataset_id_redacted or feature.version_redacted:
                    skipped["sensitive_or_unsafe_dataset_labels"] += 1
                existing = feature_index.get(feature.feature_id)
                if existing is None:
                    feature_index[feature.feature_id] = (feature, {outcome.run_id})
                else:
                    existing[1].add(outcome.run_id)

        for feature, exposed_run_ids in feature_index.values():
            candidate = _candidate_for_dataset(
                feature,
                exposed_run_ids=exposed_run_ids,
                outcomes=ordered_outcomes,
                min_dataset_runs=min_dataset_runs,
            )
            if candidate is not None:
                candidates.append(candidate)

    candidate_count = len(candidates)
    for candidate in candidates:
        p_value = float(candidate["failure_statistics"]["p_value"])
        adjusted = round(min(1.0, p_value * candidate_count), 12)
        candidate["failure_statistics"]["bonferroni_p_value"] = adjusted
        candidate["failure_statistics"]["significant_unadjusted_0_05"] = p_value < 0.05
        candidate["failure_statistics"]["significant_bonferroni_0_05"] = adjusted < 0.05
        candidate["claim"] = (
            "dataset_failure_correlation_observed_not_causal"
            if adjusted < 0.05
            else "dataset_anomaly_screening_candidate_not_significant"
        )

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            -float(item["anomaly_score"]),
            float(item["failure_statistics"]["p_value"]),
            str(item["dataset"]["dataset_id"]),
            str(item["dataset"]["version"]),
        ),
    )[:max_results]
    for index, candidate in enumerate(ordered_candidates, start=1):
        candidate["candidate_id"] = f"dataset_failure_{index:03d}"

    return {
        "schema_version": DATASET_FAILURE_CORRELATION_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "method": DATASET_FAILURE_CORRELATION_METHOD,
        "scope": "dataset_failure_correlation",
        "causal_claim": False,
        "privacy_mode": "metadata_only",
        "raw_text_in_output": False,
        "run_count": total_runs,
        "dataset_metadata_run_count": dataset_runs,
        "failure_run_count": failure_runs,
        "non_failure_run_count": non_failure_runs,
        "metric_run_count": metric_run_count,
        "candidate_count": len(ordered_candidates),
        "candidates": ordered_candidates,
        "warnings": warnings,
        "skipped_counts": dict(sorted(skipped.items())),
        "parameters": {
            "min_total_runs": min_total_runs,
            "min_dataset_runs": min_dataset_runs,
            "max_results": max_results,
            "p_value_adjustment": "bonferroni",
            "small_sample_policy": "do_not_report_candidates_below_min_total_runs",
            "dataset_identity_policy": "explicit_artifact_dataset_id_only",
        },
        "algorithm": {
            "failure_test": "Fisher exact test on 2x2 dataset exposure by failure outcome",
            "metric_anomaly": "max absolute standardized mean difference over numeric metrics",
            "anomaly_score": "max(metric_anomaly_score, absolute_failure_rate_difference)",
            "claim_wording": "correlation observed in this project, not causation confirmed",
            "complexity": "O(R * (D + M)) after artifact/metric parsing",
            "network": False,
            "database_mutation": False,
            "derived_graph_edges": False,
        },
    }


def _load_project_outcomes(
    project_root: Path, project_id: str
) -> tuple[tuple[DatasetRunOutcome, ...], dict[str, int]]:
    db_path = project_database_path(project_root)
    connection = connect_database(db_path)
    try:
        run_rows = connection.execute(
            """
            SELECT runs.run_id, runs.metrics_json, runs.artifacts_json
            FROM runs
            JOIN experiments ON experiments.id = runs.experiment_id
            WHERE experiments.project_id = ?
            ORDER BY runs.timestamp, runs.run_id
            """,
            (project_id,),
        ).fetchall()
        failure_rows = connection.execute(
            """
            SELECT failures.id, failures.run_id
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

    failure_ids_by_run: dict[str, list[str]] = {}
    for row in failure_rows:
        failure_ids_by_run.setdefault(str(row["run_id"]), []).append(str(row["id"]))

    skipped_counts = {"invalid_dataset_metadata": 0, "non_numeric_metrics": 0}
    outcomes: list[DatasetRunOutcome] = []
    for row in run_rows:
        run_id = str(row["run_id"])
        artifacts = _safe_json_array(str(row["artifacts_json"]), field="artifacts_json")
        metrics_payload = _safe_json_object(str(row["metrics_json"]), field="metrics_json")
        datasets, dataset_skipped = _datasets_from_artifacts(artifacts)
        metrics, metric_skipped = _numeric_metrics(metrics_payload)
        skipped_counts["invalid_dataset_metadata"] += dataset_skipped
        skipped_counts["non_numeric_metrics"] += metric_skipped
        failure_ids = tuple(failure_ids_by_run.get(run_id, ()))
        outcomes.append(
            DatasetRunOutcome(
                run_id=run_id,
                datasets=datasets,
                metrics=metrics,
                has_failure=bool(failure_ids),
                failure_ids=failure_ids,
            )
        )
    return tuple(outcomes), skipped_counts


def _candidate_for_dataset(
    feature: _DatasetFeature,
    *,
    exposed_run_ids: set[str],
    outcomes: tuple[DatasetRunOutcome, ...],
    min_dataset_runs: int,
) -> dict[str, Any] | None:
    exposed_failure: list[str] = []
    exposed_non_failure: list[str] = []
    unexposed_failure: list[str] = []
    unexposed_non_failure: list[str] = []
    exposed_failure_ids: list[str] = []

    for outcome in outcomes:
        exposed = outcome.run_id in exposed_run_ids
        if exposed and outcome.has_failure:
            exposed_failure.append(outcome.run_id)
            exposed_failure_ids.extend(outcome.failure_ids)
        elif exposed:
            exposed_non_failure.append(outcome.run_id)
        elif outcome.has_failure:
            unexposed_failure.append(outcome.run_id)
        else:
            unexposed_non_failure.append(outcome.run_id)

    a = len(exposed_failure)
    b = len(exposed_non_failure)
    c = len(unexposed_failure)
    d = len(unexposed_non_failure)
    exposed_total = a + b
    unexposed_total = c + d
    if exposed_total < min_dataset_runs or unexposed_total < min_dataset_runs:
        return None

    exposed_rate = a / exposed_total if exposed_total else 0.0
    unexposed_rate = c / unexposed_total if unexposed_total else 0.0
    risk_difference = exposed_rate - unexposed_rate
    metric_anomaly = _strongest_metric_anomaly(
        exposed_run_ids=exposed_run_ids,
        outcomes=outcomes,
        min_dataset_runs=min_dataset_runs,
    )
    metric_score = float(metric_anomaly["score"]) if metric_anomaly is not None else 0.0

    return {
        "candidate_id": "",
        "dataset": {
            "feature_id": feature.feature_id,
            "dataset_id": feature.dataset_id,
            "dataset_id_redacted": feature.dataset_id_redacted,
            "version": feature.version,
            "version_redacted": feature.version_redacted,
        },
        "anomaly_score": round(max(metric_score, abs(risk_difference)), 6),
        "failure_statistics": {
            "test": "fisher_exact_two_sided",
            "p_value": round(_fisher_exact_two_sided(a, b, c, d), 12),
            "risk_difference": round(risk_difference, 6),
            "dataset_failure_rate": round(exposed_rate, 6),
            "other_failure_rate": round(unexposed_rate, 6),
        },
        "metric_anomaly": metric_anomaly,
        "contingency_table": {
            "dataset_failure": a,
            "dataset_non_failure": b,
            "other_failure": c,
            "other_non_failure": d,
        },
        "sample_size": {
            "runs": len(outcomes),
            "dataset_runs": exposed_total,
            "other_runs": unexposed_total,
            "failure_runs": a + c,
            "non_failure_runs": b + d,
        },
        "evidence": {
            "dataset_failure_run_ids": _limit_ids(exposed_failure),
            "dataset_non_failure_run_ids": _limit_ids(exposed_non_failure),
            "other_failure_run_ids": _limit_ids(unexposed_failure),
            "failure_ids": _limit_ids(exposed_failure_ids),
        },
        "claim": "dataset_anomaly_screening_candidate_not_significant",
        "human_review_required": True,
    }


def _strongest_metric_anomaly(
    *,
    exposed_run_ids: set[str],
    outcomes: tuple[DatasetRunOutcome, ...],
    min_dataset_runs: int,
) -> dict[str, Any] | None:
    metric_names = sorted({name for outcome in outcomes for name in outcome.metrics})
    best: dict[str, Any] | None = None
    for metric_name in metric_names:
        exposed_values = [
            outcome.metrics[metric_name]
            for outcome in outcomes
            if outcome.run_id in exposed_run_ids and metric_name in outcome.metrics
        ]
        other_values = [
            outcome.metrics[metric_name]
            for outcome in outcomes
            if outcome.run_id not in exposed_run_ids and metric_name in outcome.metrics
        ]
        if len(exposed_values) < min_dataset_runs or len(other_values) < min_dataset_runs:
            continue
        exposed_mean = sum(exposed_values) / len(exposed_values)
        other_mean = sum(other_values) / len(other_values)
        score = _standardized_mean_difference(exposed_values, other_values)
        metric_label, metric_redacted = _safe_label(metric_name)
        direction = (
            "higher"
            if exposed_mean > other_mean
            else "lower"
            if exposed_mean < other_mean
            else "flat"
        )
        candidate = {
            "metric_name": metric_label,
            "metric_name_redacted": metric_redacted,
            "score": round(score, 6),
            "dataset_mean": _round_float(exposed_mean),
            "other_mean": _round_float(other_mean),
            "dataset_n": len(exposed_values),
            "other_n": len(other_values),
            "direction": direction,
        }
        if best is None or (
            float(candidate["score"]),
            str(candidate["metric_name"]),
        ) > (
            float(best["score"]),
            str(best["metric_name"]),
        ):
            best = candidate
    return best


def _standardized_mean_difference(first: list[float], second: list[float]) -> float:
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    first_var = _population_variance(first, first_mean)
    second_var = _population_variance(second, second_mean)
    pooled = math.sqrt((first_var + second_var) / 2)
    diff = abs(first_mean - second_mean)
    return diff / pooled if pooled > 0 else diff


def _population_variance(values: list[float], mean: float) -> float:
    return sum((value - mean) ** 2 for value in values) / len(values)


def _datasets_from_artifacts(artifacts: list[Any]) -> tuple[tuple[DatasetIdentity, ...], int]:
    datasets: list[DatasetIdentity] = []
    skipped = 0
    for artifact in artifacts:
        if not isinstance(artifact, dict) or "dataset_id" not in artifact:
            continue
        dataset_id = artifact.get("dataset_id")
        version = _dataset_version_value(artifact)
        if not _is_supported_label_value(dataset_id) or not _is_supported_label_value(
            version, allow_none=True
        ):
            skipped += 1
            continue
        dataset_id_text = str(dataset_id).strip()
        version_text = str(version).strip() if version is not None else "unknown"
        if not dataset_id_text:
            skipped += 1
            continue
        datasets.append(
            DatasetIdentity(dataset_id=dataset_id_text, version=version_text or "unknown")
        )
    deduped = {(dataset.dataset_id, dataset.version): dataset for dataset in datasets}
    return tuple(
        sorted(deduped.values(), key=lambda item: (item.dataset_id, item.version))
    ), skipped


def _dataset_version_value(artifact: dict[str, object]) -> object:
    for key in ("dataset_version", "version", "dataset_sha256"):
        if key in artifact:
            return artifact[key]
    return "unknown"


def _numeric_metrics(metrics: dict[str, Any]) -> tuple[dict[str, float], int]:
    numeric: dict[str, float] = {}
    skipped = 0
    for key, value in sorted(metrics.items()):
        key_text = str(key).strip()
        if not key_text or any(ord(char) < 32 for char in key_text):
            skipped += 1
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            skipped += 1
            continue
        numeric[key_text] = float(value)
    return numeric, skipped


def _dataset_feature(dataset: DatasetIdentity) -> _DatasetFeature:
    dataset_id, dataset_id_redacted = _safe_label(dataset.dataset_id)
    version, version_redacted = _safe_label(dataset.version)
    payload = {
        "dataset_id": dataset_id,
        "dataset_id_redacted": dataset_id_redacted,
        "version": version,
        "version_redacted": version_redacted,
    }
    return _DatasetFeature(
        dataset_id=dataset_id,
        dataset_id_redacted=dataset_id_redacted,
        version=version,
        version_redacted=version_redacted,
        feature_id=f"dataset_feature:{compute_text_hash(json.dumps(payload, sort_keys=True))[:16]}",
    )


def _safe_label(value: object) -> tuple[str, bool]:
    text = str(value).strip()
    if _is_safe_label(text):
        return text, False
    return f"sha256:{compute_text_hash(text)[:16]}", True


def _is_safe_label(text: str) -> bool:
    if not text or not _SAFE_LABEL_RE.fullmatch(text):
        return False
    lowered = text.casefold()
    if any(token in lowered for token in _SENSITIVE_TOKENS):
        return False
    if "/" in text or "\\" in text:
        return False
    return True


def _is_supported_label_value(value: object, *, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    if isinstance(value, bool):
        return False
    return isinstance(value, (str, int, float)) and not (
        isinstance(value, float) and not math.isfinite(value)
    )


def _dataset_warnings(
    *,
    total_runs: int,
    dataset_runs: int,
    failure_runs: int,
    non_failure_runs: int,
    metric_run_count: int,
    min_total_runs: int,
) -> list[str]:
    warnings: list[str] = []
    if total_runs < min_total_runs:
        warnings.append(
            "Insufficient data: dataset-failure correlation requires at least "
            f"{min_total_runs} runs before reporting dataset-failure candidates."
        )
    if dataset_runs == 0:
        warnings.append(
            "Insufficient dataset metadata: no explicit artifact dataset_id values were found."
        )
    if failure_runs == 0:
        warnings.append("No confirmed failure runs were found.")
    if non_failure_runs == 0:
        warnings.append("No non-failure comparison runs were found.")
    if metric_run_count == 0:
        warnings.append("No finite numeric metrics were found for dataset metric anomaly scoring.")
    return warnings


def _validate_parameters(*, min_total_runs: int, min_dataset_runs: int, max_results: int) -> None:
    if min_total_runs < 1:
        raise PmemValidationError("Dataset-failure correlation requires min_total_runs >= 1.")
    if min_dataset_runs < 1:
        raise PmemValidationError("Dataset-failure correlation requires min_dataset_runs >= 1.")
    if max_results < 1:
        raise PmemValidationError("Dataset-failure correlation requires max_results >= 1.")


def _safe_json_object(raw_json: str, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise PmemValidationError(f"Run {field} could not be parsed.") from exc
    if not isinstance(payload, dict):
        raise PmemValidationError(f"Run {field} must be an object.")
    return payload


def _safe_json_array(raw_json: str, *, field: str) -> list[Any]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise PmemValidationError(f"Run {field} could not be parsed.") from exc
    if not isinstance(payload, list):
        raise PmemValidationError(f"Run {field} must be an array.")
    return payload


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2
    min_a = max(0, col1 - row2)
    max_a = min(row1, col1)
    observed = _hypergeom_probability(a, row1=row1, col1=col1, total=total)
    p_value = 0.0
    for candidate_a in range(min_a, max_a + 1):
        probability = _hypergeom_probability(candidate_a, row1=row1, col1=col1, total=total)
        if probability <= observed + 1e-15:
            p_value += probability
    return min(max(p_value, 0.0), 1.0)


def _hypergeom_probability(a: int, *, row1: int, col1: int, total: int) -> float:
    return math.exp(_log_comb(col1, a) + _log_comb(total - col1, row1 - a) - _log_comb(total, row1))


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _limit_ids(values: list[str]) -> list[str]:
    return sorted(dict.fromkeys(values))[:DEFAULT_MAX_EVIDENCE_IDS]


def _round_float(value: float) -> float:
    if not math.isfinite(value):
        return value
    return round(value, 6)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
