"""anomaly detection metric outlier and reproducibility screening.

anomaly detection is a conservative anomaly-screening layer. It uses finite numeric metrics,
experiment ids, run ids, timestamps, and config hashes. It does not read or emit
raw config values, command text, failure text, artifact paths, or causal claims.
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

ANOMALY_DETECTION_SCHEMA_VERSION = "anomaly-detection-v1"
ANOMALY_DETECTION_METHOD = "iqr_outlier_same_config_variance_v1"
DEFAULT_MIN_EXPERIMENT_METRIC_RUNS = 8
DEFAULT_MIN_CONFIG_GROUP_RUNS = 4
DEFAULT_IQR_MULTIPLIER = 1.5
DEFAULT_MIN_METRIC_RANGE = 0.10
DEFAULT_MIN_STANDARD_DEVIATION = 0.05
DEFAULT_MAX_RESULTS = 25

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


@dataclass(frozen=True, slots=True)
class AnomalyRunOutcome:
    """One run's finite numeric metrics and config identity for anomaly detection."""

    run_id: str
    experiment_id: str
    timestamp: str
    metrics: dict[str, float]
    config_hash: str | None = None


@dataclass(frozen=True, slots=True)
class _MetricPoint:
    run_id: str
    experiment_id: str
    timestamp: str
    metric_name: str
    metric_label: str
    metric_redacted: bool
    value: float
    config_hash: str


def anomaly_detection_payload(
    project_root: str | Path,
    *,
    min_experiment_metric_runs: int = DEFAULT_MIN_EXPERIMENT_METRIC_RUNS,
    min_config_group_runs: int = DEFAULT_MIN_CONFIG_GROUP_RUNS,
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
    min_metric_range: float = DEFAULT_MIN_METRIC_RANGE,
    min_standard_deviation: float = DEFAULT_MIN_STANDARD_DEVIATION,
    max_results: int = DEFAULT_MAX_RESULTS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return deterministic anomaly detection metric anomaly candidates for one project."""

    _validate_parameters(
        min_experiment_metric_runs=min_experiment_metric_runs,
        min_config_group_runs=min_config_group_runs,
        iqr_multiplier=iqr_multiplier,
        min_metric_range=min_metric_range,
        min_standard_deviation=min_standard_deviation,
        max_results=max_results,
    )
    context = require_project_context(project_root)
    runs, skipped_counts = _load_project_runs(context.root, context.project.id)
    return anomaly_detection_from_outcomes(
        runs,
        primary_metric=context.project.primary_metric,
        min_experiment_metric_runs=min_experiment_metric_runs,
        min_config_group_runs=min_config_group_runs,
        iqr_multiplier=iqr_multiplier,
        min_metric_range=min_metric_range,
        min_standard_deviation=min_standard_deviation,
        max_results=max_results,
        generated_at=generated_at,
        skipped_counts=skipped_counts,
    )


def anomaly_detection_from_outcomes(
    runs: tuple[AnomalyRunOutcome, ...],
    *,
    primary_metric: str | None = None,
    min_experiment_metric_runs: int = DEFAULT_MIN_EXPERIMENT_METRIC_RUNS,
    min_config_group_runs: int = DEFAULT_MIN_CONFIG_GROUP_RUNS,
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
    min_metric_range: float = DEFAULT_MIN_METRIC_RANGE,
    min_standard_deviation: float = DEFAULT_MIN_STANDARD_DEVIATION,
    max_results: int = DEFAULT_MAX_RESULTS,
    generated_at: str | None = None,
    skipped_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build anomaly detection anomaly candidates from preloaded metric outcomes."""

    _validate_parameters(
        min_experiment_metric_runs=min_experiment_metric_runs,
        min_config_group_runs=min_config_group_runs,
        iqr_multiplier=iqr_multiplier,
        min_metric_range=min_metric_range,
        min_standard_deviation=min_standard_deviation,
        max_results=max_results,
    )
    skipped = {
        "missing_config_hash": 0,
        "non_numeric_metrics": 0,
        **(skipped_counts or {}),
    }
    points, point_skipped = _metric_points(runs)
    for key, count in point_skipped.items():
        skipped[key] = skipped.get(key, 0) + count
    primary_metric_label, primary_metric_redacted = (
        _safe_label(primary_metric) if primary_metric else ("", False)
    )
    outliers = _metric_outlier_candidates(
        points,
        min_experiment_metric_runs=min_experiment_metric_runs,
        iqr_multiplier=iqr_multiplier,
        max_results=max_results,
        primary_metric=primary_metric,
    )
    reproducibility = _reproducibility_candidates(
        points,
        min_config_group_runs=min_config_group_runs,
        min_metric_range=min_metric_range,
        min_standard_deviation=min_standard_deviation,
        max_results=max_results,
        primary_metric=primary_metric,
    )
    warnings = _warnings(
        run_count=len(runs),
        metric_point_count=len(points),
        outlier_count=len(outliers),
        reproducibility_count=len(reproducibility),
        min_experiment_metric_runs=min_experiment_metric_runs,
    )

    return {
        "schema_version": ANOMALY_DETECTION_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "method": ANOMALY_DETECTION_METHOD,
        "scope": "metric_outlier_reproducibility_screening",
        "causal_claim": False,
        "privacy_mode": "metadata_only",
        "raw_text_in_output": False,
        "primary_metric": primary_metric_label,
        "primary_metric_redacted": primary_metric_redacted,
        "metric_scope": "all_finite_numeric_metrics",
        "run_count": len(runs),
        "metric_point_count": len(points),
        "metric_outlier_count": len(outliers),
        "reproducibility_candidate_count": len(reproducibility),
        "metric_outliers": outliers,
        "reproducibility_candidates": reproducibility,
        "warnings": warnings,
        "skipped_counts": dict(sorted(skipped.items())),
        "parameters": {
            "min_experiment_metric_runs": min_experiment_metric_runs,
            "min_config_group_runs": min_config_group_runs,
            "iqr_multiplier": iqr_multiplier,
            "min_metric_range": min_metric_range,
            "min_standard_deviation": min_standard_deviation,
            "max_results": max_results,
            "small_sample_policy": "do_not_report_anomaly_candidates_below_minimum_group_sizes",
        },
        "algorithm": {
            "metric_outliers": "IQR fences per experiment and metric",
            "reproducibility": "same config hash plus high metric range and standard deviation",
            "claim_wording": "screening candidates for human review, not causal findings",
            "complexity": "O(R * M log(R * M)) after metric parsing",
            "network": False,
            "database_mutation": False,
            "derived_graph_edges": False,
        },
    }


def _load_project_runs(
    project_root: Path,
    project_id: str,
) -> tuple[tuple[AnomalyRunOutcome, ...], dict[str, int]]:
    db_path = project_database_path(project_root)
    connection = connect_database(db_path)
    try:
        rows = connection.execute(
            """
            SELECT runs.run_id, runs.experiment_id, runs.metrics_json,
                   runs.config_json, runs.config_hash, runs.timestamp
            FROM runs
            JOIN experiments ON experiments.id = runs.experiment_id
            WHERE experiments.project_id = ?
            ORDER BY runs.timestamp, runs.run_id
            """,
            (project_id,),
        ).fetchall()
    finally:
        connection.close()

    skipped = {"non_numeric_metrics": 0}
    outcomes: list[AnomalyRunOutcome] = []
    for row in rows:
        metrics_payload = _safe_json_object(str(row["metrics_json"]), field="metrics_json")
        metrics, metric_skipped = _numeric_metrics(metrics_payload)
        skipped["non_numeric_metrics"] += metric_skipped
        config_hash = str(row["config_hash"]) if row["config_hash"] is not None else ""
        if not config_hash:
            config_hash = compute_text_hash(_stable_config_json(str(row["config_json"])))
        outcomes.append(
            AnomalyRunOutcome(
                run_id=str(row["run_id"]),
                experiment_id=str(row["experiment_id"]),
                timestamp=str(row["timestamp"]),
                metrics=metrics,
                config_hash=config_hash,
            )
        )
    return tuple(outcomes), skipped


def _metric_outlier_candidates(
    points: tuple[_MetricPoint, ...],
    *,
    min_experiment_metric_runs: int,
    iqr_multiplier: float,
    max_results: int,
    primary_metric: str | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[_MetricPoint]] = {}
    for point in points:
        grouped.setdefault((point.experiment_id, point.metric_name), []).append(point)

    candidates: list[dict[str, Any]] = []
    for (experiment_id, metric_name), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: (item.value, item.timestamp, item.run_id))
        if len(ordered) < min_experiment_metric_runs:
            continue
        values = [item.value for item in ordered]
        q1 = _percentile(values, 0.25)
        q3 = _percentile(values, 0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        for point in ordered:
            if lower <= point.value <= upper:
                continue
            direction = "high" if point.value > upper else "low"
            fence = upper if direction == "high" else lower
            score = abs(point.value - fence) / iqr
            candidates.append(
                {
                    "candidate_id": "",
                    "kind": "metric_outlier",
                    "claim": "metric_outlier_candidate_not_causal",
                    "experiment_id": experiment_id,
                    "metric_name": point.metric_label,
                    "metric_name_redacted": point.metric_redacted,
                    "is_primary_metric": metric_name == primary_metric,
                    "run_id": point.run_id,
                    "timestamp": point.timestamp,
                    "value": _round_float(point.value),
                    "direction": direction,
                    "score": _round_float(score),
                    "iqr": _round_float(iqr),
                    "lower_fence": _round_float(lower),
                    "upper_fence": _round_float(upper),
                    "sample_size": len(ordered),
                    "evidence": {
                        "run_ids": [item.run_id for item in sorted(group, key=_point_order_key)],
                    },
                    "human_review_required": True,
                }
            )

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            not bool(item["is_primary_metric"]),
            -float(item["score"]),
            str(item["experiment_id"]),
            str(item["metric_name"]),
            str(item["run_id"]),
        ),
    )[:max_results]
    for index, candidate in enumerate(ordered_candidates, start=1):
        candidate["candidate_id"] = f"metric_outlier_{index:03d}"
    return ordered_candidates


def _reproducibility_candidates(
    points: tuple[_MetricPoint, ...],
    *,
    min_config_group_runs: int,
    min_metric_range: float,
    min_standard_deviation: float,
    max_results: int,
    primary_metric: str | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[_MetricPoint]] = {}
    for point in points:
        if point.config_hash == "missing":
            continue
        grouped.setdefault((point.config_hash, point.metric_name), []).append(point)

    candidates: list[dict[str, Any]] = []
    for (config_hash, metric_name), group in sorted(grouped.items()):
        ordered = sorted(group, key=_point_order_key)
        if len(ordered) < min_config_group_runs:
            continue
        values = [item.value for item in ordered]
        mean = sum(values) / len(values)
        stddev = _sample_stddev(values, mean)
        metric_range = max(values) - min(values)
        if stddev < min_standard_deviation or metric_range < min_metric_range:
            continue
        first = ordered[0]
        candidates.append(
            {
                "candidate_id": "",
                "kind": "same_config_metric_variance",
                "claim": "potential_reproducibility_issue_not_causal",
                "config_fingerprint": f"sha256:{config_hash[:16]}",
                "metric_name": first.metric_label,
                "metric_name_redacted": first.metric_redacted,
                "is_primary_metric": metric_name == primary_metric,
                "mean": _round_float(mean),
                "standard_deviation": _round_float(stddev),
                "range": _round_float(metric_range),
                "sample_size": len(ordered),
                "experiment_ids": sorted({item.experiment_id for item in ordered}),
                "evidence": {
                    "run_ids": [item.run_id for item in ordered],
                },
                "human_review_required": True,
            }
        )

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            not bool(item["is_primary_metric"]),
            -float(item["standard_deviation"]),
            -float(item["range"]),
            str(item["config_fingerprint"]),
            str(item["metric_name"]),
        ),
    )[:max_results]
    for index, candidate in enumerate(ordered_candidates, start=1):
        candidate["candidate_id"] = f"reproducibility_{index:03d}"
    return ordered_candidates


def _metric_points(
    runs: tuple[AnomalyRunOutcome, ...],
) -> tuple[tuple[_MetricPoint, ...], dict[str, int]]:
    points: list[_MetricPoint] = []
    skipped = {"missing_config_hash": 0, "non_numeric_metrics": 0}
    for run in sorted(runs, key=lambda item: (item.timestamp, item.run_id)):
        config_hash = str(run.config_hash or "").strip()
        if not _is_sha256(config_hash):
            skipped["missing_config_hash"] += 1
            config_hash = "missing"
        for metric_name, value in sorted(run.metrics.items()):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                skipped["non_numeric_metrics"] += 1
                continue
            metric_label, metric_redacted = _safe_label(metric_name)
            points.append(
                _MetricPoint(
                    run_id=run.run_id,
                    experiment_id=run.experiment_id,
                    timestamp=run.timestamp,
                    metric_name=metric_name,
                    metric_label=metric_label,
                    metric_redacted=metric_redacted,
                    value=float(value),
                    config_hash=config_hash,
                )
            )
    return tuple(points), skipped


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


def _warnings(
    *,
    run_count: int,
    metric_point_count: int,
    outlier_count: int,
    reproducibility_count: int,
    min_experiment_metric_runs: int,
) -> list[str]:
    warnings: list[str] = []
    if run_count < min_experiment_metric_runs:
        warnings.append(
            "Insufficient data: anomaly detection requires at least "
            f"{min_experiment_metric_runs} runs before reporting metric outliers."
        )
    if metric_point_count == 0:
        warnings.append("No finite numeric metric observations were found.")
    if outlier_count == 0:
        warnings.append("No metric outlier candidates were generated.")
    if reproducibility_count == 0:
        warnings.append("No same-config reproducibility candidates were generated.")
    return warnings


def _validate_parameters(
    *,
    min_experiment_metric_runs: int,
    min_config_group_runs: int,
    iqr_multiplier: float,
    min_metric_range: float,
    min_standard_deviation: float,
    max_results: int,
) -> None:
    if min_experiment_metric_runs < 4:
        raise PmemValidationError("Anomaly detection requires min_experiment_metric_runs >= 4.")
    if min_config_group_runs < 2:
        raise PmemValidationError("Anomaly detection requires min_config_group_runs >= 2.")
    if not math.isfinite(iqr_multiplier) or iqr_multiplier <= 0:
        raise PmemValidationError("Anomaly detection requires iqr_multiplier > 0.")
    if not math.isfinite(min_metric_range) or min_metric_range < 0:
        raise PmemValidationError("Anomaly detection requires min_metric_range >= 0.")
    if not math.isfinite(min_standard_deviation) or min_standard_deviation < 0:
        raise PmemValidationError("Anomaly detection requires min_standard_deviation >= 0.")
    if max_results < 1:
        raise PmemValidationError("Anomaly detection requires max_results >= 1.")


def _stable_config_json(raw_json: str) -> str:
    payload = _safe_json_object(raw_json, field="config_json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _safe_json_object(raw_json: str, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise PmemValidationError(f"Run {field} could not be parsed.") from exc
    if not isinstance(payload, dict):
        raise PmemValidationError(f"Run {field} must be an object.")
    return payload


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise PmemValidationError("Anomaly percentile calculation requires at least one value.")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def _sample_stddev(values: list[float], mean: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


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


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _point_order_key(point: _MetricPoint) -> tuple[str, str, str, float]:
    return (point.timestamp, point.experiment_id, point.run_id, point.value)


def _round_float(value: float) -> float:
    if not math.isfinite(value):
        return value
    return round(value, 6)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
