"""temporal metric drift and decision-shift screening.

Temporal reports are audit aids, not causal impact estimates. The module uses
only run timestamps, finite numeric metrics, and decision ids/timestamps. It
does not read or emit decision rationale, failure text, command text, or paths.
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

TEMPORAL_ANALYSIS_SCHEMA_VERSION = "temporal-analysis-v1"
TEMPORAL_ANALYSIS_METHOD = "linear_drift_decision_before_after_v1"
DEFAULT_MIN_TOTAL_RUNS = 8
DEFAULT_MIN_DECISION_SIDE_RUNS = 3
DEFAULT_MAX_DECISION_RESULTS = 25

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
class TemporalRunOutcome:
    """One run's timestamped finite metrics for temporal analysis."""

    run_id: str
    experiment_id: str
    timestamp: str
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    """One decision timestamp; description/rationale are intentionally absent."""

    decision_id: str
    created_at: str
    experiment_id: str | None = None


@dataclass(frozen=True, slots=True)
class _TimedMetric:
    run_id: str
    experiment_id: str
    timestamp: datetime
    value: float


@dataclass(frozen=True, slots=True)
class _DecisionPoint:
    decision_id: str
    created_at: datetime
    experiment_id: str | None


def temporal_analysis_payload(
    project_root: str | Path,
    *,
    min_total_runs: int = DEFAULT_MIN_TOTAL_RUNS,
    min_decision_side_runs: int = DEFAULT_MIN_DECISION_SIDE_RUNS,
    max_decision_results: int = DEFAULT_MAX_DECISION_RESULTS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic temporal analysis report for one project."""

    _validate_parameters(
        min_total_runs=min_total_runs,
        min_decision_side_runs=min_decision_side_runs,
        max_decision_results=max_decision_results,
    )
    context = require_project_context(project_root)
    runs, decisions, skipped_counts = _load_project_temporal_inputs(
        context.root,
        context.project.id,
    )
    return temporal_analysis_from_outcomes(
        runs,
        decisions,
        primary_metric=context.project.primary_metric,
        metric_direction=context.project.metric_direction,
        min_total_runs=min_total_runs,
        min_decision_side_runs=min_decision_side_runs,
        max_decision_results=max_decision_results,
        generated_at=generated_at,
        skipped_counts=skipped_counts,
    )


def temporal_analysis_from_outcomes(
    runs: tuple[TemporalRunOutcome, ...],
    decisions: tuple[DecisionEvent, ...],
    *,
    primary_metric: str | None,
    metric_direction: str | None = None,
    min_total_runs: int = DEFAULT_MIN_TOTAL_RUNS,
    min_decision_side_runs: int = DEFAULT_MIN_DECISION_SIDE_RUNS,
    max_decision_results: int = DEFAULT_MAX_DECISION_RESULTS,
    generated_at: str | None = None,
    skipped_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build temporal drift and decision-shift candidates from inputs."""

    _validate_parameters(
        min_total_runs=min_total_runs,
        min_decision_side_runs=min_decision_side_runs,
        max_decision_results=max_decision_results,
    )
    skipped = {
        "missing_primary_metric": 0,
        "missing_primary_metric_observations": 0,
        "invalid_timestamps": 0,
        "non_finite_primary_metric_values": 0,
        "non_numeric_metrics": 0,
        **(skipped_counts or {}),
    }
    metric_name = str(primary_metric).strip() if primary_metric is not None else ""
    metric_label, metric_redacted = _safe_label(metric_name) if metric_name else ("", False)
    clean_direction = _normalize_metric_direction(metric_direction)
    timed_metrics: tuple[_TimedMetric, ...] = ()
    decision_points: tuple[_DecisionPoint, ...] = ()
    warnings: list[str] = []

    if not metric_name:
        skipped["missing_primary_metric"] += 1
        warnings.append("Primary metric is required before temporal analysis can run.")
    else:
        timed_metrics, parse_skipped = _timed_metric_values(runs, metric_name)
        skipped["invalid_timestamps"] += parse_skipped["invalid_timestamps"]
        skipped["missing_primary_metric_observations"] += parse_skipped[
            "missing_primary_metric_observations"
        ]
        skipped["non_finite_primary_metric_values"] += parse_skipped[
            "non_finite_primary_metric_values"
        ]
        decision_points, decision_skipped = _decision_points(decisions)
        skipped["invalid_timestamps"] += decision_skipped
        warnings.extend(
            _dataset_warnings(
                run_count=len(runs),
                metric_run_count=len(timed_metrics),
                decision_count=len(decision_points),
                min_total_runs=min_total_runs,
            )
        )

    drift = (
        _metric_drift_candidate(
            timed_metrics,
            metric_name=metric_label,
            metric_name_redacted=metric_redacted,
            metric_direction=clean_direction,
        )
        if metric_name and len(timed_metrics) >= min_total_runs
        else None
    )
    decision_candidates = (
        _decision_shift_candidates(
            timed_metrics=timed_metrics,
            decision_points=decision_points,
            metric_name=metric_label,
            metric_name_redacted=metric_redacted,
            metric_direction=clean_direction,
            min_decision_side_runs=min_decision_side_runs,
            max_results=max_decision_results,
        )
        if metric_name and len(timed_metrics) >= min_total_runs
        else []
    )

    return {
        "schema_version": TEMPORAL_ANALYSIS_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "method": TEMPORAL_ANALYSIS_METHOD,
        "scope": "temporal_metric_drift_decision_shift",
        "causal_claim": False,
        "privacy_mode": "metadata_only",
        "raw_text_in_output": False,
        "primary_metric": metric_label,
        "primary_metric_redacted": metric_redacted,
        "metric_direction": clean_direction,
        "run_count": len(runs),
        "metric_run_count": len(timed_metrics),
        "decision_count": len(decision_points),
        "drift": drift,
        "decision_impact_candidate_count": len(decision_candidates),
        "decision_impact_candidates": decision_candidates,
        "warnings": warnings,
        "skipped_counts": dict(sorted(skipped.items())),
        "parameters": {
            "min_total_runs": min_total_runs,
            "min_decision_side_runs": min_decision_side_runs,
            "max_decision_results": max_decision_results,
            "small_sample_policy": "do_not_report_temporal_statistics_below_min_total_runs",
            "decision_policy": "before_after_screening_not_causal_impact",
        },
        "algorithm": {
            "drift": "ordinary least squares slope over elapsed days",
            "drift_significance": "two-sided normal approximation of slope t-statistic",
            "decision_shift": "before/after mean difference around decision timestamp",
            "decision_significance": "two-sided normal approximation of Welch-style t-statistic",
            "claim_wording": "temporal association observed, not causation confirmed",
            "complexity": "O(R log R + D * R) where R=runs and D=decisions",
            "network": False,
            "database_mutation": False,
            "derived_graph_edges": False,
        },
    }


def _load_project_temporal_inputs(
    project_root: Path,
    project_id: str,
) -> tuple[tuple[TemporalRunOutcome, ...], tuple[DecisionEvent, ...], dict[str, int]]:
    db_path = project_database_path(project_root)
    connection = connect_database(db_path)
    try:
        run_rows = connection.execute(
            """
            SELECT runs.run_id, runs.experiment_id, runs.metrics_json, runs.timestamp
            FROM runs
            JOIN experiments ON experiments.id = runs.experiment_id
            WHERE experiments.project_id = ?
            ORDER BY runs.timestamp, runs.run_id
            """,
            (project_id,),
        ).fetchall()
        decision_rows = connection.execute(
            """
            SELECT id, experiment_id, created_at
            FROM decisions
            WHERE project_id = ?
            ORDER BY created_at, id
            """,
            (project_id,),
        ).fetchall()
    finally:
        connection.close()

    skipped = {"non_numeric_metrics": 0}
    runs: list[TemporalRunOutcome] = []
    for row in run_rows:
        metrics_payload = _safe_json_object(str(row["metrics_json"]), field="metrics_json")
        metrics, metric_skipped = _numeric_metrics(metrics_payload)
        skipped["non_numeric_metrics"] += metric_skipped
        runs.append(
            TemporalRunOutcome(
                run_id=str(row["run_id"]),
                experiment_id=str(row["experiment_id"]),
                timestamp=str(row["timestamp"]),
                metrics=metrics,
            )
        )

    decisions = tuple(
        DecisionEvent(
            decision_id=str(row["id"]),
            experiment_id=str(row["experiment_id"]) if row["experiment_id"] is not None else None,
            created_at=str(row["created_at"]),
        )
        for row in decision_rows
    )
    return tuple(runs), decisions, skipped


def _metric_drift_candidate(
    timed_metrics: tuple[_TimedMetric, ...],
    *,
    metric_name: str,
    metric_name_redacted: bool,
    metric_direction: str | None,
) -> dict[str, Any]:
    ordered = tuple(sorted(timed_metrics, key=lambda item: (item.timestamp, item.run_id)))
    first_timestamp = ordered[0].timestamp
    x_values = [(item.timestamp - first_timestamp).total_seconds() / 86_400.0 for item in ordered]
    y_values = [item.value for item in ordered]
    regression = _linear_regression(x_values, y_values)
    slope = float(regression["slope"])
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    directional_slope = _directional_delta(slope, metric_direction)
    return {
        "metric_name": metric_name,
        "metric_name_redacted": metric_name_redacted,
        "claim": (
            "temporal_drift_candidate_not_causal"
            if float(regression["p_value"]) < 0.05
            else "temporal_drift_screening_candidate_not_significant"
        ),
        "direction": direction,
        "metric_direction_interpretation": _interpret_directional_change(directional_slope),
        "slope_per_day": _round_float(slope),
        "intercept": _round_float(float(regression["intercept"])),
        "r_squared": _round_float(float(regression["r_squared"])),
        "p_value": _round_float(float(regression["p_value"])),
        "t_statistic": _round_float(float(regression["t_statistic"])),
        "sample_size": len(ordered),
        "first_timestamp": _isoformat_utc(ordered[0].timestamp),
        "last_timestamp": _isoformat_utc(ordered[-1].timestamp),
        "evidence": {
            "run_ids": [item.run_id for item in ordered],
            "experiment_ids": sorted({item.experiment_id for item in ordered}),
        },
        "human_review_required": True,
    }


def _decision_shift_candidates(
    *,
    timed_metrics: tuple[_TimedMetric, ...],
    decision_points: tuple[_DecisionPoint, ...],
    metric_name: str,
    metric_name_redacted: bool,
    metric_direction: str | None,
    min_decision_side_runs: int,
    max_results: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    ordered_metrics = tuple(sorted(timed_metrics, key=lambda item: (item.timestamp, item.run_id)))
    for decision in sorted(decision_points, key=lambda item: (item.created_at, item.decision_id)):
        scoped = [
            metric
            for metric in ordered_metrics
            if decision.experiment_id is None or metric.experiment_id == decision.experiment_id
        ]
        before = [metric for metric in scoped if metric.timestamp < decision.created_at]
        after = [metric for metric in scoped if metric.timestamp >= decision.created_at]
        if len(before) < min_decision_side_runs or len(after) < min_decision_side_runs:
            continue
        before_values = [item.value for item in before]
        after_values = [item.value for item in after]
        before_mean = sum(before_values) / len(before_values)
        after_mean = sum(after_values) / len(after_values)
        delta = after_mean - before_mean
        directional_delta = _directional_delta(delta, metric_direction)
        test = _mean_shift_test(before_values, after_values)
        effect_size = _standardized_mean_difference(after_values, before_values)
        candidate = {
            "candidate_id": "",
            "decision_id": decision.decision_id,
            "decision_scope": "experiment" if decision.experiment_id is not None else "project",
            "experiment_id": decision.experiment_id,
            "decision_timestamp": _isoformat_utc(decision.created_at),
            "metric_name": metric_name,
            "metric_name_redacted": metric_name_redacted,
            "claim": (
                "decision_metric_shift_candidate_not_causal"
                if float(test["p_value"]) < 0.05 and abs(effect_size) >= 0.8
                else "decision_metric_shift_screening_candidate_not_significant"
            ),
            "metric_direction_interpretation": _interpret_directional_change(directional_delta),
            "before_mean": _round_float(before_mean),
            "after_mean": _round_float(after_mean),
            "delta": _round_float(delta),
            "directional_delta": _round_float(directional_delta),
            "standardized_mean_difference": _round_float(effect_size),
            "p_value": _round_float(float(test["p_value"])),
            "t_statistic": _round_float(float(test["t_statistic"])),
            "sample_size": {
                "before_runs": len(before),
                "after_runs": len(after),
            },
            "evidence": {
                "before_run_ids": [item.run_id for item in before],
                "after_run_ids": [item.run_id for item in after],
            },
            "human_review_required": True,
        }
        candidates.append(candidate)

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            float(item["p_value"]),
            -abs(float(item["standardized_mean_difference"])),
            str(item["decision_id"]),
        ),
    )[:max_results]
    for index, candidate in enumerate(ordered_candidates, start=1):
        candidate["candidate_id"] = f"decision_shift_{index:03d}"
    return ordered_candidates


def _timed_metric_values(
    runs: tuple[TemporalRunOutcome, ...],
    metric_name: str,
) -> tuple[tuple[_TimedMetric, ...], dict[str, int]]:
    skipped = {
        "invalid_timestamps": 0,
        "missing_primary_metric_observations": 0,
        "non_finite_primary_metric_values": 0,
    }
    values: list[_TimedMetric] = []
    for run in sorted(runs, key=lambda item: (item.timestamp, item.run_id)):
        if metric_name not in run.metrics:
            skipped["missing_primary_metric_observations"] += 1
            continue
        timestamp = _parse_timestamp(run.timestamp)
        if timestamp is None:
            skipped["invalid_timestamps"] += 1
            continue
        value = run.metrics[metric_name]
        if not math.isfinite(value):
            skipped["non_finite_primary_metric_values"] += 1
            continue
        values.append(
            _TimedMetric(
                run_id=run.run_id,
                experiment_id=run.experiment_id,
                timestamp=timestamp,
                value=float(value),
            )
        )
    return tuple(values), skipped


def _decision_points(
    decisions: tuple[DecisionEvent, ...],
) -> tuple[tuple[_DecisionPoint, ...], int]:
    points: list[_DecisionPoint] = []
    skipped = 0
    for decision in sorted(decisions, key=lambda item: (item.created_at, item.decision_id)):
        timestamp = _parse_timestamp(decision.created_at)
        if timestamp is None:
            skipped += 1
            continue
        points.append(
            _DecisionPoint(
                decision_id=decision.decision_id,
                experiment_id=decision.experiment_id,
                created_at=timestamp,
            )
        )
    return tuple(points), skipped


def _linear_regression(x_values: list[float], y_values: list[float]) -> dict[str, float]:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        raise PmemValidationError("Temporal linear regression requires at least 3 metric points.")
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    sxx = sum((x_value - x_mean) ** 2 for x_value in x_values)
    if sxx <= 0:
        raise PmemValidationError("Temporal analysis requires non-identical timestamps.")
    sxy = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * x_value for x_value in x_values]
    residuals = [y_value - fit for y_value, fit in zip(y_values, fitted, strict=True)]
    sse = sum(value * value for value in residuals)
    sst = sum((value - y_mean) ** 2 for value in y_values)
    r_squared = 1.0 if sst <= 0 and sse <= 0 else max(0.0, 1.0 - sse / sst) if sst > 0 else 0.0
    if sse <= 1e-18:
        t_statistic = math.copysign(float("inf"), slope) if slope else 0.0
        p_value = 0.0 if slope else 1.0
    else:
        residual_variance = sse / (len(x_values) - 2)
        slope_se = math.sqrt(residual_variance / sxx)
        t_statistic = slope / slope_se if slope_se > 0 else 0.0
        p_value = _normal_two_sided_p_value(t_statistic)
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "t_statistic": t_statistic,
        "p_value": p_value,
    }


def _mean_shift_test(before_values: list[float], after_values: list[float]) -> dict[str, float]:
    before_mean = sum(before_values) / len(before_values)
    after_mean = sum(after_values) / len(after_values)
    before_var = _sample_variance(before_values, before_mean)
    after_var = _sample_variance(after_values, after_mean)
    standard_error = math.sqrt(before_var / len(before_values) + after_var / len(after_values))
    delta = after_mean - before_mean
    if standard_error <= 1e-18:
        t_statistic = math.copysign(float("inf"), delta) if delta else 0.0
        p_value = 0.0 if delta else 1.0
    else:
        t_statistic = delta / standard_error
        p_value = _normal_two_sided_p_value(t_statistic)
    return {"t_statistic": t_statistic, "p_value": p_value}


def _standardized_mean_difference(first: list[float], second: list[float]) -> float:
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    first_var = _population_variance(first, first_mean)
    second_var = _population_variance(second, second_mean)
    pooled = math.sqrt((first_var + second_var) / 2)
    diff = first_mean - second_mean
    return diff / pooled if pooled > 0 else diff


def _sample_variance(values: list[float], mean: float) -> float:
    if len(values) <= 1:
        return 0.0
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _population_variance(values: list[float], mean: float) -> float:
    return sum((value - mean) ** 2 for value in values) / len(values)


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


def _dataset_warnings(
    *,
    run_count: int,
    metric_run_count: int,
    decision_count: int,
    min_total_runs: int,
) -> list[str]:
    warnings: list[str] = []
    if run_count < min_total_runs:
        warnings.append(
            "Insufficient data: temporal analysis requires at least "
            f"{min_total_runs} runs before reporting temporal statistics."
        )
    if metric_run_count < min_total_runs:
        warnings.append(
            "Insufficient metric observations: primary metric is missing from too many runs."
        )
    if decision_count == 0:
        warnings.append("No decisions with valid timestamps were found for before/after screening.")
    return warnings


def _validate_parameters(
    *,
    min_total_runs: int,
    min_decision_side_runs: int,
    max_decision_results: int,
) -> None:
    if min_total_runs < 3:
        raise PmemValidationError("Temporal analysis requires min_total_runs >= 3.")
    if min_decision_side_runs < 1:
        raise PmemValidationError("Temporal analysis requires min_decision_side_runs >= 1.")
    if max_decision_results < 1:
        raise PmemValidationError("Temporal analysis requires max_decision_results >= 1.")


def _normalize_metric_direction(metric_direction: str | None) -> str | None:
    if metric_direction is None:
        return None
    cleaned = metric_direction.strip().lower()
    if not cleaned:
        return None
    if cleaned not in {"max", "min"}:
        raise PmemValidationError(
            "Temporal analysis metric_direction must be 'max', 'min', or unset."
        )
    return cleaned


def _directional_delta(delta: float, metric_direction: str | None) -> float:
    if metric_direction == "min":
        return -delta
    return delta


def _interpret_directional_change(directional_delta: float) -> str:
    if directional_delta > 0:
        return "metric_improving"
    if directional_delta < 0:
        return "metric_regressing"
    return "flat"


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


def _safe_json_object(raw_json: str, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise PmemValidationError(f"Run {field} could not be parsed.") from exc
    if not isinstance(payload, dict):
        raise PmemValidationError(f"Run {field} must be an object.")
    return payload


def _parse_timestamp(raw_timestamp: str) -> datetime | None:
    text = raw_timestamp.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normal_two_sided_p_value(statistic: float) -> float:
    if math.isinf(statistic):
        return 0.0
    return min(1.0, max(0.0, math.erfc(abs(statistic) / math.sqrt(2.0))))


def _round_float(value: float) -> float:
    if not math.isfinite(value):
        return value
    return round(value, 6)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
