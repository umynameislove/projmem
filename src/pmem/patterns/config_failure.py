"""config-failure correlation mining.

The analysis is intentionally conservative: it treats config/failure
relationships as screening candidates, not explanations. Each candidate is a
2x2 Fisher exact test over ``config_key=value`` exposure and confirmed failure
presence. No raw failure text is read or emitted.
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
from pmem.repositories.sqlite import connect_database_readonly, execute, project_database_path
from pmem.services.project_context import require_project_context_readonly
from pmem.utils.hashing import compute_text_hash

CONFIG_FAILURE_CORRELATION_SCHEMA_VERSION = "config-failure-correlation-v1"
CONFIG_FAILURE_CORRELATION_METHOD = "fisher_exact_2x2_config_failure_v1"
DEFAULT_MIN_TOTAL_RUNS = 10
DEFAULT_MIN_FEATURE_GROUP_RUNS = 2
DEFAULT_MAX_RESULTS = 25
DEFAULT_MAX_EVIDENCE_IDS = 20

_SAFE_CATEGORY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_SENSITIVE_KEY_TOKENS = {
    "api",
    "api_key",
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
class RunFailureOutcome:
    """One run's config exposure and confirmed-failure outcome."""

    run_id: str
    config: dict[str, Any]
    has_failure: bool
    failure_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ConfigFeature:
    key: str
    key_redacted: bool
    value: str
    value_redacted: bool
    feature_id: str


def config_failure_correlation_payload(
    project_root: str | Path,
    *,
    min_total_runs: int = DEFAULT_MIN_TOTAL_RUNS,
    min_feature_group_runs: int = DEFAULT_MIN_FEATURE_GROUP_RUNS,
    max_results: int = DEFAULT_MAX_RESULTS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic config-failure correlation report."""

    if min_total_runs < 1:
        raise PmemValidationError("Config-failure correlation requires min_total_runs >= 1.")
    if min_feature_group_runs < 1:
        raise PmemValidationError(
            "Config-failure correlation requires min_feature_group_runs >= 1."
        )
    if max_results < 1:
        raise PmemValidationError("Config-failure correlation requires max_results >= 1.")

    context = require_project_context_readonly(project_root)
    outcomes = _load_project_outcomes(context.root, context.project.id)
    return config_failure_correlation_from_outcomes(
        outcomes,
        min_total_runs=min_total_runs,
        min_feature_group_runs=min_feature_group_runs,
        max_results=max_results,
        generated_at=generated_at,
    )


def config_failure_correlation_from_outcomes(
    outcomes: tuple[RunFailureOutcome, ...],
    *,
    min_total_runs: int = DEFAULT_MIN_TOTAL_RUNS,
    min_feature_group_runs: int = DEFAULT_MIN_FEATURE_GROUP_RUNS,
    max_results: int = DEFAULT_MAX_RESULTS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a config-failure correlation report from preloaded run/failure outcomes."""

    if min_total_runs < 1:
        raise PmemValidationError("Config-failure correlation requires min_total_runs >= 1.")
    if min_feature_group_runs < 1:
        raise PmemValidationError(
            "Config-failure correlation requires min_feature_group_runs >= 1."
        )
    if max_results < 1:
        raise PmemValidationError("Config-failure correlation requires max_results >= 1.")

    ordered_outcomes = tuple(sorted(outcomes, key=lambda item: item.run_id))
    total_runs = len(ordered_outcomes)
    failure_runs = sum(1 for item in ordered_outcomes if item.has_failure)
    non_failure_runs = total_runs - failure_runs
    warnings = _dataset_warnings(
        total_runs=total_runs,
        failure_runs=failure_runs,
        non_failure_runs=non_failure_runs,
        min_total_runs=min_total_runs,
    )
    skipped_counts = {"sensitive_config_keys": 0, "non_scalar_values": 0}
    features_by_run: dict[str, tuple[_ConfigFeature, ...]] = {}
    for outcome in ordered_outcomes:
        features, skipped = _features_for_config(outcome.config)
        features_by_run[outcome.run_id] = features
        for key, count in skipped.items():
            skipped_counts[key] = skipped_counts.get(key, 0) + count

    candidates: list[dict[str, Any]] = []
    if not warnings:
        feature_index: dict[str, tuple[_ConfigFeature, set[str]]] = {}
        for run_id, features in features_by_run.items():
            for feature in features:
                existing = feature_index.get(feature.feature_id)
                if existing is None:
                    feature_index[feature.feature_id] = (feature, {run_id})
                else:
                    existing[1].add(run_id)

        for feature, exposed_run_ids in feature_index.values():
            candidate = _candidate_for_feature(
                feature,
                exposed_run_ids=exposed_run_ids,
                outcomes=ordered_outcomes,
                min_feature_group_runs=min_feature_group_runs,
            )
            if candidate is not None:
                candidates.append(candidate)

    feature_count = len(candidates)
    for candidate in candidates:
        p_value = float(candidate["statistics"]["p_value"])
        candidate["statistics"]["bonferroni_p_value"] = round(min(1.0, p_value * feature_count), 12)
        candidate["statistics"]["significant_unadjusted_0_05"] = p_value < 0.05
        candidate["statistics"]["significant_bonferroni_0_05"] = (
            candidate["statistics"]["bonferroni_p_value"] < 0.05
        )
        candidate["claim"] = (
            "correlation_observed_not_causal"
            if candidate["statistics"]["significant_bonferroni_0_05"]
            else "screening_candidate_not_significant"
        )

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            float(item["statistics"]["p_value"]),
            -abs(float(item["statistics"]["risk_difference"])),
            str(item["feature"]["key"]),
            str(item["feature"]["value"]),
        ),
    )[:max_results]
    for index, candidate in enumerate(ordered_candidates, start=1):
        candidate["candidate_id"] = f"config_failure_{index:03d}"

    return {
        "schema_version": CONFIG_FAILURE_CORRELATION_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "method": CONFIG_FAILURE_CORRELATION_METHOD,
        "scope": "config_failure_correlation",
        "causal_claim": False,
        "privacy_mode": "metadata_only",
        "raw_text_in_output": False,
        "run_count": total_runs,
        "failure_run_count": failure_runs,
        "non_failure_run_count": non_failure_runs,
        "candidate_count": len(ordered_candidates),
        "candidates": ordered_candidates,
        "warnings": warnings,
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "parameters": {
            "min_total_runs": min_total_runs,
            "min_feature_group_runs": min_feature_group_runs,
            "max_results": max_results,
            "p_value_adjustment": "bonferroni",
            "small_sample_policy": "do_not_report_p_values_below_min_total_runs",
        },
        "algorithm": {
            "test": "Fisher exact test on 2x2 config exposure by failure outcome",
            "effect_size": "odds_ratio_with_haldane_anscombe_correction",
            "confidence_interval": "approximate_log_odds_ratio_95_percent",
            "multiple_testing": "raw p-values plus Bonferroni-adjusted p-values",
            "claim_wording": "correlation observed in this project, not causation confirmed",
            "complexity": "O(R * F) after config flattening, where R=runs and F=config features",
            "network": False,
            "database_mutation": False,
        },
    }


def _load_project_outcomes(project_root: Path, project_id: str) -> tuple[RunFailureOutcome, ...]:
    db_path = project_database_path(project_root)
    connection = connect_database_readonly(db_path)
    try:
        run_rows = execute(
            connection,
            """
            SELECT runs.run_id, runs.config_json
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

    outcomes = []
    for row in run_rows:
        run_id = str(row["run_id"])
        config = _safe_json_object(str(row["config_json"]))
        failure_ids = tuple(failure_ids_by_run.get(run_id, ()))
        outcomes.append(
            RunFailureOutcome(
                run_id=run_id,
                config=config,
                has_failure=bool(failure_ids),
                failure_ids=failure_ids,
            )
        )
    return tuple(outcomes)


def _candidate_for_feature(
    feature: _ConfigFeature,
    *,
    exposed_run_ids: set[str],
    outcomes: tuple[RunFailureOutcome, ...],
    min_feature_group_runs: int,
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
    if exposed_total < min_feature_group_runs or unexposed_total < min_feature_group_runs:
        return None

    p_value = _fisher_exact_two_sided(a, b, c, d)
    odds_ratio = _corrected_odds_ratio(a, b, c, d)
    ci_low, ci_high = _odds_ratio_ci95(a, b, c, d)
    exposed_rate = a / exposed_total if exposed_total else 0.0
    unexposed_rate = c / unexposed_total if unexposed_total else 0.0
    risk_difference = exposed_rate - unexposed_rate

    return {
        "candidate_id": "",
        "feature": {
            "feature_id": feature.feature_id,
            "key": feature.key,
            "key_redacted": feature.key_redacted,
            "value": feature.value,
            "value_redacted": feature.value_redacted,
        },
        "contingency_table": {
            "exposed_failure": a,
            "exposed_non_failure": b,
            "unexposed_failure": c,
            "unexposed_non_failure": d,
        },
        "statistics": {
            "test": "fisher_exact_two_sided",
            "p_value": round(p_value, 12),
            "odds_ratio": _round_float(odds_ratio),
            "odds_ratio_ci95": [_round_float(ci_low), _round_float(ci_high)],
            "risk_difference": round(risk_difference, 6),
            "exposed_failure_rate": round(exposed_rate, 6),
            "unexposed_failure_rate": round(unexposed_rate, 6),
        },
        "sample_size": {
            "runs": len(outcomes),
            "exposed_runs": exposed_total,
            "unexposed_runs": unexposed_total,
            "failure_runs": a + c,
            "non_failure_runs": b + d,
        },
        "evidence": {
            "exposed_failure_run_ids": _limit_ids(exposed_failure),
            "exposed_non_failure_run_ids": _limit_ids(exposed_non_failure),
            "unexposed_failure_run_ids": _limit_ids(unexposed_failure),
            "failure_ids": _limit_ids(exposed_failure_ids),
        },
        "claim": "screening_candidate_not_significant",
        "human_review_required": True,
    }


def _features_for_config(
    config: dict[str, Any],
) -> tuple[tuple[_ConfigFeature, ...], dict[str, int]]:
    skipped = {"sensitive_config_keys": 0, "non_scalar_values": 0}
    features: list[_ConfigFeature] = []
    for key, value in _flatten_config(config):
        if _is_sensitive_key(key):
            skipped["sensitive_config_keys"] += 1
            continue
        if not isinstance(value, _SCALAR_TYPES):
            skipped["non_scalar_values"] += 1
            continue
        key_label, key_redacted = _safe_key_label(key)
        value_label, redacted = _safe_value_label(value)
        feature_payload = {
            "key": key_label,
            "key_redacted": key_redacted,
            "value": value_label,
            "value_redacted": redacted,
        }
        feature_id = (
            f"config_feature:{compute_text_hash(json.dumps(feature_payload, sort_keys=True))[:16]}"
        )
        features.append(
            _ConfigFeature(
                key=key_label,
                key_redacted=key_redacted,
                value=value_label,
                value_redacted=redacted,
                feature_id=feature_id,
            )
        )
    deduped = {feature.feature_id: feature for feature in features}
    return tuple(sorted(deduped.values(), key=lambda item: (item.key, item.value))), skipped


def _flatten_config(config: dict[str, Any], prefix: str = "") -> tuple[tuple[str, object], ...]:
    flattened: list[tuple[str, object]] = []
    for key in sorted(config):
        cleaned_key = str(key).strip()
        if not cleaned_key or any(ord(char) < 32 for char in cleaned_key):
            continue
        path = f"{prefix}.{cleaned_key}" if prefix else cleaned_key
        value = config[key]
        if isinstance(value, dict):
            flattened.extend(_flatten_config(value, path))
        else:
            flattened.append((path, value))
    return tuple(flattened)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    parts = re.split(r"[^a-z0-9_]+", normalized)
    return any(
        token and any(sensitive in token for sensitive in _SENSITIVE_KEY_TOKENS) for token in parts
    )


def _safe_key_label(key: str) -> tuple[str, bool]:
    if _is_safe_category(key):
        return key, False
    return f"sha256:{compute_text_hash(key)[:16]}", True


def _safe_value_label(value: object) -> tuple[str, bool]:
    if value is None:
        return "null", False
    if isinstance(value, bool):
        return "true" if value else "false", False
    if isinstance(value, int):
        return str(value), False
    if isinstance(value, float):
        if not math.isfinite(value):
            return "non_finite_number", True
        return format(value, ".12g"), False
    text = str(value).strip()
    if _is_safe_category(text):
        return text, False
    return f"sha256:{compute_text_hash(text)[:16]}", True


def _is_safe_category(text: str) -> bool:
    if not text or not _SAFE_CATEGORY_RE.fullmatch(text):
        return False
    lowered = text.casefold()
    if any(token in lowered for token in _SENSITIVE_KEY_TOKENS):
        return False
    if "/" in text or "\\" in text:
        return False
    return True


def _dataset_warnings(
    *,
    total_runs: int,
    failure_runs: int,
    non_failure_runs: int,
    min_total_runs: int,
) -> list[str]:
    warnings: list[str] = []
    if total_runs < min_total_runs:
        warnings.append(
            "Insufficient data: config-failure correlation requires at least "
            f"{min_total_runs} runs before reporting p-values."
        )
    if failure_runs == 0:
        warnings.append("No confirmed failure runs were found.")
    if non_failure_runs == 0:
        warnings.append("No non-failure comparison runs were found.")
    return warnings


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
    row2 = total - row1
    return (
        math.exp(_log_comb(col1, a) + _log_comb(total - col1, row1 - a) - _log_comb(total, row1))
        if row2 >= 0
        else 0.0
    )


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _corrected_odds_ratio(a: int, b: int, c: int, d: int) -> float:
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def _odds_ratio_ci95(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    aa, bb, cc, dd = (a + 0.5, b + 0.5, c + 0.5, d + 0.5)
    odds_ratio = (aa * dd) / (bb * cc)
    standard_error = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lower = math.exp(math.log(odds_ratio) - 1.96 * standard_error)
    upper = math.exp(math.log(odds_ratio) + 1.96 * standard_error)
    return lower, upper


def _safe_json_object(raw_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise PmemValidationError("Run config JSON could not be parsed.") from exc
    if not isinstance(payload, dict):
        raise PmemValidationError("Run config JSON must be an object.")
    return payload


def _limit_ids(values: list[str]) -> list[str]:
    return sorted(dict.fromkeys(values))[:DEFAULT_MAX_EVIDENCE_IDS]


def _round_float(value: float) -> float:
    if not math.isfinite(value):
        return value
    return round(value, 6)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
