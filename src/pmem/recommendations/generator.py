"""recommendation generator conservative recommendation generator.

The generator emits project-local recommendation candidates only after recommendation evidence
evidence linking verifies every referenced graph node against SQLite
provenance. It does not create graph edges, expose CLI commands, call network
services, or claim causal/root-cause explanations.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.graph.ingestion import GraphDocument, build_graph_from_database_readonly
from pmem.graph.schema import NodeType, config_node_id, failure_node_id, run_node_id
from pmem.patterns.anomaly import anomaly_detection_payload
from pmem.patterns.config_failure import _features_for_config
from pmem.recommendations.evidence import link_recommendation_evidence_from_document
from pmem.recommendations.model import (
    EvidenceItem,
    EvidenceSource,
    Recommendation,
    RecommendationConfidence,
    RecommendationType,
)
from pmem.repositories.sqlite import connect_database_readonly, execute, project_database_path
from pmem.services.project_context import require_project_context_readonly
from pmem.utils.hashing import compute_text_hash

DEFAULT_MAX_RECOMMENDATIONS = 5
DEFAULT_MAX_EVIDENCE_ITEMS = 5
DEFAULT_MIN_AVOID_FAILURES = 2
DEFAULT_MIN_VERIFY_RUNS = 4
DEFAULT_MIN_PROMOTE_EXPERIMENTS = 3


@dataclass(frozen=True, slots=True)
class _RunEvidence:
    run_id: str
    experiment_id: str
    status: str
    timestamp: str
    config: dict[str, Any]
    config_hash: str | None
    primary_metric_value: float | None
    failure_ids: tuple[str, ...]

    @property
    def has_failure(self) -> bool:
        return bool(self.failure_ids)


@dataclass(frozen=True, slots=True)
class _GenerationContext:
    project_root: Path
    db_path: Path
    document: GraphDocument
    runs: tuple[_RunEvidence, ...]
    primary_metric: str | None
    metric_direction: str
    generated_at: datetime


def generate_recommendations(
    project_root: str | Path,
    *,
    generated_at: datetime | None = None,
    max_recommendations: int = DEFAULT_MAX_RECOMMENDATIONS,
) -> tuple[Recommendation, ...]:
    """Generate recommendation candidates with verified evidence only."""

    if max_recommendations < 1:
        return ()
    context = _load_generation_context(project_root, generated_at=generated_at)
    candidates = (
        _try_next_recommendation(context),
        _avoid_recommendation(context),
        _verify_recommendation(context),
        _promote_recommendation(context),
        _investigate_recommendation(context),
    )
    verified: list[Recommendation] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        if candidate is None or candidate.recommendation_id in seen_ids:
            continue
        link_recommendation_evidence_from_document(context.db_path, context.document, candidate)
        verified.append(candidate)
        seen_ids.add(candidate.recommendation_id)
        if len(verified) >= max_recommendations:
            break
    return tuple(verified)


def _load_generation_context(
    project_root: str | Path,
    *,
    generated_at: datetime | None,
) -> _GenerationContext:
    project_context = require_project_context_readonly(project_root)
    clean_generated_at = generated_at or datetime.now(timezone.utc)
    if clean_generated_at.tzinfo is None or clean_generated_at.utcoffset() is None:
        clean_generated_at = clean_generated_at.replace(tzinfo=timezone.utc)
    db_path = project_database_path(project_context.root)
    document = build_graph_from_database_readonly(db_path)
    runs = _load_run_evidence(
        db_path,
        project_id=project_context.project.id,
        primary_metric=project_context.project.primary_metric,
    )
    direction = project_context.project.metric_direction or "max"
    if direction not in {"max", "min"}:
        direction = "max"
    return _GenerationContext(
        project_root=project_context.root,
        db_path=db_path,
        document=document,
        runs=runs,
        primary_metric=project_context.project.primary_metric,
        metric_direction=direction,
        generated_at=clean_generated_at,
    )


def _load_run_evidence(
    db_path: Path,
    *,
    project_id: str,
    primary_metric: str | None,
) -> tuple[_RunEvidence, ...]:
    connection = connect_database_readonly(db_path)
    try:
        run_rows = execute(
            connection,
            """
            SELECT runs.run_id, runs.experiment_id, runs.status, runs.config_hash,
                   runs.config_json, runs.metrics_json, runs.timestamp
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

    failures_by_run: dict[str, list[str]] = defaultdict(list)
    for row in failure_rows:
        failures_by_run[str(row["run_id"])].append(str(row["id"]))

    runs: list[_RunEvidence] = []
    for row in run_rows:
        run_id = str(row["run_id"])
        runs.append(
            _RunEvidence(
                run_id=run_id,
                experiment_id=str(row["experiment_id"]),
                status=str(row["status"]),
                timestamp=str(row["timestamp"]),
                config=_safe_json_object_or_empty(str(row["config_json"])),
                config_hash=str(row["config_hash"]) if row["config_hash"] is not None else None,
                primary_metric_value=_primary_metric_value(
                    str(row["metrics_json"]), primary_metric
                ),
                failure_ids=tuple(failures_by_run.get(run_id, ())),
            )
        )
    return tuple(runs)


def _try_next_recommendation(context: _GenerationContext) -> Recommendation | None:
    best_run = _best_successful_run(context)
    if best_run is None or best_run.config_hash is None:
        return None
    supporting = [
        _run_evidence(best_run, "Best observed successful run by project primary metric."),
        _config_evidence(best_run.config_hash, "Configuration fingerprint for the best run."),
    ]
    opposing = _failure_evidence_for_config(context, best_run.config_hash)
    related_failures = _failure_evidence_for_config(context, best_run.config_hash)
    return Recommendation(
        recommendation_id=_recommendation_id(
            RecommendationType.TRY_NEXT,
            supporting,
            opposing,
            related_failures,
        ),
        type=RecommendationType.TRY_NEXT,
        title="Try one controlled variant near the best observed run",
        description=(
            f"Based on {len(supporting)} verified graph evidence items. This is a "
            "project-local follow-up candidate, not evidence of expected improvement."
        ),
        supporting_evidence=supporting,
        opposing_evidence=opposing,
        related_failures=related_failures,
        confidence=_confidence(len(supporting)),
        suggested_action=(
            "Run one small controlled variation near the linked configuration and "
            "compare the primary metric before adopting it."
        ),
        generated_at=context.generated_at,
    )


def _avoid_recommendation(context: _GenerationContext) -> Recommendation | None:
    candidates = [
        recommendation
        for recommendation in (
            _avoid_exact_config_recommendation(context),
            _avoid_config_feature_recommendation(context),
        )
        if recommendation is not None
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            -len(item.related_failures),
            -len(item.supporting_evidence),
            item.title,
            item.recommendation_id,
        ),
    )[0]


def _avoid_exact_config_recommendation(context: _GenerationContext) -> Recommendation | None:
    groups = _runs_by_config(context.runs)
    candidates: list[tuple[str, tuple[_RunEvidence, ...], tuple[str, ...]]] = []
    for config_hash, runs in groups.items():
        failure_ids = tuple(failure_id for run in runs for failure_id in run.failure_ids)
        if len(failure_ids) >= DEFAULT_MIN_AVOID_FAILURES:
            candidates.append((config_hash, runs, failure_ids))
    if not candidates:
        return None
    config_hash, runs, failure_ids = sorted(
        candidates,
        key=lambda item: (-len(item[2]), item[0]),
    )[0]
    failed_runs = tuple(run for run in runs if run.has_failure)
    successful_runs = tuple(run for run in runs if _is_successful(run))
    supporting = [
        _config_evidence(
            config_hash,
            f"Configuration fingerprint linked to {len(failure_ids)} confirmed failures.",
        ),
        *[
            _run_evidence(run, "Run observed in repeated-failure config group.")
            for run in failed_runs
        ],
    ][:DEFAULT_MAX_EVIDENCE_ITEMS]
    opposing = [
        _run_evidence(run, "Successful run in the same config group.")
        for run in successful_runs[:DEFAULT_MAX_EVIDENCE_ITEMS]
    ]
    related_failures = [
        _failure_evidence(failure_id, "Confirmed failure in repeated-failure group.")
        for failure_id in failure_ids[:DEFAULT_MAX_EVIDENCE_ITEMS]
    ]
    return Recommendation(
        recommendation_id=_recommendation_id(
            RecommendationType.AVOID,
            supporting,
            opposing,
            related_failures,
        ),
        type=RecommendationType.AVOID,
        title="Review a config group with repeated confirmed failures",
        description=(
            f"{len(failure_ids)} confirmed failures share one config fingerprint. "
            "Treat this as an avoid/review candidate, not a causal explanation."
        ),
        supporting_evidence=supporting,
        opposing_evidence=opposing,
        related_failures=related_failures,
        confidence=_confidence(len(failure_ids)),
        suggested_action=(
            "Pause broad reuse of this config fingerprint until the linked failures "
            "are reviewed against successful counter-evidence."
        ),
        generated_at=context.generated_at,
    )


def _avoid_config_feature_recommendation(context: _GenerationContext) -> Recommendation | None:
    feature_groups: dict[str, tuple[Any, list[_RunEvidence]]] = {}
    for run in context.runs:
        features, _skipped = _features_for_config(run.config)
        for feature in features:
            existing = feature_groups.get(feature.feature_id)
            if existing is None:
                feature_groups[feature.feature_id] = (feature, [run])
            else:
                existing[1].append(run)

    best_metric = _best_successful_metric(context)
    candidates: list[
        tuple[tuple[int, int, str, str], Any, tuple[_RunEvidence, ...], tuple[str, ...]]
    ] = []
    for feature, grouped_runs in feature_groups.values():
        runs = tuple(sorted(grouped_runs, key=lambda item: (item.timestamp, item.run_id)))
        failure_ids = tuple(failure_id for run in runs for failure_id in run.failure_ids)
        if len(failure_ids) < DEFAULT_MIN_AVOID_FAILURES:
            continue
        strong_counterexamples = tuple(
            run
            for run in runs
            if _is_strong_successful_run(
                run,
                best_metric=best_metric,
                metric_direction=context.metric_direction,
            )
        )
        if strong_counterexamples:
            continue
        strong_failure_labels = tuple(
            run
            for run in runs
            if run.has_failure
            and _is_strong_metric_value(
                run.primary_metric_value,
                best_metric=best_metric,
                metric_direction=context.metric_direction,
            )
        )
        if strong_failure_labels:
            continue
        failed_runs = tuple(run for run in runs if run.has_failure)
        if len(failed_runs) < DEFAULT_MIN_AVOID_FAILURES:
            continue
        candidates.append(
            (
                (-len(failure_ids), -len(failed_runs), str(feature.key), str(feature.value)),
                feature,
                failed_runs,
                failure_ids,
            )
        )

    if not candidates:
        return None

    _sort_key, feature, failed_runs, failure_ids = sorted(candidates, key=lambda item: item[0])[0]
    feature_label = f"{feature.key}={feature.value}"
    supporting = [
        _run_evidence(
            run,
            f"Run with confirmed failure linked to config feature {feature_label}.",
        )
        for run in failed_runs[:DEFAULT_MAX_EVIDENCE_ITEMS]
    ]
    related_failures = [
        _failure_evidence(failure_id, "Confirmed failure linked to config-feature avoid signal.")
        for failure_id in failure_ids[:DEFAULT_MAX_EVIDENCE_ITEMS]
    ]
    return Recommendation(
        recommendation_id=_recommendation_id(
            RecommendationType.AVOID,
            supporting,
            [],
            related_failures,
        ),
        type=RecommendationType.AVOID,
        title=f"Avoid broad reuse of config feature {feature_label}",
        description=(
            f"{len(failure_ids)} confirmed failures and {len(failed_runs)} failed runs "
            f"share config feature {feature_label}. Strong successful counter-evidence "
            "was not found for this feature. Treat this as an avoid candidate for "
            "human review, not causal proof."
        ),
        supporting_evidence=supporting,
        opposing_evidence=[],
        related_failures=related_failures,
        confidence=_confidence(len(failure_ids)),
        suggested_action=(
            f"Do not reuse {feature_label} broadly until the linked runs are reviewed; "
            "compare nearby safer values under a controlled rerun."
        ),
        generated_at=context.generated_at,
    )


def _verify_recommendation(context: _GenerationContext) -> Recommendation | None:
    anomaly_payload = _anomaly_payload(context)
    reproducibility = list(anomaly_payload.get("reproducibility_candidates", ()))
    if not reproducibility:
        return None
    candidate = reproducibility[0]
    run_ids = _candidate_run_ids(candidate)
    if len(run_ids) < DEFAULT_MIN_VERIFY_RUNS:
        return None
    supporting_runs = _runs_for_ids(context.runs, run_ids)
    supporting = [
        _run_evidence(run, "Run in same-config high-variance group.")
        for run in supporting_runs[:DEFAULT_MAX_EVIDENCE_ITEMS]
    ]
    related_failures = _failure_evidence_for_runs(supporting_runs)
    return Recommendation(
        recommendation_id=_recommendation_id(
            RecommendationType.VERIFY,
            supporting,
            [],
            related_failures,
        ),
        type=RecommendationType.VERIFY,
        title="Verify a same-config high-variance result group",
        description=(
            f"{len(supporting_runs)} same-config runs show high primary-metric spread. "
            "This is a reproducibility screening candidate."
        ),
        supporting_evidence=supporting,
        opposing_evidence=[],
        related_failures=related_failures,
        confidence=_confidence(len(supporting_runs)),
        suggested_action=(
            "Rerun the linked config under controlled conditions before using the "
            "result as stable evidence."
        ),
        generated_at=context.generated_at,
    )


def _promote_recommendation(context: _GenerationContext) -> Recommendation | None:
    top_runs = _top_successful_run_per_experiment(context)
    if len(top_runs) < DEFAULT_MIN_PROMOTE_EXPERIMENTS:
        return None
    best = _best_run(top_runs, context.metric_direction)
    if best is None:
        return None
    supporting = [
        _run_evidence(run, "Top successful run for one experiment.")
        for run in top_runs[:DEFAULT_MAX_EVIDENCE_ITEMS]
    ]
    opposing = _failure_evidence_for_config(context, best.config_hash)
    related_failures = _failure_evidence_for_config(context, best.config_hash)
    return Recommendation(
        recommendation_id=_recommendation_id(
            RecommendationType.PROMOTE,
            supporting,
            opposing,
            related_failures,
        ),
        type=RecommendationType.PROMOTE,
        title="Promote the best observed run for human review",
        description=(
            f"The linked run is the best observed primary-metric candidate across "
            f"{len(top_runs)} experiments. Promotion still requires human review."
        ),
        supporting_evidence=supporting,
        opposing_evidence=opposing,
        related_failures=related_failures,
        confidence=_confidence(len(top_runs)),
        suggested_action=(
            "Review the linked best run, compare it with experiment peers, and only "
            "promote after confirming reproducibility."
        ),
        generated_at=context.generated_at,
    )


def _investigate_recommendation(context: _GenerationContext) -> Recommendation | None:
    anomaly_payload = _anomaly_payload(context)
    outliers = list(anomaly_payload.get("metric_outliers", ()))
    if not outliers:
        return None
    candidate = outliers[0]
    run_ids = _candidate_run_ids(candidate)
    if not run_ids:
        return None
    supporting_runs = _runs_for_ids(context.runs, run_ids)
    if not supporting_runs:
        return None
    supporting = [
        _run_evidence(run, "Run linked to anomaly detection metric-outlier screening.")
        for run in supporting_runs[:DEFAULT_MAX_EVIDENCE_ITEMS]
    ]
    related_failures = _failure_evidence_for_runs(supporting_runs)
    return Recommendation(
        recommendation_id=_recommendation_id(
            RecommendationType.INVESTIGATE,
            supporting,
            [],
            related_failures,
        ),
        type=RecommendationType.INVESTIGATE,
        title="Investigate a primary-metric outlier candidate",
        description=(
            f"anomaly detection anomaly screening flagged {len(supporting_runs)} linked run(s). "
            "This is an audit prompt, not an automatic root-cause finding."
        ),
        supporting_evidence=supporting,
        opposing_evidence=[],
        related_failures=related_failures,
        confidence=_confidence(len(supporting_runs)),
        suggested_action=(
            "Inspect the linked run metadata and compare nearby runs before treating "
            "the outlier as meaningful."
        ),
        generated_at=context.generated_at,
    )


def _best_successful_run(context: _GenerationContext) -> _RunEvidence | None:
    candidates = tuple(
        run for run in context.runs if _is_successful(run) and run.primary_metric_value is not None
    )
    return _best_run(candidates, context.metric_direction)


def _best_successful_metric(context: _GenerationContext) -> float | None:
    best_run = _best_successful_run(context)
    return best_run.primary_metric_value if best_run is not None else None


def _top_successful_run_per_experiment(context: _GenerationContext) -> tuple[_RunEvidence, ...]:
    grouped: dict[str, list[_RunEvidence]] = defaultdict(list)
    for run in context.runs:
        if _is_successful(run) and run.primary_metric_value is not None:
            grouped[run.experiment_id].append(run)
    top_runs = [
        best
        for best in (_best_run(tuple(runs), context.metric_direction) for runs in grouped.values())
        if best is not None
    ]
    return tuple(
        sorted(
            top_runs,
            key=lambda run: _metric_sort_key(run, context.metric_direction),
        )
    )


def _best_run(runs: tuple[_RunEvidence, ...], metric_direction: str) -> _RunEvidence | None:
    scored = tuple(run for run in runs if run.primary_metric_value is not None)
    if not scored:
        return None
    return sorted(scored, key=lambda run: _metric_sort_key(run, metric_direction))[0]


def _metric_sort_key(run: _RunEvidence, metric_direction: str) -> tuple[float, str, str]:
    value = run.primary_metric_value
    if value is None:
        score = math.inf
    elif metric_direction == "min":
        score = float(value)
    else:
        score = -float(value)
    return (score, run.timestamp, run.run_id)


def _anomaly_payload(context: _GenerationContext) -> dict[str, Any]:
    return anomaly_detection_payload(
        context.project_root,
        generated_at=context.generated_at.isoformat(),
    )


def _runs_by_config(runs: tuple[_RunEvidence, ...]) -> dict[str, tuple[_RunEvidence, ...]]:
    grouped: dict[str, list[_RunEvidence]] = defaultdict(list)
    for run in runs:
        if run.config_hash:
            grouped[run.config_hash].append(run)
    return {
        config_hash: tuple(sorted(items, key=lambda run: (run.timestamp, run.run_id)))
        for config_hash, items in grouped.items()
    }


def _runs_for_ids(
    runs: tuple[_RunEvidence, ...], run_ids: tuple[str, ...]
) -> tuple[_RunEvidence, ...]:
    by_id = {run.run_id: run for run in runs}
    return tuple(by_id[run_id] for run_id in run_ids if run_id in by_id)


def _candidate_run_ids(candidate: dict[str, Any]) -> tuple[str, ...]:
    ordered: list[str] = []
    run_id = candidate.get("run_id")
    if isinstance(run_id, str):
        ordered.append(run_id)
    evidence = candidate.get("evidence")
    if isinstance(evidence, dict):
        raw_run_ids = evidence.get("run_ids")
        if isinstance(raw_run_ids, list):
            ordered.extend(str(item) for item in raw_run_ids if isinstance(item, str))
    seen: set[str] = set()
    deduped: list[str] = []
    for item in ordered:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return tuple(deduped)


def _failure_evidence_for_config(
    context: _GenerationContext,
    config_hash: str | None,
) -> list[EvidenceItem]:
    if not config_hash:
        return []
    failures = [
        failure_id
        for run in context.runs
        if run.config_hash == config_hash
        for failure_id in run.failure_ids
    ]
    return [
        _failure_evidence(failure_id, "Confirmed failure linked to the same config group.")
        for failure_id in failures[:DEFAULT_MAX_EVIDENCE_ITEMS]
    ]


def _failure_evidence_for_runs(runs: tuple[_RunEvidence, ...]) -> list[EvidenceItem]:
    failure_ids = [failure_id for run in runs for failure_id in run.failure_ids]
    return [
        _failure_evidence(failure_id, "Confirmed failure linked to recommendation evidence.")
        for failure_id in failure_ids[:DEFAULT_MAX_EVIDENCE_ITEMS]
    ]


def _run_evidence(run: _RunEvidence, summary: str) -> EvidenceItem:
    return EvidenceItem(
        entity_id=run_node_id(run.run_id),
        entity_type=NodeType.RUN,
        source=EvidenceSource.RUN_METRIC,
        summary=summary,
    )


def _config_evidence(config_hash: str, summary: str) -> EvidenceItem:
    return EvidenceItem(
        entity_id=config_node_id(config_hash),
        entity_type=NodeType.CONFIG,
        source=EvidenceSource.GRAPH,
        summary=summary,
    )


def _failure_evidence(failure_id: str, summary: str) -> EvidenceItem:
    return EvidenceItem(
        entity_id=failure_node_id(failure_id),
        entity_type=NodeType.FAILURE,
        source=EvidenceSource.FAILURE_RECORD,
        summary=summary,
    )


def _confidence(evidence_count: int) -> RecommendationConfidence:
    if evidence_count >= 6:
        return RecommendationConfidence.HIGH
    if evidence_count >= 3:
        return RecommendationConfidence.MEDIUM
    return RecommendationConfidence.LOW


def _recommendation_id(
    recommendation_type: RecommendationType,
    supporting: list[EvidenceItem],
    opposing: list[EvidenceItem],
    related_failures: list[EvidenceItem],
) -> str:
    evidence_keys = (
        [f"support:{item.entity_type.value}:{item.entity_id}" for item in supporting]
        + [f"oppose:{item.entity_type.value}:{item.entity_id}" for item in opposing]
        + [f"failure:{item.entity_type.value}:{item.entity_id}" for item in related_failures]
    )
    digest = compute_text_hash(
        json.dumps(
            {
                "type": recommendation_type.value,
                "evidence": sorted(evidence_keys),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )[:12]
    return f"rec_d59_{recommendation_type.value}_{digest}"


def _is_successful(run: _RunEvidence) -> bool:
    return run.status == "success" and not run.has_failure


def _is_strong_successful_run(
    run: _RunEvidence,
    *,
    best_metric: float | None,
    metric_direction: str,
) -> bool:
    """Return whether a run is strong enough to block an avoid feature signal.

    This deliberately uses only successful, failure-free runs so stale metrics
    attached to failed commands cannot become counter-evidence.
    """

    if best_metric is None or run.primary_metric_value is None or not _is_successful(run):
        return False
    return _is_strong_metric_value(
        run.primary_metric_value,
        best_metric=best_metric,
        metric_direction=metric_direction,
    )


def _is_strong_metric_value(
    value: float | None,
    *,
    best_metric: float | None,
    metric_direction: str,
) -> bool:
    if best_metric is None or value is None:
        return False
    tolerance = max(abs(best_metric) * 0.10, 0.05)
    if metric_direction == "min":
        return value <= best_metric + tolerance
    return value >= best_metric - tolerance


def _primary_metric_value(metrics_json: str, primary_metric: str | None) -> float | None:
    if not primary_metric:
        return None
    try:
        parsed = json.loads(metrics_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    value = parsed.get(primary_metric)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value_float = float(value)
    return value_float if math.isfinite(value_float) else None


def _safe_json_object_or_empty(raw_json: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
