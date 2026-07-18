"""status service read-only assembly of the ``status-v1`` payload (STS-002).

This service maps real project state (summary + graph freshness + optional
recommendation availability) onto the STS-001 contract.

Read-only guarantee (earned, not asserted). Every project read goes through a
shared read-only persistence seam:

- Summary: :func:`pmem.summary.get_project_summary_readonly` uses a
  ``mode=ro`` connection via :func:`require_project_context_readonly`; it never
  migrates, backs up, ``mkdir``s, ``chmod``s, or creates the database, and it
  rejects a symlinked database/config.
- Graph freshness: :func:`compute_graph_source_fingerprint_readonly` opens a
  read-only connection only.
- Opt-in recommendation generation reuses ``recommendation_list_payload``,
  which (after this refactor) resolves through ``require_project_context_readonly``
  and read-only connections end to end.

An out-of-date or checksum-tampered schema raises a safe ``PmemError`` instead
of migrating it.

Task boundary with STS-003: ``collect_status_state`` and
``assemble_status_payload`` never choose the ``next_action`` — the caller
supplies it. The next-action *policy* lives in :mod:`pmem.status.next_action`;
the ``build_status_payload`` convenience here simply applies that policy
(``select_next_action``) and delegates to ``assemble_status_payload`` without
duplicating any assembly logic.

Privacy: the service guarantees payload **shape** and path/control safety and
redacts unsafe project text into stable SHA-256 labels using the same absolute
path detector as the contract (:mod:`pmem.status.textsafety`), so producer and
consumer cannot drift. It cannot prove arbitrary text is secret-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pmem.domain.common import MetricDirection
from pmem.errors import PmemNotFoundError, PmemPersistenceError, PmemValidationError
from pmem.graph.incremental import compute_graph_source_fingerprint_readonly
from pmem.graph.persistence import default_graph_artifact_path, read_graph_document
from pmem.repositories.sqlite import project_database_path
from pmem.services.recommendation_operations import recommendation_list_payload
from pmem.status import (
    STATUS_SCHEMA_VERSION,
    GraphState,
    RecommendationMode,
    StatusBaseline,
    StatusBestRun,
    StatusCounts,
    StatusGraph,
    StatusMetric,
    StatusNextAction,
    StatusPayload,
    StatusProject,
    StatusRecommendations,
    StatusWarning,
    TargetStatus,
    WarningSeverity,
    WarningSource,
)
from pmem.status.next_action import select_next_action
from pmem.status.textsafety import contains_absolute_path, contains_control_chars
from pmem.summary import ProjectSummary, get_project_summary_readonly
from pmem.utils.hashing import compute_text_hash

_MAX_RECOMMENDATIONS = 50
_MAX_PROJECT_NAME_LENGTH = 120
_MAX_OBJECTIVE_LENGTH = 512
_MAX_METRIC_NAME_LENGTH = 512

_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_SEVERITY_RANK = {
    WarningSeverity.ERROR: 0,
    WarningSeverity.WARNING: 1,
    WarningSeverity.INFO: 2,
}


@dataclass(frozen=True, slots=True)
class CollectedStatusState:
    """Read-only status state without any next-action decision.

    STS-003 consumes this, selects exactly one action, and calls
    :func:`assemble_status_payload`.
    """

    project: StatusProject
    metric: StatusMetric
    counts: StatusCounts
    best_run: StatusBestRun
    baseline: StatusBaseline
    graph: StatusGraph
    recommendations: StatusRecommendations
    warnings: tuple[StatusWarning, ...]


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def collect_status_state(
    project_root: str | Path,
    *,
    evaluate_recommendations: bool = False,
    max_recommendations: int = 5,
) -> CollectedStatusState:
    """Collect read-only project status state. Never selects a next action.

    Raises a safe :class:`~pmem.errors.PmemError` (never migrating) when the
    project is uninitialized or its schema is out of date. Recommendation
    generation is opt-in and off by default.
    """

    _validate_recommendation_limit(max_recommendations)
    root = Path(project_root)
    summary = get_project_summary_readonly(root)

    warnings: list[StatusWarning] = []
    project = _map_project(summary, warnings)
    metric = _map_metric(summary, warnings)
    counts = _map_counts(summary)
    best_run = StatusBestRun(run_id=summary.best_run_id, metric_value=summary.best_metric_value)
    baseline = StatusBaseline(run_id=summary.baseline_run_id)
    _append_summary_warnings(summary, warnings)

    graph = _map_graph(root, warnings)
    recommendations = _map_recommendations(
        root,
        evaluate=evaluate_recommendations,
        max_recommendations=max_recommendations,
        warnings=warnings,
    )

    return CollectedStatusState(
        project=project,
        metric=metric,
        counts=counts,
        best_run=best_run,
        baseline=baseline,
        graph=graph,
        recommendations=recommendations,
        warnings=_ordered_warnings(warnings),
    )


def assemble_status_payload(
    state: CollectedStatusState,
    *,
    next_action: StatusNextAction,
) -> StatusPayload:
    """Assemble the final payload. The caller (STS-003) supplies the action."""

    return StatusPayload(
        schema_version=STATUS_SCHEMA_VERSION,
        project=state.project,
        metric=state.metric,
        counts=state.counts,
        best_run=state.best_run,
        baseline=state.baseline,
        graph=state.graph,
        recommendations=state.recommendations,
        warnings=state.warnings,
        next_action=next_action,
        database_mutation=False,
        network=False,
        raw_text_in_output=False,
    )


def build_status_payload(state: CollectedStatusState) -> StatusPayload:
    """Select the single next action (STS-003) and assemble the payload.

    Thin convenience wrapper that does not duplicate assembly logic: it applies
    the deterministic next-action policy and delegates to
    :func:`assemble_status_payload`.
    """

    return assemble_status_payload(state, next_action=select_next_action(state))


# --------------------------------------------------------------------------- #
# Summary -> contract mapping                                                  #
# --------------------------------------------------------------------------- #
def _map_project(summary: ProjectSummary, warnings: list[StatusWarning]) -> StatusProject:
    project_name = _safe_text_or_redact(
        summary.project_name,
        kind="project_name",
        max_length=_MAX_PROJECT_NAME_LENGTH,
        warnings=warnings,
    )
    objective = None
    if summary.objective is not None:
        objective = _safe_text_or_redact(
            summary.objective,
            kind="objective",
            max_length=_MAX_OBJECTIVE_LENGTH,
            warnings=warnings,
        )
    return StatusProject(
        project_id=summary.project_id,
        project_name=project_name,
        objective=objective,
    )


def _map_metric(summary: ProjectSummary, warnings: list[StatusWarning]) -> StatusMetric:
    primary_metric = None
    if summary.primary_metric is not None:
        primary_metric = _safe_text_or_redact(
            summary.primary_metric,
            kind="primary_metric",
            max_length=_MAX_METRIC_NAME_LENGTH,
            warnings=warnings,
        )
    direction = (
        MetricDirection(summary.metric_direction) if summary.metric_direction is not None else None
    )
    return StatusMetric(
        primary_metric=primary_metric,
        direction=direction,
        target_value=summary.target_value,
        best_value=summary.best_metric_value,
        target_status=TargetStatus(summary.target_status),
    )


def _map_counts(summary: ProjectSummary) -> StatusCounts:
    return StatusCounts(
        run_count=summary.run_count,
        successful_run_count=summary.successful_run_count,
        failed_run_count=summary.failed_run_count,
        tracked_path_count=summary.tracked_path_count,
        failure_count=summary.failure_count,
        decision_count=summary.decision_count,
        note_count=summary.note_count,
    )


def _append_summary_warnings(summary: ProjectSummary, warnings: list[StatusWarning]) -> None:
    """Derive typed warnings from structured summary fields (never copying text)."""

    if summary.tracked_path_count == 0:
        warnings.append(
            _warning(
                "no_tracked_paths",
                WarningSeverity.INFO,
                WarningSource.SUMMARY,
                "No files are tracked yet.",
                "Track a file with pmem track.",
            )
        )
    if summary.run_count == 0:
        warnings.append(
            _warning(
                "no_runs",
                WarningSeverity.INFO,
                WarningSource.SUMMARY,
                "No runs have been captured yet.",
                "Capture a run with pmem run.",
            )
        )
    elif summary.successful_run_count == 0:
        warnings.append(
            _warning(
                "no_successful_runs",
                WarningSeverity.WARNING,
                WarningSource.SUMMARY,
                "No successful runs have been captured yet.",
                "Investigate failing runs before relying on metrics.",
            )
        )
    if summary.run_count > 0 and summary.baseline_run_id is None:
        warnings.append(
            _warning(
                "no_baseline",
                WarningSeverity.INFO,
                WarningSource.SUMMARY,
                "No baseline run has been set.",
                "Set a baseline with pmem baseline.",
            )
        )
    if summary.target_status == "not_met":
        warnings.append(
            _warning(
                "target_not_met",
                WarningSeverity.WARNING,
                WarningSource.SUMMARY,
                "The project target has not been met.",
                None,
            )
        )
    if summary.target_status == "no_metric":
        warnings.append(
            _warning(
                "missing_primary_metric",
                WarningSeverity.WARNING,
                WarningSource.SUMMARY,
                "No successful run reported the configured primary metric.",
                None,
            )
        )


# --------------------------------------------------------------------------- #
# Graph state (fingerprint-based; never rebuilds/persists)                     #
# --------------------------------------------------------------------------- #
def _map_graph(root: Path, warnings: list[StatusWarning]) -> StatusGraph:
    graph_path = default_graph_artifact_path(root)

    if graph_path.is_symlink():
        warnings.append(_graph_warning("graph_symlink", WarningSeverity.ERROR))
        return _invalid_graph("graph_symlink")

    if not graph_path.exists():
        warnings.append(_graph_warning("graph_missing", WarningSeverity.INFO))
        return StatusGraph(
            state=GraphState.MISSING,
            node_count=None,
            edge_count=None,
            reason_code="graph_not_built",
        )

    try:
        document = read_graph_document(graph_path)
    except (PmemValidationError, PmemNotFoundError, PmemPersistenceError):
        warnings.append(_graph_warning("graph_invalid", WarningSeverity.ERROR))
        return _invalid_graph("graph_unreadable")

    persisted_fingerprint = document.metadata.get("source_fingerprint")
    if not isinstance(persisted_fingerprint, str) or not persisted_fingerprint.strip():
        warnings.append(_graph_warning("graph_freshness_unknown", WarningSeverity.WARNING))
        return StatusGraph(
            state=GraphState.UNKNOWN,
            node_count=None,
            edge_count=None,
            reason_code="graph_fingerprint_missing",
        )
    if not _FINGERPRINT_RE.match(persisted_fingerprint):
        warnings.append(_graph_warning("graph_invalid", WarningSeverity.ERROR))
        return _invalid_graph("graph_fingerprint_invalid")

    node_count = _validated_count(document.counts.get("nodes"), len(document.nodes))
    edge_count = _validated_count(document.counts.get("edges"), len(document.edges))
    if node_count is None or edge_count is None:
        warnings.append(_graph_warning("graph_invalid", WarningSeverity.ERROR))
        return _invalid_graph("graph_count_mismatch")

    # A SQLite source error here is NOT a graph-artifact problem: propagate it as
    # a safe PmemError instead of hiding it behind an "unknown" graph state.
    current_fingerprint = compute_graph_source_fingerprint_readonly(
        project_database_path(root)
    ).value

    if persisted_fingerprint == current_fingerprint:
        return StatusGraph(
            state=GraphState.CURRENT,
            node_count=node_count,
            edge_count=edge_count,
            reason_code=None,
        )

    warnings.append(_graph_warning("graph_stale", WarningSeverity.WARNING))
    return StatusGraph(
        state=GraphState.STALE,
        node_count=node_count,
        edge_count=edge_count,
        reason_code="graph_source_changed",
    )


def _invalid_graph(reason_code: str) -> StatusGraph:
    return StatusGraph(
        state=GraphState.INVALID,
        node_count=None,
        edge_count=None,
        reason_code=reason_code,
    )


def _validated_count(declared: object, actual: int) -> int | None:
    """Return the count only if the declared metadata matches the real length."""

    if isinstance(declared, bool) or not isinstance(declared, int):
        return None
    if declared < 0 or declared != actual:
        return None
    return actual


def _graph_warning(code: str, severity: WarningSeverity) -> StatusWarning:
    return _warning(
        code,
        severity,
        WarningSource.GRAPH,
        "The evidence graph is not current.",
        "Rebuild the evidence graph with pmem graph build.",
    )


# --------------------------------------------------------------------------- #
# Recommendation availability (opt-in generation)                             #
# --------------------------------------------------------------------------- #
def _map_recommendations(
    root: Path,
    *,
    evaluate: bool,
    max_recommendations: int,
    warnings: list[StatusWarning],
) -> StatusRecommendations:
    if not evaluate:
        return StatusRecommendations(
            mode=RecommendationMode.NOT_EVALUATED,
            candidate_count=None,
            active_count=None,
        )

    payload = recommendation_list_payload(root, max_recommendations=max_recommendations)
    candidate_count = _validated_recommendation_count(payload, max_recommendations)
    raw_warnings = payload.get("warnings") if isinstance(payload, dict) else None
    for warning in _recommendation_warnings(raw_warnings):
        warnings.append(warning)
    return StatusRecommendations(
        mode=RecommendationMode.GENERATED_ON_DEMAND,
        candidate_count=candidate_count,
        active_count=None,
    )


def _validated_recommendation_count(payload: object, max_recommendations: int) -> int:
    if not isinstance(payload, dict) or "recommendation_count" not in payload:
        raise PmemValidationError("The recommendation payload is malformed.")
    value = payload["recommendation_count"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PmemValidationError("The recommendation count must be an integer.")
    if value < 0 or value > max_recommendations:
        raise PmemValidationError("The recommendation count is out of range.")
    return value


def _recommendation_warnings(raw_warnings: object) -> list[StatusWarning]:
    if not isinstance(raw_warnings, list):
        return []
    unique: list[str] = []
    for item in raw_warnings:
        text = item.lower() if isinstance(item, str) else ""
        code = _classify_recommendation_warning(text)
        if code not in unique:
            unique.append(code)
    return [
        _warning(
            code,
            WarningSeverity.WARNING,
            WarningSource.RECOMMENDATION,
            "A recommendation data-quality signal was raised.",
            None,
        )
        for code in unique
    ]


def _classify_recommendation_warning(text: str) -> str:
    if "dataset" in text:
        return "dataset_metadata_placement"
    if "mislabel" in text:
        return "possible_failure_mislabel"
    if "stale" in text and "metric" in text:
        return "stale_failed_run_metrics"
    if "opposing" in text or "supporting" in text or "related failure" in text:
        return "recommendation_evidence_incomplete"
    if "evidence" in text:
        return "insufficient_recommendation_evidence"
    return "recommendation_warning"


# --------------------------------------------------------------------------- #
# Warning helpers                                                             #
# --------------------------------------------------------------------------- #
def _warning(
    code: str,
    severity: WarningSeverity,
    source: WarningSource,
    message: str,
    remediation: str | None,
) -> StatusWarning:
    return StatusWarning(
        code=code,
        severity=severity,
        message=message,
        source=source,
        remediation=remediation,
    )


def _ordered_warnings(warnings: list[StatusWarning]) -> tuple[StatusWarning, ...]:
    """Deduplicate by (source, code) then sort deterministically."""

    seen: set[tuple[str, str]] = set()
    unique: list[StatusWarning] = []
    for warning in warnings:
        key = (warning.source.value, warning.code)
        if key in seen:
            continue
        seen.add(key)
        unique.append(warning)
    unique.sort(key=lambda w: (_SEVERITY_RANK[w.severity], w.source.value, w.code))
    return tuple(unique)


# --------------------------------------------------------------------------- #
# Recommendation limit + privacy-safe text handling                           #
# --------------------------------------------------------------------------- #
def _validate_recommendation_limit(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PmemValidationError("The recommendation limit must be an integer.")
    if value < 1 or value > _MAX_RECOMMENDATIONS:
        raise PmemValidationError("The recommendation limit must be between 1 and 50.")


def _text_is_unsafe(value: str, *, max_length: int) -> bool:
    stripped = value.strip()
    if len(stripped) == 0 or len(stripped) > max_length:
        return True
    if contains_control_chars(value):
        return True
    return contains_absolute_path(value)


def _safe_text_or_redact(
    value: str,
    *,
    kind: str,
    max_length: int,
    warnings: list[StatusWarning],
) -> str:
    """Return the text unchanged, or a deterministic SHA-256 label if unsafe.

    Never echoes the raw text or any path. Uses SHA-256 (not ``hash()``) so the
    redaction is stable across processes.
    """

    if not _text_is_unsafe(value, max_length=max_length):
        return value
    digest = compute_text_hash(value)[:16]
    warnings.append(
        _warning(
            "status_text_redacted",
            WarningSeverity.WARNING,
            WarningSource.DATA_QUALITY,
            "Some project text was redacted from the status payload.",
            None,
        )
    )
    return f"redacted_{kind}_{digest}"
