"""status next-action deterministic policy (STS-003).

A **pure**, deterministic, total rule engine that selects exactly one
:class:`~pmem.status.StatusNextAction` from a collected status state. It never
reads or writes the filesystem, opens SQLite, calls the network, generates
recommendations, mutates its input, or invents ids/paths/evidence. It only reads
the structured state that STS-002 already collected.

Warning policy (explicit, to avoid over-claiming). The engine primarily decides
from structured state fields: counts, ``metric.target_status``,
``graph.state``/``graph.reason_code``, recommendation counts, and the
baseline/best-run ids. Warnings remain in the payload for rendering, but two
classes affect priority: an otherwise-unmapped ``error`` warning prevents a
healthy result, and recommendation warnings about stale failed-run metrics or
possible failure mislabels must be reviewed before recommendation candidates
are trusted. These two warnings have separate actions because stale metrics can
exist without a confirmed failure record. Redaction, scope, and
incomplete-evidence warnings remain visible but do not override a more specific
core-loop action.

Priority model (highest first):

    1  no runs captured                 -> capture the first run
    2  graph invalid via symlink        -> resolve the symlink (not rebuildable)
    3  graph invalid (rebuildable)      -> rebuild the graph
    4  unmapped error warning           -> inspect status safely
    5  no successful run, failures logged -> review failures
    6  explicit failed run, none logged -> record a failure
    7  no success and no failed status  -> inspect non-successful runs
    8  graph missing                    -> build the evidence graph
    9  graph stale                      -> rebuild the evidence graph
    10 graph freshness unknown          -> rebuild to a known-good graph
    11 possible failure mislabel        -> review confirmed failures
    12 stale metrics on failed runs     -> inspect run summary
    13 active persisted recs            -> review active recommendations
    14 on-demand candidates (>0)        -> review recommendations
    15 best run but no baseline         -> mark the baseline (real run id)
    16 metric/target unconfigured       -> discover how to configure them
    17 metric configured, unseen        -> capture a run that reports the metric
    18 recs evaluated, zero candidates  -> capture more evidence (no recs loop)
    19 target not met (recs unevaluated)-> review recommendations to improve
    20 no tracked files                 -> discover how to track a file
    21 healthy (fallback)               -> explore recommendations

Rules 11-21 are only reached once the graph is ``current`` (rules 2/3/8/9/10 have
already claimed every other graph state), so recommendation/baseline/target
guidance is never suggested on top of a stale or broken graph.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pmem.status.model import (
    GraphState,
    RecommendationMode,
    StatusNextAction,
    TargetStatus,
    WarningSeverity,
    WarningSource,
)

if TYPE_CHECKING:  # avoid a runtime import cycle (services depends on this module)
    from pmem.services.status_service import CollectedStatusState

# Every command is a real projmem command surface. Commands that need input the
# status state cannot provide (a file path, a run command) intentionally use the
# safe ``--help`` discovery form instead of a fabricated concrete command.
_CMD_RUN_HELP = "pmem run --help"
_CMD_TRACK_HELP = "pmem track --help"
_CMD_INIT_HELP = "pmem init --help"
_CMD_GRAPH_BUILD = "pmem graph build"
_CMD_GRAPH_HELP = "pmem graph --help"
_CMD_RECOMMEND_LIST = "pmem recommend list"
_CMD_FAILURES_LIST = "pmem failures list"
_CMD_LOG_FAILURE_HELP = "pmem log-failure --help"
_CMD_SUMMARY = "pmem summary"

_GRAPH_SYMLINK_REASON = "graph_symlink"
_POSSIBLE_FAILURE_MISLABEL = "possible_failure_mislabel"
_STALE_FAILED_RUN_METRICS = "stale_failed_run_metrics"


def select_next_action(state: CollectedStatusState) -> StatusNextAction:
    """Return exactly one next action for a valid ``status-v1`` state.

    Pure and total: the final rule matches unconditionally, so every valid state
    yields an action, and identical input always yields identical output.
    """

    for rule in _RULES:
        action = rule(state)
        if action is not None:
            return action
    return _healthy_fallback(state)


# --------------------------------------------------------------------------- #
# Individual rules (each returns an action or None)                            #
# --------------------------------------------------------------------------- #
def _rule_capture_first_run(state: CollectedStatusState) -> StatusNextAction | None:
    if state.counts.run_count != 0:
        return None
    return _action(
        "capture_first_run",
        "No runs have been captured yet; capture the first run to start building project memory.",
        _CMD_RUN_HELP,
    )


def _rule_resolve_graph_symlink(state: CollectedStatusState) -> StatusNextAction | None:
    if state.graph.state is not GraphState.INVALID:
        return None
    if state.graph.reason_code != _GRAPH_SYMLINK_REASON:
        return None
    return _action(
        "resolve_graph_symlink",
        "The evidence graph path is a symlink, which projmem refuses to read or overwrite; "
        "remove the symlink manually before rebuilding the graph.",
        _CMD_GRAPH_HELP,
    )


def _rule_rebuild_invalid_graph(state: CollectedStatusState) -> StatusNextAction | None:
    if state.graph.state is not GraphState.INVALID:
        return None
    return _action(
        "rebuild_invalid_graph",
        "The evidence graph artifact is invalid; rebuild it before trusting lineage or "
        "recommendations.",
        _CMD_GRAPH_BUILD,
    )


def _rule_review_unmapped_error(state: CollectedStatusState) -> StatusNextAction | None:
    if not any(warning.severity is WarningSeverity.ERROR for warning in state.warnings):
        return None
    return _action(
        "review_status_error",
        "A status integrity error has no more specific recovery rule; inspect the project "
        "summary before continuing.",
        _CMD_SUMMARY,
    )


def _rule_review_failed_runs(state: CollectedStatusState) -> StatusNextAction | None:
    if state.counts.successful_run_count != 0 or state.counts.failure_count <= 0:
        return None
    return _action(
        "review_failed_runs",
        "Runs were captured but none succeeded; review the recorded failures before relying on "
        "metrics.",
        _CMD_FAILURES_LIST,
    )


def _rule_record_run_failure(state: CollectedStatusState) -> StatusNextAction | None:
    if state.counts.successful_run_count != 0:
        return None
    if state.counts.failed_run_count <= 0 or state.counts.failure_count != 0:
        return None
    return _action(
        "record_run_failure",
        "Runs were captured with no success and no failure recorded; record a failure to explain "
        "what went wrong.",
        _CMD_LOG_FAILURE_HELP,
    )


def _rule_inspect_non_successful_runs(
    state: CollectedStatusState,
) -> StatusNextAction | None:
    if state.counts.successful_run_count != 0:
        return None
    if state.counts.failed_run_count != 0 or state.counts.failure_count != 0:
        return None
    return _action(
        "inspect_non_successful_runs",
        "Runs exist but none is successful or explicitly failed; inspect their statuses before "
        "deciding whether to retry or record a failure.",
        _CMD_SUMMARY,
    )


def _rule_build_missing_graph(state: CollectedStatusState) -> StatusNextAction | None:
    if state.graph.state is not GraphState.MISSING:
        return None
    return _action(
        "build_evidence_graph",
        "Successful runs exist but no evidence graph has been built; build it to enable lineage "
        "and recommendations.",
        _CMD_GRAPH_BUILD,
    )


def _rule_rebuild_stale_graph(state: CollectedStatusState) -> StatusNextAction | None:
    if state.graph.state is not GraphState.STALE:
        return None
    return _action(
        "rebuild_stale_graph",
        "The evidence graph is stale after new project activity; rebuild it before using its "
        "lineage.",
        _CMD_GRAPH_BUILD,
    )


def _rule_rebuild_unknown_graph(state: CollectedStatusState) -> StatusNextAction | None:
    if state.graph.state is not GraphState.UNKNOWN:
        return None
    return _action(
        "rebuild_unverifiable_graph",
        "The evidence graph freshness cannot be verified; rebuild it to a known-good state.",
        _CMD_GRAPH_BUILD,
    )


def _has_recommendation_warning(state: CollectedStatusState, code: str) -> bool:
    return any(
        warning.source is WarningSource.RECOMMENDATION and warning.code == code
        for warning in state.warnings
    )


def _rule_review_failure_mislabels(
    state: CollectedStatusState,
) -> StatusNextAction | None:
    if not _has_recommendation_warning(state, _POSSIBLE_FAILURE_MISLABEL):
        return None
    return _action(
        "review_failure_mislabels",
        "Recommendation evidence indicates possible failure mislabels; review confirmed "
        "failures before trusting candidates.",
        _CMD_FAILURES_LIST,
    )


def _rule_review_stale_failed_metrics(
    state: CollectedStatusState,
) -> StatusNextAction | None:
    if not _has_recommendation_warning(state, _STALE_FAILED_RUN_METRICS):
        return None
    return _action(
        "review_stale_failed_metrics",
        "Failed or non-successful runs contain primary metric values that may be stale; inspect "
        "the run summary before trusting candidates.",
        _CMD_SUMMARY,
    )


def _rule_review_active_recommendations(state: CollectedStatusState) -> StatusNextAction | None:
    recommendations = state.recommendations
    if recommendations.mode is not RecommendationMode.PERSISTED_LIFECYCLE:
        return None
    if recommendations.active_count is None or recommendations.active_count <= 0:
        return None
    return _action(
        "review_active_recommendations",
        "There are active recommendations to act on; review them before the next experiment.",
        _CMD_RECOMMEND_LIST,
    )


def _rule_review_candidates(state: CollectedStatusState) -> StatusNextAction | None:
    candidate_count = state.recommendations.candidate_count
    if candidate_count is None or candidate_count <= 0:
        return None
    return _action(
        "review_recommendations",
        "Evidence-backed recommendation candidates are available; review them for the next step.",
        _CMD_RECOMMEND_LIST,
    )


def _rule_set_baseline(state: CollectedStatusState) -> StatusNextAction | None:
    best_run_id = state.best_run.run_id
    if state.baseline.run_id is not None or best_run_id is None:
        return None
    return _action(
        "set_baseline",
        "A best run exists but no baseline is set; mark it as the baseline to enable "
        "comparison and regression checks.",
        f"pmem baseline {best_run_id}",
        related_entity_id=best_run_id,
    )


def _rule_capture_more_evidence(state: CollectedStatusState) -> StatusNextAction | None:
    # ``candidate_count == 0`` only occurs once recommendations have been
    # evaluated; not-evaluated states carry ``None``. Never re-suggest
    # ``recommend list`` here — that would be a no-op loop.
    if state.recommendations.candidate_count != 0:
        return None
    return _action(
        "capture_more_evidence",
        "Recommendations were evaluated but produced no candidates; capture more runs to give the "
        "engine more evidence to work with.",
        _CMD_RUN_HELP,
    )


def _rule_configure_metric(state: CollectedStatusState) -> StatusNextAction | None:
    if state.metric.target_status is not TargetStatus.NOT_CONFIGURED:
        return None
    return _action(
        "configure_project_metric",
        "The project has successful runs but no primary metric or target configured; review how "
        "they are set at initialization.",
        _CMD_INIT_HELP,
    )


def _rule_capture_primary_metric(state: CollectedStatusState) -> StatusNextAction | None:
    if state.metric.target_status is not TargetStatus.NO_METRIC:
        return None
    return _action(
        "capture_primary_metric",
        "No successful run reported the configured primary metric; capture a run that emits it.",
        _CMD_RUN_HELP,
    )


def _rule_improve_toward_target(state: CollectedStatusState) -> StatusNextAction | None:
    if state.metric.target_status is not TargetStatus.NOT_MET:
        return None
    return _action(
        "improve_toward_target",
        "The project target has not been met; review evidence-backed recommendations to decide "
        "what to try next.",
        _CMD_RECOMMEND_LIST,
    )


def _rule_track_project_file(state: CollectedStatusState) -> StatusNextAction | None:
    if state.counts.tracked_path_count != 0:
        return None
    return _action(
        "track_project_file",
        "No project files are tracked; track a file so runs can be tied to the code that produced "
        "them.",
        _CMD_TRACK_HELP,
    )


def _healthy_fallback(state: CollectedStatusState) -> StatusNextAction:
    return _action(
        "explore_recommendations",
        "The project is healthy with no blocking gaps; explore recommendations for the next "
        "experiment.",
        _CMD_RECOMMEND_LIST,
    )


_RULES: tuple[Callable[[CollectedStatusState], StatusNextAction | None], ...] = (
    _rule_capture_first_run,
    _rule_resolve_graph_symlink,
    _rule_rebuild_invalid_graph,
    _rule_review_unmapped_error,
    _rule_review_failed_runs,
    _rule_record_run_failure,
    _rule_inspect_non_successful_runs,
    _rule_build_missing_graph,
    _rule_rebuild_stale_graph,
    _rule_rebuild_unknown_graph,
    _rule_review_failure_mislabels,
    _rule_review_stale_failed_metrics,
    _rule_review_active_recommendations,
    _rule_review_candidates,
    _rule_set_baseline,
    _rule_configure_metric,
    _rule_capture_primary_metric,
    _rule_capture_more_evidence,
    _rule_improve_toward_target,
    _rule_track_project_file,
)


def _action(
    action_id: str,
    reason: str,
    suggested_command: str,
    *,
    related_entity_id: str | None = None,
) -> StatusNextAction:
    return StatusNextAction(
        action_id=action_id,
        reason=reason,
        suggested_command=suggested_command,
        related_entity_id=related_entity_id,
    )
