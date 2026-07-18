"""Unit tests for the deterministic next-action policy (STS-003)."""

from __future__ import annotations

import copy
import shlex
from typing import Any

import pytest
from typer.testing import CliRunner

from pmem.cli.app import app as cli_app
from pmem.services.status_service import (
    CollectedStatusState,
    assemble_status_payload,
    build_status_payload,
)
from pmem.status import (
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
    WarningSeverity,
    WarningSource,
    select_next_action,
)

_PROJECT_ID = "proj_" + "0" * 32
_BEST_RUN_ID = "run_" + "a" * 32
_BASELINE_ID = "run_" + "b" * 32
_CLI_RUNNER = CliRunner()


def _state(**overrides: Any) -> CollectedStatusState:
    """Build a healthy baseline state, overridable per field group."""

    counts = {
        "run_count": 5,
        "successful_run_count": 4,
        "failed_run_count": 1,
        "tracked_path_count": 2,
        "failure_count": 0,
        "decision_count": 0,
        "note_count": 0,
    }
    metric = {
        "primary_metric": "accuracy",
        "direction": "max",
        "target_value": 0.9,
        "best_value": 0.95,
        "target_status": "met",
    }
    best_run = {"run_id": _BEST_RUN_ID, "metric_value": 0.95}
    baseline = {"run_id": _BASELINE_ID}
    graph = {"state": "current", "node_count": 3, "edge_count": 1, "reason_code": None}
    recommendations = {"mode": "not_evaluated", "candidate_count": None, "active_count": None}
    project = {"project_id": _PROJECT_ID, "project_name": "demo", "objective": "Train a baseline"}
    warnings = overrides.get("warnings", ())

    counts.update(overrides.get("counts", {}))
    metric.update(overrides.get("metric", {}))
    best_run.update(overrides.get("best_run", {}))
    baseline.update(overrides.get("baseline", {}))
    graph.update(overrides.get("graph", {}))
    recommendations.update(overrides.get("recommendations", {}))
    project.update(overrides.get("project", {}))

    return CollectedStatusState(
        project=StatusProject(**project),
        metric=StatusMetric(**metric),
        counts=StatusCounts(**counts),
        best_run=StatusBestRun(**best_run),
        baseline=StatusBaseline(**baseline),
        graph=StatusGraph(**graph),
        recommendations=StatusRecommendations(**recommendations),
        warnings=tuple(warnings),
    )


_UNCONFIGURED_METRIC = {
    "primary_metric": None,
    "direction": None,
    "target_value": None,
    "best_value": None,
}
_EMPTY_BEST = {"run_id": None, "metric_value": None}
_INVALID = {"node_count": None, "edge_count": None}


def _warning(code: str, severity: str, source: str) -> StatusWarning:
    return StatusWarning(
        code=code,
        severity=WarningSeverity(severity),
        message="A generic, privacy-safe status message.",
        source=WarningSource(source),
        remediation=None,
    )


# --------------------------------------------------------------------------- #
# One positive test per rule: exact action_id, reason, command, entity          #
# --------------------------------------------------------------------------- #
_RULE_CASES: dict[str, tuple[dict[str, Any], str, str, str, str | None]] = {
    "capture_first_run": (
        {
            "counts": {"run_count": 0, "successful_run_count": 0, "failed_run_count": 0},
            "metric": {**_UNCONFIGURED_METRIC, "target_status": "no_runs"},
            "best_run": _EMPTY_BEST,
            "baseline": {"run_id": None},
            "graph": {"state": "missing", **_INVALID, "reason_code": "graph_not_built"},
        },
        "capture_first_run",
        "No runs have been captured yet; capture the first run to start building project memory.",
        "pmem run --help",
        None,
    ),
    "resolve_graph_symlink": (
        {"graph": {"state": "invalid", **_INVALID, "reason_code": "graph_symlink"}},
        "resolve_graph_symlink",
        "The evidence graph path is a symlink, which projmem refuses to read or overwrite; "
        "remove the symlink manually before rebuilding the graph.",
        "pmem graph --help",
        None,
    ),
    "rebuild_invalid_graph": (
        {"graph": {"state": "invalid", **_INVALID, "reason_code": "graph_unreadable"}},
        "rebuild_invalid_graph",
        "The evidence graph artifact is invalid; rebuild it before trusting lineage or "
        "recommendations.",
        "pmem graph build",
        None,
    ),
    "review_status_error": (
        {"warnings": (_warning("future_integrity_error", "error", "data_quality"),)},
        "review_status_error",
        "A status integrity error has no more specific recovery rule; inspect the project "
        "summary before continuing.",
        "pmem summary",
        None,
    ),
    "review_failed_runs": (
        {
            "counts": {
                "run_count": 4,
                "successful_run_count": 0,
                "failed_run_count": 4,
                "failure_count": 2,
            },
            "metric": {"best_value": None, "target_status": "no_successful_runs"},
            "best_run": _EMPTY_BEST,
            "baseline": {"run_id": None},
        },
        "review_failed_runs",
        "Runs were captured but none succeeded; review the recorded failures before relying on "
        "metrics.",
        "pmem failures list",
        None,
    ),
    "record_run_failure": (
        {
            "counts": {
                "run_count": 4,
                "successful_run_count": 0,
                "failed_run_count": 4,
                "failure_count": 0,
            },
            "metric": {"best_value": None, "target_status": "no_successful_runs"},
            "best_run": _EMPTY_BEST,
            "baseline": {"run_id": None},
        },
        "record_run_failure",
        "Runs were captured with no success and no failure recorded; record a failure to explain "
        "what went wrong.",
        "pmem log-failure --help",
        None,
    ),
    "inspect_non_successful_runs": (
        {
            "counts": {
                "run_count": 4,
                "successful_run_count": 0,
                "failed_run_count": 0,
                "failure_count": 0,
            },
            "metric": {"best_value": None, "target_status": "no_successful_runs"},
            "best_run": _EMPTY_BEST,
            "baseline": {"run_id": None},
        },
        "inspect_non_successful_runs",
        "Runs exist but none is successful or explicitly failed; inspect their statuses before "
        "deciding whether to retry or record a failure.",
        "pmem summary",
        None,
    ),
    "build_evidence_graph": (
        {"graph": {"state": "missing", **_INVALID, "reason_code": "graph_not_built"}},
        "build_evidence_graph",
        "Successful runs exist but no evidence graph has been built; build it to enable lineage "
        "and recommendations.",
        "pmem graph build",
        None,
    ),
    "rebuild_stale_graph": (
        {"graph": {"state": "stale", "reason_code": "graph_source_changed"}},
        "rebuild_stale_graph",
        "The evidence graph is stale after new project activity; rebuild it before using its "
        "lineage.",
        "pmem graph build",
        None,
    ),
    "rebuild_unverifiable_graph": (
        {"graph": {"state": "unknown", **_INVALID, "reason_code": "graph_fingerprint_missing"}},
        "rebuild_unverifiable_graph",
        "The evidence graph freshness cannot be verified; rebuild it to a known-good state.",
        "pmem graph build",
        None,
    ),
    "review_failure_mislabels": (
        {"warnings": (_warning("possible_failure_mislabel", "warning", "recommendation"),)},
        "review_failure_mislabels",
        "Recommendation evidence indicates possible failure mislabels; review confirmed "
        "failures before trusting candidates.",
        "pmem failures list",
        None,
    ),
    "review_stale_failed_metrics": (
        {"warnings": (_warning("stale_failed_run_metrics", "warning", "recommendation"),)},
        "review_stale_failed_metrics",
        "Failed or non-successful runs contain primary metric values that may be stale; inspect "
        "the run summary before trusting candidates.",
        "pmem summary",
        None,
    ),
    "review_active_recommendations": (
        {
            "recommendations": {
                "mode": "persisted_lifecycle",
                "candidate_count": 5,
                "active_count": 2,
            }
        },
        "review_active_recommendations",
        "There are active recommendations to act on; review them before the next experiment.",
        "pmem recommend list",
        None,
    ),
    "review_recommendations": (
        {
            "recommendations": {
                "mode": "generated_on_demand",
                "candidate_count": 4,
                "active_count": None,
            }
        },
        "review_recommendations",
        "Evidence-backed recommendation candidates are available; review them for the next step.",
        "pmem recommend list",
        None,
    ),
    "set_baseline": (
        {"baseline": {"run_id": None}},
        "set_baseline",
        "A best run exists but no baseline is set; mark it as the baseline to enable "
        "comparison and regression checks.",
        f"pmem baseline {_BEST_RUN_ID}",
        _BEST_RUN_ID,
    ),
    "capture_more_evidence": (
        {
            "recommendations": {
                "mode": "generated_on_demand",
                "candidate_count": 0,
                "active_count": None,
            }
        },
        "capture_more_evidence",
        "Recommendations were evaluated but produced no candidates; capture more runs to give the "
        "engine more evidence to work with.",
        "pmem run --help",
        None,
    ),
    "configure_project_metric": (
        {
            "metric": {**_UNCONFIGURED_METRIC, "target_status": "not_configured"},
            "best_run": _EMPTY_BEST,
        },
        "configure_project_metric",
        "The project has successful runs but no primary metric or target configured; review how "
        "they are set at initialization.",
        "pmem init --help",
        None,
    ),
    "capture_primary_metric": (
        {
            "metric": {"best_value": None, "target_status": "no_metric"},
            "best_run": _EMPTY_BEST,
        },
        "capture_primary_metric",
        "No successful run reported the configured primary metric; capture a run that emits it.",
        "pmem run --help",
        None,
    ),
    "improve_toward_target": (
        {
            "metric": {"best_value": 0.5, "target_status": "not_met"},
            "best_run": {"metric_value": 0.5},
        },
        "improve_toward_target",
        "The project target has not been met; review evidence-backed recommendations to decide "
        "what to try next.",
        "pmem recommend list",
        None,
    ),
    "track_project_file": (
        {"counts": {"tracked_path_count": 0}},
        "track_project_file",
        "No project files are tracked; track a file so runs can be tied to the code that produced "
        "them.",
        "pmem track --help",
        None,
    ),
    "explore_recommendations": (
        {},
        "explore_recommendations",
        "The project is healthy with no blocking gaps; explore recommendations for the next "
        "experiment.",
        "pmem recommend list",
        None,
    ),
}


@pytest.mark.parametrize("case", sorted(_RULE_CASES))
def test_rule_produces_exact_action(case: str) -> None:
    overrides, action_id, reason, command, entity = _RULE_CASES[case]
    action = select_next_action(_state(**overrides))
    assert action.action_id == action_id
    assert action.reason == reason
    assert action.suggested_command == command
    assert action.related_entity_id == entity


def test_reasons_and_action_ids_are_unique() -> None:
    action_ids = [aid for _o, aid, *_ in _RULE_CASES.values()]
    reasons = [reason for _o, _aid, reason, *_ in _RULE_CASES.values()]
    assert len(set(action_ids)) == len(action_ids)
    assert len(set(reasons)) == len(reasons)


# --------------------------------------------------------------------------- #
# Collision / precedence tests                                                 #
# --------------------------------------------------------------------------- #
_COLLISIONS: dict[str, tuple[dict[str, Any], str]] = {
    "no_runs_over_graph_missing": (
        {
            "counts": {
                "run_count": 0,
                "successful_run_count": 0,
                "failed_run_count": 0,
                "tracked_path_count": 0,
            },
            "metric": {**_UNCONFIGURED_METRIC, "target_status": "no_runs"},
            "best_run": _EMPTY_BEST,
            "baseline": {"run_id": None},
            "graph": {"state": "missing", **_INVALID, "reason_code": "graph_not_built"},
        },
        "capture_first_run",
    ),
    "symlink_over_generic_invalid_command": (
        {"graph": {"state": "invalid", **_INVALID, "reason_code": "graph_symlink"}},
        "resolve_graph_symlink",
    ),
    "no_successful_over_graph_stale": (
        {
            "counts": {
                "run_count": 4,
                "successful_run_count": 0,
                "failed_run_count": 4,
                "failure_count": 3,
            },
            "metric": {"best_value": None, "target_status": "no_successful_runs"},
            "best_run": _EMPTY_BEST,
            "baseline": {"run_id": None},
            "graph": {"state": "stale", "reason_code": "graph_source_changed"},
        },
        "review_failed_runs",
    ),
    "graph_invalid_over_candidates": (
        {
            "graph": {"state": "invalid", **_INVALID, "reason_code": "graph_unreadable"},
            "recommendations": {
                "mode": "generated_on_demand",
                "candidate_count": 9,
                "active_count": None,
            },
        },
        "rebuild_invalid_graph",
    ),
    "graph_missing_over_not_configured": (
        {
            "metric": {**_UNCONFIGURED_METRIC, "target_status": "not_configured"},
            "best_run": _EMPTY_BEST,
            "graph": {"state": "missing", **_INVALID, "reason_code": "graph_not_built"},
        },
        "build_evidence_graph",
    ),
    "graph_stale_over_target_met": (
        {"graph": {"state": "stale", "reason_code": "graph_source_changed"}},
        "rebuild_stale_graph",
    ),
    "baseline_over_target_not_met": (
        {
            "metric": {"best_value": 0.5, "target_status": "not_met"},
            "best_run": {"metric_value": 0.5},
            "baseline": {"run_id": None},
        },
        "set_baseline",
    ),
    "zero_candidates_over_target_not_met": (
        {
            "metric": {"best_value": 0.5, "target_status": "not_met"},
            "best_run": {"metric_value": 0.5},
            "recommendations": {
                "mode": "generated_on_demand",
                "candidate_count": 0,
                "active_count": None,
            },
        },
        "capture_more_evidence",
    ),
    "metric_configuration_over_zero_candidates": (
        {
            "metric": {**_UNCONFIGURED_METRIC, "target_status": "not_configured"},
            "best_run": _EMPTY_BEST,
            "recommendations": {
                "mode": "generated_on_demand",
                "candidate_count": 0,
                "active_count": None,
            },
        },
        "configure_project_metric",
    ),
    "missing_primary_metric_over_zero_candidates": (
        {
            "metric": {"best_value": None, "target_status": "no_metric"},
            "best_run": _EMPTY_BEST,
            "recommendations": {
                "mode": "generated_on_demand",
                "candidate_count": 0,
                "active_count": None,
            },
        },
        "capture_primary_metric",
    ),
    "zero_candidates_persisted_no_active": (
        {
            "recommendations": {
                "mode": "persisted_lifecycle",
                "candidate_count": 0,
                "active_count": 0,
            }
        },
        "capture_more_evidence",
    ),
    "active_recs_over_candidates": (
        {
            "recommendations": {
                "mode": "persisted_lifecycle",
                "candidate_count": 9,
                "active_count": 3,
            }
        },
        "review_active_recommendations",
    ),
    "error_condition_over_healthy": (
        {"graph": {"state": "invalid", **_INVALID, "reason_code": "graph_unreadable"}},
        "rebuild_invalid_graph",
    ),
    "failure_mislabel_warning_over_candidate": (
        {
            "recommendations": {
                "mode": "generated_on_demand",
                "candidate_count": 3,
                "active_count": None,
            },
            "warnings": (_warning("possible_failure_mislabel", "warning", "recommendation"),),
        },
        "review_failure_mislabels",
    ),
    "stale_metric_warning_over_candidate": (
        {
            "recommendations": {
                "mode": "generated_on_demand",
                "candidate_count": 3,
                "active_count": None,
            },
            "warnings": (_warning("stale_failed_run_metrics", "warning", "recommendation"),),
        },
        "review_stale_failed_metrics",
    ),
}


@pytest.mark.parametrize("case", sorted(_COLLISIONS))
def test_precedence_resolves_collision(case: str) -> None:
    overrides, expected_action_id = _COLLISIONS[case]
    assert select_next_action(_state(**overrides)).action_id == expected_action_id


# --------------------------------------------------------------------------- #
# Warning handling                                                            #
# --------------------------------------------------------------------------- #
def test_informational_redaction_warning_does_not_override_healthy() -> None:
    state = _state(
        warnings=(_warning("status_text_redacted", "warning", "data_quality"),),
    )
    assert select_next_action(state).action_id == "explore_recommendations"


@pytest.mark.parametrize(
    ("code", "severity", "source"),
    [
        ("no_baseline", "info", "summary"),
        ("status_text_redacted", "warning", "data_quality"),
        ("dataset_metadata_placement", "warning", "recommendation"),
        ("recommendation_evidence_incomplete", "warning", "recommendation"),
    ],
)
def test_non_structural_warnings_are_non_blocking(code: str, severity: str, source: str) -> None:
    # A fully healthy structured state stays on the healthy action regardless of
    # an attached info/warning-severity, non-structural warning.
    state = _state(warnings=(_warning(code, severity, source),))
    assert select_next_action(state).action_id == "explore_recommendations"


def test_structural_error_is_handled_even_with_extra_warnings() -> None:
    # The only error-severity condition (invalid graph) is a structured field and
    # is resolved by the graph rule even when other warnings are attached.
    state = _state(
        graph={"state": "invalid", **_INVALID, "reason_code": "graph_unreadable"},
        warnings=(
            _warning("graph_invalid", "error", "graph"),
            _warning("status_text_redacted", "warning", "data_quality"),
        ),
    )
    assert select_next_action(state).action_id == "rebuild_invalid_graph"


def test_unmapped_error_warning_never_reports_healthy() -> None:
    state = _state(
        warnings=(_warning("future_integrity_error", "error", "data_quality"),),
    )
    assert select_next_action(state).action_id == "review_status_error"


@pytest.mark.parametrize(
    ("code", "expected_action"),
    [
        ("possible_failure_mislabel", "review_failure_mislabels"),
        ("stale_failed_run_metrics", "review_stale_failed_metrics"),
    ],
)
def test_recommendation_trust_warning_blocks_candidate_review(
    code: str,
    expected_action: str,
) -> None:
    state = _state(
        recommendations={
            "mode": "generated_on_demand",
            "candidate_count": 3,
            "active_count": None,
        },
        warnings=(_warning(code, "warning", "recommendation"),),
    )
    assert select_next_action(state).action_id == expected_action


# --------------------------------------------------------------------------- #
# Purity, determinism, totality, validity                                      #
# --------------------------------------------------------------------------- #
def test_repeated_calls_are_deterministic() -> None:
    for overrides, *_ in _RULE_CASES.values():
        state = _state(**overrides)
        assert select_next_action(state).model_dump() == select_next_action(state).model_dump()


def test_select_does_not_mutate_input() -> None:
    for overrides in (co for co, *_ in _RULE_CASES.values()):
        state = _state(**overrides)
        snapshot = copy.deepcopy(state)
        select_next_action(state)
        assert state == snapshot


def test_action_is_a_valid_status_next_action() -> None:
    for overrides, *_ in _RULE_CASES.values():
        action = select_next_action(_state(**overrides))
        assert isinstance(action, StatusNextAction)
        assert StatusNextAction.model_validate(action.model_dump()) == action


def test_related_entity_id_is_never_fabricated() -> None:
    for overrides, *_ in _RULE_CASES.values():
        state = _state(**overrides)
        action = select_next_action(state)
        if action.related_entity_id is not None:
            assert action.related_entity_id == state.best_run.run_id
            assert action.suggested_command == f"pmem baseline {state.best_run.run_id}"


def test_no_project_text_leaks_into_action() -> None:
    sentinel_name = "sentinel-project-name"
    sentinel_objective = "sentinel-objective-text"
    state = _state(project={"project_name": sentinel_name, "objective": sentinel_objective})
    action = select_next_action(state)
    blob = " ".join(
        str(x)
        for x in (
            action.action_id,
            action.reason,
            action.suggested_command,
            action.related_entity_id,
        )
    )
    assert sentinel_name not in blob
    assert sentinel_objective not in blob
    assert state.project.project_id not in blob


# --------------------------------------------------------------------------- #
# Every command parses against the real Typer CLI                              #
# --------------------------------------------------------------------------- #
def test_every_suggested_command_exists_in_the_cli() -> None:
    shell_metacharacters = ("&&", "||", ";", "|", "`", "$(", "${", "&", ">", "<", "\n", "\r")
    for overrides, *_ in _RULE_CASES.values():
        command = select_next_action(_state(**overrides)).suggested_command
        tokens = shlex.split(command)
        assert tokens[0] == "pmem"
        assert not any(token in command for token in shell_metacharacters)
        with _CLI_RUNNER.isolated_filesystem():
            result = _CLI_RUNNER.invoke(cli_app, tokens[1:])
        # Help actions succeed. Operational actions parse successfully and then
        # fail with code 1 because the isolated directory is not initialized.
        expected_exit = 0 if "--help" in tokens else 1
        assert result.exit_code == expected_exit, (command, result.output, result.exception)


@pytest.mark.parametrize(
    "tokens",
    [
        ["baseline", _BEST_RUN_ID, "unexpected"],
        ["graph", "build", "unexpected"],
        ["recommend", "missing-subcommand"],
        ["failures", "list", "--missing-option"],
    ],
)
def test_cli_parser_rejects_invalid_command_tails(tokens: list[str]) -> None:
    with _CLI_RUNNER.isolated_filesystem():
        result = _CLI_RUNNER.invoke(cli_app, tokens)
    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# Assembly integration                                                         #
# --------------------------------------------------------------------------- #
def test_build_status_payload_matches_manual_assembly() -> None:
    state = _state(baseline={"run_id": None})
    built = build_status_payload(state)
    manual = assemble_status_payload(state, next_action=select_next_action(state))
    assert isinstance(built, StatusPayload)
    assert built.model_dump(mode="json") == manual.model_dump(mode="json")
    assert built.next_action.action_id == "set_baseline"


def test_build_status_payload_is_deterministic_json() -> None:
    state = _state()
    assert (
        build_status_payload(state).model_dump_json()
        == build_status_payload(state).model_dump_json()
    )


def test_fallback_reachable_for_generic_healthy_state() -> None:
    assert select_next_action(_state()).action_id == "explore_recommendations"
