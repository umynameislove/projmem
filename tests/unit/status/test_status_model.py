"""Unit tests for the ``status-v1`` payload contract (STS-001)."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import pmem.status as status_pkg
from pmem.status import STATUS_SCHEMA_VERSION, StatusPayload

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "status" / "status_payload_v1.json"
)

_ROOT_FIELDS = (
    "schema_version",
    "project",
    "metric",
    "counts",
    "best_run",
    "baseline",
    "graph",
    "recommendations",
    "warnings",
    "next_action",
    "database_mutation",
    "network",
    "raw_text_in_output",
)


def _load_fixture() -> dict[str, Any]:
    with _FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _mutated(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    payload = copy.deepcopy(_load_fixture())
    mutate(payload)
    return payload


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _variant(overrides: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(_load_fixture(), overrides)


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #
def test_golden_fixture_validates() -> None:
    payload = StatusPayload.model_validate(_load_fixture())
    assert payload.schema_version == STATUS_SCHEMA_VERSION


def test_dump_matches_fixture() -> None:
    fixture = _load_fixture()
    dumped = StatusPayload.model_validate(fixture).model_dump(mode="json")
    assert dumped == fixture


def test_public_exports_importable() -> None:
    for name in (
        "StatusPayload",
        "StatusProject",
        "StatusMetric",
        "StatusCounts",
        "StatusBestRun",
        "StatusBaseline",
        "StatusGraph",
        "StatusRecommendations",
        "StatusWarning",
        "StatusNextAction",
        "TargetStatus",
        "GraphState",
        "RecommendationMode",
        "WarningSeverity",
        "WarningSource",
        "STATUS_SCHEMA_VERSION",
    ):
        assert hasattr(status_pkg, name), name


def test_int_metric_is_accepted_as_float() -> None:
    payload = StatusPayload.model_validate(
        _variant({"metric": {"target_value": 1, "best_value": 0.87}})
    )
    assert payload.metric.target_value == 1.0
    assert isinstance(payload.metric.target_value, float)


# --------------------------------------------------------------------------- #
# Determinism & deep immutability                                              #
# --------------------------------------------------------------------------- #
def test_independent_models_serialize_identically() -> None:
    raw = _FIXTURE_PATH.read_text(encoding="utf-8")
    first = StatusPayload.model_validate_json(raw)
    second = StatusPayload.model_validate_json(raw)
    assert first.model_dump_json() == second.model_dump_json()


def test_dump_then_revalidate_is_stable() -> None:
    payload = StatusPayload.model_validate(_load_fixture())
    again = StatusPayload.model_validate(payload.model_dump(mode="json"))
    assert again.model_dump(mode="json") == payload.model_dump(mode="json")


def test_warnings_is_immutable_tuple() -> None:
    payload = StatusPayload.model_validate(_load_fixture())
    assert isinstance(payload.warnings, tuple)
    with pytest.raises(AttributeError):
        payload.warnings.append(payload.warnings[0])  # type: ignore[attr-defined]


def test_root_is_frozen() -> None:
    payload = StatusPayload.model_validate(_load_fixture())
    with pytest.raises(ValidationError):
        payload.network = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Required-field presence (no defaults hide producer mistakes)                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", _ROOT_FIELDS)
def test_missing_root_field_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        StatusPayload.model_validate(_mutated(lambda d: d.pop(field)))


_MISSING_NULLABLE: dict[str, Callable[[dict[str, Any]], None]] = {
    "objective": lambda d: d["project"].pop("objective"),
    "direction": lambda d: d["metric"].pop("direction"),
    "target_value": lambda d: d["metric"].pop("target_value"),
    "best_value": lambda d: d["metric"].pop("best_value"),
    "best_run_run_id": lambda d: d["best_run"].pop("run_id"),
    "best_run_metric_value": lambda d: d["best_run"].pop("metric_value"),
    "baseline_run_id": lambda d: d["baseline"].pop("run_id"),
    "graph_reason_code": lambda d: d["graph"].pop("reason_code"),
    "candidate_count": lambda d: d["recommendations"].pop("candidate_count"),
    "active_count": lambda d: d["recommendations"].pop("active_count"),
    "remediation": lambda d: d["warnings"][0].pop("remediation"),
    "related_entity_id": lambda d: d["next_action"].pop("related_entity_id"),
}


@pytest.mark.parametrize("field", sorted(_MISSING_NULLABLE))
def test_missing_nullable_field_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        StatusPayload.model_validate(_mutated(_MISSING_NULLABLE[field]))


# --------------------------------------------------------------------------- #
# Target-status consistency matrix (positive)                                  #
# --------------------------------------------------------------------------- #
_EMPTY_BEST = {"run_id": None, "metric_value": None}

_CONSISTENT_CASES: dict[str, dict[str, Any]] = {
    "no_runs": {
        "counts": {"run_count": 0, "successful_run_count": 0, "failed_run_count": 0},
        "metric": {
            "primary_metric": None,
            "direction": None,
            "target_value": None,
            "best_value": None,
            "target_status": "no_runs",
        },
        "best_run": _EMPTY_BEST,
    },
    "no_successful_runs": {
        "counts": {"run_count": 5, "successful_run_count": 0, "failed_run_count": 3},
        "metric": {
            "primary_metric": "accuracy",
            "direction": "max",
            "target_value": 0.9,
            "best_value": None,
            "target_status": "no_successful_runs",
        },
        "best_run": _EMPTY_BEST,
    },
    "not_configured": {
        "counts": {"run_count": 5, "successful_run_count": 3, "failed_run_count": 1},
        "metric": {
            "primary_metric": None,
            "direction": None,
            "target_value": None,
            "best_value": None,
            "target_status": "not_configured",
        },
        "best_run": _EMPTY_BEST,
    },
    "not_configured_target_only": {
        "counts": {"run_count": 5, "successful_run_count": 3, "failed_run_count": 1},
        "metric": {
            "primary_metric": "macro F1",
            "direction": "max",
            "target_value": None,
            "best_value": 0.87,
            "target_status": "not_configured",
        },
        "best_run": {"run_id": "run_" + "e" * 32, "metric_value": 0.87},
    },
    "no_metric": {
        "counts": {"run_count": 5, "successful_run_count": 3, "failed_run_count": 1},
        "metric": {
            "primary_metric": "accuracy",
            "direction": "max",
            "target_value": 0.9,
            "best_value": None,
            "target_status": "no_metric",
        },
        "best_run": _EMPTY_BEST,
    },
    "met_max": {
        "counts": {"run_count": 5, "successful_run_count": 4, "failed_run_count": 1},
        "metric": {
            "primary_metric": "accuracy",
            "direction": "max",
            "target_value": 0.9,
            "best_value": 0.95,
            "target_status": "met",
        },
        "best_run": {"run_id": "run_" + "a" * 32, "metric_value": 0.95},
    },
    "not_met_max": {
        "counts": {"run_count": 5, "successful_run_count": 4, "failed_run_count": 1},
        "metric": {
            "primary_metric": "accuracy",
            "direction": "max",
            "target_value": 0.9,
            "best_value": 0.80,
            "target_status": "not_met",
        },
        "best_run": {"run_id": "run_" + "b" * 32, "metric_value": 0.80},
    },
    "met_min": {
        "counts": {"run_count": 5, "successful_run_count": 4, "failed_run_count": 1},
        "metric": {
            "primary_metric": "loss",
            "direction": "min",
            "target_value": 0.20,
            "best_value": 0.10,
            "target_status": "met",
        },
        "best_run": {"run_id": "run_" + "c" * 32, "metric_value": 0.10},
    },
    "not_met_min": {
        "counts": {"run_count": 5, "successful_run_count": 4, "failed_run_count": 1},
        "metric": {
            "primary_metric": "loss",
            "direction": "min",
            "target_value": 0.20,
            "best_value": 0.30,
            "target_status": "not_met",
        },
        "best_run": {"run_id": "run_" + "d" * 32, "metric_value": 0.30},
    },
}


@pytest.mark.parametrize("case", sorted(_CONSISTENT_CASES))
def test_consistent_target_status_is_accepted(case: str) -> None:
    expected = _CONSISTENT_CASES[case]["metric"]["target_status"]
    payload = StatusPayload.model_validate(_variant(_CONSISTENT_CASES[case]))
    assert payload.metric.target_status.value == expected


# --------------------------------------------------------------------------- #
# Rejections (strictness, consistency, privacy, safety, limits)               #
# --------------------------------------------------------------------------- #
_REJECTIONS: dict[str, dict[str, Any]] = {
    # schema strictness
    "bad_schema_version": _variant({"schema_version": "status-v2"}),
    "extra_root_field": _mutated(lambda d: d.__setitem__("unexpected", 1)),
    "extra_nested_field": _mutated(lambda d: d["project"].__setitem__("unexpected", 1)),
    "next_action_none": _variant({"next_action": None}),
    "next_action_list": _mutated(lambda d: d.__setitem__("next_action", [d["next_action"]])),
    # counts / numerics
    "negative_count": _variant({"counts": {"note_count": -1}}),
    "string_count": _mutated(lambda d: d["counts"].__setitem__("note_count", "1")),
    "bool_count": _mutated(lambda d: d["counts"].__setitem__("note_count", True)),
    "totals_exceed_runs": _variant({"counts": {"successful_run_count": 10, "failed_run_count": 5}}),
    "nan_metric": _mutated(lambda d: d["metric"].__setitem__("target_value", float("nan"))),
    "inf_metric": _mutated(lambda d: d["metric"].__setitem__("best_value", float("inf"))),
    "bool_metric": _mutated(lambda d: d["metric"].__setitem__("target_value", True)),
    "string_metric": _mutated(lambda d: d["metric"].__setitem__("target_value", "0.9")),
    "bad_target_status_value": _variant({"metric": {"target_status": "unknown"}}),
    "bad_direction_value": _variant({"metric": {"direction": "highest"}}),
    # target-status consistency
    "met_without_metric": _variant(
        {
            "metric": {
                "primary_metric": None,
                "direction": None,
                "target_value": None,
                "best_value": None,
                "target_status": "met",
            },
            "best_run": _EMPTY_BEST,
        }
    ),
    "no_runs_with_runs": _variant(
        {"counts": {"run_count": 5}, "metric": {"target_status": "no_runs"}}
    ),
    "no_successful_with_best_run": _variant(
        {"counts": {"successful_run_count": 0, "failed_run_count": 3}}
    ),
    "best_value_ne_best_run": _variant({"metric": {"best_value": 0.5}}),
    "metric_value_without_run_id": _variant({"best_run": {"run_id": None}}),
    "wrong_met_direction": _variant({"metric": {"target_status": "met"}}),
    "metric_none_with_best": _variant(
        {"metric": {"primary_metric": None, "target_status": "not_configured"}}
    ),
    "direction_none_with_best": _variant(
        {"metric": {"direction": None, "target_status": "not_configured"}}
    ),
    # recommendation invariants
    "active_without_lifecycle": _variant({"recommendations": {"active_count": 2}}),
    "candidate_when_not_evaluated": _variant(
        {"recommendations": {"mode": "not_evaluated", "candidate_count": 1, "active_count": None}}
    ),
    "generated_candidate_null": _variant({"recommendations": {"candidate_count": None}}),
    "persisted_active_null": _variant(
        {
            "recommendations": {
                "mode": "persisted_lifecycle",
                "candidate_count": 3,
                "active_count": None,
            }
        }
    ),
    "active_exceeds_candidate": _variant(
        {
            "recommendations": {
                "mode": "persisted_lifecycle",
                "candidate_count": 1,
                "active_count": 5,
            }
        }
    ),
    "negative_candidate_count": _variant({"recommendations": {"candidate_count": -1}}),
    "bad_recommendation_mode": _variant({"recommendations": {"mode": "live"}}),
    # graph invariants
    "graph_negative_count": _variant({"graph": {"node_count": -1}}),
    "graph_current_missing_counts": _variant(
        {"graph": {"state": "current", "node_count": None, "edge_count": None, "reason_code": None}}
    ),
    "graph_current_with_reason": _variant({"graph": {"state": "current"}}),
    "graph_missing_with_counts": _variant({"graph": {"state": "missing"}}),
    "graph_one_count_only": _variant({"graph": {"edge_count": None}}),
    "graph_stale_no_reason": _variant({"graph": {"reason_code": None}}),
    "graph_bad_state": _variant({"graph": {"state": "fresh"}}),
    # privacy / flags
    "database_mutation_true": _variant({"database_mutation": True}),
    "network_true": _variant({"network": True}),
    "raw_text_true": _variant({"raw_text_in_output": True}),
    # identifiers / codes
    "warning_code_space": _mutated(
        lambda d: d["warnings"][0].__setitem__("code", "not machine readable")
    ),
    "warning_code_uppercase": _mutated(lambda d: d["warnings"][0].__setitem__("code", "Graph")),
    "action_id_semicolon": _mutated(
        lambda d: d["next_action"].__setitem__("action_id", "do; this")
    ),
    "reason_code_space": _variant({"graph": {"reason_code": "any thing here"}}),
    "project_id_slash": _variant({"project": {"project_id": "proj/one"}}),
    "run_id_whitespace": _variant({"best_run": {"run_id": "run one"}}),
    "related_entity_absolute_path": _variant(
        {"next_action": {"related_entity_id": "/Users/private/secret"}}
    ),
    # text safety
    "blank_warning_message": _mutated(lambda d: d["warnings"][0].__setitem__("message", "   ")),
    "control_char_message": _mutated(
        lambda d: d["warnings"][0].__setitem__("message", "line one\nline two")
    ),
    "command_not_pmem": _mutated(lambda d: d["next_action"].__setitem__("suggested_command", "ls")),
    "command_shell_and": _mutated(
        lambda d: d["next_action"].__setitem__("suggested_command", "pmem x && rm -rf .")
    ),
    "command_shell_semicolon": _mutated(
        lambda d: d["next_action"].__setitem__("suggested_command", "pmem x; echo hi")
    ),
    "command_shell_pipe": _mutated(
        lambda d: d["next_action"].__setitem__("suggested_command", "pmem x | grep y")
    ),
    "command_subshell": _mutated(
        lambda d: d["next_action"].__setitem__("suggested_command", "pmem run $(whoami)")
    ),
    "command_backtick": _mutated(
        lambda d: d["next_action"].__setitem__("suggested_command", "pmem run `whoami`")
    ),
    "absolute_path_message": _mutated(
        lambda d: d["warnings"][0].__setitem__("message", "See /Users/x/secret for details")
    ),
    "absolute_path_parens": _mutated(
        lambda d: d["warnings"][0].__setitem__("message", "See (/Users/x/secret) now")
    ),
    "absolute_path_quoted": _mutated(
        lambda d: d["warnings"][0].__setitem__("message", 'open "/Users/x/secret" here')
    ),
    "absolute_path_file_uri": _mutated(
        lambda d: d["warnings"][0].__setitem__("message", "open file:///Users/x now")
    ),
    "absolute_path_windows": _mutated(
        lambda d: d["warnings"][0].__setitem__("message", r"see (C:\Users\x) here")
    ),
    "absolute_path_reason": _mutated(
        lambda d: d["next_action"].__setitem__("reason", "Config /etc/pmem changed")
    ),
    "absolute_path_remediation": _mutated(
        lambda d: d["warnings"][0].__setitem__("remediation", "edit /home/user/config")
    ),
    "absolute_path_after_equals": _mutated(
        lambda d: d["warnings"][0].__setitem__("message", "path=/Users/x/secret")
    ),
    "windows_path_after_equals": _mutated(
        lambda d: d["warnings"][0].__setitem__("message", r"path=C:\Users\x\secret")
    ),
    # length limits
    "objective_too_long": _variant({"project": {"objective": "x" * 513}}),
    "project_name_too_long": _variant({"project": {"project_name": "y" * 121}}),
    "metric_name_too_long": _variant({"metric": {"primary_metric": "m" * 513}}),
    "message_too_long": _mutated(lambda d: d["warnings"][0].__setitem__("message", "z" * 513)),
    "reason_too_long": _mutated(lambda d: d["next_action"].__setitem__("reason", "r" * 513)),
    "remediation_too_long": _mutated(
        lambda d: d["warnings"][0].__setitem__("remediation", "r" * 257)
    ),
    "command_too_long": _mutated(
        lambda d: d["next_action"].__setitem__("suggested_command", "pmem " + "x" * 252)
    ),
    "code_too_long": _mutated(lambda d: d["warnings"][0].__setitem__("code", "c" * 65)),
    "identifier_too_long": _variant({"project": {"project_id": "p" * 129}}),
    "too_many_warnings": _mutated(lambda d: d.__setitem__("warnings", [d["warnings"][0]] * 101)),
}


@pytest.mark.parametrize("case", sorted(_REJECTIONS))
def test_invalid_payload_is_rejected(case: str) -> None:
    with pytest.raises(ValidationError):
        StatusPayload.model_validate(_REJECTIONS[case])


# --------------------------------------------------------------------------- #
# Focused positive checks                                                      #
# --------------------------------------------------------------------------- #
def test_exactly_one_next_action_object() -> None:
    payload = StatusPayload.model_validate(_load_fixture())
    assert not isinstance(payload.next_action, list)
    assert payload.next_action.suggested_command.startswith("pmem ")


def test_generated_on_demand_requires_candidate_and_null_active() -> None:
    payload = StatusPayload.model_validate(_load_fixture())
    assert payload.recommendations.candidate_count == 3
    assert payload.recommendations.active_count is None


def test_recommendation_modes_round_trip() -> None:
    not_evaluated = StatusPayload.model_validate(
        _variant(
            {
                "recommendations": {
                    "mode": "not_evaluated",
                    "candidate_count": None,
                    "active_count": None,
                }
            }
        )
    )
    assert not_evaluated.recommendations.candidate_count is None

    persisted = StatusPayload.model_validate(
        _variant(
            {
                "recommendations": {
                    "mode": "persisted_lifecycle",
                    "candidate_count": 5,
                    "active_count": 2,
                }
            }
        )
    )
    assert persisted.recommendations.active_count == 2


def test_graph_states_round_trip() -> None:
    current = StatusPayload.model_validate(
        _variant(
            {"graph": {"state": "current", "node_count": 40, "edge_count": 63, "reason_code": None}}
        )
    )
    assert current.graph.reason_code is None
    missing = StatusPayload.model_validate(
        _variant(
            {
                "graph": {
                    "state": "missing",
                    "node_count": None,
                    "edge_count": None,
                    "reason_code": "graph_not_built",
                }
            }
        )
    )
    assert missing.graph.node_count is None
