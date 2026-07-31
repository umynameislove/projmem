"""Contract tests for the ``doctor-v1`` payload models (DOC-001)."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import pmem.doctor as doctor_pkg
from pmem.doctor import (
    DOCTOR_SCHEMA_VERSION,
    DoctorCheckResult,
    DoctorReport,
    DoctorSummary,
    render_doctor_report_json,
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "doctor" / "doctor_report_v1.json"
)

_ROOT_FIELDS = (
    "schema_version",
    "project",
    "overall_outcome",
    "summary",
    "checks",
    "database_mutation",
    "network",
    "automatic_repair",
    "raw_text_in_output",
)

_CHECK_FIELDS = (
    "check_id",
    "category",
    "outcome",
    "severity",
    "message",
    "remediation",
    "related_entity_id",
)


def _load_fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _mutated(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    report = copy.deepcopy(_load_fixture())
    mutate(report)
    return report


def _passing_check(**overrides: Any) -> dict[str, Any]:
    check = {
        "check_id": "database.exists",
        "category": "database",
        "outcome": "pass",
        "severity": "info",
        "message": "The project database is present and readable.",
        "remediation": None,
        "related_entity_id": None,
    }
    check.update(overrides)
    return check


# --------------------------------------------------------------------------- #
# Happy path & golden snapshot                                                 #
# --------------------------------------------------------------------------- #
def test_golden_fixture_validates() -> None:
    report = DoctorReport.model_validate(_load_fixture())
    assert report.schema_version == DOCTOR_SCHEMA_VERSION
    assert DOCTOR_SCHEMA_VERSION == "doctor-v1"


def test_rendered_json_matches_the_committed_fixture_byte_for_byte() -> None:
    """Exact serialized snapshot: no ``json.loads`` on either side.

    This locks field order, 2-space indentation and the newline convention (the
    serializer does not emit a trailing newline; the committed file has one
    because ``end-of-file-fixer`` requires it). A shape change at any nesting
    level fails here and forces a deliberate fixture update.
    """

    raw = _FIXTURE_PATH.read_text(encoding="utf-8")
    report = DoctorReport.model_validate(json.loads(raw))

    assert render_doctor_report_json(report) + "\n" == raw
    assert not render_doctor_report_json(report).endswith("\n")


def test_round_trip_is_stable() -> None:
    report = DoctorReport.model_validate(_load_fixture())
    again = DoctorReport.model_validate(report.model_dump(mode="json"))

    assert again == report
    assert render_doctor_report_json(again) == render_doctor_report_json(report)


def test_public_exports_are_importable() -> None:
    for name in (
        "DOCTOR_SCHEMA_VERSION",
        "DoctorCategory",
        "DoctorCheck",
        "DoctorCheckContext",
        "DoctorCheckDefinition",
        "DoctorCheckOutcome",
        "DoctorCheckRegistry",
        "DoctorCheckResult",
        "DoctorOverallOutcome",
        "DoctorProject",
        "DoctorReport",
        "DoctorSeverity",
        "DoctorSummary",
        "build_doctor_report",
        "render_doctor_report_json",
    ):
        assert hasattr(doctor_pkg, name), name


def test_public_surface_hides_implementation_detail() -> None:
    exported = set(doctor_pkg.__all__)
    internal_helpers = {
        "canonical_check_order",
        "resolve_overall_outcome",
        "summarize_checks",
        "validate_check_id",
    }

    assert not any(name.startswith("_") for name in exported)
    assert "contains_absolute_path" not in exported
    assert "contains_control_chars" not in exported
    assert exported.isdisjoint(internal_helpers)
    assert all(not hasattr(doctor_pkg, name) for name in internal_helpers)
    assert exported == {name for name in dir(doctor_pkg) if name in exported}


# --------------------------------------------------------------------------- #
# Required fields, unknown fields, strictness                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", _ROOT_FIELDS)
def test_missing_root_field_is_rejected(field: str) -> None:
    report = _load_fixture()
    del report[field]

    with pytest.raises(ValidationError):
        DoctorReport.model_validate(report)


@pytest.mark.parametrize("field", _CHECK_FIELDS)
def test_missing_check_field_is_rejected(field: str) -> None:
    check = _passing_check()
    del check[field]

    with pytest.raises(ValidationError):
        DoctorCheckResult.model_validate(check)


@pytest.mark.parametrize("field", ("project", "remediation", "related_entity_id"))
def test_nullable_field_is_still_required(field: str) -> None:
    """Nullable never means optional: ``null`` must be sent explicitly."""

    if field == "project":
        report = _load_fixture()
        del report["project"]
        with pytest.raises(ValidationError):
            DoctorReport.model_validate(report)
        return

    check = _passing_check()
    del check[field]
    with pytest.raises(ValidationError):
        DoctorCheckResult.model_validate(check)


def test_unknown_root_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DoctorReport.model_validate(_mutated(lambda r: r.__setitem__("extra", 1)))


def test_unknown_check_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DoctorCheckResult.model_validate(_passing_check(extra="nope"))


@pytest.mark.parametrize(
    "value",
    ["7", 7.0, True],
    ids=["string", "float", "bool"],
)
def test_summary_rejects_coercion(value: Any) -> None:
    summary = _load_fixture()["summary"]
    summary["total"] = value

    with pytest.raises(ValidationError):
        DoctorSummary.model_validate(summary)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_summary_rejects_non_finite_numbers(value: float) -> None:
    """The contract carries no float field; counts reject NaN/inf outright."""

    summary = _load_fixture()["summary"]
    summary["total"] = value

    with pytest.raises(ValidationError):
        DoctorSummary.model_validate(summary)


def test_negative_count_is_rejected() -> None:
    summary = _load_fixture()["summary"]
    summary["passed"] = -1

    with pytest.raises(ValidationError):
        DoctorSummary.model_validate(summary)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", "networking"),
        ("outcome", "warned"),
        ("severity", "critical"),
        ("outcome", 1),
        ("severity", None),
    ],
)
def test_invalid_enum_value_is_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        DoctorCheckResult.model_validate(_passing_check(**{field: value}))


def test_invalid_overall_outcome_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DoctorReport.model_validate(_mutated(lambda r: r.__setitem__("overall_outcome", "broken")))


@pytest.mark.parametrize("field", ("database_mutation", "network", "automatic_repair"))
def test_safety_flag_cannot_be_true(field: str) -> None:
    with pytest.raises(ValidationError):
        DoctorReport.model_validate(_mutated(lambda r: r.__setitem__(field, True)))


def test_raw_text_flag_cannot_be_true() -> None:
    with pytest.raises(ValidationError):
        DoctorReport.model_validate(_mutated(lambda r: r.__setitem__("raw_text_in_output", True)))


def test_wrong_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DoctorReport.model_validate(
            _mutated(lambda r: r.__setitem__("schema_version", "doctor-v2"))
        )


# --------------------------------------------------------------------------- #
# check_id rules                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "check_id",
    [
        "",
        "   ",
        "database",
        "Database.exists",
        "database.Exists",
        "database..exists",
        ".database.exists",
        "database.exists.",
        "database exists",
        "database.exists ",
        "database/exists",
        "database\\exists",
        "database.exists\n",
        "database.$(id)",
        "database.exists;rm",
        "/etc/passwd",
        "1database.exists",
        "database." + "x" * 80,
    ],
)
def test_invalid_check_id_is_rejected(check_id: str) -> None:
    with pytest.raises(ValidationError):
        DoctorCheckResult.model_validate(_passing_check(check_id=check_id, category="database"))


@pytest.mark.parametrize(
    ("check_id", "category"),
    [
        ("database.exists", "database"),
        ("database.migration_checksum", "database"),
        ("permissions.private_directory", "permissions"),
        ("tracked_paths.symlink", "tracked_paths"),
        ("evidence.graph_freshness", "evidence"),
        ("environment.git_available", "environment"),
        ("database.foreign_keys.orphan_rows", "database"),
    ],
)
def test_namespaced_check_id_is_accepted(check_id: str, category: str) -> None:
    result = DoctorCheckResult.model_validate(_passing_check(check_id=check_id, category=category))

    assert result.check_id == check_id


def test_check_id_namespace_must_match_category() -> None:
    with pytest.raises(ValidationError, match="must equal category"):
        DoctorCheckResult.model_validate(
            _passing_check(check_id="database.exists", category="permissions")
        )


# --------------------------------------------------------------------------- #
# Outcome / severity / remediation coherence                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("severity", ("warning", "error"))
def test_passing_check_must_be_info(severity: str) -> None:
    with pytest.raises(ValidationError, match="must carry severity 'info'"):
        DoctorCheckResult.model_validate(_passing_check(severity=severity))


def test_passing_check_cannot_demand_remediation() -> None:
    with pytest.raises(ValidationError, match="must not carry remediation"):
        DoctorCheckResult.model_validate(_passing_check(remediation="Do something."))


def test_failing_check_cannot_be_info() -> None:
    with pytest.raises(ValidationError, match="must carry severity"):
        DoctorCheckResult.model_validate(
            _passing_check(outcome="fail", severity="info", remediation="Fix it.")
        )


def test_failing_check_requires_remediation() -> None:
    with pytest.raises(ValidationError, match="must carry remediation"):
        DoctorCheckResult.model_validate(
            _passing_check(outcome="fail", severity="error", remediation=None)
        )


def test_failing_check_rejects_blank_remediation() -> None:
    with pytest.raises(ValidationError):
        DoctorCheckResult.model_validate(
            _passing_check(outcome="fail", severity="error", remediation="   ")
        )


def test_skipped_check_cannot_be_error() -> None:
    with pytest.raises(ValidationError, match="coverage gap"):
        DoctorCheckResult.model_validate(_passing_check(outcome="skipped", severity="error"))


def test_skipped_check_requires_an_explanatory_message() -> None:
    with pytest.raises(ValidationError):
        DoctorCheckResult.model_validate(
            _passing_check(outcome="skipped", severity="info", message="")
        )


def test_skipped_check_is_not_a_pass() -> None:
    skipped = DoctorCheckResult.model_validate(
        _passing_check(
            outcome="skipped",
            severity="warning",
            message="Git was not found on PATH, so repository checks did not run.",
        )
    )

    assert skipped.outcome.value == "skipped"
    assert skipped.outcome.value != "pass"


def test_not_applicable_check_is_valid_and_distinct() -> None:
    result = DoctorCheckResult.model_validate(
        _passing_check(
            check_id="permissions.posix_modes",
            category="permissions",
            outcome="not_applicable",
            severity="info",
            message="POSIX permission checks do not apply on this platform.",
        )
    )

    assert result.outcome.value == "not_applicable"
    assert result.outcome.value not in {"pass", "skipped"}


@pytest.mark.parametrize("severity", ("warning", "error"))
def test_not_applicable_check_must_be_info(severity: str) -> None:
    with pytest.raises(ValidationError, match="not-applicable check must carry severity 'info'"):
        DoctorCheckResult.model_validate(
            _passing_check(outcome="not_applicable", severity=severity)
        )


def test_not_applicable_check_cannot_demand_remediation() -> None:
    with pytest.raises(ValidationError, match="not-applicable check must not carry remediation"):
        DoctorCheckResult.model_validate(
            _passing_check(outcome="not_applicable", remediation="Change the platform.")
        )


# --------------------------------------------------------------------------- #
# Report-level invariants                                                      #
# --------------------------------------------------------------------------- #
def test_duplicate_check_id_is_rejected() -> None:
    def _duplicate(report: dict[str, Any]) -> None:
        report["checks"].append(copy.deepcopy(report["checks"][0]))
        report["summary"]["total"] += 1
        report["summary"]["passed"] += 1
        report["summary"]["info"] += 1

    with pytest.raises(ValidationError, match="duplicate check_id"):
        DoctorReport.model_validate(_mutated(_duplicate))


def test_unsorted_checks_are_rejected() -> None:
    def _shuffle(report: dict[str, Any]) -> None:
        report["checks"] = list(reversed(report["checks"]))

    with pytest.raises(ValidationError, match="ascending check_id order"):
        DoctorReport.model_validate(_mutated(_shuffle))


@pytest.mark.parametrize(
    "field", ("total", "passed", "failed", "skipped", "not_applicable", "info", "warning")
)
def test_summary_mismatch_is_rejected(field: str) -> None:
    """Bumping one count breaks the summary's own internal totals."""

    def _bump(report: dict[str, Any]) -> None:
        report["summary"][field] += 1

    with pytest.raises(ValidationError):
        DoctorReport.model_validate(_mutated(_bump))


@pytest.mark.parametrize(
    ("left", "right"),
    [("passed", "failed"), ("info", "warning")],
)
def test_internally_consistent_but_wrong_summary_is_rejected(left: str, right: str) -> None:
    """The root cross-check, not just the summary's own arithmetic.

    Moving one unit between two counts keeps both of ``DoctorSummary``'s
    internal sums valid, so only the root validator's comparison against the
    real checks can catch it. Without this case that comparison is unreachable.
    """

    def _shift(report: dict[str, Any]) -> None:
        report["summary"][left] -= 1
        report["summary"][right] += 1

    with pytest.raises(ValidationError, match="summary counts must match"):
        DoctorReport.model_validate(_mutated(_shift))


def test_overall_outcome_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="overall_outcome must be"):
        DoctorReport.model_validate(_mutated(lambda r: r.__setitem__("overall_outcome", "healthy")))


def test_empty_checks_is_rejected() -> None:
    """A report that ran no check asserts nothing and must not exist."""

    def _empty(report: dict[str, Any]) -> None:
        report["checks"] = []
        report["overall_outcome"] = "healthy"
        report["summary"] = dict.fromkeys(report["summary"], 0)

    with pytest.raises(ValidationError):
        DoctorReport.model_validate(_mutated(_empty))


def test_report_without_project_cannot_be_healthy() -> None:
    def _anonymous_healthy(report: dict[str, Any]) -> None:
        report["project"] = None
        report["checks"] = [_passing_check()]
        report["overall_outcome"] = "healthy"
        report["summary"] = {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "not_applicable": 0,
            "info": 1,
            "warning": 0,
            "error": 0,
        }

    with pytest.raises(ValidationError, match="cannot conclude 'healthy'"):
        DoctorReport.model_validate(_mutated(_anonymous_healthy))


def test_report_without_project_is_allowed_when_not_healthy() -> None:
    report = DoctorReport.model_validate(_mutated(lambda r: r.__setitem__("project", None)))

    assert report.project is None
    assert report.overall_outcome.value == "unhealthy"


# --------------------------------------------------------------------------- #
# Immutability & JSON shape                                                    #
# --------------------------------------------------------------------------- #
def test_root_is_frozen() -> None:
    report = DoctorReport.model_validate(_load_fixture())

    with pytest.raises(ValidationError):
        report.overall_outcome = "healthy"  # type: ignore[misc]


def test_nested_check_is_frozen() -> None:
    report = DoctorReport.model_validate(_load_fixture())

    with pytest.raises(ValidationError):
        report.checks[0].message = "changed"  # type: ignore[misc]


def test_checks_is_an_immutable_tuple() -> None:
    report = DoctorReport.model_validate(_load_fixture())

    assert isinstance(report.checks, tuple)
    with pytest.raises(AttributeError):
        report.checks.append(report.checks[0])  # type: ignore[attr-defined]


def test_checks_serialize_as_a_json_array() -> None:
    document = json.loads(render_doctor_report_json(DoctorReport.model_validate(_load_fixture())))

    assert isinstance(document["checks"], list)
    assert isinstance(document["summary"], dict)
    assert all(isinstance(check, dict) for check in document["checks"])


def test_enums_serialize_as_canonical_strings() -> None:
    document = json.loads(render_doctor_report_json(DoctorReport.model_validate(_load_fixture())))

    assert document["overall_outcome"] == "unhealthy"
    for check in document["checks"]:
        assert isinstance(check["category"], str)
        assert isinstance(check["outcome"], str)
        assert isinstance(check["severity"], str)
