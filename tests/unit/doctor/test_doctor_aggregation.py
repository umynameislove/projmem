"""Decision-table tests for doctor aggregation (DOC-001)."""

from __future__ import annotations

import itertools

import pytest

from pmem.doctor import (
    DoctorCategory,
    DoctorCheckOutcome,
    DoctorCheckResult,
    DoctorOverallOutcome,
    DoctorProject,
    DoctorSeverity,
    build_doctor_report,
    render_doctor_report_json,
)
from pmem.doctor.model import canonical_check_order, resolve_overall_outcome, summarize_checks

_PROJECT = DoctorProject(
    project_id="proj_9f2c1a7b4d6e40f2a1b3c5d7e9f00112",
    project_name="AG News baseline",
)


def _check(
    check_id: str,
    outcome: DoctorCheckOutcome,
    severity: DoctorSeverity,
    *,
    category: DoctorCategory = DoctorCategory.DATABASE,
) -> DoctorCheckResult:
    remediation = "Review the diagnostic and follow the documented recovery step."
    return DoctorCheckResult(
        check_id=check_id,
        category=category,
        outcome=outcome,
        severity=severity,
        message="A deterministic fixture message for aggregation tests.",
        remediation=remediation if outcome is DoctorCheckOutcome.FAIL else None,
        related_entity_id=None,
    )


_PASS = _check("database.exists", DoctorCheckOutcome.PASS, DoctorSeverity.INFO)
_PASS_2 = _check("database.foreign_keys", DoctorCheckOutcome.PASS, DoctorSeverity.INFO)
_FAIL_WARNING = _check("database.integrity", DoctorCheckOutcome.FAIL, DoctorSeverity.WARNING)
_FAIL_ERROR = _check("database.migration_checksum", DoctorCheckOutcome.FAIL, DoctorSeverity.ERROR)
_SKIPPED = _check(
    "environment.git_available",
    DoctorCheckOutcome.SKIPPED,
    DoctorSeverity.INFO,
    category=DoctorCategory.ENVIRONMENT,
)
_NOT_APPLICABLE = _check(
    "permissions.posix_modes",
    DoctorCheckOutcome.NOT_APPLICABLE,
    DoctorSeverity.INFO,
    category=DoctorCategory.PERMISSIONS,
)


# --------------------------------------------------------------------------- #
# The decision table, row by row                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        ((_PASS,), DoctorOverallOutcome.HEALTHY),
        ((_PASS, _PASS_2), DoctorOverallOutcome.HEALTHY),
        ((_PASS, _NOT_APPLICABLE), DoctorOverallOutcome.HEALTHY),
        ((_NOT_APPLICABLE,), DoctorOverallOutcome.INCOMPLETE),
        ((_PASS, _FAIL_WARNING), DoctorOverallOutcome.DEGRADED),
        ((_PASS, _FAIL_ERROR), DoctorOverallOutcome.UNHEALTHY),
        ((_PASS, _SKIPPED), DoctorOverallOutcome.INCOMPLETE),
        # error outranks every other signal
        ((_FAIL_ERROR, _FAIL_WARNING), DoctorOverallOutcome.UNHEALTHY),
        ((_FAIL_ERROR, _SKIPPED), DoctorOverallOutcome.UNHEALTHY),
        ((_FAIL_ERROR, _FAIL_WARNING, _SKIPPED, _PASS), DoctorOverallOutcome.UNHEALTHY),
        # a definite warning failure outranks unknown coverage
        ((_FAIL_WARNING, _SKIPPED), DoctorOverallOutcome.DEGRADED),
        # a skipped check blocks a clean bill of health
        ((_PASS, _PASS_2, _SKIPPED), DoctorOverallOutcome.INCOMPLETE),
        ((_PASS, _SKIPPED, _NOT_APPLICABLE), DoctorOverallOutcome.INCOMPLETE),
    ],
    ids=[
        "single_pass_healthy",
        "all_pass_healthy",
        "not_applicable_is_neutral",
        "only_not_applicable_is_incomplete",
        "warning_failure_degraded",
        "error_failure_unhealthy",
        "skipped_incomplete",
        "error_beats_warning",
        "error_beats_skipped",
        "error_beats_everything",
        "warning_beats_skipped",
        "skipped_blocks_healthy",
        "skipped_still_blocks_with_not_applicable",
    ],
)
def test_overall_outcome_decision_table(
    checks: tuple[DoctorCheckResult, ...], expected: DoctorOverallOutcome
) -> None:
    assert resolve_overall_outcome(checks) is expected


def test_overall_outcome_is_order_independent() -> None:
    checks = (_PASS, _FAIL_WARNING, _FAIL_ERROR, _SKIPPED, _NOT_APPLICABLE)

    outcomes = {resolve_overall_outcome(tuple(order)) for order in itertools.permutations(checks)}

    assert outcomes == {DoctorOverallOutcome.UNHEALTHY}


# --------------------------------------------------------------------------- #
# Summary counting                                                             #
# --------------------------------------------------------------------------- #
def test_summarize_counts_outcomes_and_severities() -> None:
    summary = summarize_checks(
        (_PASS, _PASS_2, _FAIL_WARNING, _FAIL_ERROR, _SKIPPED, _NOT_APPLICABLE)
    )

    assert summary.total == 6
    assert (summary.passed, summary.failed, summary.skipped, summary.not_applicable) == (
        2,
        2,
        1,
        1,
    )
    assert (summary.info, summary.warning, summary.error) == (4, 1, 1)


def test_summary_totals_are_self_consistent() -> None:
    summary = summarize_checks((_PASS, _FAIL_ERROR, _SKIPPED))

    assert (
        summary.passed + summary.failed + summary.skipped + summary.not_applicable == summary.total
    )
    assert summary.info + summary.warning + summary.error == summary.total


# --------------------------------------------------------------------------- #
# Canonical ordering & report assembly                                         #
# --------------------------------------------------------------------------- #
def test_canonical_order_sorts_by_check_id() -> None:
    ordered = canonical_check_order((_SKIPPED, _FAIL_ERROR, _PASS, _FAIL_WARNING))

    assert [check.check_id for check in ordered] == sorted(
        check.check_id for check in (_SKIPPED, _FAIL_ERROR, _PASS, _FAIL_WARNING)
    )


def test_build_report_is_byte_identical_for_every_input_order() -> None:
    checks = (_PASS, _PASS_2, _FAIL_WARNING, _FAIL_ERROR, _SKIPPED)

    rendered = {
        render_doctor_report_json(build_doctor_report(project=_PROJECT, checks=tuple(order)))
        for order in itertools.permutations(checks)
    }

    assert len(rendered) == 1


def test_build_report_derives_summary_and_outcome() -> None:
    report = build_doctor_report(project=_PROJECT, checks=(_SKIPPED, _PASS, _FAIL_ERROR))

    assert report.overall_outcome is DoctorOverallOutcome.UNHEALTHY
    assert report.summary == summarize_checks(report.checks)
    assert [check.check_id for check in report.checks] == sorted(
        check.check_id for check in report.checks
    )


def test_build_report_locks_every_safety_flag_false() -> None:
    report = build_doctor_report(project=_PROJECT, checks=(_PASS,))

    assert report.database_mutation is False
    assert report.network is False
    assert report.automatic_repair is False
    assert report.raw_text_in_output is False


def test_build_report_accepts_a_null_project_when_not_healthy() -> None:
    report = build_doctor_report(project=None, checks=(_FAIL_ERROR,))

    assert report.project is None
    assert report.overall_outcome is DoctorOverallOutcome.UNHEALTHY


def test_build_report_rejects_a_null_project_that_would_be_healthy() -> None:
    with pytest.raises(ValueError, match="cannot conclude 'healthy'"):
        build_doctor_report(project=None, checks=(_PASS,))


def test_build_report_rejects_duplicate_check_ids() -> None:
    with pytest.raises(ValueError, match="duplicate check_id"):
        build_doctor_report(project=_PROJECT, checks=(_PASS, _PASS))


def test_repeated_builds_are_byte_identical() -> None:
    first = build_doctor_report(project=_PROJECT, checks=(_PASS, _FAIL_ERROR))
    second = build_doctor_report(project=_PROJECT, checks=(_PASS, _FAIL_ERROR))

    assert render_doctor_report_json(first) == render_doctor_report_json(second)
