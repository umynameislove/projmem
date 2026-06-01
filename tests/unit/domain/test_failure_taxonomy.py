"""Tests for entity schema failure taxonomy behavior.

The DOCX requires stable default tags, snake_case normalization, and allowance
for user-defined tags so clustering can use clean data without blocking users.
"""

import pytest

from pmem.domain.common import FailureCandidateKind, FailureSeverity, FailureSource
from pmem.domain.failure import Failure
from pmem.domain.failure_taxonomy import (
    DEFAULT_FAILURE_TAGS,
    is_default_tag,
    normalize_tag,
    normalize_tags,
)
from pmem.domain.target import FailureCandidate


def test_default_failure_tags_match_d3_contract() -> None:
    """The ten default tags from the roadmap should be present."""

    assert set(DEFAULT_FAILURE_TAGS) == {
        "data_quality",
        "config_error",
        "oom",
        "gradient_issue",
        "convergence",
        "reproducibility",
        "environment",
        "logic_error",
        "timeout",
        "data_pipeline",
    }


def test_failure_severity_values_match_d11_contract() -> None:
    """failure taxonomy locks the allowed confirmed-failure severity values."""

    assert {severity.value for severity in FailureSeverity} == {
        "critical",
        "high",
        "medium",
        "low",
    }


def test_failure_source_values_match_d11_contract() -> None:
    """failure taxonomy locks the allowed confirmed-failure source values."""

    assert {source.value for source in FailureSource} == {
        "user_confirmed",
        "auto_technical",
        "promoted_candidate",
    }


def test_tags_normalize_to_snake_case_and_deduplicate() -> None:
    """Tags should be normalized before storage in tags_json."""

    assert normalize_tag(" Data Quality! ") == "data_quality"
    assert normalize_tags(["Data Quality", "data-quality", "Custom Tag"]) == [
        "data_quality",
        "custom_tag",
    ]


def test_unknown_user_defined_tag_is_allowed_after_normalization() -> None:
    """Unknown tags should not be rejected; only normalized."""

    failure = Failure(
        id="fail_1",
        run_id="run_1",
        error_type="MetricRegression",
        description="Accuracy dropped below the baseline.",
        tags=["new Weird Tag"],
    )

    assert failure.tags == ["new_weird_tag"]
    assert not is_default_tag("new Weird Tag")


def test_empty_tag_is_rejected() -> None:
    """Blank tags would create useless search keys."""

    with pytest.raises(ValueError, match="failure tags cannot be empty"):
        normalize_tag("   ")


def test_failure_optional_blank_text_is_rejected() -> None:
    """Optional failure text may be omitted, but present text cannot be blank."""

    with pytest.raises(ValueError, match="cannot be blank"):
        Failure(
            id="f1",
            run_id="r1",
            error_type="X",
            description="Y",
            root_cause="   ",
        )


def test_failure_source_and_severity_are_constrained() -> None:
    """Confirmed failures need an explicit source and severity vocabulary."""

    failure = Failure(
        id="fail_2",
        run_id="run_2",
        error_type="FileNotFoundError",
        description="Metrics file was missing.",
        severity=FailureSeverity.HIGH,
        source=FailureSource.AUTO_TECHNICAL,
        tags=["data pipeline"],
    )

    assert failure.source == FailureSource.AUTO_TECHNICAL
    assert failure.severity == FailureSeverity.HIGH
    assert failure.tags == ["data_pipeline"]


def test_failure_candidate_normalizes_suggested_tag() -> None:
    """Failure candidates carry suggested tags but remain unconfirmed."""

    candidate = FailureCandidate(
        kind=FailureCandidateKind.BASELINE_REGRESSION,
        evidence="accuracy 0.83 is below baseline 0.86",
        suggested_tag="Convergence",
    )

    assert candidate.suggested_tag == "convergence"
    assert candidate.resolved is False
