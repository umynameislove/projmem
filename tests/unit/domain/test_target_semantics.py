"""Tests for target, baseline, and failure-candidate semantics.

These are direct checks for the DOCX addition: no target means no research
failure judgment, while target/baseline/constraint misses become candidates.
"""

import pytest
from pydantic import ValidationError

from pmem.domain.common import EvaluationFlag, FailureCandidateKind, MetricDirection, RunStatus
from pmem.domain.target import (
    BaselineReference,
    FailureCandidate,
    FailureCriterion,
    ProjectTarget,
    RunEvaluation,
    TargetSpec,
)


def test_project_target_accepts_full_init_context() -> None:
    """`pmem init` should be able to validate objective/metric/target context."""

    target = ProjectTarget(
        current_objective="Build AG News classifier CPU-friendly enough to dogfood pmem.",
        primary_metric="accuracy",
        metric_direction=MetricDirection.MAX,
        target=TargetSpec(
            target_value=0.90,
            baseline=BaselineReference(
                label="logistic regression",
                metric_name="accuracy",
                metric_value=0.86,
            ),
            constraints={"runtime_sec_max": 600, "cpu_only": True},
        ),
        failure_criteria=[
            FailureCriterion(
                expression="accuracy < baseline",
                candidate_kind=FailureCandidateKind.BASELINE_REGRESSION,
                tag="Convergence",
            )
        ],
    )

    assert target.metric_direction == MetricDirection.MAX
    assert target.target.target_value == 0.90
    assert target.failure_criteria[0].tag == "convergence"


def test_target_value_requires_metric_name_and_direction() -> None:
    """A numeric target is not meaningful without metric context."""

    with pytest.raises(ValidationError, match="primary_metric is required"):
        ProjectTarget(target=TargetSpec(target_value=0.90), metric_direction=MetricDirection.MAX)

    with pytest.raises(ValidationError, match="metric_direction is required"):
        ProjectTarget(primary_metric="accuracy", target=TargetSpec(target_value=0.90))


def test_metric_direction_only_accepts_max_or_min() -> None:
    """The direction vocabulary should stay exactly `max` or `min`."""

    parsed = ProjectTarget.model_validate({"primary_metric": "loss", "metric_direction": "min"})
    assert parsed.metric_direction == MetricDirection.MIN

    with pytest.raises(ValidationError):
        ProjectTarget.model_validate(
            {"primary_metric": "accuracy", "metric_direction": "higher_better"}
        )


def test_target_and_baseline_values_must_be_finite() -> None:
    """NaN/inf would make rule-based comparisons unreliable."""

    with pytest.raises(ValidationError, match="target_value must be finite"):
        TargetSpec(target_value=float("nan"))

    with pytest.raises(ValidationError, match="baseline metric_value must be finite"):
        BaselineReference(metric_value=float("inf"))


def test_run_evaluation_separates_technical_status_from_research_failure() -> None:
    """A successful command can still miss target/baseline as a candidate."""

    evaluation = RunEvaluation(
        technical_status=RunStatus.SUCCESS,
        primary_metric="accuracy",
        primary_metric_value=0.83,
        target_value=0.90,
        baseline_value=0.86,
        flags=[EvaluationFlag.TARGET_MISSED, EvaluationFlag.BASELINE_REGRESSED],
    )

    assert evaluation.technical_status == RunStatus.SUCCESS
    assert EvaluationFlag.TARGET_MISSED in evaluation.flags
    assert EvaluationFlag.BASELINE_REGRESSED in evaluation.flags


def test_metric_value_requires_metric_name() -> None:
    """A metric value without its metric key cannot be interpreted later."""

    with pytest.raises(ValidationError, match="primary_metric is required"):
        RunEvaluation(primary_metric_value=0.83)


def test_candidate_evidence_is_required() -> None:
    """Candidates must explain why the user should review them."""

    with pytest.raises(ValidationError, match="failure candidate evidence cannot be blank"):
        FailureCandidate(kind=FailureCandidateKind.TARGET_MISS, evidence=" ")
