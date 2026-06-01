"""Entity model tests for the entity schema schema contract."""

import pytest
from pydantic import ValidationError

from pmem.domain.common import FailureCandidateKind, FailureSource, MetricDirection, RunStatus
from pmem.domain.decision import Decision
from pmem.domain.experiment import Experiment
from pmem.domain.failure import Failure
from pmem.domain.note import Note
from pmem.domain.project import Project
from pmem.domain.run import Run
from pmem.domain.target import FailureCandidate, TargetSpec
from pmem.domain.tracked_path import TrackedPath


def test_project_entity_carries_target_columns() -> None:
    """Project should expose the entity schema target columns as typed fields."""

    project = Project(
        id="proj_1",
        name="ag-news-classifier",
        current_objective="Build AG News classifier",
        primary_metric="accuracy",
        metric_direction=MetricDirection.MAX,
        target=TargetSpec(target_value=0.9),
    )

    assert project.current_objective == "Build AG News classifier"
    assert project.primary_metric == "accuracy"
    assert project.target.target_value == 0.9


def test_experiment_entity_supports_baseline_and_target_override() -> None:
    """Experiment can mark the baseline and override project target context."""

    experiment = Experiment(
        id="exp_1",
        project_id="proj_1",
        name="baseline-logreg",
        is_baseline=True,
        primary_metric="accuracy",
        target=TargetSpec(target_value=0.86),
    )

    assert experiment.is_baseline is True
    assert experiment.target is not None


def test_run_entity_holds_evaluation_and_candidates() -> None:
    """Run stores candidates separately from confirmed Failure rows."""

    run = Run(
        run_id="run_1",
        experiment_id="exp_1",
        command="python train.py",
        cwd="/tmp/project",
        exit_code=0,
        status=RunStatus.SUCCESS,
        metrics={"accuracy": 0.83},
        failure_candidates=[
            FailureCandidate(
                kind=FailureCandidateKind.BASELINE_REGRESSION,
                evidence="accuracy 0.83 is below baseline 0.86",
            )
        ],
    )

    assert run.status == RunStatus.SUCCESS
    assert len(run.failure_candidates) == 1


def test_successful_run_rejects_nonzero_exit_code() -> None:
    """Technical status should not contradict the captured exit code."""

    with pytest.raises(ValidationError, match="status=success requires exit_code"):
        Run(
            run_id="run_2",
            experiment_id="exp_1",
            command="python train.py",
            cwd="/tmp/project",
            exit_code=1,
            status=RunStatus.SUCCESS,
        )


def test_failure_entity_records_source() -> None:
    """Confirmed failures should say how they entered the system."""

    failure = Failure(
        id="fail_1",
        run_id="run_1",
        error_type="FileNotFoundError",
        description="Metrics file missing.",
        source=FailureSource.PROMOTED_CANDIDATE,
    )

    assert failure.source == FailureSource.PROMOTED_CANDIDATE


def test_decision_and_note_keep_optional_context_links() -> None:
    """Decisions/notes can attach to experiment/run context without requiring it."""

    decision = Decision(
        id="dec_1",
        project_id="proj_1",
        experiment_id="exp_1",
        description="Use logistic regression as baseline.",
        rationale="Fast CPU baseline for dogfood.",
    )
    note = Note(
        id="note_1",
        project_id="proj_1",
        experiment_id="exp_1",
        run_id="run_1",
        content="Question: does macro F1 diverge from accuracy?",
        tags=["Open Question"],
    )

    assert decision.experiment_id == "exp_1"
    assert note.run_id == "run_1"
    assert note.tags == ["open_question"]


def test_decision_related_experiment_blank_id_is_rejected() -> None:
    """Related experiment ids are optional, but present ids must be meaningful."""

    with pytest.raises(ValidationError, match="related experiment ids cannot be blank"):
        Decision(
            id="dec_2",
            project_id="proj_1",
            description="Use small baseline.",
            related_experiments=["   "],
        )


def test_tracked_path_normalizes_tag_and_rejects_negative_size() -> None:
    """Tracked paths carry SHA-256 hashes and optional normalized tags."""

    tracked = TrackedPath(
        id="track_1",
        project_id="proj_1",
        path="data/ag_news.csv",
        hash="a" * 64,
        tag="Raw Data",
        size_bytes=123,
    )

    assert tracked.tag == "raw_data"

    with pytest.raises(ValidationError, match="size_bytes cannot be negative"):
        TrackedPath(
            id="track_2",
            project_id="proj_1",
            path="data/ag_news.csv",
            hash="a" * 64,
            size_bytes=-1,
        )
