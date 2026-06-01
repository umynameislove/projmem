"""Additional domain validation edge cases."""

import pytest
from pydantic import ValidationError

from pmem.domain.decision import Decision
from pmem.domain.experiment import Experiment
from pmem.domain.failure import Failure
from pmem.domain.note import Note
from pmem.domain.project import Project
from pmem.domain.run import Run
from pmem.domain.target import FailureCriterion, RunEvaluation, TargetSpec
from pmem.domain.tracked_path import TrackedPath


@pytest.mark.parametrize(
    ("model", "kwargs", "message"),
    [
        (Project, {"id": " ", "name": "demo"}, "project id/name cannot be blank"),
        (Project, {"id": "proj_1", "name": "demo", "goal": " "}, "project text fields"),
        (Experiment, {"id": " ", "project_id": "proj_1", "name": "exp"}, "experiment id"),
        (
            Experiment,
            {"id": "exp_1", "project_id": "proj_1", "name": "exp", "hypothesis": " "},
            "experiment text",
        ),
        (Run, {"run_id": " ", "experiment_id": "exp_1", "command": "cmd", "cwd": "/tmp"}, "run_id"),
        (
            Failure,
            {"id": " ", "run_id": "run_1", "error_type": "E", "description": "desc"},
            "failure id",
        ),
        (
            Failure,
            {
                "id": "fail_1",
                "run_id": "run_1",
                "error_type": "E",
                "description": "desc",
                "lesson": " ",
            },
            "failure optional",
        ),
        (Decision, {"id": " ", "project_id": "proj_1", "description": "desc"}, "decision id"),
        (
            Decision,
            {"id": "dec_1", "project_id": "proj_1", "description": "desc", "author": " "},
            "decision optional",
        ),
        (Note, {"id": " ", "project_id": "proj_1", "content": "note"}, "note id"),
        (
            Note,
            {"id": "note_1", "project_id": "proj_1", "content": "note", "run_id": " "},
            "note optional",
        ),
        (
            TrackedPath,
            {"id": " ", "project_id": "proj_1", "path": "x", "hash": "a" * 64},
            "tracked path",
        ),
    ],
)
def test_blank_text_edges_are_rejected(model, kwargs, message) -> None:
    """Domain models should reject meaningless blank text."""

    with pytest.raises(ValidationError, match=message):
        model(**kwargs)


def test_run_duration_cannot_be_negative() -> None:
    """Run duration validation complements the DB CHECK constraint."""

    with pytest.raises(ValidationError, match="duration_sec cannot be negative"):
        Run(run_id="run_1", experiment_id="exp_1", command="cmd", cwd="/tmp", duration_sec=-0.1)


def test_failure_criterion_expression_cannot_be_blank() -> None:
    """Failure criteria need an expression that can be reviewed."""

    with pytest.raises(ValidationError, match="failure criterion expression cannot be blank"):
        FailureCriterion(expression=" ")


def test_run_evaluation_rejects_blank_metric_name() -> None:
    """Blank metric names cannot be used as stable JSON keys."""

    with pytest.raises(ValidationError, match="primary_metric cannot be blank"):
        RunEvaluation(primary_metric=" ")


def test_target_spec_rejects_infinite_target_value() -> None:
    """Infinite targets cannot be compared safely."""

    with pytest.raises(ValidationError, match="target_value must be finite"):
        TargetSpec(target_value=float("inf"))
