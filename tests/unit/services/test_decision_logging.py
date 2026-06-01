"""Service tests for `pmem log-decision`."""

import json
import sys

import pytest

from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.repositories.sqlite import connect_database
from pmem.services.decision_logging import log_decision
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command


def test_log_decision_persists_project_decision(tmp_path) -> None:
    """decision logging should store durable project decisions."""

    init_project(tmp_path, project_name="demo")
    record = log_decision(
        tmp_path,
        description="Use logistic regression as the first baseline.",
        rationale="It is CPU-friendly and easy to inspect.",
        author="owner-a",
    )

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        row = connection.execute(
            "SELECT description, rationale, author, related_experiments_json FROM decisions"
        ).fetchone()
    finally:
        connection.close()

    assert record.id.startswith("dec_")
    assert row["description"] == "Use logistic regression as the first baseline."
    assert row["rationale"] == "It is CPU-friendly and easy to inspect."
    assert row["author"] == "owner-a"
    assert json.loads(row["related_experiments_json"]) == []


def test_log_decision_accepts_current_project_experiment_link(tmp_path) -> None:
    """Experiment links should be validated against the current project."""

    init_project(tmp_path, project_name="demo")
    run = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    record = log_decision(
        tmp_path,
        description="Keep default experiment as baseline.",
        experiment_id=run.record.experiment_id,
        related_experiments=(run.record.experiment_id,),
    )

    assert record.experiment_id == run.record.experiment_id


def test_log_decision_requires_initialized_project(tmp_path) -> None:
    """Decision logging must not create implicit project state."""

    with pytest.raises(PmemNotFoundError, match="pmem init"):
        log_decision(tmp_path, description="Use baseline.")


def test_log_decision_rejects_missing_experiment_link(tmp_path) -> None:
    """Experiment links should fail cleanly when they do not exist."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemNotFoundError, match="Experiment was not found"):
        log_decision(
            tmp_path,
            description="Use baseline.",
            experiment_id="exp_missing",
        )


def test_log_decision_rejects_blank_related_experiment(tmp_path) -> None:
    """Related experiment ids are optional but cannot be blank."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemValidationError, match="cannot be blank"):
        log_decision(
            tmp_path,
            description="Use baseline.",
            related_experiments=("   ",),
        )


def test_log_decision_rejects_blank_optional_text(tmp_path) -> None:
    """Optional rationale/author values cannot be whitespace-only."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemValidationError, match="cannot be blank"):
        log_decision(tmp_path, description="Use baseline.", rationale="  ")
