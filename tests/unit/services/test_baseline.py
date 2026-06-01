"""Service tests for baseline baseline tracking."""

import json

import pytest

from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.projects import ProjectRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database
from pmem.services.baseline import compare_run_to_baseline, set_baseline_run
from pmem.services.project_init import init_project

NOW = "2026-05-18T00:00:00Z"


def _create_project_runs(tmp_path) -> tuple[str, str]:
    """Create a project with a baseline run and a later comparable run."""

    init_project(tmp_path, project_name="demo")
    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        project = ProjectRepository(connection).list_projects()[0]
        experiment = ExperimentRepository(connection).get_or_create_default(
            project_id=project.id,
            timestamp=NOW,
        )
        runs = RunRepository(connection)
        runs.create(
            run_id="run_base",
            experiment_id=experiment.id,
            command="python train.py",
            cwd=".",
            exit_code=0,
            status="success",
            metrics={"accuracy": 0.8, "loss": 0.5, "passed": True},
            timestamp=NOW,
        )
        runs.create(
            run_id="run_new",
            experiment_id=experiment.id,
            command="python train.py",
            cwd=".",
            exit_code=0,
            status="success",
            metrics={"accuracy": 0.85, "loss": 0.4, "note": "ok"},
            timestamp=NOW,
        )
        return "run_base", "run_new"
    finally:
        connection.close()


def test_set_baseline_run_persists_experiment_metadata(tmp_path) -> None:
    """A run can be marked as experiment baseline without a new migration."""

    baseline_run_id, _ = _create_project_runs(tmp_path)

    result = set_baseline_run(tmp_path, run_id=baseline_run_id)

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        metadata = json.loads(
            connection.execute("SELECT metadata_json FROM experiments").fetchone()[0]
        )
    finally:
        connection.close()

    assert result.run_id == baseline_run_id
    assert result.metric_count == 2
    assert metadata["baseline_run_id"] == baseline_run_id
    assert metadata["baseline_metrics"] == {"accuracy": 0.8, "loss": 0.5}


def test_compare_run_to_baseline_persists_evaluation(tmp_path) -> None:
    """A later run can be compared and the comparison is stored on the run."""

    baseline_run_id, new_run_id = _create_project_runs(tmp_path)
    set_baseline_run(tmp_path, run_id=baseline_run_id)

    result = compare_run_to_baseline(tmp_path, run_id=new_run_id)

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        evaluation = json.loads(
            connection.execute(
                "SELECT evaluation_json FROM runs WHERE run_id = ?",
                (new_run_id,),
            ).fetchone()[0]
        )
    finally:
        connection.close()

    assert result.baseline_run_id == baseline_run_id
    assert result.metric_deltas == pytest.approx({"accuracy": 0.05, "loss": -0.1})
    assert evaluation["baseline_comparison"]["baseline_run_id"] == baseline_run_id
    assert evaluation["baseline_comparison"]["metric_deltas"] == pytest.approx(
        {"accuracy": 0.05, "loss": -0.1}
    )


def test_compare_run_requires_existing_baseline(tmp_path) -> None:
    """Comparison should fail cleanly before a baseline run is set."""

    _, new_run_id = _create_project_runs(tmp_path)

    with pytest.raises(PmemNotFoundError, match="No baseline run"):
        compare_run_to_baseline(tmp_path, run_id=new_run_id)


def test_set_baseline_rejects_missing_run(tmp_path) -> None:
    """Baseline assignment should fail cleanly for an unknown run id."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemNotFoundError, match="Run was not found"):
        set_baseline_run(tmp_path, run_id="run_missing")


def test_compare_rejects_baseline_from_different_experiment(tmp_path) -> None:
    """Baseline comparison should reject corrupted cross-experiment metadata."""

    _, new_run_id = _create_project_runs(tmp_path)
    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        project = ProjectRepository(connection).list_projects()[0]
        experiments = ExperimentRepository(connection)
        other_experiment = experiments.create(
            experiment_id="exp_other",
            project_id=project.id,
            name="other",
            created_at=NOW,
            updated_at=NOW,
        )
        RunRepository(connection).create(
            run_id="run_other",
            experiment_id=other_experiment.id,
            command="python train.py",
            cwd=".",
            exit_code=0,
            status="success",
            metrics={"accuracy": 0.1},
            timestamp=NOW,
        )
        current_run = RunRepository(connection).get_by_id(new_run_id)
        assert current_run is not None
        current_experiment_id = current_run.experiment_id
        experiments.update_metadata(
            experiment_id=current_experiment_id,
            metadata={"baseline_run_id": "run_other"},
            updated_at=NOW,
        )
    finally:
        connection.close()

    with pytest.raises(PmemValidationError, match="different experiment"):
        compare_run_to_baseline(tmp_path, run_id=new_run_id)


def test_baseline_requires_initialized_project(tmp_path) -> None:
    """Baseline commands must not create implicit project state."""

    with pytest.raises(PmemNotFoundError, match="pmem init"):
        set_baseline_run(tmp_path, run_id="run_missing")
