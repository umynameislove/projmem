"""Tests for the confirmed-failure repository."""

import json

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import apply_migrations
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.projects import ProjectRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database

NOW = "2026-05-18T00:00:00Z"


@pytest.fixture()
def repository(tmp_path):
    """Return a failure repository backed by one migrated run."""

    db_path = tmp_path / "pmem.db"
    apply_migrations(db_path)
    connection = connect_database(db_path)
    try:
        ProjectRepository(connection).create(
            project_id="proj_1",
            name="demo",
            created_at=NOW,
            updated_at=NOW,
        )
        ExperimentRepository(connection).create(
            experiment_id="exp_1",
            project_id="proj_1",
            name="default",
            created_at=NOW,
            updated_at=NOW,
        )
        RunRepository(connection).create(
            run_id="run_1",
            experiment_id="exp_1",
            command="python train.py",
            cwd=".",
            exit_code=1,
            status="failed",
            timestamp=NOW,
        )
        yield FailureRepository(connection)
    finally:
        connection.close()


def test_create_and_read_failure(repository: FailureRepository) -> None:
    """A confirmed failure should round-trip with deterministic tag JSON."""

    record = repository.create(
        failure_id="fail_1",
        run_id="run_1",
        error_type="MetricRegression",
        description="Accuracy dropped below target.",
        root_cause="Bad config",
        lesson="Check learning rate",
        severity="high",
        tags=["config_error", "convergence"],
        source="user_confirmed",
        created_at=NOW,
    )

    assert repository.get_by_id("fail_1") == record
    assert repository.list_for_run("run_1") == (record,)
    assert repository.list_for_project("proj_1") == (record,)
    assert json.loads(record.tags_json) == ["config_error", "convergence"]


def test_orphan_run_is_rejected(repository: FailureRepository) -> None:
    """Failures must reference an existing run."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(
            failure_id="fail_orphan",
            run_id="missing",
            error_type="ValueError",
            description="bad",
            root_cause=None,
            lesson=None,
            severity="medium",
            tags=[],
            source="user_confirmed",
            created_at=NOW,
        )


def test_invalid_severity_is_rejected(repository: FailureRepository) -> None:
    """The DB severity CHECK protects invalid repository writes."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(
            failure_id="fail_bad",
            run_id="run_1",
            error_type="ValueError",
            description="bad",
            root_cause=None,
            lesson=None,
            severity="urgent",
            tags=[],
            source="user_confirmed",
            created_at=NOW,
        )
