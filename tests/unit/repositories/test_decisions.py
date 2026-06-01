"""Tests for the decision repository."""

import json

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import apply_migrations
from pmem.repositories.decisions import DecisionRepository
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.projects import ProjectRepository
from pmem.repositories.sqlite import connect_database

NOW = "2026-05-18T00:00:00Z"


@pytest.fixture()
def repository(tmp_path):
    """Return a decision repository backed by one project and experiment."""

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
        yield DecisionRepository(connection)
    finally:
        connection.close()


def test_create_and_read_decision(repository: DecisionRepository) -> None:
    """A decision should round-trip with deterministic related-experiment JSON."""

    record = repository.create(
        decision_id="dec_1",
        project_id="proj_1",
        experiment_id="exp_1",
        description="Use linear baseline first.",
        rationale="Cheaper and easier to inspect.",
        related_experiments=["exp_1"],
        created_at=NOW,
        author="owner-a",
    )

    assert repository.get_by_id("dec_1") == record
    assert repository.list_for_project("proj_1") == (record,)
    assert json.loads(record.related_experiments_json) == ["exp_1"]


def test_orphan_project_is_rejected(repository: DecisionRepository) -> None:
    """Decisions must reference an existing project."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(
            decision_id="dec_orphan",
            project_id="missing",
            description="bad",
            created_at=NOW,
        )
