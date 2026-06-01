"""Tests for the project repository used by `pmem init`."""

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import apply_migrations
from pmem.repositories.projects import ProjectRepository
from pmem.repositories.sqlite import connect_database

NOW = "2026-05-15T00:00:00Z"


@pytest.fixture()
def repository(tmp_path):
    """Return a project repository backed by migrated SQLite."""

    db_path = tmp_path / "pmem.db"
    apply_migrations(db_path)
    connection = connect_database(db_path)
    try:
        yield ProjectRepository(connection)
    finally:
        connection.close()


def test_create_and_read_project_by_id_and_name(repository: ProjectRepository) -> None:
    """A created project should be readable through both lookup paths."""

    record = repository.create(
        project_id="proj_1",
        name="demo",
        created_at=NOW,
        updated_at=NOW,
    )

    assert record.id == "proj_1"
    assert repository.get_by_id("proj_1") == record
    assert repository.get_by_name("demo") == record


def test_create_project_with_init_metadata(repository: ProjectRepository) -> None:
    """project init init flags should persist into existing schema v1 columns."""

    record = repository.create(
        project_id="proj_1",
        name="demo",
        created_at=NOW,
        updated_at=NOW,
        goal="Win AG News",
        current_objective="Train baseline",
        primary_metric="accuracy",
        metric_direction="max",
        target={"target_value": 0.9},
    )

    assert record.goal == "Win AG News"
    assert record.current_objective == "Train baseline"
    assert record.primary_metric == "accuracy"
    assert record.metric_direction == "max"
    assert record.target_json == '{"target_value":0.9}'
    assert repository.get_by_id("proj_1") == record


def test_duplicate_project_name_is_rejected(repository: ProjectRepository) -> None:
    """The DB unique constraint should prevent duplicate project names."""

    repository.create(project_id="proj_1", name="demo", created_at=NOW, updated_at=NOW)

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(project_id="proj_2", name="demo", created_at=NOW, updated_at=NOW)

    assert len(repository.list_projects()) == 1


def test_project_not_null_and_blank_constraints_are_enforced(
    repository: ProjectRepository,
) -> None:
    """Required project fields should be protected by DB constraints."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(project_id="", name="demo", created_at=NOW, updated_at=NOW)

    assert repository.list_projects() == ()


def test_sql_injection_like_project_name_is_stored_as_data(
    repository: ProjectRepository,
) -> None:
    """Repository writes must parameterize user-controlled project names."""

    dangerous_name = "'; DROP TABLE projects; --"

    record = repository.create(
        project_id="proj_injection",
        name=dangerous_name,
        created_at=NOW,
        updated_at=NOW,
    )

    assert repository.get_by_name(dangerous_name) == record
    assert repository.get_by_id("proj_injection") == record


def test_project_create_failure_rolls_back_transaction(repository: ProjectRepository) -> None:
    """A failed insert must not leave partial project state."""

    repository.create(project_id="proj_1", name="demo", created_at=NOW, updated_at=NOW)

    with pytest.raises(PmemPersistenceError):
        repository.create(project_id="proj_2", name="demo", created_at=NOW, updated_at=NOW)

    assert [project.id for project in repository.list_projects()] == ["proj_1"]
