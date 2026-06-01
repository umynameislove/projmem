"""Tests for the experiment repository used by `pmem run`."""

import json

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import apply_migrations
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.projects import ProjectRepository
from pmem.repositories.sqlite import connect_database

NOW = "2026-05-15T00:00:00Z"


@pytest.fixture()
def repository(tmp_path):
    """Return an experiment repository backed by migrated SQLite."""

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
        yield ExperimentRepository(connection)
    finally:
        connection.close()


def test_get_or_create_default_experiment_is_idempotent(
    repository: ExperimentRepository,
) -> None:
    """run capture should create exactly one default experiment per project."""

    first = repository.get_or_create_default(project_id="proj_1", timestamp=NOW)
    second = repository.get_or_create_default(project_id="proj_1", timestamp=NOW)

    assert first == second
    assert first.name == "default"
    assert repository.get_by_project_and_name("proj_1", "default") == first


def test_get_or_create_default_handles_duplicate_insert_race(
    repository: ExperimentRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent default insert should fall back to the existing row."""

    existing = repository.create(
        experiment_id="exp_default_proj_1",
        project_id="proj_1",
        name="default",
        created_at=NOW,
        updated_at=NOW,
    )
    lookup_results = iter([None, existing])

    def racing_lookup(project_id: str, name: str):
        assert project_id == "proj_1"
        assert name == "default"
        return next(lookup_results)

    monkeypatch.setattr(repository, "get_by_project_and_name", racing_lookup)

    result = repository.get_or_create_default(project_id="proj_1", timestamp=NOW)
    row_count = repository._connection.execute(
        "SELECT COUNT(*) FROM experiments WHERE project_id = ? AND name = ?",
        ("proj_1", "default"),
    ).fetchone()[0]

    assert result == existing
    assert row_count == 1


def test_create_and_read_experiment(repository: ExperimentRepository) -> None:
    """Explicit experiment rows should round-trip through id and name lookups."""

    record = repository.create(
        experiment_id="exp_1",
        project_id="proj_1",
        name="baseline",
        hypothesis="try small classifier",
        created_at=NOW,
        updated_at=NOW,
        primary_metric="accuracy",
        target={"target_value": 0.9},
    )

    assert repository.get_by_id("exp_1") == record
    assert repository.get_by_project_and_name("proj_1", "baseline") == record
    assert repository.list_for_project("proj_1") == (record,)
    assert record.target_json == '{"target_value":0.9}'


def test_update_experiment_metadata(repository: ExperimentRepository) -> None:
    """baseline should persist baseline metadata without schema changes."""

    repository.create(
        experiment_id="exp_1",
        project_id="proj_1",
        name="baseline",
        created_at=NOW,
        updated_at=NOW,
    )

    updated = repository.update_metadata(
        experiment_id="exp_1",
        metadata={"baseline_run_id": "run_1"},
        updated_at="2026-05-18T00:00:00Z",
    )

    assert json.loads(updated.metadata_json) == {"baseline_run_id": "run_1"}
    assert updated.updated_at == "2026-05-18T00:00:00Z"


def test_duplicate_experiment_name_is_rejected(
    repository: ExperimentRepository,
) -> None:
    """The DB unique constraint should protect default experiment idempotency."""

    repository.create(
        experiment_id="exp_1",
        project_id="proj_1",
        name="same",
        created_at=NOW,
        updated_at=NOW,
    )

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(
            experiment_id="exp_2",
            project_id="proj_1",
            name="same",
            created_at=NOW,
            updated_at=NOW,
        )


def test_orphan_project_id_is_rejected(repository: ExperimentRepository) -> None:
    """Experiments must reference an existing project."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(
            experiment_id="exp_orphan",
            project_id="missing",
            name="bad",
            created_at=NOW,
            updated_at=NOW,
        )


def test_sql_injection_like_experiment_name_is_stored_as_data(
    repository: ExperimentRepository,
) -> None:
    """Dangerous-looking experiment names should not execute SQL."""

    dangerous = "default'; DROP TABLE experiments; --"

    record = repository.create(
        experiment_id="exp_injection",
        project_id="proj_1",
        name=dangerous,
        created_at=NOW,
        updated_at=NOW,
    )

    assert repository.get_by_project_and_name("proj_1", dangerous) == record
