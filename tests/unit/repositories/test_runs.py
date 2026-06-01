"""Tests for the run repository used by `pmem run`."""

import json

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import apply_migrations
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.projects import ProjectRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database

NOW = "2026-05-15T00:00:00Z"
HASH = "a" * 64


@pytest.fixture()
def repository(tmp_path):
    """Return a run repository backed by migrated SQLite."""

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
        yield RunRepository(connection)
    finally:
        connection.close()


def test_create_and_read_run(repository: RunRepository) -> None:
    """A captured run should round-trip with deterministic JSON payloads."""

    record = repository.create(
        run_id="run_1",
        experiment_id="exp_1",
        name="baseline",
        command="python train.py",
        cwd=".",
        exit_code=0,
        status="success",
        duration_sec=1.2,
        seed="13",
        stdout_path=".pmem/artifacts/runs/run_1/stdout.txt",
        stderr_path=".pmem/artifacts/runs/run_1/stderr.txt",
        stdout_preview="ok",
        stderr_preview="",
        env={"python_version": "3.10.0"},
        config={"lr": 0.1},
        config_hash=HASH,
        metrics={"accuracy": 0.9},
        artifacts=[{"path": "model.bin", "sha256": HASH, "size_bytes": 12}],
        git={},
        timestamp=NOW,
    )

    assert repository.get_by_id("run_1") == record
    assert repository.list_for_experiment("exp_1") == (record,)
    assert repository.list_for_project("proj_1") == (record,)
    assert json.loads(record.metrics_json) == {"accuracy": 0.9}
    assert json.loads(record.artifacts_json)[0]["sha256"] == HASH


def test_update_run_evaluation(repository: RunRepository) -> None:
    """baseline should store baseline comparison in evaluation_json."""

    repository.create(
        run_id="run_1",
        experiment_id="exp_1",
        command="python train.py",
        cwd=".",
        exit_code=0,
        status="success",
        timestamp=NOW,
    )

    updated = repository.update_evaluation(
        run_id="run_1",
        evaluation={"baseline_comparison": {"baseline_run_id": "run_0"}},
    )

    assert json.loads(updated.evaluation_json) == {
        "baseline_comparison": {"baseline_run_id": "run_0"}
    }


def test_success_run_cannot_have_nonzero_exit_code(
    repository: RunRepository,
) -> None:
    """The schema success/exit-code invariant should be enforced."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(
            run_id="run_bad",
            experiment_id="exp_1",
            command="python train.py",
            cwd=".",
            exit_code=2,
            status="success",
            timestamp=NOW,
        )


def test_orphan_experiment_id_is_rejected(repository: RunRepository) -> None:
    """Runs must reference an existing experiment."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(
            run_id="run_orphan",
            experiment_id="missing",
            command="python train.py",
            cwd=".",
            exit_code=0,
            status="success",
            timestamp=NOW,
        )


def test_preview_length_constraint_is_enforced(repository: RunRepository) -> None:
    """SQLite should reject previews that exceed the project init 2048-char contract."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(
            run_id="run_long_preview",
            experiment_id="exp_1",
            command="python train.py",
            cwd=".",
            exit_code=0,
            status="success",
            stdout_preview="x" * 2049,
            timestamp=NOW,
        )


def test_invalid_config_hash_is_rejected(repository: RunRepository) -> None:
    """Config hashes must be lowercase SHA-256 values when present."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(
            run_id="run_bad_hash",
            experiment_id="exp_1",
            command="python train.py",
            cwd=".",
            exit_code=0,
            status="success",
            config_hash="not-a-hash",
            timestamp=NOW,
        )


def test_sql_injection_like_command_is_stored_as_data(
    repository: RunRepository,
) -> None:
    """Run commands should be parameterized, not executed as SQL."""

    dangerous = "python train.py'; DROP TABLE runs; --"

    record = repository.create(
        run_id="run_injection",
        experiment_id="exp_1",
        command=dangerous,
        cwd=".",
        exit_code=0,
        status="success",
        timestamp=NOW,
    )

    assert repository.get_by_id("run_injection") == record
