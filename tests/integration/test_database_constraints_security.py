"""database database constraints and security-oriented tests."""

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import apply_migrations
from pmem.repositories.sqlite import connect_database, execute, query_one

NOW = "2026-05-15T00:00:00Z"
HASH = "a" * 64


@pytest.fixture()
def migrated_connection(tmp_path):
    """Return a migrated SQLite connection for constraint tests."""

    db_path = tmp_path / "pmem.db"
    apply_migrations(db_path)
    connection = connect_database(db_path)
    try:
        yield connection
    finally:
        connection.close()


def _insert_project(connection, project_id: str = "proj_1", name: str = "demo") -> None:
    execute(
        connection,
        """
        INSERT INTO projects(id, name, target_json, failure_criteria_json, created_at, updated_at,
                             metadata_json)
        VALUES (?, ?, '{}', '[]', ?, ?, '{}')
        """,
        (project_id, name, NOW, NOW),
    )
    connection.commit()


def _insert_experiment(
    connection,
    experiment_id: str = "exp_1",
    project_id: str = "proj_1",
    name: str = "baseline",
    *,
    is_baseline: int = 0,
    status: str = "active",
) -> None:
    execute(
        connection,
        """
        INSERT INTO experiments(id, project_id, name, status, is_baseline, created_at, updated_at,
                                metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, '{}')
        """,
        (experiment_id, project_id, name, status, is_baseline, NOW, NOW),
    )
    connection.commit()


def _insert_run(connection) -> None:
    execute(
        connection,
        """
        INSERT INTO runs(run_id, experiment_id, command, cwd, exit_code, status, duration_sec,
                         timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("run_1", "exp_1", "python train.py", "/tmp/project", 0, "success", 1.2, NOW),
    )
    connection.commit()


def test_foreign_keys_reject_orphan_experiment(migrated_connection) -> None:
    """Experiment rows must reference an existing project."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        _insert_experiment(migrated_connection, project_id="missing_project")


def test_invalid_project_status_is_rejected(migrated_connection) -> None:
    """Database CHECK constraints should protect enum state."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        execute(
            migrated_connection,
            """
            INSERT INTO projects(id, name, status, target_json, failure_criteria_json, created_at,
                                 updated_at, metadata_json)
            VALUES (?, ?, ?, '{}', '[]', ?, ?, '{}')
            """,
            ("proj_bad", "bad", "deleted", NOW, NOW),
        )


def test_invalid_json_is_rejected(migrated_connection) -> None:
    """Required JSON columns should reject malformed payloads."""

    with pytest.raises(PmemPersistenceError):
        execute(
            migrated_connection,
            """
            INSERT INTO projects(id, name, target_json, failure_criteria_json, created_at,
                                 updated_at, metadata_json)
            VALUES (?, ?, ?, '[]', ?, ?, '{}')
            """,
            ("proj_bad_json", "bad-json", "{not-json", NOW, NOW),
        )


def test_duplicate_experiment_name_is_rejected_per_project(migrated_connection) -> None:
    """Experiment names are unique inside one project."""

    _insert_project(migrated_connection)
    _insert_experiment(migrated_connection, experiment_id="exp_1", name="same")

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        _insert_experiment(migrated_connection, experiment_id="exp_2", name="same")


def test_only_one_active_baseline_per_project(migrated_connection) -> None:
    """The partial unique baseline index keeps summary deterministic."""

    _insert_project(migrated_connection)
    _insert_experiment(
        migrated_connection,
        experiment_id="exp_1",
        name="baseline-1",
        is_baseline=1,
    )

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        _insert_experiment(
            migrated_connection,
            experiment_id="exp_2",
            name="baseline-2",
            is_baseline=1,
        )

    _insert_experiment(
        migrated_connection,
        experiment_id="exp_3",
        name="old-baseline",
        is_baseline=1,
        status="abandoned",
    )


def test_success_run_cannot_have_nonzero_exit_code(migrated_connection) -> None:
    """Technical status cannot contradict exit code at DB level."""

    _insert_project(migrated_connection)
    _insert_experiment(migrated_connection)

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        execute(
            migrated_connection,
            """
            INSERT INTO runs(run_id, experiment_id, command, cwd, exit_code, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("run_bad", "exp_1", "python train.py", "/tmp/project", 1, "success", NOW),
        )


def test_tracked_path_hash_and_duplicate_constraints(migrated_connection) -> None:
    """Tracked paths require SHA-256 hashes and unique paths per project."""

    _insert_project(migrated_connection)

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        execute(
            migrated_connection,
            """
            INSERT INTO tracked_paths(id, project_id, path, hash, last_checked, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("track_bad", "proj_1", "data.csv", "not-a-hash", NOW, NOW),
        )

    execute(
        migrated_connection,
        """
        INSERT INTO tracked_paths(id, project_id, path, hash, last_checked, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("track_1", "proj_1", "data.csv", HASH, NOW, NOW),
    )

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        execute(
            migrated_connection,
            """
            INSERT INTO tracked_paths(id, project_id, path, hash, last_checked, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("track_2", "proj_1", "data.csv", HASH, NOW, NOW),
        )


def test_failure_constraints_reject_invalid_severity_and_source(migrated_connection) -> None:
    """Failure severity/source are protected at DB level."""

    _insert_project(migrated_connection)
    _insert_experiment(migrated_connection)
    _insert_run(migrated_connection)

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        execute(
            migrated_connection,
            """
            INSERT INTO failures(id, run_id, error_type, description, severity, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("fail_1", "run_1", "MetricRegression", "bad", "urgent", "unknown", NOW),
        )


def test_sql_injection_like_input_is_stored_as_data(migrated_connection) -> None:
    """Parameterized writes store dangerous-looking input without executing it."""

    malicious_name = "'; DROP TABLE projects; --"
    _insert_project(migrated_connection, project_id="proj_injection", name=malicious_name)

    row = query_one(
        migrated_connection,
        "SELECT name FROM projects WHERE id = ?",
        ("proj_injection",),
    )
    table_row = query_one(
        migrated_connection,
        "SELECT name FROM sqlite_master WHERE type = ? AND name = ?",
        ("table", "projects"),
    )

    assert row is not None
    assert row["name"] == malicious_name
    assert table_row is not None


def test_raw_database_error_does_not_leak_sql_or_input(migrated_connection) -> None:
    """Public persistence errors should not echo SQL or dangerous input."""

    dangerous = "' OR '1'='1"

    with pytest.raises(PmemPersistenceError) as exc_info:
        execute(
            migrated_connection,
            "INSERT INTO missing_table(value) VALUES (?)",
            (dangerous,),
        )

    public_message = str(exc_info.value)
    assert "missing_table" not in public_message
    assert dangerous not in public_message
    assert public_message == "Database operation failed."
