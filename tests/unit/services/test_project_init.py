"""Service tests for the `pmem init` workflow."""

import json

import pytest

from pmem.errors import PmemConflictError, PmemValidationError
from pmem.repositories.projects import ProjectRepository
from pmem.repositories.sqlite import connect_database
from pmem.services.config import (
    ProjectConfig,
    project_config_path,
    read_project_config,
    write_project_config_if_missing,
)
from pmem.services.database import ensure_database
from pmem.services.project_init import init_project, validate_init_metadata, validate_project_name


def test_init_project_creates_local_state_and_project_row(tmp_path) -> None:
    """First init should create folders, DB, config, migration, and project row."""

    result = init_project(tmp_path, project_name="demo")

    assert result.already_initialized is False
    assert result.pmem_dir == tmp_path / ".pmem"
    assert result.db_path == tmp_path / ".pmem" / "pmem.db"
    assert result.config_path == tmp_path / ".pmem" / "config.yaml"
    assert result.artifacts_dir == tmp_path / ".pmem" / "artifacts"
    assert result.snapshots_dir == tmp_path / ".pmem" / "snapshots"
    assert result.pmem_dir.is_dir()
    assert result.db_path.is_file()
    assert result.config_path.is_file()
    assert result.artifacts_dir.is_dir()
    assert result.snapshots_dir.is_dir()
    assert result.migration_result.applied_versions == (
        "0001_schema_v1",
        "0002_phase2_portability",
    )

    connection = connect_database(result.db_path)
    try:
        row = connection.execute("SELECT id, name FROM projects").fetchone()
        migration_count = connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
    finally:
        connection.close()

    assert row["id"] == result.project_id
    assert row["name"] == "demo"
    assert migration_count == 2


def test_init_project_persists_optional_context_flags(tmp_path) -> None:
    """project init init flags should populate schema v1 project context columns."""

    result = init_project(
        tmp_path,
        project_name="demo",
        goal="Improve classifier",
        current_objective="Train CPU baseline",
        primary_metric="accuracy",
        metric_direction="max",
        target_value=0.9,
    )

    connection = connect_database(result.db_path)
    try:
        row = connection.execute(
            """
            SELECT goal, current_objective, primary_metric, metric_direction, target_json
            FROM projects
            WHERE id = ?
            """,
            (result.project_id,),
        ).fetchone()
    finally:
        connection.close()

    assert row["goal"] == "Improve classifier"
    assert row["current_objective"] == "Train CPU baseline"
    assert row["primary_metric"] == "accuracy"
    assert row["metric_direction"] == "max"
    assert json.loads(row["target_json"])["target_value"] == 0.9


def test_init_project_is_idempotent_and_preserves_config(tmp_path) -> None:
    """Repeated init must not change project identity or overwrite config."""

    first = init_project(tmp_path, project_name="demo")
    config_path = project_config_path(tmp_path)
    original_config_text = config_path.read_text(encoding="utf-8")
    marker = tmp_path / ".pmem" / "artifacts" / "keep.txt"
    marker.write_text("keep me\n", encoding="utf-8")

    second = init_project(tmp_path)

    assert second.already_initialized is True
    assert second.project_id == first.project_id
    assert second.project_name == first.project_name
    assert config_path.read_text(encoding="utf-8") == original_config_text
    assert marker.read_text(encoding="utf-8") == "keep me\n"

    connection = connect_database(second.db_path)
    try:
        project_count = connection.execute("SELECT count(*) FROM projects").fetchone()[0]
    finally:
        connection.close()

    assert project_count == 1


def test_init_project_rejects_name_change_after_init(tmp_path) -> None:
    """Init must not silently change project identity after config exists."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemConflictError, match="different name"):
        init_project(tmp_path, project_name="other")


def test_init_project_rejects_context_change_after_init(tmp_path) -> None:
    """Repeated init should not silently mutate stored project context."""

    init_project(tmp_path, project_name="demo", current_objective="baseline")

    with pytest.raises(PmemConflictError, match="different objective"):
        init_project(tmp_path, current_objective="new objective")


def test_init_project_keeps_sql_injection_like_name_as_data(tmp_path) -> None:
    """Project names should be parameterized, not executed as SQL."""

    dangerous_name = "' OR '1'='1"

    result = init_project(tmp_path, project_name=dangerous_name)
    config = read_project_config(result.config_path)

    assert result.project_name == dangerous_name
    assert config.project_name == dangerous_name


def test_init_project_database_integrity_checks_pass(tmp_path) -> None:
    """The database migration integrity guarantees should still hold after init."""

    result = init_project(tmp_path, project_name="demo")
    connection = connect_database(result.db_path)
    try:
        foreign_key_enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()

    assert foreign_key_enabled == 1
    assert foreign_key_rows == []
    assert integrity == "ok"


def test_init_project_uses_existing_single_project_when_config_is_missing(tmp_path) -> None:
    """Init should recover config from an existing single-project database."""

    first = init_project(tmp_path, project_name="demo")
    project_config_path(tmp_path).unlink()

    second = init_project(tmp_path)

    assert second.project_id == first.project_id
    assert second.already_initialized is False
    assert read_project_config(second.config_path).project_id == first.project_id


def test_init_project_recreates_missing_project_row_from_config(tmp_path) -> None:
    """A config-only project identity should be restored into SQLite."""

    ensure_database(tmp_path)
    config_path = project_config_path(tmp_path)
    write_project_config_if_missing(
        config_path,
        ProjectConfig(
            version=1,
            project_id="proj_config_only",
            project_name="demo",
            created_at="2026-05-15T00:00:00Z",
        ),
    )

    result = init_project(tmp_path)

    assert result.project_id == "proj_config_only"
    connection = connect_database(result.db_path)
    try:
        row = connection.execute(
            "SELECT id FROM projects WHERE id = ?", (result.project_id,)
        ).fetchone()
    finally:
        connection.close()

    assert row["id"] == "proj_config_only"


def test_init_project_rejects_multiple_project_rows_without_config(tmp_path) -> None:
    """Single-project local-memory should fail closed if DB state is ambiguous."""

    first = init_project(tmp_path, project_name="demo")
    project_config_path(tmp_path).unlink()
    connection = connect_database(first.db_path)
    try:
        ProjectRepository(connection).create(
            project_id="proj_2",
            name="other",
            created_at="2026-05-15T00:00:00Z",
            updated_at="2026-05-15T00:00:00Z",
        )
    finally:
        connection.close()

    with pytest.raises(PmemConflictError, match="Multiple project records"):
        init_project(tmp_path)


def test_init_project_rejects_name_change_when_single_row_has_no_config(tmp_path) -> None:
    """A missing config should not allow project name mutation."""

    init_project(tmp_path, project_name="demo")
    project_config_path(tmp_path).unlink()

    with pytest.raises(PmemConflictError, match="different name"):
        init_project(tmp_path, project_name="other")


def test_validate_project_name_rejects_unsafe_names() -> None:
    """Blank, oversized, and control-character names should fail before DB writes."""

    with pytest.raises(PmemValidationError, match="blank"):
        validate_project_name(" ")
    with pytest.raises(PmemValidationError, match="too long"):
        validate_project_name("x" * 121)
    with pytest.raises(PmemValidationError, match="control"):
        validate_project_name("bad\nname")


def test_validate_init_metadata_requires_metric_context_for_target() -> None:
    """A numeric target is not meaningful without metric name and direction."""

    with pytest.raises(PmemValidationError, match="requires metric"):
        validate_init_metadata(
            goal=None,
            current_objective=None,
            primary_metric=None,
            metric_direction=None,
            target_value=0.9,
        )
    with pytest.raises(PmemValidationError, match="max"):
        validate_init_metadata(
            goal=None,
            current_objective=None,
            primary_metric="accuracy",
            metric_direction="up",
            target_value=None,
        )


def test_init_project_error_message_does_not_expose_sqlite_details(tmp_path) -> None:
    """Expected init errors should be app-level messages."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemConflictError) as exc_info:
        init_project(tmp_path, project_name="other")

    message = str(exc_info.value)
    assert "sqlite" not in message.lower()
    assert "SELECT" not in message
    assert str(tmp_path) not in message
