"""Tests for SQLite connection and safe execution helpers."""

import sqlite3

import pytest

from pmem.errors import PmemPersistenceError, PmemSecurityError
from pmem.repositories.sqlite import connect_database, execute, project_database_path, query_one


def test_project_database_path_uses_project_local_pmem_dir(tmp_path) -> None:
    """The DB path should be derived from the project root, not hard-coded."""

    assert project_database_path(tmp_path) == tmp_path / ".pmem" / "pmem.db"


def test_project_database_path_rejects_blank_root() -> None:
    """Blank project roots are config mistakes."""

    with pytest.raises(PmemSecurityError, match="Project root cannot be blank"):
        project_database_path(" ")


def test_connect_database_creates_parent_and_enables_foreign_keys(tmp_path) -> None:
    """Every connection should enforce FK constraints."""

    db_path = tmp_path / "nested" / "pmem.db"
    connection = connect_database(db_path)

    try:
        fk_enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        connection.close()

    assert db_path.exists()
    assert fk_enabled == 1
    assert busy_timeout == 5000
    assert db_path.stat().st_mode & 0o777 == 0o600


def test_connect_database_rejects_directory_path(tmp_path) -> None:
    """A directory cannot be opened as the SQLite database file."""

    with pytest.raises(PmemSecurityError, match="Database path points to a directory"):
        connect_database(tmp_path)


def test_execute_maps_raw_sqlite_errors(tmp_path) -> None:
    """Raw DB errors should become safe app-level errors."""

    connection = connect_database(tmp_path / "pmem.db")
    try:
        with pytest.raises(PmemPersistenceError) as exc_info:
            execute(connection, "SELECT * FROM missing_table")
    finally:
        connection.close()

    assert str(exc_info.value) == "Database operation failed."
    assert isinstance(exc_info.value.__cause__, sqlite3.Error)


def test_query_one_uses_safe_execution_path(tmp_path) -> None:
    """query_one should return one row through the helper."""

    connection = connect_database(tmp_path / "pmem.db")
    try:
        execute(connection, "CREATE TABLE demo (id TEXT PRIMARY KEY, value TEXT)")
        execute(connection, "INSERT INTO demo(id, value) VALUES (?, ?)", ("id_1", "value"))
        row = query_one(connection, "SELECT value FROM demo WHERE id = ?", ("id_1",))
    finally:
        connection.close()

    assert row is not None
    assert row["value"] == "value"
