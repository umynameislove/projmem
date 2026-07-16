"""Tests for SQLite connection and safe execution helpers."""

import sqlite3

import pytest

from pmem.errors import PmemPersistenceError, PmemSecurityError
from pmem.repositories.sqlite import (
    connect_database,
    connect_database_readonly,
    execute,
    project_database_path,
    query_one,
)


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


def test_readonly_connection_cannot_write_or_change_file_metadata(tmp_path) -> None:
    db_path = tmp_path / "pmem.db"
    writable = connect_database(db_path)
    try:
        writable.execute("CREATE TABLE demo (id INTEGER PRIMARY KEY)")
        writable.commit()
    finally:
        writable.close()
    db_path.chmod(0o644)
    before = (db_path.read_bytes(), db_path.stat().st_mtime_ns, db_path.stat().st_mode)

    readonly = connect_database_readonly(db_path)
    try:
        assert readonly.execute("SELECT COUNT(*) FROM demo").fetchone()[0] == 0
        with pytest.raises(sqlite3.OperationalError):
            readonly.execute("INSERT INTO demo DEFAULT VALUES")
    finally:
        readonly.close()

    assert (db_path.read_bytes(), db_path.stat().st_mtime_ns, db_path.stat().st_mode) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["pmem.db"]


def test_readonly_connection_in_checkpointed_wal_mode_creates_no_sidecars(tmp_path) -> None:
    db_path = tmp_path / "pmem.db"
    writable = sqlite3.connect(db_path)
    try:
        assert writable.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writable.execute("CREATE TABLE demo (value TEXT)")
        writable.execute("INSERT INTO demo VALUES ('checkpointed')")
        writable.commit()
    finally:
        writable.close()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["pmem.db"]

    readonly = connect_database_readonly(db_path)
    try:
        assert readonly.execute("SELECT value FROM demo").fetchone()[0] == "checkpointed"
    finally:
        readonly.close()

    assert sorted(path.name for path in tmp_path.iterdir()) == ["pmem.db"]


def test_readonly_connection_rejects_active_wal_without_touching_sidecars(tmp_path) -> None:
    db_path = tmp_path / "pmem.db"
    writer = sqlite3.connect(db_path)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("CREATE TABLE demo (value TEXT)")
        writer.execute("INSERT INTO demo VALUES ('uncheckpointed')")
        writer.commit()
        before = {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)
            for path in tmp_path.iterdir()
        }

        with pytest.raises(PmemPersistenceError, match="active SQLite sidecar state"):
            connect_database_readonly(db_path)

        after = {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)
            for path in tmp_path.iterdir()
        }
        assert after == before
    finally:
        writer.close()
