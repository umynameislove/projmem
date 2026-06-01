"""database migration reliability tests."""

import sqlite3

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import apply_migrations
from pmem.migrations.schema_v1 import SCHEMA_V1, Migration
from pmem.migrations.schema_v2 import SCHEMA_V2
from pmem.repositories.sqlite import connect_database

EXPECTED_TABLES = {
    "schema_migrations",
    "projects",
    "experiments",
    "runs",
    "failures",
    "decisions",
    "notes",
    "tracked_paths",
    "export_packages",
    "import_jobs",
    "shared_paths",
    "audit_events",
}


def _table_names(db_path) -> set[str]:
    connection = connect_database(db_path)
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    finally:
        connection.close()
    return {row["name"] for row in rows}


def test_fresh_database_migrates_to_schema_v1(tmp_path) -> None:
    """Fresh DB migration should create all database tables and version row."""

    db_path = tmp_path / "pmem.db"
    result = apply_migrations(db_path)

    connection = connect_database(db_path)
    try:
        version_rows = connection.execute(
            "SELECT version, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()

    assert result.applied_versions == ("0001_schema_v1", "0002_phase2_portability")
    assert result.skipped_versions == ()
    assert result.backup_path is None
    assert EXPECTED_TABLES <= _table_names(db_path)
    assert [(row["version"], row["checksum"]) for row in version_rows] == [
        ("0001_schema_v1", SCHEMA_V1.checksum),
        ("0002_phase2_portability", SCHEMA_V2.checksum),
    ]
    assert integrity == "ok"
    assert foreign_key_rows == []


def test_migration_runner_skips_already_applied_version(tmp_path) -> None:
    """Re-running the migration should not create duplicate state."""

    db_path = tmp_path / "pmem.db"
    apply_migrations(db_path)

    result = apply_migrations(db_path)

    connection = connect_database(db_path)
    try:
        count = connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
    finally:
        connection.close()

    assert result.applied_versions == ()
    assert result.skipped_versions == ("0001_schema_v1", "0002_phase2_portability")
    assert result.backup_path is None
    assert count == 2


def test_existing_database_is_backed_up_before_pending_migration(tmp_path) -> None:
    """A non-empty DB should be copied before schema changes."""

    db_path = tmp_path / "pmem.db"
    raw_connection = sqlite3.connect(db_path)
    raw_connection.execute("CREATE TABLE preexisting (id TEXT)")
    raw_connection.commit()
    raw_connection.close()

    result = apply_migrations(db_path)

    assert result.backup_path is not None
    assert result.backup_path.exists()
    assert result.backup_path.read_bytes() != b""
    assert "preexisting" in _table_names(result.backup_path)
    assert EXPECTED_TABLES <= _table_names(db_path)


def test_existing_schema_v1_database_upgrades_to_portability(tmp_path) -> None:
    """Upgrade a local-memory DB without removing existing tables or data."""

    db_path = tmp_path / "pmem.db"
    apply_migrations(db_path, migrations=(SCHEMA_V1,))
    connection = connect_database(db_path)
    try:
        connection.execute(
            """
            INSERT INTO projects(id, name, created_at, updated_at)
            VALUES ('proj_existing', 'existing', '2026-05-21T00:00:00Z', '2026-05-21T00:00:00Z')
            """
        )
        connection.commit()
    finally:
        connection.close()

    result = apply_migrations(db_path)

    connection = connect_database(db_path)
    try:
        project_count = connection.execute("SELECT count(*) FROM projects").fetchone()[0]
        versions = [
            str(row["version"])
            for row in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]
        indexes = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()

    assert result.applied_versions == ("0002_phase2_portability",)
    assert result.backup_path is not None
    assert project_count == 1
    assert versions == ["0001_schema_v1", "0002_phase2_portability"]
    assert {
        "idx_export_packages_manifest_hash",
        "idx_import_jobs_source_hash",
        "idx_shared_paths_alias",
        "idx_audit_events_entity",
    } <= indexes
    assert integrity == "ok"
    assert foreign_key_rows == []


def test_failed_migration_rolls_back_partial_schema(tmp_path) -> None:
    """A failed migration must leave no partially created table."""

    db_path = tmp_path / "pmem.db"
    bad_migration = Migration(
        version="0001_bad",
        checksum="a" * 64,
        sql="""
        CREATE TABLE created_before_failure (id TEXT);
        INSERT INTO missing_table(id) VALUES ('x');
        """,
    )

    with pytest.raises(PmemPersistenceError, match="Migration failed"):
        apply_migrations(db_path, migrations=(bad_migration,), create_backup=False)

    assert "created_before_failure" not in _table_names(db_path)


def test_migration_checksum_mismatch_is_rejected(tmp_path) -> None:
    """Recorded checksum drift means local DB state is not trusted."""

    db_path = tmp_path / "pmem.db"
    apply_migrations(db_path)

    connection = connect_database(db_path)
    try:
        connection.execute(
            "UPDATE schema_migrations SET checksum = ? WHERE version = ?",
            ("b" * 64, "0001_schema_v1"),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PmemPersistenceError, match="checksum mismatch"):
        apply_migrations(db_path)
