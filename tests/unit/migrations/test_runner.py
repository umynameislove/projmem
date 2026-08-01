"""Unit tests for migration runner internals."""

import dataclasses
import re
import sqlite3

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import (
    CURRENT_MIGRATIONS,
    SchemaState,
    _iter_sql_statements,
    _utc_now_iso,
    inspect_schema,
    verify_schema_current,
)


def test_iter_sql_statements_splits_complete_statements() -> None:
    """Migration scripts should be applied statement by statement."""

    statements = _iter_sql_statements(
        """
        CREATE TABLE one (id TEXT);
        CREATE TABLE two (id TEXT);
        """
    )

    assert len(statements) == 2
    assert statements[0].startswith("CREATE TABLE one")
    assert statements[1].startswith("CREATE TABLE two")


def test_iter_sql_statements_rejects_incomplete_sql() -> None:
    """Incomplete SQL should fail before a partial migration starts."""

    with pytest.raises(PmemPersistenceError, match="incomplete statement"):
        _iter_sql_statements("CREATE TABLE broken (")


def test_utc_now_iso_is_utc_second_precision() -> None:
    """Migration timestamps should be compact UTC ISO strings."""

    timestamp = _utc_now_iso()

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", timestamp)


def test_verify_schema_current_maps_corrupt_database_error(tmp_path) -> None:
    db_path = tmp_path / "pmem.db"
    db_path.write_bytes(b"not a sqlite database")
    connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        with pytest.raises(PmemPersistenceError) as exc_info:
            verify_schema_current(connection)
    finally:
        connection.close()

    assert str(exc_info.value) == "The project database could not be read."
    assert isinstance(exc_info.value.__cause__, sqlite3.Error)
    assert str(db_path) not in str(exc_info.value)
    assert "file is not a database" not in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Read-only schema inspection seam (DOC-002)                                   #
# --------------------------------------------------------------------------- #
def _migrated_connection(tmp_path):
    from pmem.migrations.runner import apply_migrations

    db_path = tmp_path / ".pmem" / "pmem.db"
    apply_migrations(db_path)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def test_inspect_schema_reports_current_for_a_migrated_database(tmp_path) -> None:
    connection = _migrated_connection(tmp_path)
    try:
        inspection = inspect_schema(connection)
    finally:
        connection.close()

    assert inspection.state is SchemaState.CURRENT
    assert inspection.missing_versions == ()
    assert inspection.mismatched_versions == ()
    assert inspection.unknown_versions == ()
    assert inspection.recorded_version_count == len(CURRENT_MIGRATIONS)


def test_inspect_schema_reports_missing_migrations(tmp_path) -> None:
    connection = _migrated_connection(tmp_path)
    try:
        connection.execute(
            "DELETE FROM schema_migrations WHERE version = ?", (CURRENT_MIGRATIONS[-1].version,)
        )
        connection.commit()
        inspection = inspect_schema(connection)
    finally:
        connection.close()

    assert inspection.state is SchemaState.MISSING_MIGRATIONS
    assert inspection.missing_versions == (CURRENT_MIGRATIONS[-1].version,)
    assert inspection.mismatched_versions == ()
    assert inspection.unknown_versions == ()


def test_inspect_schema_reports_checksum_mismatch(tmp_path) -> None:
    connection = _migrated_connection(tmp_path)
    try:
        connection.execute(
            "UPDATE schema_migrations SET checksum = ? WHERE version = ?",
            ("a" * 64, CURRENT_MIGRATIONS[0].version),
        )
        connection.commit()
        inspection = inspect_schema(connection)
    finally:
        connection.close()

    assert inspection.state is SchemaState.CHECKSUM_MISMATCH
    assert inspection.mismatched_versions == (CURRENT_MIGRATIONS[0].version,)
    assert inspection.unknown_versions == ()


def test_checksum_mismatch_outranks_missing_migration(tmp_path) -> None:
    """Precedence must mirror ``verify_schema_current`` exactly."""

    connection = _migrated_connection(tmp_path)
    try:
        connection.execute(
            "UPDATE schema_migrations SET checksum = ? WHERE version = ?",
            ("a" * 64, CURRENT_MIGRATIONS[0].version),
        )
        connection.execute(
            "DELETE FROM schema_migrations WHERE version = ?", (CURRENT_MIGRATIONS[-1].version,)
        )
        connection.commit()
        inspection = inspect_schema(connection)

        assert inspection.state is SchemaState.CHECKSUM_MISMATCH
        assert inspection.missing_versions == (CURRENT_MIGRATIONS[-1].version,)
        with pytest.raises(PmemPersistenceError, match="checksum"):
            verify_schema_current(connection)
    finally:
        connection.close()


def test_inspect_schema_reports_empty_database(tmp_path) -> None:
    db_path = tmp_path / "empty.db"
    db_path.write_bytes(b"")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        inspection = inspect_schema(connection)
    finally:
        connection.close()

    assert inspection.state is SchemaState.MISSING_MIGRATIONS
    assert inspection.recorded_version_count == 0
    assert set(inspection.missing_versions) == {m.version for m in CURRENT_MIGRATIONS}
    assert inspection.unknown_versions == ()


def test_inspect_schema_accepts_a_plain_tuple_row_connection(tmp_path) -> None:
    connection = _migrated_connection(tmp_path)
    db_path = tmp_path / ".pmem" / "pmem.db"
    connection.close()
    plain_connection = sqlite3.connect(db_path)
    try:
        inspection = inspect_schema(plain_connection)
    finally:
        plain_connection.close()

    assert inspection.state is SchemaState.CURRENT


def test_inspect_schema_reports_unknown_recorded_versions(tmp_path) -> None:
    connection = _migrated_connection(tmp_path)
    try:
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at, checksum) VALUES (?, ?, ?)",
            ("999", "2026-01-01T00:00:00Z", "a" * 64),
        )
        connection.commit()
        inspection = inspect_schema(connection)
    finally:
        connection.close()

    assert inspection.state is SchemaState.CURRENT
    assert inspection.unknown_versions == ("999",)


def test_inspect_schema_maps_corrupt_database_to_a_safe_error(tmp_path) -> None:
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"not a sqlite database at all" * 100)
    connection = sqlite3.connect(db_path)
    try:
        with pytest.raises(PmemPersistenceError) as excinfo:
            inspect_schema(connection)
    finally:
        connection.close()

    assert "file is not a database" not in str(excinfo.value)
    assert str(db_path) not in str(excinfo.value)


def test_inspect_schema_does_not_mutate_the_database(tmp_path) -> None:
    db_path = tmp_path / ".pmem" / "pmem.db"
    connection = _migrated_connection(tmp_path)
    try:
        before = (db_path.read_bytes(), db_path.stat().st_mtime_ns)
        inspect_schema(connection)
        inspect_schema(connection)
    finally:
        connection.close()

    assert (db_path.read_bytes(), db_path.stat().st_mtime_ns) == before


def test_inspection_result_is_immutable(tmp_path) -> None:
    connection = _migrated_connection(tmp_path)
    try:
        inspection = inspect_schema(connection)
    finally:
        connection.close()

    with pytest.raises(dataclasses.FrozenInstanceError):
        inspection.state = SchemaState.CURRENT  # type: ignore[misc]


def test_verify_schema_current_still_accepts_a_migrated_database(tmp_path) -> None:
    """The refactor must not change the behaviour existing callers rely on."""

    connection = _migrated_connection(tmp_path)
    try:
        assert verify_schema_current(connection) is None
    finally:
        connection.close()


def test_verify_schema_current_still_rejects_a_stale_schema(tmp_path) -> None:
    connection = _migrated_connection(tmp_path)
    try:
        connection.execute(
            "DELETE FROM schema_migrations WHERE version = ?", (CURRENT_MIGRATIONS[-1].version,)
        )
        connection.commit()
        with pytest.raises(PmemPersistenceError, match="out of date"):
            verify_schema_current(connection)
    finally:
        connection.close()
