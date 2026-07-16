"""Unit tests for migration runner internals."""

import re
import sqlite3

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import _iter_sql_statements, _utc_now_iso, verify_schema_current


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
