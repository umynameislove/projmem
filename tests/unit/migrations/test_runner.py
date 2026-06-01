"""Unit tests for migration runner internals."""

import re

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import _iter_sql_statements, _utc_now_iso


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
