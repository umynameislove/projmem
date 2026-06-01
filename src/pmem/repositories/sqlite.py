"""SQLite connection and safe execution helpers.

This module is intentionally small for database. It centralizes connection PRAGMAs,
path handling, parameterized execution, and raw SQLite error mapping before
feature repositories start writing project data.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pmem.errors import PmemPersistenceError, PmemSecurityError

PMEM_DIRNAME = ".pmem"
PMEM_DB_FILENAME = "pmem.db"
SQLITE_TIMEOUT_SECONDS = 5.0


def project_database_path(project_root: str | Path) -> Path:
    """Return the local-memory database path for a project root."""

    root = Path(project_root)
    if not str(root).strip():
        raise PmemSecurityError("Project root cannot be blank.")
    return root / PMEM_DIRNAME / PMEM_DB_FILENAME


def connect_database(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with project-safe defaults."""

    path = Path(db_path)
    if path.exists() and path.is_dir():
        raise PmemSecurityError("Database path points to a directory.")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = sqlite3.connect(path, timeout=SQLITE_TIMEOUT_SECONDS)
    except sqlite3.Error as exc:
        raise PmemPersistenceError() from exc

    _restrict_file_permissions(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def execute(
    connection: sqlite3.Connection,
    sql: str,
    parameters: Sequence[Any] = (),
) -> sqlite3.Cursor:
    """Execute SQL with parameters and map raw SQLite errors."""

    try:
        return connection.execute(sql, parameters)
    except sqlite3.IntegrityError as exc:
        raise PmemPersistenceError("Database constraint violation.") from exc
    except sqlite3.Error as exc:
        raise PmemPersistenceError() from exc


def query_one(
    connection: sqlite3.Connection,
    sql: str,
    parameters: Sequence[Any] = (),
) -> sqlite3.Row | None:
    """Return one row using the safe execution path."""

    return execute(connection, sql, parameters).fetchone()


def _restrict_file_permissions(path: Path) -> None:
    """Restrict project-local DB readability to the current OS user."""

    try:
        path.chmod(0o600)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PmemSecurityError("Database permissions could not be restricted.") from exc
