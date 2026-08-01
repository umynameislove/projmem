"""SQLite connection and safe execution helpers.

This module is intentionally small for database. It centralizes connection PRAGMAs,
path handling, parameterized execution, and raw SQLite error mapping before
feature repositories start writing project data.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pmem.errors import PmemNotFoundError, PmemPersistenceError, PmemSecurityError

PMEM_DIRNAME = ".pmem"
PMEM_DB_FILENAME = "pmem.db"
SQLITE_TIMEOUT_SECONDS = 5.0

_READONLY_SIDECAR_MESSAGE = (
    "The project database has active SQLite sidecar state. Close active projmem commands and retry."
)


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


def connect_database_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open a strictly read-only SQLite connection.

    Unlike :func:`connect_database`, this never creates directories, changes
    permissions, or creates the database. It opens an immutable snapshot only
    after proving that no rollback/WAL sidecar exists. Active sidecar state is
    rejected rather than ignored, because immutable mode must never silently
    omit uncheckpointed WAL data.
    """

    path = Path(db_path)
    if path.is_symlink():
        raise PmemSecurityError("Database path must not be a symlink.")
    if not path.exists():
        raise PmemNotFoundError("Project database was not found.")
    if path.is_dir():
        raise PmemSecurityError("Database path points to a directory.")

    _reject_sqlite_sidecars(path)
    uri = f"{Path(os.path.abspath(path)).as_uri()}?mode=ro&immutable=1"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA foreign_keys = ON")
        # Close the check/open race. Immutable SQLite cannot create sidecars;
        # anything appearing here belongs to another process and invalidates
        # the snapshot as current project state.
        _reject_sqlite_sidecars(path)
    except PmemPersistenceError:
        if connection is not None:
            connection.close()
        raise
    except sqlite3.Error as exc:
        if connection is not None:
            connection.close()
        raise PmemPersistenceError() from exc
    return connection


def has_active_sqlite_sidecars(db_path: str | Path) -> bool:
    """Return whether journal/WAL/master-journal state exists beside ``db_path``.

    Read-only: it lists the containing directory and inspects names only. It
    never opens, checkpoints, moves or removes a sidecar.

    Public because a diagnostic caller needs to distinguish "another command is
    running" from "this database is broken" *before* attempting to connect.
    Sharing this predicate keeps the sidecar naming policy in exactly one
    place; a second copy would be free to drift from the policy that
    :func:`connect_database_readonly` actually enforces.

    May raise :class:`OSError`; callers decide how to map it.
    """

    path = Path(db_path)
    exact_names = {
        f"{path.name}-journal",
        f"{path.name}-shm",
        f"{path.name}-wal",
    }
    master_journal_prefix = f"{path.name}-mj"
    return any(
        entry.name in exact_names or entry.name.startswith(master_journal_prefix)
        for entry in path.parent.iterdir()
    )


def _reject_sqlite_sidecars(path: Path) -> None:
    """Fail closed when journal/WAL state exists beside ``path``."""

    try:
        if has_active_sqlite_sidecars(path):
            raise PmemPersistenceError(_READONLY_SIDECAR_MESSAGE)
    except PmemPersistenceError:
        raise
    except OSError as exc:
        raise PmemPersistenceError() from exc


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
