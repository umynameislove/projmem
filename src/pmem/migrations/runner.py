"""Migration runner for project-local SQLite databases."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pmem.errors import PmemPersistenceError
from pmem.migrations.schema_v1 import SCHEMA_V1, Migration
from pmem.migrations.schema_v2 import SCHEMA_V2
from pmem.repositories.sqlite import connect_database, execute

CURRENT_MIGRATIONS: tuple[Migration, ...] = (SCHEMA_V1, SCHEMA_V2)


@dataclass(frozen=True)
class MigrationResult:
    """Result of applying pending migrations."""

    db_path: Path
    applied_versions: tuple[str, ...]
    skipped_versions: tuple[str, ...]
    backup_path: Path | None


class SchemaState(Enum):
    """Typed schema condition for read-only callers.

    Exists so a diagnostic caller can distinguish *missing migrations* from a
    *checksum mismatch* without parsing an exception message. Message parsing
    would couple the caller to error wording that is meant to stay free to
    change, and it would silently start misreporting the moment the wording
    moved.
    """

    CURRENT = "current"
    MISSING_MIGRATIONS = "missing_migrations"
    CHECKSUM_MISMATCH = "checksum_mismatch"


@dataclass(frozen=True)
class SchemaInspection:
    """Immutable, read-only description of the recorded migration state.

    ``missing_versions`` and ``mismatched_versions`` carry migration *version
    names* only. No checksum value, no SQL and no path is ever included, so the
    structure is safe to reason about in a diagnostic that must not leak them.
    Callers are still responsible for not rendering version names into public
    output if their own contract forbids it.
    """

    state: SchemaState
    missing_versions: tuple[str, ...]
    mismatched_versions: tuple[str, ...]
    unknown_versions: tuple[str, ...]
    recorded_version_count: int


def inspect_schema(
    connection: sqlite3.Connection,
    migrations: tuple[Migration, ...] = CURRENT_MIGRATIONS,
) -> SchemaInspection:
    """Describe the recorded migration state without mutating anything.

    Strictly read-only: it issues two ``SELECT`` statements and never creates,
    migrates, backs up or repairs. A database that cannot be read raises the
    same safe :class:`PmemPersistenceError` that
    :func:`verify_schema_current` raises.

    ``state`` mirrors the precedence :func:`verify_schema_current` enforces --
    a checksum mismatch outranks a missing migration -- so the two functions
    can never disagree about which problem to report first.
    """

    try:
        applied = _load_applied_migrations(connection)
    except sqlite3.Error as exc:
        raise PmemPersistenceError("The project database could not be read.") from exc

    expected = {migration.version: migration.checksum for migration in migrations}
    mismatched = tuple(
        version
        for version, checksum in sorted(applied.items())
        if version in expected and expected[version] != checksum
    )
    missing = tuple(
        migration.version for migration in migrations if migration.version not in applied
    )
    unknown = tuple(version for version in sorted(applied) if version not in expected)

    if mismatched:
        state = SchemaState.CHECKSUM_MISMATCH
    elif missing:
        state = SchemaState.MISSING_MIGRATIONS
    else:
        state = SchemaState.CURRENT

    return SchemaInspection(
        state=state,
        missing_versions=missing,
        mismatched_versions=mismatched,
        unknown_versions=unknown,
        recorded_version_count=len(applied),
    )


def apply_migrations(
    db_path: str | Path,
    migrations: tuple[Migration, ...] = CURRENT_MIGRATIONS,
    *,
    create_backup: bool = True,
) -> MigrationResult:
    """Apply pending migrations to a SQLite database."""

    path = Path(db_path)
    connection = connect_database(path)
    backup_path: Path | None = None

    try:
        applied = _load_applied_migrations(connection)
        _check_applied_checksums(applied, migrations)
        pending = tuple(migration for migration in migrations if migration.version not in applied)

        if pending and create_backup and path.exists() and path.stat().st_size > 0:
            backup_path = _backup_database(path)

        applied_versions: list[str] = []
        for migration in pending:
            _apply_one(connection, migration)
            applied_versions.append(migration.version)

        _assert_database_integrity(connection)
        skipped_versions = tuple(
            migration.version for migration in migrations if migration.version in applied
        )
        return MigrationResult(
            db_path=path,
            applied_versions=tuple(applied_versions),
            skipped_versions=skipped_versions,
            backup_path=backup_path,
        )
    finally:
        connection.close()


def verify_schema_current(
    connection: sqlite3.Connection,
    migrations: tuple[Migration, ...] = CURRENT_MIGRATIONS,
) -> None:
    """Verify (never migrate) that every known migration is applied and intact.

    Raises a safe :class:`PmemPersistenceError` when a migration is missing or
    when a recorded checksum no longer matches. Used by read-only paths that
    must not run migrations. The error message never leaks SQL, paths, or the
    expected/actual checksum values.

    Implemented on top of :func:`inspect_schema` so the raising path and the
    diagnostic path can never disagree about what the schema state is.
    """

    inspection = inspect_schema(connection, migrations)
    if inspection.state is SchemaState.CHECKSUM_MISMATCH:
        raise PmemPersistenceError("Migration checksum mismatch.")
    if inspection.state is SchemaState.MISSING_MIGRATIONS:
        raise PmemPersistenceError(
            "The projmem database schema is out of date. "
            "Run `pmem init` to migrate before this read-only command."
        )


def _load_applied_migrations(connection: sqlite3.Connection) -> dict[str, str]:
    """Load migration versions if the tracking table already exists."""

    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = ? AND name = ?",
        ("table", "schema_migrations"),
    ).fetchone()
    if row is None:
        return {}

    rows = connection.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    # Positional access works for both plain tuple rows and ``sqlite3.Row``.
    # The public inspection seam must not depend on a caller-specific
    # ``row_factory`` merely to read two fixed columns.
    return {str(row[0]): str(row[1]) for row in rows}


def _check_applied_checksums(
    applied: dict[str, str],
    migrations: tuple[Migration, ...],
) -> None:
    """Reject local DBs whose recorded migration checksum changed."""

    expected = {migration.version: migration.checksum for migration in migrations}
    for version, checksum in applied.items():
        if version in expected and expected[version] != checksum:
            raise PmemPersistenceError("Migration checksum mismatch.")


def _backup_database(db_path: Path) -> Path:
    """Copy an existing DB before changing schema."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_name(f"{db_path.name}.{timestamp}.bak")
    shutil.copy2(db_path, backup_path)
    return backup_path


def _apply_one(connection: sqlite3.Connection, migration: Migration) -> None:
    """Apply one migration in a transaction and record its checksum."""

    try:
        connection.execute("BEGIN")
        for statement in _iter_sql_statements(migration.sql):
            connection.execute(statement)
        execute(
            connection,
            "INSERT INTO schema_migrations(version, applied_at, checksum) VALUES (?, ?, ?)",
            (migration.version, _utc_now_iso(), migration.checksum),
        )
        connection.execute("COMMIT")
    except sqlite3.Error as exc:
        connection.execute("ROLLBACK")
        raise PmemPersistenceError("Migration failed.") from exc
    except PmemPersistenceError:
        connection.execute("ROLLBACK")
        raise


def _iter_sql_statements(script: str) -> tuple[str, ...]:
    """Split a migration script into SQLite statements."""

    statements: list[str] = []
    pending: list[str] = []
    for line in script.splitlines():
        pending.append(line)
        candidate = "\n".join(pending).strip()
        if candidate and sqlite3.complete_statement(candidate):
            statements.append(candidate)
            pending = []

    remainder = "\n".join(pending).strip()
    if remainder:
        raise PmemPersistenceError("Migration SQL contains an incomplete statement.")
    return tuple(statements)


def _assert_database_integrity(connection: sqlite3.Connection) -> None:
    """Run SQLite integrity checks after migration."""

    fk_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    if fk_rows:
        raise PmemPersistenceError("Database foreign key check failed.")

    row = connection.execute("PRAGMA integrity_check").fetchone()
    if row is None or row[0] != "ok":
        raise PmemPersistenceError("Database integrity check failed.")


def _utc_now_iso() -> str:
    """Return compact UTC ISO timestamp for migration records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
