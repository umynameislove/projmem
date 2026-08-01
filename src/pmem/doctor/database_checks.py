"""Database diagnostics for ``pmem doctor`` (DOC-002).

Six checks, expressed entirely through the merged ``doctor-v1`` contract:

``database.exists``
    the project database is present and is a regular file reached without a
    symlink;
``database.readable``
    a strictly read-only connection opens and answers a probe query;
``database.integrity``
    ``PRAGMA integrity_check`` reports ``ok``;
``database.foreign_keys``
    ``PRAGMA foreign_key_check`` reports no violation;
``database.migrations_current``
    every known migration is recorded as applied;
``database.migration_checksums``
    every recorded migration checksum still matches the shipped migration.

Read-only by construction. Every database read goes through
:func:`pmem.repositories.sqlite.connect_database_readonly`, which opens
``mode=ro&immutable=1`` with ``PRAGMA query_only = ON`` after proving no
SQLite sidecar exists. This module never calls ``connect_database``,
``ensure_database`` or ``apply_migrations``; never creates ``.pmem``, the
database or any file; never chmods, renames, copies or unlinks; and issues no
statement other than the read-only probe, the two read-only ``PRAGMA``
inspections and the two ``SELECT`` statements inside
:func:`pmem.migrations.runner.inspect_schema`.

Privacy. Every ``message`` and ``remediation`` is a hand-written constant.
Nothing is interpolated -- not a path, not SQL, not a SQLite error string, not
a checksum, not a table or column name, not a violation count. The raw
``sqlite3`` exception is caught and discarded at the boundary; only its
*category* survives, as an internal enum. Expected diagnostic conditions
therefore become typed results, while an unexpected programmer error is
deliberately allowed to propagate rather than being flattened into a
misleading ``fail``.

Determinism. :func:`run_database_checks` defines one diagnostic invocation: it
collects one immutable snapshot and maps all six definitions over that snapshot,
so no two results in one report can describe different states of the file.
Standalone definitions do not retain state between executions.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeAlias

from pmem.doctor.model import (
    DoctorCategory,
    DoctorCheckOutcome,
    DoctorCheckResult,
    DoctorSeverity,
)
from pmem.doctor.registry import DoctorCheckContext, DoctorCheckDefinition
from pmem.errors import PmemError
from pmem.migrations.runner import SchemaInspection, SchemaState, inspect_schema
from pmem.repositories.sqlite import (
    PMEM_DIRNAME,
    connect_database_readonly,
    has_active_sqlite_sidecars,
    project_database_path,
)

CHECK_DATABASE_EXISTS = "database.exists"
CHECK_DATABASE_READABLE = "database.readable"
CHECK_DATABASE_INTEGRITY = "database.integrity"
CHECK_DATABASE_FOREIGN_KEYS = "database.foreign_keys"
CHECK_DATABASE_MIGRATIONS_CURRENT = "database.migrations_current"
CHECK_DATABASE_MIGRATION_CHECKSUMS = "database.migration_checksums"

# Canonical (ascending ``check_id``) order. Note that ``migration_checksums``
# sorts before ``migrations_current`` because ``_`` (0x5F) precedes ``s``
# (0x73); the factory derives this order with ``sorted`` rather than trusting
# this literal, and a test asserts the two agree.
DATABASE_CHECK_IDS: tuple[str, ...] = (
    CHECK_DATABASE_EXISTS,
    CHECK_DATABASE_FOREIGN_KEYS,
    CHECK_DATABASE_INTEGRITY,
    CHECK_DATABASE_MIGRATION_CHECKSUMS,
    CHECK_DATABASE_MIGRATIONS_CURRENT,
    CHECK_DATABASE_READABLE,
)

# The probe query proves the file really is a SQLite database. Opening is not
# enough: a read-only immutable connection to random bytes succeeds, and only
# the first real statement raises. Verified against the shipped SQLite in
# ``tests/unit/doctor/test_database_checks.py``.
_PROBE_SQL = "SELECT count(*) FROM sqlite_master"
_INTEGRITY_SQL = "PRAGMA integrity_check"
_FOREIGN_KEY_SQL = "PRAGMA foreign_key_check"
_INTEGRITY_OK = "ok"


# --------------------------------------------------------------------------- #
# Internal snapshot vocabulary                                                 #
# --------------------------------------------------------------------------- #
class PathState(Enum):
    """How the database path itself resolved."""

    OK = "ok"
    MISSING = "missing"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    PARENT_SYMLINK = "parent_symlink"
    NOT_REGULAR_FILE = "not_regular_file"
    ROOT_SYMLINK = "root_symlink"


class ReadState(Enum):
    """Whether a read-only connection could be established and used."""

    OK = "ok"
    SIDECAR_ACTIVE = "sidecar_active"
    SIDECAR_STATE_UNREADABLE = "sidecar_state_unreadable"
    UNREADABLE = "unreadable"
    NOT_INSPECTED = "not_inspected"


class ProbeState(Enum):
    """Result of a read-only inspection that may not have been reachable."""

    OK = "ok"
    FAILED = "failed"
    NOT_INSPECTED = "not_inspected"


@dataclass(frozen=True, slots=True)
class DatabaseInspection:
    """Immutable snapshot of the project database, collected exactly once.

    Holds only closed vocabularies and a version-name-only schema inspection.
    No path, no SQLite error text, no checksum and no table name is retained,
    so nothing sensitive can reach a result by accident.
    """

    path_state: PathState
    read_state: ReadState
    integrity_state: ProbeState
    foreign_key_state: ProbeState
    schema: SchemaInspection | None


def inspect_database(project_root: str | Path) -> DatabaseInspection:
    """Collect one read-only snapshot of the project database. Never mutates.

    Order matters and is deliberate: path safety is settled before anything is
    opened, and sidecar state is settled before a connection is attempted, so a
    symlinked or busy database is never followed or read.
    """

    root = Path(project_root)
    db_path = project_database_path(root)

    path_state = _classify_path(root, db_path)
    if path_state is not PathState.OK:
        return DatabaseInspection(
            path_state=path_state,
            read_state=ReadState.NOT_INSPECTED,
            integrity_state=ProbeState.NOT_INSPECTED,
            foreign_key_state=ProbeState.NOT_INSPECTED,
            schema=None,
        )

    sidecar_state = _sidecar_state(db_path)
    if sidecar_state is SidecarState.PRESENT:
        return DatabaseInspection(
            path_state=path_state,
            read_state=ReadState.SIDECAR_ACTIVE,
            integrity_state=ProbeState.NOT_INSPECTED,
            foreign_key_state=ProbeState.NOT_INSPECTED,
            schema=None,
        )
    if sidecar_state is SidecarState.UNKNOWN:
        return DatabaseInspection(
            path_state=path_state,
            read_state=ReadState.SIDECAR_STATE_UNREADABLE,
            integrity_state=ProbeState.NOT_INSPECTED,
            foreign_key_state=ProbeState.NOT_INSPECTED,
            schema=None,
        )

    return _inspect_open_database(db_path, path_state)


def _classify_path(root: Path, db_path: Path) -> PathState:
    """Classify the database path without following a symlink or reading it."""

    try:
        if root.is_symlink():
            return PathState.ROOT_SYMLINK
        if (root / PMEM_DIRNAME).is_symlink():
            return PathState.PARENT_SYMLINK
        if db_path.is_symlink():
            return PathState.SYMLINK
        if not db_path.exists():
            return PathState.MISSING
        if db_path.is_dir():
            return PathState.DIRECTORY
        if not db_path.is_file():
            return PathState.NOT_REGULAR_FILE
    except OSError:
        # A path that cannot even be stat-ed is treated as unusable rather than
        # crashing the diagnostic. Fail closed.
        return PathState.NOT_REGULAR_FILE
    return PathState.OK


class SidecarState(Enum):
    """Whether sidecar state was observed or could be determined safely."""

    ABSENT = "absent"
    PRESENT = "present"
    UNKNOWN = "unknown"


def _sidecar_state(db_path: Path) -> SidecarState:
    """Inspect sidecar names without turning an I/O error into a false claim."""

    try:
        present = has_active_sqlite_sidecars(db_path)
    except OSError:
        return SidecarState.UNKNOWN
    return SidecarState.PRESENT if present else SidecarState.ABSENT


def _inspect_open_database(db_path: Path, path_state: PathState) -> DatabaseInspection:
    """Open one read-only connection and collect every database observation."""

    try:
        connection = connect_database_readonly(db_path)
    except PmemError:
        # ``connect_database_readonly`` already maps raw driver failures onto a
        # safe typed error. Re-check sidecars to keep "another command is
        # running" distinct from "this database is broken" even when the
        # sidecar appeared during the connect race.
        sidecar_state = _sidecar_state(db_path)
        read_state = (
            ReadState.SIDECAR_ACTIVE
            if sidecar_state is SidecarState.PRESENT
            else ReadState.UNREADABLE
        )
        return DatabaseInspection(
            path_state=path_state,
            read_state=read_state,
            integrity_state=ProbeState.NOT_INSPECTED,
            foreign_key_state=ProbeState.NOT_INSPECTED,
            schema=None,
        )

    try:
        try:
            connection.execute(_PROBE_SQL).fetchone()
        except sqlite3.DatabaseError:
            return DatabaseInspection(
                path_state=path_state,
                read_state=ReadState.UNREADABLE,
                integrity_state=ProbeState.NOT_INSPECTED,
                foreign_key_state=ProbeState.NOT_INSPECTED,
                schema=None,
            )

        integrity_state = _integrity_state(connection)
        if integrity_state is ProbeState.FAILED:
            # Once structural integrity is known to be broken, subsequent
            # query results are not trustworthy enough to claim PASS.
            return DatabaseInspection(
                path_state=path_state,
                read_state=ReadState.OK,
                integrity_state=integrity_state,
                foreign_key_state=ProbeState.NOT_INSPECTED,
                schema=None,
            )
        foreign_key_state = _foreign_key_state(connection)
        schema = _schema_inspection(connection)
        return DatabaseInspection(
            path_state=path_state,
            read_state=ReadState.OK,
            integrity_state=integrity_state,
            foreign_key_state=foreign_key_state,
            schema=schema,
        )
    finally:
        connection.close()


def _integrity_state(connection: sqlite3.Connection) -> ProbeState:
    """Run the read-only integrity pragma.

    Two distinct corruption shapes were verified against the shipped SQLite: a
    damaged interior page makes the pragma *raise*, while other damage makes it
    *return* rows other than ``ok``. Both are treated as failure.
    """

    try:
        rows = connection.execute(_INTEGRITY_SQL).fetchall()
    except sqlite3.DatabaseError:
        return ProbeState.FAILED
    if len(rows) != 1 or rows[0][0] != _INTEGRITY_OK:
        return ProbeState.FAILED
    return ProbeState.OK


def _foreign_key_state(connection: sqlite3.Connection) -> ProbeState:
    """Run the read-only foreign-key pragma, discarding every returned row.

    The pragma returns the offending table name, rowid and parent table. None
    of that is retained: only whether the result set was empty.
    """

    try:
        rows = connection.execute(_FOREIGN_KEY_SQL).fetchall()
    except sqlite3.DatabaseError:
        return ProbeState.FAILED
    return ProbeState.OK if not rows else ProbeState.FAILED


def _schema_inspection(connection: sqlite3.Connection) -> SchemaInspection | None:
    """Inspect recorded migrations through the typed read-only seam."""

    try:
        return inspect_schema(connection)
    except PmemError:
        return None


# --------------------------------------------------------------------------- #
# Stable, hand-written result text                                             #
# --------------------------------------------------------------------------- #
_REMEDIATION_INIT = "Run `pmem init` in the project directory to create the project database."
_REMEDIATION_REPLACE_DIRECTORY = (
    "A directory occupies the database location. Move it aside, then run `pmem init`."
)
_REMEDIATION_SYMLINK = (
    "projmem refuses to read or write a symlinked database. Replace the symlink with the "
    "real database file, then re-run the diagnostic."
)
_REMEDIATION_PARENT_SYMLINK = (
    "projmem refuses to read a symlinked project state directory. Replace it with a real "
    "directory, then re-run the diagnostic."
)
_REMEDIATION_ROOT_SYMLINK = (
    "Run the diagnostic from the real project directory rather than through a symlink."
)
_REMEDIATION_UNREADABLE = (
    "The database could not be read as a SQLite database. Restore the most recent backup "
    "kept beside it, then re-run the diagnostic."
)
_REMEDIATION_SIDECAR = (
    "Another projmem command is holding the database. Let it finish, then re-run the diagnostic."
)
_REMEDIATION_STATE_DIRECTORY = (
    "The project state directory could not be inspected safely. Check its access permissions, "
    "then re-run the diagnostic."
)
_REMEDIATION_INTEGRITY = (
    "SQLite reported structural damage. Restore the most recent backup kept beside the "
    "database, then re-run the diagnostic."
)
_REMEDIATION_FOREIGN_KEYS = (
    "Evidence records reference rows that no longer exist. Restore the most recent backup "
    "kept beside the database, then re-run the diagnostic."
)
_REMEDIATION_MIGRATIONS = (
    "The database schema is older than this projmem version. Run `pmem init` in the project "
    "directory to migrate it."
)
_REMEDIATION_CHECKSUMS = (
    "A recorded migration no longer matches the migration shipped with this projmem "
    "version. Restore the most recent backup kept beside the database, then re-run the "
    "diagnostic."
)


def _result(
    check_id: str,
    outcome: DoctorCheckOutcome,
    severity: DoctorSeverity,
    message: str,
    remediation: str | None = None,
) -> DoctorCheckResult:
    return DoctorCheckResult(
        check_id=check_id,
        category=DoctorCategory.DATABASE,
        outcome=outcome,
        severity=severity,
        message=message,
        remediation=remediation,
        related_entity_id=None,
    )


def _passed(check_id: str, message: str) -> DoctorCheckResult:
    return _result(check_id, DoctorCheckOutcome.PASS, DoctorSeverity.INFO, message)


def _failed(check_id: str, message: str, remediation: str) -> DoctorCheckResult:
    return _result(check_id, DoctorCheckOutcome.FAIL, DoctorSeverity.ERROR, message, remediation)


def _blocked(check_id: str, message: str) -> DoctorCheckResult:
    """A downstream check that could not run because a prerequisite failed.

    Severity is ``info`` on purpose. The prerequisite already reported the one
    actionable failure at ``error`` severity; repeating that alarm on every
    dependent check would multiply attention without adding a distinct action.
    The outcome is still ``skipped``, never ``pass``, so the report can never
    claim a check succeeded when it did not run.
    """

    return _result(check_id, DoctorCheckOutcome.SKIPPED, DoctorSeverity.INFO, message)


_BLOCKED_MESSAGE = "The check did not run because an earlier database check failed."


# --------------------------------------------------------------------------- #
# Snapshot -> result mapping (one function per stable check id)                #
# --------------------------------------------------------------------------- #
def _check_exists(inspection: DatabaseInspection) -> DoctorCheckResult:
    check_id = CHECK_DATABASE_EXISTS
    state = inspection.path_state
    if state is PathState.OK:
        return _passed(check_id, "The project database is present and is a regular file.")
    if state is PathState.MISSING:
        return _failed(
            check_id,
            "The project database is missing, so projmem has no memory to inspect.",
            _REMEDIATION_INIT,
        )
    if state is PathState.DIRECTORY:
        return _failed(
            check_id,
            "A directory occupies the database location instead of a database file.",
            _REMEDIATION_REPLACE_DIRECTORY,
        )
    if state is PathState.SYMLINK:
        return _failed(
            check_id,
            "The database location is a symlink, which projmem refuses to follow.",
            _REMEDIATION_SYMLINK,
        )
    if state is PathState.PARENT_SYMLINK:
        return _failed(
            check_id,
            "The project state directory is a symlink, which projmem refuses to follow.",
            _REMEDIATION_PARENT_SYMLINK,
        )
    if state is PathState.ROOT_SYMLINK:
        return _failed(
            check_id,
            "The supplied project root is a symlink, which projmem refuses to follow.",
            _REMEDIATION_ROOT_SYMLINK,
        )
    return _failed(
        check_id,
        "The database location is not a regular file.",
        _REMEDIATION_SYMLINK,
    )


def _check_readable(inspection: DatabaseInspection) -> DoctorCheckResult:
    check_id = CHECK_DATABASE_READABLE
    state = inspection.read_state
    if state is ReadState.OK:
        return _passed(check_id, "The project database opened for reading and answered a probe.")
    if state is ReadState.SIDECAR_ACTIVE:
        # Not a failure: the database is not proven bad, projmem simply refused
        # to read a snapshot that may omit uncheckpointed data. ``skipped`` at
        # ``warning`` severity makes the report ``incomplete``, which is the
        # honest conclusion for "health could not be determined right now".
        return _result(
            check_id,
            DoctorCheckOutcome.SKIPPED,
            DoctorSeverity.WARNING,
            "The database is in use by another projmem command, so it was not read.",
            _REMEDIATION_SIDECAR,
        )
    if state is ReadState.UNREADABLE:
        return _failed(
            check_id,
            "The database could not be opened and read as a SQLite database.",
            _REMEDIATION_UNREADABLE,
        )
    if state is ReadState.SIDECAR_STATE_UNREADABLE:
        return _failed(
            check_id,
            "The project state directory could not be inspected safely.",
            _REMEDIATION_STATE_DIRECTORY,
        )
    return _blocked(check_id, _BLOCKED_MESSAGE)


def _probe_result(
    check_id: str,
    state: ProbeState,
    passed_message: str,
    failed_message: str,
    remediation: str,
) -> DoctorCheckResult:
    if state is ProbeState.OK:
        return _passed(check_id, passed_message)
    if state is ProbeState.FAILED:
        return _failed(check_id, failed_message, remediation)
    return _blocked(check_id, _BLOCKED_MESSAGE)


def _check_integrity(inspection: DatabaseInspection) -> DoctorCheckResult:
    return _probe_result(
        CHECK_DATABASE_INTEGRITY,
        inspection.integrity_state,
        "SQLite reported no structural damage in the project database.",
        "SQLite reported structural damage in the project database.",
        _REMEDIATION_INTEGRITY,
    )


def _check_foreign_keys(inspection: DatabaseInspection) -> DoctorCheckResult:
    # ``error``, not ``warning``: a dangling reference means the evidence chain
    # a recommendation would be built from is already broken, and the migration
    # runner treats the same condition as a hard failure
    # (``runner._assert_database_integrity``). Reporting it as a warning would
    # let a report conclude ``degraded`` while recommendations silently rest on
    # missing evidence.
    return _probe_result(
        CHECK_DATABASE_FOREIGN_KEYS,
        inspection.foreign_key_state,
        "Every evidence record references a row that exists.",
        "Some evidence records reference rows that no longer exist.",
        _REMEDIATION_FOREIGN_KEYS,
    )


def _check_migrations_current(inspection: DatabaseInspection) -> DoctorCheckResult:
    check_id = CHECK_DATABASE_MIGRATIONS_CURRENT
    schema = inspection.schema
    if schema is None:
        return _blocked(check_id, _BLOCKED_MESSAGE)
    if schema.missing_versions:
        # ``error``: every read-only command already refuses to run against an
        # out-of-date schema (``runner.verify_schema_current``), so the tool is
        # unusable until this is fixed.
        return _failed(
            check_id,
            "The database schema is missing one or more required migrations.",
            _REMEDIATION_MIGRATIONS,
        )
    return _passed(check_id, "Every required migration is recorded as applied.")


def _check_migration_checksums(inspection: DatabaseInspection) -> DoctorCheckResult:
    check_id = CHECK_DATABASE_MIGRATION_CHECKSUMS
    schema = inspection.schema
    if schema is None:
        return _blocked(check_id, _BLOCKED_MESSAGE)
    if schema.state is SchemaState.CHECKSUM_MISMATCH:
        return _failed(
            check_id,
            "A recorded migration checksum no longer matches the shipped migration.",
            _REMEDIATION_CHECKSUMS,
        )
    known_recorded_count = schema.recorded_version_count - len(schema.unknown_versions)
    if known_recorded_count == 0:
        # Nothing has been recorded, so there is no checksum to verify. Saying
        # ``pass`` here would let an empty file look partly healthy.
        return _result(
            check_id,
            DoctorCheckOutcome.SKIPPED,
            DoctorSeverity.INFO,
            "No migration is recorded yet, so no checksum could be verified.",
        )
    return _passed(
        check_id,
        "Every known recorded migration checksum matches the shipped migration.",
    )


# --------------------------------------------------------------------------- #
# Registry factory                                                             #
# --------------------------------------------------------------------------- #
_ResultBuilder: TypeAlias = Callable[["DatabaseInspection"], DoctorCheckResult]
_InspectionProvider: TypeAlias = Callable[[DoctorCheckContext], DatabaseInspection]


def _definitions(
    inspection_provider: _InspectionProvider,
) -> tuple[DoctorCheckDefinition, ...]:
    """Build definitions around an explicit inspection provider."""

    def _bind(check_id: str, builder: _ResultBuilder) -> DoctorCheckDefinition:
        def _run(context: DoctorCheckContext) -> DoctorCheckResult:
            return builder(inspection_provider(context))

        return DoctorCheckDefinition(
            check_id=check_id,
            category=DoctorCategory.DATABASE,
            run=_run,
        )

    definitions = (
        _bind(CHECK_DATABASE_EXISTS, _check_exists),
        _bind(CHECK_DATABASE_FOREIGN_KEYS, _check_foreign_keys),
        _bind(CHECK_DATABASE_INTEGRITY, _check_integrity),
        _bind(CHECK_DATABASE_MIGRATIONS_CURRENT, _check_migrations_current),
        _bind(CHECK_DATABASE_MIGRATION_CHECKSUMS, _check_migration_checksums),
        _bind(CHECK_DATABASE_READABLE, _check_readable),
    )
    return tuple(sorted(definitions, key=lambda definition: definition.check_id))


def database_check_definitions() -> tuple[DoctorCheckDefinition, ...]:
    """Return the six database check definitions in canonical ``check_id`` order.

    Deterministic and side-effect free: calling it opens no database, reads no
    file and runs no check. The returned definitions share one lazily collected
    Definitions are safe to retain: each standalone execution inspects current
    state and never returns a cached result from an earlier invocation. Report
    assembly should use :func:`run_database_checks` to share one snapshot.
    """

    return _definitions(lambda context: inspect_database(context.project_root))


def run_database_checks(context: DoctorCheckContext) -> tuple[DoctorCheckResult, ...]:
    """Execute all database checks over one invocation-scoped snapshot."""

    inspection = inspect_database(context.project_root)
    definitions = _definitions(lambda _context: inspection)
    return tuple(definition.execute(context) for definition in definitions)
