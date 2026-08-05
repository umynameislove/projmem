"""Tracked-path diagnostics for ``pmem doctor`` (DOC-004).

Four checks, expressed entirely through the merged ``doctor-v1`` contract:

``tracked_paths.records_safe``
    every stored record is canonical, project-relative and non-internal, with
    valid hash and size metadata;
``tracked_paths.symlink``
    no tracked path or any of its parent components is a symlink;
``tracked_paths.present``
    every safe record still resolves to a regular file;
``tracked_paths.content_current``
    the observed SHA-256 equals the stored SHA-256, and the file did not change
    while it was being hashed.

Stored records are not trusted. The tracking service validates on write, but a
database can be edited, restored from an old backup, or corrupted afterwards.
Every record is therefore re-validated here against the same rules the service
applies, and a record that would have to be *normalized* to become safe is
rejected rather than quietly repaired.

Descriptor-anchored traversal. No tracked content path is ever resolved and no
tracked content file is ever opened by path. The project root is anchored
*before* the approved config/database seam is called, using
``O_DIRECTORY | O_NOFOLLOW``,
each parent component is inspected with ``follow_symlinks=False`` and opened
relative to the previous descriptor, and the identity of every opened directory
is proved against the name that was inspected. The leaf is opened with
``O_NOFOLLOW | O_NONBLOCK`` and re-checked with ``fstat``. This closes the
check-then-open race that ``compute_file_hash(path)`` is subject to, which is
why that helper is deliberately not used.

Content is read but never retained. Hashing streams the descriptor in fixed
chunks; only the enum conclusion survives. The observed digest is compared and
discarded -- it never reaches the snapshot and never reaches a result.

Privacy. Every ``message`` and ``remediation`` is a hand-written constant.
Nothing is interpolated: not a path, not a file name, not a stored or observed
hash, not a tag, not a size, not a count, not a timestamp, not a project or
record identifier, and not an ``OSError``.

Read-only. Records are read through the existing read-only seam
(``require_project_context_readonly``, which opens ``mode=ro&immutable=1`` and
verifies the schema without migrating). No hash is refreshed, no
``last_checked`` is touched, no row is written, and nothing on the filesystem is
created, renamed, chmod-ed or removed.

Resource bounds. The database query returns at most the record budget plus one
sentinel row, so excess cardinality is detected before Python materialises or
sorts an unbounded result. Content hashing has a separate byte budget.

Determinism. One immutable snapshot backs every result of a single
:func:`run_tracked_path_checks` invocation. Nothing is cached between
invocations. Records are returned in canonical database order, classified into
counted buckets, and the results depend only on those counts, never on database
row order or filesystem iteration order.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import stat
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
from pmem.doctor.pathsafety import (
    anchored_traversal_supported,
    close_quietly,
    open_directory,
    open_directory_at,
    open_file_at,
    same_directory_binding,
    same_file_binding,
    same_path_binding,
)
from pmem.doctor.registry import DoctorCheckContext, DoctorCheckDefinition
from pmem.errors import PmemError
from pmem.repositories.sqlite import PMEM_DIRNAME
from pmem.repositories.tracked_paths import TrackedPathRecord, TrackedPathRepository
from pmem.services.project_context import require_project_context_readonly
from pmem.services.tracking import MAX_TRACKED_PATH_LENGTH

CHECK_TRACKED_PATHS_CONTENT_CURRENT = "tracked_paths.content_current"
CHECK_TRACKED_PATHS_PRESENT = "tracked_paths.present"
CHECK_TRACKED_PATHS_RECORDS_SAFE = "tracked_paths.records_safe"
CHECK_TRACKED_PATHS_SYMLINK = "tracked_paths.symlink"

# Canonical (ascending ``check_id``) order. The factory derives this with
# ``sorted`` rather than trusting the literal; a test asserts they agree.
TRACKED_PATH_CHECK_IDS: tuple[str, ...] = (
    CHECK_TRACKED_PATHS_CONTENT_CURRENT,
    CHECK_TRACKED_PATHS_PRESENT,
    CHECK_TRACKED_PATHS_RECORDS_SAFE,
    CHECK_TRACKED_PATHS_SYMLINK,
)

# Resource budgets. The record query is capped at the limit plus one sentinel
# row, and the byte budget stops the reader mid-file after at most one further
# chunk. Exceeding either yields an explicitly incomplete result and never a
# pass over the unexamined remainder.
MAX_INSPECTED_RECORDS = 2000
MAX_TOTAL_HASHED_BYTES = 512 * 1024 * 1024
HASH_CHUNK_SIZE = 1024 * 1024

_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}\Z")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


# --------------------------------------------------------------------------- #
# Internal vocabulary                                                          #
# --------------------------------------------------------------------------- #
class SourceState(Enum):
    """Whether tracked-path records could be read at all."""

    OK = "ok"
    UNAVAILABLE = "unavailable"
    NOT_SUPPORTED = "not_supported"


class RecordVerdict(Enum):
    """Classification of one stored record and its filesystem state."""

    UNSAFE_RECORD = "unsafe_record"
    SYMLINK = "symlink"
    MISSING = "missing"
    WRONG_TYPE = "wrong_type"
    UNREADABLE = "unreadable"
    RACED = "raced"
    BUDGET_SKIPPED = "budget_skipped"
    CHANGED = "changed"
    CURRENT = "current"


@dataclass(frozen=True, slots=True)
class TrackedPathSnapshot:
    """Immutable, content-free view of every tracked record.

    Only counts survive. No path, hash, name, tag, size, timestamp, identifier
    or exception text is retained, so nothing sensitive can reach a result even
    by accident.
    """

    source: SourceState
    record_count: int
    unsafe_record_count: int
    symlink_count: int
    missing_count: int
    wrong_type_count: int
    unreadable_count: int
    raced_count: int
    budget_skipped_count: int
    changed_count: int
    current_count: int
    record_limit_exceeded: bool
    root_binding_changed: bool

    @property
    def inspected_count(self) -> int:
        """Records whose filesystem state was actually established."""

        return self.record_count - self.unsafe_record_count


# --------------------------------------------------------------------------- #
# Stored-record validation                                                     #
# --------------------------------------------------------------------------- #
def stored_path_is_safe(value: str) -> bool:
    """Return whether a stored path is canonical, relative and non-internal.

    Deliberately a predicate, never a normalizer: a stored path that would need
    to be rewritten to become safe is unsafe evidence and must be reported, not
    silently repaired.
    """

    if not value or not value.strip():
        return False
    if len(value) > MAX_TRACKED_PATH_LENGTH:
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return False
    if "\\" in value:
        return False
    # A leading slash covers both an absolute POSIX path and a ``//server``
    # UNC reference, so no separate UNC branch is needed.
    if value.startswith("/"):
        return False
    if _WINDOWS_DRIVE_RE.match(value):
        return False

    components = value.split("/")
    if any(component in ("", ".", "..") for component in components):
        return False
    if components[0].casefold() == PMEM_DIRNAME.casefold():
        return False
    return True


def stored_hash_is_safe(value: str) -> bool:
    """Return whether a stored digest is exactly 64 lowercase hex characters."""

    return bool(_HEX_SHA256_RE.match(value))


def stored_size_is_safe(value: int | None) -> bool:
    """Return whether a stored size is absent or non-negative."""

    return value is None or value >= 0


def record_is_safe(record: TrackedPathRecord) -> bool:
    """Return whether every stored field of one record is trustworthy."""

    return (
        stored_path_is_safe(record.path)
        and stored_hash_is_safe(record.sha256)
        and stored_size_is_safe(record.size_bytes)
    )


# --------------------------------------------------------------------------- #
# Descriptor-anchored inspection                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _Budget:
    """Mutable-free byte budget carried through one invocation."""

    remaining_bytes: int


def _hash_open_file(descriptor: int, remaining_bytes: int) -> tuple[str | None, int]:
    """Stream a descriptor into SHA-256, stopping once the budget is spent.

    Returns ``(digest_or_None, bytes_consumed)``. ``None`` means the byte budget
    was exhausted, so no conclusion about the content may be drawn. The reader
    stops after at most one chunk beyond the budget, so a hostile file is never
    read to its end.
    """

    digest = hashlib.sha256()
    consumed = 0
    while True:
        chunk = os.read(descriptor, HASH_CHUNK_SIZE)
        if not chunk:
            return digest.hexdigest(), consumed
        consumed += len(chunk)
        if consumed > remaining_bytes:
            return None, consumed
        digest.update(chunk)


def _inspect_leaf(
    parent_fd: int, name: str, record: TrackedPathRecord, budget: _Budget
) -> tuple[RecordVerdict, int]:
    """Inspect and hash one leaf that is a direct child of ``parent_fd``."""

    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return RecordVerdict.MISSING, 0
    except OSError:
        return RecordVerdict.UNREADABLE, 0

    if stat.S_ISLNK(before.st_mode):
        return RecordVerdict.SYMLINK, 0
    if not stat.S_ISREG(before.st_mode):
        return RecordVerdict.WRONG_TYPE, 0
    if budget.remaining_bytes <= 0:
        return RecordVerdict.BUDGET_SKIPPED, 0

    try:
        descriptor = open_file_at(parent_fd, name)
    except FileNotFoundError:
        return RecordVerdict.MISSING, 0
    except OSError:
        # ``ELOOP`` from ``O_NOFOLLOW`` lands here when the entry was replaced
        # by a symlink between the stat and the open.
        return RecordVerdict.UNREADABLE, 0

    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            return RecordVerdict.WRONG_TYPE, 0
        if not same_file_binding(parent_fd, name, descriptor):
            return RecordVerdict.RACED, 0

        digest, consumed = _hash_open_file(descriptor, budget.remaining_bytes)
        if digest is None:
            return RecordVerdict.BUDGET_SKIPPED, consumed

        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            return RecordVerdict.RACED, consumed
        if digest != record.sha256:
            return RecordVerdict.CHANGED, consumed
        return RecordVerdict.CURRENT, consumed
    finally:
        close_quietly(descriptor)


def _inspect_record(
    root_fd: int, record: TrackedPathRecord, budget: _Budget
) -> tuple[RecordVerdict, int]:
    """Walk one record's components from the anchored root descriptor."""

    components = record.path.split("/")
    parents, leaf = components[:-1], components[-1]

    parent_fd = root_fd
    opened: list[int] = []
    try:
        for name in parents:
            try:
                status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return RecordVerdict.MISSING, 0
            except OSError:
                return RecordVerdict.UNREADABLE, 0
            if stat.S_ISLNK(status.st_mode):
                return RecordVerdict.SYMLINK, 0
            if not stat.S_ISDIR(status.st_mode):
                return RecordVerdict.WRONG_TYPE, 0

            try:
                child_fd = open_directory_at(parent_fd, name)
            except FileNotFoundError:
                return RecordVerdict.MISSING, 0
            except OSError:
                return RecordVerdict.UNREADABLE, 0
            opened.append(child_fd)
            if not same_directory_binding(parent_fd, name, child_fd):
                return RecordVerdict.RACED, 0
            parent_fd = child_fd

        return _inspect_leaf(parent_fd, leaf, record, budget)
    finally:
        for descriptor in reversed(opened):
            close_quietly(descriptor)


# --------------------------------------------------------------------------- #
# Snapshot collection                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _LoadedRecords:
    """Bounded record prefix plus proof that additional rows exist."""

    records: tuple[TrackedPathRecord, ...]
    limit_exceeded: bool


def _load_records(project_root: Path) -> _LoadedRecords | None:
    """Read at most the inspection budget plus one sentinel row."""

    from pmem.repositories.sqlite import connect_database_readonly, project_database_path

    try:
        context = require_project_context_readonly(project_root)
        connection = connect_database_readonly(project_database_path(project_root))
    except PmemError:
        return None
    try:
        rows = TrackedPathRepository(connection).list_for_project_limited(
            context.project.id,
            limit=MAX_INSPECTED_RECORDS + 1,
        )
        return _LoadedRecords(
            records=rows[:MAX_INSPECTED_RECORDS],
            limit_exceeded=len(rows) > MAX_INSPECTED_RECORDS,
        )
    except (PmemError, sqlite3.Error):
        return None
    finally:
        connection.close()


def _empty_snapshot(source: SourceState) -> TrackedPathSnapshot:
    return TrackedPathSnapshot(
        source=source,
        record_count=0,
        unsafe_record_count=0,
        symlink_count=0,
        missing_count=0,
        wrong_type_count=0,
        unreadable_count=0,
        raced_count=0,
        budget_skipped_count=0,
        changed_count=0,
        current_count=0,
        record_limit_exceeded=False,
        root_binding_changed=False,
    )


def _root_changed_snapshot(loaded: _LoadedRecords | None = None) -> TrackedPathSnapshot:
    """Return an incomplete snapshot without inspecting a rebound project root."""

    return TrackedPathSnapshot(
        source=SourceState.OK,
        record_count=len(loaded.records) if loaded is not None else 0,
        unsafe_record_count=0,
        symlink_count=0,
        missing_count=0,
        wrong_type_count=0,
        unreadable_count=0,
        raced_count=0,
        budget_skipped_count=0,
        changed_count=0,
        current_count=0,
        record_limit_exceeded=loaded.limit_exceeded if loaded is not None else False,
        root_binding_changed=True,
    )


def collect_tracked_path_snapshot(project_root: str | Path) -> TrackedPathSnapshot:
    """Collect one immutable, content-free tracked-path snapshot."""

    root = Path(project_root)
    if not anchored_traversal_supported():
        return _empty_snapshot(SourceState.NOT_SUPPORTED)

    # Anchor the caller-supplied root before reading config or SQLite.  In
    # particular, this rejects a root that is itself a symlink before a target
    # outside the requested project can be read through it.
    try:
        root_fd = open_directory(root)
    except OSError:
        return _empty_snapshot(SourceState.UNAVAILABLE)

    try:
        if not same_path_binding(root, root_fd):
            return _root_changed_snapshot()

        loaded = _load_records(root)
        if loaded is None:
            return _empty_snapshot(SourceState.UNAVAILABLE)
        if not same_path_binding(root, root_fd):
            return _root_changed_snapshot(loaded)
        if not loaded.records and not loaded.limit_exceeded:
            return _empty_snapshot(SourceState.OK)

        tally: dict[RecordVerdict, int] = {verdict: 0 for verdict in RecordVerdict}
        remaining_bytes = MAX_TOTAL_HASHED_BYTES
        for record in loaded.records:
            if not record_is_safe(record):
                tally[RecordVerdict.UNSAFE_RECORD] += 1
                continue
            verdict, consumed = _inspect_record(
                root_fd, record, _Budget(remaining_bytes=remaining_bytes)
            )
            remaining_bytes = max(0, remaining_bytes - consumed)
            tally[verdict] += 1

        root_changed = not same_path_binding(root, root_fd)
        return TrackedPathSnapshot(
            source=SourceState.OK,
            record_count=len(loaded.records),
            unsafe_record_count=tally[RecordVerdict.UNSAFE_RECORD],
            symlink_count=tally[RecordVerdict.SYMLINK],
            missing_count=tally[RecordVerdict.MISSING],
            wrong_type_count=tally[RecordVerdict.WRONG_TYPE],
            unreadable_count=tally[RecordVerdict.UNREADABLE],
            raced_count=tally[RecordVerdict.RACED],
            budget_skipped_count=tally[RecordVerdict.BUDGET_SKIPPED],
            changed_count=tally[RecordVerdict.CHANGED],
            current_count=tally[RecordVerdict.CURRENT],
            record_limit_exceeded=loaded.limit_exceeded,
            root_binding_changed=root_changed,
        )
    finally:
        close_quietly(root_fd)


# --------------------------------------------------------------------------- #
# Stable, hand-written result text                                             #
# --------------------------------------------------------------------------- #
_SOURCE_UNAVAILABLE_MESSAGE = (
    "Tracked-path records could not be read, so tracked evidence was not judged."
)
_NOT_SUPPORTED_MESSAGE = (
    "Link-safe path inspection does not apply on this platform, so tracked evidence was not judged."
)
_NO_RECORDS_MESSAGE = "No files are tracked yet, so there is no tracked evidence to judge."
_BLOCKED_MESSAGE = "The check did not run because tracked records could not be trusted."
_ROOT_CHANGED_MESSAGE = (
    "The project directory changed identity while tracked evidence was being inspected."
)

_REMEDIATION_UNSAFE_RECORD = (
    "A tracked record no longer describes a safe project-relative file. Review the "
    "tracked list and re-track the affected files from the project directory."
)
_REMEDIATION_SYMLINK = (
    "A tracked location became a link, which projmem refuses to follow. Replace the link "
    "with the real file, then re-track it."
)
_REMEDIATION_MISSING = (
    "Restore the tracked files, or re-track the project so the evidence list matches what "
    "is on disk."
)
_REMEDIATION_WRONG_TYPE = (
    "Something other than a regular file now occupies a tracked location. Restore the "
    "file, then re-track it."
)
_REMEDIATION_CHANGED = (
    "Tracked files changed since they were recorded, so run evidence may no longer match "
    "the code that produced it. Re-track them to record the current contents."
)
_REMEDIATION_INCOMPLETE = (
    "Some tracked files could not be inspected in this pass. Close anything writing to "
    "them and re-run the diagnostic."
)
_REMEDIATION_BUDGET = (
    "There is more tracked evidence than the diagnostic inspects in one pass. Reduce the "
    "number or size of tracked files, then re-run the diagnostic."
)
_REMEDIATION_ROOT_CHANGED = (
    "Re-run the diagnostic without moving or replacing the project directory while it runs."
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
        category=DoctorCategory.TRACKED_PATHS,
        outcome=outcome,
        severity=severity,
        message=message,
        remediation=remediation,
        related_entity_id=None,
    )


def _passed(check_id: str, message: str) -> DoctorCheckResult:
    return _result(check_id, DoctorCheckOutcome.PASS, DoctorSeverity.INFO, message)


def _failed(
    check_id: str,
    message: str,
    remediation: str,
    severity: DoctorSeverity = DoctorSeverity.ERROR,
) -> DoctorCheckResult:
    return _result(check_id, DoctorCheckOutcome.FAIL, severity, message, remediation)


def _not_applicable(check_id: str, message: str) -> DoctorCheckResult:
    return _result(check_id, DoctorCheckOutcome.NOT_APPLICABLE, DoctorSeverity.INFO, message)


def _blocked(check_id: str, message: str = _BLOCKED_MESSAGE) -> DoctorCheckResult:
    """A dependent check that could not run. Never ``pass``."""

    return _result(check_id, DoctorCheckOutcome.SKIPPED, DoctorSeverity.INFO, message)


def _incomplete(check_id: str, message: str, remediation: str) -> DoctorCheckResult:
    """A coverage gap: something could not be established. Never ``pass``."""

    return _result(
        check_id, DoctorCheckOutcome.SKIPPED, DoctorSeverity.WARNING, message, remediation
    )


def _environment_result(check_id: str, snapshot: TrackedPathSnapshot) -> DoctorCheckResult | None:
    """Return the answer shared by every check when nothing could be judged."""

    if snapshot.source is SourceState.NOT_SUPPORTED:
        return _not_applicable(check_id, _NOT_SUPPORTED_MESSAGE)
    if snapshot.source is SourceState.UNAVAILABLE:
        # The database diagnostics own database failures; do not repeat them.
        return _blocked(check_id, _SOURCE_UNAVAILABLE_MESSAGE)
    if snapshot.root_binding_changed:
        return _incomplete(check_id, _ROOT_CHANGED_MESSAGE, _REMEDIATION_ROOT_CHANGED)
    if snapshot.record_count == 0:
        return _not_applicable(check_id, _NO_RECORDS_MESSAGE)
    return None


# --------------------------------------------------------------------------- #
# Snapshot -> result mapping (one function per stable check id)                #
# --------------------------------------------------------------------------- #
def _check_records_safe(snapshot: TrackedPathSnapshot) -> DoctorCheckResult:
    check_id = CHECK_TRACKED_PATHS_RECORDS_SAFE
    environment = _environment_result(check_id, snapshot)
    if environment is not None:
        return environment
    if snapshot.unsafe_record_count:
        return _failed(
            check_id,
            "A stored tracked-path record is not a safe project-relative file reference.",
            _REMEDIATION_UNSAFE_RECORD,
        )
    if snapshot.record_limit_exceeded:
        return _incomplete(
            check_id,
            "There are more tracked records than one diagnostic pass validates.",
            _REMEDIATION_BUDGET,
        )
    return _passed(check_id, "Every tracked-path record is a safe project-relative reference.")


def _check_symlink(snapshot: TrackedPathSnapshot) -> DoctorCheckResult:
    check_id = CHECK_TRACKED_PATHS_SYMLINK
    environment = _environment_result(check_id, snapshot)
    if environment is not None:
        return environment
    if snapshot.unsafe_record_count:
        return _blocked(check_id)
    if snapshot.symlink_count:
        return _failed(
            check_id,
            "A tracked location, or a directory leading to it, is a link.",
            _REMEDIATION_SYMLINK,
        )
    if snapshot.raced_count or snapshot.unreadable_count or snapshot.record_limit_exceeded:
        # A component whose identity changed mid-walk may have been swapped for
        # a link; absence of a link was never established, so this is a gap.
        return _incomplete(
            check_id,
            "A tracked location changed while it was being inspected.",
            _REMEDIATION_INCOMPLETE,
        )
    return _passed(check_id, "No tracked location is reached through a link.")


def _check_present(snapshot: TrackedPathSnapshot) -> DoctorCheckResult:
    check_id = CHECK_TRACKED_PATHS_PRESENT
    environment = _environment_result(check_id, snapshot)
    if environment is not None:
        return environment
    if snapshot.unsafe_record_count or snapshot.symlink_count:
        return _blocked(check_id)
    if snapshot.missing_count:
        return _failed(
            check_id, "A tracked file is missing from the project.", _REMEDIATION_MISSING
        )
    if snapshot.wrong_type_count:
        return _failed(
            check_id,
            "A tracked location is no longer a regular file.",
            _REMEDIATION_WRONG_TYPE,
        )
    if snapshot.budget_skipped_count or snapshot.record_limit_exceeded:
        # The inspected prefix proves nothing about the remainder.
        return _incomplete(
            check_id,
            "There is more tracked evidence than one diagnostic pass inspects.",
            _REMEDIATION_BUDGET,
        )
    if snapshot.unreadable_count or snapshot.raced_count:
        return _incomplete(
            check_id,
            "A tracked file could not be inspected in this pass.",
            _REMEDIATION_INCOMPLETE,
        )
    return _passed(check_id, "Every tracked file is present as a regular file.")


def _check_content_current(snapshot: TrackedPathSnapshot) -> DoctorCheckResult:
    check_id = CHECK_TRACKED_PATHS_CONTENT_CURRENT
    environment = _environment_result(check_id, snapshot)
    if environment is not None:
        return environment
    if (
        snapshot.unsafe_record_count
        or snapshot.symlink_count
        or snapshot.missing_count
        or snapshot.wrong_type_count
    ):
        return _blocked(check_id)
    if snapshot.changed_count:
        # Stale evidence, not corruption: a warning, not an error.
        return _failed(
            check_id,
            "A tracked file no longer matches the contents recorded for it.",
            _REMEDIATION_CHANGED,
            severity=DoctorSeverity.WARNING,
        )
    if snapshot.budget_skipped_count or snapshot.record_limit_exceeded:
        return _incomplete(
            check_id,
            "There is more tracked evidence than one diagnostic pass inspects.",
            _REMEDIATION_BUDGET,
        )
    if snapshot.raced_count or snapshot.unreadable_count:
        return _incomplete(
            check_id,
            "A tracked file changed or could not be read while it was being inspected.",
            _REMEDIATION_INCOMPLETE,
        )
    return _passed(check_id, "Every tracked file still matches the contents recorded for it.")


_ResultBuilder: TypeAlias = Callable[[TrackedPathSnapshot], DoctorCheckResult]

_RESULT_BUILDERS: tuple[tuple[str, _ResultBuilder], ...] = (
    (CHECK_TRACKED_PATHS_CONTENT_CURRENT, _check_content_current),
    (CHECK_TRACKED_PATHS_PRESENT, _check_present),
    (CHECK_TRACKED_PATHS_RECORDS_SAFE, _check_records_safe),
    (CHECK_TRACKED_PATHS_SYMLINK, _check_symlink),
)


# --------------------------------------------------------------------------- #
# Public entry points                                                          #
# --------------------------------------------------------------------------- #
def tracked_path_check_definitions() -> tuple[DoctorCheckDefinition, ...]:
    """Return the four definitions in canonical order.

    Side-effect free: building them reads nothing. Each definition collects its
    own fresh snapshot when executed, so it can never serve a stale
    observation. Callers wanting one consistent view should use
    :func:`run_tracked_path_checks`.
    """

    def _bind(check_id: str, builder: _ResultBuilder) -> DoctorCheckDefinition:
        def _run(context: DoctorCheckContext) -> DoctorCheckResult:
            return builder(collect_tracked_path_snapshot(context.project_root))

        return DoctorCheckDefinition(
            check_id=check_id,
            category=DoctorCategory.TRACKED_PATHS,
            run=_run,
        )

    definitions = tuple(_bind(check_id, builder) for check_id, builder in _RESULT_BUILDERS)
    return tuple(sorted(definitions, key=lambda definition: definition.check_id))


def run_tracked_path_checks(context: DoctorCheckContext) -> tuple[DoctorCheckResult, ...]:
    """Run every tracked-path check against one shared snapshot."""

    snapshot = collect_tracked_path_snapshot(context.project_root)
    results = tuple(builder(snapshot) for _, builder in _RESULT_BUILDERS)
    return tuple(sorted(results, key=lambda result: result.check_id))
