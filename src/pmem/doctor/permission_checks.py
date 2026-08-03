"""Filesystem permission diagnostics for ``pmem doctor`` (DOC-003).

Five checks, expressed entirely through the merged ``doctor-v1`` contract:

``permissions.config``
    the project configuration file is owner-only;
``permissions.database``
    the project database file is owner-only;
``permissions.graph``
    the evidence graph artifact is owner-only, when one exists;
``permissions.pmem_directory``
    the private state directory and the two directories created beside it are
    owner-only;
``permissions.run_artifacts``
    every directory and file recorded under the private run-artifact subtree is
    owner-only.

Read-only by construction. Managed directories are opened read-only with
``O_DIRECTORY | O_NOFOLLOW`` and descendants are inspected relative to those
anchored descriptors. The bounded ``scandir`` traversal reads names only. This
module never chmods, chowns, mkdirs, touches, opens regular-file content,
writes, truncates, renames, unlinks, migrates, initializes, checkpoints SQLite,
removes a sidecar, calls the network, or emits telemetry. Artifact *content* is
never observed -- only identity-safe type, ownership and mode metadata.

Symlinks are never followed. Every managed path component is inspected with
``follow_symlinks=False`` and every traversed directory is opened relative to
an already anchored descriptor with ``O_NOFOLLOW``. ``Path.resolve()`` is never
called. A symlink is reported as a symlink and can never produce a ``pass``;
the target is neither stat-ed, opened, nor named.

Privacy. Every ``message`` and ``remediation`` is a hand-written constant.
Nothing is interpolated -- not a path, not a file name, not a symlink target,
not a project name, not a username, not an entry count, not the mode actually
read from the filesystem, and not an ``OSError``. Observed modes live only
inside the internal snapshot, which is never serialized.

Determinism. One immutable snapshot backs every result of a single
:func:`run_permission_checks` invocation, so no two results can describe
different filesystem states. Nothing is cached between invocations, so a mode
changed between two runs is observed by the second run. Results are returned in
canonical ``check_id`` order regardless of execution order.
"""

from __future__ import annotations

import os
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
from pmem.doctor.registry import DoctorCheckContext, DoctorCheckDefinition
from pmem.graph.persistence import default_graph_artifact_path
from pmem.repositories.sqlite import PMEM_DIRNAME, project_database_path
from pmem.services.config import project_config_path
from pmem.services.project_init import ARTIFACTS_DIRNAME, SNAPSHOTS_DIRNAME

CHECK_PERMISSIONS_CONFIG = "permissions.config"
CHECK_PERMISSIONS_DATABASE = "permissions.database"
CHECK_PERMISSIONS_GRAPH = "permissions.graph"
CHECK_PERMISSIONS_PMEM_DIRECTORY = "permissions.pmem_directory"
CHECK_PERMISSIONS_RUN_ARTIFACTS = "permissions.run_artifacts"

# Canonical (ascending ``check_id``) order. Derived with ``sorted`` by the
# factory rather than trusted from this literal; a test asserts they agree.
PERMISSION_CHECK_IDS: tuple[str, ...] = (
    CHECK_PERMISSIONS_CONFIG,
    CHECK_PERMISSIONS_DATABASE,
    CHECK_PERMISSIONS_GRAPH,
    CHECK_PERMISSIONS_PMEM_DIRECTORY,
    CHECK_PERMISSIONS_RUN_ARTIFACTS,
)

# Owner-only expectations. A private file needs owner read+write; a private
# directory additionally needs owner execute in order to be traversed.
REQUIRED_FILE_MODE = 0o600
REQUIRED_DIRECTORY_MODE = 0o700
GROUP_OTHER_MASK = 0o077

# Traversal budget for the private run-artifact subtree. The tree is bounded in
# practice (one directory per run, a handful of files each), but a budget keeps
# a pathological or hostile tree from turning a diagnostic into a long scan.
# Exceeding it yields an explicitly incomplete result, never a silent pass.
MAX_SCANNED_ENTRIES = 5000


# --------------------------------------------------------------------------- #
# Internal snapshot vocabulary                                                 #
# --------------------------------------------------------------------------- #
class EntryState(Enum):
    """What a no-follow stat revealed about one inspected path."""

    OK = "ok"
    MISSING = "missing"
    SYMLINK = "symlink"
    WRONG_TYPE = "wrong_type"
    UNREADABLE = "unreadable"


class ModeVerdict(Enum):
    """How an observed mode compares with the owner-only expectation."""

    OK = "ok"
    EXPOSED = "exposed"
    OWNER_UNUSABLE = "owner_unusable"


class ScanState(Enum):
    """Outcome of the bounded run-artifact traversal."""

    OK = "ok"
    ROOT_MISSING = "root_missing"
    ROOT_NOT_INSPECTABLE = "root_not_inspectable"
    EMPTY = "empty"
    UNSAFE_ENTRY = "unsafe_entry"
    SYMLINK_ENTRY = "symlink_entry"
    UNLISTABLE = "unlistable"
    BUDGET_EXCEEDED = "budget_exceeded"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class EntryPermission:
    """One inspected path. ``mode`` stays internal and is never serialized."""

    state: EntryState
    mode: int | None
    is_directory: bool
    owner_matches_current_user: bool | None = None

    def verdict(self) -> ModeVerdict:
        """Classify the observed mode against the owner-only expectation."""

        if self.mode is None:
            return ModeVerdict.OWNER_UNUSABLE
        if self.mode & GROUP_OTHER_MASK:
            return ModeVerdict.EXPOSED
        if self.owner_matches_current_user is False:
            return ModeVerdict.OWNER_UNUSABLE
        required = REQUIRED_DIRECTORY_MODE if self.is_directory else REQUIRED_FILE_MODE
        if self.mode & required != required:
            return ModeVerdict.OWNER_UNUSABLE
        return ModeVerdict.OK


@dataclass(frozen=True, slots=True)
class PermissionSnapshot:
    """Immutable view of every path the permission checks care about."""

    posix_modes_supported: bool
    project_root_is_symlink: bool
    inspection_incomplete: bool
    pmem_directory: EntryPermission
    artifacts_directory: EntryPermission
    snapshots_directory: EntryPermission
    database: EntryPermission
    config: EntryPermission
    graph: EntryPermission
    run_artifact_scan: ScanState


def posix_modes_are_supported() -> bool:
    """Return whether POSIX permission bits describe this platform.

    Windows exposes an ``st_mode`` that does not represent its ACLs, so
    inspecting the bits there would produce a confident and wrong answer. This
    is a module-level function rather than an inline ``os.name`` test so tests
    can supply the answer explicitly instead of monkeypatching ``os.name``,
    which would also change ``pathlib`` and ``pytest`` behaviour.
    """

    return (
        os.name == "posix"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "geteuid")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
    )


# --------------------------------------------------------------------------- #
# Snapshot collection                                                          #
# --------------------------------------------------------------------------- #
def _entry_from_stat(status: os.stat_result, *, expect_directory: bool) -> EntryPermission:
    """Classify one no-follow stat result without retaining identity metadata."""

    if stat.S_ISLNK(status.st_mode):
        return EntryPermission(EntryState.SYMLINK, None, expect_directory)
    is_directory = stat.S_ISDIR(status.st_mode)
    is_regular_file = stat.S_ISREG(status.st_mode)
    if expect_directory and not is_directory:
        return EntryPermission(EntryState.WRONG_TYPE, None, expect_directory)
    if not expect_directory and not is_regular_file:
        return EntryPermission(EntryState.WRONG_TYPE, None, expect_directory)
    return EntryPermission(
        EntryState.OK,
        stat.S_IMODE(status.st_mode),
        is_directory,
        status.st_uid == os.geteuid(),
    )


def _inspect_at(parent_fd: int, name: str, *, expect_directory: bool) -> EntryPermission:
    """Inspect one direct child of an already anchored directory."""

    try:
        status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return EntryPermission(EntryState.MISSING, None, expect_directory)
    except OSError:
        return EntryPermission(EntryState.UNREADABLE, None, expect_directory)
    return _entry_from_stat(status, expect_directory=expect_directory)


def _directory_open_flags() -> int:
    """Flags that open a directory itself and reject a symlink at the final component."""

    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_directory(path: Path) -> int:
    return os.open(path, _directory_open_flags())


def _open_directory_at(parent_fd: int, name: str) -> int:
    return os.open(name, _directory_open_flags(), dir_fd=parent_fd)


def _same_directory_binding(parent_fd: int, name: str, child_fd: int) -> bool:
    """Prove ``name`` still identifies the directory held by ``child_fd``."""

    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(child_fd)
    except OSError:
        return False
    return (
        stat.S_ISDIR(named.st_mode)
        and named.st_dev == opened.st_dev
        and named.st_ino == opened.st_ino
    )


def _same_path_binding(path: Path, directory_fd: int) -> bool:
    """Prove an externally supplied root path still names its opened directory."""

    try:
        named = path.lstat()
        opened = os.fstat(directory_fd)
    except OSError:
        return False
    return (
        stat.S_ISDIR(named.st_mode)
        and named.st_dev == opened.st_dev
        and named.st_ino == opened.st_ino
    )


def _bounded_directory_names(directory_fd: int, remaining: int) -> tuple[str, ...] | None:
    """Read at most ``remaining + 1`` names; ``None`` means the budget was exceeded."""

    names: list[str] = []
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            names.append(entry.name)
            if len(names) > remaining:
                return None
    return tuple(sorted(names))


def _scan_open_run_tree(run_root_fd: int) -> ScanState:
    """Inspect an FD-anchored run tree without following a replace-race symlink."""

    pending: list[int] = [run_root_fd]
    open_fds: set[int] = {run_root_fd}
    scanned = 0
    saw_entry = False
    try:
        while pending:
            directory_fd = pending.pop()
            try:
                names = _bounded_directory_names(
                    directory_fd,
                    MAX_SCANNED_ENTRIES - scanned,
                )
            except FileNotFoundError:
                return ScanState.INCOMPLETE
            except OSError:
                return ScanState.UNLISTABLE
            if names is None:
                return ScanState.BUDGET_EXCEEDED

            for name in names:
                scanned += 1
                saw_entry = True
                entry = _inspect_at(directory_fd, name, expect_directory=False)
                if entry.state is EntryState.MISSING:
                    return ScanState.INCOMPLETE
                if entry.state is EntryState.SYMLINK:
                    return ScanState.SYMLINK_ENTRY
                if entry.state is EntryState.UNREADABLE:
                    return ScanState.INCOMPLETE

                try:
                    raw_status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return ScanState.INCOMPLETE
                except OSError:
                    return ScanState.INCOMPLETE

                is_directory = stat.S_ISDIR(raw_status.st_mode)
                is_regular_file = stat.S_ISREG(raw_status.st_mode)
                if not is_directory and not is_regular_file:
                    return ScanState.UNSAFE_ENTRY
                classified = _entry_from_stat(raw_status, expect_directory=is_directory)
                if classified.verdict() is not ModeVerdict.OK:
                    return ScanState.UNSAFE_ENTRY
                if not is_directory:
                    continue

                try:
                    child_fd = _open_directory_at(directory_fd, name)
                except FileNotFoundError:
                    return ScanState.INCOMPLETE
                except OSError:
                    # Includes a symlink substituted after the no-follow stat.
                    return ScanState.INCOMPLETE
                open_fds.add(child_fd)
                if not _same_directory_binding(directory_fd, name, child_fd):
                    return ScanState.INCOMPLETE
                pending.append(child_fd)

            os.close(directory_fd)
            open_fds.remove(directory_fd)
    finally:
        for descriptor in open_fds:
            os.close(descriptor)

    return ScanState.OK if saw_entry else ScanState.EMPTY


def _scan_run_artifacts(pmem_fd: int, artifacts: EntryPermission) -> ScanState:
    """Scan exactly the managed ``artifacts/runs`` subtree through anchored FDs."""

    if artifacts.state is EntryState.MISSING:
        return ScanState.ROOT_MISSING
    if artifacts.state is not EntryState.OK or artifacts.verdict() is not ModeVerdict.OK:
        return ScanState.ROOT_NOT_INSPECTABLE

    try:
        artifacts_fd = _open_directory_at(pmem_fd, ARTIFACTS_DIRNAME)
    except FileNotFoundError:
        return ScanState.INCOMPLETE
    except OSError:
        return ScanState.ROOT_NOT_INSPECTABLE
    try:
        if not _same_directory_binding(pmem_fd, ARTIFACTS_DIRNAME, artifacts_fd):
            return ScanState.INCOMPLETE
        runs = _inspect_at(artifacts_fd, "runs", expect_directory=True)
        if runs.state is EntryState.MISSING:
            return ScanState.ROOT_MISSING
        if runs.state is EntryState.SYMLINK:
            return ScanState.SYMLINK_ENTRY
        if runs.state is EntryState.WRONG_TYPE:
            return ScanState.UNSAFE_ENTRY
        if runs.state is not EntryState.OK:
            return ScanState.INCOMPLETE
        if runs.verdict() is not ModeVerdict.OK:
            return ScanState.UNSAFE_ENTRY
        try:
            runs_fd = _open_directory_at(artifacts_fd, "runs")
        except FileNotFoundError:
            return ScanState.INCOMPLETE
        except OSError:
            return ScanState.INCOMPLETE
        if not _same_directory_binding(artifacts_fd, "runs", runs_fd):
            os.close(runs_fd)
            return ScanState.INCOMPLETE
        return _scan_open_run_tree(runs_fd)
    finally:
        os.close(artifacts_fd)


def collect_permission_snapshot(
    project_root: str | Path,
    *,
    posix_modes_supported: bool | None = None,
) -> PermissionSnapshot:
    """Collect one immutable permission snapshot. Never mutates anything.

    ``posix_modes_supported`` is injectable so the non-POSIX branch is
    deterministically testable. When the platform lacks meaningful permission
    bits or secure descriptor-relative primitives, the snapshot is produced
    without issuing filesystem inspection calls.
    """

    root = Path(project_root)
    supported = (
        posix_modes_are_supported() if posix_modes_supported is None else (posix_modes_supported)
    )
    if not supported:
        return _unsupported_snapshot()

    if root.is_symlink():
        # Every inspected path would be reached through the link. Refuse rather
        # than silently verifying something outside the project.
        return _unsupported_snapshot(posix_modes_supported=True, project_root_is_symlink=True)

    try:
        root_fd = _open_directory(root)
    except OSError:
        is_link = root.is_symlink()
        return _unsupported_snapshot(
            posix_modes_supported=True,
            project_root_is_symlink=is_link,
            inspection_incomplete=not is_link,
        )
    try:
        pmem_directory = _inspect_at(root_fd, PMEM_DIRNAME, expect_directory=True)
        if pmem_directory.state is not EntryState.OK:
            return _descendants_blocked_snapshot(pmem_directory)

        try:
            pmem_fd = _open_directory_at(root_fd, PMEM_DIRNAME)
        except OSError:
            refreshed = _inspect_at(root_fd, PMEM_DIRNAME, expect_directory=True)
            return _descendants_blocked_snapshot(
                refreshed,
                inspection_incomplete=refreshed.state is EntryState.OK,
            )
        try:
            if not _same_directory_binding(root_fd, PMEM_DIRNAME, pmem_fd):
                return _descendants_blocked_snapshot(
                    EntryPermission(EntryState.UNREADABLE, None, True),
                    inspection_incomplete=True,
                )
            artifacts = _inspect_at(pmem_fd, ARTIFACTS_DIRNAME, expect_directory=True)
            snapshots = _inspect_at(pmem_fd, SNAPSHOTS_DIRNAME, expect_directory=True)
            database = _inspect_at(
                pmem_fd, project_database_path(root).name, expect_directory=False
            )
            config = _inspect_at(pmem_fd, project_config_path(root).name, expect_directory=False)
            graph = _inspect_at(
                pmem_fd,
                default_graph_artifact_path(root).name,
                expect_directory=False,
            )
            run_scan = _scan_run_artifacts(pmem_fd, artifacts)
            binding_changed = not _same_path_binding(root, root_fd) or not _same_directory_binding(
                root_fd, PMEM_DIRNAME, pmem_fd
            )
            return PermissionSnapshot(
                posix_modes_supported=True,
                project_root_is_symlink=False,
                inspection_incomplete=binding_changed,
                pmem_directory=pmem_directory,
                artifacts_directory=artifacts,
                snapshots_directory=snapshots,
                database=database,
                config=config,
                graph=graph,
                run_artifact_scan=(ScanState.INCOMPLETE if binding_changed else run_scan),
            )
        finally:
            os.close(pmem_fd)
    finally:
        os.close(root_fd)


def _descendants_blocked_snapshot(
    pmem_directory: EntryPermission,
    *,
    inspection_incomplete: bool = False,
) -> PermissionSnapshot:
    """Return without touching descendants when the managed parent is unsafe."""

    missing = pmem_directory.state is EntryState.MISSING
    file_state = EntryState.MISSING if missing else EntryState.UNREADABLE
    directory_state = EntryState.MISSING if missing else EntryState.UNREADABLE
    return PermissionSnapshot(
        posix_modes_supported=True,
        project_root_is_symlink=False,
        inspection_incomplete=inspection_incomplete,
        pmem_directory=pmem_directory,
        artifacts_directory=EntryPermission(directory_state, None, True),
        snapshots_directory=EntryPermission(directory_state, None, True),
        database=EntryPermission(file_state, None, False),
        config=EntryPermission(file_state, None, False),
        graph=EntryPermission(file_state, None, False),
        run_artifact_scan=(ScanState.ROOT_MISSING if missing else ScanState.ROOT_NOT_INSPECTABLE),
    )


def _unsupported_snapshot(
    *,
    posix_modes_supported: bool = False,
    project_root_is_symlink: bool = False,
    inspection_incomplete: bool = False,
) -> PermissionSnapshot:
    unreadable = EntryPermission(EntryState.UNREADABLE, None, False)
    unreadable_directory = EntryPermission(EntryState.UNREADABLE, None, True)
    return PermissionSnapshot(
        posix_modes_supported=posix_modes_supported,
        project_root_is_symlink=project_root_is_symlink,
        inspection_incomplete=inspection_incomplete,
        pmem_directory=unreadable_directory,
        artifacts_directory=unreadable_directory,
        snapshots_directory=unreadable_directory,
        database=unreadable,
        config=unreadable,
        graph=unreadable,
        run_artifact_scan=ScanState.ROOT_NOT_INSPECTABLE,
    )


# --------------------------------------------------------------------------- #
# Stable, hand-written result text                                             #
# --------------------------------------------------------------------------- #
_NOT_APPLICABLE_MESSAGE = (
    "Permission-bit inspection does not apply on this platform, so access was not judged."
)
_ROOT_SYMLINK_MESSAGE = (
    "The project directory is reached through a link, so permissions were not judged."
)
_INCOMPLETE_INSPECTION_MESSAGE = (
    "The private project state changed during inspection, so permissions were not judged."
)
_NOT_INITIALIZED_MESSAGE = (
    "The private project state directory does not exist, so permissions were not judged."
)
_BLOCKED_MESSAGE = "The check did not run because the private state directory was not inspectable."

_REMEDIATION_RESTRICT_DIRECTORY = (
    "Restrict the private project state directories to the current user only, so that "
    "no other account can list or enter them."
)
_REMEDIATION_RESTRICT_RUN_ARTIFACTS = (
    "Restrict every stored run artifact file and directory to the current user only, then "
    "re-run the diagnostic."
)
_REMEDIATION_RESTRICT_FILE = (
    "Restrict the file to the current user only, so that no other account can read or modify it."
)
_REMEDIATION_OWNER_DIRECTORY = (
    "Grant the current user read, write and traverse access to the private project state "
    "directories, which projmem needs in order to use them."
)
_REMEDIATION_OWNER_FILE = (
    "Grant the current user read and write access to the file, which projmem needs in "
    "order to use it."
)
_REMEDIATION_REPLACE_LINK = (
    "Replace the link with the real directory or file, then re-run the diagnostic. "
    "projmem refuses to judge state it would have to follow a link to reach."
)
_REMEDIATION_UNEXPECTED_TYPE = (
    "Something other than the expected directory or file occupies this location. Move it "
    "aside and let projmem recreate its own state."
)
_REMEDIATION_UNREADABLE = (
    "Grant the current user permission to inspect the private project state, then re-run "
    "the diagnostic."
)
_REMEDIATION_UNLISTABLE = (
    "Grant the current user permission to list the private run-artifact directories, then "
    "re-run the diagnostic."
)
_REMEDIATION_BUDGET = (
    "The run-artifact tree is larger than the diagnostic inspects in one pass. Archive or "
    "remove old runs, then re-run the diagnostic."
)
_REMEDIATION_RETRY = "Wait for other local project activity to finish, then re-run the diagnostic."
_REMEDIATION_RESTORE_DIRECTORIES = (
    "Reinitialize or restore the required private project state directories, then re-run "
    "the diagnostic."
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
        category=DoctorCategory.PERMISSIONS,
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


def _not_applicable(check_id: str, message: str) -> DoctorCheckResult:
    return _result(check_id, DoctorCheckOutcome.NOT_APPLICABLE, DoctorSeverity.INFO, message)


def _blocked(check_id: str) -> DoctorCheckResult:
    """A check that could not run because a prerequisite was not inspectable.

    ``skipped`` at ``info`` severity: the prerequisite already reported the one
    actionable problem, and the outcome is never ``pass``, so the report cannot
    claim a check succeeded when it did not run.
    """

    return _result(check_id, DoctorCheckOutcome.SKIPPED, DoctorSeverity.INFO, _BLOCKED_MESSAGE)


def _environment_result(check_id: str, snapshot: PermissionSnapshot) -> DoctorCheckResult | None:
    """Return the platform/root-level answer shared by every check, if any."""

    if not snapshot.posix_modes_supported:
        return _not_applicable(check_id, _NOT_APPLICABLE_MESSAGE)
    if snapshot.project_root_is_symlink:
        return _result(
            check_id,
            DoctorCheckOutcome.SKIPPED,
            DoctorSeverity.WARNING,
            _ROOT_SYMLINK_MESSAGE,
            _REMEDIATION_REPLACE_LINK,
        )
    if snapshot.inspection_incomplete:
        return _result(
            check_id,
            DoctorCheckOutcome.SKIPPED,
            DoctorSeverity.WARNING,
            _INCOMPLETE_INSPECTION_MESSAGE,
            _REMEDIATION_RETRY,
        )
    return None


def _entry_failure(
    check_id: str,
    entry: EntryPermission,
    *,
    exposed_message: str,
    owner_message: str,
    symlink_message: str,
    wrong_type_message: str,
    unreadable_message: str,
) -> DoctorCheckResult | None:
    """Map a non-``pass`` entry onto a stable result, or ``None`` when healthy."""

    if entry.state is EntryState.SYMLINK:
        return _failed(check_id, symlink_message, _REMEDIATION_REPLACE_LINK)
    if entry.state is EntryState.WRONG_TYPE:
        return _failed(check_id, wrong_type_message, _REMEDIATION_UNEXPECTED_TYPE)
    if entry.state is EntryState.UNREADABLE:
        return _failed(check_id, unreadable_message, _REMEDIATION_UNREADABLE)

    verdict = entry.verdict()
    if verdict is ModeVerdict.EXPOSED:
        remediation = (
            _REMEDIATION_RESTRICT_DIRECTORY if entry.is_directory else _REMEDIATION_RESTRICT_FILE
        )
        return _failed(check_id, exposed_message, remediation)
    if verdict is ModeVerdict.OWNER_UNUSABLE:
        remediation = (
            _REMEDIATION_OWNER_DIRECTORY if entry.is_directory else _REMEDIATION_OWNER_FILE
        )
        return _failed(check_id, owner_message, remediation)
    return None


# --------------------------------------------------------------------------- #
# Snapshot -> result mapping (one function per stable check id)                #
# --------------------------------------------------------------------------- #
def _check_pmem_directory(snapshot: PermissionSnapshot) -> DoctorCheckResult:
    check_id = CHECK_PERMISSIONS_PMEM_DIRECTORY
    environment = _environment_result(check_id, snapshot)
    if environment is not None:
        return environment

    if snapshot.pmem_directory.state is EntryState.MISSING:
        # Not a permission defect: the project simply is not initialized. The
        # database diagnostics own that failure; repeating it here would add a
        # second alarm with no distinct action.
        return _result(
            check_id,
            DoctorCheckOutcome.SKIPPED,
            DoctorSeverity.WARNING,
            _NOT_INITIALIZED_MESSAGE,
        )

    for entry in (
        snapshot.pmem_directory,
        snapshot.artifacts_directory,
        snapshot.snapshots_directory,
    ):
        if entry.state is EntryState.MISSING:
            return _result(
                check_id,
                DoctorCheckOutcome.SKIPPED,
                DoctorSeverity.WARNING,
                "A required private project state directory does not exist, so its permissions "
                "were not judged.",
                _REMEDIATION_RESTORE_DIRECTORIES,
            )
        failure = _entry_failure(
            check_id,
            entry,
            exposed_message=(
                "A private project state directory can be entered or listed by other "
                "accounts on this machine."
            ),
            owner_message=(
                "A private project state directory cannot be read or traversed by the current user."
            ),
            symlink_message=(
                "A private project state directory is a link, which projmem refuses to follow."
            ),
            wrong_type_message=(
                "Something other than a directory occupies a private project state location."
            ),
            unreadable_message=("A private project state directory could not be inspected."),
        )
        if failure is not None:
            return failure

    return _passed(
        check_id, "The private project state directories are restricted to the current user."
    )


def _private_file_check(
    check_id: str,
    entry: EntryPermission,
    snapshot: PermissionSnapshot,
    *,
    missing_result: Callable[[str], DoctorCheckResult],
    exposed_message: str,
    owner_message: str,
    symlink_message: str,
    wrong_type_message: str,
    unreadable_message: str,
    passed_message: str,
) -> DoctorCheckResult:
    environment = _environment_result(check_id, snapshot)
    if environment is not None:
        return environment
    if snapshot.pmem_directory.state is not EntryState.OK:
        return _blocked(check_id)
    if entry.state is EntryState.MISSING:
        return missing_result(check_id)

    failure = _entry_failure(
        check_id,
        entry,
        exposed_message=exposed_message,
        owner_message=owner_message,
        symlink_message=symlink_message,
        wrong_type_message=wrong_type_message,
        unreadable_message=unreadable_message,
    )
    return failure if failure is not None else _passed(check_id, passed_message)


def _check_database(snapshot: PermissionSnapshot) -> DoctorCheckResult:
    return _private_file_check(
        CHECK_PERMISSIONS_DATABASE,
        snapshot.database,
        snapshot,
        missing_result=lambda check_id: _result(
            check_id,
            DoctorCheckOutcome.SKIPPED,
            DoctorSeverity.INFO,
            "The project database does not exist, so its permissions were not judged.",
        ),
        exposed_message="The project database can be read or modified by other accounts.",
        owner_message="The project database cannot be read or written by the current user.",
        symlink_message="The project database is a link, which projmem refuses to follow.",
        wrong_type_message="Something other than a regular file occupies the database location.",
        unreadable_message="The project database permissions could not be inspected.",
        passed_message="The project database is restricted to the current user.",
    )


def _check_config(snapshot: PermissionSnapshot) -> DoctorCheckResult:
    return _private_file_check(
        CHECK_PERMISSIONS_CONFIG,
        snapshot.config,
        snapshot,
        missing_result=lambda check_id: _result(
            check_id,
            DoctorCheckOutcome.SKIPPED,
            DoctorSeverity.INFO,
            "The project configuration does not exist, so its permissions were not judged.",
        ),
        exposed_message="The project configuration can be read or modified by other accounts.",
        owner_message=("The project configuration cannot be read or written by the current user."),
        symlink_message="The project configuration is a link, which projmem refuses to follow.",
        wrong_type_message=(
            "Something other than a regular file occupies the configuration location."
        ),
        unreadable_message="The project configuration permissions could not be inspected.",
        passed_message="The project configuration is restricted to the current user.",
    )


def _check_graph(snapshot: PermissionSnapshot) -> DoctorCheckResult:
    return _private_file_check(
        CHECK_PERMISSIONS_GRAPH,
        snapshot.graph,
        snapshot,
        # An evidence graph that was never built is normal, not a defect.
        missing_result=lambda check_id: _not_applicable(
            check_id,
            "No evidence graph has been built yet, so there are no permissions to judge.",
        ),
        exposed_message="The evidence graph can be read or modified by other accounts.",
        owner_message="The evidence graph cannot be read or written by the current user.",
        symlink_message="The evidence graph is a link, which projmem refuses to follow.",
        wrong_type_message=(
            "Something other than a regular file occupies the evidence graph location."
        ),
        unreadable_message="The evidence graph permissions could not be inspected.",
        passed_message="The evidence graph is restricted to the current user.",
    )


def _check_run_artifacts(snapshot: PermissionSnapshot) -> DoctorCheckResult:
    check_id = CHECK_PERMISSIONS_RUN_ARTIFACTS
    environment = _environment_result(check_id, snapshot)
    if environment is not None:
        return environment
    if snapshot.pmem_directory.state is not EntryState.OK:
        return _blocked(check_id)
    if (
        snapshot.artifacts_directory.state is not EntryState.OK
        or snapshot.artifacts_directory.verdict() is not ModeVerdict.OK
    ):
        return _blocked(check_id)

    scan = snapshot.run_artifact_scan
    if scan is ScanState.ROOT_MISSING or scan is ScanState.EMPTY:
        # No run has stored an artifact yet. Nothing to judge, not a failure.
        return _not_applicable(
            check_id, "No run artifacts have been stored yet, so there are none to judge."
        )
    if scan is ScanState.ROOT_NOT_INSPECTABLE:
        return _blocked(check_id)
    if scan is ScanState.UNLISTABLE:
        return _failed(
            check_id,
            "A private run-artifact directory could not be listed.",
            _REMEDIATION_UNLISTABLE,
        )
    if scan is ScanState.BUDGET_EXCEEDED:
        # Explicitly incomplete: the prefix that was inspected proves nothing
        # about the rest, so this must never read as a pass.
        return _result(
            check_id,
            DoctorCheckOutcome.SKIPPED,
            DoctorSeverity.WARNING,
            "The run-artifact tree is too large to inspect fully in one diagnostic pass.",
            _REMEDIATION_BUDGET,
        )
    if scan is ScanState.INCOMPLETE:
        return _result(
            check_id,
            DoctorCheckOutcome.SKIPPED,
            DoctorSeverity.WARNING,
            _INCOMPLETE_INSPECTION_MESSAGE,
            _REMEDIATION_RETRY,
        )
    if scan is ScanState.SYMLINK_ENTRY:
        return _failed(
            check_id,
            "A stored run artifact is a link, which projmem refuses to follow.",
            _REMEDIATION_REPLACE_LINK,
        )
    if scan is ScanState.UNSAFE_ENTRY:
        return _failed(
            check_id,
            "A stored run artifact is reachable by other accounts, is a link, or could not "
            "be inspected.",
            _REMEDIATION_RESTRICT_RUN_ARTIFACTS,
        )
    return _passed(check_id, "Stored run artifacts are restricted to the current user.")


_ResultBuilder: TypeAlias = Callable[[PermissionSnapshot], DoctorCheckResult]

_RESULT_BUILDERS: tuple[tuple[str, _ResultBuilder], ...] = (
    (CHECK_PERMISSIONS_CONFIG, _check_config),
    (CHECK_PERMISSIONS_DATABASE, _check_database),
    (CHECK_PERMISSIONS_GRAPH, _check_graph),
    (CHECK_PERMISSIONS_PMEM_DIRECTORY, _check_pmem_directory),
    (CHECK_PERMISSIONS_RUN_ARTIFACTS, _check_run_artifacts),
)


# --------------------------------------------------------------------------- #
# Public entry points                                                          #
# --------------------------------------------------------------------------- #
def permission_check_definitions() -> tuple[DoctorCheckDefinition, ...]:
    """Return the five permission check definitions in canonical order.

    Deterministic and side-effect free: calling it inspects nothing. Each
    definition collects its **own** fresh snapshot when executed, so a
    definition can never serve a stale observation. Callers that want all five
    results to describe one consistent filesystem state should use
    :func:`run_permission_checks` instead.
    """

    def _bind(check_id: str, builder: _ResultBuilder) -> DoctorCheckDefinition:
        def _run(context: DoctorCheckContext) -> DoctorCheckResult:
            return builder(collect_permission_snapshot(context.project_root))

        return DoctorCheckDefinition(
            check_id=check_id,
            category=DoctorCategory.PERMISSIONS,
            run=_run,
        )

    definitions = tuple(_bind(check_id, builder) for check_id, builder in _RESULT_BUILDERS)
    return tuple(sorted(definitions, key=lambda definition: definition.check_id))


def run_permission_checks(context: DoctorCheckContext) -> tuple[DoctorCheckResult, ...]:
    """Run every permission check against one shared snapshot.

    This is the entry point that guarantees internal consistency: all five
    results describe the same observation of the filesystem, so the report can
    never say the state directory is safe while claiming its contents are not.
    Nothing is retained between calls, so a mode changed after this returns is
    observed by the next invocation.
    """

    snapshot = collect_permission_snapshot(context.project_root)
    results = tuple(builder(snapshot) for _, builder in _RESULT_BUILDERS)
    return tuple(sorted(results, key=lambda result: result.check_id))
