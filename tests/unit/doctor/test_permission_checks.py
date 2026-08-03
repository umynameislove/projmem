"""Permission doctor check tests (DOC-003).

Behavioural cases build real directory trees and set real modes. Narrow fault
injection is used only for races and OS failures that cannot be reproduced
deterministically with portable filesystem operations.

Note on fixtures: ``init_project`` currently creates its directories with a
plain ``mkdir`` and no ``chmod``, so a freshly initialized project inherits the
umask (0755 under the default 0022). The healthy fixtures below therefore build
the owner-only tree explicitly. That is a *test* fixture choice, not a change to
any writer -- DOC-003 is a diagnostic and must keep reporting the real baseline
honestly.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import pmem.doctor.permission_checks as permission_module
from pmem.doctor import (
    DoctorCategory,
    DoctorCheckContext,
    DoctorCheckOutcome,
    DoctorCheckResult,
    DoctorSeverity,
)
from pmem.doctor.permission_checks import (
    MAX_SCANNED_ENTRIES,
    PERMISSION_CHECK_IDS,
    EntryState,
    ModeVerdict,
    PermissionSnapshot,
    ScanState,
    collect_permission_snapshot,
    permission_check_definitions,
    run_permission_checks,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="POSIX permission bits are required for these cases"
)

_CONFIG = "permissions.config"
_DATABASE = "permissions.database"
_GRAPH = "permissions.graph"
_PMEM_DIR = "permissions.pmem_directory"
_RUN_ARTIFACTS = "permissions.run_artifacts"


# --------------------------------------------------------------------------- #
# Fixtures that build real trees with real modes                               #
# --------------------------------------------------------------------------- #
def _private_file(path: Path, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    path.chmod(mode)
    return path


def _private_dir(path: Path, mode: int = 0o700) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(mode)
    return path


def _healthy_project(root: Path, *, with_graph: bool = True, with_run: bool = True) -> Path:
    """Build an owner-only project tree by hand (see module docstring)."""

    pmem = _private_dir(root / ".pmem")
    _private_dir(pmem / "artifacts")
    _private_dir(pmem / "snapshots")
    _private_file(pmem / "pmem.db")
    _private_file(pmem / "config.yaml")
    if with_graph:
        _private_file(pmem / "graph.json")
    if with_run:
        run_dir = _private_dir(pmem / "artifacts" / "runs")
        one_run = _private_dir(run_dir / "run_0123456789abcdef")
        _private_file(one_run / "stdout.txt")
        _private_file(one_run / "stderr.txt")
    return root


def _results(root: Path) -> dict[str, DoctorCheckResult]:
    context = DoctorCheckContext(project_root=root)
    return {result.check_id: result for result in run_permission_checks(context)}


def _pairs(root: Path) -> dict[str, tuple[str, str]]:
    return {
        check_id: (result.outcome.value, result.severity.value)
        for check_id, result in _results(root).items()
    }


# --------------------------------------------------------------------------- #
# A. Contract / factory                                                        #
# --------------------------------------------------------------------------- #
def test_factory_returns_the_five_stable_ids_in_canonical_order() -> None:
    definitions = permission_check_definitions()

    assert tuple(d.check_id for d in definitions) == PERMISSION_CHECK_IDS
    assert PERMISSION_CHECK_IDS == tuple(sorted(PERMISSION_CHECK_IDS))
    assert set(PERMISSION_CHECK_IDS) == {
        _CONFIG,
        _DATABASE,
        _GRAPH,
        _PMEM_DIR,
        _RUN_ARTIFACTS,
    }


def test_every_definition_is_a_permissions_definition() -> None:
    for definition in permission_check_definitions():
        assert definition.category is DoctorCategory.PERMISSIONS
        assert definition.check_id.startswith("permissions.")


def test_definition_and_result_identity_match(tmp_path: Path) -> None:
    context = DoctorCheckContext(project_root=_healthy_project(tmp_path))

    for definition in permission_check_definitions():
        result = definition.execute(context)
        assert result.check_id == definition.check_id
        assert result.category is definition.category


def test_two_factories_share_nothing(tmp_path: Path) -> None:
    first = permission_check_definitions()
    second = permission_check_definitions()

    assert first is not second
    assert [d.check_id for d in first] == [d.check_id for d in second]
    assert all(a is not b for a, b in zip(first, second, strict=True))


def test_module_exposes_no_mutable_registry() -> None:
    for name in ("_REGISTRY", "REGISTRY", "_CACHE", "CACHE", "_SNAPSHOT"):
        assert not hasattr(permission_module, name), name


def test_factory_and_import_perform_no_filesystem_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("building definitions must not touch the filesystem")

    monkeypatch.setattr(Path, "lstat", _forbidden)
    monkeypatch.setattr(os, "scandir", _forbidden)
    monkeypatch.setattr(permission_module, "_open_directory", _forbidden)
    monkeypatch.setattr(permission_module, "_inspect_at", _forbidden)

    definitions = permission_check_definitions()

    assert len(definitions) == 5


def test_definitions_do_not_hold_a_stale_snapshot(tmp_path: Path) -> None:
    """A definition reused after a mode change must observe the new mode."""

    root = _healthy_project(tmp_path)
    context = DoctorCheckContext(project_root=root)
    definition = next(d for d in permission_check_definitions() if d.check_id == _PMEM_DIR)

    assert definition.execute(context).outcome is DoctorCheckOutcome.PASS
    (root / ".pmem").chmod(0o755)
    assert definition.execute(context).outcome is DoctorCheckOutcome.FAIL


# --------------------------------------------------------------------------- #
# B. POSIX decision table                                                      #
# --------------------------------------------------------------------------- #
def test_healthy_project_passes_every_check(tmp_path: Path) -> None:
    results = _results(_healthy_project(tmp_path))

    assert set(results) == set(PERMISSION_CHECK_IDS)
    for check_id, result in results.items():
        assert result.outcome is DoctorCheckOutcome.PASS, check_id
        assert result.severity is DoctorSeverity.INFO, check_id
        assert result.remediation is None, check_id


@pytest.mark.parametrize("mode", [0o644, 0o660, 0o666, 0o604, 0o640, 0o606])
@pytest.mark.parametrize(
    ("relative", "check_id"),
    [
        (".pmem/pmem.db", _DATABASE),
        (".pmem/config.yaml", _CONFIG),
        (".pmem/graph.json", _GRAPH),
    ],
)
def test_group_or_other_readable_file_fails(
    tmp_path: Path, relative: str, check_id: str, mode: int
) -> None:
    root = _healthy_project(tmp_path)
    (root / relative).chmod(mode)

    result = _results(root)[check_id]

    assert result.outcome is DoctorCheckOutcome.FAIL
    assert result.severity is DoctorSeverity.ERROR
    assert result.remediation is not None


@pytest.mark.parametrize("mode", [0o400, 0o200, 0o000])
@pytest.mark.parametrize(
    ("relative", "check_id"),
    [
        (".pmem/pmem.db", _DATABASE),
        (".pmem/config.yaml", _CONFIG),
        (".pmem/graph.json", _GRAPH),
    ],
)
def test_owner_missing_file_rights_fails_as_unusable(
    tmp_path: Path, relative: str, check_id: str, mode: int
) -> None:
    root = _healthy_project(tmp_path)
    target = root / relative
    target.chmod(mode)
    try:
        result = _results(root)[check_id]
    finally:
        target.chmod(0o600)

    assert result.outcome is DoctorCheckOutcome.FAIL
    # a usability problem, not an exposure: the remediation must say so
    assert "Grant the current user" in (result.remediation or "")


@pytest.mark.parametrize("mode", [0o755, 0o770, 0o777, 0o705, 0o750])
@pytest.mark.parametrize("relative", [".pmem", ".pmem/artifacts", ".pmem/snapshots"])
def test_group_or_other_accessible_directory_fails(
    tmp_path: Path, relative: str, mode: int
) -> None:
    root = _healthy_project(tmp_path)
    (root / relative).chmod(mode)

    result = _results(root)[_PMEM_DIR]

    assert result.outcome is DoctorCheckOutcome.FAIL
    assert result.severity is DoctorSeverity.ERROR
    assert result.remediation is not None


@pytest.mark.parametrize("mode", [0o600, 0o500, 0o000])
def test_directory_missing_owner_traverse_fails_as_unusable(tmp_path: Path, mode: int) -> None:
    root = _healthy_project(tmp_path)
    target = root / ".pmem" / "snapshots"
    target.chmod(mode)
    try:
        result = _results(root)[_PMEM_DIR]
    finally:
        target.chmod(0o700)

    assert result.outcome is DoctorCheckOutcome.FAIL
    assert "Grant the current user" in (result.remediation or "")


@pytest.mark.parametrize("mode", [0o755, 0o770, 0o777])
def test_unsafe_run_directory_fails(tmp_path: Path, mode: int) -> None:
    root = _healthy_project(tmp_path)
    (root / ".pmem" / "artifacts" / "runs" / "run_0123456789abcdef").chmod(mode)

    result = _results(root)[_RUN_ARTIFACTS]

    assert result.outcome is DoctorCheckOutcome.FAIL
    assert result.severity is DoctorSeverity.ERROR


@pytest.mark.parametrize("mode", [0o644, 0o666, 0o604])
def test_unsafe_run_file_fails(tmp_path: Path, mode: int) -> None:
    root = _healthy_project(tmp_path)
    stdout = root / ".pmem" / "artifacts" / "runs" / "run_0123456789abcdef" / "stdout.txt"
    stdout.chmod(mode)

    result = _results(root)[_RUN_ARTIFACTS]

    assert result.outcome is DoctorCheckOutcome.FAIL


def test_one_unsafe_entry_in_a_large_tree_still_fails(tmp_path: Path) -> None:
    """A single exposed leaf must not be diluted by many safe siblings."""

    root = _healthy_project(tmp_path)
    runs = root / ".pmem" / "artifacts" / "runs"
    for index in range(25):
        run_dir = _private_dir(runs / f"run_{index:016x}")
        _private_file(run_dir / "stdout.txt")
        _private_file(run_dir / "stderr.txt")
    assert _results(root)[_RUN_ARTIFACTS].outcome is DoctorCheckOutcome.PASS

    (runs / "run_000000000000000c" / "stderr.txt").chmod(0o644)

    assert _results(root)[_RUN_ARTIFACTS].outcome is DoctorCheckOutcome.FAIL


# --------------------------------------------------------------------------- #
# B (continued). Missing and optional state                                    #
# --------------------------------------------------------------------------- #
def test_uninitialized_project_skips_without_erroring(tmp_path: Path) -> None:
    pairs = _pairs(tmp_path)

    assert pairs[_PMEM_DIR] == ("skipped", "warning")
    for check_id in (_CONFIG, _DATABASE, _GRAPH, _RUN_ARTIFACTS):
        assert pairs[check_id][0] == "skipped", check_id
    assert not (tmp_path / ".pmem").exists()


def test_missing_graph_is_not_applicable_not_a_failure(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path, with_graph=False)

    result = _results(root)[_GRAPH]

    assert result.outcome is DoctorCheckOutcome.NOT_APPLICABLE
    assert result.severity is DoctorSeverity.INFO
    assert result.remediation is None


def test_missing_run_artifacts_is_not_applicable_not_a_failure(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path, with_run=False)

    result = _results(root)[_RUN_ARTIFACTS]

    assert result.outcome is DoctorCheckOutcome.NOT_APPLICABLE
    assert result.remediation is None


def test_unrelated_artifact_does_not_count_as_a_run_artifact(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path, with_run=False)
    _private_file(root / ".pmem" / "artifacts" / "unrelated.bin")

    result = _results(root)[_RUN_ARTIFACTS]

    assert result.outcome is DoctorCheckOutcome.NOT_APPLICABLE
    assert result.remediation is None


def test_empty_artifacts_and_snapshots_directories_are_not_failures(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path, with_run=False)

    results = _results(root)

    assert results[_PMEM_DIR].outcome is DoctorCheckOutcome.PASS
    assert results[_RUN_ARTIFACTS].outcome is DoctorCheckOutcome.NOT_APPLICABLE


@pytest.mark.parametrize(
    ("relative", "check_id"),
    [(".pmem/pmem.db", _DATABASE), (".pmem/config.yaml", _CONFIG)],
)
def test_missing_mandatory_file_skips_rather_than_duplicating_an_error(
    tmp_path: Path, relative: str, check_id: str
) -> None:
    root = _healthy_project(tmp_path)
    (root / relative).unlink()

    result = _results(root)[check_id]

    assert result.outcome is DoctorCheckOutcome.SKIPPED
    assert result.severity is DoctorSeverity.INFO


def test_downstream_checks_do_not_pass_when_state_directory_is_unsafe(tmp_path: Path) -> None:
    """A missing prerequisite must never let a dependent check read as pass."""

    root = tmp_path
    (root / ".pmem").mkdir()
    # no config/database/graph, .pmem left at an exposed mode
    (root / ".pmem").chmod(0o755)

    results = _results(root)

    assert results[_PMEM_DIR].outcome is DoctorCheckOutcome.FAIL
    for check_id in (_CONFIG, _DATABASE, _GRAPH, _RUN_ARTIFACTS):
        assert results[check_id].outcome is not DoctorCheckOutcome.PASS, check_id


# --------------------------------------------------------------------------- #
# C. Platform fallback                                                         #
# --------------------------------------------------------------------------- #
def test_non_posix_returns_not_applicable_for_every_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission_module, "posix_modes_are_supported", lambda: False)

    results = _results(_healthy_project(tmp_path))

    assert set(results) == set(PERMISSION_CHECK_IDS)
    for check_id, result in results.items():
        assert result.outcome is DoctorCheckOutcome.NOT_APPLICABLE, check_id
        assert result.severity is DoctorSeverity.INFO, check_id
        assert result.remediation is None, check_id
        assert "does not apply on this platform" in result.message


def test_non_posix_never_pretends_to_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(permission_module, "posix_modes_are_supported", lambda: False)

    for result in run_permission_checks(DoctorCheckContext(project_root=tmp_path)):
        assert result.outcome is not DoctorCheckOutcome.PASS


def test_non_posix_performs_no_filesystem_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission_module, "posix_modes_are_supported", lambda: False)

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("the non-POSIX branch must not inspect the filesystem")

    monkeypatch.setattr(Path, "lstat", _forbidden)
    monkeypatch.setattr(Path, "is_symlink", _forbidden)
    monkeypatch.setattr(os, "scandir", _forbidden)
    monkeypatch.setattr(permission_module, "_open_directory", _forbidden)
    monkeypatch.setattr(permission_module, "_inspect_at", _forbidden)

    results = run_permission_checks(DoctorCheckContext(project_root=tmp_path))

    assert len(results) == 5


def test_non_posix_output_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(permission_module, "posix_modes_are_supported", lambda: False)
    context = DoctorCheckContext(project_root=tmp_path)

    assert run_permission_checks(context) == run_permission_checks(context)


def test_injected_platform_flag_overrides_detection(tmp_path: Path) -> None:
    snapshot = collect_permission_snapshot(tmp_path, posix_modes_supported=False)

    assert snapshot.posix_modes_supported is False
    assert isinstance(snapshot, PermissionSnapshot)


# --------------------------------------------------------------------------- #
# D. Symlinks (behaviour; the security suite proves targets stay untouched)    #
# --------------------------------------------------------------------------- #
def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")


def test_symlinked_project_root_skips_every_check(tmp_path: Path) -> None:
    real = _healthy_project(tmp_path / "real")
    link = tmp_path / "linked"
    _symlink_or_skip(link, real)

    results = _results(link)

    for check_id, result in results.items():
        assert result.outcome is DoctorCheckOutcome.SKIPPED, check_id
        assert result.severity is DoctorSeverity.WARNING, check_id
        assert result.outcome is not DoctorCheckOutcome.PASS


def test_symlinked_state_directory_fails(tmp_path: Path) -> None:
    outside = _private_dir(tmp_path.parent / "outside-pmem-dir")
    _symlink_or_skip(tmp_path / ".pmem", outside)

    results = _results(tmp_path)
    snapshot = collect_permission_snapshot(tmp_path)

    assert results[_PMEM_DIR].outcome is DoctorCheckOutcome.FAIL
    assert "is a link" in results[_PMEM_DIR].message
    assert snapshot.pmem_directory.state is EntryState.SYMLINK
    for check_id in (_CONFIG, _DATABASE, _GRAPH, _RUN_ARTIFACTS):
        assert results[check_id].outcome is not DoctorCheckOutcome.PASS, check_id


def test_symlinked_state_directory_short_circuits_every_descendant_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = _private_dir(tmp_path.parent / "outside-no-read")
    _private_file(outside / "pmem.db")
    _symlink_or_skip(tmp_path / ".pmem", outside)
    real_inspect = permission_module._inspect_at
    inspected_names: list[str] = []

    def _record(parent_fd: int, name: str, *, expect_directory: bool) -> object:
        inspected_names.append(name)
        return real_inspect(parent_fd, name, expect_directory=expect_directory)

    monkeypatch.setattr(permission_module, "_inspect_at", _record)

    snapshot = collect_permission_snapshot(tmp_path)

    assert snapshot.pmem_directory.state is EntryState.SYMLINK
    assert inspected_names == [".pmem"]


def test_directory_swap_to_symlink_is_incomplete_and_target_is_not_scanned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _healthy_project(tmp_path)
    runs = root / ".pmem" / "artifacts" / "runs"
    parked = runs.with_name("parked-runs")
    outside = _private_dir(tmp_path.parent / "outside-swap-target")
    _private_file(outside / "sensitive.bin")
    real_open_at = permission_module._open_directory_at
    swapped = {"done": False}

    def _swap(parent_fd: int, name: str) -> int:
        if name == "runs" and not swapped["done"]:
            swapped["done"] = True
            runs.rename(parked)
            _symlink_or_skip(runs, outside)
        return real_open_at(parent_fd, name)

    def _must_not_scan(_directory_fd: int, _remaining: int) -> object:
        raise AssertionError("a substituted symlink target must not be scanned")

    monkeypatch.setattr(permission_module, "_open_directory_at", _swap)
    monkeypatch.setattr(permission_module, "_bounded_directory_names", _must_not_scan)

    result = _results(root)[_RUN_ARTIFACTS]

    assert swapped["done"] is True
    assert result.outcome is DoctorCheckOutcome.SKIPPED
    assert result.severity is DoctorSeverity.WARNING


def test_project_root_binding_change_makes_every_result_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _healthy_project(tmp_path)
    monkeypatch.setattr(permission_module, "_same_path_binding", lambda _path, _fd: False)

    results = _results(root)

    assert all(result.outcome is DoctorCheckOutcome.SKIPPED for result in results.values())
    assert all(result.severity is DoctorSeverity.WARNING for result in results.values())


@pytest.mark.parametrize(
    ("relative", "check_id"),
    [
        (".pmem/pmem.db", _DATABASE),
        (".pmem/config.yaml", _CONFIG),
        (".pmem/graph.json", _GRAPH),
    ],
)
def test_symlinked_private_file_fails(tmp_path: Path, relative: str, check_id: str) -> None:
    root = _healthy_project(tmp_path)
    target = root / relative
    target.unlink()
    outside = _private_file(tmp_path.parent / f"outside-{check_id.split('.')[-1]}")
    _symlink_or_skip(target, outside)

    result = _results(root)[check_id]
    snapshot = collect_permission_snapshot(root)

    assert result.outcome is DoctorCheckOutcome.FAIL
    # classified as a link, not merely as "wrong type": the operator must be
    # told to replace the link, not that something unexpected is in the way
    assert "is a link" in result.message
    assert "Replace the link" in (result.remediation or "")
    assert getattr(snapshot, check_id.split(".")[-1]).state is EntryState.SYMLINK
    assert target.is_symlink()  # never replaced


def test_symlinked_nested_run_artifact_fails(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    stdout = root / ".pmem" / "artifacts" / "runs" / "run_0123456789abcdef" / "stdout.txt"
    stdout.unlink()
    _symlink_or_skip(stdout, tmp_path.parent / "outside-artifact")

    result = _results(root)[_RUN_ARTIFACTS]
    assert result.outcome is DoctorCheckOutcome.FAIL
    assert "is a link" in result.message
    assert "Replace the link" in (result.remediation or "")


def test_symlinked_run_root_fails_with_link_remediation(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path, with_run=False)
    runs = root / ".pmem" / "artifacts" / "runs"
    _symlink_or_skip(runs, _private_dir(tmp_path.parent / "outside-runs"))

    result = _results(root)[_RUN_ARTIFACTS]

    assert result.outcome is DoctorCheckOutcome.FAIL
    assert "is a link" in result.message
    assert "Replace the link" in (result.remediation or "")


def test_broken_symlink_fails_rather_than_reading_as_missing(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    database = root / ".pmem" / "pmem.db"
    database.unlink()
    _symlink_or_skip(database, tmp_path / "does-not-exist")

    result = _results(root)[_DATABASE]

    assert result.outcome is DoctorCheckOutcome.FAIL
    assert result.outcome is not DoctorCheckOutcome.SKIPPED
    # a broken link is still a link, never "missing" and never "wrong type"
    assert "is a link" in result.message
    assert collect_permission_snapshot(root).database.state is EntryState.SYMLINK


# --------------------------------------------------------------------------- #
# E. Errors and races                                                          #
# --------------------------------------------------------------------------- #
def test_unreadable_directory_listing_fails_closed(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    runs = root / ".pmem" / "artifacts" / "runs"
    runs.chmod(0o300)  # owner may enter but not list
    if os.access(runs, os.R_OK):
        runs.chmod(0o700)
        pytest.skip("permission bits are not enforced for the current user")
    try:
        result = _results(root)[_RUN_ARTIFACTS]
    finally:
        runs.chmod(0o700)

    assert result.outcome is DoctorCheckOutcome.FAIL
    assert result.outcome is not DoctorCheckOutcome.PASS


def test_lstat_permission_denied_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _healthy_project(tmp_path)
    real_inspect = permission_module._inspect_at

    def _deny(parent_fd: int, name: str, *, expect_directory: bool) -> object:
        if name == "pmem.db":
            return permission_module.EntryPermission(EntryState.UNREADABLE, None, False)
        return real_inspect(parent_fd, name, expect_directory=expect_directory)

    monkeypatch.setattr(permission_module, "_inspect_at", _deny)

    result = _results(root)[_DATABASE]

    assert result.outcome is DoctorCheckOutcome.FAIL
    assert "PermissionError" not in result.message


def test_file_vanishing_during_scan_is_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _healthy_project(tmp_path)
    real_inspect = permission_module._inspect_at
    removed = {"done": False}

    def _vanish(parent_fd: int, name: str, *, expect_directory: bool) -> object:
        if name == "stderr.txt" and not removed["done"]:
            removed["done"] = True
            return permission_module.EntryPermission(EntryState.MISSING, None, False)
        return real_inspect(parent_fd, name, expect_directory=expect_directory)

    monkeypatch.setattr(permission_module, "_inspect_at", _vanish)

    result = _results(root)[_RUN_ARTIFACTS]

    assert result.outcome is DoctorCheckOutcome.SKIPPED
    assert result.severity is DoctorSeverity.WARNING


@pytest.mark.parametrize("relative", [".pmem/pmem.db", ".pmem/config.yaml"])
def test_fifo_in_place_of_a_private_file_fails(tmp_path: Path, relative: str) -> None:
    root = _healthy_project(tmp_path)
    target = root / relative
    target.unlink()
    try:
        os.mkfifo(target)
    except (AttributeError, OSError, NotImplementedError):
        pytest.skip("named pipes are not supported on this platform")

    check_id = _DATABASE if relative.endswith("pmem.db") else _CONFIG
    result = _results(root)[check_id]

    assert result.outcome is DoctorCheckOutcome.FAIL


def test_file_in_place_of_a_private_directory_fails(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    snapshots = root / ".pmem" / "snapshots"
    snapshots.rmdir()
    _private_file(snapshots)

    assert _results(root)[_PMEM_DIR].outcome is DoctorCheckOutcome.FAIL


def test_no_expected_filesystem_condition_raises(tmp_path: Path) -> None:
    """Every modelled condition must become a typed result, never an exception."""

    root = _healthy_project(tmp_path)
    database = root / ".pmem" / "pmem.db"
    for mode in (0o000, 0o644, 0o777, 0o400):
        database.chmod(mode)
        assert len(_results(root)) == 5
    database.chmod(0o600)


def test_a_programmer_error_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bug must propagate, not be flattened into a misleading failure."""

    def _bug(*args: object, **kwargs: object) -> object:
        raise TypeError("programmer error")

    monkeypatch.setattr(permission_module, "_inspect_at", _bug)

    with pytest.raises(TypeError, match="programmer error"):
        run_permission_checks(DoctorCheckContext(project_root=tmp_path))


# --------------------------------------------------------------------------- #
# Traversal budget boundary                                                    #
# --------------------------------------------------------------------------- #
def test_traversal_budget_is_a_declared_bounded_constant() -> None:
    """The budget must be an explicit, finite constant, not an accident."""

    assert isinstance(MAX_SCANNED_ENTRIES, int)
    assert 0 < MAX_SCANNED_ENTRIES <= 100_000


@pytest.mark.parametrize("delta", [-1, 0, 1], ids=["limit_minus_1", "limit", "limit_plus_1"])
def test_traversal_budget_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delta: int
) -> None:
    monkeypatch.setattr(permission_module, "MAX_SCANNED_ENTRIES", 10)
    root = _healthy_project(tmp_path, with_run=False)
    artifacts = root / ".pmem" / "artifacts" / "runs"
    _private_dir(artifacts)
    for index in range(10 + delta):
        _private_file(artifacts / f"entry_{index:04d}.txt")

    result = _results(root)[_RUN_ARTIFACTS]

    if delta <= 0:
        assert result.outcome is DoctorCheckOutcome.PASS
    else:
        assert result.outcome is DoctorCheckOutcome.SKIPPED
        assert result.severity is DoctorSeverity.WARNING
        assert result.remediation is not None
        assert result.outcome is not DoctorCheckOutcome.PASS


def test_budget_stops_consuming_directory_entries_at_limit_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(permission_module, "MAX_SCANNED_ENTRIES", 10)
    consumed = {"count": 0}

    class _Entry:
        def __init__(self, index: int) -> None:
            self.name = f"entry_{index:04d}"

    class _Scandir:
        def __enter__(self) -> object:
            def _entries() -> object:
                for index in range(10_000):
                    consumed["count"] += 1
                    yield _Entry(index)

            return _entries()

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(os, "scandir", lambda _fd: _Scandir())

    assert permission_module._bounded_directory_names(123, 10) is None
    assert consumed["count"] == 11


# --------------------------------------------------------------------------- #
# G. Determinism                                                               #
# --------------------------------------------------------------------------- #
def test_same_state_yields_identical_results(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    context = DoctorCheckContext(project_root=root)

    assert run_permission_checks(context) == run_permission_checks(context)


def test_results_are_in_canonical_order_regardless_of_execution_order(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    context = DoctorCheckContext(project_root=root)

    entry_point = {r.check_id: r for r in run_permission_checks(context)}
    reversed_definitions = {
        d.check_id: d.execute(context) for d in reversed(permission_check_definitions())
    }

    assert [r.check_id for r in run_permission_checks(context)] == list(PERMISSION_CHECK_IDS)
    assert entry_point == reversed_definitions


def test_a_mode_change_between_invocations_is_observed(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    context = DoctorCheckContext(project_root=root)

    assert _results(root)[_DATABASE].outcome is DoctorCheckOutcome.PASS
    (root / ".pmem" / "pmem.db").chmod(0o644)
    assert _results(root)[_DATABASE].outcome is DoctorCheckOutcome.FAIL
    (root / ".pmem" / "pmem.db").chmod(0o600)
    assert run_permission_checks(context)[1].outcome is DoctorCheckOutcome.PASS


def test_results_carry_no_volatile_field(tmp_path: Path) -> None:
    for result in run_permission_checks(
        DoctorCheckContext(project_root=_healthy_project(tmp_path))
    ):
        dumped = result.model_dump(mode="json")
        assert set(dumped) == {
            "check_id",
            "category",
            "outcome",
            "severity",
            "message",
            "remediation",
            "related_entity_id",
        }
        assert dumped["related_entity_id"] is None


# --------------------------------------------------------------------------- #
# Snapshot internals                                                           #
# --------------------------------------------------------------------------- #
def test_snapshot_is_immutable(tmp_path: Path) -> None:
    snapshot = collect_permission_snapshot(_healthy_project(tmp_path))

    assert snapshot.pmem_directory.state is EntryState.OK
    assert snapshot.run_artifact_scan is ScanState.OK
    with pytest.raises(AttributeError):
        snapshot.posix_modes_supported = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mode", "is_directory", "expected"),
    [
        (0o600, False, ModeVerdict.OK),
        (0o700, True, ModeVerdict.OK),
        (0o644, False, ModeVerdict.EXPOSED),
        (0o755, True, ModeVerdict.EXPOSED),
        (0o400, False, ModeVerdict.OWNER_UNUSABLE),
        (0o600, True, ModeVerdict.OWNER_UNUSABLE),
        (0o000, False, ModeVerdict.OWNER_UNUSABLE),
    ],
)
def test_mode_verdict_table(mode: int, is_directory: bool, expected: ModeVerdict) -> None:
    entry = permission_module.EntryPermission(EntryState.OK, mode, is_directory)

    assert entry.verdict() is expected


def test_owner_identity_mismatch_never_reports_current_user_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _healthy_project(tmp_path)
    real_euid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: real_euid + 1)

    results = _results(root)

    assert results[_PMEM_DIR].outcome is DoctorCheckOutcome.FAIL
    assert results[_DATABASE].outcome is DoctorCheckOutcome.FAIL
    assert "current user" in (results[_DATABASE].remediation or "")


def test_measured_writer_baseline_is_reported_honestly(tmp_path: Path) -> None:
    """A freshly initialized project is reported as it really is, not as wished.

    ``init_project`` creates its directories with a bare ``mkdir``, so under the
    default umask they land at 0755. DOC-003 is a diagnostic: it must surface
    that, and this test pins the behaviour so a future writer-hardening change
    is a deliberate, visible decision.
    """

    from pmem.services.project_init import init_project

    init_project(tmp_path, project_name="baseline-probe", primary_metric="accuracy")
    observed = stat.S_IMODE((tmp_path / ".pmem").lstat().st_mode)

    if observed & 0o077:
        assert _results(tmp_path)[_PMEM_DIR].outcome is DoctorCheckOutcome.FAIL
    else:  # a stricter umask was in effect
        assert _results(tmp_path)[_PMEM_DIR].outcome is DoctorCheckOutcome.PASS


# --------------------------------------------------------------------------- #
# Defensive branches inside the bounded traversal                              #
# --------------------------------------------------------------------------- #
def test_symlinked_artifacts_directory_blocks_the_scan(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path, with_run=False)
    artifacts = root / ".pmem" / "artifacts"
    artifacts.rmdir()
    _symlink_or_skip(artifacts, _private_dir(tmp_path.parent / "outside-artifacts"))

    results = _results(root)

    assert collect_permission_snapshot(root).run_artifact_scan is ScanState.ROOT_NOT_INSPECTABLE
    assert results[_RUN_ARTIFACTS].outcome is DoctorCheckOutcome.SKIPPED
    assert results[_RUN_ARTIFACTS].outcome is not DoctorCheckOutcome.PASS
    assert results[_PMEM_DIR].outcome is DoctorCheckOutcome.FAIL


def test_unlistable_subdirectory_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _healthy_project(tmp_path)
    real_reader = permission_module._bounded_directory_names
    calls = {"count": 0}

    def _deny(directory_fd: int, remaining: int) -> object:
        calls["count"] += 1
        if calls["count"] == 2:
            raise PermissionError("cannot list")
        return real_reader(directory_fd, remaining)

    monkeypatch.setattr(permission_module, "_bounded_directory_names", _deny)

    result = _results(root)[_RUN_ARTIFACTS]
    assert result.outcome is DoctorCheckOutcome.FAIL
    assert "could not be listed" in result.message


def test_subdirectory_vanishing_before_listing_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _healthy_project(tmp_path)
    real_reader = permission_module._bounded_directory_names
    vanished = {"done": False}

    def _vanish(directory_fd: int, remaining: int) -> object:
        if not vanished["done"]:
            vanished["done"] = True
            return real_reader(directory_fd, remaining)
        raise FileNotFoundError("vanished")

    monkeypatch.setattr(permission_module, "_bounded_directory_names", _vanish)

    result = _results(root)[_RUN_ARTIFACTS]
    assert result.outcome is DoctorCheckOutcome.SKIPPED
    assert result.severity is DoctorSeverity.WARNING


def test_entry_stat_error_is_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _healthy_project(tmp_path)
    real_stat = os.stat

    def _fail(name: object, *args: object, **kwargs: object) -> os.stat_result:
        if name == "stderr.txt":
            raise OSError("cannot classify")
        return real_stat(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "stat", _fail)
    monkeypatch.setattr(permission_module, "posix_modes_are_supported", lambda: True)

    assert collect_permission_snapshot(root).run_artifact_scan is ScanState.INCOMPLETE
    result = _results(root)[_RUN_ARTIFACTS]
    assert result.outcome is DoctorCheckOutcome.SKIPPED
    assert result.severity is DoctorSeverity.WARNING


def test_unexpected_file_type_inside_the_run_tree_fails(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    fifo = root / ".pmem" / "artifacts" / "runs" / "run_0123456789abcdef" / "pipe"
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError, NotImplementedError):
        pytest.skip("named pipes are not supported on this platform")

    assert collect_permission_snapshot(root).run_artifact_scan is ScanState.UNSAFE_ENTRY
    assert _results(root)[_RUN_ARTIFACTS].outcome is DoctorCheckOutcome.FAIL


def test_missing_required_state_directory_is_incomplete(tmp_path: Path) -> None:
    """A missing init-created directory cannot produce a permission PASS."""

    root = _healthy_project(tmp_path, with_run=False)
    (root / ".pmem" / "snapshots").rmdir()

    result = _results(root)[_PMEM_DIR]
    assert result.outcome is DoctorCheckOutcome.SKIPPED
    assert result.severity is DoctorSeverity.WARNING


def test_entry_without_a_mode_is_treated_as_unusable() -> None:
    entry = permission_module.EntryPermission(EntryState.OK, None, False)

    assert entry.verdict() is ModeVerdict.OWNER_UNUSABLE
