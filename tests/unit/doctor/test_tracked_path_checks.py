"""Tracked-path doctor check tests (DOC-004).

Every case builds a real project with a real SQLite database and real files on
disk. Nothing about the filesystem or the database is faked, so a production
path that stopped hashing, stopped walking components, or started following a
link turns these tests red rather than leaving them green.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat as stat_module
import uuid
from pathlib import Path

import pytest

import pmem.doctor.tracked_path_checks as tracked_module
from pmem.doctor import (
    DoctorCategory,
    DoctorCheckContext,
    DoctorCheckOutcome,
    DoctorCheckResult,
    DoctorSeverity,
)
from pmem.doctor.tracked_path_checks import (
    MAX_INSPECTED_RECORDS,
    MAX_TOTAL_HASHED_BYTES,
    TRACKED_PATH_CHECK_IDS,
    SourceState,
    TrackedPathSnapshot,
    collect_tracked_path_snapshot,
    record_is_safe,
    run_tracked_path_checks,
    stored_hash_is_safe,
    stored_path_is_safe,
    stored_size_is_safe,
    tracked_path_check_definitions,
)
from pmem.repositories.sqlite import project_database_path
from pmem.services.config import project_config_path, read_project_config
from pmem.services.project_init import init_project
from pmem.services.tracking import MAX_TRACKED_PATH_LENGTH, track_path
from pmem.utils.hashing import compute_file_hash

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="descriptor-anchored traversal requires POSIX"
)

_CONTENT = "tracked_paths.content_current"
_PRESENT = "tracked_paths.present"
_RECORDS = "tracked_paths.records_safe"
_SYMLINK = "tracked_paths.symlink"


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _project(root: Path) -> Path:
    init_project(root, project_name="tracked-doctor", primary_metric="accuracy")
    return root


def _write(root: Path, relative: str, content: bytes = b"print('hi')\n") -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _tracked(root: Path, relative: str, content: bytes = b"print('hi')\n") -> Path:
    target = _write(root, relative, content)
    track_path(root, relative)
    return target


def _project_id(root: Path) -> str:
    return read_project_config(project_config_path(root)).project_id


def _insert_raw_record(
    root: Path,
    *,
    path: str,
    sha256: str | None = None,
    size_bytes: int | None = 1,
    tag: str | None = None,
) -> None:
    """Write a record straight into SQLite, bypassing service validation.

    This is how a restored backup, a hand-edited database or a corrupted row
    would look, and it is exactly what the diagnostic must not trust.
    """

    connection = sqlite3.connect(project_database_path(root))
    connection.execute(
        "INSERT INTO tracked_paths"
        "(id, project_id, path, tag, hash, size_bytes, last_checked, created_at) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
        (
            f"track_{uuid.uuid4().hex}",
            _project_id(root),
            path,
            tag,
            ("a" * 64) if sha256 is None else sha256,
            size_bytes,
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        ),
    )
    connection.commit()
    connection.close()


def _results(root: Path) -> dict[str, DoctorCheckResult]:
    return {
        result.check_id: result
        for result in run_tracked_path_checks(DoctorCheckContext(project_root=root))
    }


def _pairs(root: Path) -> dict[str, tuple[str, str]]:
    return {
        check_id: (result.outcome.value, result.severity.value)
        for check_id, result in _results(root).items()
    }


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")


# --------------------------------------------------------------------------- #
# A. Registry / contract                                                       #
# --------------------------------------------------------------------------- #
def test_factory_returns_the_four_stable_ids_in_canonical_order() -> None:
    definitions = tracked_path_check_definitions()

    assert tuple(d.check_id for d in definitions) == TRACKED_PATH_CHECK_IDS
    assert TRACKED_PATH_CHECK_IDS == tuple(sorted(TRACKED_PATH_CHECK_IDS))
    assert set(TRACKED_PATH_CHECK_IDS) == {_CONTENT, _PRESENT, _RECORDS, _SYMLINK}


def test_every_definition_is_a_tracked_paths_definition() -> None:
    for definition in tracked_path_check_definitions():
        assert definition.category is DoctorCategory.TRACKED_PATHS
        assert definition.check_id.startswith("tracked_paths.")


def test_definition_and_result_identity_match(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py")
    context = DoctorCheckContext(project_root=root)

    for definition in tracked_path_check_definitions():
        result = definition.execute(context)
        assert result.check_id == definition.check_id
        assert result.category is definition.category


def test_two_factories_share_nothing() -> None:
    first = tracked_path_check_definitions()
    second = tracked_path_check_definitions()

    assert first is not second
    assert all(a is not b for a, b in zip(first, second, strict=True))


def test_module_exposes_no_singleton_or_cache() -> None:
    for name in ("_REGISTRY", "REGISTRY", "_CACHE", "CACHE", "_SNAPSHOT", "_SHARED"):
        assert not hasattr(tracked_module, name), name


def test_factory_performs_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("building definitions must not touch disk")

    monkeypatch.setattr(os, "open", _forbidden)
    monkeypatch.setattr(sqlite3, "connect", _forbidden)

    assert len(tracked_path_check_definitions()) == 4


def test_snapshot_is_frozen(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py")
    snapshot = collect_tracked_path_snapshot(root)

    assert isinstance(snapshot, TrackedPathSnapshot)
    assert snapshot.source is SourceState.OK
    with pytest.raises(AttributeError):
        snapshot.record_count = 99  # type: ignore[misc]


def test_definitions_do_not_hold_a_stale_snapshot(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = _tracked(root, "src/train.py")
    context = DoctorCheckContext(project_root=root)
    definition = next(d for d in tracked_path_check_definitions() if d.check_id == _CONTENT)

    assert definition.execute(context).outcome is DoctorCheckOutcome.PASS
    target.write_bytes(b"changed\n")
    assert definition.execute(context).outcome is DoctorCheckOutcome.FAIL


# --------------------------------------------------------------------------- #
# B. Stored records                                                            #
# --------------------------------------------------------------------------- #
def test_zero_records_is_not_applicable(tmp_path: Path) -> None:
    pairs = _pairs(_project(tmp_path))

    for check_id in TRACKED_PATH_CHECK_IDS:
        assert pairs[check_id] == ("not_applicable", "info"), check_id


def test_single_and_multiple_current_records_pass(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _tracked(root, "a.py", b"a\n")
    assert all(r.outcome is DoctorCheckOutcome.PASS for r in _results(root).values())

    _tracked(root, "nested/deep/b.py", b"b\n")
    _tracked(root, "c.py", b"c\n")
    results = _results(root)

    for check_id, result in results.items():
        assert result.outcome is DoctorCheckOutcome.PASS, check_id
        assert result.severity is DoctorSeverity.INFO, check_id
        assert result.remediation is None, check_id


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_stored_path_is_rejected_by_predicate_and_by_schema(
    tmp_path: Path, blank: str
) -> None:
    """The schema CHECK already blocks a blank path; the predicate agrees.

    Verified at runtime: ``length(trim(path)) > 0`` makes the row uninsertable,
    so this case cannot be exercised end to end and is asserted at the
    predicate boundary instead.
    """

    assert stored_path_is_safe(blank) is False

    root = _project(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_raw_record(root, path=blank)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        # SQLite ``trim()`` strips spaces only, so a tab-only path really is
        # insertable and must be caught by the diagnostic, not by the schema.
        "\t",
        "/etc/passwd",
        "//server/share/file.txt",
        "C:/Windows/System32/config",
        "c:\\Windows\\file.txt",
        "..",
        "../outside.txt",
        "src/../../outside.txt",
        "src/../train.py",
        ".pmem/pmem.db",
        ".PMEM/pmem.db",
        ".PmEm/config.yaml",
        ".pMeM/graph.json",
        "src\\train.py",
        "src//train.py",
        "./train.py",
        "src/./train.py",
        "train.py/",
        "tr\x01ain.py",
        "train\x7f.py",
        "train\n.py",
        "a" * (MAX_TRACKED_PATH_LENGTH + 1),
    ],
)
def test_unsafe_stored_path_is_rejected(tmp_path: Path, unsafe_path: str) -> None:
    assert stored_path_is_safe(unsafe_path) is False

    root = _project(tmp_path)
    _insert_raw_record(root, path=unsafe_path)
    results = _results(root)

    assert results[_RECORDS].outcome is DoctorCheckOutcome.FAIL
    assert results[_RECORDS].severity is DoctorSeverity.ERROR
    assert results[_RECORDS].remediation is not None
    for check_id in (_SYMLINK, _PRESENT, _CONTENT):
        assert results[check_id].outcome is DoctorCheckOutcome.SKIPPED, check_id
        assert results[check_id].outcome is not DoctorCheckOutcome.PASS, check_id


@pytest.mark.parametrize(
    "safe_path",
    ["train.py", "src/train.py", "a/b/c/d.py", "dữ-liệu/tệp.py", ".hidden/file.py", "x.pmem"],
)
def test_safe_stored_path_is_accepted(safe_path: str) -> None:
    assert stored_path_is_safe(safe_path) is True


@pytest.mark.parametrize(
    "bad_hash",
    ["", "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65, " " + "a" * 63],
)
def test_invalid_stored_hash_is_rejected(tmp_path: Path, bad_hash: str) -> None:
    """The schema CHECK blocks every malformed digest; the predicate agrees.

    Verified at runtime: ``length(hash) = 64 AND hash NOT GLOB '*[^0-9a-f]*'``
    makes such a row uninsertable, so the diagnostic can only be exercised at
    the predicate boundary. The predicate still exists because a restored or
    externally written database file need not have been created by this schema.
    """

    assert stored_hash_is_safe(bad_hash) is False

    root = _project(tmp_path)
    _write(root, "train.py")
    with pytest.raises(sqlite3.IntegrityError):
        _insert_raw_record(root, path="train.py", sha256=bad_hash)


def test_negative_stored_size_is_rejected() -> None:
    assert stored_size_is_safe(-1) is False
    assert stored_size_is_safe(0) is True
    assert stored_size_is_safe(None) is True


def test_null_size_record_is_still_safe(tmp_path: Path) -> None:
    """``size_bytes`` is nullable in the schema, so ``NULL`` must not be unsafe."""

    root = _project(tmp_path)
    target = _write(root, "train.py")
    _insert_raw_record(root, path="train.py", sha256=compute_file_hash(target), size_bytes=None)

    results = _results(root)

    assert results[_RECORDS].outcome is DoctorCheckOutcome.PASS
    assert results[_CONTENT].outcome is DoctorCheckOutcome.PASS


def test_unicode_filename_is_tracked_and_never_leaked(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _tracked(root, "dữ-liệu/tệp-huấn-luyện.py", b"x\n")

    results = _results(root)
    rendered = json.dumps([r.model_dump(mode="json") for r in results.values()])

    assert results[_CONTENT].outcome is DoctorCheckOutcome.PASS
    assert "dữ-liệu" not in rendered
    assert "tệp-huấn-luyện" not in rendered


def test_database_row_order_does_not_change_results(tmp_path: Path) -> None:
    root = _project(tmp_path)
    for name in ("z.py", "a.py", "m.py"):
        _tracked(root, name, name.encode())
    forward = json.dumps([r.model_dump(mode="json") for r in _results(root).values()])

    # rewrite the rows in the opposite order, preserving content
    connection = sqlite3.connect(project_database_path(root))
    rows = connection.execute("SELECT * FROM tracked_paths ORDER BY path DESC").fetchall()
    connection.execute("DELETE FROM tracked_paths")
    for row in rows:
        connection.execute("INSERT INTO tracked_paths VALUES(?,?,?,?,?,?,?,?)", tuple(row))
    connection.commit()
    connection.close()

    assert json.dumps([r.model_dump(mode="json") for r in _results(root).values()]) == forward


# --------------------------------------------------------------------------- #
# C. Filesystem state                                                          #
# --------------------------------------------------------------------------- #
def test_unchanged_file_is_not_a_false_positive(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py", b"stable content\n")

    for _ in range(3):
        assert _results(root)[_CONTENT].outcome is DoctorCheckOutcome.PASS


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        (b"aaaa\n", b"bbbb\n"),  # same size, different content
        (b"aaaa\n", b"much longer content here\n"),  # different size
        (b"aaaa\n", b""),  # truncated
    ],
    ids=["same_size", "larger", "truncated"],
)
def test_changed_content_fails_as_a_warning(
    tmp_path: Path, original: bytes, replacement: bytes
) -> None:
    root = _project(tmp_path)
    target = _tracked(root, "src/train.py", original)
    target.write_bytes(replacement)

    results = _results(root)

    assert results[_CONTENT].outcome is DoctorCheckOutcome.FAIL
    assert results[_CONTENT].severity is DoctorSeverity.WARNING  # stale, not corrupt
    assert results[_PRESENT].outcome is DoctorCheckOutcome.PASS
    assert results[_SYMLINK].outcome is DoctorCheckOutcome.PASS
    assert results[_RECORDS].outcome is DoctorCheckOutcome.PASS


def test_missing_file_fails_and_blocks_content(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = _tracked(root, "src/train.py")
    target.unlink()

    results = _results(root)

    assert results[_PRESENT].outcome is DoctorCheckOutcome.FAIL
    assert results[_PRESENT].severity is DoctorSeverity.ERROR
    assert results[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED
    assert results[_SYMLINK].outcome is DoctorCheckOutcome.PASS


def test_missing_parent_directory_fails_present(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py")
    (root / "src" / "train.py").unlink()
    (root / "src").rmdir()

    assert _results(root)[_PRESENT].outcome is DoctorCheckOutcome.FAIL


def test_file_replaced_by_a_directory_fails_present(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = _tracked(root, "src/train.py")
    target.unlink()
    target.mkdir()

    results = _results(root)

    assert results[_PRESENT].outcome is DoctorCheckOutcome.FAIL
    assert results[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED


def test_file_replaced_by_a_fifo_fails_present(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = _tracked(root, "src/train.py")
    target.unlink()
    try:
        os.mkfifo(target)
    except (AttributeError, OSError, NotImplementedError):
        pytest.skip("named pipes are not supported on this platform")

    results = _results(root)

    assert results[_PRESENT].outcome is DoctorCheckOutcome.FAIL
    assert results[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED


def test_leaf_replaced_by_symlink_fails_symlink_check(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = _tracked(root, "src/train.py")
    outside = tmp_path.parent / "outside-leaf.py"
    outside.write_bytes(b"outside\n")
    target.unlink()
    _symlink_or_skip(target, outside)

    results = _results(root)

    assert results[_SYMLINK].outcome is DoctorCheckOutcome.FAIL
    assert results[_SYMLINK].severity is DoctorSeverity.ERROR
    assert results[_PRESENT].outcome is DoctorCheckOutcome.SKIPPED
    assert results[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED
    assert target.is_symlink()


def test_parent_component_replaced_by_symlink_fails_symlink_check(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py")
    outside_dir = tmp_path.parent / "outside-src"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "train.py").write_bytes(b"outside\n")
    (root / "src" / "train.py").unlink()
    (root / "src").rmdir()
    _symlink_or_skip(root / "src", outside_dir)

    results = _results(root)

    assert results[_SYMLINK].outcome is DoctorCheckOutcome.FAIL
    assert results[_PRESENT].outcome is DoctorCheckOutcome.SKIPPED
    assert results[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED


def test_broken_symlink_fails_symlink_not_missing(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = _tracked(root, "src/train.py")
    target.unlink()
    _symlink_or_skip(target, tmp_path / "does-not-exist")

    results = _results(root)

    assert results[_SYMLINK].outcome is DoctorCheckOutcome.FAIL
    assert results[_PRESENT].outcome is DoctorCheckOutcome.SKIPPED


def test_unreadable_file_is_incomplete_never_pass(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = _tracked(root, "src/train.py")
    target.chmod(0o000)
    if os.access(target, os.R_OK):
        target.chmod(0o600)
        pytest.skip("permission bits are not enforced for the current user")
    try:
        results = _results(root)
    finally:
        target.chmod(0o600)

    assert results[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED
    assert results[_CONTENT].severity is DoctorSeverity.WARNING
    assert results[_CONTENT].outcome is not DoctorCheckOutcome.PASS


def test_file_changed_while_hashing_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file rewritten mid-hash must never be reported current."""

    root = _project(tmp_path)
    target = _tracked(root, "src/train.py", b"x" * 4096)
    real_read = os.read
    fired = {"done": False}

    def _mutate_during_read(descriptor: int, size: int) -> bytes:
        chunk = real_read(descriptor, size)
        if chunk and not fired["done"]:
            fired["done"] = True
            target.write_bytes(b"y" * 8192)
        return chunk

    monkeypatch.setattr(os, "read", _mutate_during_read)

    result = _results(root)[_CONTENT]

    assert fired["done"]
    assert result.outcome is DoctorCheckOutcome.SKIPPED
    assert result.severity is DoctorSeverity.WARNING
    assert result.outcome is not DoctorCheckOutcome.PASS


def test_project_root_rebinding_makes_the_invocation_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py")

    monkeypatch.setattr(tracked_module, "same_path_binding", lambda path, fd: False)

    results = _results(root)

    for check_id, result in results.items():
        assert result.outcome is DoctorCheckOutcome.SKIPPED, check_id
        assert result.severity is DoctorSeverity.WARNING, check_id


def test_project_root_rebinding_after_record_load_blocks_file_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root replaced while SQLite is read must not be used for the file walk."""

    root = _project(tmp_path)
    _tracked(root, "src/train.py")
    calls = 0
    real_binding = tracked_module.same_path_binding

    def _changes_after_load(path: Path, descriptor: int) -> bool:
        nonlocal calls
        calls += 1
        return real_binding(path, descriptor) if calls == 1 else False

    def _forbidden_inspection(*args: object, **kwargs: object) -> object:
        raise AssertionError("a rebound root must not be inspected")

    monkeypatch.setattr(tracked_module, "same_path_binding", _changes_after_load)
    monkeypatch.setattr(tracked_module, "_inspect_record", _forbidden_inspection)

    results = _results(root)

    assert calls == 2
    for result in results.values():
        assert result.outcome is DoctorCheckOutcome.SKIPPED
        assert result.severity is DoctorSeverity.WARNING


def test_symlinked_project_root_is_rejected_before_records_are_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The root guard must run before config or SQLite can be read via a link."""

    target = _project(tmp_path / "outside")
    root = tmp_path / "project-link"
    _symlink_or_skip(root, target)
    called = False

    def _forbidden_load(project_root: Path) -> object:
        nonlocal called
        called = True
        raise AssertionError("records must not be loaded through a symlinked root")

    monkeypatch.setattr(tracked_module, "_load_records", _forbidden_load)

    results = _results(root)

    assert called is False
    assert all(result.outcome is DoctorCheckOutcome.SKIPPED for result in results.values())


def test_directory_raced_into_a_symlink_between_stat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A component whose identity changes after inspection must not be trusted."""

    root = _project(tmp_path)
    _tracked(root, "src/train.py")

    monkeypatch.setattr(tracked_module, "same_directory_binding", lambda fd, name, child: False)

    results = _results(root)

    assert results[_PRESENT].outcome is not DoctorCheckOutcome.PASS
    assert results[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED


# --------------------------------------------------------------------------- #
# E. Resource budgets                                                          #
# --------------------------------------------------------------------------- #
def test_declared_budgets_are_bounded_constants() -> None:
    assert isinstance(MAX_INSPECTED_RECORDS, int) and 0 < MAX_INSPECTED_RECORDS <= 100_000
    assert isinstance(MAX_TOTAL_HASHED_BYTES, int) and 0 < MAX_TOTAL_HASHED_BYTES


@pytest.mark.parametrize("delta", [-1, 0, 1], ids=["limit_minus_1", "limit", "limit_plus_1"])
def test_record_budget_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delta: int
) -> None:
    monkeypatch.setattr(tracked_module, "MAX_INSPECTED_RECORDS", 5)
    root = _project(tmp_path)
    for index in range(5 + delta):
        _tracked(root, f"f{index:03d}.py", f"content-{index}\n".encode())

    results = _results(root)

    if delta <= 0:
        assert results[_PRESENT].outcome is DoctorCheckOutcome.PASS
        assert results[_CONTENT].outcome is DoctorCheckOutcome.PASS
    else:
        assert results[_PRESENT].outcome is DoctorCheckOutcome.SKIPPED
        assert results[_PRESENT].severity is DoctorSeverity.WARNING
        assert results[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED
        assert results[_RECORDS].outcome is DoctorCheckOutcome.SKIPPED
        assert results[_RECORDS].severity is DoctorSeverity.WARNING
        assert results[_SYMLINK].outcome is DoctorCheckOutcome.SKIPPED
        assert results[_SYMLINK].severity is DoctorSeverity.WARNING
        assert results[_PRESENT].outcome is not DoctorCheckOutcome.PASS


def test_record_loader_uses_the_bounded_repository_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap must constrain SQL materialisation, not only the later walk."""

    from pmem.repositories.tracked_paths import TrackedPathRepository

    root = _project(tmp_path)
    _tracked(root, "a.py")
    calls: list[int] = []
    real_limited = TrackedPathRepository.list_for_project_limited

    def _forbidden_unbounded(self: object, project_id: str) -> object:
        raise AssertionError("doctor must not materialise the unbounded record list")

    def _counting_limited(self: TrackedPathRepository, project_id: str, *, limit: int) -> object:
        calls.append(limit)
        return real_limited(self, project_id, limit=limit)

    monkeypatch.setattr(TrackedPathRepository, "list_for_project", _forbidden_unbounded)
    monkeypatch.setattr(TrackedPathRepository, "list_for_project_limited", _counting_limited)

    assert _results(root)[_CONTENT].outcome is DoctorCheckOutcome.PASS
    assert calls == [MAX_INSPECTED_RECORDS + 1]


def test_byte_budget_stops_the_reader_without_reading_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file larger than the budget must not be read to its end."""

    monkeypatch.setattr(tracked_module, "HASH_CHUNK_SIZE", 1024)
    monkeypatch.setattr(tracked_module, "MAX_TOTAL_HASHED_BYTES", 2048)
    root = _project(tmp_path)
    _tracked(root, "big.bin", b"z" * (1024 * 64))

    consumed = {"bytes": 0}
    real_read = os.read

    def _counting_read(descriptor: int, size: int) -> bytes:
        chunk = real_read(descriptor, size)
        consumed["bytes"] += len(chunk)
        return chunk

    monkeypatch.setattr(os, "read", _counting_read)

    result = _results(root)[_CONTENT]

    assert result.outcome is DoctorCheckOutcome.SKIPPED
    assert result.outcome is not DoctorCheckOutcome.PASS
    # stopped at the budget plus at most one chunk, not the whole 64 KiB file
    assert consumed["bytes"] <= 2048 + 1024


def test_several_files_together_exhaust_the_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tracked_module, "HASH_CHUNK_SIZE", 512)
    monkeypatch.setattr(tracked_module, "MAX_TOTAL_HASHED_BYTES", 1024)
    root = _project(tmp_path)
    for index in range(4):
        _tracked(root, f"f{index}.bin", b"q" * 700)

    result = _results(root)[_CONTENT]

    assert result.outcome is DoctorCheckOutcome.SKIPPED
    assert result.remediation is not None


def test_a_file_declaring_a_small_size_but_larger_on_disk_still_hits_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget follows bytes actually read, not the stored ``size_bytes``."""

    monkeypatch.setattr(tracked_module, "HASH_CHUNK_SIZE", 256)
    monkeypatch.setattr(tracked_module, "MAX_TOTAL_HASHED_BYTES", 512)
    root = _project(tmp_path)
    target = _write(root, "lying.bin", b"w" * 8192)
    _insert_raw_record(root, path="lying.bin", sha256=compute_file_hash(target), size_bytes=1)

    assert _results(root)[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED


# --------------------------------------------------------------------------- #
# F. Determinism                                                               #
# --------------------------------------------------------------------------- #
def test_same_state_yields_identical_results(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _tracked(root, "a.py", b"a\n")
    _tracked(root, "b/c.py", b"c\n")
    context = DoctorCheckContext(project_root=root)

    assert run_tracked_path_checks(context) == run_tracked_path_checks(context)


def test_results_are_in_canonical_order_regardless_of_execution_order(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _tracked(root, "a.py")
    context = DoctorCheckContext(project_root=root)

    entry_point = {r.check_id: r for r in run_tracked_path_checks(context)}
    reversed_defs = {
        d.check_id: d.execute(context) for d in reversed(tracked_path_check_definitions())
    }

    assert [r.check_id for r in run_tracked_path_checks(context)] == list(TRACKED_PATH_CHECK_IDS)
    assert entry_point == reversed_defs


def test_state_change_between_invocations_is_observed(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = _tracked(root, "a.py", b"a\n")

    assert _results(root)[_CONTENT].outcome is DoctorCheckOutcome.PASS
    target.write_bytes(b"b\n")
    assert _results(root)[_CONTENT].outcome is DoctorCheckOutcome.FAIL


def test_one_invocation_collects_exactly_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _tracked(root, "a.py")
    calls = {"count": 0}
    real = tracked_module.collect_tracked_path_snapshot

    def _counting(project_root: object) -> object:
        calls["count"] += 1
        return real(project_root)  # type: ignore[arg-type]

    monkeypatch.setattr(tracked_module, "collect_tracked_path_snapshot", _counting)

    run_tracked_path_checks(DoctorCheckContext(project_root=root))

    assert calls["count"] == 1


def test_results_carry_no_volatile_field(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _tracked(root, "a.py")

    for result in run_tracked_path_checks(DoctorCheckContext(project_root=root)):
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
# G. Error mapping                                                             #
# --------------------------------------------------------------------------- #
def test_uninitialized_project_is_blocked_not_crashed(tmp_path: Path) -> None:
    results = _results(tmp_path)

    for check_id, result in results.items():
        assert result.outcome is DoctorCheckOutcome.SKIPPED, check_id
        assert result.outcome is not DoctorCheckOutcome.PASS, check_id


def test_corrupt_database_is_blocked_without_leaking(tmp_path: Path) -> None:
    root = _project(tmp_path)
    project_database_path(root).write_bytes(os.urandom(2048))

    results = _results(root)
    rendered = json.dumps([r.model_dump(mode="json") for r in results.values()])

    for result in results.values():
        assert result.outcome is DoctorCheckOutcome.SKIPPED
    assert "sqlite3" not in rendered
    assert "not a database" not in rendered


def test_non_posix_platform_is_not_applicable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracked_module, "anchored_traversal_supported", lambda: False)

    results = _results(Path("/nonexistent-project-root"))

    for check_id, result in results.items():
        assert result.outcome is DoctorCheckOutcome.NOT_APPLICABLE, check_id
        assert result.remediation is None, check_id


def test_module_uses_no_broad_exception_handler() -> None:
    source = Path(tracked_module.__file__).read_text(encoding="utf-8")

    assert "except Exception" not in source
    assert "except BaseException" not in source
    assert "except:" not in source


def test_programmer_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    _tracked(root, "a.py")

    def _bug(*args: object, **kwargs: object) -> object:
        raise TypeError("programmer error")

    monkeypatch.setattr(tracked_module, "_inspect_record", _bug)

    with pytest.raises(TypeError, match="programmer error"):
        run_tracked_path_checks(DoctorCheckContext(project_root=root))


def test_assertion_error_is_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    _tracked(root, "a.py")

    def _bug(*args: object, **kwargs: object) -> object:
        raise AssertionError("invariant violated")

    monkeypatch.setattr(tracked_module, "record_is_safe", _bug)

    with pytest.raises(AssertionError, match="invariant violated"):
        run_tracked_path_checks(DoctorCheckContext(project_root=root))


def test_record_is_safe_combines_every_field() -> None:
    from pmem.repositories.tracked_paths import TrackedPathRecord

    def _record(**overrides: object) -> TrackedPathRecord:
        base = {
            "id": "track_1",
            "project_id": "proj_1",
            "path": "a.py",
            "sha256": "a" * 64,
            "tag": None,
            "size_bytes": 1,
            "last_checked": "t",
            "created_at": "t",
        }
        base.update(overrides)
        return TrackedPathRecord(**base)  # type: ignore[arg-type]

    assert record_is_safe(_record()) is True
    assert record_is_safe(_record(path="../x")) is False
    assert record_is_safe(_record(sha256="zz")) is False
    assert record_is_safe(_record(size_bytes=-5)) is False


# --------------------------------------------------------------------------- #
# Defensive branches                                                           #
# --------------------------------------------------------------------------- #
def test_inspected_count_excludes_unsafe_records(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _tracked(root, "a.py")
    _insert_raw_record(root, path="../outside.py")

    snapshot = collect_tracked_path_snapshot(root)

    assert snapshot.record_count == 2
    assert snapshot.unsafe_record_count == 1
    assert snapshot.inspected_count == 1


def test_windows_drive_prefix_is_rejected_by_the_predicate() -> None:
    for value in ("C:/x", "c:\\x", "Z:", "a:/b"):
        assert stored_path_is_safe(value) is False


def test_leaf_stat_permission_error_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py")
    real_stat = os.stat

    def _deny(name: object, *args: object, **kwargs: object) -> object:
        if name == "train.py":
            raise PermissionError("denied")
        return real_stat(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "stat", _deny)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {_deny})

    results = _results(root)

    assert results[_PRESENT].outcome is DoctorCheckOutcome.SKIPPED
    assert results[_PRESENT].outcome is not DoctorCheckOutcome.PASS
    assert results[_SYMLINK].outcome is DoctorCheckOutcome.SKIPPED
    assert results[_SYMLINK].severity is DoctorSeverity.WARNING


def test_leaf_vanishing_between_stat_and_open_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py")
    real_open = os.open

    def _vanish(name: object, *args: object, **kwargs: object) -> int:
        if name == "train.py":
            raise FileNotFoundError("vanished")
        return real_open(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", _vanish)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {_vanish})

    assert _results(root)[_PRESENT].outcome is DoctorCheckOutcome.FAIL


def test_leaf_raced_into_another_inode_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py")

    monkeypatch.setattr(tracked_module, "same_file_binding", lambda fd, name, child: False)

    results = _results(root)

    assert results[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED
    assert results[_CONTENT].severity is DoctorSeverity.WARNING


def test_opened_leaf_that_is_not_a_regular_file_is_wrong_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fstat`` after open is the last line of defence against a type race."""

    body = b"unique-body-for-fstat-race\n"
    root = _project(tmp_path)
    _tracked(root, "src/train.py", body)
    real_fstat = os.fstat

    class _FifoStat:
        st_mode = stat_module.S_IFIFO | 0o600
        st_dev = 1
        st_ino = 1
        st_size = 0
        st_mtime_ns = 0

    def _fifo_for_the_tracked_file(descriptor: int) -> object:
        status = real_fstat(descriptor)
        if stat_module.S_ISREG(status.st_mode) and status.st_size == len(body):
            return _FifoStat()
        return status

    monkeypatch.setattr(os, "fstat", _fifo_for_the_tracked_file)

    results = _results(root)

    assert results[_PRESENT].outcome is DoctorCheckOutcome.FAIL
    assert results[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED


def test_parent_directory_stat_error_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py")
    real_stat = os.stat

    def _deny(name: object, *args: object, **kwargs: object) -> object:
        if name == "src":
            raise PermissionError("denied")
        return real_stat(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "stat", _deny)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {_deny})

    assert _results(root)[_PRESENT].outcome is not DoctorCheckOutcome.PASS


def test_parent_directory_open_errors_are_mapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py")
    real_open = os.open

    def _deny(name: object, *args: object, **kwargs: object) -> int:
        if name == "src":
            raise PermissionError("denied")
        return real_open(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", _deny)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {_deny})

    assert _results(root)[_PRESENT].outcome is not DoctorCheckOutcome.PASS


def test_parent_directory_vanishing_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _tracked(root, "src/train.py")
    real_open = os.open

    def _vanish(name: object, *args: object, **kwargs: object) -> int:
        if name == "src":
            raise FileNotFoundError("vanished")
        return real_open(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", _vanish)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {_vanish})

    assert _results(root)[_PRESENT].outcome is DoctorCheckOutcome.FAIL


def test_unopenable_project_root_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _tracked(root, "a.py")
    real_open = os.open

    def _deny(path: object, *args: object, **kwargs: object) -> int:
        if kwargs.get("dir_fd") is None and str(path) == str(root):
            raise PermissionError("denied")
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", _deny)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {_deny})

    for result in _results(root).values():
        assert result.outcome is DoctorCheckOutcome.SKIPPED


def test_repository_error_is_mapped_to_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pmem.repositories.tracked_paths import TrackedPathRepository

    root = _project(tmp_path)
    _tracked(root, "a.py")

    def _raise(self: object, project_id: str, *, limit: int) -> object:
        raise sqlite3.OperationalError("no such table: tracked_paths")

    monkeypatch.setattr(TrackedPathRepository, "list_for_project_limited", _raise)

    for result in _results(root).values():
        assert result.outcome is DoctorCheckOutcome.SKIPPED
        assert result.outcome is not DoctorCheckOutcome.PASS


def test_pathsafety_binding_helpers_fail_closed_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity proofs must answer ``False`` when the syscall itself fails."""

    from pmem.doctor import pathsafety

    root = _project(tmp_path)
    descriptor = pathsafety.open_directory(root)
    try:

        def _raise(*args: object, **kwargs: object) -> object:
            raise PermissionError("denied")

        monkeypatch.setattr(os, "stat", _raise)
        monkeypatch.setattr(Path, "lstat", _raise)

        assert pathsafety.same_directory_binding(descriptor, "x", descriptor) is False
        assert pathsafety.same_file_binding(descriptor, "x", descriptor) is False
        assert pathsafety.same_path_binding(root, descriptor) is False
    finally:
        pathsafety.close_quietly(descriptor)


def test_close_quietly_swallows_a_close_failure() -> None:
    from pmem.doctor import pathsafety

    descriptor = os.open(os.devnull, os.O_RDONLY)
    os.close(descriptor)

    pathsafety.close_quietly(descriptor)  # already closed: must not raise


def test_anchored_open_flags_refuse_to_follow_a_link() -> None:
    """``O_NOFOLLOW`` is asserted structurally, on purpose.

    Behaviourally it is a *second* barrier: if it were removed, an open that
    followed a raced symlink would still be caught by the dev/inode identity
    proof in :func:`same_file_binding`, so no outcome-level test can isolate it.
    Asserting the flag directly is what keeps the first barrier from being
    deleted silently.
    """

    from pmem.doctor.pathsafety import directory_open_flags, file_open_flags

    assert file_open_flags() & os.O_NOFOLLOW
    assert file_open_flags() & os.O_RDONLY == os.O_RDONLY
    assert directory_open_flags() & os.O_NOFOLLOW
    assert directory_open_flags() & os.O_DIRECTORY
    # write intent must never be requested
    for flag_name in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"):
        flag = getattr(os, flag_name, 0)
        if flag:
            assert not (file_open_flags() & flag), flag_name
            assert not (directory_open_flags() & flag), flag_name


def test_unc_style_double_slash_prefix_is_rejected() -> None:
    """``//server/share`` is a UNC reference, rejected by the leading-slash rule."""

    for value in ("//server/share/file.txt", "//x", "//"):
        assert stored_path_is_safe(value) is False


def test_parent_component_that_is_a_regular_file_is_wrong_type(tmp_path: Path) -> None:
    """A record whose parent became a file, not a directory, cannot be walked."""

    root = _project(tmp_path)
    _tracked(root, "src/train.py")
    (root / "src" / "train.py").unlink()
    (root / "src").rmdir()
    (root / "src").write_bytes(b"now a regular file\n")

    results = _results(root)

    assert results[_PRESENT].outcome is DoctorCheckOutcome.FAIL
    assert results[_CONTENT].outcome is DoctorCheckOutcome.SKIPPED


def test_sidecar_locked_database_is_blocked_not_crashed(tmp_path: Path) -> None:
    """An active SQLite sidecar makes the read-only connection refuse to open."""

    root = _project(tmp_path)
    _tracked(root, "a.py")
    database = project_database_path(root)
    database.with_name(database.name + "-wal").write_bytes(b"")

    results = _results(root)

    for check_id, result in results.items():
        assert result.outcome is DoctorCheckOutcome.SKIPPED, check_id
        assert result.outcome is not DoctorCheckOutcome.PASS, check_id
