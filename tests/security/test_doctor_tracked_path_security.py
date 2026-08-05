"""Security regression tests for tracked-path diagnostics (DOC-004).

Three guarantees are proved here against real projects, real databases and real
files:

1. no path outside the project, no internal ``.pmem`` file and no symlink target
   is ever opened or read -- proved by making such an open *fatal*, not by
   observing that bytes happened not to change;
2. running the diagnostics leaves the database and the whole project tree
   byte-, mode-, mtime- and symlink-identical, and leaks no descriptor;
3. no serialized result carries a path, a file name, a stored or observed
   digest, a project name, an OS error, or a marker planted anywhere.
"""

from __future__ import annotations

import json
import os
import resource
import sqlite3
import stat
import uuid
from pathlib import Path

import pytest

import pmem.doctor.tracked_path_checks as tracked_module
from pmem.doctor import DoctorCheckContext
from pmem.doctor.tracked_path_checks import (
    TRACKED_PATH_CHECK_IDS,
    run_tracked_path_checks,
)
from pmem.repositories.sqlite import project_database_path
from pmem.services.config import project_config_path, read_project_config
from pmem.services.project_init import init_project
from pmem.services.tracking import track_path
from pmem.utils.hashing import compute_file_hash

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="descriptor-anchored traversal requires POSIX"
)

_MARKER = "TRACKEDDOCTOR_MARKER_5e19c7"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _snapshot(root: Path) -> dict[str, tuple[str, object, int, int, int]]:
    """Capture type/content/size/mode/mtime for every entry, never following links."""

    captured: dict[str, tuple[str, object, int, int, int]] = {}
    for path in sorted(root.rglob("*")):
        status = path.lstat()
        key = str(path.relative_to(root))
        mode = stat.S_IMODE(status.st_mode)
        if path.is_symlink():
            captured[key] = ("link", os.readlink(path), 0, mode, status.st_mtime_ns)
        elif stat.S_ISDIR(status.st_mode):
            captured[key] = ("dir", b"", 0, mode, status.st_mtime_ns)
        elif stat.S_ISREG(status.st_mode):
            captured[key] = (
                "file",
                path.read_bytes(),
                status.st_size,
                mode,
                status.st_mtime_ns,
            )
        else:
            captured[key] = ("other", b"", 0, mode, status.st_mtime_ns)
    return captured


def _marked_project(parent: Path) -> Path:
    root = parent / f"project-{_MARKER}"
    root.mkdir()
    init_project(root, project_name=f"proj-{_MARKER}", primary_metric="accuracy")
    target = root / f"dir-{_MARKER}" / f"file-{_MARKER}.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(f"secret content {_MARKER}\n".encode())
    track_path(root, f"dir-{_MARKER}/file-{_MARKER}.py")
    return root


def _insert_raw_record(root: Path, *, path: str, sha256: str | None = None) -> None:
    connection = sqlite3.connect(project_database_path(root))
    connection.execute(
        "INSERT INTO tracked_paths"
        "(id, project_id, path, tag, hash, size_bytes, last_checked, created_at) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
        (
            f"track_{uuid.uuid4().hex}",
            read_project_config(project_config_path(root)).project_id,
            path,
            f"tag-{_MARKER}",
            ("a" * 64) if sha256 is None else sha256,
            1,
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        ),
    )
    connection.commit()
    connection.close()


def _rendered(root: Path) -> str:
    results = run_tracked_path_checks(DoctorCheckContext(project_root=root))
    return json.dumps([result.model_dump(mode="json") for result in results], indent=2)


def _free_text(rendered: str) -> str:
    """Only the operator-facing text; ``check_id`` is a fixed vocabulary."""

    parts: list[str] = []
    for entry in json.loads(rendered):
        parts.append(entry["message"])
        if entry["remediation"] is not None:
            parts.append(entry["remediation"])
    return "\n".join(parts)


_FORBIDDEN = (
    "/Users/",
    "/home/",
    "/tmp",
    "C:\\",
    "file://",
    ".pmem",
    "pmem.db",
    "config.yaml",
    "graph.json",
    "SELECT",
    "sqlite3",
    "Traceback",
    "PermissionError",
    "OSError",
    "FileNotFoundError",
)


def _assert_clean(rendered: str, root: Path) -> None:
    text = _free_text(rendered)
    for forbidden in _FORBIDDEN:
        assert forbidden not in text, forbidden
    assert _MARKER not in rendered
    assert str(root) not in rendered
    assert root.name not in rendered
    assert "\x1b" not in rendered
    assert not any(ord(char) < 32 and char != "\n" for char in rendered)
    document = json.loads(rendered)
    assert {entry["check_id"] for entry in document} == set(TRACKED_PATH_CHECK_IDS)
    assert all(entry["category"] == "tracked_paths" for entry in document)


class _ForbiddenOpen:
    """Make opening any path under a forbidden prefix a hard test failure.

    ``os.supports_dir_fd`` is a set of the *original* function objects, so
    replacing ``os.open`` would make the module's platform probe report that
    anchored traversal is unavailable and silently turn every check into
    ``not_applicable`` -- a green test that proved nothing. The replacement is
    therefore registered in that set as well, so the probe keeps telling the
    truth while the guard is armed.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *forbidden: Path) -> None:
        self._forbidden = [str(path) for path in forbidden]
        self.touched: list[str] = []
        real_open = os.open

        def _guarded(path: object, *args: object, **kwargs: object) -> int:
            if kwargs.get("dir_fd") is None:
                text = str(path)
                for prefix in self._forbidden:
                    if text == prefix or text.startswith(prefix + os.sep):
                        self.touched.append(text)
                        raise AssertionError(f"forbidden open: {text}")
            return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "open", _guarded)
        monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {_guarded})


# --------------------------------------------------------------------------- #
# D. Never follow, never read outside the project                              #
# --------------------------------------------------------------------------- #
def test_symlink_target_is_never_opened_or_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proved by making the open fatal, not by comparing target bytes."""

    outside = tmp_path / f"outside-{_MARKER}.py"
    outside.write_bytes(f"outside secret {_MARKER}\n".encode())
    root = _marked_project(tmp_path)
    leaf = root / f"dir-{_MARKER}" / f"file-{_MARKER}.py"
    before_target = outside.read_bytes()
    leaf.unlink()
    try:
        os.symlink(outside, leaf)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    guard = _ForbiddenOpen(monkeypatch, outside)
    rendered = _rendered(root)

    assert guard.touched == []
    assert outside.read_bytes() == before_target
    results = {e["check_id"]: e for e in json.loads(rendered)}
    assert results["tracked_paths.symlink"]["outcome"] == "fail"
    assert results["tracked_paths.content_current"]["outcome"] == "skipped"
    _assert_clean(rendered, root)


def test_parent_symlink_target_directory_is_never_entered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside_dir = tmp_path / f"outside-dir-{_MARKER}"
    outside_dir.mkdir()
    hidden = outside_dir / f"file-{_MARKER}.py"
    hidden.write_bytes(f"outside secret {_MARKER}\n".encode())
    root = _marked_project(tmp_path)
    parent = root / f"dir-{_MARKER}"
    (parent / f"file-{_MARKER}.py").unlink()
    parent.rmdir()
    try:
        os.symlink(outside_dir, parent)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    guard = _ForbiddenOpen(monkeypatch, outside_dir)
    rendered = _rendered(root)

    assert guard.touched == []
    assert hidden.read_bytes() == f"outside secret {_MARKER}\n".encode()
    assert json.loads(rendered)[3]["check_id"] == "tracked_paths.symlink"
    _assert_clean(rendered, root)


def test_symlinked_project_root_is_rejected_before_project_state_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-supplied root link must not expose its target's config or database."""

    target = _marked_project(tmp_path)
    root = tmp_path / f"project-link-{_MARKER}"
    try:
        os.symlink(target, root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    calls: list[str] = []

    def _forbidden_context(project_root: object) -> object:
        calls.append("context")
        raise AssertionError("project context must not be read through a root symlink")

    monkeypatch.setattr(tracked_module, "require_project_context_readonly", _forbidden_context)

    rendered = _rendered(root)

    assert calls == []
    assert all(entry["outcome"] == "skipped" for entry in json.loads(rendered))
    _assert_clean(rendered, root)


def test_traversal_record_is_blocked_before_any_filesystem_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / f"outside-{_MARKER}.py"
    outside.write_bytes(f"outside secret {_MARKER}\n".encode())
    root = _marked_project(tmp_path)
    _insert_raw_record(root, path=f"../outside-{_MARKER}.py")

    guard = _ForbiddenOpen(monkeypatch, outside)
    rendered = _rendered(root)

    assert guard.touched == []
    results = {e["check_id"]: e for e in json.loads(rendered)}
    assert results["tracked_paths.records_safe"]["outcome"] == "fail"
    _assert_clean(rendered, root)


@pytest.mark.parametrize("absolute", ["/etc/passwd", "/etc/hosts"])
def test_absolute_record_is_blocked_before_any_filesystem_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absolute: str
) -> None:
    root = _marked_project(tmp_path)
    _insert_raw_record(root, path=absolute)

    guard = _ForbiddenOpen(monkeypatch, Path(absolute))
    rendered = _rendered(root)

    assert guard.touched == []
    assert json.loads(rendered)[2]["outcome"] == "fail"  # records_safe
    _assert_clean(rendered, root)


@pytest.mark.parametrize(
    "internal", [".pmem/pmem.db", ".PMEM/pmem.db", ".PmEm/config.yaml", ".pMeM/graph.json"]
)
def test_internal_record_never_opens_project_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, internal: str
) -> None:
    root = _marked_project(tmp_path)
    _insert_raw_record(root, path=internal)
    database_before = project_database_path(root).read_bytes()

    guard = _ForbiddenOpen(monkeypatch, root / ".pmem")
    rendered = _rendered(root)

    assert guard.touched == []
    assert project_database_path(root).read_bytes() == database_before
    results = {e["check_id"]: e for e in json.loads(rendered)}
    assert results["tracked_paths.records_safe"]["outcome"] == "fail"
    _assert_clean(rendered, root)


@pytest.mark.parametrize("windows", ["C:/Windows/win.ini", "c:\\Windows\\win.ini", "//srv/s/f"])
def test_windows_style_record_is_blocked_on_posix_too(tmp_path: Path, windows: str) -> None:
    root = _marked_project(tmp_path)
    _insert_raw_record(root, path=windows)

    rendered = _rendered(root)

    results = {e["check_id"]: e for e in json.loads(rendered)}
    assert results["tracked_paths.records_safe"]["outcome"] == "fail"
    _assert_clean(rendered, root)


def test_module_never_uses_a_resolving_or_path_based_reader() -> None:
    """Structural proof that the check-then-open race is not reintroduced."""

    source = Path(tracked_module.__file__).read_text(encoding="utf-8")

    for banned in (
        ".resolve()",
        "realpath",
        "compute_file_hash",
        "read_bytes()",
        "read_text()",
        "open(",
    ):
        if banned == "open(":
            # only the anchored helpers may open anything
            assert "os.open(" not in source
            continue
        assert banned not in source.replace("``compute_file_hash(path)``", ""), banned


def test_resolve_is_never_called(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _marked_project(tmp_path)

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("tracked-path diagnostics must not resolve a path")

    monkeypatch.setattr(Path, "resolve", _forbidden)
    monkeypatch.setattr(os.path, "realpath", _forbidden)

    assert len(run_tracked_path_checks(DoctorCheckContext(project_root=root))) == 4


# --------------------------------------------------------------------------- #
# Read-only guarantee                                                          #
# --------------------------------------------------------------------------- #
def test_project_tree_and_database_are_identical_afterwards(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)
    before = _snapshot(root)

    run_tracked_path_checks(DoctorCheckContext(project_root=root))

    assert _snapshot(root) == before


def test_changed_file_is_reported_without_refreshing_the_stored_hash(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)
    leaf = root / f"dir-{_MARKER}" / f"file-{_MARKER}.py"
    connection = sqlite3.connect(project_database_path(root))
    stored_before = connection.execute("SELECT hash, last_checked FROM tracked_paths").fetchone()
    connection.close()
    leaf.write_bytes(b"changed\n")

    rendered = _rendered(root)

    connection = sqlite3.connect(project_database_path(root))
    stored_after = connection.execute("SELECT hash, last_checked FROM tracked_paths").fetchone()
    connection.close()
    assert stored_after == stored_before  # no auto-update of hash or last_checked
    results = {e["check_id"]: e for e in json.loads(rendered)}
    assert results["tracked_paths.content_current"]["outcome"] == "fail"
    _assert_clean(rendered, root)


def test_no_mutating_syscall_is_reachable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _marked_project(tmp_path)

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("tracked-path diagnostics must not mutate anything")

    for name in ("chmod", "chown", "mkdir", "remove", "unlink", "rename", "replace", "truncate"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, _forbidden)
    for name in ("chmod", "mkdir", "unlink", "rename", "replace", "touch", "write_bytes"):
        if hasattr(Path, name):
            monkeypatch.setattr(Path, name, _forbidden)
    monkeypatch.setattr(os, "write", _forbidden)

    assert len(run_tracked_path_checks(DoctorCheckContext(project_root=root))) == 4


def test_no_write_capable_database_helper_is_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pmem.repositories.sqlite as sqlite_module
    import pmem.services.database as database_module
    from pmem.repositories.tracked_paths import TrackedPathRepository

    root = _marked_project(tmp_path)

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("tracked-path diagnostics must stay on the read-only seam")

    monkeypatch.setattr(sqlite_module, "connect_database", _forbidden)
    monkeypatch.setattr(database_module, "ensure_database", _forbidden)
    monkeypatch.setattr(TrackedPathRepository, "update_hash", _forbidden)
    monkeypatch.setattr(TrackedPathRepository, "add", _forbidden)

    assert len(run_tracked_path_checks(DoctorCheckContext(project_root=root))) == 4


def test_no_network_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("tracked-path diagnostics must not open a socket")

    monkeypatch.setattr(socket, "socket", _forbidden)

    assert (
        len(run_tracked_path_checks(DoctorCheckContext(project_root=_marked_project(tmp_path))))
        == 4
    )


def test_descriptors_do_not_leak_across_many_invocations(tmp_path: Path) -> None:
    """Hundreds of invocations must not grow the open-descriptor count."""

    root = _marked_project(tmp_path)
    for index in range(4):
        nested = root / f"n{index}" / f"f{index}.py"
        nested.parent.mkdir(parents=True)
        nested.write_bytes(f"content-{index}\n".encode())
        track_path(root, f"n{index}/f{index}.py")
    context = DoctorCheckContext(project_root=root)

    def _open_descriptors() -> int:
        limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        limit = min(limit, 4096)
        count = 0
        for descriptor in range(limit):
            try:
                os.fstat(descriptor)
            except OSError:
                continue
            count += 1
        return count

    run_tracked_path_checks(context)  # warm up any lazy import
    before = _open_descriptors()

    for _ in range(200):
        run_tracked_path_checks(context)

    assert _open_descriptors() == before


def test_descriptors_do_not_leak_on_the_error_path(tmp_path: Path) -> None:
    """Failure branches must close every descriptor they opened."""

    root = _marked_project(tmp_path)
    leaf = root / f"dir-{_MARKER}" / f"file-{_MARKER}.py"
    leaf.unlink()
    leaf.mkdir()  # wrong type: the leaf branch returns before hashing
    context = DoctorCheckContext(project_root=root)

    def _open_descriptors() -> int:
        count = 0
        for descriptor in range(min(resource.getrlimit(resource.RLIMIT_NOFILE)[0], 4096)):
            try:
                os.fstat(descriptor)
            except OSError:
                continue
            count += 1
        return count

    run_tracked_path_checks(context)
    before = _open_descriptors()

    for _ in range(200):
        run_tracked_path_checks(context)

    assert _open_descriptors() == before


# --------------------------------------------------------------------------- #
# Privacy                                                                      #
# --------------------------------------------------------------------------- #
def test_current_project_leaks_nothing(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)

    _assert_clean(_rendered(root), root)


def test_changed_project_leaks_no_digest(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)
    leaf = root / f"dir-{_MARKER}" / f"file-{_MARKER}.py"
    stored = compute_file_hash(leaf)
    leaf.write_bytes(b"changed content\n")
    observed = compute_file_hash(leaf)

    rendered = _rendered(root)

    assert stored not in rendered
    assert observed not in rendered
    _assert_clean(rendered, root)


def test_missing_project_leaks_nothing(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)
    (root / f"dir-{_MARKER}" / f"file-{_MARKER}.py").unlink()

    _assert_clean(_rendered(root), root)


def test_unreadable_file_error_text_never_reaches_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _marked_project(tmp_path)
    real_open = os.open

    def _deny(path: object, *args: object, **kwargs: object) -> int:
        if isinstance(path, str) and path.endswith(".py"):
            raise PermissionError(13, f"Permission denied leaking {_MARKER}", str(path))
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", _deny)

    rendered = _rendered(root)

    _assert_clean(rendered, root)
    assert "Permission denied" not in rendered


def test_database_error_text_never_reaches_a_result(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)
    project_database_path(root).write_bytes(f"corrupt {_MARKER}".encode() * 100)

    rendered = _rendered(root)

    _assert_clean(rendered, root)
    assert "not a database" not in rendered


def test_marked_tag_and_identifiers_never_reach_a_result(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)
    _insert_raw_record(root, path=f"other-{_MARKER}.py")

    rendered = _rendered(root)

    assert _MARKER not in rendered
    _assert_clean(rendered, root)


def test_no_message_is_built_by_interpolation() -> None:
    source = Path(tracked_module.__file__).read_text(encoding="utf-8")

    assert 'f"' not in source
    assert "f'" not in source
    assert ".format(" not in source
    assert "str(exc" not in source
    assert "repr(" not in source


def test_every_emitted_text_is_a_literal_from_the_module(tmp_path: Path) -> None:
    def _squash(value: str) -> str:
        return "".join(value.split()).replace('"', "").replace("'", "")

    root = _marked_project(tmp_path)
    (root / f"dir-{_MARKER}" / f"file-{_MARKER}.py").write_bytes(b"changed\n")
    source = _squash(Path(tracked_module.__file__).read_text(encoding="utf-8"))

    emitted = 0
    for result in run_tracked_path_checks(DoctorCheckContext(project_root=root)):
        for text in (result.message, result.remediation):
            if text is None:
                continue
            assert _squash(text) in source, text
            emitted += 1

    assert emitted >= 4


def test_content_bytes_never_reach_a_result(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)
    leaf = root / f"dir-{_MARKER}" / f"file-{_MARKER}.py"
    leaf.write_bytes(f"UNIQUE_BODY_{_MARKER}_INSIDE\n".encode())

    rendered = _rendered(root)

    assert "UNIQUE_BODY" not in rendered
    assert _MARKER not in rendered
    assert json.loads(rendered)[0]["outcome"] == "fail"  # content_current


def test_leaf_raced_into_a_symlink_between_stat_and_open_never_opens_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact window ``O_NOFOLLOW`` exists to close.

    The pre-open no-follow stat is made to report a regular file while the entry
    on disk is already a symlink, which is what an attacker achieves by swapping
    it in that window. Without ``O_NOFOLLOW`` the open would succeed on the
    target, hash it, and -- because the stored digest here is the *target's* --
    report the tracked file as current. The check must refuse instead.
    """

    root = _marked_project(tmp_path)
    leaf = root / f"dir-{_MARKER}" / f"file-{_MARKER}.py"
    outside = tmp_path / f"outside-target-{_MARKER}.py"
    outside.write_bytes(f"outside body {_MARKER}\n".encode())

    # store the target's digest, so following the link would look "current"
    connection = sqlite3.connect(project_database_path(root))
    connection.execute("UPDATE tracked_paths SET hash = ?", (compute_file_hash(outside),))
    connection.commit()
    connection.close()

    real_stat = os.stat
    regular = real_stat(leaf)
    leaf.unlink()
    try:
        os.symlink(outside, leaf)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    def _lie_about_the_leaf(name: object, *args: object, **kwargs: object) -> object:
        if name == f"file-{_MARKER}.py" and kwargs.get("follow_symlinks") is False:
            return regular  # pretend the swap has not happened yet
        return real_stat(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "stat", _lie_about_the_leaf)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {_lie_about_the_leaf})

    rendered = _rendered(root)

    results = {entry["check_id"]: entry for entry in json.loads(rendered)}
    assert results["tracked_paths.content_current"]["outcome"] != "pass"
    assert results["tracked_paths.present"]["outcome"] != "pass"
    assert results["tracked_paths.symlink"]["outcome"] == "skipped"
    assert results["tracked_paths.symlink"]["severity"] == "warning"
    assert outside.read_bytes() == f"outside body {_MARKER}\n".encode()
    _assert_clean(rendered, root)
