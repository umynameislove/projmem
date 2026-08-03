"""Read-only and privacy regression tests for permission diagnostics (DOC-003).

Two guarantees are proved here against real directory trees:

1. running every permission diagnostic leaves the project byte-, mode-, mtime-
   and symlink-identical, and creates nothing;
2. no serialized result can carry a path, a file name, a symlink target, a
   project name, an OS error, or a marker planted anywhere in the tree.

Sensitive markers are planted in the project directory name, in file names, in
an artifact name and in a symlink target, then the whole result set is
serialized and searched.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

import pmem.doctor.permission_checks as permission_module
from pmem.doctor import DoctorCheckContext, DoctorCheckOutcome
from pmem.doctor.permission_checks import (
    PERMISSION_CHECK_IDS,
    permission_check_definitions,
    run_permission_checks,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="POSIX permission bits are required for these cases"
)

_MARKER = "PERMDOCTOR_MARKER_8b31d0a7"


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
    root_status = root.lstat()
    captured["<root>"] = (
        "dir",
        b"",
        0,
        stat.S_IMODE(root_status.st_mode),
        root_status.st_mtime_ns,
    )
    return captured


def _private_file(path: Path, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"private-content")
    path.chmod(mode)
    return path


def _private_dir(path: Path, mode: int = 0o700) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(mode)
    return path


def _marked_project(parent: Path) -> Path:
    """Build a project whose name, files and artifacts all carry the marker."""

    root = _private_dir(parent / f"project-{_MARKER}")
    pmem = _private_dir(root / ".pmem")
    _private_dir(pmem / "artifacts")
    _private_dir(pmem / "snapshots")
    _private_file(pmem / "pmem.db")
    _private_file(pmem / "config.yaml")
    _private_file(pmem / "graph.json")
    run_dir = _private_dir(pmem / "artifacts" / "runs" / f"run_{_MARKER}")
    _private_file(run_dir / "stdout.txt")
    _private_file(run_dir / f"artifact-{_MARKER}.bin")
    return root


def _rendered(root: Path) -> str:
    results = run_permission_checks(DoctorCheckContext(project_root=root))
    return json.dumps([result.model_dump(mode="json") for result in results], indent=2)


def _free_text(rendered: str) -> str:
    """Return only the operator-facing text of a serialized result set.

    ``check_id`` values are a fixed, spec-mandated vocabulary and are validated
    separately; scanning them for forbidden substrings would false-positive on
    ``permissions.pmem_directory``, which contains ``.pmem`` by construction.
    Privacy is a property of the free text, and that is what this returns.
    """

    document = json.loads(rendered)
    parts: list[str] = []
    for entry in document:
        parts.append(entry["message"])
        if entry["remediation"] is not None:
            parts.append(entry["remediation"])
    return "\n".join(parts)


_FORBIDDEN = (
    "/Users/",
    "/home/",
    "C:\\",
    ".pmem",
    "pmem.db",
    "config.yaml",
    "graph.json",
    "Traceback",
    "PermissionError",
    "OSError",
    "FileNotFoundError",
    "artifacts",
    "snapshots",
    "stdout.txt",
    "stderr.txt",
)


def _assert_clean(rendered: str, root: Path) -> None:
    text = _free_text(rendered)
    for forbidden in _FORBIDDEN:
        assert forbidden not in text, forbidden
    assert _MARKER not in rendered  # the marker must not appear anywhere at all
    assert str(root) not in rendered
    assert root.name not in rendered
    assert "/tmp" not in rendered
    assert "\x1b" not in rendered
    assert not any(ord(char) < 32 and char != "\n" for char in rendered)
    document = json.loads(rendered)
    # check_id is a fixed vocabulary, validated by equality rather than scanning
    assert {entry["check_id"] for entry in document} == set(PERMISSION_CHECK_IDS)
    assert all(entry["category"] == "permissions" for entry in document)


# --------------------------------------------------------------------------- #
# Read-only: nothing on disk may change                                        #
# --------------------------------------------------------------------------- #
def test_healthy_project_is_identical_after_diagnostics(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)
    before = _snapshot(root)

    run_permission_checks(DoctorCheckContext(project_root=root))

    assert _snapshot(root) == before


def test_unsafe_project_is_not_repaired(tmp_path: Path) -> None:
    """The diagnostic must report exposure, never chmod it away."""

    root = _marked_project(tmp_path)
    (root / ".pmem").chmod(0o755)
    (root / ".pmem" / "pmem.db").chmod(0o644)
    before = _snapshot(root)

    results = {r.check_id: r for r in run_permission_checks(DoctorCheckContext(project_root=root))}

    assert _snapshot(root) == before
    assert stat.S_IMODE((root / ".pmem").lstat().st_mode) == 0o755
    assert stat.S_IMODE((root / ".pmem" / "pmem.db").lstat().st_mode) == 0o644
    assert results["permissions.pmem_directory"].outcome is DoctorCheckOutcome.FAIL
    assert results["permissions.database"].outcome is DoctorCheckOutcome.FAIL


def test_diagnostics_never_call_a_mutating_syscall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any write-shaped call is an outright bug, so make one fatal."""

    root = _marked_project(tmp_path)

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("permission diagnostics must not mutate the filesystem")

    for name in ("chmod", "chown", "mkdir", "remove", "unlink", "rename", "replace", "truncate"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, _forbidden)
    for name in ("chmod", "mkdir", "unlink", "rename", "replace", "touch", "write_bytes"):
        if hasattr(Path, name):
            monkeypatch.setattr(Path, name, _forbidden)
    monkeypatch.setattr(Path, "open", _forbidden)
    monkeypatch.setattr("builtins.open", _forbidden)

    results = run_permission_checks(DoctorCheckContext(project_root=root))

    assert len(results) == 5


def test_uninitialized_project_creates_nothing(tmp_path: Path) -> None:
    before = sorted(path.name for path in tmp_path.iterdir())

    run_permission_checks(DoctorCheckContext(project_root=tmp_path))

    assert sorted(path.name for path in tmp_path.iterdir()) == before
    assert not (tmp_path / ".pmem").exists()


def test_symlink_targets_outside_the_project_are_untouched(tmp_path: Path) -> None:
    outside = tmp_path / f"outside-{_MARKER}.db"
    outside.write_text(f"outside content {_MARKER}", encoding="utf-8")
    outside.chmod(0o600)
    before_bytes = outside.read_bytes()
    before_mode = stat.S_IMODE(outside.lstat().st_mode)
    before_mtime = outside.lstat().st_mtime_ns

    root = _marked_project(tmp_path)
    database = root / ".pmem" / "pmem.db"
    database.unlink()
    try:
        os.symlink(outside, database)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    rendered = _rendered(root)

    assert outside.read_bytes() == before_bytes
    assert stat.S_IMODE(outside.lstat().st_mode) == before_mode
    assert outside.lstat().st_mtime_ns == before_mtime
    assert database.is_symlink()
    _assert_clean(rendered, root)


def test_symlinked_directory_target_is_neither_entered_nor_changed(tmp_path: Path) -> None:
    outside_dir = _private_dir(tmp_path / f"outside-dir-{_MARKER}")
    hidden = _private_file(outside_dir / f"hidden-{_MARKER}.txt")
    root = _private_dir(tmp_path / f"project-{_MARKER}")
    try:
        os.symlink(outside_dir, root / ".pmem")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")
    before = _snapshot(outside_dir)

    rendered = _rendered(root)

    assert _snapshot(outside_dir) == before
    assert hidden.read_bytes() == b"private-content"
    _assert_clean(rendered, root)


def test_file_content_is_never_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening any project file for reading would be a privacy regression."""

    root = _marked_project(tmp_path)

    def _forbidden_open(*args: object, **kwargs: object) -> object:
        raise AssertionError("permission diagnostics must not open file content")

    monkeypatch.setattr("builtins.open", _forbidden_open)
    monkeypatch.setattr(Path, "read_bytes", _forbidden_open)
    monkeypatch.setattr(Path, "read_text", _forbidden_open)

    results = run_permission_checks(DoctorCheckContext(project_root=root))

    assert len(results) == 5


def test_no_network_or_database_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import socket
    import sqlite3

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("permission diagnostics must not open a socket or a database")

    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(sqlite3, "connect", _forbidden)

    results = run_permission_checks(DoctorCheckContext(project_root=_marked_project(tmp_path)))

    assert len(results) == 5


def test_resolve_is_never_used_on_an_inspected_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``resolve()`` would follow links and defeat the whole guarantee."""

    root = _marked_project(tmp_path)

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("permission diagnostics must not resolve inspected paths")

    monkeypatch.setattr(Path, "resolve", _forbidden)
    monkeypatch.setattr(os.path, "realpath", _forbidden)

    results = run_permission_checks(DoctorCheckContext(project_root=root))

    assert len(results) == 5


# --------------------------------------------------------------------------- #
# Privacy: nothing sensitive may reach the serialized results                  #
# --------------------------------------------------------------------------- #
def test_healthy_results_leak_nothing(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)

    _assert_clean(_rendered(root), root)


def test_unsafe_results_leak_nothing(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)
    (root / ".pmem").chmod(0o755)
    (root / ".pmem" / "config.yaml").chmod(0o666)
    (root / ".pmem" / "artifacts" / "runs" / f"run_{_MARKER}").chmod(0o777)

    _assert_clean(_rendered(root), root)


def test_uninitialized_results_leak_nothing(tmp_path: Path) -> None:
    root = _private_dir(tmp_path / f"project-{_MARKER}")

    _assert_clean(_rendered(root), root)


def test_os_error_text_never_reaches_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw OS error carries a path; it must be discarded at the boundary."""

    root = _marked_project(tmp_path)
    real_stat = os.stat

    def _raise(name: object, *args: object, **kwargs: object) -> os.stat_result:
        if name in {"pmem.db", "config.yaml", "graph.json"}:
            raise PermissionError(13, f"Permission denied leaking {_MARKER}", str(name))
        return real_stat(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "stat", _raise)
    monkeypatch.setattr(permission_module, "posix_modes_are_supported", lambda: True)

    rendered = _rendered(root)

    _assert_clean(rendered, root)
    assert "Permission denied" not in rendered


def test_scandir_error_text_never_reaches_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _marked_project(tmp_path)
    real_reader = permission_module._bounded_directory_names
    calls = {"count": 0}

    def _raise(directory_fd: int, remaining: int) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError(13, f"Cannot list {_MARKER}", "sensitive-path")
        return real_reader(directory_fd, remaining)

    monkeypatch.setattr(permission_module, "_bounded_directory_names", _raise)

    rendered = _rendered(root)

    _assert_clean(rendered, root)
    assert "Cannot list" not in rendered


def test_observed_mode_is_never_rendered(tmp_path: Path) -> None:
    """Modes live in the internal snapshot only; results must not echo them."""

    root = _marked_project(tmp_path)
    (root / ".pmem" / "pmem.db").chmod(0o642)

    rendered = _rendered(root)
    text = _free_text(rendered)

    # every spelling a leaked mode could take: octal literal, octal digits,
    # and the decimal value ``str(mode)`` would produce
    for spelling in ("0o642", "0642", "642", "0o600", "0600", "600", "418", "384", "33188"):
        assert spelling not in text, spelling
    # stronger and future-proof: the operator-facing text carries no digits at
    # all, so no numeric observation can hide in it
    assert not any(char.isdigit() for char in text)
    _assert_clean(rendered, root)


def test_no_message_is_built_by_interpolation(tmp_path: Path) -> None:
    """Interpolated text would break the privacy guarantee; pin it structurally."""

    source = Path(permission_module.__file__).read_text(encoding="utf-8")

    assert 'f"' not in source
    assert "f'" not in source
    assert ".format(" not in source
    assert "% (" not in source
    assert "str(exc" not in source
    assert "repr(" not in source


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda parent: _marked_project(parent), id="healthy"),
        pytest.param(
            lambda parent: _private_dir(parent / f"project-{_MARKER}"), id="uninitialized"
        ),
    ],
)
def test_every_emitted_text_is_a_literal_from_the_module(tmp_path: Path, build: object) -> None:
    """Each message/remediation must be a constant defined in the source, verbatim."""

    def _squash(value: str) -> str:
        """Drop quotes and whitespace so implicit concatenation still matches."""

        return "".join(value.split()).replace('"', "").replace("'", "")

    root = build(tmp_path)  # type: ignore[operator]
    source = _squash(Path(permission_module.__file__).read_text(encoding="utf-8"))

    emitted = 0
    for result in run_permission_checks(DoctorCheckContext(project_root=root)):
        for text in (result.message, result.remediation):
            if text is None:
                continue
            assert _squash(text) in source, text
            emitted += 1

    assert emitted >= 5  # the loop actually asserted something


def test_definitions_and_entry_point_agree_and_both_stay_clean(tmp_path: Path) -> None:
    root = _marked_project(tmp_path)
    context = DoctorCheckContext(project_root=root)

    from_definitions = {d.check_id: d.execute(context) for d in permission_check_definitions()}
    from_entry_point = {r.check_id: r for r in run_permission_checks(context)}

    assert from_definitions == from_entry_point
    rendered = json.dumps(
        [
            result.model_dump(mode="json")
            for result in sorted(from_definitions.values(), key=lambda r: r.check_id)
        ]
    )
    for forbidden in _FORBIDDEN:
        assert forbidden not in _free_text(rendered), forbidden
    assert _MARKER not in rendered
