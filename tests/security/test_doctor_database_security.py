"""Read-only and privacy regression tests for database doctor checks (DOC-002).

Two guarantees are proved here, both against real SQLite files:

1. running every database diagnostic leaves the project byte-, mtime- and
   mode-identical, and creates nothing;
2. no serialized result can carry a path, raw SQL, a raw SQLite error, a
   checksum, a table name or a secret planted in the database.

The tests are written to go red if the production code regresses: swapping the
read-only connection for a writable one, dropping the symlink guard, running a
migration, or interpolating ``str(exc)`` each breaks at least one assertion.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from pmem.doctor import (
    DoctorCheckOutcome,
    DoctorCheckResult,
    DoctorProject,
    build_doctor_report,
    render_doctor_report_json,
)
from pmem.doctor.database_checks import (
    DATABASE_CHECK_IDS,
    run_database_checks,
)
from pmem.doctor.registry import DoctorCheckContext
from pmem.migrations.runner import CURRENT_MIGRATIONS
from pmem.repositories.sqlite import PMEM_DIRNAME, project_database_path
from pmem.services.project_init import init_project

_SENSITIVE_MARKER = "DOCTOR_DB_MARKER_4f7a91c3"
_PROJECT = DoctorProject(
    project_id="proj_9f2c1a7b4d6e40f2a1b3c5d7e9f00112",
    project_name="AG News baseline",
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _snapshot(root: Path) -> dict[str, tuple[bool, bytes | str, int, int]]:
    """Capture bytes/mtime/mode for every entry under the project root.

    Symlinks are recorded by their target text and never followed, so a test
    that accidentally dereferenced one would show up as a changed snapshot.
    """

    snapshot: dict[str, tuple[bool, bytes | str, int, int]] = {}
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        key = str(path.relative_to(root))
        if path.is_symlink():
            snapshot[key] = (True, os.readlink(path), stat.st_mtime_ns, stat.st_mode)
        elif path.is_file():
            snapshot[key] = (False, path.read_bytes(), stat.st_mtime_ns, stat.st_mode)
        else:
            snapshot[key] = (False, b"<dir>", stat.st_mtime_ns, stat.st_mode)
    return snapshot


def _run_all(root: Path) -> tuple[DoctorCheckResult, ...]:
    context = DoctorCheckContext(project_root=root)
    return run_database_checks(context)


def _rendered(root: Path) -> str:
    """Render a full report. Uses a fixed placeholder identity on purpose.

    The identity is a constant unrelated to the project under test, so any
    project text that appears in the output must have come from a database
    check -- which is exactly what these tests are trying to catch.
    """

    results = _run_all(root)
    report = build_doctor_report(project=_PROJECT, checks=results)
    return render_doctor_report_json(report)


def _project_with_secret(tmp_path: Path) -> Path:
    """A real migrated project whose database contains a unique sensitive marker."""

    init_project(tmp_path, project_name="doctor-secret", primary_metric="accuracy")
    connection = sqlite3.connect(project_database_path(tmp_path))
    connection.execute(
        "UPDATE projects SET current_objective = ?",
        (f"objective containing {_SENSITIVE_MARKER}",),
    )
    connection.commit()
    connection.close()
    return tmp_path


# --------------------------------------------------------------------------- #
# Read-only regression: nothing on disk may change                             #
# --------------------------------------------------------------------------- #
def test_healthy_project_is_byte_identical_after_diagnostics(tmp_path: Path) -> None:
    init_project(tmp_path, project_name="doctor-readonly", primary_metric="accuracy")
    before = _snapshot(tmp_path)

    _run_all(tmp_path)

    assert _snapshot(tmp_path) == before


def test_diagnostics_do_not_chmod_a_group_readable_database(tmp_path: Path) -> None:
    """``connect_database`` would chmod to 0600; the read-only seam must not."""

    init_project(tmp_path, project_name="doctor-mode", primary_metric="accuracy")
    db_path = project_database_path(tmp_path)
    db_path.chmod(0o644)
    before_mode = db_path.stat().st_mode

    _run_all(tmp_path)

    assert db_path.stat().st_mode == before_mode
    assert before_mode & 0o777 == 0o644


def test_diagnostics_create_no_sidecar_or_backup(tmp_path: Path) -> None:
    init_project(tmp_path, project_name="doctor-nosidecar", primary_metric="accuracy")

    _run_all(tmp_path)

    pmem_dir = tmp_path / PMEM_DIRNAME
    assert not list(pmem_dir.glob("pmem.db-*"))
    assert not list(pmem_dir.rglob("*.bak"))


def test_missing_project_is_not_initialized_by_diagnostics(tmp_path: Path) -> None:
    before = sorted(path.name for path in tmp_path.iterdir())

    results = _run_all(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == before
    assert not (tmp_path / PMEM_DIRNAME).exists()
    assert results[0].outcome is DoctorCheckOutcome.FAIL


def test_out_of_date_schema_is_not_migrated(tmp_path: Path) -> None:
    """The diagnostic must report a stale schema, never repair it."""

    init_project(tmp_path, project_name="doctor-nomigrate", primary_metric="accuracy")
    db_path = project_database_path(tmp_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        "DELETE FROM schema_migrations WHERE version = ?", (CURRENT_MIGRATIONS[-1].version,)
    )
    connection.commit()
    connection.close()
    before = _snapshot(tmp_path)

    _run_all(tmp_path)

    assert _snapshot(tmp_path) == before
    recorded = (
        sqlite3.connect(db_path).execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
    )
    assert recorded == len(CURRENT_MIGRATIONS) - 1  # still not migrated


def test_corrupt_database_is_never_replaced_or_backed_up(tmp_path: Path) -> None:
    db_path = project_database_path(tmp_path)
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(os.urandom(4096))
    before = _snapshot(tmp_path)

    _run_all(tmp_path)

    assert _snapshot(tmp_path) == before
    assert not list(db_path.parent.glob("*.bak"))


def test_active_sidecars_are_never_removed_or_checkpointed(tmp_path: Path) -> None:
    init_project(tmp_path, project_name="doctor-wal", primary_metric="accuracy")
    db_path = project_database_path(tmp_path)
    for suffix in ("-wal", "-shm"):
        db_path.with_name(db_path.name + suffix).write_bytes(b"sidecar")
    before = _snapshot(tmp_path)

    _run_all(tmp_path)

    assert _snapshot(tmp_path) == before
    for suffix in ("-wal", "-shm"):
        assert db_path.with_name(db_path.name + suffix).read_bytes() == b"sidecar"


def test_symlinked_database_target_is_neither_read_nor_written(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-doctor-security.db"
    outside.write_text(f"outside content {_SENSITIVE_MARKER}", encoding="utf-8")
    db_path = project_database_path(tmp_path)
    db_path.parent.mkdir(parents=True)
    try:
        os.symlink(outside, db_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")
    before_outside = outside.read_bytes()
    before = _snapshot(tmp_path)

    rendered = _rendered(tmp_path)

    assert _snapshot(tmp_path) == before
    assert outside.read_bytes() == before_outside
    assert _SENSITIVE_MARKER not in rendered
    assert db_path.is_symlink()  # guard did not replace the link


def test_symlinked_pmem_directory_target_is_untouched(tmp_path: Path) -> None:
    outside_dir = tmp_path.parent / "outside-doctor-pmem-dir"
    outside_dir.mkdir(exist_ok=True)
    secret_file = outside_dir / "pmem.db"
    secret_file.write_text(_SENSITIVE_MARKER, encoding="utf-8")
    try:
        os.symlink(outside_dir, tmp_path / PMEM_DIRNAME)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    rendered = _rendered(tmp_path)

    assert secret_file.read_text(encoding="utf-8") == _SENSITIVE_MARKER
    assert _SENSITIVE_MARKER not in rendered


def test_unreadable_database_is_skipped_safely_without_mutation(tmp_path: Path) -> None:
    init_project(tmp_path, project_name="doctor-perm", primary_metric="accuracy")
    db_path = project_database_path(tmp_path)
    original_mode = db_path.stat().st_mode & 0o777
    before = _snapshot(tmp_path)  # taken while the file is still readable

    db_path.chmod(0o000)
    if os.access(db_path, os.R_OK):  # root, or a filesystem without POSIX modes
        db_path.chmod(original_mode)
        pytest.skip("permission bits are not enforced for the current user")
    try:
        rendered = _rendered(tmp_path)
    finally:
        # chmod updates ctime, not mtime, so restoring the mode leaves the
        # snapshot comparison below meaningful.
        db_path.chmod(original_mode)

    assert _snapshot(tmp_path) == before
    document = json.loads(rendered)
    readable = next(
        check for check in document["checks"] if check["check_id"] == "database.readable"
    )
    assert readable["outcome"] != "pass"
    _assert_report_is_clean(rendered, tmp_path)


# --------------------------------------------------------------------------- #
# Privacy: nothing sensitive may reach the serialized report                   #
# --------------------------------------------------------------------------- #
_FORBIDDEN_SUBSTRINGS = (
    "file is not a database",
    "database disk image is malformed",
    "attempt to write a readonly database",
    "SELECT",
    "PRAGMA",
    "sqlite3.",
    "Traceback",
    "sqlite_master",
    "schema_migrations",
    "integrity_check",
    "foreign_key_check",
    ".pmem",
    "pmem.db",
)


def _assert_report_is_clean(rendered: str, root: Path) -> None:
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        assert forbidden not in rendered, forbidden
    assert _SENSITIVE_MARKER not in rendered
    assert str(root) not in rendered
    assert str(root.resolve()) not in rendered
    assert "\x1b" not in rendered
    assert not any(ord(char) < 32 and char != "\n" for char in rendered)
    document = json.loads(rendered)
    assert document["schema_version"] == "doctor-v1"
    assert {check["check_id"] for check in document["checks"]} == set(DATABASE_CHECK_IDS)


def test_healthy_report_leaks_nothing(tmp_path: Path) -> None:
    root = _project_with_secret(tmp_path)

    _assert_report_is_clean(_rendered(root), root)


def test_missing_database_report_leaks_nothing(tmp_path: Path) -> None:
    _assert_report_is_clean(_rendered(tmp_path), tmp_path)


def test_corrupt_database_report_leaks_no_sqlite_error(tmp_path: Path) -> None:
    db_path = project_database_path(tmp_path)
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(_SENSITIVE_MARKER.encode("utf-8") * 200)

    _assert_report_is_clean(_rendered(tmp_path), tmp_path)


def test_foreign_key_violation_report_names_no_table_or_row(tmp_path: Path) -> None:
    root = _project_with_secret(tmp_path)
    connection = sqlite3.connect(project_database_path(root))
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute(
        "INSERT INTO experiments"
        "(id, project_id, name, hypothesis, status, created_at, updated_at) "
        "VALUES(?, 'proj_missing', 'orphan', 'h', 'active', "
        "'2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        (f"exp_{_SENSITIVE_MARKER}",),
    )
    connection.commit()
    connection.close()

    rendered = _rendered(root)

    _assert_report_is_clean(rendered, root)
    for table_name in ("experiments", "projects", "runs", "exp_orphan"):
        assert table_name not in rendered, table_name


def test_checksum_mismatch_report_leaks_no_checksum(tmp_path: Path) -> None:
    root = _project_with_secret(tmp_path)
    tampered = "b" * 64
    connection = sqlite3.connect(project_database_path(root))
    connection.execute(
        "UPDATE schema_migrations SET checksum = ? WHERE version = ?",
        (tampered, CURRENT_MIGRATIONS[0].version),
    )
    connection.commit()
    connection.close()

    rendered = _rendered(root)

    _assert_report_is_clean(rendered, root)
    assert tampered not in rendered
    for migration in CURRENT_MIGRATIONS:
        assert migration.checksum not in rendered
        assert migration.version not in rendered


def test_missing_migration_report_leaks_no_version_or_sql(tmp_path: Path) -> None:
    root = _project_with_secret(tmp_path)
    connection = sqlite3.connect(project_database_path(root))
    connection.execute(
        "DELETE FROM schema_migrations WHERE version = ?", (CURRENT_MIGRATIONS[-1].version,)
    )
    connection.commit()
    connection.close()

    rendered = _rendered(root)

    _assert_report_is_clean(rendered, root)
    for migration in CURRENT_MIGRATIONS:
        assert migration.version not in rendered
        assert "CREATE TABLE" not in rendered


def test_sidecar_report_leaks_nothing(tmp_path: Path) -> None:
    root = _project_with_secret(tmp_path)
    db_path = project_database_path(root)
    db_path.with_name(db_path.name + "-wal").write_bytes(_SENSITIVE_MARKER.encode("utf-8"))

    _assert_report_is_clean(_rendered(root), root)


@pytest.mark.parametrize(
    "project_name",
    ["project with spaces", "dự-án-thử", 'quote"name'],
)
def test_unusual_project_names_never_reach_the_database_report(
    tmp_path: Path, project_name: str
) -> None:
    """Database results are hand-written, so project text cannot flow into them."""

    init_project(tmp_path, project_name=project_name, primary_metric="accuracy")

    rendered = _rendered(tmp_path)

    assert project_name not in rendered
    _assert_report_is_clean(rendered, tmp_path)
