"""Service tests for the `pmem track` workflow."""

import os

import pytest

import pmem.services.tracking as tracking_service
from pmem.errors import PmemNotFoundError, PmemSecurityError, PmemValidationError
from pmem.repositories.sqlite import connect_database
from pmem.services.project_init import init_project
from pmem.services.tracking import track_path, validate_track_path
from pmem.utils.hashing import compute_file_hash


def test_track_path_requires_initialized_project(tmp_path) -> None:
    """Tracking should not create `.pmem/` implicitly."""

    target = tmp_path / "README.md"
    target.write_text("hello\n", encoding="utf-8")

    with pytest.raises(PmemNotFoundError, match="pmem init"):
        track_path(tmp_path, "README.md")

    assert not (tmp_path / ".pmem").exists()


def test_track_file_stores_sha256_and_size(tmp_path) -> None:
    """Tracking a regular file should hash content and persist one DB row."""

    init_project(tmp_path, project_name="demo")
    target = tmp_path / "README.md"
    target.write_text("hello\n", encoding="utf-8")

    result = track_path(tmp_path, "README.md")

    assert result.already_tracked is False
    assert result.path == "README.md"
    assert result.sha256 == compute_file_hash(target)
    assert result.size_bytes == target.stat().st_size

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        row = connection.execute(
            "SELECT path, hash, size_bytes FROM tracked_paths WHERE project_id = ?",
            (result.project_id,),
        ).fetchone()
    finally:
        connection.close()

    assert row["path"] == "README.md"
    assert row["hash"] == result.sha256
    assert row["size_bytes"] == target.stat().st_size


def test_track_duplicate_returns_existing_hash_without_new_row(tmp_path) -> None:
    """file tracking duplicate policy is create-only: report the existing stored hash."""

    init_project(tmp_path, project_name="demo")
    target = tmp_path / "README.md"
    target.write_text("first\n", encoding="utf-8")
    first = track_path(tmp_path, "README.md")
    target.write_text("changed\n", encoding="utf-8")

    second = track_path(tmp_path, "README.md")

    assert second.already_tracked is True
    assert second.sha256 == first.sha256

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        count = connection.execute("SELECT count(*) FROM tracked_paths").fetchone()[0]
    finally:
        connection.close()

    assert count == 1


def test_track_update_refreshes_hash_for_changed_file(tmp_path) -> None:
    """`pmem track --update` should refresh stale hashes without duplicate rows."""

    init_project(tmp_path, project_name="demo")
    target = tmp_path / "README.md"
    target.write_text("first\n", encoding="utf-8")
    first = track_path(tmp_path, "README.md")
    target.write_text("changed\n", encoding="utf-8")

    second = track_path(tmp_path, "README.md", update=True)

    assert second.already_tracked is True
    assert second.updated is True
    assert second.sha256 == compute_file_hash(target)
    assert second.sha256 != first.sha256

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        count = connection.execute("SELECT count(*) FROM tracked_paths").fetchone()[0]
        stored_hash = connection.execute("SELECT hash FROM tracked_paths").fetchone()[0]
    finally:
        connection.close()

    assert count == 1
    assert stored_hash == second.sha256


def test_track_file_inside_nested_directory_normalizes_path(tmp_path) -> None:
    """Normalized paths should be project-relative and deterministic."""

    init_project(tmp_path, project_name="demo")
    nested = tmp_path / "docs"
    nested.mkdir()
    target = nested / "guide.md"
    target.write_text("guide\n", encoding="utf-8")

    result = track_path(tmp_path, "docs/../docs/guide.md")

    assert result.path == "docs/guide.md"


def test_track_sql_injection_like_path_is_safe(tmp_path) -> None:
    """A dangerous-looking filename should be stored as data."""

    init_project(tmp_path, project_name="demo")
    dangerous_name = "README.md'; DROP TABLE tracked_paths; --"
    (tmp_path / dangerous_name).write_text("safe data\n", encoding="utf-8")

    result = track_path(tmp_path, dangerous_name)

    assert result.path == dangerous_name

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        table_exists = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name = ?",
            ("table", "tracked_paths"),
        ).fetchone()
    finally:
        connection.close()

    assert table_exists is not None


def test_validate_track_path_rejects_pmem_internal_files(tmp_path) -> None:
    """`.pmem/` must never become tracked evidence."""

    (tmp_path / ".pmem").mkdir()
    (tmp_path / ".pmem" / "pmem.db").write_text("db", encoding="utf-8")

    with pytest.raises(PmemSecurityError, match="internal files"):
        validate_track_path(tmp_path, ".pmem/pmem.db")
    with pytest.raises(PmemSecurityError, match="internal files"):
        validate_track_path(tmp_path, ".pmem/config.yaml")
    with pytest.raises(PmemSecurityError, match="internal files"):
        validate_track_path(tmp_path, ".PMEM/pmem.db")
    with pytest.raises(PmemSecurityError, match="internal files"):
        validate_track_path(tmp_path, ".PmEm/config.yaml")
    with pytest.raises(PmemSecurityError, match="internal files"):
        validate_track_path(tmp_path, "docs/../.pMeM/secret.txt")


def test_validate_track_path_rejects_missing_directory_absolute_and_traversal(
    tmp_path,
) -> None:
    """file tracking path policy should fail closed for unsafe path shapes."""

    with pytest.raises(PmemNotFoundError, match="does not exist"):
        validate_track_path(tmp_path, "missing.txt")

    (tmp_path / "data").mkdir()
    with pytest.raises(PmemValidationError, match="Directory tracking"):
        validate_track_path(tmp_path, "data")

    with pytest.raises(PmemSecurityError, match="relative"):
        validate_track_path(tmp_path, str(tmp_path / "README.md"))

    with pytest.raises(PmemSecurityError, match="inside"):
        validate_track_path(tmp_path, "../outside.txt")


def test_validate_track_path_rejects_control_oversized_and_symlink(tmp_path) -> None:
    """Malformed and symlink paths should be rejected before hashing."""

    target = tmp_path / "target.txt"
    target.write_text("target\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    os.symlink(target, link)

    with pytest.raises(PmemValidationError, match="control"):
        validate_track_path(tmp_path, "bad\nname.txt")
    with pytest.raises(PmemValidationError, match="too long"):
        validate_track_path(tmp_path, f"{'x' * 513}.txt")
    with pytest.raises(PmemSecurityError, match="Symlink"):
        validate_track_path(tmp_path, "link.txt")


def test_track_error_message_does_not_expose_internals(tmp_path) -> None:
    """Service errors should be safe public messages."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemSecurityError) as exc_info:
        track_path(tmp_path, ".pmem/pmem.db")

    message = str(exc_info.value)
    assert "sqlite" not in message.lower()
    assert "SELECT" not in message
    assert str(tmp_path) not in message


def test_track_fails_cleanly_when_config_project_row_is_missing(tmp_path) -> None:
    """Tracking should reject inconsistent config/database state."""

    init_project(tmp_path, project_name="demo")
    target = tmp_path / "README.md"
    target.write_text("hello\n", encoding="utf-8")
    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        connection.execute("DELETE FROM projects")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PmemNotFoundError, match="pmem init"):
        track_path(tmp_path, "README.md")


def test_track_hash_read_error_is_mapped_to_safe_validation_error(
    monkeypatch,
    tmp_path,
) -> None:
    """Filesystem read failures should not leak raw OS details."""

    init_project(tmp_path, project_name="demo")
    target = tmp_path / "README.md"
    target.write_text("hello\n", encoding="utf-8")

    def fail_hash(_path):
        raise OSError("raw absolute path leak")

    monkeypatch.setattr(tracking_service, "compute_file_hash", fail_hash)

    with pytest.raises(PmemValidationError) as exc_info:
        track_path(tmp_path, "README.md")

    assert str(exc_info.value) == "Tracked path could not be read."
    assert "raw absolute path leak" not in str(exc_info.value)


def test_validate_track_path_rejects_blank_path(tmp_path) -> None:
    """Blank path input should fail before filesystem access."""

    with pytest.raises(PmemValidationError, match="blank"):
        validate_track_path(tmp_path, " ")


def test_validate_track_path_accepts_dot_component_without_changing_output(tmp_path) -> None:
    """Safe normalization should handle harmless `./` components."""

    target = tmp_path / "README.md"
    target.write_text("hello\n", encoding="utf-8")

    result = validate_track_path(tmp_path, "./README.md")

    assert result.relative_path == "README.md"


def test_validate_track_path_rejects_non_regular_file(tmp_path) -> None:
    """file tracking should track regular files only."""

    fifo_path = tmp_path / "fifo"
    os.mkfifo(fifo_path)

    with pytest.raises(PmemValidationError, match="regular files"):
        validate_track_path(tmp_path, "fifo")
