"""Tests for the tracked-path repository used by `pmem track`."""

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import apply_migrations
from pmem.repositories.projects import ProjectRepository
from pmem.repositories.sqlite import connect_database
from pmem.repositories.tracked_paths import TrackedPathRepository

NOW = "2026-05-15T00:00:00Z"
HASH = "a" * 64


@pytest.fixture()
def repositories(tmp_path):
    """Return project and tracked-path repositories backed by migrated SQLite."""

    db_path = tmp_path / "pmem.db"
    apply_migrations(db_path)
    connection = connect_database(db_path)
    try:
        project_repository = ProjectRepository(connection)
        tracked_repository = TrackedPathRepository(connection)
        project_repository.create(
            project_id="proj_1",
            name="demo",
            created_at=NOW,
            updated_at=NOW,
        )
        yield tracked_repository
    finally:
        connection.close()


def test_add_and_read_tracked_path(repositories: TrackedPathRepository) -> None:
    """A tracked file should be readable by project and normalized path."""

    record = repositories.add(
        tracked_path_id="track_1",
        project_id="proj_1",
        path="README.md",
        sha256=HASH,
        size_bytes=12,
        last_checked=NOW,
        created_at=NOW,
    )

    assert repositories.get_by_project_and_path("proj_1", "README.md") == record
    assert repositories.list_for_project("proj_1") == (record,)


def test_duplicate_tracked_path_is_rejected(repositories: TrackedPathRepository) -> None:
    """The unique project/path constraint should prevent duplicate tracking rows."""

    repositories.add(
        tracked_path_id="track_1",
        project_id="proj_1",
        path="README.md",
        sha256=HASH,
        size_bytes=12,
        last_checked=NOW,
        created_at=NOW,
    )

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repositories.add(
            tracked_path_id="track_2",
            project_id="proj_1",
            path="README.md",
            sha256=HASH,
            size_bytes=12,
            last_checked=NOW,
            created_at=NOW,
        )

    assert len(repositories.list_for_project("proj_1")) == 1


def test_invalid_hash_is_rejected(repositories: TrackedPathRepository) -> None:
    """DB CHECK constraint should reject non-SHA-256 values."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repositories.add(
            tracked_path_id="track_bad",
            project_id="proj_1",
            path="README.md",
            sha256="not-a-hash",
            size_bytes=12,
            last_checked=NOW,
            created_at=NOW,
        )


def test_orphan_project_id_is_rejected(repositories: TrackedPathRepository) -> None:
    """Tracked paths must reference an existing project."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repositories.add(
            tracked_path_id="track_orphan",
            project_id="missing",
            path="README.md",
            sha256=HASH,
            size_bytes=12,
            last_checked=NOW,
            created_at=NOW,
        )


def test_sql_injection_like_path_is_stored_as_data(
    repositories: TrackedPathRepository,
) -> None:
    """Dangerous-looking paths should not be executed as SQL."""

    dangerous_path = "README.md'; DROP TABLE tracked_paths; --"

    record = repositories.add(
        tracked_path_id="track_injection",
        project_id="proj_1",
        path=dangerous_path,
        sha256=HASH,
        size_bytes=12,
        last_checked=NOW,
        created_at=NOW,
    )

    assert repositories.get_by_project_and_path("proj_1", dangerous_path) == record
