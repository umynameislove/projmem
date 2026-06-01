"""Service-level database database workflow tests."""

from pmem.services.database import ensure_database


def test_ensure_database_creates_project_local_db(tmp_path) -> None:
    """Service workflow should create `.pmem/pmem.db` under the project root."""

    result = ensure_database(tmp_path)

    assert result.db_path == tmp_path / ".pmem" / "pmem.db"
    assert result.db_path.exists()
    assert result.applied_versions == ("0001_schema_v1", "0002_phase2_portability")


def test_ensure_database_is_idempotent(tmp_path) -> None:
    """Running the service twice should skip the already applied migration."""

    ensure_database(tmp_path)
    result = ensure_database(tmp_path)

    assert result.applied_versions == ()
    assert result.skipped_versions == ("0001_schema_v1", "0002_phase2_portability")
