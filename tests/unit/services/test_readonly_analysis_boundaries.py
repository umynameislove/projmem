"""Regression tests for public analysis APIs moved to the read-only seam."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pmem.errors import PmemPersistenceError
from pmem.patterns.anomaly import anomaly_detection_payload
from pmem.patterns.config_failure import config_failure_correlation_payload
from pmem.patterns.dataset_failure import dataset_failure_correlation_payload
from pmem.patterns.temporal import temporal_analysis_payload
from pmem.services.project_init import init_project
from pmem.services.recommendation_operations import recommendation_list_payload

ReadOperation = Callable[[Path], dict[str, Any]]

_READ_OPERATIONS: tuple[tuple[str, ReadOperation], ...] = (
    ("anomaly", anomaly_detection_payload),
    ("config_failure", config_failure_correlation_payload),
    ("dataset_failure", dataset_failure_correlation_payload),
    ("temporal", temporal_analysis_payload),
    ("recommendations", recommendation_list_payload),
)


def _snapshot(root: Path) -> dict[str, tuple[bytes, int, int]]:
    return {
        str(path.relative_to(root)): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
            path.stat().st_mode,
        )
        for path in sorted((root / ".pmem").rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(("_name", "operation"), _READ_OPERATIONS, ids=lambda value: str(value))
def test_public_analysis_api_preserves_project_files_and_mode(
    tmp_path: Path, _name: str, operation: ReadOperation
) -> None:
    init_project(tmp_path, project_name=f"readonly-{_name}", primary_metric="accuracy")
    db_path = tmp_path / ".pmem" / "pmem.db"
    db_path.chmod(0o644)
    before = _snapshot(tmp_path)

    payload = operation(tmp_path)

    assert isinstance(payload, dict)
    assert _snapshot(tmp_path) == before
    assert db_path.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(("_name", "operation"), _READ_OPERATIONS, ids=lambda value: str(value))
def test_public_analysis_api_maps_corrupt_database_to_safe_error(
    tmp_path: Path, _name: str, operation: ReadOperation
) -> None:
    init_project(tmp_path, project_name=f"corrupt-{_name}")
    db_path = tmp_path / ".pmem" / "pmem.db"
    db_path.write_bytes(b"not sqlite")
    before = _snapshot(tmp_path)

    with pytest.raises(PmemPersistenceError) as exc_info:
        operation(tmp_path)

    assert _snapshot(tmp_path) == before
    assert "file is not a database" not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)
