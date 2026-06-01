"""Service tests for `pmem log-failure`."""

import json
import sys

import pytest

from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.repositories.sqlite import connect_database
from pmem.services.failure_logging import log_failure
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command


def _create_run(tmp_path) -> str:
    """Initialize a temp project and return one captured run id."""

    init_project(tmp_path, project_name="demo")
    result = run_command(
        tmp_path,
        [sys.executable, "-c", "import sys; print('bad'); sys.exit(2)"],
    )
    return result.record.run_id


def test_log_failure_persists_confirmed_failure(tmp_path) -> None:
    """failure logging should store one validated failure for an existing run."""

    run_id = _create_run(tmp_path)
    record = log_failure(
        tmp_path,
        run_id=run_id,
        error_type="MetricRegression",
        description="Accuracy dropped below target.",
        root_cause="Learning rate too high",
        lesson="Try a smaller learning rate",
        severity="high",
        tags=("Config Error", "convergence"),
        source="user_confirmed",
    )

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        row = connection.execute(
            "SELECT run_id, error_type, severity, tags_json, source FROM failures WHERE id = ?",
            (record.id,),
        ).fetchone()
    finally:
        connection.close()

    assert row["run_id"] == run_id
    assert row["error_type"] == "MetricRegression"
    assert row["severity"] == "high"
    assert json.loads(row["tags_json"]) == ["config_error", "convergence"]
    assert row["source"] == "user_confirmed"


def test_log_failure_requires_initialized_project(tmp_path) -> None:
    """failure logging must not create implicit project state."""

    with pytest.raises(PmemNotFoundError, match="pmem init"):
        log_failure(
            tmp_path,
            run_id="run_1",
            error_type="ValueError",
            description="bad",
        )


def test_log_failure_rejects_invalid_taxonomy(tmp_path) -> None:
    """Invalid severity/source values should fail before persistence."""

    run_id = _create_run(tmp_path)

    with pytest.raises(PmemValidationError, match="severity"):
        log_failure(
            tmp_path,
            run_id=run_id,
            error_type="ValueError",
            description="bad",
            severity="urgent",
        )


def test_log_failure_rejects_invalid_source(tmp_path) -> None:
    """Invalid source values should fail before persistence."""

    run_id = _create_run(tmp_path)

    with pytest.raises(PmemValidationError, match="source"):
        log_failure(
            tmp_path,
            run_id=run_id,
            error_type="ValueError",
            description="bad",
            source="unknown",
        )


def test_log_failure_rejects_blank_tag(tmp_path) -> None:
    """Blank failure tags should not create useless JSON search keys."""

    run_id = _create_run(tmp_path)

    with pytest.raises(PmemValidationError, match="tags cannot be blank"):
        log_failure(
            tmp_path,
            run_id=run_id,
            error_type="ValueError",
            description="bad",
            tags=("   ",),
        )


def test_log_failure_rejects_missing_run(tmp_path) -> None:
    """Failure logging should fail cleanly when run_id does not exist."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemNotFoundError, match="Run was not found"):
        log_failure(
            tmp_path,
            run_id="run_missing",
            error_type="ValueError",
            description="bad",
        )


def test_log_failure_rejects_blank_optional_text(tmp_path) -> None:
    """Optional free-text fields cannot be whitespace-only."""

    run_id = _create_run(tmp_path)

    with pytest.raises(PmemValidationError, match="cannot be blank"):
        log_failure(
            tmp_path,
            run_id=run_id,
            error_type="ValueError",
            description="bad",
            root_cause="   ",
        )
