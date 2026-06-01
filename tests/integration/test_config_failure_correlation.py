"""config-failure correlation integration tests."""

from __future__ import annotations

import json

from pmem.patterns.config_failure import config_failure_correlation_payload
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_init import init_project
from pmem.utils.hashing import compute_text_hash

NOW = "2026-05-29T00:00:00Z"


def test_config_failure_correlation_finds_known_synthetic_signal(tmp_path) -> None:
    """Find a known config/failure association without raw text leakage."""

    _seed_config_failure_project(tmp_path)

    payload = config_failure_correlation_payload(tmp_path, generated_at=NOW)
    raw_json = json.dumps(payload, sort_keys=True)
    optimizer_candidate = next(
        candidate
        for candidate in payload["candidates"]
        if candidate["feature"]["key"] == "optimizer" and candidate["feature"]["value"] == "adam"
    )

    assert payload["run_count"] == 12
    assert payload["failure_run_count"] == 6
    assert payload["non_failure_run_count"] == 6
    assert payload["candidate_count"] >= 1
    assert optimizer_candidate["statistics"]["p_value"] < 0.01
    assert optimizer_candidate["statistics"]["risk_difference"] == 1.0
    assert optimizer_candidate["statistics"]["odds_ratio_ci95"][0] > 1.0
    assert optimizer_candidate["claim"] == "correlation_observed_not_causal"
    assert "PRIVATE failure description" not in raw_json
    assert "PRIVATE root cause" not in raw_json
    assert "PRIVATE lesson" not in raw_json
    assert "private/data.csv" not in raw_json
    assert "caused" not in raw_json.casefold()


def test_config_failure_correlation_handles_empty_project(tmp_path) -> None:
    """Empty projects should produce a clear insufficient-data report."""

    init_project(tmp_path, project_name="empty-config-correlation")

    payload = config_failure_correlation_payload(tmp_path, generated_at=NOW)

    assert payload["run_count"] == 0
    assert payload["candidate_count"] == 0
    assert payload["candidates"] == []
    assert any("Insufficient data" in warning for warning in payload["warnings"])
    assert any("No confirmed failure" in warning for warning in payload["warnings"])


def _seed_config_failure_project(tmp_path) -> None:
    init_result = init_project(tmp_path, project_name="config-correlation")
    db_path = project_database_path(tmp_path)
    connection = connect_database(db_path)
    try:
        ExperimentRepository(connection).create(
            experiment_id="exp_d50",
            project_id=init_result.project_id,
            name="d50",
            created_at=NOW,
            updated_at=NOW,
        )
        runs = RunRepository(connection)
        failures = FailureRepository(connection)
        for index in range(6):
            run_id = f"run_adam_{index}"
            config = {"optimizer": "adam", "lr": 0.001, "dataset_path": "private/data.csv"}
            runs.create(
                run_id=run_id,
                experiment_id="exp_d50",
                command="python train.py",
                cwd=".",
                exit_code=1,
                status="failed",
                config=config,
                config_hash=compute_text_hash(json.dumps(config, sort_keys=True)),
                timestamp=f"2026-05-29T00:00:{index:02d}Z",
            )
            failures.create(
                failure_id=f"failure_adam_{index}",
                run_id=run_id,
                error_type="MetricRegression",
                description="PRIVATE failure description",
                root_cause="PRIVATE root cause",
                lesson="PRIVATE lesson",
                severity="high",
                tags=["config"],
                source="user_confirmed",
                created_at=f"2026-05-29T00:01:{index:02d}Z",
            )
        for index in range(6):
            config = {"optimizer": "sgd", "lr": 0.01, "dataset_path": "public_set"}
            runs.create(
                run_id=f"run_sgd_{index}",
                experiment_id="exp_d50",
                command="python train.py",
                cwd=".",
                exit_code=0,
                status="success",
                config=config,
                config_hash=compute_text_hash(json.dumps(config, sort_keys=True)),
                timestamp=f"2026-05-29T00:02:{index:02d}Z",
            )
    finally:
        connection.close()
