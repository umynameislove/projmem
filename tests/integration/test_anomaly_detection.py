"""anomaly detection anomaly detection integration tests."""

from __future__ import annotations

import json

from pmem.patterns.anomaly import anomaly_detection_payload
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_init import init_project
from pmem.utils.hashing import compute_text_hash

NOW = "2026-05-30T00:00:00Z"


def test_anomaly_detection_finds_outlier_and_unreproducible_config(tmp_path) -> None:
    """anomaly detection should detect synthetic anomaly signals without leaking raw config."""

    _seed_anomaly_project(tmp_path)

    payload = anomaly_detection_payload(tmp_path, generated_at=NOW)
    raw_json = json.dumps(payload, sort_keys=True)
    outlier = payload["metric_outliers"][0]
    repro = payload["reproducibility_candidates"][0]

    assert payload["run_count"] == 13
    assert payload["metric_point_count"] == 13
    assert outlier["run_id"] == "run_d54_outlier_high"
    assert outlier["direction"] == "high"
    assert repro["sample_size"] == 4
    assert repro["range"] >= 0.10
    assert repro["standard_deviation"] >= 0.05
    assert "PRIVATE command" not in raw_json
    assert "private/config.yaml" not in raw_json
    assert "caused" not in raw_json.casefold()


def test_anomaly_detection_handles_sparse_project(tmp_path) -> None:
    """Empty projects should get explicit insufficient-data warnings."""

    init_project(tmp_path, project_name="d54-empty", primary_metric="accuracy")

    payload = anomaly_detection_payload(tmp_path, generated_at=NOW)

    assert payload["run_count"] == 0
    assert payload["metric_outlier_count"] == 0
    assert payload["reproducibility_candidate_count"] == 0
    assert any("Insufficient data" in warning for warning in payload["warnings"])
    assert any("No finite numeric metric" in warning for warning in payload["warnings"])


def _seed_anomaly_project(tmp_path) -> None:
    init_result = init_project(
        tmp_path,
        project_name="anomaly-detection",
        primary_metric="accuracy",
        metric_direction="max",
    )
    db_path = project_database_path(tmp_path)
    connection = connect_database(db_path)
    try:
        experiments = ExperimentRepository(connection)
        experiments.create(
            experiment_id="exp_d54_outlier",
            project_id=init_result.project_id,
            name="d54-outlier",
            created_at=NOW,
            updated_at=NOW,
            primary_metric="accuracy",
        )
        experiments.create(
            experiment_id="exp_d54_repro",
            project_id=init_result.project_id,
            name="d54-repro",
            created_at=NOW,
            updated_at=NOW,
            primary_metric="accuracy",
        )
        runs = RunRepository(connection)
        for index in range(8):
            config = {"source": "private/config.yaml", "run": index}
            runs.create(
                run_id=f"run_d54_normal_{index}",
                experiment_id="exp_d54_outlier",
                command="PRIVATE command normal",
                cwd=".",
                exit_code=0,
                status="success",
                config=config,
                config_hash=compute_text_hash(json.dumps(config, sort_keys=True)),
                metrics={"accuracy": 0.80 + index * 0.01},
                timestamp=f"2026-05-{index + 1:02d}T00:00:00Z",
            )
        outlier_config = {"source": "private/config.yaml", "run": "outlier"}
        runs.create(
            run_id="run_d54_outlier_high",
            experiment_id="exp_d54_outlier",
            command="PRIVATE command outlier",
            cwd=".",
            exit_code=0,
            status="success",
            config=outlier_config,
            config_hash=compute_text_hash(json.dumps(outlier_config, sort_keys=True)),
            metrics={"accuracy": 1.50},
            timestamp="2026-05-09T00:00:00Z",
        )
        shared_config = {"model": "same", "source": "private/config.yaml"}
        shared_hash = compute_text_hash(json.dumps(shared_config, sort_keys=True))
        for index, value in enumerate((0.40, 0.95, 0.42, 0.90)):
            runs.create(
                run_id=f"run_d54_repro_{index}",
                experiment_id="exp_d54_repro",
                command="PRIVATE command repro",
                cwd=".",
                exit_code=0,
                status="success",
                config=shared_config,
                config_hash=shared_hash,
                metrics={"accuracy": value},
                timestamp=f"2026-06-{index + 1:02d}T00:00:00Z",
            )
    finally:
        connection.close()
