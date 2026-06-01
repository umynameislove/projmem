"""dataset-failure correlation dataset-failure correlation integration tests."""

from __future__ import annotations

import json
import sys

from pmem.patterns.dataset_failure import dataset_failure_correlation_payload
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command
from pmem.utils.hashing import compute_text_hash

NOW = "2026-05-30T00:00:00Z"
HASH = "a" * 64


def test_dataset_failure_correlation_finds_known_synthetic_signal(tmp_path) -> None:
    """Use explicit dataset metadata without leaking raw artifact paths."""

    _seed_dataset_failure_project(tmp_path)

    payload = dataset_failure_correlation_payload(tmp_path, generated_at=NOW)
    raw_json = json.dumps(payload, sort_keys=True)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["dataset"]["dataset_id"] == "bead" and item["dataset"]["version"] == "v_bad"
    )

    assert payload["run_count"] == 12
    assert payload["dataset_metadata_run_count"] == 12
    assert payload["failure_run_count"] == 6
    assert candidate["failure_statistics"]["p_value"] < 0.01
    assert candidate["failure_statistics"]["risk_difference"] == 1.0
    assert candidate["metric_anomaly"]["metric_name"] == "accuracy"
    assert candidate["metric_anomaly"]["score"] > 5.0
    assert candidate["claim"] == "dataset_failure_correlation_observed_not_causal"
    assert "PRIVATE failure description" not in raw_json
    assert "PRIVATE root cause" not in raw_json
    assert "PRIVATE lesson" not in raw_json
    assert "datasets/private/v_bad.csv" not in raw_json
    assert "caused" not in raw_json.casefold()


def test_dataset_failure_correlation_handles_missing_dataset_metadata(tmp_path) -> None:
    """Projects without explicit dataset_id metadata should degrade gracefully."""

    init_result = init_project(tmp_path, project_name="missing-dataset-metadata")
    db_path = project_database_path(tmp_path)
    connection = connect_database(db_path)
    try:
        ExperimentRepository(connection).create(
            experiment_id="exp_d51_empty",
            project_id=init_result.project_id,
            name="d51-empty",
            created_at=NOW,
            updated_at=NOW,
        )
        runs = RunRepository(connection)
        for index in range(10):
            runs.create(
                run_id=f"run_no_dataset_{index}",
                experiment_id="exp_d51_empty",
                command="python train.py",
                cwd=".",
                exit_code=0,
                status="success",
                metrics={"accuracy": 0.8 + index * 0.01},
                artifacts=[{"path": f"datasets/raw_{index}.csv", "sha256": HASH}],
                timestamp=f"2026-05-30T00:00:{index:02d}Z",
            )
    finally:
        connection.close()

    payload = dataset_failure_correlation_payload(tmp_path, generated_at=NOW)

    assert payload["dataset_metadata_run_count"] == 0
    assert payload["candidate_count"] == 0
    assert any("Insufficient dataset metadata" in warning for warning in payload["warnings"])


def test_dataset_failure_correlation_uses_pmem_run_dataset_metadata(tmp_path) -> None:
    """The normal run workflow should feed dataset-failure screening."""

    init_project(tmp_path, project_name="dataset-cli-source", primary_metric="accuracy")
    failure_ids: list[tuple[str, str]] = []
    for index in range(5):
        bad = run_command(
            tmp_path,
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import json; "
                "Path('metrics.json').write_text(json.dumps({'accuracy': 0.10}), encoding='utf-8')",
            ],
            metrics_path="metrics.json",
            dataset_id="fashion_mnist",
            dataset_version="bad_split",
        )
        failure_ids.append((f"failure_dataset_{index}", bad.record.run_id))
        run_command(
            tmp_path,
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import json; "
                "Path('metrics.json').write_text(json.dumps({'accuracy': 0.91}), encoding='utf-8')",
            ],
            metrics_path="metrics.json",
            dataset_id="fashion_mnist",
            dataset_version="good_split",
        )

    connection = connect_database(project_database_path(tmp_path))
    try:
        failures = FailureRepository(connection)
        for index, (failure_id, run_id) in enumerate(failure_ids):
            failures.create(
                failure_id=failure_id,
                run_id=run_id,
                error_type="data_quality",
                description="PRIVATE dataset issue",
                root_cause="PRIVATE root cause",
                lesson="PRIVATE lesson",
                severity="high",
                tags=["dataset"],
                source="user_confirmed",
                created_at=f"2026-05-30T00:10:{index:02d}Z",
            )
    finally:
        connection.close()

    payload = dataset_failure_correlation_payload(tmp_path, generated_at=NOW)
    raw_json = json.dumps(payload, sort_keys=True)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["dataset"]["dataset_id"] == "fashion_mnist"
        and item["dataset"]["version"] == "bad_split"
    )

    assert payload["run_count"] == 10
    assert payload["dataset_metadata_run_count"] == 10
    assert candidate["failure_statistics"]["risk_difference"] == 1.0
    assert "PRIVATE" not in raw_json
    assert "metrics.json" not in raw_json


def _seed_dataset_failure_project(tmp_path) -> None:
    init_result = init_project(tmp_path, project_name="dataset-correlation")
    db_path = project_database_path(tmp_path)
    connection = connect_database(db_path)
    try:
        ExperimentRepository(connection).create(
            experiment_id="exp_d51",
            project_id=init_result.project_id,
            name="d51",
            created_at=NOW,
            updated_at=NOW,
        )
        runs = RunRepository(connection)
        failures = FailureRepository(connection)
        for index in range(6):
            run_id = f"run_bad_dataset_{index}"
            artifacts = [
                {
                    "path": "datasets/private/v_bad.csv",
                    "sha256": compute_text_hash(f"bad-{index}"),
                    "size_bytes": 100 + index,
                    "dataset_id": "bead",
                    "version": "v_bad",
                }
            ]
            runs.create(
                run_id=run_id,
                experiment_id="exp_d51",
                command="python train.py",
                cwd=".",
                exit_code=1,
                status="failed",
                metrics={"accuracy": 0.35 + index * 0.01},
                artifacts=artifacts,
                timestamp=f"2026-05-30T00:01:{index:02d}Z",
            )
            failures.create(
                failure_id=f"failure_bad_dataset_{index}",
                run_id=run_id,
                error_type="DatasetShift",
                description="PRIVATE failure description",
                root_cause="PRIVATE root cause",
                lesson="PRIVATE lesson",
                severity="high",
                tags=["dataset"],
                source="user_confirmed",
                created_at=f"2026-05-30T00:02:{index:02d}Z",
            )
        for index in range(6):
            artifacts = [
                {
                    "path": "datasets/public/v_good.csv",
                    "sha256": compute_text_hash(f"good-{index}"),
                    "size_bytes": 200 + index,
                    "dataset_id": "bead",
                    "version": "v_good",
                }
            ]
            runs.create(
                run_id=f"run_good_dataset_{index}",
                experiment_id="exp_d51",
                command="python train.py",
                cwd=".",
                exit_code=0,
                status="success",
                metrics={"accuracy": 0.91 + index * 0.01},
                artifacts=artifacts,
                timestamp=f"2026-05-30T00:03:{index:02d}Z",
            )
    finally:
        connection.close()
