"""recommendation generator integration tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pmem.recommendations import RecommendationType, generate_recommendations
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_init import init_project
from pmem.utils.hashing import compute_text_hash

NOW = datetime(2026, 5, 31, 0, 0, tzinfo=timezone.utc)
NOW_TEXT = "2026-05-31T00:00:00Z"


def test_generates_five_verified_recommendation_types_from_synthetic_project(
    tmp_path,
) -> None:
    """Generate all five recommendation types without fabricated evidence."""

    _seed_d59_project(tmp_path)

    recommendations = generate_recommendations(tmp_path, generated_at=NOW)
    payload = json.dumps([item.model_dump(mode="json") for item in recommendations])

    assert {item.type for item in recommendations} == set(RecommendationType)
    assert len(recommendations) == 5
    assert all(item.supporting_evidence for item in recommendations)
    assert all(
        "based on" in item.description.casefold() or item.type is not RecommendationType.TRY_NEXT
        for item in recommendations
    )
    assert any(
        item.related_failures for item in recommendations if item.type is RecommendationType.AVOID
    )
    assert "PRIVATE" not in payload
    assert "python train.py" not in payload
    assert "root cause" not in payload.casefold()
    assert "SUPPORTS::" not in payload
    assert "CONTRADICTS::" not in payload


def _seed_d59_project(tmp_path) -> None:
    init_result = init_project(
        tmp_path,
        project_name="d59-generator",
        primary_metric="accuracy",
        metric_direction="max",
    )
    connection = connect_database(project_database_path(tmp_path))
    try:
        experiments = ExperimentRepository(connection)
        runs = RunRepository(connection)
        failures = FailureRepository(connection)
        for experiment_id in ("exp_a", "exp_b", "exp_c"):
            experiments.create(
                experiment_id=experiment_id,
                project_id=init_result.project_id,
                name=experiment_id,
                created_at=NOW_TEXT,
                updated_at=NOW_TEXT,
            )
        for index, value in enumerate((0.78, 0.79, 0.80, 0.81, 0.82, 0.79, 0.80, 0.81)):
            _create_run(
                runs,
                run_id=f"run_normal_{index}",
                experiment_id="exp_a",
                config={"family": "normal", "index": index},
                metric=value,
                timestamp=f"2026-05-31T00:00:{index:02d}Z",
            )
        _create_run(
            runs,
            run_id="run_outlier_high",
            experiment_id="exp_a",
            config={"family": "outlier"},
            metric=0.99,
            timestamp="2026-05-31T00:00:59Z",
        )
        for index, value in enumerate((0.45, 0.92, 0.50, 0.95)):
            _create_run(
                runs,
                run_id=f"run_var_{index}",
                experiment_id="exp_b",
                config={"family": "variance"},
                metric=value,
                timestamp=f"2026-05-31T00:01:{index:02d}Z",
            )
        for index in range(5):
            run_id = f"run_bad_{index}"
            _create_run(
                runs,
                run_id=run_id,
                experiment_id="exp_c",
                config={"family": "bad"},
                metric=0.30 + index * 0.01,
                timestamp=f"2026-05-31T00:02:{index:02d}Z",
                status="failed",
                exit_code=1,
            )
            failures.create(
                failure_id=f"failure_bad_{index}",
                run_id=run_id,
                error_type="MetricRegression",
                description="PRIVATE failure description",
                root_cause="PRIVATE root cause",
                lesson="PRIVATE lesson",
                severity="high",
                tags=["metric"],
                source="user_confirmed",
                created_at=f"2026-05-31T00:03:{index:02d}Z",
            )
        _create_run(
            runs,
            run_id="run_promote_c",
            experiment_id="exp_c",
            config={"family": "promote"},
            metric=0.90,
            timestamp="2026-05-31T00:04:00Z",
        )
    finally:
        connection.close()


def _create_run(
    runs: RunRepository,
    *,
    run_id: str,
    experiment_id: str,
    config: dict[str, object],
    metric: float,
    timestamp: str,
    status: str = "success",
    exit_code: int = 0,
) -> None:
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    runs.create(
        run_id=run_id,
        experiment_id=experiment_id,
        command="python train.py",
        cwd=".",
        exit_code=exit_code,
        status=status,
        config=config,
        config_hash=compute_text_hash(config_json),
        metrics={"accuracy": metric},
        timestamp=timestamp,
    )
