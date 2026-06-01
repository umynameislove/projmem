"""recommendation generator tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pmem.recommendations import (
    RecommendationConfidence,
    RecommendationType,
    generate_recommendations,
)
from pmem.recommendations.generator import (
    _candidate_run_ids,
    _confidence,
    _metric_sort_key,
    _primary_metric_value,
    _RunEvidence,
    _runs_by_config,
)
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_init import init_project
from pmem.utils.hashing import compute_text_hash

NOW = datetime(2026, 5, 31, 0, 0, tzinfo=timezone.utc)
NOW_TEXT = "2026-05-31T00:00:00Z"


def test_generate_recommendations_emits_all_five_d59_types_with_verified_evidence(
    tmp_path,
) -> None:
    """Synthetic recommendation generator data should produce the locked recommendation types."""

    _seed_recommendation_project(tmp_path)

    recommendations = generate_recommendations(tmp_path, generated_at=NOW)
    payload = json.dumps(
        [item.model_dump(mode="json") for item in recommendations],
        sort_keys=True,
    )
    by_type = {item.type: item for item in recommendations}

    assert tuple(item.type for item in recommendations) == (
        RecommendationType.TRY_NEXT,
        RecommendationType.AVOID,
        RecommendationType.VERIFY,
        RecommendationType.PROMOTE,
        RecommendationType.INVESTIGATE,
    )
    assert by_type[RecommendationType.AVOID].related_failures
    assert by_type[RecommendationType.VERIFY].confidence.value in {"medium", "high"}
    assert len(by_type[RecommendationType.PROMOTE].supporting_evidence) >= 3
    assert (
        by_type[RecommendationType.INVESTIGATE]
        .supporting_evidence[0]
        .entity_id.startswith("run:run_outlier_high")
    )
    assert all(item.generated_at == NOW for item in recommendations)
    assert "PRIVATE" not in payload
    assert "python train.py" not in payload
    assert "SUPPORTS::" not in payload
    assert "CONTRADICTS::" not in payload
    assert "caused" not in payload.casefold()


def test_generate_recommendations_handles_empty_project_gracefully(tmp_path) -> None:
    """Do not fabricate recommendations when project evidence is absent."""

    init_project(tmp_path, project_name="empty-recommendations", primary_metric="accuracy")

    assert generate_recommendations(tmp_path, generated_at=NOW) == ()


def test_generate_recommendations_respects_max_recommendations(tmp_path) -> None:
    """Callers should be able to cap candidate count deterministically."""

    _seed_recommendation_project(tmp_path)

    recommendations = generate_recommendations(
        tmp_path,
        generated_at=NOW,
        max_recommendations=2,
    )

    assert tuple(item.type for item in recommendations) == (
        RecommendationType.TRY_NEXT,
        RecommendationType.AVOID,
    )


def test_generate_recommendations_zero_limit_does_not_require_project(tmp_path) -> None:
    """A zero recommendation cap should short-circuit without touching project state."""

    assert generate_recommendations(tmp_path, max_recommendations=0) == ()


def test_generator_helper_edges_are_deterministic_and_safe() -> None:
    """recommendation generator helper edge cases should stay deterministic for sparse projects."""

    missing_metric = _RunEvidence(
        run_id="run_missing",
        experiment_id="exp",
        status="success",
        timestamp=NOW_TEXT,
        config={},
        config_hash=None,
        primary_metric_value=None,
        failure_ids=(),
    )
    low_metric = _RunEvidence(
        run_id="run_low",
        experiment_id="exp",
        status="success",
        timestamp=NOW_TEXT,
        config={"family": "low"},
        config_hash="cfg",
        primary_metric_value=0.1,
        failure_ids=(),
    )

    assert _primary_metric_value("{}", None) is None
    assert _primary_metric_value("{bad", "accuracy") is None
    assert _primary_metric_value("[]", "accuracy") is None
    assert _primary_metric_value(json.dumps({"accuracy": True}), "accuracy") is None
    assert _primary_metric_value(json.dumps({"accuracy": "0.9"}), "accuracy") is None
    assert _candidate_run_ids({"run_id": "run_1"}) == ("run_1",)
    assert _candidate_run_ids({"evidence": {"run_ids": ["run_1", "run_1", "run_2"]}}) == (
        "run_1",
        "run_2",
    )
    assert _candidate_run_ids({"evidence": "not-a-dict"}) == ()
    assert _runs_by_config((missing_metric, low_metric)) == {"cfg": (low_metric,)}
    assert _metric_sort_key(missing_metric, "max")[0] == float("inf")
    assert _metric_sort_key(low_metric, "min")[0] == 0.1
    assert _confidence(1) is RecommendationConfidence.LOW
    assert _confidence(3) is RecommendationConfidence.MEDIUM
    assert _confidence(6) is RecommendationConfidence.HIGH


def test_feature_level_avoid_surfaces_bad_config_feature_without_exact_config_match(
    tmp_path,
) -> None:
    """Cfile tracking dogfood regression: avoid should catch a bad feature across configs."""

    init_result = init_project(
        tmp_path,
        project_name="feature-level-avoid",
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

        for index, metric in enumerate((0.82, 0.83, 0.81)):
            _create_run(
                runs,
                run_id=f"run_good_{index}",
                experiment_id="exp_a",
                config={"optimizer": "adam", "lr": 0.001, "seed": index},
                metric=metric,
                timestamp=f"2026-05-31T00:10:{index:02d}Z",
            )

        for index, metric in enumerate((0.795, 0.778, 0.808)):
            run_id = f"run_mislabel_lr_01_{index}"
            _create_run(
                runs,
                run_id=run_id,
                experiment_id="exp_b",
                config={"optimizer": "sgd", "lr": 0.1, "seed": index},
                metric=metric,
                timestamp=f"2026-05-31T00:11:{index:02d}Z",
            )
            failures.create(
                failure_id=f"failure_mislabel_{index}",
                run_id=run_id,
                error_type="convergence",
                description="PRIVATE mislabeled failure text",
                root_cause="PRIVATE root cause",
                lesson="PRIVATE lesson",
                severity="medium",
                tags=["convergence"],
                source="user_confirmed",
                created_at=f"2026-05-31T00:12:{index:02d}Z",
            )

        for index, metric in enumerate((0.119, 0.0985, 0.1545, 0.1175)):
            run_id = f"run_bad_lr_1_{index}"
            _create_run(
                runs,
                run_id=run_id,
                experiment_id="exp_c",
                config={"optimizer": "adam" if index % 2 else "sgd", "lr": 1.0, "seed": index},
                metric=metric,
                timestamp=f"2026-05-31T00:13:{index:02d}Z",
                status="failed",
                exit_code=1,
            )
            failures.create(
                failure_id=f"failure_bad_lr_1_{index}",
                run_id=run_id,
                error_type="config_error",
                description="PRIVATE bad lr text",
                root_cause="PRIVATE root cause",
                lesson="PRIVATE lesson",
                severity="high",
                tags=["config"],
                source="user_confirmed",
                created_at=f"2026-05-31T00:14:{index:02d}Z",
            )
    finally:
        connection.close()

    recommendations = generate_recommendations(tmp_path, generated_at=NOW)
    avoid = next(item for item in recommendations if item.type is RecommendationType.AVOID)
    payload = json.dumps(avoid.model_dump(mode="json"), sort_keys=True)

    assert "lr=1" in avoid.title
    assert "lr=0.1" not in payload
    assert "causal proof" in avoid.description
    assert avoid.related_failures
    assert all("run_bad_lr_1" in item.entity_id for item in avoid.supporting_evidence)
    assert "PRIVATE" not in payload


def _seed_recommendation_project(tmp_path) -> None:
    init_result = init_project(
        tmp_path,
        project_name="recommendation-generator",
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
                description="PRIVATE failure text",
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
