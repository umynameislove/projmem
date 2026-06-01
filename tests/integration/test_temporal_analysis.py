"""temporal metric drift and decision-shift integration tests."""

from __future__ import annotations

import json

from pmem.patterns.temporal import temporal_analysis_payload
from pmem.repositories.decisions import DecisionRepository
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_init import init_project

NOW = "2026-05-30T00:00:00Z"


def test_temporal_analysis_finds_known_midpoint_shift(tmp_path) -> None:
    """temporal analysis should detect synthetic drift without leaking decision text."""

    _seed_temporal_project(tmp_path)

    payload = temporal_analysis_payload(tmp_path, generated_at=NOW)
    raw_json = json.dumps(payload, sort_keys=True)
    drift = payload["drift"]
    decision = payload["decision_impact_candidates"][0]

    assert payload["run_count"] == 12
    assert payload["metric_run_count"] == 12
    assert payload["decision_count"] == 1
    assert drift["metric_name"] == "accuracy"
    assert drift["slope_per_day"] > 0
    assert drift["p_value"] < 0.05
    assert decision["decision_id"] == "decision_d53_midpoint"
    assert decision["decision_scope"] == "experiment"
    assert decision["directional_delta"] > 0
    assert decision["p_value"] < 0.05
    assert decision["claim"] == "decision_metric_shift_candidate_not_causal"
    assert "PRIVATE decision description" not in raw_json
    assert "PRIVATE decision rationale" not in raw_json
    assert "PRIVATE command" not in raw_json
    assert "caused" not in raw_json.casefold()


def test_temporal_analysis_handles_missing_primary_metric(tmp_path) -> None:
    """Projects without primary metric should get a safe explanation."""

    init_project(tmp_path, project_name="d53-no-primary-metric")

    payload = temporal_analysis_payload(tmp_path, generated_at=NOW)

    assert payload["primary_metric"] == ""
    assert payload["drift"] is None
    assert payload["decision_impact_candidates"] == []
    assert any("Primary metric is required" in warning for warning in payload["warnings"])


def _seed_temporal_project(tmp_path) -> None:
    init_result = init_project(
        tmp_path,
        project_name="temporal-analysis",
        primary_metric="accuracy",
        metric_direction="max",
    )
    db_path = project_database_path(tmp_path)
    connection = connect_database(db_path)
    try:
        ExperimentRepository(connection).create(
            experiment_id="exp_d53",
            project_id=init_result.project_id,
            name="d53",
            created_at=NOW,
            updated_at=NOW,
            primary_metric="accuracy",
        )
        runs = RunRepository(connection)
        for index in range(6):
            runs.create(
                run_id=f"run_d53_before_{index}",
                experiment_id="exp_d53",
                command="PRIVATE command before",
                cwd=".",
                exit_code=0,
                status="success",
                metrics={"accuracy": 0.50 + index * 0.01},
                timestamp=f"2026-05-{index + 1:02d}T00:00:00Z",
            )
        DecisionRepository(connection).create(
            decision_id="decision_d53_midpoint",
            project_id=init_result.project_id,
            experiment_id="exp_d53",
            description="PRIVATE decision description",
            rationale="PRIVATE decision rationale",
            created_at="2026-05-07T00:00:00Z",
        )
        for index in range(6):
            runs.create(
                run_id=f"run_d53_after_{index}",
                experiment_id="exp_d53",
                command="PRIVATE command after",
                cwd=".",
                exit_code=0,
                status="success",
                metrics={"accuracy": 0.80 + index * 0.01},
                timestamp=f"2026-05-{index + 7:02d}T00:00:00Z",
            )
    finally:
        connection.close()
