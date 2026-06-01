"""recurring failure detection integration tests."""

from __future__ import annotations

import json

from pmem.patterns.recurring_failures import recurring_failure_report_payload
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_init import init_project

NOW = "2026-05-30T01:00:00Z"


def test_recurring_failure_detection_finds_known_recurring_tag(tmp_path) -> None:
    """recurring failure detection should group recurring failures without leaking raw free text."""

    _seed_recurring_failure_project(tmp_path)

    payload = recurring_failure_report_payload(tmp_path, generated_at=NOW)
    raw_json = json.dumps(payload, sort_keys=True)
    recurring = [cluster for cluster in payload["clusters"] if cluster["recurring"]]
    timeout_cluster = next(
        cluster for cluster in recurring if cluster["dominant_signals"]["tag"].get("timeout") == 5
    )

    assert payload["record_count"] == 10
    assert payload["recurring_cluster_count"] >= 1
    assert timeout_cluster["size"] == 5
    assert len(timeout_cluster["run_ids"]) == 5
    assert timeout_cluster["claim"] == "recurring_failure_candidate_not_root_cause"
    assert "PRIVATE timeout failure" not in raw_json
    assert "PRIVATE root cause" not in raw_json
    assert "PRIVATE lesson" not in raw_json
    assert "caused" not in raw_json.casefold()
    assert payload["algorithm"]["optional_nlp_dependency_required"] is False


def test_recurring_failure_detection_handles_empty_project(tmp_path) -> None:
    """Empty projects should return a clear no-failure report."""

    init_project(tmp_path, project_name="empty-recurring-failures")

    payload = recurring_failure_report_payload(tmp_path, generated_at=NOW)

    assert payload["record_count"] == 0
    assert payload["cluster_count"] == 0
    assert payload["recurring_cluster_count"] == 0
    assert any("No confirmed failure" in warning for warning in payload["warnings"])


def _seed_recurring_failure_project(tmp_path) -> None:
    init_result = init_project(tmp_path, project_name="recurring-failures")
    db_path = project_database_path(tmp_path)
    connection = connect_database(db_path)
    try:
        ExperimentRepository(connection).create(
            experiment_id="exp_d52",
            project_id=init_result.project_id,
            name="d52",
            created_at=NOW,
            updated_at=NOW,
        )
        runs = RunRepository(connection)
        failures = FailureRepository(connection)
        for index in range(5):
            run_id = f"run_timeout_{index}"
            runs.create(
                run_id=run_id,
                experiment_id="exp_d52",
                command="python train.py",
                cwd=".",
                exit_code=1,
                status="failed",
                config={"optimizer": "adam", "fold": index},
                config_hash=f"{index + 1:064x}"[-64:],
                metrics={"accuracy": 0.4 + index * 0.01},
                timestamp=f"2026-05-30T01:00:{index:02d}Z",
            )
            failures.create(
                failure_id=f"failure_timeout_{index}",
                run_id=run_id,
                error_type="TimeoutError",
                description="PRIVATE timeout failure",
                root_cause="PRIVATE root cause",
                lesson="PRIVATE lesson",
                severity="high",
                tags=["timeout"],
                source="user_confirmed",
                created_at=f"2026-05-30T01:01:{index:02d}Z",
            )
        for index in range(5):
            run_id = f"run_other_{index}"
            runs.create(
                run_id=run_id,
                experiment_id="exp_d52",
                command="python train.py",
                cwd=".",
                exit_code=1,
                status="failed",
                config={"optimizer": "sgd", "fold": index},
                config_hash=f"{index + 11:064x}"[-64:],
                metrics={"accuracy": 0.8 + index * 0.01},
                timestamp=f"2026-05-30T01:02:{index:02d}Z",
            )
            failures.create(
                failure_id=f"failure_other_{index}",
                run_id=run_id,
                error_type=f"UniqueError{index}",
                description=f"PRIVATE unrelated failure {index}",
                root_cause="PRIVATE other root cause",
                lesson="PRIVATE other lesson",
                severity="medium",
                tags=[f"unique_{index}"],
                source="user_confirmed",
                created_at=f"2026-05-30T01:03:{index:02d}Z",
            )
    finally:
        connection.close()
