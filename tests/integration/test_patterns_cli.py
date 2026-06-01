"""pattern CLI pattern detection CLI integration tests."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.decisions import DecisionRepository
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_init import init_project
from pmem.utils.hashing import compute_text_hash

runner = CliRunner()
NOW = "2026-05-30T02:00:00Z"


def test_patterns_help_exposes_d55_commands() -> None:
    """pattern CLI commands should be discoverable from root and subcommand help."""

    root_help = runner.invoke(app, ["--help"])
    patterns_help = runner.invoke(app, ["patterns", "--help"])

    assert root_help.exit_code == 0
    assert "patterns" in root_help.stdout
    assert patterns_help.exit_code == 0
    assert "list" in patterns_help.stdout
    assert "config-failure" in patterns_help.stdout
    assert "dataset-failure" in patterns_help.stdout
    assert "recurring-failures" in patterns_help.stdout
    assert "temporal" in patterns_help.stdout
    assert "anomalies" in patterns_help.stdout


def test_patterns_cli_handles_sparse_project_with_helpful_warnings(monkeypatch, tmp_path) -> None:
    """Sparse projects should return safe insufficient-data reports."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "d55-empty"]).exit_code == 0

    listing = runner.invoke(app, ["patterns", "list", "--json"])
    config = runner.invoke(app, ["patterns", "config-failure", "--json"])
    temporal = runner.invoke(app, ["patterns", "temporal", "--json"])
    anomalies = runner.invoke(app, ["patterns", "anomalies", "--json"])

    listing_payload = json.loads(listing.stdout)
    config_payload = json.loads(config.stdout)
    temporal_payload = json.loads(temporal.stdout)
    anomalies_payload = json.loads(anomalies.stdout)

    assert listing.exit_code == 0
    assert listing_payload["schema_version"] == "pattern-list-result-v1"
    assert listing_payload["candidate_count"] == 0
    assert any("Insufficient data" in warning for warning in listing_payload["warnings"])
    assert config.exit_code == 0
    assert config_payload["summary"]["candidate_count"] == 0
    assert temporal.exit_code == 0
    assert temporal_payload["summary"]["candidate_count"] == 0
    assert anomalies.exit_code == 0
    assert anomalies_payload["summary"]["candidate_count"] == 0


def test_patterns_cli_reports_known_patterns_without_private_text(monkeypatch, tmp_path) -> None:
    """pattern CLI should surface known pattern-analysis signals without leaking raw inputs."""

    monkeypatch.chdir(tmp_path)
    _seed_pattern_project(tmp_path)

    listing = runner.invoke(app, ["patterns", "list", "--json"])
    config = runner.invoke(app, ["patterns", "config-failure", "--json"])
    dataset = runner.invoke(app, ["patterns", "dataset-failure", "--json"])
    recurring = runner.invoke(app, ["patterns", "recurring-failures", "--json"])
    temporal = runner.invoke(app, ["patterns", "temporal", "--json"])
    anomalies = runner.invoke(app, ["patterns", "anomalies", "--json"])
    text_listing = runner.invoke(app, ["patterns", "list"])
    text_config = runner.invoke(app, ["patterns", "config-failure"])
    text_dataset = runner.invoke(app, ["patterns", "dataset-failure"])
    text_recurring = runner.invoke(app, ["patterns", "recurring-failures"])
    text_temporal = runner.invoke(app, ["patterns", "temporal"])
    text_anomalies = runner.invoke(app, ["patterns", "anomalies"])
    combined = "\n".join(
        [
            listing.stdout,
            config.stdout,
            dataset.stdout,
            recurring.stdout,
            temporal.stdout,
            anomalies.stdout,
            text_listing.stdout,
            text_config.stdout,
            text_dataset.stdout,
            text_recurring.stdout,
            text_temporal.stdout,
            text_anomalies.stdout,
        ]
    )
    listing_payload = json.loads(listing.stdout)
    config_payload = json.loads(config.stdout)
    dataset_payload = json.loads(dataset.stdout)
    recurring_payload = json.loads(recurring.stdout)
    temporal_payload = json.loads(temporal.stdout)
    anomalies_payload = json.loads(anomalies.stdout)

    assert listing.exit_code == 0
    assert listing_payload["candidate_count"] >= 4
    assert config.exit_code == 0
    assert config_payload["summary"]["candidate_count"] >= 1
    assert dataset.exit_code == 0
    assert dataset_payload["summary"]["candidate_count"] >= 1
    assert recurring.exit_code == 0
    assert recurring_payload["summary"]["candidate_count"] >= 1
    assert temporal.exit_code == 0
    assert temporal_payload["summary"]["candidate_count"] >= 1
    assert anomalies.exit_code == 0
    assert anomalies_payload["summary"]["candidate_count"] >= 1
    assert text_listing.exit_code == 0
    assert "Pattern reports" in text_listing.stdout
    assert text_config.exit_code == 0
    assert "Pattern: config_failure" in text_config.stdout
    assert text_dataset.exit_code == 0
    assert "Pattern: dataset_failure" in text_dataset.stdout
    assert text_recurring.exit_code == 0
    assert "Pattern: recurring_failures" in text_recurring.stdout
    assert text_temporal.exit_code == 0
    assert "Pattern: temporal" in text_temporal.stdout
    assert text_anomalies.exit_code == 0
    assert "Pattern: anomalies" in text_anomalies.stdout
    assert "PRIVATE" not in combined
    assert "private/data.csv" not in combined
    assert "private/config.yaml" not in combined
    assert "caused" not in combined.casefold()


def test_patterns_recurring_include_text_requires_confirm(monkeypatch, tmp_path) -> None:
    """Raw failure text analysis must keep the failure export confirmation gate."""

    monkeypatch.chdir(tmp_path)
    _seed_pattern_project(tmp_path)

    result = runner.invoke(app, ["patterns", "recurring-failures", "--include-text", "--json"])

    assert result.exit_code == 1
    assert "requires --confirm" in result.stdout
    assert "Traceback" not in result.stdout


def _seed_pattern_project(tmp_path) -> None:
    init_result = init_project(
        tmp_path,
        project_name="d55-patterns",
        primary_metric="accuracy",
        metric_direction="max",
    )
    db_path = project_database_path(tmp_path)
    connection = connect_database(db_path)
    try:
        experiments = ExperimentRepository(connection)
        experiments.create(
            experiment_id="exp_d55_main",
            project_id=init_result.project_id,
            name="d55-main",
            created_at=NOW,
            updated_at=NOW,
            primary_metric="accuracy",
        )
        experiments.create(
            experiment_id="exp_d55_anomaly",
            project_id=init_result.project_id,
            name="d55-anomaly",
            created_at=NOW,
            updated_at=NOW,
            primary_metric="accuracy",
        )
        runs = RunRepository(connection)
        failures = FailureRepository(connection)
        for index in range(6):
            _create_run(
                runs,
                run_id=f"run_d55_adam_{index}",
                experiment_id="exp_d55_main",
                config={"optimizer": "adam", "dataset_path": "private/data.csv"},
                metrics={"accuracy": 0.50 + index * 0.01},
                artifacts=[{"dataset_id": "badset", "version": "v1"}],
                status="failed",
                exit_code=1,
                timestamp=f"2026-05-{index + 1:02d}T00:00:00Z",
            )
            failures.create(
                failure_id=f"failure_d55_timeout_{index}",
                run_id=f"run_d55_adam_{index}",
                error_type="TimeoutError",
                description="PRIVATE timeout failure",
                root_cause="PRIVATE root cause",
                lesson="PRIVATE lesson",
                severity="high",
                tags=["timeout"],
                source="user_confirmed",
                created_at=f"2026-05-{index + 1:02d}T00:10:00Z",
            )
        DecisionRepository(connection).create(
            decision_id="decision_d55_midpoint",
            project_id=init_result.project_id,
            experiment_id="exp_d55_main",
            description="PRIVATE decision description",
            rationale="PRIVATE decision rationale",
            created_at="2026-05-07T00:00:00Z",
        )
        for index in range(6):
            _create_run(
                runs,
                run_id=f"run_d55_sgd_{index}",
                experiment_id="exp_d55_main",
                config={"optimizer": "sgd", "dataset_path": "public_set"},
                metrics={"accuracy": 0.80 + index * 0.01},
                artifacts=[{"dataset_id": "goodset", "version": "v1"}],
                status="success",
                exit_code=0,
                timestamp=f"2026-05-{index + 7:02d}T00:00:00Z",
            )
        for index in range(8):
            _create_run(
                runs,
                run_id=f"run_d55_normal_{index}",
                experiment_id="exp_d55_anomaly",
                config={"source": "private/config.yaml", "run": index},
                metrics={"accuracy": 0.70 + index * 0.01},
                artifacts=[],
                status="success",
                exit_code=0,
                timestamp=f"2026-06-{index + 1:02d}T00:00:00Z",
            )
        _create_run(
            runs,
            run_id="run_d55_outlier_high",
            experiment_id="exp_d55_anomaly",
            config={"source": "private/config.yaml", "run": "outlier"},
            metrics={"accuracy": 1.50},
            artifacts=[],
            status="success",
            exit_code=0,
            timestamp="2026-06-09T00:00:00Z",
        )
        shared_config: dict[str, object] = {"model": "same", "source": "private/config.yaml"}
        for index, value in enumerate((0.40, 0.95, 0.42, 0.90)):
            _create_run(
                runs,
                run_id=f"run_d55_repro_{index}",
                experiment_id="exp_d55_anomaly",
                config=shared_config,
                metrics={"accuracy": value},
                artifacts=[],
                status="success",
                exit_code=0,
                timestamp=f"2026-06-{index + 10:02d}T00:00:00Z",
            )
    finally:
        connection.close()


def _create_run(
    runs: RunRepository,
    *,
    run_id: str,
    experiment_id: str,
    config: dict[str, object],
    metrics: dict[str, object],
    artifacts: list[dict[str, object]],
    status: str,
    exit_code: int,
    timestamp: str,
) -> None:
    runs.create(
        run_id=run_id,
        experiment_id=experiment_id,
        command="PRIVATE training command",
        cwd=".",
        exit_code=exit_code,
        status=status,
        config=config,
        config_hash=compute_text_hash(json.dumps(config, sort_keys=True)),
        metrics=metrics,
        artifacts=artifacts,
        timestamp=timestamp,
    )
