"""recommendation CLI recommendation CLI integration tests."""

from __future__ import annotations

import json
import stat

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_init import init_project
from pmem.utils.hashing import compute_text_hash

runner = CliRunner()
NOW_TEXT = "2026-05-31T00:00:00Z"


def test_recommend_help_exposes_d60_commands() -> None:
    """recommendation CLI commands should be discoverable from root and subcommand help."""

    root_help = runner.invoke(app, ["--help"])
    recommend_help = runner.invoke(app, ["recommend", "--help"])

    assert root_help.exit_code == 0
    assert "recommend" in root_help.stdout
    assert recommend_help.exit_code == 0
    assert "list" in recommend_help.stdout
    assert "run" in recommend_help.stdout
    assert "export" in recommend_help.stdout


def test_recommend_cli_handles_sparse_project_with_helpful_message(monkeypatch, tmp_path) -> None:
    """Sparse projects should not emit fabricated recommendation candidates."""

    monkeypatch.chdir(tmp_path)
    init_project(tmp_path, project_name="d60-empty", primary_metric="accuracy")

    listing = runner.invoke(app, ["recommend", "list", "--json"])
    text_listing = runner.invoke(app, ["recommend", "list"])
    payload = json.loads(listing.stdout)

    assert listing.exit_code == 0
    assert payload["recommendation_count"] == 0
    assert any("Insufficient project evidence" in warning for warning in payload["warnings"])
    assert text_listing.exit_code == 0
    assert "Recommendation candidates" in text_listing.stdout
    assert "- none" in text_listing.stdout


def test_recommend_cli_lists_details_and_exports_without_private_text(
    monkeypatch,
    tmp_path,
) -> None:
    """Expose recommendation candidates through privacy-safe CLI output."""

    monkeypatch.chdir(tmp_path)
    _seed_d59_project(tmp_path)

    listing = runner.invoke(app, ["recommend", "list", "--json"])
    list_payload = json.loads(listing.stdout)
    recommendation_id = list_payload["recommendations"][0]["recommendation_id"]
    detail = runner.invoke(app, ["recommend", "run", recommendation_id, "--json"])
    detail_payload = json.loads(detail.stdout)
    text_detail = runner.invoke(app, ["recommend", "run", recommendation_id])
    export = runner.invoke(
        app,
        ["recommend", "export", "--out", "exports/recommendations.json", "--json"],
    )
    export_payload = json.loads(export.stdout)
    exported_json = (tmp_path / "exports" / "recommendations.json").read_text(encoding="utf-8")
    combined = "\n".join(
        [
            listing.stdout,
            detail.stdout,
            text_detail.stdout,
            export.stdout,
            exported_json,
        ]
    )

    assert listing.exit_code == 0
    assert list_payload["recommendation_count"] == 5
    assert list_payload["basis_counts"]["experiments"] == 3
    assert list_payload["basis_counts"]["runs"] >= 10
    assert list_payload["basis_counts"]["failures"] >= 5
    assert detail.exit_code == 0
    assert detail_payload["recommendation"]["recommendation_id"] == recommendation_id
    assert text_detail.exit_code == 0
    assert recommendation_id in text_detail.stdout
    assert "why:" in text_detail.stdout
    assert "supporting_evidence" not in text_detail.stdout
    assert export.exit_code == 0
    assert export_payload["output_path"] == "exports/recommendations.json"
    assert stat.S_IMODE((tmp_path / "exports" / "recommendations.json").stat().st_mode) == 0o600
    assert "PRIVATE" not in combined
    assert "python train.py" not in combined
    assert "SUPPORTS::" not in combined
    assert "CONTRADICTS::" not in combined
    assert "caused" not in combined.casefold()


def test_recommend_cli_rejects_missing_id_and_unsafe_export_path(
    monkeypatch,
    tmp_path,
) -> None:
    """recommendation CLI should fail closed for unknown ids and unsafe export paths."""

    monkeypatch.chdir(tmp_path)
    _seed_d59_project(tmp_path)

    missing = runner.invoke(app, ["recommend", "run", "rec_missing", "--json"])
    unsafe_export = runner.invoke(app, ["recommend", "export", "--out", "../escape.json"])

    assert missing.exit_code == 1
    assert "Recommendation candidate was not found" in missing.stdout
    assert "Traceback" not in missing.stdout
    assert unsafe_export.exit_code == 1
    assert "cannot contain traversal" in unsafe_export.stdout
    assert "Traceback" not in unsafe_export.stdout


def _seed_d59_project(tmp_path) -> None:
    init_result = init_project(
        tmp_path,
        project_name="d60-recommend-cli",
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
