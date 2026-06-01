"""project summary summary, timeline/status, and offline performance workflow tests."""

from __future__ import annotations

import json
import sys
from time import perf_counter

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database

runner = CliRunner()


def test_summary_no_runs_target_state(monkeypatch, tmp_path) -> None:
    """summary summary should handle an initialized project with no runs."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        [
            "init",
            "--name",
            "demo",
            "--objective",
            "Train classifier",
            "--metric",
            "accuracy",
            "--metric-direction",
            "max",
            "--target",
            "0.9",
        ],
    )

    result = runner.invoke(app, ["summary"])

    assert result.exit_code == 0
    assert "Project: demo" in result.stdout
    assert "Objective: Train classifier" in result.stdout
    assert "Run count: 0" in result.stdout
    assert "Best run: none" in result.stdout
    assert "Target status: no_runs" in result.stdout
    assert "- run: missing - 0 run(s), 0 successful" in result.stdout


def test_summary_all_failed_runs_target_state(monkeypatch, tmp_path) -> None:
    """summary summary should not promote failed runs as best runs."""

    monkeypatch.chdir(tmp_path)
    _init_target_project()
    failed = runner.invoke(app, ["run", "--", sys.executable, "-c", "import sys; sys.exit(2)"])

    result = runner.invoke(app, ["summary"])

    assert failed.exit_code == 0
    assert "failed" in failed.stdout
    assert result.exit_code == 0
    assert "Run count: 1" in result.stdout
    assert "Best run: none" in result.stdout
    assert "Target status: no_successful_runs" in result.stdout
    assert "- No successful runs captured yet." in result.stdout


def test_summary_target_met_text_and_json(monkeypatch, tmp_path) -> None:
    """summary summary should expose target-met state in text and JSON output."""

    monkeypatch.chdir(tmp_path)
    _init_target_project()
    run_id = _run_with_accuracy(tmp_path, 0.91)

    text = runner.invoke(app, ["summary"])
    json_result = runner.invoke(app, ["summary", "--output", "json"])
    payload = json.loads(json_result.stdout)

    assert text.exit_code == 0
    assert f"Best run: {run_id}" in text.stdout
    assert "Best metric value: 0.91" in text.stdout
    assert "Target status: met" in text.stdout
    assert "- target: met - target status: met" in text.stdout
    assert json_result.exit_code == 0
    assert payload["project_name"] == "demo"
    assert payload["objective"] == "Train classifier"
    assert payload["run_count"] == 1
    assert payload["best_run_id"] == run_id
    assert payload["best_metric_value"] == 0.91
    assert payload["target_status"] == "met"
    assert {item["name"] for item in payload["timeline"]} == {
        "init",
        "track",
        "run",
        "baseline",
        "memory",
        "target",
    }


def test_summary_target_not_met(monkeypatch, tmp_path) -> None:
    """summary summary should report target-not-met when best metric misses target."""

    monkeypatch.chdir(tmp_path)
    _init_target_project()
    _run_with_accuracy(tmp_path, 0.7)

    result = runner.invoke(app, ["summary"])

    assert result.exit_code == 0
    assert "Target status: not_met" in result.stdout
    assert "- Target is not met." in result.stdout


def test_summary_without_target_configuration_reports_not_configured(monkeypatch, tmp_path) -> None:
    """timeline smoke: a successful run without metric config should still summarize cleanly."""

    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "--name", "s-noconf"])
    run_result = runner.invoke(app, ["run", "--", sys.executable, "-c", "print('ok')"])

    result = runner.invoke(app, ["summary"])

    assert init_result.exit_code == 0
    assert run_result.exit_code == 0
    assert result.exit_code == 0
    assert "Target status: not_configured" in result.stdout


def test_summary_success_without_numeric_metric_reports_no_metric(monkeypatch, tmp_path) -> None:
    """summary summary should ignore boolean metrics for target comparison."""

    monkeypatch.chdir(tmp_path)
    _init_target_project()
    script = (
        "from pathlib import Path; import json; "
        "Path('metrics.json').write_text(json.dumps({'accuracy': True}), encoding='utf-8')"
    )
    run_result = runner.invoke(
        app,
        ["run", "--metrics", "metrics.json", "--", sys.executable, "-c", script],
    )

    result = runner.invoke(app, ["summary"])

    assert run_result.exit_code == 0
    assert result.exit_code == 0
    assert "Target status: no_metric" in result.stdout
    assert "- No successful run has metric accuracy." in result.stdout


def test_summary_min_direction_baseline_and_memory_timeline(monkeypatch, tmp_path) -> None:
    """project summary should cover min metrics, baseline, tracked files, and memory rows."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    init_result = runner.invoke(
        app,
        [
            "init",
            "--name",
            "demo",
            "--objective",
            "Reduce loss",
            "--metric",
            "loss",
            "--metric-direction",
            "min",
            "--target",
            "0.5",
        ],
    )
    track_result = runner.invoke(app, ["track", "README.md"])
    run_id = _run_with_metric(tmp_path, "loss", 0.4)
    failure_result = runner.invoke(
        app,
        ["log-failure", run_id, "MetricRegression", "Loss spiked once."],
    )
    decision_result = runner.invoke(
        app,
        ["log-decision", "Keep the low-loss baseline.", "--rationale", "It is reproducible."],
    )
    note_result = runner.invoke(app, ["note", "Review loss curve.", "--run-id", run_id])
    baseline_result = runner.invoke(app, ["baseline", run_id])

    result = runner.invoke(app, ["summary", "--output", "json"])
    payload = json.loads(result.stdout)
    timeline = {item["name"]: item for item in payload["timeline"]}

    assert init_result.exit_code == 0
    assert track_result.exit_code == 0
    assert failure_result.exit_code == 0
    assert decision_result.exit_code == 0
    assert note_result.exit_code == 0
    assert baseline_result.exit_code == 0
    assert result.exit_code == 0
    assert payload["target_status"] == "met"
    assert payload["best_metric_value"] == 0.4
    assert payload["baseline_run_id"] == run_id
    assert payload["warnings"] == []
    assert timeline["track"]["status"] == "done"
    assert timeline["baseline"]["status"] == "done"
    assert timeline["memory"]["status"] == "done"


def test_summary_requires_init_clean_error(monkeypatch, tmp_path) -> None:
    """summary missing-init errors should stay app-level and traceback-free."""

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["summary"])

    assert result.exit_code == 1
    assert "projmem is not initialized. Run `pmem init` first." in result.stdout
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()


def test_core_cli_commands_stay_under_two_seconds_on_cold_db(monkeypatch, tmp_path) -> None:
    """timeline should keep core local-first commands responsive on a cold DB."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")

    init_duration, init_result = _timed_invoke(app, ["init", "--name", "demo"])
    track_duration, track_result = _timed_invoke(app, ["track", "README.md"])
    run_duration, run_result = _timed_invoke(
        app,
        ["run", "--", sys.executable, "-c", "print('ok')"],
    )

    assert init_result.exit_code == 0
    assert track_result.exit_code == 0
    assert run_result.exit_code == 0
    assert init_duration < 2.0
    assert track_duration < 2.0
    assert run_duration < 2.0


def test_summary_workflow_keeps_database_integrity(monkeypatch, tmp_path) -> None:
    """project summary CLI workflow should leave SQLite integrity checks clean."""

    monkeypatch.chdir(tmp_path)
    _init_target_project()
    _run_with_accuracy(tmp_path, 0.91)

    summary = runner.invoke(app, ["summary", "--output", "json"])
    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()

    assert summary.exit_code == 0
    assert integrity == "ok"
    assert foreign_key_rows == []


def _init_target_project() -> None:
    """Create a target-aware project for summary scenarios."""

    result = runner.invoke(
        app,
        [
            "init",
            "--name",
            "demo",
            "--objective",
            "Train classifier",
            "--metric",
            "accuracy",
            "--metric-direction",
            "max",
            "--target",
            "0.9",
        ],
    )
    assert result.exit_code == 0


def _run_with_accuracy(tmp_path, accuracy: float) -> str:
    """Capture a successful run with one accuracy metric."""

    return _run_with_metric(tmp_path, "accuracy", accuracy)


def _run_with_metric(tmp_path, metric: str, value: float) -> str:
    """Capture a successful run with one numeric metric."""

    script = (
        "from pathlib import Path; import json; "
        f"Path('metrics.json').write_text(json.dumps({{{metric!r}: {value}}}), "
        "encoding='utf-8'); print('ok')"
    )
    result = runner.invoke(
        app,
        ["run", "--metrics", "metrics.json", "--", sys.executable, "-c", script],
    )
    assert result.exit_code == 0
    assert "success" in result.stdout
    assert (tmp_path / "metrics.json").is_file()
    return result.stdout.split()[1]


def _timed_invoke(cli_app, args: list[str]):
    """Return elapsed seconds and CLI result for one command invocation."""

    start = perf_counter()
    result = runner.invoke(cli_app, args)
    return perf_counter() - start, result
