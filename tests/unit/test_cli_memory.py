"""CLI tests for memory logging memory commands."""

import json
import sys

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database

runner = CliRunner()


def _init_and_run(monkeypatch, tmp_path) -> str:
    """Initialize a temp CLI project and return one run id."""

    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "--name", "demo"])
    run_result = runner.invoke(app, ["run", "--", sys.executable, "-c", "print('ok')"])

    assert init_result.exit_code == 0
    assert run_result.exit_code == 0
    return run_result.stdout.split()[1]


def test_cli_log_failure_success_path(monkeypatch, tmp_path) -> None:
    """`pmem log-failure` should create a confirmed failure for a run."""

    run_id = _init_and_run(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "log-failure",
            run_id,
            "MetricRegression",
            "Accuracy dropped below target.",
            "--severity",
            "high",
            "--tag",
            "Config Error",
            "--root-cause",
            "learning rate too high",
            "--lesson",
            "try smaller lr",
        ],
    )

    assert result.exit_code == 0
    assert "Logged failure fail_" in result.stdout
    assert f"run_id: {run_id}" in result.stdout
    assert "severity: high" in result.stdout
    assert "Traceback" not in result.stdout


def test_cli_log_failure_requires_init(monkeypatch, tmp_path) -> None:
    """Failure logging before init should fail cleanly."""

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["log-failure", "run_1", "ValueError", "bad"],
    )

    assert result.exit_code == 1
    assert "projmem is not initialized. Run `pmem init` first." in result.stdout
    assert "Traceback" not in result.stdout


def test_cli_log_failure_invalid_taxonomy_is_clean(monkeypatch, tmp_path) -> None:
    """Invalid taxonomy values should not expose raw validation internals."""

    run_id = _init_and_run(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["log-failure", run_id, "ValueError", "bad", "--source", "unknown"],
    )

    assert result.exit_code == 1
    assert "Failure source must be" in result.stdout
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()


def test_cli_log_failure_json_output_matches_schema(monkeypatch, tmp_path) -> None:
    """failure JSON should expose a stable JSON payload for log-failure."""

    run_id = _init_and_run(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "log-failure",
            run_id,
            "MetricRegression",
            "Accuracy dropped below target.",
            "--severity",
            "high",
            "--source",
            "user_confirmed",
            "--tag",
            "data quality",
            "--output",
            "json",
        ],
    )

    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert set(payload) == {
        "id",
        "run_id",
        "error_type",
        "severity",
        "tags",
        "source",
        "created_at",
    }
    assert payload["id"].startswith("fail_")
    assert payload["run_id"] == run_id
    assert payload["error_type"] == "MetricRegression"
    assert payload["severity"] == "high"
    assert payload["tags"] == ["data_quality"]
    assert payload["source"] == "user_confirmed"
    assert "description" not in payload
    assert "root_cause" not in payload
    assert "lesson" not in payload


def test_cli_log_failure_invalid_output_format_is_clean(monkeypatch, tmp_path) -> None:
    """Unsupported output modes should fail before DB writes."""

    run_id = _init_and_run(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "log-failure",
            run_id,
            "MetricRegression",
            "Accuracy dropped below target.",
            "--output",
            "xml",
        ],
    )

    assert result.exit_code == 1
    assert "Output format must be text or json." in result.stdout
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()


def test_cli_log_decision_success_path(monkeypatch, tmp_path) -> None:
    """`pmem log-decision` should store project decisions."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "demo"])

    result = runner.invoke(
        app,
        [
            "log-decision",
            "Use logistic regression baseline.",
            "--rationale",
            "fast CPU check",
        ],
    )

    assert result.exit_code == 0
    assert "Logged decision dec_" in result.stdout
    assert "project_id: proj_" in result.stdout


def test_cli_log_decision_prints_experiment_link(monkeypatch, tmp_path) -> None:
    """Decision output should include an experiment id when one is linked."""

    _init_and_run(monkeypatch, tmp_path)
    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        experiment_id = connection.execute("SELECT id FROM experiments").fetchone()[0]
    finally:
        connection.close()

    result = runner.invoke(
        app,
        [
            "log-decision",
            "Keep default experiment.",
            "--experiment-id",
            experiment_id,
        ],
    )

    assert result.exit_code == 0
    assert f"experiment_id: {experiment_id}" in result.stdout


def test_cli_log_decision_missing_experiment_is_clean(monkeypatch, tmp_path) -> None:
    """Invalid decision experiment links should fail without raw internals."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "demo"])

    result = runner.invoke(
        app,
        ["log-decision", "Use baseline.", "--experiment-id", "exp_missing"],
    )

    assert result.exit_code == 1
    assert "Experiment was not found." in result.stdout
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()


def test_cli_note_success_path(monkeypatch, tmp_path) -> None:
    """`pmem note` should store lightweight project notes."""

    run_id = _init_and_run(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["note", "Inspect metric variance.", "--run-id", run_id, "--tag", "Open Question"],
    )

    assert result.exit_code == 0
    assert "Logged note note_" in result.stdout
    assert f"run_id: {run_id}" in result.stdout


def test_cli_note_blank_tag_is_clean(monkeypatch, tmp_path) -> None:
    """Blank note tags should fail through the CLI without raw internals."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "demo"])

    result = runner.invoke(app, ["note", "hello", "--tag", "   "])

    assert result.exit_code == 1
    assert "Note tags cannot be blank." in result.stdout
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()
