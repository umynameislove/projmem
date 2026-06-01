"""failure pattern and summary CLI tests for failure pattern reports and UX summary."""

from __future__ import annotations

import json
import sys

from typer.testing import CliRunner

from pmem.cli.app import app

runner = CliRunner()


def test_failures_patterns_json_reports_heuristic_candidates(monkeypatch, tmp_path) -> None:
    """failure pattern report JSON should expose explainable candidates without raw failure text."""

    _create_cli_failure(monkeypatch, tmp_path, tag="data quality")
    _create_second_failure(tag="data quality")

    result = runner.invoke(
        app,
        [
            "failures",
            "patterns",
            "--dimension",
            "64",
            "--threshold",
            "0.2",
            "--json",
        ],
    )
    payload = json.loads(result.stdout)
    raw_json = json.dumps(payload, sort_keys=True)

    assert result.exit_code == 0
    assert payload["schema_version"] == "failure-pattern-report-v1"
    assert payload["pattern_count"] >= 1
    assert payload["patterns"][0]["heuristic_label"].endswith("pattern candidate")
    assert payload["patterns"][0]["explanation"]
    assert payload["algorithm"]["human_review_required"] is True
    assert "SECRET training text" not in raw_json


def test_failures_patterns_include_text_requires_confirm(monkeypatch, tmp_path) -> None:
    """failure pattern report text-derived signals must be explicitly confirmed."""

    _create_cli_failure(monkeypatch, tmp_path)

    rejected = runner.invoke(app, ["failures", "patterns", "--include-text", "--json"])
    accepted = runner.invoke(
        app,
        ["failures", "patterns", "--include-text", "--confirm", "--json"],
    )
    payload = json.loads(accepted.stdout)
    raw_json = json.dumps(payload, sort_keys=True)

    assert rejected.exit_code == 1
    assert "requires --confirm" in rejected.stdout
    assert "SECRET training text" not in rejected.stdout
    assert accepted.exit_code == 0
    assert payload["privacy_mode"] == "explicit_text_derived"
    assert "derived_from_free_text" in raw_json
    assert "SECRET training text" not in raw_json


def test_failures_patterns_text_output_is_review_oriented(monkeypatch, tmp_path) -> None:
    """failure pattern report text mode should avoid strong causal language."""

    _create_cli_failure(monkeypatch, tmp_path)

    result = runner.invoke(app, ["failures", "patterns", "--dimension", "64"])

    assert result.exit_code == 0
    assert "Failure pattern candidates:" in result.stdout
    assert "human review required" in result.stdout
    assert "pattern candidate" in result.stdout
    assert "root cause" not in result.stdout.lower()
    assert "SECRET training text" not in result.stdout


def test_failures_summary_handles_empty_project(monkeypatch, tmp_path) -> None:
    """failure analysis summary should summarize an initialized project with no failures."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "summary-demo"]).exit_code == 0

    result = runner.invoke(app, ["failures", "summary", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["schema_version"] == "failure-analysis-summary-v1"
    assert payload["status"] == "no_failures"
    assert payload["record_count"] == 0
    assert payload["top_patterns"] == []


def test_failures_summary_json_reports_top_patterns(monkeypatch, tmp_path) -> None:
    """failure analysis summary JSON should expose compact top candidates and next audit actions."""

    _create_cli_failure(monkeypatch, tmp_path, tag="config error")
    _create_second_failure(tag="config error")

    result = runner.invoke(
        app,
        ["failures", "summary", "--dimension", "64", "--threshold", "0.2", "--json"],
    )
    payload = json.loads(result.stdout)
    raw_json = json.dumps(payload, sort_keys=True)

    assert result.exit_code == 0
    assert payload["status"] == "pattern_candidates_available"
    assert payload["top_patterns"]
    assert payload["human_review_required"] is True
    assert "Review top pattern candidates" in payload["next_actions"][0]
    assert "SECRET training text" not in raw_json


def test_failures_summary_text_output_is_readable(monkeypatch, tmp_path) -> None:
    """failure analysis summary text output should be concise and action-oriented."""

    _create_cli_failure(monkeypatch, tmp_path)

    result = runner.invoke(app, ["failures", "summary", "--dimension", "64"])

    assert result.exit_code == 0
    assert "Failure analysis summary" in result.stdout
    assert "Top pattern candidates:" in result.stdout
    assert "Next actions:" in result.stdout
    assert "SECRET training text" not in result.stdout


def test_project_summary_backward_compatibility(monkeypatch, tmp_path) -> None:
    """failure analysis summary must not change the existing `pmem summary` JSON contract."""

    _create_cli_failure(monkeypatch, tmp_path)

    result = runner.invoke(app, ["summary", "--output", "json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert "failure_analysis" not in payload
    assert payload["failure_count"] == 1
    assert {item["name"] for item in payload["timeline"]} == {
        "init",
        "track",
        "run",
        "baseline",
        "memory",
        "target",
    }


def test_failures_patterns_text_output_empty_project(monkeypatch, tmp_path) -> None:
    """L1333-1334: patterns text output with no failures prints '- none'."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "empty-demo"]).exit_code == 0

    result = runner.invoke(app, ["failures", "patterns"])

    assert result.exit_code == 0
    assert "Failure pattern candidates: 0" in result.stdout
    assert "- none" in result.stdout


def test_failures_summary_text_output_empty_project(monkeypatch, tmp_path) -> None:
    """L1360: summary text output with no failures prints '- none' for top_patterns."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "empty-demo"]).exit_code == 0

    result = runner.invoke(app, ["failures", "summary"])

    assert result.exit_code == 0
    assert "Failure analysis summary" in result.stdout
    assert "status: no_failures" in result.stdout
    assert "- none" in result.stdout


def test_failures_patterns_and_summary_help_are_available() -> None:
    """failure pattern and summary commands should appear in CLI help."""

    failures_help = runner.invoke(app, ["failures", "--help"])
    patterns_help = runner.invoke(app, ["failures", "patterns", "--help"])
    summary_help = runner.invoke(app, ["failures", "summary", "--help"])

    assert failures_help.exit_code == 0
    assert "patterns" in failures_help.stdout
    assert "summary" in failures_help.stdout
    assert patterns_help.exit_code == 0
    assert "Generate human-reviewable failure pattern candidates" in patterns_help.stdout
    assert summary_help.exit_code == 0
    assert "Summarize failure analysis status" in summary_help.stdout


def _create_cli_failure(monkeypatch, tmp_path, *, tag: str = "data quality") -> str:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "patterns-demo"]).exit_code == 0
    return _create_second_failure(tag=tag)


def _create_second_failure(*, tag: str = "data quality") -> str:
    run_result = runner.invoke(app, ["run", "--", sys.executable, "-c", "print('ok')"])
    assert run_result.exit_code == 0
    run_id = run_result.stdout.split()[1]
    failure_result = runner.invoke(
        app,
        [
            "log-failure",
            run_id,
            "MetricRegression",
            "SECRET training text should not appear.",
            "--severity",
            "high",
            "--source",
            "user_confirmed",
            "--tag",
            tag,
            "--root-cause",
            "Private root cause",
            "--lesson",
            "Private lesson",
        ],
    )
    assert failure_result.exit_code == 0
    return run_id
