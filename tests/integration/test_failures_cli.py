"""failure export CLI tests for privacy-safe failure list/export commands."""

from __future__ import annotations

import json
import sys

from typer.testing import CliRunner

from pmem.cli.app import app

runner = CliRunner()


def test_failures_cli_handles_empty_project(monkeypatch, tmp_path) -> None:
    """Empty projects should produce a valid zero-record JSON payload."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "failure-demo"]).exit_code == 0

    result = runner.invoke(app, ["failures", "list", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["schema_version"] == "failure-export-v1"
    assert payload["record_count"] == 0
    assert payload["records"] == []


def test_failures_list_default_does_not_expose_raw_text(monkeypatch, tmp_path) -> None:
    """Default text output should be useful without printing private failure text."""

    _create_cli_failure(monkeypatch, tmp_path)

    result = runner.invoke(app, ["failures", "list"])

    assert result.exit_code == 0
    assert "Failures: 1" in result.stdout
    assert "privacy_mode: redacted" in result.stdout
    assert "MetricRegression" in result.stdout
    assert "SECRET accuracy dropped" not in result.stdout
    assert "Bad data split" not in result.stdout
    assert "Traceback" not in result.stdout


def test_failures_list_json_default_excludes_raw_text(monkeypatch, tmp_path) -> None:
    """Default JSON is the failure analysis substrate and must not include raw text."""

    _create_cli_failure(monkeypatch, tmp_path)

    result = runner.invoke(app, ["failures", "list", "--json"])
    payload = json.loads(result.stdout)
    record = payload["records"][0]

    assert result.exit_code == 0
    assert payload["privacy_mode"] == "redacted"
    assert payload["include_text"] is False
    assert record["text_included"] is False
    assert record["tags"] == ["data_quality"]
    assert "description" not in record
    assert "root_cause" not in record
    assert "lesson" not in record


def test_failures_include_text_requires_confirm(monkeypatch, tmp_path) -> None:
    """Raw free text should never appear from an accidental flag alone."""

    _create_cli_failure(monkeypatch, tmp_path)

    rejected = runner.invoke(app, ["failures", "list", "--include-text", "--json"])
    accepted = runner.invoke(
        app,
        ["failures", "list", "--include-text", "--confirm", "--json"],
    )
    payload = json.loads(accepted.stdout)

    assert rejected.exit_code == 1
    assert "requires --confirm" in rejected.stdout
    assert "SECRET accuracy dropped" not in rejected.stdout
    assert accepted.exit_code == 0
    assert payload["privacy_mode"] == "explicit_text"
    assert payload["records"][0]["description"] == "SECRET accuracy dropped."


def test_failures_list_text_can_print_confirmed_raw_text(monkeypatch, tmp_path) -> None:
    """Confirmed text output should exercise the failure export text rendering branch."""

    _create_cli_failure(monkeypatch, tmp_path)

    result = runner.invoke(app, ["failures", "list", "--include-text", "--confirm"])

    assert result.exit_code == 0
    assert "description: SECRET accuracy dropped." in result.stdout
    assert "root_cause: Bad data split" in result.stdout
    assert "lesson: Audit split seed" in result.stdout
    assert "Traceback" not in result.stdout


def test_failures_export_writes_file_and_metadata(monkeypatch, tmp_path) -> None:
    """Export writes a reviewable JSON file and can emit parseable result metadata."""

    _create_cli_failure(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["failures", "export", "--out", "exports/failures.json", "--json"],
    )
    result_payload = json.loads(result.stdout)
    export_payload = json.loads((tmp_path / "exports" / "failures.json").read_text("utf-8"))

    assert result.exit_code == 0
    assert result_payload["ok"] is True
    assert result_payload["path"] == "exports/failures.json"
    assert result_payload["record_count"] == 1
    assert export_payload["include_text"] is False
    assert "description" not in export_payload["records"][0]


def test_failures_export_include_text_requires_confirm(monkeypatch, tmp_path) -> None:
    """Text-bearing exports require an explicit confirmation gate."""

    _create_cli_failure(monkeypatch, tmp_path)

    rejected = runner.invoke(
        app,
        ["failures", "export", "--out", "failures.json", "--include-text"],
    )
    accepted = runner.invoke(
        app,
        [
            "failures",
            "export",
            "--out",
            "failures-with-text.json",
            "--include-text",
            "--confirm",
            "--json",
        ],
    )
    payload = json.loads((tmp_path / "failures-with-text.json").read_text("utf-8"))

    assert rejected.exit_code == 1
    assert "requires --confirm" in rejected.stdout
    assert accepted.exit_code == 0
    assert payload["privacy_mode"] == "explicit_text"
    assert payload["records"][0]["description"] == "SECRET accuracy dropped."


def test_failures_export_rejects_unsafe_output_path(monkeypatch, tmp_path) -> None:
    """Export output should not write outside the project or inside `.pmem`."""

    _create_cli_failure(monkeypatch, tmp_path)

    traversal = runner.invoke(app, ["failures", "export", "--out", "../failures.json"])
    inside_pmem = runner.invoke(app, ["failures", "export", "--out", ".PMEM/failures.json"])
    absolute = runner.invoke(app, ["failures", "export", "--out", str(tmp_path / "x.json")])

    assert traversal.exit_code == 1
    assert inside_pmem.exit_code == 1
    assert absolute.exit_code == 1
    assert "Traceback" not in traversal.stdout
    assert "Traceback" not in inside_pmem.stdout
    assert str(tmp_path) not in traversal.stdout


def test_failures_help_is_available() -> None:
    """failure export commands should appear in CLI help."""

    root_help = runner.invoke(app, ["--help"])
    failures_help = runner.invoke(app, ["failures", "--help"])
    list_help = runner.invoke(app, ["failures", "list", "--help"])
    export_help = runner.invoke(app, ["failures", "export", "--help"])

    assert root_help.exit_code == 0
    assert "failures" in root_help.stdout
    assert failures_help.exit_code == 0
    assert "list" in failures_help.stdout
    assert "export" in failures_help.stdout
    assert list_help.exit_code == 0
    assert "List confirmed failures" in list_help.stdout
    assert export_help.exit_code == 0
    assert "Export confirmed failures" in export_help.stdout


def _create_cli_failure(monkeypatch, tmp_path) -> str:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "failure-demo"]).exit_code == 0
    run_result = runner.invoke(app, ["run", "--", sys.executable, "-c", "print('ok')"])
    assert run_result.exit_code == 0
    run_id = run_result.stdout.split()[1]
    failure_result = runner.invoke(
        app,
        [
            "log-failure",
            run_id,
            "MetricRegression",
            "SECRET accuracy dropped.",
            "--severity",
            "high",
            "--source",
            "user_confirmed",
            "--tag",
            "data quality",
            "--root-cause",
            "Bad data split",
            "--lesson",
            "Audit split seed",
        ],
    )
    assert failure_result.exit_code == 0
    return run_id
