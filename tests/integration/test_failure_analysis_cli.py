"""failure embedding and clustering CLI tests for local failure embedding and clustering."""

from __future__ import annotations

import json
import sys

from typer.testing import CliRunner

from pmem.cli.app import app

runner = CliRunner()


def test_failures_embed_json_excludes_raw_text_by_default(monkeypatch, tmp_path) -> None:
    """failure embeddings default CLI output should not expose failure free text."""

    _create_cli_failure(monkeypatch, tmp_path)

    result = runner.invoke(app, ["failures", "embed", "--dimension", "64", "--json"])
    payload = json.loads(result.stdout)
    raw_json = json.dumps(payload, sort_keys=True)

    assert result.exit_code == 0
    assert payload["schema_version"] == "failure-embedding-v1"
    assert payload["privacy_mode"] == "structured_only"
    assert payload["record_count"] == 1
    assert len(payload["records"][0]["vector"]) == 64
    assert "SECRET training text" not in raw_json


def test_failures_embed_include_text_requires_confirm(monkeypatch, tmp_path) -> None:
    """failure embeddings should gate text-derived embeddings behind explicit confirmation."""

    _create_cli_failure(monkeypatch, tmp_path)

    rejected = runner.invoke(app, ["failures", "embed", "--include-text", "--json"])
    accepted = runner.invoke(
        app,
        ["failures", "embed", "--include-text", "--confirm", "--json"],
    )
    payload = json.loads(accepted.stdout)
    raw_json = json.dumps(payload, sort_keys=True)

    assert rejected.exit_code == 1
    assert "requires --confirm" in rejected.stdout
    assert accepted.exit_code == 0
    assert payload["privacy_mode"] == "explicit_text_derived"
    assert "derived_from_free_text" in raw_json
    assert "SECRET training text" not in raw_json


def test_failures_cluster_json_reports_clusters_and_projection(monkeypatch, tmp_path) -> None:
    """failure clustering should produce deterministic cluster and point structures."""

    _create_cli_failure(monkeypatch, tmp_path, error_type="MetricRegression", tag="data quality")
    _create_second_failure(error_type="MetricRegression", tag="data quality")

    result = runner.invoke(
        app,
        [
            "failures",
            "cluster",
            "--include-text",
            "--confirm",
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
    assert payload["schema_version"] == "failure-cluster-v1"
    assert payload["record_count"] == 2
    assert payload["cluster_count"] >= 1
    assert payload["projection"]["method"] == "top_variance_axes"
    assert len(payload["points"]) == 2
    assert "SECRET training text" not in raw_json


def test_failures_cluster_include_text_requires_confirm(monkeypatch, tmp_path) -> None:
    """Cluster analysis must not silently consume raw failure text."""

    _create_cli_failure(monkeypatch, tmp_path)

    result = runner.invoke(app, ["failures", "cluster", "--include-text", "--json"])

    assert result.exit_code == 1
    assert "requires --confirm" in result.stdout
    assert "SECRET training text" not in result.stdout


def test_failures_embed_human_readable_output(monkeypatch, tmp_path) -> None:
    """Embed text output (no --json) covers the console.print path."""

    _create_cli_failure(monkeypatch, tmp_path)

    result = runner.invoke(app, ["failures", "embed", "--dimension", "64"])

    assert result.exit_code == 0
    assert "Failure embeddings:" in result.stdout
    assert "method:" in result.stdout
    assert "dimension:" in result.stdout
    assert "privacy_mode:" in result.stdout
    assert "SECRET training text" not in result.stdout


def test_failures_cluster_human_readable_output(monkeypatch, tmp_path) -> None:
    """Cluster text output (no --json) covers the cluster console.print path."""

    _create_cli_failure(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "failures",
            "cluster",
            "--dimension",
            "64",
            "--threshold",
            "0.1",
        ],
    )

    assert result.exit_code == 0
    assert "Failure clusters:" in result.stdout
    assert "records:" in result.stdout
    assert "threshold:" in result.stdout
    assert "privacy_mode:" in result.stdout
    assert "SECRET training text" not in result.stdout


def test_failures_analysis_help_is_available() -> None:
    """failure embedding and clustering commands should appear in CLI help."""

    failures_help = runner.invoke(app, ["failures", "--help"])
    embed_help = runner.invoke(app, ["failures", "embed", "--help"])
    cluster_help = runner.invoke(app, ["failures", "cluster", "--help"])

    assert failures_help.exit_code == 0
    assert "embed" in failures_help.stdout
    assert "cluster" in failures_help.stdout
    assert embed_help.exit_code == 0
    assert "Compute deterministic local failure embeddings" in embed_help.stdout
    assert cluster_help.exit_code == 0
    assert "Cluster local failure embeddings" in cluster_help.stdout


def _create_cli_failure(
    monkeypatch,
    tmp_path,
    *,
    error_type: str = "MetricRegression",
    tag: str = "data quality",
) -> str:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "analysis-demo"]).exit_code == 0
    run_result = runner.invoke(app, ["run", "--", sys.executable, "-c", "print('ok')"])
    assert run_result.exit_code == 0
    run_id = run_result.stdout.split()[1]
    failure_result = runner.invoke(
        app,
        [
            "log-failure",
            run_id,
            error_type,
            "SECRET training text should not appear.",
            "--severity",
            "high",
            "--source",
            "user_confirmed",
            "--tag",
            tag,
            "--root-cause",
            "Synthetic private root cause",
            "--lesson",
            "Synthetic private lesson",
        ],
    )
    assert failure_result.exit_code == 0
    return run_id


def _create_second_failure(*, error_type: str, tag: str) -> str:
    run_result = runner.invoke(app, ["run", "--", sys.executable, "-c", "print('ok')"])
    assert run_result.exit_code == 0
    run_id = run_result.stdout.split()[1]
    failure_result = runner.invoke(
        app,
        [
            "log-failure",
            run_id,
            error_type,
            "SECRET training text should not appear again.",
            "--severity",
            "high",
            "--source",
            "user_confirmed",
            "--tag",
            tag,
        ],
    )
    assert failure_result.exit_code == 0
    return run_id
