"""End-to-end text workflow tests for ``pmem status`` (STS-004)."""

from __future__ import annotations

import json
import sys

import pytest
from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.status import StatusPayload

runner = CliRunner()


def test_status_guides_project_through_empty_sparse_stale_and_mature_states(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("demo\n", encoding="utf-8")

    init_result = runner.invoke(
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
    empty = runner.invoke(app, ["status"])

    assert init_result.exit_code == 0
    _assert_one_action(empty, "capture_first_run", "pmem run --help")
    assert "Graph: missing" in empty.stdout
    assert "Recommendations: not_evaluated" in empty.stdout

    run_id = _run_with_accuracy(tmp_path, 0.91)
    sparse = runner.invoke(app, ["status"])
    _assert_one_action(sparse, "build_evidence_graph", "pmem graph build")

    graph_build = runner.invoke(app, ["graph", "build"])
    needs_baseline = runner.invoke(app, ["status"])
    assert graph_build.exit_code == 0
    _assert_one_action(needs_baseline, "set_baseline", f"pmem baseline {run_id}")
    assert f"Related entity: {run_id}" in needs_baseline.stdout

    baseline = runner.invoke(app, ["baseline", run_id])
    stale_after_baseline = runner.invoke(app, ["status"])
    assert baseline.exit_code == 0
    _assert_one_action(stale_after_baseline, "rebuild_stale_graph", "pmem graph build")
    assert "Graph: stale" in stale_after_baseline.stdout

    assert runner.invoke(app, ["graph", "build"]).exit_code == 0
    needs_tracking = runner.invoke(app, ["status"])
    _assert_one_action(needs_tracking, "track_project_file", "pmem track --help")

    assert runner.invoke(app, ["track", "README.md"]).exit_code == 0
    stale_after_tracking = runner.invoke(app, ["status"])
    _assert_one_action(stale_after_tracking, "rebuild_stale_graph", "pmem graph build")

    assert runner.invoke(app, ["graph", "build"]).exit_code == 0
    mature = runner.invoke(app, ["status"])
    _assert_one_action(mature, "explore_recommendations", "pmem recommend list")
    assert "Target status: met" in mature.stdout
    assert "Warnings\n- none" in mature.stdout
    assert "Safety: database_mutation=false network=false raw_text_in_output=false" in mature.stdout


def test_status_uninitialized_project_has_clean_error_and_creates_nothing(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    assert "projmem is not initialized. Run `pmem init` first." in result.stdout
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()
    assert str(tmp_path) not in result.stdout
    assert not (tmp_path / ".pmem").exists()


def test_status_text_is_deterministic(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "deterministic"]).exit_code == 0

    first = runner.invoke(app, ["status"])
    second = runner.invoke(app, ["status"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.stdout == second.stdout


def test_status_json_agrees_with_text_and_locks_the_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        runner.invoke(
            app,
            [
                "init",
                "--name",
                "json-demo",
                "--objective",
                "Train",
                "--metric",
                "accuracy",
                "--metric-direction",
                "max",
                "--target",
                "0.9",
            ],
        ).exit_code
        == 0
    )

    def _both() -> tuple[dict, str]:
        text = runner.invoke(app, ["status"])
        raw = runner.invoke(app, ["status", "--json"])
        assert text.exit_code == 0
        assert raw.exit_code == 0
        document = json.loads(raw.stdout)
        # the JSON document is a valid, versioned status-v1 payload (contract lock)
        assert StatusPayload.model_validate(document).model_dump(mode="json") == document
        assert document["schema_version"] == "status-v1"
        # text and JSON agree on the single next action
        assert f"Action: {document['next_action']['action_id']}" in text.stdout
        assert f"Command: {document['next_action']['suggested_command']}" in text.stdout
        # a second JSON run is byte-for-byte identical
        assert runner.invoke(app, ["status", "--json"]).stdout == raw.stdout
        return document, raw.stdout

    empty, _ = _both()
    assert empty["counts"]["run_count"] == 0
    assert empty["next_action"]["action_id"] == "capture_first_run"

    run_id = _run_with_accuracy(tmp_path, 0.95)
    runner.invoke(app, ["graph", "build"])
    with_best, _ = _both()
    assert with_best["best_run"]["run_id"] == run_id
    assert with_best["next_action"]["action_id"] == "set_baseline"
    assert with_best["next_action"]["related_entity_id"] == run_id


def test_status_json_uninitialized_error_creates_nothing(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 1
    assert "projmem is not initialized. Run `pmem init` first." in result.stdout
    assert '"schema_version"' not in result.stdout
    assert "Traceback" not in result.stdout
    assert str(tmp_path) not in result.stdout
    assert not (tmp_path / ".pmem").exists()
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_status_json_reports_failed_not_met_met_and_stale_states(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        runner.invoke(
            app,
            [
                "init",
                "--name",
                "json-state-matrix",
                "--metric",
                "accuracy",
                "--metric-direction",
                "max",
                "--target",
                "0.9",
            ],
        ).exit_code
        == 0
    )

    failed = runner.invoke(
        app,
        ["run", "--", sys.executable, "-c", "import sys; sys.exit(2)"],
    )
    assert failed.exit_code == 0
    no_success = json.loads(runner.invoke(app, ["status", "--json"]).stdout)
    assert no_success["metric"]["target_status"] == "no_successful_runs"
    assert no_success["counts"]["successful_run_count"] == 0
    assert no_success["counts"]["failed_run_count"] == 1

    _run_with_accuracy(tmp_path, 0.5)
    not_met = json.loads(runner.invoke(app, ["status", "--json"]).stdout)
    assert not_met["metric"]["target_status"] == "not_met"
    assert not_met["metric"]["best_value"] == 0.5

    assert runner.invoke(app, ["graph", "build"]).exit_code == 0
    best_run_id = _run_with_accuracy(tmp_path, 0.95)
    met_and_stale = json.loads(runner.invoke(app, ["status", "--json"]).stdout)
    assert met_and_stale["metric"]["target_status"] == "met"
    assert met_and_stale["best_run"]["run_id"] == best_run_id
    assert met_and_stale["graph"]["state"] == "stale"
    assert met_and_stale["graph"]["reason_code"] == "graph_source_changed"


def _run_with_accuracy(tmp_path, accuracy: float) -> str:
    script = (
        "from pathlib import Path; import json; "
        f"Path('metrics.json').write_text(json.dumps({{'accuracy': {accuracy}}}), "
        "encoding='utf-8')"
    )
    result = runner.invoke(
        app,
        ["run", "--metrics", "metrics.json", "--", sys.executable, "-c", script],
    )
    assert result.exit_code == 0
    assert (tmp_path / "metrics.json").is_file()
    return result.stdout.split()[1]


def _assert_one_action(result, action_id: str, command: str) -> None:
    assert result.exit_code == 0
    assert result.stdout.count("Next action") == 1
    assert result.stdout.count("Action: ") == 1
    assert f"Action: {action_id}" in result.stdout
    assert f"Command: {command}" in result.stdout
