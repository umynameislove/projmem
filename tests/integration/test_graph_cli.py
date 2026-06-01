"""graph CLI graph CLI integration tests."""

from __future__ import annotations

import json
import stat
import sys

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.graph.persistence import default_graph_artifact_path
from pmem.graph.schema import EdgeType, run_node_id
from pmem.services.decision_logging import log_decision
from pmem.services.failure_logging import log_failure
from pmem.services.note_logging import add_note
from pmem.services.project_init import init_project
from pmem.services.run_capture import RunCaptureResult, run_command

runner = CliRunner()


def test_graph_help_exposes_d47_commands() -> None:
    root_help = runner.invoke(app, ["--help"])
    graph_help = runner.invoke(app, ["graph", "--help"])

    assert root_help.exit_code == 0
    assert "graph" in root_help.stdout
    assert graph_help.exit_code == 0
    assert "build" in graph_help.stdout
    assert "status" in graph_help.stdout
    assert "query" in graph_help.stdout
    assert "lineage" in graph_help.stdout
    assert "export" in graph_help.stdout


def test_graph_status_is_graceful_before_build(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "graph-status-demo"]).exit_code == 0

    result = runner.invoke(app, ["graph", "status", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["exists"] is False
    assert "pmem graph build" in payload["message"]


def test_graph_cli_build_query_and_export_are_privacy_safe(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    run_result = _seed_project(tmp_path)
    run_node = run_node_id(run_result.record.run_id)

    build = runner.invoke(app, ["graph", "build", "--json"])
    build_payload = json.loads(build.stdout)
    status = runner.invoke(app, ["graph", "status", "--json"])
    status_payload = json.loads(status.stdout)
    query = runner.invoke(
        app,
        [
            "graph",
            "query",
            "--node",
            run_node,
            "--edge-type",
            EdgeType.BELONGS_TO.value,
            "--depth",
            "1",
            "--path-to",
            run_node,
            "--json",
        ],
    )
    query_payload = json.loads(query.stdout)
    lineage = runner.invoke(
        app,
        ["graph", "lineage", "--run-id", run_result.record.run_id, "--json"],
    )
    lineage_payload = json.loads(lineage.stdout)
    rejected_export = runner.invoke(app, ["graph", "export", "--out", "graph-export.json"])
    export = runner.invoke(
        app,
        ["graph", "export", "--out", "exports/graph.json", "--confirm", "--json"],
    )
    export_payload = json.loads(export.stdout)
    graph_path = default_graph_artifact_path(tmp_path)
    exported_json = (tmp_path / "exports" / "graph.json").read_text(encoding="utf-8")
    combined = json.dumps(
        {
            "build": build_payload,
            "status": status_payload,
            "query": query_payload,
            "lineage": lineage_payload,
            "export": export_payload,
        },
        sort_keys=True,
    )

    assert build.exit_code == 0
    assert build_payload["ok"] is True
    assert build_payload["counts"]["nodes"] >= 1
    assert build_payload["counts"]["edges"] >= 1
    assert status.exit_code == 0
    assert status_payload["exists"] is True
    assert query.exit_code == 0
    assert query_payload["found"] is True
    assert query_payload["node"] == {"id": run_node, "type": "run"}
    assert query_payload["path"]["found"] is True
    assert query_payload["subgraph"]["counts"]["nodes"] >= 1
    assert lineage.exit_code == 0
    assert lineage_payload["schema_version"] == "graph-lineage-result-v1"
    assert lineage_payload["lineage"]["counts"]["hops"] >= 1
    assert rejected_export.exit_code == 1
    assert "requires --confirm" in rejected_export.stdout
    assert export.exit_code == 0
    assert export_payload["output_path"] == "exports/graph.json"
    assert stat.S_IMODE(graph_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "exports" / "graph.json").stat().st_mode) == 0o600
    assert "PRIVATE" not in combined
    assert "PRIVATE" not in exported_json
    assert "artifact.txt" not in combined
    assert "README.md" not in combined
    assert "python -c" not in combined
    assert "attributes" not in json.dumps(query_payload, sort_keys=True)


def test_graph_cli_text_output_is_minimal_and_readable(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    run_result = _seed_project(tmp_path)
    run_node = run_node_id(run_result.record.run_id)

    build = runner.invoke(app, ["graph", "build"])
    status = runner.invoke(app, ["graph", "status"])
    query = runner.invoke(
        app,
        [
            "graph",
            "query",
            "--node",
            run_node,
            "--depth",
            "1",
            "--path-to",
            run_node,
        ],
    )
    lineage = runner.invoke(app, ["graph", "lineage", "--run-id", run_result.record.run_id])
    export = runner.invoke(app, ["graph", "export", "--out", "graph-review.json", "--confirm"])
    combined = "\n".join([build.stdout, status.stdout, query.stdout, lineage.stdout, export.stdout])

    assert build.exit_code == 0
    assert "Graph built." in build.stdout
    assert "nodes:" in build.stdout
    assert status.exit_code == 0
    assert "Graph artifact: present" in status.stdout
    assert query.exit_code == 0
    assert "Graph query:" in query.stdout
    assert "path_found: True" in query.stdout
    assert "subgraph_nodes:" in query.stdout
    assert lineage.exit_code == 0
    assert "Graph lineage:" in lineage.stdout
    assert "hops:" in lineage.stdout
    assert export.exit_code == 0
    assert "Exported graph: graph-review.json" in export.stdout
    assert "PRIVATE" not in combined
    assert "artifact.txt" not in combined


def test_graph_query_requires_existing_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "graph-query-demo"]).exit_code == 0

    result = runner.invoke(app, ["graph", "query", "--node", "run:missing", "--json"])

    assert result.exit_code == 1
    assert "Run `pmem graph build` first" in result.stdout
    assert "Traceback" not in result.stdout


def _seed_project(tmp_path) -> RunCaptureResult:
    init_project(tmp_path, project_name="graph-cli-demo", primary_metric="accuracy")
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 0.91}), encoding="utf-8")
    (tmp_path / "artifact.txt").write_text("artifact-data", encoding="utf-8")
    run_result = run_command(
        tmp_path,
        [sys.executable, "-c", "print('ok')"],
        metrics_path="metrics.json",
        artifact_paths=("artifact.txt",),
    )
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type="PrivacyRegression",
        description="PRIVATE failure text",
        root_cause="PRIVATE root cause",
        lesson="PRIVATE lesson",
    )
    log_decision(
        tmp_path,
        description="PRIVATE decision text",
        rationale="PRIVATE rationale",
        experiment_id=run_result.record.experiment_id,
    )
    add_note(tmp_path, content="PRIVATE note text", run_id=run_result.record.run_id)
    return run_result
