"""MCP stdio MCP stdio CLI integration tests."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.graph.schema import run_node_id
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.graph_operations import build_graph_artifact
from pmem.services.project_init import init_project
from pmem.utils.hashing import compute_text_hash

runner = CliRunner()
NOW = "2026-05-31T13:00:00Z"


def test_mcp_help_is_available() -> None:
    """MCP stdio should expose a root-level pmem mcp command."""

    root_help = runner.invoke(app, ["--help"])
    mcp_help = runner.invoke(app, ["mcp", "--help"])

    assert root_help.exit_code == 0
    assert "mcp" in root_help.stdout
    assert mcp_help.exit_code == 0
    assert "stdio MCP" in mcp_help.stdout


def test_mcp_stdio_mock_client_gets_context_pack(monkeypatch, tmp_path) -> None:
    """Mock MCP clients should receive structured project context over stdin/stdout."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)

    request = {
        "jsonrpc": "2.0",
        "id": "ctx-1",
        "method": "tools/call",
        "params": {"name": "get_context_pack", "arguments": {"max_items": 5}},
    }
    result = runner.invoke(app, ["mcp"], input=json.dumps(request) + "\n")
    response = json.loads(result.stdout)
    payload = json.loads(response["result"]["content"][0]["text"])
    combined = json.dumps(response, sort_keys=True)

    assert result.exit_code == 0
    assert response["id"] == "ctx-1"
    assert payload["schema_version"] == "mcp-context-pack-v1"
    assert payload["transport"] == "stdio"
    assert payload["project_summary"]["run_count"] == 1
    assert payload["failures"]["record_count"] == 1
    assert payload["budget"]["within_budget"] is True
    assert "PRIVATE" not in combined
    assert "python train.py" not in combined
    assert payload["database_mutation"] is False
    assert payload["network"] is False


def test_mcp_stdio_tools_list_and_graph_neighbors(monkeypatch, tmp_path) -> None:
    """MCP stdio should expose tools/list and graph neighbor calls after graph build."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    build_graph_artifact(tmp_path)
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_graph_neighbors",
                "arguments": {"node_id": run_node_id("run_d61"), "direction": "both", "depth": 1},
            },
        },
    ]
    result = runner.invoke(
        app,
        ["mcp"],
        input="\n".join(json.dumps(message) for message in messages) + "\n",
    )
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    tool_names = [tool["name"] for tool in responses[0]["result"]["tools"]]
    graph_payload = json.loads(responses[1]["result"]["content"][0]["text"])
    combined = json.dumps(responses, sort_keys=True)

    assert result.exit_code == 0
    assert len(tool_names) == 7
    assert "get_context_pack" in tool_names
    assert graph_payload["found"] is True
    assert graph_payload["neighbor_count"] >= 1
    assert "attributes" not in combined
    assert "PRIVATE" not in combined


def test_mcp_stdio_unknown_tool_returns_jsonrpc_error(monkeypatch, tmp_path) -> None:
    """Expected MCP validation failures should be JSON-RPC errors, not tracebacks."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    request = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {"name": "missing_tool", "arguments": {}},
    }

    result = runner.invoke(app, ["mcp"], input=json.dumps(request) + "\n")
    response = json.loads(result.stdout)

    assert result.exit_code == 0
    assert response["id"] == 9
    assert response["error"]["code"] == -32000
    assert "Unknown MCP tool" in response["error"]["message"]
    assert "Traceback" not in result.stdout


def _seed_project(tmp_path) -> None:
    init_result = init_project(
        tmp_path,
        project_name="PRIVATE d61 project",
        primary_metric="accuracy",
        metric_direction="max",
    )
    connection = connect_database(project_database_path(tmp_path))
    try:
        experiments = ExperimentRepository(connection)
        runs = RunRepository(connection)
        failures = FailureRepository(connection)
        experiments.create(
            experiment_id="exp_d61",
            project_id=init_result.project_id,
            name="PRIVATE experiment",
            hypothesis="PRIVATE hypothesis",
            created_at=NOW,
            updated_at=NOW,
        )
        config_json = json.dumps({"family": "d61"}, sort_keys=True, separators=(",", ":"))
        runs.create(
            run_id="run_d61",
            experiment_id="exp_d61",
            command="python train.py --PRIVATE",
            cwd=".",
            exit_code=1,
            status="failed",
            config={"family": "d61"},
            config_hash=compute_text_hash(config_json),
            metrics={"accuracy": 0.42},
            timestamp=NOW,
        )
        failures.create(
            failure_id="failure_d61",
            run_id="run_d61",
            error_type="ValueError",
            description="PRIVATE failure text",
            root_cause="PRIVATE root",
            lesson="PRIVATE lesson",
            severity="high",
            tags=["metric"],
            source="user_confirmed",
            created_at=NOW,
        )
    finally:
        connection.close()
