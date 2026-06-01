"""MCP stdio stdio MCP service tests."""

from __future__ import annotations

import io
import json

import pytest

from pmem.errors import PmemValidationError
from pmem.graph.schema import run_node_id
from pmem.repositories.decisions import DecisionRepository
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.notes import NoteRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services import mcp_operations as operations
from pmem.services.graph_operations import build_graph_artifact
from pmem.services.mcp_operations import (
    MCP_DEFAULT_TOKEN_BUDGET,
    call_mcp_tool,
    handle_mcp_message,
    mcp_tool_specs,
    run_mcp_stdio,
)
from pmem.services.project_init import init_project
from pmem.utils.hashing import compute_text_hash

NOW = "2026-05-31T12:00:00Z"


def test_mcp_tool_specs_expose_d61_tools() -> None:
    """MCP stdio should expose the seven locked MCP tools in stable order."""

    specs = mcp_tool_specs()

    assert [spec.name for spec in specs] == [
        "get_project_summary",
        "get_current_state",
        "get_related_experiments",
        "get_failures",
        "get_recommendations",
        "get_graph_neighbors",
        "get_context_pack",
    ]
    assert all(spec.input_schema["type"] == "object" for spec in specs)


def test_context_pack_omits_raw_text_and_fits_budget(tmp_path) -> None:
    """MCP stdio context packs should be bounded and metadata-only by default."""

    _seed_private_project(tmp_path)

    payload = call_mcp_tool(
        tmp_path,
        "get_context_pack",
        {"max_items": 5, "token_budget": MCP_DEFAULT_TOKEN_BUDGET},
    )
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == "mcp-context-pack-v1"
    assert payload["budget"]["within_budget"] is True
    assert payload["budget"]["estimated_tokens"] <= MCP_DEFAULT_TOKEN_BUDGET
    assert payload["project_summary"]["project_name_included"] is False
    assert payload["project_summary"]["objective_included"] is False
    assert payload["related_experiments"]["records"][0]["name_included"] is False
    assert payload["related_experiments"]["records"][0]["hypothesis_included"] is False
    assert payload["failures"]["records"][0]["text_included"] is False
    assert "PRIVATE" not in raw_json
    assert "python train.py" not in raw_json
    assert payload["database_mutation"] is False
    assert payload["network"] is False


def test_core_mcp_tools_return_metadata_only_payloads(tmp_path) -> None:
    """MCP stdio core tools should be callable without exposing raw project text."""

    _seed_private_project(tmp_path)

    current = call_mcp_tool(tmp_path, "get_current_state", {"max_recommendations": 3})
    experiments = call_mcp_tool(tmp_path, "get_related_experiments", {"max_items": 2})
    failures = call_mcp_tool(tmp_path, "get_failures", {"max_items": 2})
    recommendations = call_mcp_tool(tmp_path, "get_recommendations", {"max_recommendations": 3})
    raw_json = json.dumps(
        {
            "current": current,
            "experiments": experiments,
            "failures": failures,
            "recommendations": recommendations,
        },
        sort_keys=True,
    )

    assert current["schema_version"] == "mcp-current-state-v1"
    assert experiments["returned_count"] == 1
    assert failures["returned_count"] == 1
    assert "recommendation_count" in recommendations
    assert "PRIVATE" not in raw_json
    assert "python train.py" not in raw_json


def test_context_pack_truncates_when_budget_is_tight(tmp_path) -> None:
    """MCP stdio should degrade with warnings instead of crashing on small budgets."""

    _seed_private_project(tmp_path)

    payload = call_mcp_tool(tmp_path, "get_context_pack", {"max_items": 10, "token_budget": 1000})

    assert payload["budget"]["token_budget"] == 1000
    assert any("budget" in warning for warning in payload["warnings"])
    assert payload["failures"]["returned_count"] <= 1


def test_context_pack_can_include_optional_graph_node_context(tmp_path) -> None:
    """MCP stdio context packs can include safe graph context for one requested node."""

    _seed_private_project(tmp_path)
    build_graph_artifact(tmp_path)

    payload = call_mcp_tool(
        tmp_path,
        "get_context_pack",
        {"graph_node_id": run_node_id("run_1"), "max_items": 3},
    )

    assert payload["graph"]["found"] is True
    assert payload["graph"]["subgraph"]["counts"]["nodes"] >= 1


def test_graph_neighbors_degrades_when_graph_artifact_is_missing(tmp_path) -> None:
    """MCP stdio graph tool should not require graph build before startup."""

    _seed_private_project(tmp_path)

    payload = call_mcp_tool(tmp_path, "get_graph_neighbors", {"node_id": run_node_id("run_1")})

    assert payload["available"] is False
    assert payload["neighbor_count"] == 0
    assert "Run `pmem graph build` first" in payload["message"]


def test_graph_neighbors_returns_safe_payload_after_graph_build(tmp_path) -> None:
    """MCP stdio graph neighbors should reuse graph CLI safe graph query payloads."""

    _seed_private_project(tmp_path)
    build_graph_artifact(tmp_path)

    payload = call_mcp_tool(
        tmp_path,
        "get_graph_neighbors",
        {"node_id": run_node_id("run_1"), "direction": "both", "depth": 1},
    )
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["found"] is True
    assert payload["neighbor_count"] >= 1
    assert "subgraph" in payload
    assert "attributes" not in raw_json
    assert "PRIVATE" not in raw_json


def test_handle_mcp_message_lists_and_calls_tools(tmp_path) -> None:
    """The JSON-RPC dispatcher should support tools/list and tools/call."""

    _seed_private_project(tmp_path)

    listed = handle_mcp_message(tmp_path, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    called = handle_mcp_message(
        tmp_path,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "get_project_summary", "arguments": {}},
        },
    )

    assert listed is not None
    assert listed["result"]["tools"][0]["name"] == "get_project_summary"
    assert called is not None
    tool_text = called["result"]["content"][0]["text"]
    assert json.loads(tool_text)["schema_version"] == "mcp-project-summary-v1"


def test_handle_mcp_message_rejects_invalid_jsonrpc_shapes(tmp_path) -> None:
    """Malformed JSON-RPC messages should return structured errors."""

    _seed_private_project(tmp_path)

    missing_method = handle_mcp_message(tmp_path, {"jsonrpc": "2.0", "id": 1})
    notification = handle_mcp_message(
        tmp_path,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    ping = handle_mcp_message(tmp_path, {"jsonrpc": "2.0", "id": 2, "method": "ping"})
    bad_params = handle_mcp_message(
        tmp_path,
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": []},
    )
    missing_name = handle_mcp_message(
        tmp_path,
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"arguments": {}}},
    )
    none_arguments = handle_mcp_message(
        tmp_path,
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "get_project_summary", "arguments": None},
        },
    )
    bad_arguments = handle_mcp_message(
        tmp_path,
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "get_project_summary", "arguments": []},
        },
    )
    unsupported = handle_mcp_message(
        tmp_path,
        {"jsonrpc": "2.0", "id": 7, "method": "resources/list"},
    )

    assert missing_method is not None
    assert missing_method["error"]["code"] == -32600
    assert notification is None
    assert ping == {"jsonrpc": "2.0", "id": 2, "result": {}}
    assert bad_params is not None and bad_params["error"]["code"] == -32000
    assert missing_name is not None and missing_name["error"]["code"] == -32000
    assert none_arguments is not None and "result" in none_arguments
    assert bad_arguments is not None and bad_arguments["error"]["code"] == -32000
    assert unsupported is not None and unsupported["error"]["code"] == -32601


def test_mcp_stdio_loop_handles_mock_client_lines(tmp_path) -> None:
    """A mock stdio client should receive one JSON-RPC response per request line."""

    _seed_private_project(tmp_path)
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {"name": "get_context_pack", "arguments": {"max_items": 3}},
                    }
                ),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    run_mcp_stdio(tmp_path, stdin, stdout)
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    context_text = responses[2]["result"]["content"][0]["text"]

    assert [response["id"] for response in responses] == [1, 2, 3]
    assert responses[0]["result"]["transport"] == "stdio"
    assert len(responses[1]["result"]["tools"]) == 7
    assert json.loads(context_text)["schema_version"] == "mcp-context-pack-v1"
    assert "PRIVATE" not in stdout.getvalue()


def test_mcp_stdio_loop_handles_blank_invalid_and_non_object_lines(tmp_path) -> None:
    """The stdio loop should fail closed for malformed client input."""

    _seed_private_project(tmp_path)
    stdin = io.StringIO("\nnot-json\n[]\n")
    stdout = io.StringIO()

    run_mcp_stdio(tmp_path, stdin, stdout)
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert [response["error"]["code"] for response in responses] == [-32700, -32600]


def test_mcp_rejects_unknown_tool_and_invalid_arguments(tmp_path) -> None:
    """MCP stdio should fail closed with expected validation errors."""

    _seed_private_project(tmp_path)

    with pytest.raises(PmemValidationError, match="Unknown MCP tool"):
        call_mcp_tool(tmp_path, "unknown_tool", {})
    with pytest.raises(PmemValidationError, match="node_id"):
        call_mcp_tool(tmp_path, "get_graph_neighbors", {})
    with pytest.raises(PmemValidationError, match="token_budget"):
        call_mcp_tool(tmp_path, "get_context_pack", {"token_budget": 999})
    with pytest.raises(PmemValidationError, match="max_items"):
        call_mcp_tool(tmp_path, "get_failures", {"max_items": True})
    with pytest.raises(PmemValidationError, match="max_recommendations"):
        call_mcp_tool(tmp_path, "get_recommendations", {"max_recommendations": 51})
    with pytest.raises(PmemValidationError, match="Optional string"):
        call_mcp_tool(tmp_path, "get_graph_neighbors", {"node_id": "run:1", "edge_type": 5})


def test_failure_payload_defensively_handles_non_list_records(monkeypatch, tmp_path) -> None:
    """MCP stdio should tolerate malformed lower-layer failure payload shape."""

    _seed_private_project(tmp_path)
    monkeypatch.setattr(
        operations,
        "failure_export_payload",
        lambda *_args, **_kwargs: {"record_count": 1, "records": "bad"},
    )

    payload = call_mcp_tool(tmp_path, "get_failures", {"max_items": 5})

    assert payload["record_count"] == 1
    assert payload["returned_count"] == 0
    assert payload["records"] == []


def test_private_helpers_keep_shape_for_unexpected_truncation_input() -> None:
    """Small defensive helpers should keep unexpected input unchanged."""

    assert operations._truncate_records("bad", max_items=1) == "bad"


def _seed_private_project(tmp_path) -> None:
    init_result = init_project(
        tmp_path,
        project_name="PRIVATE project name",
        primary_metric="accuracy",
        metric_direction="max",
    )
    connection = connect_database(project_database_path(tmp_path))
    try:
        experiments = ExperimentRepository(connection)
        runs = RunRepository(connection)
        failures = FailureRepository(connection)
        decisions = DecisionRepository(connection)
        notes = NoteRepository(connection)
        experiments.create(
            experiment_id="exp_private",
            project_id=init_result.project_id,
            name="PRIVATE experiment name",
            hypothesis="PRIVATE hypothesis",
            created_at=NOW,
            updated_at=NOW,
        )
        config_json = json.dumps({"family": "safe"}, sort_keys=True, separators=(",", ":"))
        runs.create(
            run_id="run_1",
            experiment_id="exp_private",
            command="python train.py --PRIVATE-token",
            cwd=".",
            exit_code=1,
            status="failed",
            config={"family": "safe"},
            config_hash=compute_text_hash(config_json),
            metrics={"accuracy": 0.5},
            artifacts=[
                {
                    "path": "outputs/private-artifact.txt",
                    "sha256": compute_text_hash("artifact"),
                    "size_bytes": 10,
                }
            ],
            timestamp=NOW,
        )
        failures.create(
            failure_id="failure_1",
            run_id="run_1",
            error_type="ValueError",
            description="PRIVATE failure description",
            root_cause="PRIVATE root cause",
            lesson="PRIVATE lesson",
            severity="high",
            tags=["metric"],
            source="user_confirmed",
            created_at=NOW,
        )
        decisions.create(
            decision_id="decision_1",
            project_id=init_result.project_id,
            experiment_id="exp_private",
            description="PRIVATE decision text",
            rationale="PRIVATE rationale",
            created_at=NOW,
        )
        notes.create(
            note_id="note_1",
            project_id=init_result.project_id,
            experiment_id="exp_private",
            run_id="run_1",
            content="PRIVATE note content",
            tags=["audit"],
            context={"private": "PRIVATE context"},
            resolved=False,
            created_at=NOW,
        )
    finally:
        connection.close()
