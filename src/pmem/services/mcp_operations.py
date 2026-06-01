"""MCP stdio local stdio MCP tool operations.

This module intentionally avoids importing the CLI layer or starting a network
server. It implements a small JSON-RPC stdio surface around existing projmem
services so FastAPI adapter can add HTTP later without sharing transport code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from pmem import __version__
from pmem.errors import PmemError, PmemNotFoundError, PmemValidationError
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.failure_exports import failure_export_payload
from pmem.services.graph_operations import graph_query_payload, graph_status_payload
from pmem.services.project_context import require_project_context
from pmem.services.recommendation_operations import recommendation_list_payload
from pmem.summary import get_project_summary, summary_json_payload

MCP_STDIO_RESULT_VERSION = "mcp-stdio-result-v1"
MCP_CONTEXT_PACK_VERSION = "mcp-context-pack-v1"
MCP_DEFAULT_TOKEN_BUDGET = 100_000
MCP_MAX_ITEMS = 50

_TOOL_NAMES = (
    "get_project_summary",
    "get_current_state",
    "get_related_experiments",
    "get_failures",
    "get_recommendations",
    "get_graph_neighbors",
    "get_context_pack",
)


@dataclass(frozen=True, slots=True)
class McpToolSpec:
    """One stdio MCP tool contract exposed by MCP stdio."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def mcp_tool_specs() -> tuple[McpToolSpec, ...]:
    """Return MCP stdio tool specs in stable order."""

    return (
        McpToolSpec(
            name="get_project_summary",
            description="Return metadata-only project summary counts and status.",
            input_schema=_object_schema(),
        ),
        McpToolSpec(
            name="get_current_state",
            description="Return project summary, graph status, and recommendation counts.",
            input_schema=_object_schema(
                max_recommendations=_integer_schema(1, MCP_MAX_ITEMS),
            ),
        ),
        McpToolSpec(
            name="get_related_experiments",
            description="Return metadata-only experiment records without hypotheses.",
            input_schema=_object_schema(max_items=_integer_schema(1, MCP_MAX_ITEMS)),
        ),
        McpToolSpec(
            name="get_failures",
            description="Return confirmed failure metadata without raw text.",
            input_schema=_object_schema(max_items=_integer_schema(1, MCP_MAX_ITEMS)),
        ),
        McpToolSpec(
            name="get_recommendations",
            description="Return evidence-backed recommendation candidates.",
            input_schema=_object_schema(
                max_recommendations=_integer_schema(1, MCP_MAX_ITEMS),
            ),
        ),
        McpToolSpec(
            name="get_graph_neighbors",
            description="Return metadata-only neighbors for one graph node.",
            input_schema=_object_schema(
                required=("node_id",),
                node_id={"type": "string"},
                edge_type={"type": "string"},
                direction={"type": "string", "enum": ["in", "out", "both"]},
                depth=_integer_schema(0, 3),
            ),
        ),
        McpToolSpec(
            name="get_context_pack",
            description="Return bounded structured context for local human review.",
            input_schema=_object_schema(
                max_items=_integer_schema(1, MCP_MAX_ITEMS),
                token_budget=_integer_schema(1_000, MCP_DEFAULT_TOKEN_BUDGET),
                graph_node_id={"type": "string"},
            ),
        ),
    )


def call_mcp_tool(
    project_root: str | Path,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch one MCP stdio MCP tool and return a JSON-serializable payload."""

    args = arguments or {}
    if name == "get_project_summary":
        return _project_summary_payload(project_root)
    if name == "get_current_state":
        return _current_state_payload(
            project_root,
            max_recommendations=_clean_int(
                args.get("max_recommendations", 5),
                name="max_recommendations",
                minimum=1,
                maximum=MCP_MAX_ITEMS,
            ),
        )
    if name == "get_related_experiments":
        return _related_experiments_payload(
            project_root,
            max_items=_clean_int(
                args.get("max_items", 10),
                name="max_items",
                minimum=1,
                maximum=MCP_MAX_ITEMS,
            ),
        )
    if name == "get_failures":
        return _failures_payload(
            project_root,
            max_items=_clean_int(
                args.get("max_items", 10),
                name="max_items",
                minimum=1,
                maximum=MCP_MAX_ITEMS,
            ),
        )
    if name == "get_recommendations":
        return recommendation_list_payload(
            project_root,
            max_recommendations=_clean_int(
                args.get("max_recommendations", 5),
                name="max_recommendations",
                minimum=1,
                maximum=MCP_MAX_ITEMS,
            ),
        )
    if name == "get_graph_neighbors":
        return _graph_neighbors_payload(project_root, args)
    if name == "get_context_pack":
        return get_context_pack(project_root, args)
    raise PmemValidationError(f"Unknown MCP tool: {name}")


def get_context_pack(
    project_root: str | Path,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a bounded, metadata-only MCP stdio context pack."""

    args = arguments or {}
    max_items = _clean_int(
        args.get("max_items", 10),
        name="max_items",
        minimum=1,
        maximum=MCP_MAX_ITEMS,
    )
    token_budget = _clean_int(
        args.get("token_budget", MCP_DEFAULT_TOKEN_BUDGET),
        name="token_budget",
        minimum=1_000,
        maximum=MCP_DEFAULT_TOKEN_BUDGET,
    )
    payload = {
        "schema_version": MCP_CONTEXT_PACK_VERSION,
        "transport": "stdio",
        "project_summary": _project_summary_payload(project_root),
        "current_state": _current_state_payload(
            project_root,
            max_recommendations=min(max_items, 10),
        ),
        "related_experiments": _related_experiments_payload(project_root, max_items=max_items),
        "failures": _failures_payload(project_root, max_items=max_items),
        "recommendations": recommendation_list_payload(
            project_root,
            max_recommendations=min(max_items, 10),
        ),
        "graph": _optional_graph_context(project_root, args),
        "privacy": _mcp_privacy_policy(),
        "warnings": [
            "Raw project, failure, decision, note, command, stdout, stderr, and path text "
            "is omitted from MCP stdio context packs by default."
        ],
        "database_mutation": False,
        "network": False,
    }
    return _fit_context_budget(payload, token_budget=token_budget, max_items=max_items)


def handle_mcp_message(project_root: str | Path, message: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message for the local stdio transport."""

    message_id = message.get("id")
    method = message.get("method")
    if not isinstance(method, str):
        return _jsonrpc_error(message_id, -32600, "JSON-RPC method must be a string.")
    if method.startswith("notifications/"):
        return None
    try:
        if method == "initialize":
            return _jsonrpc_result(message_id, _initialize_result())
        if method == "ping":
            return _jsonrpc_result(message_id, {})
        if method == "tools/list":
            return _jsonrpc_result(
                message_id,
                {"tools": [spec.to_dict() for spec in mcp_tool_specs()]},
            )
        if method == "tools/call":
            params = message.get("params")
            if not isinstance(params, dict):
                raise PmemValidationError("tools/call params must be an object.")
            tool_name = params.get("name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise PmemValidationError("tools/call requires a non-empty tool name.")
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise PmemValidationError("tools/call arguments must be an object.")
            payload = call_mcp_tool(project_root, tool_name.strip(), arguments)
            return _jsonrpc_result(message_id, _tool_result(payload))
        return _jsonrpc_error(message_id, -32601, f"Unsupported MCP method: {method}")
    except PmemError as exc:
        return _jsonrpc_error(message_id, -32000, str(exc))


def run_mcp_stdio(project_root: str | Path, stdin: TextIO, stdout: TextIO) -> None:
    """Run the local stdio JSON-RPC loop until stdin closes."""

    for line in stdin:
        cleaned = line.strip()
        if not cleaned:
            continue
        try:
            message = json.loads(cleaned)
            if not isinstance(message, dict):
                response = _jsonrpc_error(None, -32600, "JSON-RPC message must be an object.")
            else:
                response = handle_mcp_message(project_root, message)
        except json.JSONDecodeError:
            response = _jsonrpc_error(None, -32700, "Invalid JSON-RPC message.")
        if response is not None:
            stdout.write(_canonical_json(response) + "\n")
            stdout.flush()


def _project_summary_payload(project_root: str | Path) -> dict[str, Any]:
    summary = summary_json_payload(get_project_summary(project_root))
    return {
        "schema_version": "mcp-project-summary-v1",
        "project_id": summary["project_id"],
        "project_name_included": False,
        "objective_included": False,
        "primary_metric": summary["primary_metric"],
        "metric_direction": summary["metric_direction"],
        "target_value": summary["target_value"],
        "run_count": summary["run_count"],
        "successful_run_count": summary["successful_run_count"],
        "failed_run_count": summary["failed_run_count"],
        "best_run_id": summary["best_run_id"],
        "best_metric_value": summary["best_metric_value"],
        "target_status": summary["target_status"],
        "tracked_path_count": summary["tracked_path_count"],
        "failure_count": summary["failure_count"],
        "decision_count": summary["decision_count"],
        "note_count": summary["note_count"],
        "baseline_run_id": summary["baseline_run_id"],
        "warnings": summary["warnings"],
        "privacy_mode": "metadata_only",
        "raw_text_in_output": False,
        "database_mutation": False,
    }


def _current_state_payload(project_root: str | Path, *, max_recommendations: int) -> dict[str, Any]:
    recommendations = recommendation_list_payload(
        project_root,
        max_recommendations=max_recommendations,
    )
    graph_status = graph_status_payload(project_root)
    return {
        "schema_version": "mcp-current-state-v1",
        "project_summary": _project_summary_payload(project_root),
        "graph_status": graph_status,
        "recommendation_count": recommendations["recommendation_count"],
        "recommendation_basis_counts": recommendations["basis_counts"],
        "recommendation_warnings": recommendations["warnings"],
        "privacy_mode": "metadata_only",
        "raw_text_in_output": False,
        "database_mutation": False,
    }


def _related_experiments_payload(project_root: str | Path, *, max_items: int) -> dict[str, Any]:
    context = require_project_context(project_root)
    connection = connect_database(project_database_path(context.root))
    try:
        experiments = ExperimentRepository(connection).list_for_project(context.project.id)
    finally:
        connection.close()
    records = [
        {
            "id": experiment.id,
            "project_id": experiment.project_id,
            "name_included": False,
            "hypothesis_included": False,
            "status": experiment.status,
            "is_baseline": experiment.is_baseline,
            "primary_metric": experiment.primary_metric,
            "has_target": experiment.target_json is not None,
            "created_at": experiment.created_at,
            "updated_at": experiment.updated_at,
        }
        for experiment in experiments[:max_items]
    ]
    return {
        "schema_version": "mcp-related-experiments-v1",
        "record_count": len(experiments),
        "returned_count": len(records),
        "records": records,
        "privacy_mode": "metadata_only",
        "raw_text_in_output": False,
        "database_mutation": False,
    }


def _failures_payload(project_root: str | Path, *, max_items: int) -> dict[str, Any]:
    payload = failure_export_payload(project_root, include_text=False)
    records = payload["records"]
    if not isinstance(records, list):
        records = []
    return {
        "schema_version": "mcp-failures-v1",
        "record_count": payload["record_count"],
        "returned_count": len(records[:max_items]),
        "records": records[:max_items],
        "privacy_mode": "metadata_only",
        "raw_text_in_output": False,
        "database_mutation": False,
    }


def _graph_neighbors_payload(project_root: str | Path, args: dict[str, Any]) -> dict[str, Any]:
    node_id = args.get("node_id")
    if not isinstance(node_id, str) or not node_id.strip():
        raise PmemValidationError("get_graph_neighbors requires node_id.")
    try:
        return graph_query_payload(
            project_root,
            node_id=node_id.strip(),
            edge_type=_optional_string(args.get("edge_type")),
            direction=_optional_string(args.get("direction")) or "both",
            depth=_optional_int(args.get("depth"), name="depth"),
        )
    except PmemNotFoundError as exc:
        return {
            "schema_version": "mcp-graph-neighbors-v1",
            "available": False,
            "node_id": node_id.strip(),
            "message": str(exc),
            "neighbors": [],
            "neighbor_count": 0,
            "privacy_mode": "metadata_only",
            "raw_text_in_output": False,
            "database_mutation": False,
        }


def _optional_graph_context(project_root: str | Path, args: dict[str, Any]) -> dict[str, Any]:
    node_id = args.get("graph_node_id")
    if isinstance(node_id, str) and node_id.strip():
        return _graph_neighbors_payload(project_root, {"node_id": node_id.strip(), "depth": 1})
    return graph_status_payload(project_root)


def _fit_context_budget(
    payload: dict[str, Any],
    *,
    token_budget: int,
    max_items: int,
) -> dict[str, Any]:
    warnings = list(payload["warnings"])
    result = dict(payload)
    estimated_tokens = _estimated_tokens(_canonical_json(result))
    if estimated_tokens > token_budget:
        warnings.append("Context pack exceeded budget; list fields were truncated.")
        result["related_experiments"] = _truncate_records(
            result["related_experiments"],
            max_items=1,
        )
        result["failures"] = _truncate_records(result["failures"], max_items=1)
        recommendations = result["recommendations"]
        if isinstance(recommendations, dict):
            recommendations = dict(recommendations)
            items = recommendations.get("recommendations")
            if isinstance(items, list):
                recommendations["recommendations"] = items[:1]
                recommendations["recommendation_count"] = len(items)
                recommendations["returned_count"] = len(items[:1])
            result["recommendations"] = recommendations
        result["warnings"] = sorted(set(warnings))
        estimated_tokens = _estimated_tokens(_canonical_json(result))
    result["budget"] = {
        "token_budget": token_budget,
        "estimated_tokens": estimated_tokens,
        "within_budget": estimated_tokens <= token_budget,
        "token_counter": "conservative_json_character_count",
        "max_items": max_items,
    }
    if estimated_tokens > token_budget:
        result["warnings"] = sorted(
            set([*warnings, "Context pack still exceeds budget after truncation."])
        )
    else:
        result["warnings"] = sorted(set(result["warnings"]))
    return result


def _truncate_records(payload: Any, *, max_items: int) -> Any:
    if not isinstance(payload, dict):
        return payload
    result = dict(payload)
    records = result.get("records")
    if isinstance(records, list):
        result["records"] = records[:max_items]
        result["returned_count"] = len(records[:max_items])
    return result


def _initialize_result() -> dict[str, Any]:
    return {
        "schema_version": MCP_STDIO_RESULT_VERSION,
        "protocolVersion": "projmem-stdio-jsonrpc-v1",
        "serverInfo": {"name": "projmem", "version": __version__},
        "capabilities": {"tools": {}},
        "transport": "stdio",
        "network": False,
    }


def _tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    text = _canonical_json(payload)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload,
        "isError": False,
    }


def _mcp_privacy_policy() -> dict[str, Any]:
    return {
        "privacy_mode": "metadata_only",
        "raw_failure_text": "omitted",
        "raw_decision_text": "omitted",
        "raw_note_text": "omitted",
        "raw_command_text": "omitted",
        "raw_stdout_stderr": "omitted",
        "raw_paths": "omitted",
        "database_mutation": False,
        "network": False,
    }


def _object_schema(
    *,
    required: tuple[str, ...] = (),
    **properties: dict[str, Any],
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def _integer_schema(minimum: int, maximum: int) -> dict[str, int | str]:
    return {"type": "integer", "minimum": minimum, "maximum": maximum}


def _clean_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PmemValidationError(f"{name} must be an integer.")
    if value < minimum:
        raise PmemValidationError(f"{name} must be at least {minimum}.")
    if value > maximum:
        raise PmemValidationError(f"{name} must be {maximum} or less.")
    return value


def _optional_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _clean_int(value, name=name, minimum=0, maximum=3)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PmemValidationError("Optional string argument must be a string.")
    cleaned = value.strip()
    return cleaned or None


def _jsonrpc_result(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _estimated_tokens(text: str) -> int:
    # Conservative for MCP stdio: budget by JSON character count instead of relying on
    # a tokenizer dependency in the core install.
    return len(text)
