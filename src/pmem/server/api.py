"""FastAPI adapter localhost-first FastAPI adapter.

The HTTP layer is intentionally thin: it delegates to graph/recommendation/MCP metadata-only
services and never imports the CLI layer. The default bind stays on loopback.
"""

from __future__ import annotations

from ipaddress import ip_address
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from pmem.errors import (
    PmemError,
    PmemNotFoundError,
    PmemSecurityError,
    PmemValidationError,
)
from pmem.services.graph_operations import graph_lineage_payload
from pmem.services.mcp_operations import call_mcp_tool

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8765
API_RESULT_VERSION = "api-result-v1"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def create_api_app(project_root: str | Path) -> FastAPI:
    """Create the local FastAPI adapter REST adapter for one initialized project."""

    root = Path(project_root)
    app = FastAPI(
        title="projmem local API",
        version=API_RESULT_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.exception_handler(PmemError)
    async def handle_pmem_error(_request: Request, exc: PmemError) -> JSONResponse:
        return JSONResponse(
            status_code=_status_code(exc),
            content={
                "schema_version": API_RESULT_VERSION,
                "ok": False,
                "error": str(exc),
            },
        )

    @app.get("/summary")
    def get_summary() -> dict[str, Any]:
        return _api_payload("summary", call_mcp_tool(root, "get_project_summary"))

    @app.get("/current-state")
    def get_current_state(
        max_recommendations: int = Query(default=5, ge=1, le=50),
    ) -> dict[str, Any]:
        return _api_payload(
            "current-state",
            call_mcp_tool(
                root,
                "get_current_state",
                {"max_recommendations": max_recommendations},
            ),
        )

    @app.get("/recommendations")
    def get_recommendations(
        max_recommendations: int = Query(default=5, ge=1, le=50),
    ) -> dict[str, Any]:
        return _api_payload(
            "recommendations",
            call_mcp_tool(
                root,
                "get_recommendations",
                {"max_recommendations": max_recommendations},
            ),
        )

    @app.get("/failures")
    def get_failures(
        max_items: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, Any]:
        return _api_payload(
            "failures",
            call_mcp_tool(root, "get_failures", {"max_items": max_items}),
        )

    @app.get("/graph/neighbors")
    def get_graph_neighbors(
        node_id: str = Query(min_length=1),
        edge_type: str | None = Query(default=None, min_length=1),
        direction: str = Query(default="both", pattern="^(in|out|both)$"),
        depth: int | None = Query(default=None, ge=0, le=3),
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "node_id": node_id,
            "direction": direction,
        }
        if edge_type is not None:
            arguments["edge_type"] = edge_type
        if depth is not None:
            arguments["depth"] = depth
        return _api_payload(
            "graph-neighbors",
            call_mcp_tool(root, "get_graph_neighbors", arguments),
        )

    @app.get("/graph/lineage")
    def get_graph_lineage(run_id: str = Query(min_length=1)) -> dict[str, Any]:
        return _api_payload(
            "graph-lineage",
            graph_lineage_payload(root, run_id=run_id),
        )

    return app


def serve_configuration_payload(
    *,
    host: str = DEFAULT_API_HOST,
    port: int = DEFAULT_API_PORT,
    confirm_non_local_bind: bool = False,
) -> dict[str, Any]:
    """Validate and describe one FastAPI adapter bind configuration."""

    cleaned_host = _clean_host(host)
    cleaned_port = _clean_port(port)
    loopback = _is_loopback_host(cleaned_host)
    if not loopback and not confirm_non_local_bind:
        raise PmemSecurityError(
            "Non-local API bind requires --confirm-non-local-bind. "
            "The default 127.0.0.1 bind is safer."
        )
    warnings = []
    if not loopback:
        warnings.append(
            "API is binding outside loopback. Project metadata may be visible on the network."
        )
    return {
        "schema_version": API_RESULT_VERSION,
        "host": cleaned_host,
        "port": cleaned_port,
        "loopback_only": loopback,
        "warnings": warnings,
        "authentication": "none",
        "privacy_mode": "metadata_only",
        "database_mutation": False,
    }


def run_api_server(
    project_root: str | Path,
    *,
    host: str = DEFAULT_API_HOST,
    port: int = DEFAULT_API_PORT,
    confirm_non_local_bind: bool = False,
) -> dict[str, Any]:
    """Run the blocking FastAPI adapter Uvicorn server after bind validation.

    The returned configuration is reachable when an injected or patched runner
    exits, which keeps the bind contract directly testable.
    """

    configuration = serve_configuration_payload(
        host=host,
        port=port,
        confirm_non_local_bind=confirm_non_local_bind,
    )
    app = create_api_app(project_root)
    uvicorn.run(
        app,
        host=str(configuration["host"]),
        port=int(configuration["port"]),
        log_level="info",
    )
    return configuration


def _api_payload(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": API_RESULT_VERSION,
        "endpoint": endpoint,
        "payload": payload,
        "privacy_mode": "metadata_only",
        "database_mutation": False,
    }


def _status_code(exc: PmemError) -> int:
    if isinstance(exc, PmemNotFoundError):
        return 404
    if isinstance(exc, PmemSecurityError):
        return 403
    if isinstance(exc, PmemValidationError):
        return 422
    return 400


def _clean_host(host: str) -> str:
    cleaned = host.strip()
    if not cleaned:
        raise PmemValidationError("API host cannot be blank.")
    if "\x00" in cleaned or any(ord(char) < 32 for char in cleaned):
        raise PmemSecurityError("API host contains unsafe characters.")
    return cleaned


def _clean_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int):
        raise PmemValidationError("API port must be an integer.")
    if not 1 <= port <= 65_535:
        raise PmemValidationError("API port must be between 1 and 65535.")
    return port


def _is_loopback_host(host: str) -> bool:
    if host.casefold() in _LOOPBACK_HOSTS:
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped_address = getattr(address, "ipv4_mapped", None)
    return mapped_address is not None and mapped_address.is_loopback
