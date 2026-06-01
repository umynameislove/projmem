"""graph CLI privacy-safe graph CLI operations.

The service layer keeps Typer concerns out of the graph modules while reusing
the graph ingestion and query contracts: ingest SQLite into a ``GraphDocument``, round-trip
through the graph engine, query via graph query, and trace via lineage when needed by future
CLI surface. It never creates derived edges or exposes raw graph node
attributes in query output.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from pmem.errors import (
    PmemNotFoundError,
    PmemPersistenceError,
    PmemSecurityError,
    PmemValidationError,
)
from pmem.graph.engine import GraphEngine
from pmem.graph.incremental import build_graph_full, build_graph_incremental
from pmem.graph.ingestion import GraphEdge, GraphNode
from pmem.graph.lineage import GraphLineageService
from pmem.graph.persistence import (
    default_graph_artifact_path,
    read_graph_document,
    write_graph_document,
)
from pmem.graph.privacy import GRAPH_JSON_FILE_MODE, graph_artifact_policy
from pmem.graph.query import GraphQueryService
from pmem.repositories.sqlite import PMEM_DIRNAME
from pmem.services.project_context import require_project_context

GRAPH_BUILD_RESULT_VERSION = "graph-build-result-v1"
GRAPH_STATUS_RESULT_VERSION = "graph-status-result-v1"
GRAPH_QUERY_RESULT_VERSION = "graph-query-result-v1"
GRAPH_LINEAGE_RESULT_VERSION = "graph-lineage-result-v1"
GRAPH_EXPORT_RESULT_VERSION = "graph-export-result-v1"


@dataclass(frozen=True, slots=True)
class GraphExportResult:
    """Result of exporting the private graph artifact to an explicit file."""

    output_path: Path
    display_path: str
    payload: dict[str, Any]


def build_graph_artifact(project_root: str | Path, *, incremental: bool = False) -> dict[str, Any]:
    """Build and persist the private project-local graph artifact."""

    context = require_project_context(project_root)
    result = (
        build_graph_incremental(context.root) if incremental else build_graph_full(context.root)
    )
    engine_document = result.document
    graph_path = default_graph_artifact_path(context.root)
    if result.should_persist:
        write_graph_document(engine_document, graph_path, project_root=context.root)
    mode = graph_path.stat().st_mode & 0o777
    return {
        "schema_version": GRAPH_BUILD_RESULT_VERSION,
        "ok": True,
        "mode": result.mode,
        "graph_schema_version": engine_document.schema_version,
        "graph_path": _display_path(context.root, graph_path),
        "counts": engine_document.counts,
        "warnings": list(result.warnings or engine_document.warnings),
        "skipped_counts": engine_document.skipped_counts,
        "source_changed": result.source_changed,
        "graph_changed": result.graph_changed,
        "previous_source_fingerprint": result.previous_fingerprint,
        "source_fingerprint": result.current_fingerprint,
        "source_fingerprint_prefix": result.current_fingerprint[:19],
        "source_table_counts": result.table_counts,
        "persisted": result.should_persist,
        "file_mode": f"0o{mode:03o}",
        "privacy": graph_artifact_policy(),
        "database_mutation": False,
    }


def graph_status_payload(project_root: str | Path) -> dict[str, Any]:
    """Return graph artifact status without exposing nodes or edges."""

    context = require_project_context(project_root)
    graph_path = default_graph_artifact_path(context.root)
    if not graph_path.exists():
        return {
            "schema_version": GRAPH_STATUS_RESULT_VERSION,
            "exists": False,
            "graph_path": _display_path(context.root, graph_path),
            "message": "No graph artifact found. Run `pmem graph build` first.",
            "counts": {},
            "warnings": [],
            "skipped_counts": {},
            "privacy": graph_artifact_policy(),
        }

    document = read_graph_document(graph_path)
    mode = graph_path.stat().st_mode & 0o777
    source_fingerprint = document.metadata.get("source_fingerprint")
    return {
        "schema_version": GRAPH_STATUS_RESULT_VERSION,
        "exists": True,
        "graph_schema_version": document.schema_version,
        "graph_path": _display_path(context.root, graph_path),
        "counts": document.counts,
        "warnings": list(document.warnings),
        "skipped_counts": document.skipped_counts,
        "build_mode": document.metadata.get("build_mode"),
        "source_fingerprint": source_fingerprint,
        "source_fingerprint_prefix": (
            source_fingerprint[:19] if isinstance(source_fingerprint, str) else None
        ),
        "source_fingerprint_computed_at": document.metadata.get("source_fingerprint_computed_at"),
        "source_table_counts": document.metadata.get("source_table_counts", {}),
        "updated_at": document.metadata.get("updated_at"),
        "full_rebuild_at": document.metadata.get("full_rebuild_at"),
        "incremental_requested": document.metadata.get("incremental_requested"),
        "incremental_applied": document.metadata.get("incremental_applied"),
        "fallback_reason": document.metadata.get("fallback_reason"),
        "file_mode": f"0o{mode:03o}",
        "privacy": graph_artifact_policy(),
    }


def graph_query_payload(
    project_root: str | Path,
    *,
    node_id: str,
    edge_type: str | None = None,
    direction: str = "both",
    depth: int | None = None,
    path_to: str | None = None,
) -> dict[str, Any]:
    """Query the persisted graph artifact with metadata-only output."""

    context = require_project_context(project_root)
    query = _query_service_from_artifact(context.root)
    node = query.get_node(node_id)
    neighbors = query.get_neighbors(node_id, edge_type=edge_type, direction=direction)
    payload: dict[str, Any] = {
        "schema_version": GRAPH_QUERY_RESULT_VERSION,
        "node_id": node_id,
        "found": node is not None,
        "node": _safe_node_payload(node) if node is not None else None,
        "neighbors": [neighbor.to_dict() for neighbor in neighbors],
        "neighbor_count": len(neighbors),
        "direction": direction,
        "edge_type": edge_type,
        "database_mutation": False,
    }
    if path_to is not None:
        payload["path"] = query.get_path(node_id, path_to).to_dict()
    if depth is not None:
        subgraph = query.get_subgraph(node_id, depth=depth, edge_type=edge_type)
        payload["subgraph"] = {
            "root_node_id": subgraph.root_node_id,
            "depth": subgraph.depth,
            "nodes": [_safe_node_payload(item) for item in subgraph.nodes],
            "edges": [_safe_edge_payload(item) for item in subgraph.edges],
            "warnings": list(subgraph.warnings),
            "counts": {"nodes": len(subgraph.nodes), "edges": len(subgraph.edges)},
        }
    return payload


def graph_lineage_payload(project_root: str | Path, *, run_id: str) -> dict[str, Any]:
    """Trace one run lineage from the persisted graph artifact."""

    context = require_project_context(project_root)
    query = _query_service_from_artifact(context.root)
    lineage = GraphLineageService(query).trace_run_lineage(run_id)
    return {
        "schema_version": GRAPH_LINEAGE_RESULT_VERSION,
        "database_mutation": False,
        "lineage": lineage.to_dict(),
    }


def export_graph_artifact(
    project_root: str | Path,
    *,
    output_path: str | Path,
    confirm: bool,
) -> GraphExportResult:
    """Export the private derived graph artifact to an explicit review file."""

    if not confirm:
        raise PmemValidationError("Graph export requires --confirm because graph data is private.")
    context = require_project_context(project_root)
    graph_path = default_graph_artifact_path(context.root)
    document = read_graph_document(graph_path)
    output = _resolve_graph_export_path(context.root, output_path)
    payload = {
        "schema_version": GRAPH_EXPORT_RESULT_VERSION,
        "graph_schema_version": document.schema_version,
        "privacy": graph_artifact_policy(),
        "graph": document.to_dict(),
    }
    _write_private_json(output, payload)
    return GraphExportResult(
        output_path=output,
        display_path=_display_path(context.root, output),
        payload={
            "schema_version": GRAPH_EXPORT_RESULT_VERSION,
            "ok": True,
            "output_path": _display_path(context.root, output),
            "graph_path": _display_path(context.root, graph_path),
            "counts": document.counts,
            "privacy": graph_artifact_policy(),
            "file_mode": f"0o{(output.stat().st_mode & 0o777):03o}",
        },
    )


def _query_service_from_artifact(project_root: Path) -> GraphQueryService:
    graph_path = default_graph_artifact_path(project_root)
    if not graph_path.exists():
        raise PmemNotFoundError("Graph artifact was not found. Run `pmem graph build` first.")
    document = read_graph_document(graph_path)
    return GraphQueryService(GraphEngine.from_document(document))


def _safe_node_payload(node: GraphNode) -> dict[str, str]:
    return {"id": node.node_id, "type": node.node_type.value}


def _safe_edge_payload(edge: GraphEdge) -> dict[str, object]:
    return {
        "id": edge.edge_id,
        "type": edge.edge_type.value,
        "source": edge.source,
        "target": edge.target,
        "edge_class": edge.edge_class.value,
        "provenance": [item.to_dict() for item in edge.provenance],
    }


def _resolve_graph_export_path(project_root: Path, user_path: str | Path) -> Path:
    raw_text = str(user_path).strip()
    if not raw_text:
        raise PmemValidationError("Graph export path cannot be blank.")
    if "\\" in raw_text or "\x00" in raw_text or any(ord(char) < 32 for char in raw_text):
        raise PmemSecurityError("Graph export path contains unsafe characters.")
    raw_path = Path(raw_text)
    if raw_path.is_absolute() or PureWindowsPath(raw_text).is_absolute():
        raise PmemSecurityError("Graph export path must be project-relative.")
    if any(part == ".." for part in raw_path.parts):
        raise PmemSecurityError("Graph export path cannot contain traversal segments.")
    if any(part.casefold() == PMEM_DIRNAME.casefold() for part in raw_path.parts):
        raise PmemSecurityError("Graph export path cannot point inside .pmem.")

    root = project_root.resolve()
    output = root / raw_path
    parent = output.parent.resolve(strict=False)
    if root != parent and root not in parent.parents:
        raise PmemSecurityError("Graph export path must stay inside the project.")
    _reject_symlink_parts(root, raw_path.parent)
    if output.exists() and output.is_dir():
        raise PmemSecurityError("Graph export path must point to a file, not a directory.")
    if output.exists() and output.is_symlink():
        raise PmemSecurityError("Graph export path cannot be a symlink.")
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_parts(root, raw_path.parent)
    if parent.is_symlink():
        raise PmemSecurityError("Graph export path cannot contain symlinks.")
    return parent / output.name


def _write_private_json(output: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temp_path = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, GRAPH_JSON_FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output)
        os.chmod(output, GRAPH_JSON_FILE_MODE)
    except OSError as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise PmemPersistenceError("Graph export file could not be written.") from exc


def _reject_symlink_parts(project_root: Path, relative_parent: Path) -> None:
    current = project_root
    for part in relative_parent.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.exists() and current.is_symlink():
            raise PmemSecurityError("Graph export path cannot contain symlinks.")


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name
