"""graph engine graph document serialization and private artifact persistence."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from pmem.errors import (
    PmemNotFoundError,
    PmemPersistenceError,
    PmemSecurityError,
    PmemValidationError,
)
from pmem.graph.ingestion import GraphDocument, GraphEdge, GraphNode
from pmem.graph.privacy import GRAPH_JSON_FILE_MODE, GRAPH_JSON_RELATIVE_PATH
from pmem.graph.provenance import GraphProvenance
from pmem.graph.schema import GRAPH_SCHEMA_VERSION, EdgeClass, EdgeType, NodeType


def default_graph_artifact_path(project_root: str | Path) -> Path:
    """Return the private project-local graph artifact path."""

    return Path(project_root) / GRAPH_JSON_RELATIVE_PATH


def write_graph_document(
    document: GraphDocument,
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> None:
    """Write a graph document atomically with private file permissions."""

    _validate_document(document)
    target = _validated_write_path(path, project_root=project_root)
    if not target.parent.exists():
        raise PmemNotFoundError("Graph artifact directory was not found.")

    payload = json.dumps(document.to_dict(), indent=2, sort_keys=True) + "\n"
    temp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, GRAPH_JSON_FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        os.chmod(target, GRAPH_JSON_FILE_MODE)
    except OSError as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise PmemPersistenceError("Graph artifact write failed.") from exc


def read_graph_document(path: str | Path) -> GraphDocument:
    """Read and validate a persisted graph document."""

    target = Path(path)
    if not target.exists():
        raise PmemNotFoundError("Graph artifact was not found.")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PmemValidationError("Graph artifact is not valid JSON.") from exc
    except OSError as exc:
        raise PmemPersistenceError("Graph artifact read failed.") from exc

    document = _document_from_payload(payload)
    _validate_document(document)
    return document


def _validated_write_path(path: str | Path, *, project_root: str | Path | None) -> Path:
    target = Path(path)
    _validate_graph_artifact_shape(target)
    if project_root is None:
        return target

    root = Path(project_root).resolve()
    expected = default_graph_artifact_path(root).resolve()
    resolved = target.resolve()
    if resolved != expected:
        raise PmemSecurityError("Graph artifact path must be project-local .pmem/graph.json.")
    if resolved.is_symlink():
        raise PmemSecurityError("Graph artifact path must not be a symlink.")
    return resolved


def _validate_graph_artifact_shape(path: Path) -> None:
    if path.name != "graph.json" or path.parent.name != ".pmem":
        raise PmemSecurityError("Graph artifact path must be .pmem/graph.json.")
    if ".." in path.parts:
        raise PmemSecurityError("Graph artifact path must not traverse directories.")
    if path.exists() and path.is_symlink():
        raise PmemSecurityError("Graph artifact path must not be a symlink.")


def _document_from_payload(payload: object) -> GraphDocument:
    if not isinstance(payload, dict):
        raise PmemValidationError("Graph artifact must be a JSON object.")
    if payload.get("schema_version") != GRAPH_SCHEMA_VERSION:
        raise PmemValidationError("Unsupported graph schema version.")

    nodes_value = payload.get("nodes")
    edges_value = payload.get("edges")
    if not isinstance(nodes_value, list) or not isinstance(edges_value, list):
        raise PmemValidationError("Graph artifact nodes and edges must be arrays.")

    counts = payload.get("counts", {})
    metadata = payload.get("metadata", {})
    skipped_counts = payload.get("skipped_counts", {})
    warnings = payload.get("warnings", [])
    if not isinstance(counts, dict):
        raise PmemValidationError("Graph artifact counts must be an object.")
    if not isinstance(metadata, dict):
        raise PmemValidationError("Graph artifact metadata must be an object.")
    if not isinstance(skipped_counts, dict):
        raise PmemValidationError("Graph artifact skipped_counts must be an object.")
    for value in skipped_counts.values():
        if not isinstance(value, int) or isinstance(value, bool):
            raise PmemValidationError("Graph artifact skipped_counts values must be integers.")
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise PmemValidationError("Graph artifact warnings must be strings.")

    document = GraphDocument(
        schema_version=str(payload["schema_version"]),
        method=_required_string(payload, "method"),
        nodes=tuple(_node_from_payload(item) for item in nodes_value),
        edges=tuple(_edge_from_payload(item) for item in edges_value),
        counts={str(key): value for key, value in counts.items()},
        warnings=tuple(warnings),
        skipped_counts={str(key): value for key, value in skipped_counts.items()},
        metadata={str(key): value for key, value in metadata.items()},
    )
    return document


def _node_from_payload(payload: object) -> GraphNode:
    if not isinstance(payload, dict):
        raise PmemValidationError("Graph node entries must be objects.")
    attributes = payload.get("attributes")
    if not isinstance(attributes, dict):
        raise PmemValidationError("Graph node attributes must be an object.")
    try:
        node_type = NodeType(_required_string(payload, "type"))
    except ValueError as exc:
        raise PmemValidationError("Graph node type is not supported.") from exc
    return GraphNode(
        node_id=_required_string(payload, "id"),
        node_type=node_type,
        attributes={str(key): value for key, value in attributes.items()},
        provenance=_provenance_tuple(payload.get("provenance")),
    )


def _edge_from_payload(payload: object) -> GraphEdge:
    if not isinstance(payload, dict):
        raise PmemValidationError("Graph edge entries must be objects.")
    attributes = payload.get("attributes")
    if not isinstance(attributes, dict):
        raise PmemValidationError("Graph edge attributes must be an object.")
    try:
        edge_type = EdgeType(_required_string(payload, "type"))
        edge_class = EdgeClass(_required_string(payload, "edge_class"))
    except ValueError as exc:
        raise PmemValidationError("Graph edge type or class is not supported.") from exc
    return GraphEdge(
        edge_id=_required_string(payload, "id"),
        edge_type=edge_type,
        source=_required_string(payload, "source"),
        target=_required_string(payload, "target"),
        edge_class=edge_class,
        attributes={str(key): value for key, value in attributes.items()},
        provenance=_provenance_tuple(payload.get("provenance")),
    )


def _provenance_tuple(payload: object) -> tuple[GraphProvenance, ...]:
    if not isinstance(payload, list) or not payload:
        raise PmemValidationError("Graph provenance must be a non-empty array.")
    items: list[GraphProvenance] = []
    required = ("source_table", "source_pk", "source_field", "creation_rule")
    for item in payload:
        if not isinstance(item, dict):
            raise PmemValidationError("Graph provenance entries must be objects.")
        if any(not str(item.get(key, "")).strip() for key in required):
            raise PmemValidationError("Graph provenance entries must include source evidence.")
        items.append(
            GraphProvenance(
                source_table=str(item["source_table"]),
                source_pk=str(item["source_pk"]),
                source_field=str(item["source_field"]),
                creation_rule=str(item["creation_rule"]),
            )
        )
    return tuple(items)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PmemValidationError(f"Graph artifact field {key} is required.")
    return value


def _validate_document(document: GraphDocument) -> None:
    if not isinstance(document, GraphDocument):
        raise PmemValidationError("Graph persistence requires a GraphDocument.")
    if document.schema_version != GRAPH_SCHEMA_VERSION:
        raise PmemValidationError("Unsupported graph schema version.")
    node_ids: set[str] = set()
    for node in document.nodes:
        if not isinstance(node, GraphNode):
            raise PmemValidationError("Graph document nodes must be GraphNode objects.")
        if not node.node_id.strip() or not node.provenance:
            raise PmemValidationError("Graph document nodes require id and provenance.")
        node_ids.add(node.node_id)
    for edge in document.edges:
        if not isinstance(edge, GraphEdge):
            raise PmemValidationError("Graph document edges must be GraphEdge objects.")
        if not edge.edge_id.strip() or not edge.provenance:
            raise PmemValidationError("Graph document edges require id and provenance.")
        if edge.source not in node_ids or edge.target not in node_ids:
            raise PmemValidationError("Graph artifact references a missing node.")
