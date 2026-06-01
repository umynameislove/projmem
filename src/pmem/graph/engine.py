"""NetworkX graph engine.

The engine wraps an in-memory ``networkx.MultiDiGraph`` around graph documents.
It deliberately does not read SQLite, write graph artifacts, expose CLI
commands, or implement query APIs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import networkx as nx

from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.graph.ingestion import (
    GraphDocument,
    GraphEdge,
    GraphNode,
)
from pmem.graph.provenance import GraphProvenance
from pmem.graph.schema import GRAPH_SCHEMA_VERSION, EdgeClass, EdgeType, NodeType

GRAPH_ENGINE_METHOD = "networkx-multidigraph-v1"
GRAPH_ENGINE_ALLOWED_EDGE_CLASSES = frozenset({EdgeClass.DIRECT, EdgeClass.CONDITIONAL_DIRECT})


class GraphEngine:
    """NetworkX-backed CRUD foundation for evidence graph documents."""

    def __init__(
        self,
        *,
        schema_version: str = GRAPH_SCHEMA_VERSION,
        method: str = GRAPH_ENGINE_METHOD,
        metadata: Mapping[str, Any] | None = None,
        warnings: tuple[str, ...] = (),
        skipped_counts: Mapping[str, int] | None = None,
    ) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._edge_index: dict[str, tuple[str, str, str]] = {}
        self._schema_version = schema_version
        self._method = method
        self._metadata = _stable_data(dict(metadata or {}))
        self._warnings = tuple(warnings)
        self._skipped_counts = dict(sorted((skipped_counts or {}).items()))

    @classmethod
    def from_document(cls, document: GraphDocument) -> GraphEngine:
        """Build an engine from a graph document."""

        if document.schema_version != GRAPH_SCHEMA_VERSION:
            raise PmemValidationError("Unsupported graph schema version.")
        engine = cls(
            schema_version=document.schema_version,
            method=document.method,
            metadata=document.metadata,
            warnings=document.warnings,
            skipped_counts=document.skipped_counts,
        )
        for node in sorted(document.nodes, key=lambda item: item.node_id):
            engine.add_node(node)
        for edge in sorted(document.edges, key=lambda item: item.edge_id):
            engine.add_edge(edge)
        return engine

    def to_document(self) -> GraphDocument:
        """Return a deterministic graph document snapshot."""

        nodes: list[GraphNode] = []
        for node_id in sorted(self._graph.nodes):
            node = self.get_node(str(node_id))
            if node is None:
                raise PmemValidationError("Graph engine contains a malformed node.")
            nodes.append(node)
        edges = tuple(self._iter_edges_for_document())
        return GraphDocument(
            schema_version=self._schema_version,
            method=self._method,
            nodes=tuple(nodes),
            edges=edges,
            counts=self.counts(),
            warnings=self._warnings,
            skipped_counts=self._skipped_counts,
            metadata=dict(self._metadata),
        )

    def add_node(self, node: GraphNode) -> None:
        """Add or replace one graph node."""

        _validate_node(node)
        self._graph.add_node(
            node.node_id,
            node_id=node.node_id,
            node_type=node.node_type.value,
            attributes=_stable_data(node.attributes),
            provenance=[item.to_dict() for item in node.provenance],
        )

    def update_node(self, node_id: str, attributes: Mapping[str, Any]) -> None:
        """Merge attributes into an existing node."""

        if node_id not in self._graph:
            raise PmemNotFoundError("Graph node was not found.")
        if not isinstance(attributes, Mapping):
            raise PmemValidationError("Graph node attributes must be a mapping.")
        existing = dict(self._graph.nodes[node_id].get("attributes", {}))
        existing.update(dict(attributes))
        self._graph.nodes[node_id]["attributes"] = _stable_data(existing)

    def delete_node(self, node_id: str) -> None:
        """Delete a node and its incident edges."""

        if node_id not in self._graph:
            raise PmemNotFoundError("Graph node was not found.")
        self._graph.remove_node(node_id)
        self._rebuild_edge_index()

    def add_edge(self, edge: GraphEdge) -> None:
        """Add or replace one graph edge by canonical edge id."""

        _validate_edge(edge)
        if edge.edge_class not in GRAPH_ENGINE_ALLOWED_EDGE_CLASSES:
            raise PmemValidationError("Graph engine only accepts direct edge classes.")
        if edge.source not in self._graph or edge.target not in self._graph:
            raise PmemNotFoundError("Graph edge source or target node was not found.")
        existing = self._edge_index.get(edge.edge_id)
        if existing is not None:
            self._graph.remove_edge(*existing)
        self._graph.add_edge(
            edge.source,
            edge.target,
            key=edge.edge_id,
            edge_id=edge.edge_id,
            edge_type=edge.edge_type.value,
            edge_class=edge.edge_class.value,
            attributes=_stable_data(edge.attributes),
            provenance=[item.to_dict() for item in edge.provenance],
        )
        self._edge_index[edge.edge_id] = (edge.source, edge.target, edge.edge_id)

    def delete_edge(self, edge_id: str) -> None:
        """Delete one edge by canonical edge id."""

        found = self._edge_location(edge_id)
        if found is None:
            raise PmemNotFoundError("Graph edge was not found.")
        source, target, key = found
        self._graph.remove_edge(source, target, key)
        self._edge_index.pop(edge_id, None)

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return one graph node without exposing NetworkX internals."""

        if node_id not in self._graph:
            return None
        data = self._graph.nodes[node_id]
        return _node_from_engine_data(data)

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        """Return one graph edge by canonical edge id."""

        found = self._edge_location(edge_id)
        if found is None:
            return None
        source, target, key = found
        return _edge_from_engine_data(source, target, self._graph.edges[source, target, key])

    def counts(self) -> dict[str, Any]:
        """Return deterministic node/edge counts."""

        node_type_counts = Counter(
            str(data["node_type"]) for _, data in self._graph.nodes(data=True)
        )
        edge_type_counts = Counter(
            str(data["edge_type"]) for _, _, _, data in self._graph.edges(keys=True, data=True)
        )
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "node_types": dict(sorted(node_type_counts.items())),
            "edge_types": dict(sorted(edge_type_counts.items())),
        }

    def _edge_location(self, edge_id: str) -> tuple[str, str, str] | None:
        return self._edge_index.get(edge_id)

    def _iter_edges_for_document(self) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        edge_rows = sorted(
            self._graph.edges(keys=True, data=True),
            key=lambda item: str(item[3].get("edge_id", "")),
        )
        for source, target, key, data in edge_rows:
            graph_edge_id = data.get("edge_id")
            if not isinstance(graph_edge_id, str) or not graph_edge_id.strip():
                raise PmemValidationError("Graph engine contains a malformed edge.")
            location = self._edge_location(graph_edge_id)
            if location != (str(source), str(target), str(key)):
                raise PmemValidationError("Graph engine edge index is inconsistent.")
            edges.append(_edge_from_engine_data(str(source), str(target), data))
        return edges

    def _rebuild_edge_index(self) -> None:
        self._edge_index = {}
        for source, target, key, data in self._graph.edges(keys=True, data=True):
            graph_edge_id = data.get("edge_id")
            if isinstance(graph_edge_id, str) and graph_edge_id.strip():
                self._edge_index[graph_edge_id] = (str(source), str(target), str(key))


def _node_from_engine_data(data: Mapping[str, Any]) -> GraphNode:
    provenance_items = _provenance_from_data(data.get("provenance"))
    return GraphNode(
        node_id=str(data["node_id"]),
        node_type=NodeType(str(data["node_type"])),
        attributes=_stable_data(data.get("attributes", {})),
        provenance=tuple(provenance_items),
    )


def _edge_from_engine_data(source: str, target: str, data: Mapping[str, Any]) -> GraphEdge:
    provenance_items = _provenance_from_data(data.get("provenance"))
    return GraphEdge(
        edge_id=str(data["edge_id"]),
        edge_type=EdgeType(str(data["edge_type"])),
        source=source,
        target=target,
        edge_class=EdgeClass(str(data["edge_class"])),
        attributes=_stable_data(data.get("attributes", {})),
        provenance=tuple(provenance_items),
    )


def _validate_node(node: GraphNode) -> None:
    if not isinstance(node, GraphNode):
        raise PmemValidationError("Graph engine requires GraphNode objects.")
    if not node.node_id.strip():
        raise PmemValidationError("Graph node id is required.")
    if not node.provenance:
        raise PmemValidationError("Graph nodes require provenance.")


def _validate_edge(edge: GraphEdge) -> None:
    if not isinstance(edge, GraphEdge):
        raise PmemValidationError("Graph engine requires GraphEdge objects.")
    if not edge.edge_id.strip() or not edge.source.strip() or not edge.target.strip():
        raise PmemValidationError("Graph edge id, source, and target are required.")
    if not edge.provenance:
        raise PmemValidationError("Graph edges require provenance.")


def _provenance_from_data(value: object) -> tuple[GraphProvenance, ...]:
    if not isinstance(value, list) or not value:
        raise PmemValidationError("Graph provenance must be a non-empty list.")
    items: list[GraphProvenance] = []
    for item in value:
        if not isinstance(item, dict):
            raise PmemValidationError("Graph provenance items must be objects.")
        required = ("source_table", "source_pk", "source_field", "creation_rule")
        if any(not str(item.get(key, "")).strip() for key in required):
            raise PmemValidationError("Graph provenance items must include source evidence.")
        items.append(
            GraphProvenance(
                source_table=str(item.get("source_table", "")),
                source_pk=str(item.get("source_pk", "")),
                source_field=str(item.get("source_field", "")),
                creation_rule=str(item.get("creation_rule", "")),
            )
        )
    return tuple(items)


def _stable_data(value: Any) -> Any:
    return deepcopy(_stable_order(value))


def _stable_order(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_order(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_stable_order(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_order(item) for item in value]
    return value
