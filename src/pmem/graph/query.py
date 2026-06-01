"""graph query deterministic graph query API.

The query layer reads a snapshot from ``GraphEngine`` and builds lightweight
adjacency indexes for neighbors, path search, and bounded subgraph extraction.
It does not mutate the graph, read SQLite, write graph artifacts, or expose CLI
commands.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from pmem.errors import PmemValidationError
from pmem.graph.engine import GraphEngine
from pmem.graph.ingestion import GraphEdge, GraphNode
from pmem.graph.provenance import GraphProvenance
from pmem.graph.schema import EdgeType, NodeType

GRAPH_QUERY_METHOD = "graph-query-v1"
VALID_DIRECTIONS = frozenset({"in", "out", "both"})


@dataclass(frozen=True, slots=True)
class GraphNeighbor:
    """One adjacent node reached through one graph edge."""

    node_id: str
    edge_id: str
    edge_type: str
    direction: str
    node_type: str
    provenance: tuple[GraphProvenance, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready neighbor data without node attributes or raw text."""

        return {
            "node_id": self.node_id,
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "direction": self.direction,
            "node_type": self.node_type,
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(frozen=True, slots=True)
class GraphPath:
    """Deterministic path search result."""

    source_node_id: str
    target_node_id: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    found: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready path data."""

        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "found": self.found,
        }


@dataclass(frozen=True, slots=True)
class GraphSubgraph:
    """Bounded neighborhood extraction result.

    Subgraph output includes node attributes from the upstream ``GraphDocument``.
    Query output therefore relies on the graph ingestion and graph engine
    privacy contract: graph node attributes must already be metadata-only and
    must not contain raw free text or paths.
    """

    root_node_id: str
    depth: int
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready subgraph data."""

        return {
            "root_node_id": self.root_node_id,
            "depth": self.depth,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "warnings": list(self.warnings),
            "counts": {"nodes": len(self.nodes), "edges": len(self.edges)},
        }


class GraphQueryService:
    """Read-only deterministic queries over a graph engine snapshot."""

    def __init__(self, engine: GraphEngine) -> None:
        document = engine.to_document()
        self._nodes = {node.node_id: node for node in document.nodes}
        self._edges = {edge.edge_id: edge for edge in document.edges}
        self._out_edges: dict[str, list[GraphEdge]] = {}
        self._in_edges: dict[str, list[GraphEdge]] = {}
        for edge in sorted(document.edges, key=lambda item: item.edge_id):
            self._out_edges.setdefault(edge.source, []).append(edge)
            self._in_edges.setdefault(edge.target, []).append(edge)

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return one node from the query snapshot."""

        return self._nodes.get(node_id)

    def nodes_by_type(self, node_type: NodeType | str) -> tuple[GraphNode, ...]:
        """Return nodes of one type in stable order."""

        selected_type = _validate_node_type(node_type)
        return tuple(
            sorted(
                (node for node in self._nodes.values() if node.node_type is selected_type),
                key=lambda item: item.node_id,
            )
        )

    def get_neighbors(
        self,
        node_id: str,
        *,
        edge_type: EdgeType | str | None = None,
        direction: str = "both",
    ) -> tuple[GraphNeighbor, ...]:
        """Return deterministic adjacent nodes.

        Unknown nodes return an empty tuple so callers can degrade gracefully.
        """

        selected_direction = _validate_direction(direction)
        selected_edge_type = _validate_edge_type(edge_type)
        if node_id not in self._nodes:
            return ()

        neighbors: list[GraphNeighbor] = []
        if selected_direction in {"out", "both"}:
            for edge in self._out_edges.get(node_id, ()):
                if _edge_type_matches(edge, selected_edge_type):
                    neighbor_node = self._nodes.get(edge.target)
                    if neighbor_node is not None:
                        neighbors.append(_neighbor(edge, neighbor_node, "out"))
        if selected_direction in {"in", "both"}:
            for edge in self._in_edges.get(node_id, ()):
                if _edge_type_matches(edge, selected_edge_type):
                    neighbor_node = self._nodes.get(edge.source)
                    if neighbor_node is not None:
                        neighbors.append(_neighbor(edge, neighbor_node, "in"))
        return tuple(
            sorted(
                neighbors,
                key=lambda item: (item.node_id, item.edge_type, item.edge_id, item.direction),
            )
        )

    def get_path(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        max_depth: int | None = None,
    ) -> GraphPath:
        """Find a deterministic undirected evidence path between two nodes."""

        if max_depth is not None and max_depth < 0:
            raise PmemValidationError("Graph path max_depth must be non-negative.")
        if source_node_id not in self._nodes or target_node_id not in self._nodes:
            return GraphPath(source_node_id, target_node_id, (), (), False)
        if source_node_id == target_node_id:
            return GraphPath(source_node_id, target_node_id, (source_node_id,), (), True)

        queue: deque[tuple[str, tuple[str, ...], tuple[str, ...]]] = deque(
            [(source_node_id, (source_node_id,), ())]
        )
        visited = {source_node_id}
        while queue:
            current, path_nodes, path_edges = queue.popleft()
            if max_depth is not None and len(path_edges) >= max_depth:
                continue
            for neighbor in self.get_neighbors(current, direction="both"):
                if neighbor.node_id in visited:
                    continue
                next_nodes = (*path_nodes, neighbor.node_id)
                next_edges = (*path_edges, neighbor.edge_id)
                if neighbor.node_id == target_node_id:
                    return GraphPath(source_node_id, target_node_id, next_nodes, next_edges, True)
                visited.add(neighbor.node_id)
                queue.append((neighbor.node_id, next_nodes, next_edges))

        return GraphPath(source_node_id, target_node_id, (), (), False)

    def get_subgraph(
        self,
        root_node_id: str,
        *,
        depth: int = 1,
        edge_type: EdgeType | str | None = None,
    ) -> GraphSubgraph:
        """Return a deterministic bounded neighborhood around one root node."""

        if depth < 0:
            raise PmemValidationError("Graph subgraph depth must be non-negative.")
        selected_edge_type = _validate_edge_type(edge_type)
        root = self._nodes.get(root_node_id)
        if root is None:
            return GraphSubgraph(
                root_node_id=root_node_id,
                depth=depth,
                nodes=(),
                edges=(),
                warnings=("Root node was not found.",),
            )

        visited = {root_node_id}
        edge_ids: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(root_node_id, 0)])
        while queue:
            current, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            for neighbor in self.get_neighbors(
                current,
                edge_type=selected_edge_type,
                direction="both",
            ):
                edge_ids.add(neighbor.edge_id)
                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    queue.append((neighbor.node_id, current_depth + 1))

        nodes = tuple(self._nodes[node_id] for node_id in sorted(visited))
        edges = tuple(self._edges[edge_id] for edge_id in sorted(edge_ids))
        return GraphSubgraph(root_node_id=root_node_id, depth=depth, nodes=nodes, edges=edges)


def _neighbor(edge: GraphEdge, node: GraphNode, direction: str) -> GraphNeighbor:
    return GraphNeighbor(
        node_id=node.node_id,
        edge_id=edge.edge_id,
        edge_type=edge.edge_type.value,
        direction=direction,
        node_type=node.node_type.value,
        provenance=edge.provenance,
    )


def _edge_type_matches(edge: GraphEdge, selected_edge_type: EdgeType | None) -> bool:
    return selected_edge_type is None or edge.edge_type is selected_edge_type


def _validate_direction(direction: str) -> str:
    if direction not in VALID_DIRECTIONS:
        raise PmemValidationError("Graph query direction must be one of: in, out, both.")
    return direction


def _validate_edge_type(edge_type: EdgeType | str | None) -> EdgeType | None:
    if edge_type is None:
        return None
    if isinstance(edge_type, EdgeType):
        return edge_type
    try:
        return EdgeType(str(edge_type))
    except ValueError as exc:
        raise PmemValidationError("Graph query edge_type is not part of the graph schema.") from exc


def _validate_node_type(node_type: NodeType | str) -> NodeType:
    if isinstance(node_type, NodeType):
        return node_type
    try:
        return NodeType(str(node_type))
    except ValueError as exc:
        raise PmemValidationError("Graph query node_type is not part of the graph schema.") from exc
