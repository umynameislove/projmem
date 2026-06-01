"""lineage evidence lineage tracking over direct graph edges."""

from __future__ import annotations

from dataclasses import dataclass

from pmem.errors import PmemValidationError
from pmem.graph.ingestion import GraphNode
from pmem.graph.provenance import GraphProvenance
from pmem.graph.query import GraphNeighbor, GraphQueryService
from pmem.graph.schema import EdgeType, NodeType, run_node_id

GRAPH_LINEAGE_METHOD = "run-lineage-v1"


@dataclass(frozen=True, slots=True)
class LineageHop:
    """One ordered evidence item in a run lineage chain."""

    entity_type: str
    entity_id: str
    node_id: str
    edge_id: str | None
    edge_type: str | None
    direction: str
    provenance: tuple[GraphProvenance, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready lineage hop data without raw node attributes."""

        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "node_id": self.node_id,
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "direction": self.direction,
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(frozen=True, slots=True)
class RunLineage:
    """Ordered lineage result for one run."""

    run_node_id: str
    hops: tuple[LineageHop, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready lineage data."""

        return {
            "method": GRAPH_LINEAGE_METHOD,
            "run_node_id": self.run_node_id,
            "hops": [hop.to_dict() for hop in self.hops],
            "warnings": list(self.warnings),
            "counts": {"hops": len(self.hops)},
        }


class GraphLineageService:
    """Trace run evidence context using only existing graph relationships."""

    def __init__(self, query_service: GraphQueryService) -> None:
        self._query = query_service

    def trace_run_lineage(self, run_id: str) -> RunLineage:
        """Trace a run's direct evidence neighborhood.

        The result is an ordered audit chain, not a causal explanation.
        """

        target_run_node_id = _coerce_run_node_id(run_id)
        run_node = self._query.get_node(target_run_node_id)
        if run_node is None:
            return RunLineage(
                run_node_id=target_run_node_id,
                hops=(),
                warnings=("Run node was not found.",),
            )

        hops: list[LineageHop] = [_node_hop(run_node, direction="self")]
        warnings: list[str] = []
        seen: set[tuple[str, str | None, str]] = {(run_node.node_id, None, "self")}

        experiment_nodes = self._append_neighbors(
            hops,
            seen,
            source_node_id=target_run_node_id,
            edge_type=EdgeType.BELONGS_TO,
            direction="out",
            missing_warning="No experiment context is linked to this run.",
        )
        experiment_node = experiment_nodes[0] if experiment_nodes else None
        project_node = self._project_context_from_experiment(experiment_node, warnings=warnings)
        if project_node is not None:
            self._append_hop(hops, seen, _node_hop(project_node, direction="attribute"))

        if project_node is not None:
            self._append_neighbors(
                hops,
                seen,
                source_node_id=project_node.node_id,
                edge_type=EdgeType.TRACKS_CODE,
                direction="out",
                missing_warning="No tracked code module is linked to this project.",
                warnings=warnings,
            )

        self._append_neighbors(
            hops,
            seen,
            source_node_id=target_run_node_id,
            edge_type=EdgeType.USES_CONFIG,
            direction="out",
            missing_warning="No config node is linked to this run.",
            warnings=warnings,
        )
        self._append_neighbors(
            hops,
            seen,
            source_node_id=target_run_node_id,
            edge_type=EdgeType.PRODUCES_METRIC,
            direction="out",
            missing_warning="No metric nodes are linked to this run.",
            warnings=warnings,
        )
        self._append_neighbors(
            hops,
            seen,
            source_node_id=target_run_node_id,
            edge_type=EdgeType.PRODUCES_ARTIFACT,
            direction="out",
            missing_warning="No artifact nodes are linked to this run.",
            warnings=warnings,
        )
        self._append_neighbors(
            hops,
            seen,
            source_node_id=target_run_node_id,
            edge_type=EdgeType.OBSERVED_IN,
            direction="in",
            missing_warning="No failure nodes are linked to this run.",
            warnings=warnings,
        )
        self._append_neighbors(
            hops,
            seen,
            source_node_id=target_run_node_id,
            edge_type=EdgeType.NOTE_ON,
            direction="in",
            missing_warning="No run notes are linked to this run.",
            warnings=warnings,
        )

        if experiment_node is not None:
            self._append_neighbors(
                hops,
                seen,
                source_node_id=experiment_node.node_id,
                edge_type=EdgeType.NOTE_IN_EXPERIMENT,
                direction="in",
                missing_warning="No experiment notes are linked to this run's experiment.",
                warnings=warnings,
            )
            self._append_neighbors(
                hops,
                seen,
                source_node_id=experiment_node.node_id,
                edge_type=EdgeType.DECISION_IN_EXPERIMENT,
                direction="in",
                missing_warning="No experiment decisions are linked to this run's experiment.",
                warnings=warnings,
            )

        if project_node is not None:
            self._append_neighbors(
                hops,
                seen,
                source_node_id=project_node.node_id,
                edge_type=EdgeType.DECISION_IN_PROJECT,
                direction="in",
                missing_warning="No project decisions are linked to this run.",
                warnings=warnings,
            )

        return RunLineage(
            run_node_id=target_run_node_id,
            hops=tuple(hops),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _project_context_from_experiment(
        self,
        experiment_node: GraphNode | None,
        *,
        warnings: list[str],
    ) -> GraphNode | None:
        if experiment_node is None:
            return None
        project_id = experiment_node.attributes.get("project_id")
        if not isinstance(project_id, str):
            warnings.append("Experiment context does not include a project_id attribute.")
            return None
        project_node = self._query.get_node(project_id)
        if project_node is None:
            warnings.append("Experiment project context node was not found.")
        return project_node

    def _append_neighbors(
        self,
        hops: list[LineageHop],
        seen: set[tuple[str, str | None, str]],
        *,
        source_node_id: str,
        edge_type: EdgeType,
        direction: str,
        missing_warning: str,
        warnings: list[str] | None = None,
    ) -> tuple[GraphNode, ...]:
        neighbors = self._query.get_neighbors(
            source_node_id,
            edge_type=edge_type,
            direction=direction,
        )
        if not neighbors:
            if warnings is not None:
                warnings.append(missing_warning)
            return ()

        nodes: list[GraphNode] = []
        for neighbor in neighbors:
            node = self._query.get_node(neighbor.node_id)
            if node is None:
                continue
            nodes.append(node)
            self._append_hop(hops, seen, _neighbor_hop(neighbor))
        return tuple(nodes)

    @staticmethod
    def _append_hop(
        hops: list[LineageHop],
        seen: set[tuple[str, str | None, str]],
        hop: LineageHop,
    ) -> None:
        key = (hop.node_id, hop.edge_id, hop.direction)
        if key not in seen:
            seen.add(key)
            hops.append(hop)


def _node_hop(node: GraphNode, *, direction: str) -> LineageHop:
    return LineageHop(
        entity_type=node.node_type.value,
        entity_id=node.node_id,
        node_id=node.node_id,
        edge_id=None,
        edge_type=None,
        direction=direction,
        provenance=node.provenance,
    )


def _neighbor_hop(neighbor: GraphNeighbor) -> LineageHop:
    return LineageHop(
        entity_type=neighbor.node_type,
        entity_id=neighbor.node_id,
        node_id=neighbor.node_id,
        edge_id=neighbor.edge_id,
        edge_type=neighbor.edge_type,
        direction=neighbor.direction,
        provenance=neighbor.provenance,
    )


def _coerce_run_node_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id.strip():
        raise PmemValidationError("Run lineage requires a non-empty run id.")
    cleaned = run_id.strip()
    if cleaned.startswith(f"{NodeType.RUN.value}:"):
        return cleaned
    return run_node_id(cleaned)
