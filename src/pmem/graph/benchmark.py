"""NetworkX graph query benchmark.

This module is deliberately local and dependency-free. It builds synthetic
metadata-only ``GraphDocument`` objects, runs the graph query layer over the
NetworkX engine, and records enough timing evidence to decide whether projmem
should stay on NetworkX or open a Neo4j migration plan.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from pmem.errors import PmemValidationError
from pmem.graph.engine import GraphEngine
from pmem.graph.ingestion import GraphDocument, GraphEdge, GraphNode
from pmem.graph.provenance import provenance
from pmem.graph.query import GraphQueryService
from pmem.graph.schema import (
    GRAPH_SCHEMA_VERSION,
    EdgeClass,
    EdgeType,
    NodeType,
    edge_id,
    experiment_node_id,
    run_node_id,
)

NETWORKX_BENCHMARK_METHOD = "networkx-query-benchmark-v1"
NEO4J_MIGRATION_THRESHOLD_SECONDS = 2.0
MIGRATION_DECISION_NODE_COUNT = 5_000
DEFAULT_BENCHMARK_SIZES = (1_000, 5_000, 10_000)
DEFAULT_BENCHMARK_ITERATIONS = 5


@dataclass(frozen=True, slots=True)
class GraphQueryBenchmarkResult:
    """One synthetic graph benchmark result."""

    node_count: int
    edge_count: int
    iterations: int
    build_seconds: float
    query_p50_seconds: float
    query_p95_seconds: float
    query_p99_seconds: float
    query_max_seconds: float
    operation: str = "neighbors_path_subgraph"

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready benchmark data."""

        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "iterations": self.iterations,
            "build_seconds": round(self.build_seconds, 6),
            "query_p50_seconds": round(self.query_p50_seconds, 6),
            "query_p95_seconds": round(self.query_p95_seconds, 6),
            "query_p99_seconds": round(self.query_p99_seconds, 6),
            "query_max_seconds": round(self.query_max_seconds, 6),
            "operation": self.operation,
        }


@dataclass(frozen=True, slots=True)
class Neo4jMigrationGateResult:
    """Migration decision based on NetworkX benchmark evidence."""

    schema_version: str
    method: str
    threshold_seconds: float
    decision_node_count: int
    decision: str
    rationale: str
    results: tuple[GraphQueryBenchmarkResult, ...]
    database_mutation: bool = False
    network: bool = False
    raw_text_in_output: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready migration gate evidence."""

        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "threshold_seconds": self.threshold_seconds,
            "decision_node_count": self.decision_node_count,
            "decision": self.decision,
            "rationale": self.rationale,
            "results": [result.to_dict() for result in self.results],
            "database_mutation": self.database_mutation,
            "network": self.network,
            "raw_text_in_output": self.raw_text_in_output,
        }


def run_networkx_query_benchmark(
    *,
    sizes: tuple[int, ...] = DEFAULT_BENCHMARK_SIZES,
    iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
    threshold_seconds: float = NEO4J_MIGRATION_THRESHOLD_SECONDS,
) -> Neo4jMigrationGateResult:
    """Run the NetworkX migration benchmark.

    The benchmark is synthetic by design: it avoids reading private project data
    and measures the current graph engine/query path under predictable graph
    sizes. The decision is based on the 5K-node P99 migration threshold.
    """

    _validate_benchmark_inputs(
        sizes=sizes, iterations=iterations, threshold_seconds=threshold_seconds
    )
    results = tuple(_benchmark_size(size, iterations=iterations) for size in sizes)
    decision_result = _result_for_decision(results)
    if decision_result.query_p99_seconds < threshold_seconds:
        decision = "stay_networkx"
        rationale = (
            f"NetworkX P99 query time at {decision_result.node_count} nodes "
            f"was {decision_result.query_p99_seconds:.6f}s, below the "
            f"{threshold_seconds:.1f}s graph migration threshold."
        )
    else:
        decision = "create_neo4j_migration_plan"
        rationale = (
            f"NetworkX P99 query time at {decision_result.node_count} nodes "
            f"was {decision_result.query_p99_seconds:.6f}s, meeting or exceeding "
            f"the {threshold_seconds:.1f}s graph migration threshold."
        )
    return Neo4jMigrationGateResult(
        schema_version="neo4j-migration-gate-v1",
        method=NETWORKX_BENCHMARK_METHOD,
        threshold_seconds=threshold_seconds,
        decision_node_count=MIGRATION_DECISION_NODE_COUNT,
        decision=decision,
        rationale=rationale,
        results=results,
    )


def synthetic_query_benchmark_document(node_count: int) -> GraphDocument:
    """Build a metadata-only synthetic graph with one experiment hub."""

    if node_count < 2:
        raise PmemValidationError("Synthetic graph benchmark requires at least 2 nodes.")
    experiment = GraphNode(
        node_id=experiment_node_id("d64_exp"),
        node_type=NodeType.EXPERIMENT,
        attributes={"synthetic": True},
        provenance=(
            provenance(
                source_table="d64_synthetic",
                source_pk="experiment:d64_exp",
                source_field="id",
                creation_rule="d64 benchmark synthetic experiment",
            ),
        ),
    )
    runs = tuple(_synthetic_run(index) for index in range(node_count - 1))
    edges = tuple(_synthetic_run_edge(run.node_id, experiment.node_id) for run in runs)
    return GraphDocument(
        schema_version=GRAPH_SCHEMA_VERSION,
        method=NETWORKX_BENCHMARK_METHOD,
        nodes=(experiment, *runs),
        edges=edges,
        counts={
            "nodes": node_count,
            "edges": len(edges),
            "node_types": {
                NodeType.EXPERIMENT.value: 1,
                NodeType.RUN.value: len(runs),
            },
            "edge_types": {EdgeType.BELONGS_TO.value: len(edges)},
        },
        warnings=(),
        skipped_counts={},
        metadata={
            "synthetic": True,
            "database_mutation": False,
            "raw_text_in_output": False,
            "network": False,
        },
    )


def _benchmark_size(node_count: int, *, iterations: int) -> GraphQueryBenchmarkResult:
    document = synthetic_query_benchmark_document(node_count)
    start = time.perf_counter()
    engine = GraphEngine.from_document(document)
    service = GraphQueryService(engine)
    build_seconds = time.perf_counter() - start
    experiment_id = experiment_node_id("d64_exp")
    first_run_id = run_node_id("d64_run_000000")
    last_run_id = run_node_id(f"d64_run_{node_count - 2:06d}")
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        neighbors = service.get_neighbors(experiment_id, direction="in")
        path = service.get_path(first_run_id, last_run_id)
        subgraph = service.get_subgraph(experiment_id, depth=1)
        elapsed = time.perf_counter() - start
        if len(neighbors) != node_count - 1 or not path.found or len(subgraph.nodes) != node_count:
            raise PmemValidationError("Graph benchmark query returned inconsistent results.")
        samples.append(elapsed)
    return GraphQueryBenchmarkResult(
        node_count=node_count,
        edge_count=len(document.edges),
        iterations=iterations,
        build_seconds=build_seconds,
        query_p50_seconds=_percentile(samples, 50),
        query_p95_seconds=_percentile(samples, 95),
        query_p99_seconds=_percentile(samples, 99),
        query_max_seconds=max(samples),
    )


def _synthetic_run(index: int) -> GraphNode:
    node_id = run_node_id(f"d64_run_{index:06d}")
    return GraphNode(
        node_id=node_id,
        node_type=NodeType.RUN,
        attributes={"synthetic": True, "ordinal": index},
        provenance=(
            provenance(
                source_table="d64_synthetic",
                source_pk=node_id,
                source_field="run_id",
                creation_rule="d64 benchmark synthetic run",
            ),
        ),
    )


def _synthetic_run_edge(source: str, target: str) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id(EdgeType.BELONGS_TO, source, target),
        edge_type=EdgeType.BELONGS_TO,
        source=source,
        target=target,
        edge_class=EdgeClass.DIRECT,
        attributes={"synthetic": True},
        provenance=(
            provenance(
                source_table="d64_synthetic",
                source_pk=f"{source}->{target}",
                source_field="experiment_id",
                creation_rule="d64 benchmark synthetic BELONGS_TO edge",
            ),
        ),
    )


def _validate_benchmark_inputs(
    *,
    sizes: tuple[int, ...],
    iterations: int,
    threshold_seconds: float,
) -> None:
    if not sizes:
        raise PmemValidationError("Graph benchmark sizes cannot be empty.")
    if any(size < 2 for size in sizes):
        raise PmemValidationError("Graph benchmark sizes must be at least 2 nodes.")
    if iterations < 1:
        raise PmemValidationError("Graph benchmark iterations must be at least 1.")
    if threshold_seconds <= 0:
        raise PmemValidationError("Graph benchmark threshold must be positive.")


def _result_for_decision(
    results: tuple[GraphQueryBenchmarkResult, ...],
) -> GraphQueryBenchmarkResult:
    for result in sorted(results, key=lambda item: item.node_count):
        if result.node_count >= MIGRATION_DECISION_NODE_COUNT:
            return result
    raise PmemValidationError(
        "Graph benchmark must include at least one result at or above 5K nodes."
    )


def _percentile(samples: list[float], percentile: int) -> float:
    if not samples:
        raise PmemValidationError("Graph benchmark samples cannot be empty.")
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile / 100 * len(ordered)) - 1))
    return ordered[index]
