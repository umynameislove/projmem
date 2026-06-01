"""NetworkX migration benchmark gate tests."""

from __future__ import annotations

import json

import pytest

from pmem.errors import PmemValidationError
from pmem.graph.benchmark import (
    NEO4J_MIGRATION_THRESHOLD_SECONDS,
    NETWORKX_BENCHMARK_METHOD,
    run_networkx_query_benchmark,
    synthetic_query_benchmark_document,
)
from pmem.graph.schema import EdgeType, NodeType


def test_synthetic_benchmark_document_is_metadata_only_and_sized() -> None:
    """NetworkX benchmark synthetic benchmark documents should not use private project data."""

    document = synthetic_query_benchmark_document(100)
    payload = document.to_dict()
    raw_json = json.dumps(payload, sort_keys=True)

    assert document.method == NETWORKX_BENCHMARK_METHOD
    assert document.counts["nodes"] == 100
    assert document.counts["edges"] == 99
    assert document.counts["node_types"][NodeType.EXPERIMENT.value] == 1
    assert document.counts["node_types"][NodeType.RUN.value] == 99
    assert document.counts["edge_types"][EdgeType.BELONGS_TO.value] == 99
    assert payload["metadata"]["database_mutation"] is False
    assert payload["metadata"]["network"] is False
    assert "PRIVATE" not in raw_json
    assert "/Users/" not in raw_json
    assert "neo4j" not in raw_json.casefold()


def test_networkx_benchmark_gate_stays_networkx_for_5k_nodes() -> None:
    """NetworkX benchmark should keep NetworkX when the 5K-node P99 gate is below 2 seconds."""

    result = run_networkx_query_benchmark(sizes=(1_000, 5_000), iterations=2)
    payload = result.to_dict()
    five_k = payload["results"][1]

    assert payload["schema_version"] == "neo4j-migration-gate-v1"
    assert payload["method"] == NETWORKX_BENCHMARK_METHOD
    assert payload["decision"] == "stay_networkx"
    assert payload["threshold_seconds"] == NEO4J_MIGRATION_THRESHOLD_SECONDS
    assert five_k["node_count"] == 5_000
    assert five_k["query_p99_seconds"] < NEO4J_MIGRATION_THRESHOLD_SECONDS
    assert payload["database_mutation"] is False
    assert payload["network"] is False
    assert payload["raw_text_in_output"] is False


def test_networkx_benchmark_rejects_invalid_gate_inputs() -> None:
    """Fail closed for inputs that cannot support the migration decision."""

    with pytest.raises(PmemValidationError, match="sizes"):
        run_networkx_query_benchmark(sizes=(), iterations=1)
    with pytest.raises(PmemValidationError, match="at least 2 nodes"):
        synthetic_query_benchmark_document(1)
    with pytest.raises(PmemValidationError, match="iterations"):
        run_networkx_query_benchmark(sizes=(5_000,), iterations=0)
    with pytest.raises(PmemValidationError, match="5K"):
        run_networkx_query_benchmark(sizes=(1_000,), iterations=1)
