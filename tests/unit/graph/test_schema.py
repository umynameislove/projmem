"""graph schema graph schema contract tests."""

from __future__ import annotations

import json
import math

import networkx as nx
import pytest

from pmem.errors import PmemSecurityError, PmemValidationError
from pmem.graph.schema import (
    EDGE_SPECS,
    GRAPH_SCHEMA_VERSION,
    NODE_TYPES,
    EdgeClass,
    EdgeType,
    NodeType,
    artifact_node_id,
    canonical_schema_json,
    code_module_node_id,
    edge_id,
    graph_schema_document,
    is_supported_metric_value,
    metric_payload,
    normalize_project_relative_path,
)
from pmem.utils.hashing import compute_text_hash


def test_networkx_is_available_as_core_graph_dependency() -> None:
    """graph scope locks NetworkX as the core evidence graph dependency."""

    graph = nx.Graph()
    graph.add_edge("run:1", "experiment:1")

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1


def test_graph_schema_has_locked_node_and_edge_vocabulary() -> None:
    """graph schema should expose a deterministic node/edge schema snapshot."""

    document = graph_schema_document()
    edge_types = {spec["edge_type"] for spec in document["edge_specs"]}

    assert document["schema_version"] == GRAPH_SCHEMA_VERSION
    assert document["node_count"] == 11
    assert [node.value for node in NODE_TYPES] == [
        "project",
        "experiment",
        "run",
        "config",
        "dataset",
        "metric",
        "artifact",
        "failure",
        "decision",
        "note",
        "code_module",
    ]
    assert document["edge_count"] == len(EDGE_SPECS)
    assert EdgeType.OBSERVED_IN.value in edge_types
    assert "CAUSED_BY" not in edge_types
    assert all("CAUSE" not in edge_type for edge_type in edge_types)
    assert document["causal_failure_edge"] is False


def test_edge_specs_cover_all_classes_without_fabricating_deferred_edges() -> None:
    """graph ingestion can filter direct edges while later phases keep derived/deferred specs."""

    by_type = {spec.edge_type: spec for spec in EDGE_SPECS}

    assert by_type[EdgeType.BELONGS_TO].edge_class is EdgeClass.DIRECT
    assert by_type[EdgeType.OBSERVED_IN].edge_class is EdgeClass.DIRECT
    assert by_type[EdgeType.USES_CONFIG].edge_class is EdgeClass.CONDITIONAL_DIRECT
    assert by_type[EdgeType.SUPPORTS].edge_class is EdgeClass.DERIVED
    assert by_type[EdgeType.BASED_ON].edge_class is EdgeClass.OPTIONAL
    assert by_type[EdgeType.TRAINED_ON].edge_class is EdgeClass.DEFERRED


def test_edge_id_is_deterministic_and_rejects_unknown_edge_type() -> None:
    """Edge IDs must be stable across rebuilds and restricted to graph schema."""

    source = "failure:failure_123"
    target = "run:run_456"

    assert edge_id(EdgeType.OBSERVED_IN, source, target) == (
        "OBSERVED_IN::failure:failure_123::run:run_456"
    )
    assert edge_id("OBSERVED_IN", source, target) == edge_id(EdgeType.OBSERVED_IN, source, target)
    with pytest.raises(PmemValidationError, match="edge type"):
        edge_id("CAUSED_BY", source, target)


def test_code_module_node_id_uses_existing_text_hash_policy() -> None:
    """CodeModule IDs hash normalized project-relative paths via compute_text_hash."""

    expected_hash = compute_text_hash("src/train.py")

    assert code_module_node_id("project_1", "src/train.py") == f"code:project_1:{expected_hash}"
    assert code_module_node_id("project_1", "src\\train.py") == f"code:project_1:{expected_hash}"


def test_artifact_node_id_hashes_normalized_project_path() -> None:
    """Artifact IDs should avoid exposing raw path strings."""

    expected_hash = compute_text_hash("outputs/metrics.json")
    node_id = artifact_node_id("run_1", "outputs\\metrics.json")

    assert node_id == f"artifact:run_1:{expected_hash}"
    assert "outputs" not in node_id


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/tmp/train.py",
        "C:\\tmp\\train.py",
        "../secret.py",
        "src/../../secret.py",
        ".PMEM/pmem.db",
        "src/\x00model.py",
    ],
)
def test_graph_path_normalization_rejects_unsafe_paths(unsafe_path: str) -> None:
    """Graph path policy preserves traversal and .pmem safety."""

    with pytest.raises((PmemSecurityError, PmemValidationError)):
        normalize_project_relative_path(unsafe_path)


def test_graph_path_normalization_is_stable_for_safe_relative_paths() -> None:
    assert normalize_project_relative_path("./src/model.py") == "src/model.py"
    assert normalize_project_relative_path("src\\model.py") == "src/model.py"


def test_metric_policy_accepts_only_finite_numeric_non_boolean_values() -> None:
    """Metric fan-out should not create nodes from non-numeric or unsafe values."""

    assert is_supported_metric_value(0)
    assert is_supported_metric_value(0.5)
    assert not is_supported_metric_value(True)
    assert not is_supported_metric_value("0.5")
    assert not is_supported_metric_value(None)
    assert not is_supported_metric_value(float("nan"))
    assert not is_supported_metric_value(float("inf"))

    payload = metric_payload(
        run_id="run_1",
        metric_name="accuracy",
        value=0.91,
        primary_metric="accuracy",
    )
    assert payload["node_id"] == "metric:run_1:accuracy"
    assert payload["is_primary_metric"] is True
    assert math.isclose(payload["value"], 0.91)


def test_canonical_schema_json_is_byte_stable() -> None:
    first = canonical_schema_json()
    second = canonical_schema_json()

    assert first == second
    assert json.loads(first)["schema_version"] == GRAPH_SCHEMA_VERSION
    assert "OBSERVED_IN" in first
    assert "CAUSED_BY" not in first


def test_all_node_types_are_unique() -> None:
    assert len({node.value for node in NodeType}) == len(NodeType)
