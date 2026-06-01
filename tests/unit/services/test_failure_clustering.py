"""failure clustering tests for deterministic local failure clustering."""

from __future__ import annotations

import json
import sys

import pytest

from pmem.errors import PmemValidationError
from pmem.services.failure_clustering import (
    FAILURE_CLUSTER_SCHEMA_VERSION,
    _connected_components,
    failure_cluster_payload,
)
from pmem.services.failure_logging import log_failure
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command


def test_failure_cluster_payload_handles_empty_project(tmp_path) -> None:
    """failure clustering should produce a stable empty report."""

    init_project(tmp_path, project_name="demo")

    payload = failure_cluster_payload(
        tmp_path,
        generated_at="2026-01-01T00:00:00Z",
        dimension=32,
    )

    assert payload["schema_version"] == FAILURE_CLUSTER_SCHEMA_VERSION
    assert payload["record_count"] == 0
    assert payload["cluster_count"] == 0
    assert payload["clusters"] == []
    assert payload["points"] == []


def test_failure_cluster_payload_groups_similar_failures(tmp_path) -> None:
    """Cosine-threshold connected components should group similar records."""

    _create_failure(
        tmp_path,
        error_type="MetricRegression",
        description="Accuracy dropped after noisy labels.",
        tag="data quality",
    )
    _create_failure(
        tmp_path,
        error_type="MetricRegression",
        description="Accuracy dropped after bad validation split.",
        tag="data quality",
    )
    _create_failure(
        tmp_path,
        error_type="OOM",
        description="GPU memory exhausted during larger batch.",
        tag="oom",
    )

    payload = failure_cluster_payload(
        tmp_path,
        include_text=True,
        generated_at="2026-01-01T00:00:00Z",
        dimension=128,
        similarity_threshold=0.8,
    )
    sizes = sorted(cluster["size"] for cluster in payload["clusters"])
    cluster_with_two = next(cluster for cluster in payload["clusters"] if cluster["size"] == 2)
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["cluster_count"] == 2
    assert sizes == [1, 2]
    assert cluster_with_two["tag_counts"] == {"data_quality": 2}
    assert len(payload["points"]) == 3
    assert all({"x", "y", "cluster_id", "failure_id"} <= set(point) for point in payload["points"])
    assert "Accuracy dropped" not in raw_json
    assert "GPU memory" not in raw_json


def test_failure_cluster_payload_is_deterministic(tmp_path) -> None:
    """Frozen timestamp and same records should produce equal reports."""

    _create_failure(
        tmp_path,
        error_type="ConfigError",
        description="Learning rate caused divergence.",
        tag="config error",
    )

    first = failure_cluster_payload(
        tmp_path,
        include_text=True,
        generated_at="2026-01-01T00:00:00Z",
        dimension=64,
        similarity_threshold=0.4,
    )
    second = failure_cluster_payload(
        tmp_path,
        include_text=True,
        generated_at="2026-01-01T00:00:00Z",
        dimension=64,
        similarity_threshold=0.4,
    )

    assert first == second
    assert first["projection"]["method"] == "top_variance_axes"
    assert first["clusters"][0]["prototype_failure_id"] == first["clusters"][0]["failure_ids"][0]


def test_failure_cluster_payload_rejects_invalid_threshold(tmp_path) -> None:
    """Similarity threshold must stay in cosine range."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemValidationError, match="threshold"):
        failure_cluster_payload(tmp_path, similarity_threshold=1.5)


def test_connected_components_covers_reversed_root_union_branch() -> None:
    """Synthetic vectors should cover the union branch independent of UUID order."""

    records = [
        _synthetic_record("fail_a", [1.0, 0.0]),
        _synthetic_record("fail_b", [0.0, 1.0]),
        _synthetic_record("fail_c", [0.7071067812, 0.7071067812]),
    ]

    components = _connected_components(records, threshold=0.6)

    assert len(components) == 1
    assert [record["failure_id"] for record in components[0]] == ["fail_a", "fail_b", "fail_c"]


def _create_failure(tmp_path, *, error_type: str, description: str, tag: str) -> None:
    if not (tmp_path / ".pmem").exists():
        init_project(tmp_path, project_name="demo")
    run_result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type=error_type,
        description=description,
        root_cause="Synthetic test root cause",
        lesson="Synthetic test lesson",
        severity="high",
        tags=(tag,),
        source="user_confirmed",
    )


def _synthetic_record(failure_id: str, vector: list[float]) -> dict[str, object]:
    return {
        "failure_id": failure_id,
        "run_id": f"run_{failure_id}",
        "error_type": "Synthetic",
        "severity": "medium",
        "source": "user_confirmed",
        "tags": [],
        "created_at": "2026-01-01T00:00:00Z",
        "vector": vector,
    }
