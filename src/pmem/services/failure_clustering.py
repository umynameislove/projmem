"""Deterministic local clustering for failure-analysis layer failure clustering."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.errors import PmemValidationError
from pmem.services.failure_embeddings import (
    DEFAULT_EMBEDDING_DIMENSION,
    FAILURE_EMBEDDING_SCHEMA_VERSION,
    cosine_similarity,
    failure_embedding_payload,
    validate_embedding_dimension,
)

FAILURE_CLUSTER_SCHEMA_VERSION = "failure-cluster-v1"
DEFAULT_SIMILARITY_THRESHOLD = 0.42


def failure_cluster_payload(
    project_root: str | Path,
    *,
    include_text: bool = False,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return deterministic clusters and 2D projection for failure embeddings."""

    clean_dimension = validate_embedding_dimension(dimension)
    clean_threshold = validate_similarity_threshold(similarity_threshold)
    embeddings = failure_embedding_payload(
        project_root,
        include_text=include_text,
        dimension=clean_dimension,
        generated_at=generated_at,
    )
    records = list(embeddings["records"])
    components = _connected_components(records, threshold=clean_threshold)
    clusters = _cluster_summaries(components)
    points = _projection_points(records, components)
    return {
        "schema_version": FAILURE_CLUSTER_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "embedding_schema_version": FAILURE_EMBEDDING_SCHEMA_VERSION,
        "embedding_method": embeddings["method"],
        "dimension": clean_dimension,
        "similarity_threshold": clean_threshold,
        "privacy_mode": embeddings["privacy_mode"],
        "include_text": include_text,
        "record_count": len(records),
        "cluster_count": len(clusters),
        "clusters": clusters,
        "points": points,
        "projection": {
            "method": "top_variance_axes",
            "dimensions": 2,
            "network": False,
            "raw_text_in_output": False,
        },
        "privacy_flags": embeddings["privacy_flags"],
    }


def validate_similarity_threshold(value: float) -> float:
    """Validate failure clustering cosine similarity threshold."""

    if value < 0.0 or value > 1.0:
        raise PmemValidationError("Similarity threshold must be between 0.0 and 1.0.")
    return round(float(value), 6)


def _connected_components(
    records: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda item: str(item["failure_id"]))
    parent = {str(record["failure_id"]): str(record["failure_id"]) for record in ordered}

    def find(item_id: str) -> str:
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            score = cosine_similarity(left["vector"], right["vector"])
            if score >= threshold:
                union(str(left["failure_id"]), str(right["failure_id"]))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in ordered:
        grouped.setdefault(find(str(record["failure_id"])), []).append(record)
    return [grouped[key] for key in sorted(grouped)]


def _cluster_summaries(components: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for index, members in enumerate(components, start=1):
        failure_ids = [str(member["failure_id"]) for member in members]
        clusters.append(
            {
                "cluster_id": f"cluster_{index:03d}",
                "size": len(members),
                "failure_ids": failure_ids,
                "prototype_failure_id": _prototype_failure_id(members),
                "error_type_counts": _counter_dict(member["error_type"] for member in members),
                "severity_counts": _counter_dict(member["severity"] for member in members),
                "source_counts": _counter_dict(member["source"] for member in members),
                "tag_counts": _counter_dict(
                    tag for member in members for tag in member.get("tags", [])
                ),
            }
        )
    return clusters


def _prototype_failure_id(members: list[dict[str, Any]]) -> str:
    if len(members) == 1:
        return str(members[0]["failure_id"])
    scores: list[tuple[float, str]] = []
    for candidate in members:
        candidate_id = str(candidate["failure_id"])
        similarities = [
            cosine_similarity(candidate["vector"], other["vector"])
            for other in members
            if other is not candidate
        ]
        average = sum(similarities) / len(similarities) if similarities else 0.0
        scores.append((average, candidate_id))
    return sorted(scores, key=lambda item: (-item[0], item[1]))[0][1]


def _projection_points(
    records: list[dict[str, Any]],
    components: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not records:
        return []
    vectors = [record["vector"] for record in records]
    x_axis, y_axis = _top_variance_axes(vectors)
    means = [sum(vector[axis] for vector in vectors) / len(vectors) for axis in (x_axis, y_axis)]
    cluster_by_failure: dict[str, str] = {}
    for index, members in enumerate(components, start=1):
        for member in members:
            cluster_by_failure[str(member["failure_id"])] = f"cluster_{index:03d}"
    return [
        {
            "failure_id": str(record["failure_id"]),
            "cluster_id": cluster_by_failure[str(record["failure_id"])],
            "x": round(float(record["vector"][x_axis] - means[0]), 10),
            "y": round(float(record["vector"][y_axis] - means[1]), 10),
        }
        for record in sorted(records, key=lambda item: str(item["failure_id"]))
    ]


def _top_variance_axes(vectors: list[list[float]]) -> tuple[int, int]:
    dimension = len(vectors[0])
    variances: list[tuple[float, int]] = []
    for axis in range(dimension):
        values = [vector[axis] for vector in vectors]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        variances.append((variance, axis))
    ordered = sorted(variances, key=lambda item: (-item[0], item[1]))
    x_axis = ordered[0][1]
    y_axis = ordered[1][1] if len(ordered) > 1 else x_axis
    return x_axis, y_axis


def _counter_dict(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values if str(value).strip()).items()))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
