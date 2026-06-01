"""Recurring failure detection using embeddings plus graph context.

Recurring failure candidates are audit aids. They combine local failure
embeddings, clustering output, and graph context overlap. They do not claim
shared root cause, create graph edges, or require optional NLP packages.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.errors import PmemValidationError
from pmem.graph.ingestion import GraphDocument, GraphEdge, GraphNode, build_graph_from_project
from pmem.graph.schema import EdgeType, NodeType
from pmem.services.failure_clustering import (
    DEFAULT_SIMILARITY_THRESHOLD,
    FAILURE_CLUSTER_SCHEMA_VERSION,
    failure_cluster_payload,
    validate_similarity_threshold,
)
from pmem.services.failure_embeddings import (
    DEFAULT_EMBEDDING_DIMENSION,
    FAILURE_EMBEDDING_SCHEMA_VERSION,
    cosine_similarity,
    failure_embedding_payload,
    validate_embedding_dimension,
)

RECURRING_FAILURE_SCHEMA_VERSION = "recurring-failure-report-v1"
RECURRING_FAILURE_METHOD = "embedding_cluster_graph_context_v1"
DEFAULT_RECURRENCE_THRESHOLD = 0.55
DEFAULT_GRAPH_WEIGHT = 0.35
DEFAULT_MIN_CLUSTER_SIZE = 1


@dataclass(frozen=True, slots=True)
class _FailureContext:
    failure_id: str
    run_id: str
    features: frozenset[str]


def recurring_failure_report_payload(
    project_root: str | Path,
    *,
    include_text: bool = False,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    recurrence_threshold: float = DEFAULT_RECURRENCE_THRESHOLD,
    graph_weight: float = DEFAULT_GRAPH_WEIGHT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return deterministic recurring failure candidates for a project."""

    clean_dimension = validate_embedding_dimension(dimension)
    clean_similarity_threshold = validate_similarity_threshold(similarity_threshold)
    clean_recurrence_threshold = _validate_unit_interval(
        recurrence_threshold, field_name="recurrence_threshold"
    )
    clean_graph_weight = _validate_unit_interval(graph_weight, field_name="graph_weight")
    timestamp = generated_at or _utc_now_iso()
    embedding_payload = failure_embedding_payload(
        project_root,
        include_text=include_text,
        dimension=clean_dimension,
        generated_at=timestamp,
    )
    cluster_payload = failure_cluster_payload(
        project_root,
        include_text=include_text,
        dimension=clean_dimension,
        similarity_threshold=clean_similarity_threshold,
        generated_at=timestamp,
    )
    graph_document = build_graph_from_project(project_root)
    return recurring_failure_report_from_inputs(
        embedding_payload=embedding_payload,
        cluster_payload=cluster_payload,
        graph_document=graph_document,
        recurrence_threshold=clean_recurrence_threshold,
        graph_weight=clean_graph_weight,
        generated_at=timestamp,
    )


def recurring_failure_report_from_inputs(
    *,
    embedding_payload: dict[str, Any],
    cluster_payload: dict[str, Any],
    graph_document: GraphDocument,
    recurrence_threshold: float = DEFAULT_RECURRENCE_THRESHOLD,
    graph_weight: float = DEFAULT_GRAPH_WEIGHT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a recurring failure report from embeddings, clusters, and graph data."""

    _validate_embedding_payload(embedding_payload)
    _validate_cluster_payload(cluster_payload)
    clean_recurrence_threshold = _validate_unit_interval(
        recurrence_threshold, field_name="recurrence_threshold"
    )
    clean_graph_weight = _validate_unit_interval(graph_weight, field_name="graph_weight")
    embedding_weight = round(1.0 - clean_graph_weight, 6)
    records = sorted(
        (dict(record) for record in embedding_payload["records"]),
        key=lambda item: str(item["failure_id"]),
    )
    contexts = _failure_contexts(graph_document)
    seed_cluster_by_failure = _seed_cluster_by_failure(cluster_payload)
    links = _recurring_links(
        records=records,
        contexts=contexts,
        seed_cluster_by_failure=seed_cluster_by_failure,
        recurrence_threshold=clean_recurrence_threshold,
        embedding_weight=embedding_weight,
        graph_weight=clean_graph_weight,
    )
    components = _components(records, links)
    clusters = [
        _cluster_summary(
            index=index,
            members=members,
            records_by_failure={str(record["failure_id"]): record for record in records},
            contexts=contexts,
            seed_cluster_by_failure=seed_cluster_by_failure,
            links=links,
        )
        for index, members in enumerate(components, start=1)
    ]
    recurring_clusters = [cluster for cluster in clusters if cluster["recurring"]]
    warnings = _warnings(
        record_count=len(records),
        context_count=len(contexts),
        recurring_count=len(recurring_clusters),
    )
    return {
        "schema_version": RECURRING_FAILURE_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "method": RECURRING_FAILURE_METHOD,
        "embedding_schema_version": FAILURE_EMBEDDING_SCHEMA_VERSION,
        "cluster_schema_version": FAILURE_CLUSTER_SCHEMA_VERSION,
        "embedding_method": embedding_payload["method"],
        "dimension": embedding_payload["dimension"],
        "similarity_threshold": cluster_payload["similarity_threshold"],
        "recurrence_threshold": clean_recurrence_threshold,
        "embedding_weight": embedding_weight,
        "graph_weight": clean_graph_weight,
        "privacy_mode": embedding_payload["privacy_mode"],
        "include_text": bool(embedding_payload["include_text"]),
        "record_count": len(records),
        "cluster_count": len(clusters),
        "recurring_cluster_count": len(recurring_clusters),
        "clusters": clusters,
        "warnings": warnings,
        "privacy_flags": _privacy_flags(
            include_text=bool(embedding_payload["include_text"]),
            record_count=len(records),
        ),
        "algorithm": {
            "seed": "failure-cluster-v1 connected components",
            "embedding_similarity": "cosine similarity over local hashing vectors",
            "graph_similarity": "Jaccard overlap over metadata-only graph context",
            "combination": "weighted sum of embedding and graph similarity",
            "causal_claim": False,
            "human_review_required": True,
            "raw_text_in_output": False,
            "network": False,
            "optional_nlp_dependency_required": False,
            "database_mutation": False,
            "derived_graph_edges": False,
        },
    }


def _recurring_links(
    *,
    records: list[dict[str, Any]],
    contexts: dict[str, _FailureContext],
    seed_cluster_by_failure: dict[str, str],
    recurrence_threshold: float,
    embedding_weight: float,
    graph_weight: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    links: dict[tuple[str, str], dict[str, Any]] = {}
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            left_id = str(left["failure_id"])
            right_id = str(right["failure_id"])
            embedding_score = cosine_similarity(left["vector"], right["vector"])
            graph_score = _graph_similarity(
                contexts.get(left_id),
                contexts.get(right_id),
            )
            combined = round(embedding_score * embedding_weight + graph_score * graph_weight, 6)
            same_seed_cluster = seed_cluster_by_failure.get(
                left_id
            ) is not None and seed_cluster_by_failure.get(left_id) == seed_cluster_by_failure.get(
                right_id
            )
            if same_seed_cluster or combined >= recurrence_threshold:
                links[(left_id, right_id)] = {
                    "left_failure_id": left_id,
                    "right_failure_id": right_id,
                    "embedding_similarity": round(embedding_score, 6),
                    "graph_similarity": round(graph_score, 6),
                    "combined_similarity": combined,
                    "same_seed_cluster": same_seed_cluster,
                }
    return links


def _components(
    records: list[dict[str, Any]],
    links: dict[tuple[str, str], dict[str, Any]],
) -> list[list[str]]:
    failure_ids = [str(record["failure_id"]) for record in records]
    parent = {failure_id: failure_id for failure_id in failure_ids}

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

    for left_id, right_id in sorted(links):
        union(left_id, right_id)

    grouped: dict[str, list[str]] = {}
    for failure_id in failure_ids:
        grouped.setdefault(find(failure_id), []).append(failure_id)
    return [sorted(grouped[root]) for root in sorted(grouped)]


def _cluster_summary(
    *,
    index: int,
    members: list[str],
    records_by_failure: dict[str, dict[str, Any]],
    contexts: dict[str, _FailureContext],
    seed_cluster_by_failure: dict[str, str],
    links: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    member_records = [records_by_failure[failure_id] for failure_id in members]
    member_links = [
        link for pair, link in sorted(links.items()) if pair[0] in members and pair[1] in members
    ]
    embedding_scores = [float(link["embedding_similarity"]) for link in member_links]
    graph_scores = [float(link["graph_similarity"]) for link in member_links]
    combined_scores = [float(link["combined_similarity"]) for link in member_links]
    run_ids = sorted(
        {
            context.run_id
            for failure_id in members
            if (context := contexts.get(failure_id)) is not None and context.run_id
        }
    )
    source_clusters = sorted(
        {
            seed_cluster_by_failure[failure_id]
            for failure_id in members
            if failure_id in seed_cluster_by_failure
        }
    )
    dominant = _dominant_signals(member_records)
    size = len(members)
    recurring = size >= 2
    return {
        "recurring_cluster_id": f"recurring_failure_{index:03d}",
        "recurring": recurring,
        "claim": (
            "recurring_failure_candidate_not_root_cause"
            if recurring
            else "single_failure_no_recurrence_claim"
        ),
        "size": size,
        "failure_ids": members,
        "run_ids": run_ids,
        "prototype_failure_id": _prototype_failure_id(member_records),
        "source_cluster_ids": source_clusters,
        "score": {
            "mean_embedding_similarity": _mean_or_zero(embedding_scores),
            "mean_graph_similarity": _mean_or_zero(graph_scores),
            "mean_combined_similarity": _mean_or_zero(combined_scores),
            "max_combined_similarity": round(max(combined_scores), 6) if combined_scores else 0.0,
        },
        "dominant_signals": dominant,
        "evidence": {
            "failure_ids": members,
            "run_ids": run_ids,
            "pair_links": member_links[:20],
        },
        "explanation": _cluster_explanation(size=size, dominant=dominant),
        "human_review_required": True,
    }


def _failure_contexts(document: GraphDocument) -> dict[str, _FailureContext]:
    nodes_by_id = {node.node_id: node for node in document.nodes}
    outgoing: dict[str, list[GraphEdge]] = {}
    incoming: dict[str, list[GraphEdge]] = {}
    for edge in document.edges:
        outgoing.setdefault(edge.source, []).append(edge)
        incoming.setdefault(edge.target, []).append(edge)

    contexts: dict[str, _FailureContext] = {}
    for node in document.nodes:
        if node.node_type is not NodeType.FAILURE:
            continue
        failure_id = _failure_id_from_node(node)
        run_id = str(node.attributes.get("run_id") or "")
        features = set(_failure_node_features(node))
        for edge in outgoing.get(node.node_id, ()):
            features.add(f"edge:{edge.edge_type.value}")
            features.add(f"target_type:{_node_type_value(nodes_by_id.get(edge.target))}")
            if edge.edge_type is EdgeType.OBSERVED_IN:
                run_node = nodes_by_id.get(edge.target)
                if run_node is not None:
                    run_id = edge.target
                    features.update(_run_context_features(run_node, outgoing, nodes_by_id))
        contexts[failure_id] = _FailureContext(
            failure_id=failure_id,
            run_id=run_id,
            features=frozenset(feature for feature in features if feature),
        )
    return contexts


def _run_context_features(
    run_node: GraphNode,
    outgoing: dict[str, list[GraphEdge]],
    nodes_by_id: dict[str, GraphNode],
) -> set[str]:
    features = {
        f"run_status:{run_node.attributes.get('status')}",
        f"run_exit_code:{run_node.attributes.get('exit_code')}",
        f"experiment:{run_node.attributes.get('experiment_id')}",
        f"run_has_config:{run_node.attributes.get('has_config')}",
    }
    for edge in outgoing.get(run_node.node_id, ()):
        target = nodes_by_id.get(edge.target)
        features.add(f"run_edge:{edge.edge_type.value}")
        features.add(f"run_target_type:{_node_type_value(target)}")
        if edge.edge_type is EdgeType.USES_CONFIG:
            features.add(f"config:{edge.target}")
        elif edge.edge_type is EdgeType.PRODUCES_METRIC and target is not None:
            features.add(f"metric:{target.attributes.get('metric_name')}")
        elif edge.edge_type is EdgeType.PRODUCES_ARTIFACT:
            features.add(f"artifact:{edge.target}")
    return {feature for feature in features if not feature.endswith(":None")}


def _failure_node_features(node: GraphNode) -> set[str]:
    features = {
        f"error_type:{node.attributes.get('error_type')}",
        f"severity:{node.attributes.get('severity')}",
        f"source:{node.attributes.get('source')}",
    }
    for tag in node.attributes.get("tags", ()):
        features.add(f"tag:{tag}")
    return {feature for feature in features if not feature.endswith(":None")}


def _graph_similarity(left: _FailureContext | None, right: _FailureContext | None) -> float:
    if left is None or right is None:
        return 0.0
    union = left.features | right.features
    if not union:
        return 0.0
    return round(len(left.features & right.features) / len(union), 10)


def _seed_cluster_by_failure(cluster_payload: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for cluster in cluster_payload["clusters"]:
        cluster_id = str(cluster["cluster_id"])
        for failure_id in cluster["failure_ids"]:
            result[str(failure_id)] = cluster_id
    return result


def _dominant_signals(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "error_type": _counter_dict(record["error_type"] for record in records),
        "severity": _counter_dict(record["severity"] for record in records),
        "source": _counter_dict(record["source"] for record in records),
        "tag": _counter_dict(tag for record in records for tag in record.get("tags", ())),
    }


def _prototype_failure_id(records: list[dict[str, Any]]) -> str:
    if len(records) == 1:
        return str(records[0]["failure_id"])
    scores: list[tuple[float, str]] = []
    for record in records:
        failure_id = str(record["failure_id"])
        similarities = [
            cosine_similarity(record["vector"], other["vector"])
            for other in records
            if other is not record
        ]
        average = sum(similarities) / len(similarities) if similarities else 0.0
        scores.append((average, failure_id))
    return sorted(scores, key=lambda item: (-item[0], item[1]))[0][1]


def _counter_dict(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values if str(value).strip()).items()))


def _cluster_explanation(*, size: int, dominant: dict[str, dict[str, int]]) -> str:
    if size < 2:
        return "Single failure retained as a singleton; no recurrence claim is made."
    tag_counts = dominant.get("tag", {})
    if tag_counts:
        tag = sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return (
            f"Recurring candidate groups {size} failures with shared embedding/graph "
            f"evidence and dominant tag `{tag}`; human review is required."
        )
    return (
        f"Recurring candidate groups {size} failures with shared embedding/graph "
        "evidence; human review is required."
    )


def _warnings(*, record_count: int, context_count: int, recurring_count: int) -> list[str]:
    warnings: list[str] = []
    if record_count == 0:
        warnings.append("No confirmed failure records were found.")
    if context_count < record_count:
        warnings.append("Some failures were missing graph context; graph similarity was reduced.")
    if record_count and recurring_count == 0:
        warnings.append("No recurring failure candidates met the current evidence threshold.")
    return warnings


def _privacy_flags(*, include_text: bool, record_count: int) -> list[dict[str, Any]]:
    if not record_count:
        return []
    if include_text:
        return [
            {
                "code": "text_derived_similarity",
                "severity": "warning",
                "message": (
                    "Similarity scores may be derived from free text; treat output as private."
                ),
            }
        ]
    return [
        {
            "code": "metadata_only_similarity",
            "severity": "info",
            "message": "Recurring detection used structured metadata and graph context by default.",
        }
    ]


def _validate_embedding_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != FAILURE_EMBEDDING_SCHEMA_VERSION:
        raise PmemValidationError(
            "Recurring failure detection requires failure-embedding-v1 input."
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise PmemValidationError("Embedding payload must include a records list.")
    for record in records:
        if not isinstance(record, dict):
            raise PmemValidationError("Embedding records must be objects.")
        required = {"failure_id", "run_id", "error_type", "severity", "source", "tags", "vector"}
        if any(key not in record for key in required):
            raise PmemValidationError("Embedding record is missing recurring-analysis fields.")


def _validate_cluster_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != FAILURE_CLUSTER_SCHEMA_VERSION:
        raise PmemValidationError("Recurring failure detection requires failure-cluster-v1 input.")
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        raise PmemValidationError("Cluster payload must include a clusters list.")
    for cluster in clusters:
        if not isinstance(cluster, dict) or not isinstance(cluster.get("failure_ids"), list):
            raise PmemValidationError("Cluster entries must include failure_ids.")


def _validate_unit_interval(value: float, *, field_name: str) -> float:
    number = round(float(value), 6)
    if number < 0.0 or number > 1.0:
        raise PmemValidationError(f"{field_name} must be between 0.0 and 1.0.")
    return number


def _failure_id_from_node(node: GraphNode) -> str:
    prefix = "failure:"
    return node.node_id[len(prefix) :] if node.node_id.startswith(prefix) else node.node_id


def _node_type_value(node: GraphNode | None) -> str:
    return node.node_type.value if node is not None else "missing"


def _mean_or_zero(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
