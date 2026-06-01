"""Human-reviewable failure pattern reports for failure-analysis layer failure pattern and summary.

Pattern reports are deterministic audit aids. They summarize failure clustering clusters with
metadata-driven labels and explicitly avoid causal/root-cause claims.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.errors import PmemValidationError
from pmem.services.failure_clustering import (
    DEFAULT_SIMILARITY_THRESHOLD,
    FAILURE_CLUSTER_SCHEMA_VERSION,
    failure_cluster_payload,
)
from pmem.services.failure_embeddings import DEFAULT_EMBEDDING_DIMENSION
from pmem.services.failure_exports import list_failure_records

FAILURE_PATTERN_SCHEMA_VERSION = "failure-pattern-report-v1"
FAILURE_PATTERN_METHOD = "cluster_metadata_heuristic_v1"
FAILURE_ANALYSIS_SUMMARY_SCHEMA_VERSION = "failure-analysis-summary-v1"
DEFAULT_PATTERN_LIMIT = 5

_TEXT_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_SENSITIVE_TEXT_TOKENS = {
    "api",
    "apikey",
    "api_key",
    "auth",
    "credential",
    "credentials",
    "key",
    "password",
    "private",
    "secret",
    "token",
}
_STOP_WORDS = {
    "a",
    "after",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def failure_pattern_report_payload(
    project_root: str | Path,
    *,
    include_text: bool = False,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    generated_at: str | None = None,
    max_terms: int = DEFAULT_PATTERN_LIMIT,
) -> dict[str, Any]:
    """Build a deterministic human-reviewable pattern candidate report."""

    timestamp = generated_at or _utc_now_iso()
    clusters = failure_cluster_payload(
        project_root,
        include_text=include_text,
        dimension=dimension,
        similarity_threshold=similarity_threshold,
        generated_at=timestamp,
    )
    text_terms_by_cluster: dict[str, list[dict[str, object]]] = {}
    if include_text and clusters["cluster_count"]:
        records = {
            str(record["id"]): record
            for record in list_failure_records(project_root, include_text=True)
        }
        text_terms_by_cluster = _text_terms_by_cluster(
            clusters["clusters"],
            records=records,
            max_terms=max_terms,
        )
    return failure_pattern_report_from_clusters(
        clusters,
        generated_at=timestamp,
        text_terms_by_cluster=text_terms_by_cluster,
        max_terms=max_terms,
    )


def failure_pattern_report_from_clusters(
    cluster_payload: dict[str, Any],
    *,
    generated_at: str | None = None,
    text_terms_by_cluster: dict[str, list[dict[str, object]]] | None = None,
    max_terms: int = DEFAULT_PATTERN_LIMIT,
) -> dict[str, Any]:
    """Create pattern candidates from a failure clustering cluster payload."""

    _validate_cluster_payload(cluster_payload)
    clusters = list(cluster_payload["clusters"])
    patterns = [
        _pattern_for_cluster(
            cluster,
            total_records=int(cluster_payload["record_count"]),
            text_terms=(text_terms_by_cluster or {}).get(str(cluster["cluster_id"]), []),
            max_terms=max_terms,
        )
        for cluster in clusters
    ]
    ordered_patterns = sorted(
        patterns,
        key=lambda item: (
            -int(item["evidence_count"]),
            str(item["heuristic_label"]),
            str(item["cluster_id"]),
        ),
    )
    for index, pattern in enumerate(ordered_patterns, start=1):
        pattern["pattern_id"] = f"pattern_{index:03d}"

    include_text = bool(cluster_payload["include_text"])
    return {
        "schema_version": FAILURE_PATTERN_SCHEMA_VERSION,
        "generated_at": generated_at or str(cluster_payload["generated_at"]),
        "method": FAILURE_PATTERN_METHOD,
        "source_schema_version": FAILURE_CLUSTER_SCHEMA_VERSION,
        "embedding_schema_version": cluster_payload["embedding_schema_version"],
        "embedding_method": cluster_payload["embedding_method"],
        "dimension": cluster_payload["dimension"],
        "similarity_threshold": cluster_payload["similarity_threshold"],
        "privacy_mode": cluster_payload["privacy_mode"],
        "include_text": include_text,
        "record_count": cluster_payload["record_count"],
        "cluster_count": cluster_payload["cluster_count"],
        "pattern_count": len(ordered_patterns),
        "patterns": ordered_patterns,
        "warnings": _report_warnings(
            include_text=include_text, pattern_count=len(ordered_patterns)
        ),
        "algorithm": {
            "labeling": "metadata_first_heuristic",
            "causal_claim": False,
            "human_review_required": True,
            "raw_text_in_output": False,
            "network": False,
            "complexity": (
                "failure pattern labeling is O(C log C + N); "
                "failure clustering is pairwise O(N^2 * D)."
            ),
        },
        "privacy_flags": _privacy_flags(
            include_text=include_text, pattern_count=len(ordered_patterns)
        ),
    }


def failure_analysis_summary_payload(
    project_root: str | Path,
    *,
    include_text: bool = False,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    generated_at: str | None = None,
    max_patterns: int = 3,
) -> dict[str, Any]:
    """Return a compact failure analysis summary failure analysis summary for CLI UX."""

    timestamp = generated_at or _utc_now_iso()
    report = failure_pattern_report_payload(
        project_root,
        include_text=include_text,
        dimension=dimension,
        similarity_threshold=similarity_threshold,
        generated_at=timestamp,
    )
    top_patterns = [
        {
            "pattern_id": pattern["pattern_id"],
            "cluster_id": pattern["cluster_id"],
            "heuristic_label": pattern["heuristic_label"],
            "evidence_count": pattern["evidence_count"],
            "heuristic_score": pattern["heuristic_score"],
            "review_recommendation": pattern["review_recommendation"],
        }
        for pattern in report["patterns"][:max_patterns]
    ]
    return {
        "schema_version": FAILURE_ANALYSIS_SUMMARY_SCHEMA_VERSION,
        "generated_at": timestamp,
        "source_schema_version": report["schema_version"],
        "privacy_mode": report["privacy_mode"],
        "include_text": report["include_text"],
        "record_count": report["record_count"],
        "cluster_count": report["cluster_count"],
        "pattern_count": report["pattern_count"],
        "status": _summary_status(
            record_count=report["record_count"], pattern_count=report["pattern_count"]
        ),
        "top_patterns": top_patterns,
        "warnings": report["warnings"],
        "next_actions": _next_actions(report),
        "artifact_source": "generated_on_demand",
        "human_review_required": True,
    }


def _pattern_for_cluster(
    cluster: dict[str, Any],
    *,
    total_records: int,
    text_terms: list[dict[str, object]],
    max_terms: int,
) -> dict[str, Any]:
    size = int(cluster["size"])
    tag = _dominant_item(cluster.get("tag_counts", {}))
    error_type = _dominant_item(cluster.get("error_type_counts", {}))
    severity = _dominant_item(cluster.get("severity_counts", {}))
    source = _dominant_item(cluster.get("source_counts", {}))
    label_basis = _label_basis(tag=tag, error_type=error_type, severity=severity, source=source)
    evidence_count = max(size, 0)
    label_count_value = label_basis["count"]
    label_count = label_count_value if isinstance(label_count_value, int) else 0
    support = label_count / evidence_count if evidence_count else 0.0
    recurrence = min(evidence_count / max(total_records, 1), 1.0)
    heuristic_score = round(min(1.0, support * 0.75 + recurrence * 0.25), 4)
    cluster_id = str(cluster["cluster_id"])
    label = f"{_humanize(str(label_basis['value']))} pattern candidate"
    if evidence_count == 1:
        label = f"single-evidence {label}"

    top_terms = text_terms[:max_terms]
    return {
        "pattern_id": "",
        "cluster_id": cluster_id,
        "heuristic_label": label,
        "label_source": str(label_basis["source"]),
        "heuristic_score": heuristic_score,
        "evidence_count": evidence_count,
        "failure_ids": [str(item) for item in cluster["failure_ids"]],
        "prototype_failure_id": str(cluster["prototype_failure_id"]),
        "dominant_signals": {
            "tag": _signal_payload(tag),
            "error_type": _signal_payload(error_type),
            "severity": _signal_payload(severity),
            "source": _signal_payload(source),
        },
        "counts": {
            "error_type": dict(cluster.get("error_type_counts", {})),
            "severity": dict(cluster.get("severity_counts", {})),
            "source": dict(cluster.get("source_counts", {})),
            "tag": dict(cluster.get("tag_counts", {})),
        },
        "text_terms": top_terms,
        "explanation": _explanation(
            label_basis=label_basis, evidence_count=evidence_count, cluster_id=cluster_id
        ),
        "review_recommendation": (
            f"Review {cluster_id} before treating this label as a confirmed finding."
        ),
    }


def _validate_cluster_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != FAILURE_CLUSTER_SCHEMA_VERSION:
        raise PmemValidationError("Pattern report requires failure-cluster-v1 input.")
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        raise PmemValidationError("Cluster payload must include a clusters list.")
    required_top_level = {
        "generated_at",
        "embedding_schema_version",
        "embedding_method",
        "dimension",
        "similarity_threshold",
        "privacy_mode",
        "include_text",
        "record_count",
        "cluster_count",
    }
    missing = sorted(key for key in required_top_level if key not in payload)
    if missing:
        raise PmemValidationError("Cluster payload is missing required pattern-report fields.")
    for cluster in clusters:
        if not isinstance(cluster, dict):
            raise PmemValidationError("Cluster entries must be objects.")
        required_cluster = {
            "cluster_id",
            "size",
            "failure_ids",
            "prototype_failure_id",
            "error_type_counts",
            "severity_counts",
            "source_counts",
            "tag_counts",
        }
        if any(key not in cluster for key in required_cluster):
            raise PmemValidationError("Cluster entry is missing pattern-report fields.")
        if not isinstance(cluster["failure_ids"], list):
            raise PmemValidationError("Cluster failure_ids must be a list.")


def _dominant_item(counts: object) -> tuple[str, int] | None:
    if not isinstance(counts, dict) or not counts:
        return None
    normalized = [(str(key), int(value)) for key, value in counts.items() if int(value) > 0]
    if not normalized:
        return None
    return sorted(normalized, key=lambda item: (-item[1], item[0]))[0]


def _label_basis(
    *,
    tag: tuple[str, int] | None,
    error_type: tuple[str, int] | None,
    severity: tuple[str, int] | None,
    source: tuple[str, int] | None,
) -> dict[str, object]:
    if tag is not None:
        return {"source": "tag", "value": tag[0], "count": tag[1]}
    if error_type is not None:
        return {"source": "error_type", "value": error_type[0], "count": error_type[1]}
    if severity is not None:
        return {"source": "severity", "value": severity[0], "count": severity[1]}
    if source is not None:
        return {"source": "source", "value": source[0], "count": source[1]}
    return {"source": "fallback", "value": "unlabeled", "count": 0}


def _text_terms_by_cluster(
    clusters: list[dict[str, Any]],
    *,
    records: dict[str, dict[str, Any]],
    max_terms: int,
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for cluster in clusters:
        counter: Counter[str] = Counter()
        for failure_id in cluster["failure_ids"]:
            record = records.get(str(failure_id))
            if record is None:
                continue
            counter.update(_safe_text_tokens(record))
        result[str(cluster["cluster_id"])] = [
            {"term": term, "count": count}
            for term, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
                :max_terms
            ]
        ]
    return result


def _safe_text_tokens(record: dict[str, Any]) -> list[str]:
    raw = " ".join(
        str(record.get(field) or "") for field in ("description", "root_cause", "lesson")
    )
    tokens: list[str] = []
    for match in _TEXT_TOKEN_RE.finditer(raw.casefold()):
        token = match.group(0)
        if len(token) < 3 or token in _STOP_WORDS or token in _SENSITIVE_TEXT_TOKENS:
            continue
        if any(secret in token for secret in _SENSITIVE_TEXT_TOKENS):
            continue
        tokens.append(token)
    return tokens


def _signal_payload(item: tuple[str, int] | None) -> dict[str, object] | None:
    if item is None:
        return None
    return {"value": item[0], "count": item[1]}


def _explanation(*, label_basis: dict[str, object], evidence_count: int, cluster_id: str) -> str:
    return (
        f"Heuristic label selected from {label_basis['source']}={label_basis['value']} "
        f"covering {label_basis['count']}/{evidence_count} record(s) in {cluster_id}."
    )


def _humanize(value: str) -> str:
    cleaned = value.replace("_", " ").replace("-", " ").strip()
    return " ".join(cleaned.split()) or "unlabeled"


def _summary_status(*, record_count: int, pattern_count: int) -> str:
    if record_count == 0:
        return "no_failures"
    if pattern_count == 0:
        return "no_patterns"
    return "pattern_candidates_available"


def _next_actions(report: dict[str, Any]) -> list[str]:
    if report["record_count"] == 0:
        return ["Log confirmed failures before running pattern analysis."]
    if report["pattern_count"] == 0:
        return ["Review failure records; no pattern candidates were generated."]
    return [
        "Review top pattern candidates before treating labels as findings.",
        "Inspect cluster evidence counts and dominant signals.",
        "Use --include-text --confirm only if it is safe to derive signals from raw failure text.",
    ]


def _report_warnings(*, include_text: bool, pattern_count: int) -> list[str]:
    warnings = [
        "Pattern labels are heuristic candidates, not confirmed root causes.",
        "Human review is required before using a pattern in reports or decisions.",
    ]
    if include_text:
        warnings.append(
            "This report is derived from raw failure text and should be treated as private."
        )
    if pattern_count == 0:
        warnings.append("No pattern candidates were generated.")
    return warnings


def _privacy_flags(*, include_text: bool, pattern_count: int) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = [
        {
            "code": "heuristic_labels_only",
            "severity": "info",
            "message": "Pattern labels are audit aids and must be human-reviewed.",
        }
    ]
    if include_text:
        flags.append(
            {
                "code": "derived_from_free_text",
                "severity": "warning",
                "message": (
                    "Pattern report used confirmed raw failure text; treat output as private."
                ),
            }
        )
    elif pattern_count:
        flags.append(
            {
                "code": "raw_text_excluded",
                "severity": "info",
                "message": "Pattern report used structured metadata only.",
            }
        )
    return flags


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
