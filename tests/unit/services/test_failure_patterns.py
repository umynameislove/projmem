"""failure pattern and summary tests for privacy-safe failure pattern reports."""

from __future__ import annotations

import json
import sys

import pytest

from pmem.errors import PmemValidationError
from pmem.services.failure_logging import log_failure
from pmem.services.failure_patterns import (
    FAILURE_ANALYSIS_SUMMARY_SCHEMA_VERSION,
    FAILURE_PATTERN_SCHEMA_VERSION,
    failure_analysis_summary_payload,
    failure_pattern_report_from_clusters,
    failure_pattern_report_payload,
)
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command


def test_pattern_report_handles_empty_project(tmp_path) -> None:
    """failure pattern report should produce a stable empty pattern report."""

    init_project(tmp_path, project_name="patterns-demo")

    payload = failure_pattern_report_payload(
        tmp_path,
        generated_at="2026-01-01T00:00:00Z",
        dimension=32,
    )

    assert payload["schema_version"] == FAILURE_PATTERN_SCHEMA_VERSION
    assert payload["record_count"] == 0
    assert payload["pattern_count"] == 0
    assert payload["patterns"] == []
    assert payload["algorithm"]["causal_claim"] is False


def test_pattern_report_labels_single_and_recurring_clusters(tmp_path) -> None:
    """failure pattern report labels are metadata-first audit candidates, not root-cause claims."""

    payload = failure_pattern_report_from_clusters(
        _synthetic_cluster_payload(
            clusters=[
                {
                    "cluster_id": "cluster_001",
                    "size": 2,
                    "failure_ids": ["failure_a", "failure_b"],
                    "prototype_failure_id": "failure_a",
                    "error_type_counts": {"MetricRegression": 2},
                    "severity_counts": {"high": 2},
                    "source_counts": {"user_confirmed": 2},
                    "tag_counts": {"data_quality": 2},
                },
                {
                    "cluster_id": "cluster_002",
                    "size": 1,
                    "failure_ids": ["failure_c"],
                    "prototype_failure_id": "failure_c",
                    "error_type_counts": {"OOM": 1},
                    "severity_counts": {"medium": 1},
                    "source_counts": {"user_confirmed": 1},
                    "tag_counts": {"oom": 1},
                },
            ]
        ),
        generated_at="2026-01-01T00:00:00Z",
    )
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["pattern_count"] == 2
    assert payload["patterns"][0]["pattern_id"] == "pattern_001"
    assert payload["patterns"][0]["heuristic_label"] == "data quality pattern candidate"
    assert payload["patterns"][0]["label_source"] == "tag"
    assert payload["patterns"][0]["evidence_count"] == 2
    assert payload["patterns"][0]["review_recommendation"].startswith("Review cluster_")
    assert "SECRET token" not in raw_json
    assert "true root cause" not in raw_json.casefold()


def test_pattern_report_is_deterministic(tmp_path) -> None:
    """Frozen timestamp and same DB should produce equal reports."""

    _create_failure(tmp_path, error_type="ConfigError", tag="config error")

    first = failure_pattern_report_payload(
        tmp_path,
        generated_at="2026-01-01T00:00:00Z",
        dimension=64,
        similarity_threshold=0.4,
    )
    second = failure_pattern_report_payload(
        tmp_path,
        generated_at="2026-01-01T00:00:00Z",
        dimension=64,
        similarity_threshold=0.4,
    )

    assert first == second


def test_pattern_report_default_excludes_raw_text(tmp_path) -> None:
    """failure pattern report default output must not expose failure free text."""

    _create_failure(tmp_path, error_type="MetricRegression", tag="data quality")

    payload = failure_pattern_report_payload(
        tmp_path,
        generated_at="2026-01-01T00:00:00Z",
        dimension=64,
    )
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["privacy_mode"] == "structured_only"
    assert payload["include_text"] is False
    assert "description" not in raw_json
    assert "SECRET token" not in raw_json
    assert "private root cause" not in raw_json


def test_pattern_report_rejects_malformed_cluster_payload() -> None:
    """failure pattern report should fail closed on malformed cluster artifacts."""

    with pytest.raises(PmemValidationError, match="failure-cluster-v1"):
        failure_pattern_report_from_clusters({"schema_version": "wrong"})

    with pytest.raises(PmemValidationError, match="clusters list"):
        failure_pattern_report_from_clusters(_synthetic_cluster_payload(clusters=None))


def test_pattern_report_scales_over_many_cluster_summaries() -> None:
    """Pattern labeling over cluster summaries stays linear-ish."""

    clusters = []
    for index in range(60):
        clusters.append(
            {
                "cluster_id": f"cluster_{index + 1:03d}",
                "size": 1,
                "failure_ids": [f"failure_{index:03d}"],
                "prototype_failure_id": f"failure_{index:03d}",
                "error_type_counts": {"MetricRegression": 1},
                "severity_counts": {"medium": 1},
                "source_counts": {"user_confirmed": 1},
                "tag_counts": {"data_quality": 1},
            }
        )
    payload = failure_pattern_report_from_clusters(
        _synthetic_cluster_payload(clusters=clusters),
        generated_at="2026-01-01T00:00:00Z",
    )

    assert payload["pattern_count"] == 60
    assert payload["patterns"][0]["pattern_id"] == "pattern_001"
    assert payload["patterns"][-1]["pattern_id"] == "pattern_060"


def test_failure_analysis_summary_payload_reports_next_actions(tmp_path) -> None:
    """failure analysis summary summary should surface audit next actions without raw text."""

    _create_failure(tmp_path, error_type="MetricRegression", tag="data quality")

    payload = failure_analysis_summary_payload(
        tmp_path,
        generated_at="2026-01-01T00:00:00Z",
        dimension=64,
    )
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == FAILURE_ANALYSIS_SUMMARY_SCHEMA_VERSION
    assert payload["status"] == "pattern_candidates_available"
    assert payload["top_patterns"]
    assert payload["human_review_required"] is True
    assert "Review top pattern candidates" in payload["next_actions"][0]
    assert "SECRET token" not in raw_json


def test_safe_text_tokens_filters_sensitive_substrings() -> None:
    """L374: tokens containing a sensitive word as substring must be filtered.

    The second guard catches e.g. 'tokenizer' (contains 'token'),
    'authenticate' (contains 'auth'), 'private_data' (contains 'private').
    This prevents subtle info-leakage where a secret-adjacent word slips
    the exact-match check but would still expose context.
    """
    from pmem.services.failure_patterns import _safe_text_tokens

    record = {
        "description": "tokenizer_crash authenticate_routine normal_word",
        "root_cause": "private_config_missing",
        "lesson": "rotate_credentials_annually",
    }
    tokens = _safe_text_tokens(record)

    # Exact-match sensitive tokens already filtered by L371; substring tested here:
    assert "tokenizer_crash" not in tokens  # "token" in "tokenizer_crash"
    assert "authenticate_routine" not in tokens  # "auth" in "authenticate_routine"
    assert "private_config_missing" not in tokens  # "private" in it
    assert "rotate_credentials_annually" not in tokens  # "credential" in it
    # Non-sensitive token should survive
    assert "normal_word" in tokens


def test_pattern_report_label_basis_uses_priority_order() -> None:
    """L332-338: label_basis must prefer tag > error_type > severity > source > fallback."""

    # 1. Only error_type available (no tags)
    no_tag_cluster = _synthetic_cluster_payload(
        clusters=[
            {
                "cluster_id": "cluster_001",
                "size": 1,
                "failure_ids": ["f1"],
                "prototype_failure_id": "f1",
                "error_type_counts": {"OOM": 1},
                "severity_counts": {"high": 1},
                "source_counts": {"user_confirmed": 1},
                "tag_counts": {},  # no tags
            }
        ]
    )
    payload = failure_pattern_report_from_clusters(
        no_tag_cluster, generated_at="2026-01-01T00:00:00Z"
    )
    assert payload["patterns"][0]["label_source"] == "error_type"
    assert "oom" in payload["patterns"][0]["heuristic_label"].lower()

    # 2. Only severity available (no tags, no error_type)
    no_tag_no_type_cluster = _synthetic_cluster_payload(
        clusters=[
            {
                "cluster_id": "cluster_001",
                "size": 1,
                "failure_ids": ["f1"],
                "prototype_failure_id": "f1",
                "error_type_counts": {},
                "severity_counts": {"critical": 1},
                "source_counts": {},
                "tag_counts": {},
            }
        ]
    )
    payload2 = failure_pattern_report_from_clusters(
        no_tag_no_type_cluster, generated_at="2026-01-01T00:00:00Z"
    )
    assert payload2["patterns"][0]["label_source"] == "severity"

    # 3. Fallback — all counts empty
    fallback_cluster = _synthetic_cluster_payload(
        clusters=[
            {
                "cluster_id": "cluster_001",
                "size": 1,
                "failure_ids": ["f1"],
                "prototype_failure_id": "f1",
                "error_type_counts": {},
                "severity_counts": {},
                "source_counts": {},
                "tag_counts": {},
            }
        ]
    )
    payload3 = failure_pattern_report_from_clusters(
        fallback_cluster, generated_at="2026-01-01T00:00:00Z"
    )
    assert payload3["patterns"][0]["label_source"] == "fallback"
    assert "unlabeled" in payload3["patterns"][0]["heuristic_label"]


def test_pattern_report_rejects_missing_top_level_fields() -> None:
    """L294: cluster payload missing required top-level fields must be rejected."""

    # Has correct schema_version and clusters list, but is missing required fields
    # like generated_at, embedding_schema_version, dimension, etc.
    with pytest.raises(PmemValidationError, match="missing required"):
        failure_pattern_report_from_clusters(
            {
                "schema_version": "failure-cluster-v1",
                "clusters": [],
            }
        )


def test_pattern_report_rejects_cluster_missing_required_field() -> None:
    """L309: cluster entry missing a required field (e.g. error_type_counts) must fail."""

    payload = _synthetic_cluster_payload(
        clusters=[
            {
                "cluster_id": "cluster_001",
                "size": 1,
                "failure_ids": ["f1"],
                "prototype_failure_id": "f1",
                # error_type_counts is intentionally omitted
                "severity_counts": {"high": 1},
                "source_counts": {"user_confirmed": 1},
                "tag_counts": {"data_quality": 1},
            }
        ]
    )
    with pytest.raises(PmemValidationError, match="missing pattern-report fields"):
        failure_pattern_report_from_clusters(payload)


def test_analysis_summary_status_no_patterns_when_no_clusters() -> None:
    """L401/L409: status='no_patterns' and correct next_action when clusters=0 but records>0."""

    # Synthetic payload: has records but no clusters (and thus no patterns)
    cluster_payload = _synthetic_cluster_payload(clusters=[])
    # Override record_count to simulate "failures exist but nothing clustered"
    cluster_payload = dict(cluster_payload)
    cluster_payload["record_count"] = 2

    payload = failure_pattern_report_from_clusters(
        cluster_payload, generated_at="2026-01-01T00:00:00Z"
    )

    assert payload["pattern_count"] == 0

    # Test _summary_status and _next_actions directly for the no_patterns state
    from pmem.services.failure_patterns import _next_actions, _summary_status

    assert _summary_status(record_count=2, pattern_count=0) == "no_patterns"
    next_acts = _next_actions({"record_count": 2, "pattern_count": 0})
    assert "no pattern candidates" in next_acts[0].lower()


def _create_failure(tmp_path, *, error_type: str, tag: str) -> None:
    if not (tmp_path / ".pmem").exists():
        init_project(tmp_path, project_name="patterns-demo")
    run_result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type=error_type,
        description="SECRET token failure text should stay private.",
        root_cause="Synthetic private root cause",
        lesson="Synthetic private lesson",
        severity="high",
        tags=(tag,),
        source="user_confirmed",
    )


def _synthetic_cluster_payload(*, clusters) -> dict[str, object]:
    return {
        "schema_version": "failure-cluster-v1",
        "generated_at": "2026-01-01T00:00:00Z",
        "embedding_schema_version": "failure-embedding-v1",
        "embedding_method": "local_hashing_tf_l2",
        "dimension": 32,
        "similarity_threshold": 0.42,
        "privacy_mode": "structured_only",
        "include_text": False,
        "record_count": len(clusters) if isinstance(clusters, list) else 0,
        "cluster_count": len(clusters) if isinstance(clusters, list) else 0,
        "clusters": clusters,
    }
