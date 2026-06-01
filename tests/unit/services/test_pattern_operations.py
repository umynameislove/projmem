"""pattern CLI pattern operation service tests."""

from __future__ import annotations

from pmem.services import pattern_operations
from pmem.services.pattern_operations import pattern_list_payload

NOW = "2026-05-30T02:00:00Z"


def test_pattern_list_payload_summarizes_all_d50_d54_reports(monkeypatch, tmp_path) -> None:
    """pattern CLI list output should summarize every pattern-mining layer pattern API safely."""

    monkeypatch.setattr(
        pattern_operations,
        "config_failure_correlation_payload",
        lambda *args, **kwargs: _report(
            "config-failure-correlation-v1",
            "config_failure_correlation",
            candidate_count=2,
        ),
    )
    monkeypatch.setattr(
        pattern_operations,
        "dataset_failure_correlation_payload",
        lambda *args, **kwargs: _report(
            "dataset-failure-correlation-v1",
            "dataset_failure_correlation",
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(
        pattern_operations,
        "recurring_failure_report_payload",
        lambda *args, **kwargs: _report(
            "recurring-failure-report-v1",
            "recurring_failure_detection",
            recurring_cluster_count=3,
        ),
    )
    monkeypatch.setattr(
        pattern_operations,
        "temporal_analysis_payload",
        lambda *args, **kwargs: _report(
            "temporal-analysis-v1",
            "temporal_metric_drift_decision_shift",
            drift={"claim": "temporal_drift_candidate_not_causal"},
            decision_impact_candidate_count=1,
        ),
    )
    monkeypatch.setattr(
        pattern_operations,
        "anomaly_detection_payload",
        lambda *args, **kwargs: _report(
            "anomaly-detection-v1",
            "metric_outlier_reproducibility_screening",
            metric_outlier_count=1,
            reproducibility_candidate_count=1,
        ),
    )

    payload = pattern_list_payload(tmp_path, generated_at=NOW)

    assert payload["schema_version"] == "pattern-list-result-v1"
    assert payload["candidate_count"] == 10
    assert payload["raw_text_in_output"] is False
    assert payload["database_mutation"] is False
    assert [item["pattern"] for item in payload["patterns"]] == [
        "anomalies",
        "config_failure",
        "dataset_failure",
        "recurring_failures",
        "temporal",
    ]
    assert all(item["causal_claim"] is False for item in payload["patterns"])


def test_pattern_summary_handles_future_empty_reports() -> None:
    """Unknown future pattern kinds should degrade to a no-candidate summary."""

    summary = pattern_operations._pattern_summary(  # noqa: SLF001
        "future_pattern",
        _report("future-pattern-v1", "future_pattern"),
    )

    assert summary["candidate_count"] == 0
    assert summary["status"] == "no_candidates"


def _report(schema_version: str, scope: str, **extra) -> dict:
    return {
        "schema_version": schema_version,
        "scope": scope,
        "privacy_mode": "metadata_only",
        "raw_text_in_output": False,
        "causal_claim": False,
        "warnings": [],
        **extra,
    }
