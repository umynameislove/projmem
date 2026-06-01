"""pattern CLI privacy-safe pattern CLI service operations.

This layer keeps Typer rendering out of the pattern-analysis pattern modules. It wires
the existing local-only analysis APIs into stable JSON payloads for
``pmem patterns`` without creating graph edges, recommendations, network calls,
or raw-text output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.patterns import (
    anomaly_detection_payload,
    config_failure_correlation_payload,
    dataset_failure_correlation_payload,
    recurring_failure_report_payload,
    temporal_analysis_payload,
)

PATTERN_LIST_RESULT_VERSION = "pattern-list-result-v1"
PATTERN_CLI_RESULT_VERSION = "pattern-cli-result-v1"


def pattern_list_payload(
    project_root: str | Path,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return a metadata-only summary of all pattern-analysis pattern reports."""

    timestamp = generated_at or _utc_now_iso()
    reports = {
        "config_failure": config_failure_correlation_payload(
            project_root,
            generated_at=timestamp,
        ),
        "dataset_failure": dataset_failure_correlation_payload(
            project_root,
            generated_at=timestamp,
        ),
        "recurring_failures": recurring_failure_report_payload(
            project_root,
            include_text=False,
            generated_at=timestamp,
        ),
        "temporal": temporal_analysis_payload(
            project_root,
            generated_at=timestamp,
        ),
        "anomalies": anomaly_detection_payload(
            project_root,
            generated_at=timestamp,
        ),
    }
    summaries = tuple(
        _pattern_summary(pattern_key, report)
        for pattern_key, report in sorted(reports.items(), key=lambda item: item[0])
    )
    return {
        "schema_version": PATTERN_LIST_RESULT_VERSION,
        "generated_at": timestamp,
        "privacy_mode": "metadata_only",
        "raw_text_in_output": False,
        "database_mutation": False,
        "network": False,
        "derived_graph_edges": False,
        "causal_claim": False,
        "pattern_count": len(summaries),
        "candidate_count": sum(int(item["candidate_count"]) for item in summaries),
        "patterns": [dict(item) for item in summaries],
        "warnings": _aggregate_warnings(summaries),
    }


def config_failure_cli_payload(project_root: str | Path) -> dict[str, Any]:
    """Return config-failure correlation for the patterns CLI."""

    return _wrap_pattern_payload(
        "config_failure",
        config_failure_correlation_payload(project_root),
    )


def dataset_failure_cli_payload(project_root: str | Path) -> dict[str, Any]:
    """Return dataset-failure correlation dataset-failure correlation for the patterns CLI."""

    return _wrap_pattern_payload(
        "dataset_failure",
        dataset_failure_correlation_payload(project_root),
    )


def recurring_failures_cli_payload(
    project_root: str | Path,
    *,
    include_text: bool = False,
) -> dict[str, Any]:
    """Return recurring failure candidates for the patterns CLI."""

    return _wrap_pattern_payload(
        "recurring_failures",
        recurring_failure_report_payload(project_root, include_text=include_text),
    )


def temporal_cli_payload(project_root: str | Path) -> dict[str, Any]:
    """Return temporal analysis for the patterns CLI."""

    return _wrap_pattern_payload(
        "temporal",
        temporal_analysis_payload(project_root),
    )


def anomalies_cli_payload(project_root: str | Path) -> dict[str, Any]:
    """Return anomaly detection anomaly detection for the patterns CLI."""

    return _wrap_pattern_payload(
        "anomalies",
        anomaly_detection_payload(project_root),
    )


def _wrap_pattern_payload(pattern_key: str, report: dict[str, Any]) -> dict[str, Any]:
    summary = _pattern_summary(pattern_key, report)
    return {
        "schema_version": PATTERN_CLI_RESULT_VERSION,
        "pattern": pattern_key,
        "summary": dict(summary),
        "report": report,
        "privacy_mode": "metadata_only",
        "raw_text_in_output": False,
        "database_mutation": False,
        "network": False,
        "derived_graph_edges": False,
        "causal_claim": False,
    }


def _pattern_summary(pattern_key: str, report: dict[str, Any]) -> dict[str, Any]:
    candidate_count = _candidate_count(pattern_key, report)
    warnings = tuple(str(item) for item in report.get("warnings", ()) if str(item).strip())
    return {
        "pattern": pattern_key,
        "schema_version": str(report.get("schema_version", "")),
        "scope": str(report.get("scope", "")),
        "status": _status(candidate_count=candidate_count, warnings=warnings),
        "candidate_count": candidate_count,
        "warning_count": len(warnings),
        "warnings": list(warnings),
        "privacy_mode": str(report.get("privacy_mode", "metadata_only")),
        "raw_text_in_output": bool(report.get("raw_text_in_output", False)),
        "causal_claim": bool(report.get("causal_claim", False)),
    }


def _candidate_count(pattern_key: str, report: dict[str, Any]) -> int:
    if pattern_key in {"config_failure", "dataset_failure"}:
        return int(report.get("candidate_count", 0))
    if pattern_key == "recurring_failures":
        return int(report.get("recurring_cluster_count", 0))
    if pattern_key == "temporal":
        drift_count = 1 if report.get("drift") is not None else 0
        return drift_count + int(report.get("decision_impact_candidate_count", 0))
    if pattern_key == "anomalies":
        return int(report.get("metric_outlier_count", 0)) + int(
            report.get("reproducibility_candidate_count", 0)
        )
    return 0


def _status(*, candidate_count: int, warnings: tuple[str, ...]) -> str:
    if candidate_count > 0:
        return "candidates_found"
    if warnings:
        return "insufficient_data_or_no_signal"
    return "no_candidates"


def _aggregate_warnings(summaries: tuple[dict[str, Any], ...]) -> list[str]:
    warnings: list[str] = []
    for summary in summaries:
        pattern = str(summary["pattern"])
        for warning in summary["warnings"]:
            warnings.append(f"{pattern}: {warning}")
    return warnings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
