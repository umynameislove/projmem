"""dataset-failure correlation dataset-failure correlation unit tests."""

from __future__ import annotations

import json

import pytest

from pmem.errors import PmemValidationError
from pmem.patterns import dataset_failure
from pmem.patterns.dataset_failure import (
    DATASET_FAILURE_CORRELATION_SCHEMA_VERSION,
    DatasetIdentity,
    DatasetRunOutcome,
    dataset_failure_correlation_from_outcomes,
)

NOW = "2026-05-30T00:00:00Z"


def test_dataset_failure_detects_known_synthetic_dataset_signal() -> None:
    """A known dataset/failure association should rank without causal wording."""

    outcomes = _known_dataset_outcomes()

    payload = dataset_failure_correlation_from_outcomes(outcomes, generated_at=NOW)
    candidate = _candidate_for(payload, dataset_id="bead", version="v_bad")

    assert payload["schema_version"] == DATASET_FAILURE_CORRELATION_SCHEMA_VERSION
    assert payload["causal_claim"] is False
    assert payload["candidate_count"] >= 1
    assert candidate["contingency_table"] == {
        "dataset_failure": 6,
        "dataset_non_failure": 0,
        "other_failure": 0,
        "other_non_failure": 6,
    }
    assert candidate["failure_statistics"]["p_value"] < 0.01
    assert candidate["failure_statistics"]["risk_difference"] == 1.0
    assert candidate["metric_anomaly"]["metric_name"] == "accuracy"
    assert candidate["metric_anomaly"]["score"] > 5.0
    assert candidate["anomaly_score"] == candidate["metric_anomaly"]["score"]
    assert candidate["claim"] == "dataset_failure_correlation_observed_not_causal"
    assert "caused" not in json.dumps(payload, sort_keys=True).casefold()


def test_dataset_failure_gracefully_handles_missing_dataset_metadata() -> None:
    """No explicit dataset_id should produce an insufficient metadata message."""

    outcomes = tuple(
        DatasetRunOutcome(
            run_id=f"run_{index}",
            datasets=(),
            metrics={"accuracy": 0.8},
            has_failure=index < 5,
            failure_ids=(f"failure_{index}",) if index < 5 else (),
        )
        for index in range(10)
    )

    payload = dataset_failure_correlation_from_outcomes(outcomes, generated_at=NOW)

    assert payload["candidate_count"] == 0
    assert payload["candidates"] == []
    assert any("Insufficient dataset metadata" in warning for warning in payload["warnings"])


def test_dataset_failure_suppresses_small_sample_candidates() -> None:
    """dataset-failure correlation should not report candidates when the project is too small."""

    outcomes = (
        DatasetRunOutcome("run_1", (DatasetIdentity("bead", "v1"),), {"accuracy": 0.5}, True),
        DatasetRunOutcome("run_2", (DatasetIdentity("bead", "v2"),), {"accuracy": 0.9}, False),
    )

    payload = dataset_failure_correlation_from_outcomes(
        outcomes,
        min_total_runs=10,
        generated_at=NOW,
    )

    assert payload["candidate_count"] == 0
    assert any("Insufficient data" in warning for warning in payload["warnings"])


def test_dataset_failure_redacts_unsafe_dataset_labels_and_metric_names() -> None:
    """Dataset ids, versions, and metric names should not leak raw private labels."""

    outcomes = tuple(
        [
            DatasetRunOutcome(
                run_id=f"fail_{index}",
                datasets=(DatasetIdentity("/private/data/bead", "token-version"),),
                metrics={"/private/metric": 0.1 + index * 0.01},
                has_failure=True,
                failure_ids=(f"failure_{index}",),
            )
            for index in range(5)
        ]
        + [
            DatasetRunOutcome(
                run_id=f"ok_{index}",
                datasets=(DatasetIdentity("public", "v1"),),
                metrics={"/private/metric": 0.9 + index * 0.01},
                has_failure=False,
            )
            for index in range(5)
        ]
    )

    payload = dataset_failure_correlation_from_outcomes(outcomes, generated_at=NOW)
    raw_json = json.dumps(payload, sort_keys=True)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["dataset"]["dataset_id"].startswith("sha256:")
    )

    assert candidate["dataset"]["dataset_id_redacted"] is True
    assert candidate["dataset"]["version_redacted"] is True
    assert candidate["metric_anomaly"]["metric_name_redacted"] is True
    assert "/private/data/bead" not in raw_json
    assert "token-version" not in raw_json
    assert "/private/metric" not in raw_json


def test_dataset_failure_handles_missing_metrics_without_crashing() -> None:
    """Failure association can still be reported when metric anomaly is unavailable."""

    outcomes = tuple(
        [
            DatasetRunOutcome(
                run_id=f"fail_{index}",
                datasets=(DatasetIdentity("bead", "v_bad"),),
                metrics={},
                has_failure=True,
                failure_ids=(f"failure_{index}",),
            )
            for index in range(5)
        ]
        + [
            DatasetRunOutcome(
                run_id=f"ok_{index}",
                datasets=(DatasetIdentity("bead", "v_good"),),
                metrics={},
                has_failure=False,
            )
            for index in range(5)
        ]
    )

    payload = dataset_failure_correlation_from_outcomes(outcomes, generated_at=NOW)

    assert payload["candidate_count"] == 0
    assert any("No finite numeric metrics" in warning for warning in payload["warnings"])


def test_dataset_failure_validates_parameters() -> None:
    """Invalid thresholds should fail before producing misleading output."""

    with pytest.raises(PmemValidationError, match="min_total_runs"):
        dataset_failure_correlation_from_outcomes((), min_total_runs=0)
    with pytest.raises(PmemValidationError, match="min_dataset_runs"):
        dataset_failure_correlation_from_outcomes((), min_dataset_runs=0)
    with pytest.raises(PmemValidationError, match="max_results"):
        dataset_failure_correlation_from_outcomes((), max_results=0)


def test_dataset_failure_payload_validates_parameters_before_project_lookup(tmp_path) -> None:
    """Project-level API should reject invalid thresholds before touching the DB."""

    with pytest.raises(PmemValidationError, match="min_total_runs"):
        dataset_failure.dataset_failure_correlation_payload(tmp_path, min_total_runs=0)
    with pytest.raises(PmemValidationError, match="min_dataset_runs"):
        dataset_failure.dataset_failure_correlation_payload(tmp_path, min_dataset_runs=0)
    with pytest.raises(PmemValidationError, match="max_results"):
        dataset_failure.dataset_failure_correlation_payload(tmp_path, max_results=0)


def test_dataset_failure_private_helpers_fail_closed() -> None:
    """Defensive helpers should reject malformed JSON and unsupported metadata."""

    with pytest.raises(PmemValidationError, match="could not be parsed"):
        dataset_failure._safe_json_object("{", field="metrics_json")  # noqa: SLF001
    with pytest.raises(PmemValidationError, match="must be an object"):
        dataset_failure._safe_json_object("[]", field="metrics_json")  # noqa: SLF001
    with pytest.raises(PmemValidationError, match="must be an array"):
        dataset_failure._safe_json_array("{}", field="artifacts_json")  # noqa: SLF001

    datasets, skipped = dataset_failure._datasets_from_artifacts(  # noqa: SLF001
        [
            {"dataset_id": ["bad"], "version": "v1"},
            {"dataset_id": "", "version": "v1"},
            {"dataset_id": None, "version": "v1"},
            {"dataset_id": True, "version": "v1"},
        ]
    )
    metrics, metric_skipped = dataset_failure._numeric_metrics(  # noqa: SLF001
        {"ok": 1, "bad": "x", "flag": True, "nan": float("nan"), "": 2}
    )

    assert datasets == ()
    assert skipped == 4
    assert metrics == {"ok": 1.0}
    assert metric_skipped == 4
    assert dataset_failure._log_comb(1, 2) == float("-inf")  # noqa: SLF001


def _known_dataset_outcomes() -> tuple[DatasetRunOutcome, ...]:
    return tuple(
        [
            DatasetRunOutcome(
                run_id=f"run_bad_{index}",
                datasets=(DatasetIdentity("bead", "v_bad"),),
                metrics={"accuracy": 0.35 + index * 0.01},
                has_failure=True,
                failure_ids=(f"failure_{index}",),
            )
            for index in range(6)
        ]
        + [
            DatasetRunOutcome(
                run_id=f"run_good_{index}",
                datasets=(DatasetIdentity("bead", "v_good"),),
                metrics={"accuracy": 0.91 + index * 0.01},
                has_failure=False,
            )
            for index in range(6)
        ]
    )


def _candidate_for(payload: dict[str, object], *, dataset_id: str, version: str) -> dict:
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert isinstance(candidate, dict)
        dataset = candidate["dataset"]
        assert isinstance(dataset, dict)
        if dataset["dataset_id"] == dataset_id and dataset["version"] == version:
            return candidate
    raise AssertionError(f"missing candidate for {dataset_id}@{version}")
