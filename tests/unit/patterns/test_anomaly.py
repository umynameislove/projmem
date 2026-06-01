"""anomaly detection anomaly detection unit tests."""

from __future__ import annotations

import json

import pytest

from pmem.errors import PmemValidationError
from pmem.patterns import anomaly
from pmem.patterns.anomaly import (
    ANOMALY_DETECTION_SCHEMA_VERSION,
    AnomalyRunOutcome,
    anomaly_detection_from_outcomes,
)
from pmem.utils.hashing import compute_text_hash

NOW = "2026-05-30T00:00:00Z"


def test_anomaly_detection_finds_outlier_and_reproducibility_candidate() -> None:
    """Synthetic IQR outlier and same-config variance should be detected."""

    runs = _known_anomaly_runs()

    payload = anomaly_detection_from_outcomes(
        runs,
        primary_metric="accuracy",
        generated_at=NOW,
    )
    outlier = payload["metric_outliers"][0]
    repro = payload["reproducibility_candidates"][0]
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == ANOMALY_DETECTION_SCHEMA_VERSION
    assert payload["causal_claim"] is False
    assert outlier["run_id"] == "run_outlier_high"
    assert outlier["kind"] == "metric_outlier"
    assert outlier["direction"] == "high"
    assert outlier["claim"] == "metric_outlier_candidate_not_causal"
    assert repro["kind"] == "same_config_metric_variance"
    assert repro["sample_size"] == 4
    assert repro["standard_deviation"] >= 0.05
    assert repro["range"] >= 0.10
    assert repro["claim"] == "potential_reproducibility_issue_not_causal"
    assert "caused" not in raw_json.casefold()
    assert "root cause" not in raw_json.casefold()


def test_anomaly_detection_handles_insufficient_data() -> None:
    """Small projects should return warnings instead of anomaly claims."""

    runs = (
        AnomalyRunOutcome(
            "run_1",
            "exp",
            "2026-05-01T00:00:00Z",
            {"accuracy": 0.8},
            "a" * 64,
        ),
    )

    payload = anomaly_detection_from_outcomes(runs, primary_metric="accuracy", generated_at=NOW)

    assert payload["metric_outlier_count"] == 0
    assert payload["reproducibility_candidate_count"] == 0
    assert any("Insufficient data" in warning for warning in payload["warnings"])


def test_anomaly_detection_redacts_unsafe_metric_labels_and_config_identity() -> None:
    """Metric labels may be private and config values must not appear in output."""

    config_hash = compute_text_hash('{"private_path":"/secret/config.yaml"}')
    runs = tuple(
        AnomalyRunOutcome(
            run_id=f"run_{index}",
            experiment_id="exp",
            timestamp=f"2026-05-{index + 1:02d}T00:00:00Z",
            metrics={"/private/metric": 0.80 + index * 0.01},
            config_hash=config_hash,
        )
        for index in range(8)
    ) + (
        AnomalyRunOutcome(
            "run_private_outlier",
            "exp",
            "2026-05-09T00:00:00Z",
            {"/private/metric": 1.5},
            None,
        ),
    )

    payload = anomaly_detection_from_outcomes(
        runs,
        primary_metric="/private/metric",
        generated_at=NOW,
    )
    raw_json = json.dumps(payload, sort_keys=True)
    outlier = payload["metric_outliers"][0]

    assert payload["primary_metric"].startswith("sha256:")
    assert outlier["metric_name_redacted"] is True
    assert payload["skipped_counts"]["missing_config_hash"] == 1
    assert "/private/metric" not in raw_json
    assert "/secret/config.yaml" not in raw_json
    assert config_hash not in raw_json


def test_anomaly_detection_is_deterministic_for_input_ordering() -> None:
    """Input ordering should not change candidate ids or evidence ordering."""

    runs = _known_anomaly_runs()

    first = anomaly_detection_from_outcomes(runs, primary_metric="accuracy", generated_at=NOW)
    second = anomaly_detection_from_outcomes(
        tuple(reversed(runs)),
        primary_metric="accuracy",
        generated_at=NOW,
    )

    assert first == second


def test_anomaly_detection_validates_parameters() -> None:
    """Invalid thresholds should fail before producing misleading output."""

    with pytest.raises(PmemValidationError, match="min_experiment_metric_runs"):
        anomaly_detection_from_outcomes((), min_experiment_metric_runs=3)
    with pytest.raises(PmemValidationError, match="min_config_group_runs"):
        anomaly_detection_from_outcomes((), min_config_group_runs=1)
    with pytest.raises(PmemValidationError, match="iqr_multiplier"):
        anomaly_detection_from_outcomes((), iqr_multiplier=0)
    with pytest.raises(PmemValidationError, match="min_metric_range"):
        anomaly_detection_from_outcomes((), min_metric_range=-1)
    with pytest.raises(PmemValidationError, match="min_standard_deviation"):
        anomaly_detection_from_outcomes((), min_standard_deviation=-1)
    with pytest.raises(PmemValidationError, match="max_results"):
        anomaly_detection_from_outcomes((), max_results=0)


def test_anomaly_private_helpers_fail_closed() -> None:
    """Defensive helpers should reject malformed JSON and handle edge math."""

    with pytest.raises(PmemValidationError, match="could not be parsed"):
        anomaly._safe_json_object("{", field="metrics_json")  # noqa: SLF001
    with pytest.raises(PmemValidationError, match="must be an object"):
        anomaly._safe_json_object("[]", field="metrics_json")  # noqa: SLF001
    with pytest.raises(PmemValidationError, match="percentile"):
        anomaly._percentile([], 0.5)  # noqa: SLF001

    metrics, skipped = anomaly._numeric_metrics(  # noqa: SLF001
        {"accuracy": 1, "bad": "x", "flag": True, "nan": float("nan"), "": 2}
    )

    assert metrics == {"accuracy": 1.0}
    assert skipped == 4
    assert anomaly._sample_stddev([1.0], 1.0) == 0.0  # noqa: SLF001
    assert anomaly._is_sha256("a" * 64) is True  # noqa: SLF001
    assert anomaly._is_sha256("A" * 64) is False  # noqa: SLF001
    assert anomaly._round_float(float("inf")) == float("inf")  # noqa: SLF001


def _known_anomaly_runs() -> tuple[AnomalyRunOutcome, ...]:
    unique_runs = [
        AnomalyRunOutcome(
            run_id=f"run_normal_{index}",
            experiment_id="exp_outlier",
            timestamp=f"2026-05-{index + 1:02d}T00:00:00Z",
            metrics={"accuracy": 0.80 + index * 0.01},
            config_hash=compute_text_hash(f"unique-{index}"),
        )
        for index in range(8)
    ]
    outlier = AnomalyRunOutcome(
        run_id="run_outlier_high",
        experiment_id="exp_outlier",
        timestamp="2026-05-09T00:00:00Z",
        metrics={"accuracy": 1.50},
        config_hash=compute_text_hash("unique-outlier"),
    )
    shared_hash = compute_text_hash("shared-config")
    repro_runs = [
        AnomalyRunOutcome(
            run_id=f"run_repro_{index}",
            experiment_id="exp_repro",
            timestamp=f"2026-06-{index + 1:02d}T00:00:00Z",
            metrics={"accuracy": value},
            config_hash=shared_hash,
        )
        for index, value in enumerate((0.40, 0.95, 0.42, 0.90))
    ]
    return tuple(unique_runs + [outlier] + repro_runs)
