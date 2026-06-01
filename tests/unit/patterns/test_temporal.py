"""temporal analysis unit tests."""

from __future__ import annotations

import json

import pytest

from pmem.errors import PmemValidationError
from pmem.patterns import temporal
from pmem.patterns.temporal import (
    TEMPORAL_ANALYSIS_SCHEMA_VERSION,
    DecisionEvent,
    TemporalRunOutcome,
    temporal_analysis_from_outcomes,
)

NOW = "2026-05-30T00:00:00Z"


def test_temporal_analysis_detects_known_drift_and_decision_shift() -> None:
    """Synthetic midpoint drift should be reported without causal wording."""

    runs = _known_temporal_runs()
    decisions = (
        DecisionEvent(
            decision_id="decision_midpoint",
            experiment_id="exp_temporal",
            created_at="2026-05-07T00:00:00Z",
        ),
    )

    payload = temporal_analysis_from_outcomes(
        runs,
        decisions,
        primary_metric="accuracy",
        metric_direction="max",
        generated_at=NOW,
    )
    raw_json = json.dumps(payload, sort_keys=True)
    drift = payload["drift"]
    decision = payload["decision_impact_candidates"][0]

    assert payload["schema_version"] == TEMPORAL_ANALYSIS_SCHEMA_VERSION
    assert payload["causal_claim"] is False
    assert drift["slope_per_day"] > 0
    assert drift["p_value"] < 0.05
    assert drift["claim"] == "temporal_drift_candidate_not_causal"
    assert decision["decision_id"] == "decision_midpoint"
    assert decision["sample_size"] == {"before_runs": 6, "after_runs": 6}
    assert decision["directional_delta"] > 0
    assert decision["metric_direction_interpretation"] == "metric_improving"
    assert decision["p_value"] < 0.05
    assert decision["claim"] == "decision_metric_shift_candidate_not_causal"
    assert "caused" not in raw_json.casefold()
    assert "root cause" not in raw_json.casefold()


def test_temporal_analysis_requires_primary_metric() -> None:
    """temporal analysis should not infer a primary metric from arbitrary metric keys."""

    payload = temporal_analysis_from_outcomes(
        _known_temporal_runs(),
        (),
        primary_metric=None,
        generated_at=NOW,
    )

    assert payload["primary_metric"] == ""
    assert payload["drift"] is None
    assert payload["decision_impact_candidates"] == []
    assert payload["skipped_counts"]["missing_primary_metric"] == 1
    assert any("Primary metric is required" in warning for warning in payload["warnings"])


def test_temporal_analysis_suppresses_underpowered_samples() -> None:
    """Small time series should degrade gracefully instead of reporting p-values."""

    runs = (
        TemporalRunOutcome("run_1", "exp", "2026-05-01T00:00:00Z", {"accuracy": 0.5}),
        TemporalRunOutcome("run_2", "exp", "2026-05-02T00:00:00Z", {"accuracy": 0.8}),
    )

    payload = temporal_analysis_from_outcomes(
        runs,
        (),
        primary_metric="accuracy",
        min_total_runs=8,
        generated_at=NOW,
    )

    assert payload["drift"] is None
    assert payload["decision_impact_candidate_count"] == 0
    assert any("Insufficient data" in warning for warning in payload["warnings"])


def test_temporal_analysis_redacts_unsafe_metric_label() -> None:
    """Metric names can be private labels and must be hashed in output."""

    runs = tuple(
        TemporalRunOutcome(
            run_id=f"run_{index}",
            experiment_id="exp",
            timestamp=f"2026-05-{index + 1:02d}T00:00:00Z",
            metrics={"/private/metric": 0.5 + index * 0.02},
        )
        for index in range(8)
    )

    payload = temporal_analysis_from_outcomes(
        runs,
        (),
        primary_metric="/private/metric",
        min_total_runs=8,
        generated_at=NOW,
    )
    raw_json = json.dumps(payload, sort_keys=True)

    assert payload["primary_metric"].startswith("sha256:")
    assert payload["primary_metric_redacted"] is True
    assert payload["drift"]["metric_name_redacted"] is True
    assert "/private/metric" not in raw_json


def test_temporal_analysis_handles_min_direction_and_project_decisions() -> None:
    """Lower-is-better metrics should interpret negative deltas as improvement."""

    runs = tuple(
        [
            TemporalRunOutcome(
                run_id=f"loss_before_{index}",
                experiment_id="exp_a",
                timestamp=f"2026-05-{index + 1:02d}T00:00:00Z",
                metrics={"loss": 0.80 - index * 0.01},
            )
            for index in range(4)
        ]
        + [
            TemporalRunOutcome(
                run_id=f"loss_after_{index}",
                experiment_id="exp_b" if index == 0 else "exp_a",
                timestamp=f"2026-05-{index + 5:02d}T00:00:00Z",
                metrics={"loss": 0.50 - index * 0.01},
            )
            for index in range(4)
        ]
    )
    decisions = (
        DecisionEvent("decision_project", "2026-05-05T00:00:00Z"),
        DecisionEvent("decision_too_late", "2026-06-01T00:00:00Z"),
    )

    payload = temporal_analysis_from_outcomes(
        runs,
        decisions,
        primary_metric="loss",
        metric_direction="min",
        min_total_runs=8,
        generated_at=NOW,
    )
    decision = payload["decision_impact_candidates"][0]

    assert payload["decision_impact_candidate_count"] == 1
    assert decision["decision_id"] == "decision_project"
    assert decision["decision_scope"] == "project"
    assert decision["directional_delta"] > 0
    assert decision["metric_direction_interpretation"] == "metric_improving"


def test_temporal_analysis_skips_bad_timestamps_and_missing_metrics() -> None:
    """Malformed rows should be counted and skipped without raw traceback data."""

    runs = (
        TemporalRunOutcome("run_ok_1", "exp", "2026-05-01T00:00:00Z", {"accuracy": 0.5}),
        TemporalRunOutcome("run_ok_2", "exp", "2026-05-02T00:00:00Z", {"accuracy": 0.6}),
        TemporalRunOutcome("run_ok_3", "exp", "2026-05-03T00:00:00Z", {"accuracy": 0.7}),
        TemporalRunOutcome("run_missing", "exp", "2026-05-04T00:00:00Z", {"loss": 1.0}),
        TemporalRunOutcome("run_bad_time", "exp", "not-a-date", {"accuracy": 0.8}),
        TemporalRunOutcome("run_nan", "exp", "2026-05-05T00:00:00Z", {"accuracy": float("nan")}),
    )
    decisions = (
        DecisionEvent("decision_bad_time", "not-a-date", "exp"),
        DecisionEvent("decision_ok", "2026-05-02T12:00:00Z", "exp"),
    )

    payload = temporal_analysis_from_outcomes(
        runs,
        decisions,
        primary_metric="accuracy",
        min_total_runs=3,
        min_decision_side_runs=1,
        generated_at=NOW,
    )

    assert payload["metric_run_count"] == 3
    assert payload["decision_count"] == 1
    assert payload["skipped_counts"]["invalid_timestamps"] == 2
    assert payload["skipped_counts"]["missing_primary_metric_observations"] == 1
    assert payload["skipped_counts"]["non_finite_primary_metric_values"] == 1


def test_temporal_analysis_is_deterministic_for_input_order() -> None:
    """Input ordering should not change report payloads when timestamp ids match."""

    runs = _known_temporal_runs()
    decisions = (DecisionEvent("decision_midpoint", "2026-05-07T00:00:00Z", "exp_temporal"),)

    first = temporal_analysis_from_outcomes(
        runs,
        decisions,
        primary_metric="accuracy",
        metric_direction="max",
        generated_at=NOW,
    )
    second = temporal_analysis_from_outcomes(
        tuple(reversed(runs)),
        tuple(reversed(decisions)),
        primary_metric="accuracy",
        metric_direction="max",
        generated_at=NOW,
    )

    assert first == second


def test_temporal_analysis_validates_parameters() -> None:
    """Invalid thresholds should fail before producing misleading output."""

    with pytest.raises(PmemValidationError, match="min_total_runs"):
        temporal_analysis_from_outcomes((), (), primary_metric="accuracy", min_total_runs=2)
    with pytest.raises(PmemValidationError, match="min_decision_side_runs"):
        temporal_analysis_from_outcomes(
            (),
            (),
            primary_metric="accuracy",
            min_decision_side_runs=0,
        )
    with pytest.raises(PmemValidationError, match="max_decision_results"):
        temporal_analysis_from_outcomes(
            (),
            (),
            primary_metric="accuracy",
            max_decision_results=0,
        )
    with pytest.raises(PmemValidationError, match="metric_direction"):
        temporal_analysis_from_outcomes((), (), primary_metric="accuracy", metric_direction="up")


def test_temporal_private_helpers_fail_closed() -> None:
    """Defensive helpers should reject malformed JSON and unsupported shapes."""

    with pytest.raises(PmemValidationError, match="could not be parsed"):
        temporal._safe_json_object("{", field="metrics_json")  # noqa: SLF001
    with pytest.raises(PmemValidationError, match="must be an object"):
        temporal._safe_json_object("[]", field="metrics_json")  # noqa: SLF001

    metrics, skipped = temporal._numeric_metrics(  # noqa: SLF001
        {"accuracy": 1, "bad": "x", "flag": True, "nan": float("nan"), "": 2}
    )

    assert metrics == {"accuracy": 1.0}
    assert skipped == 4
    assert temporal._normalize_metric_direction(" ") is None  # noqa: SLF001
    assert temporal._interpret_directional_change(-1.0) == "metric_regressing"  # noqa: SLF001
    assert temporal._interpret_directional_change(0.0) == "flat"  # noqa: SLF001
    assert temporal._sample_variance([1.0], 1.0) == 0.0  # noqa: SLF001
    assert temporal._parse_timestamp("") is None  # noqa: SLF001
    assert temporal._parse_timestamp("not-a-date") is None  # noqa: SLF001
    assert temporal._parse_timestamp("2026-05-30T00:00:00") is not None  # noqa: SLF001
    assert temporal._normal_two_sided_p_value(float("inf")) == 0.0  # noqa: SLF001
    assert temporal._round_float(float("inf")) == float("inf")  # noqa: SLF001


def test_temporal_private_math_helpers_cover_degenerate_cases() -> None:
    """Degenerate math branches should be explicit and deterministic."""

    with pytest.raises(PmemValidationError, match="at least 3"):
        temporal._linear_regression([1.0, 2.0], [1.0, 2.0])  # noqa: SLF001
    with pytest.raises(PmemValidationError, match="non-identical"):
        temporal._linear_regression([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])  # noqa: SLF001

    flat_regression = temporal._linear_regression([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])  # noqa: SLF001
    flat_shift = temporal._mean_shift_test([1.0, 1.0], [1.0, 1.0])  # noqa: SLF001

    assert flat_regression["p_value"] == 1.0
    assert flat_shift == {"t_statistic": 0.0, "p_value": 1.0}


def _known_temporal_runs() -> tuple[TemporalRunOutcome, ...]:
    before = [
        TemporalRunOutcome(
            run_id=f"run_before_{index}",
            experiment_id="exp_temporal",
            timestamp=f"2026-05-{index + 1:02d}T00:00:00Z",
            metrics={"accuracy": 0.50 + index * 0.01},
        )
        for index in range(6)
    ]
    after = [
        TemporalRunOutcome(
            run_id=f"run_after_{index}",
            experiment_id="exp_temporal",
            timestamp=f"2026-05-{index + 7:02d}T00:00:00Z",
            metrics={"accuracy": 0.80 + index * 0.01},
        )
        for index in range(6)
    ]
    return tuple(before + after)
