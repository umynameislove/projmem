"""config-failure correlation unit tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from pmem.errors import PmemValidationError
from pmem.patterns import config_failure
from pmem.patterns.config_failure import (
    CONFIG_FAILURE_CORRELATION_SCHEMA_VERSION,
    RunFailureOutcome,
    config_failure_correlation_from_outcomes,
)

NOW = "2026-05-29T00:00:00Z"


def test_config_failure_detects_known_synthetic_correlation() -> None:
    """A known synthetic association should rank first without causal wording."""

    outcomes = tuple(
        [
            RunFailureOutcome(
                run_id=f"run_fail_{index}",
                config={"optimizer": "adam", "lr": 0.001},
                has_failure=True,
                failure_ids=(f"failure_{index}",),
            )
            for index in range(6)
        ]
        + [
            RunFailureOutcome(
                run_id=f"run_ok_{index}",
                config={"optimizer": "sgd", "lr": 0.01},
                has_failure=False,
            )
            for index in range(6)
        ]
    )

    payload = config_failure_correlation_from_outcomes(outcomes, generated_at=NOW)
    optimizer_candidate = _candidate_for(payload, key="optimizer", value="adam")

    assert payload["schema_version"] == CONFIG_FAILURE_CORRELATION_SCHEMA_VERSION
    assert payload["candidate_count"] >= 1
    assert payload["causal_claim"] is False
    assert optimizer_candidate["contingency_table"] == {
        "exposed_failure": 6,
        "exposed_non_failure": 0,
        "unexposed_failure": 0,
        "unexposed_non_failure": 6,
    }
    assert optimizer_candidate["statistics"]["p_value"] < 0.01
    assert optimizer_candidate["statistics"]["odds_ratio"] > 1.0
    assert optimizer_candidate["claim"] == "correlation_observed_not_causal"
    assert "caused" not in json.dumps(payload, sort_keys=True).casefold()


def test_config_failure_insufficient_data_suppresses_p_values() -> None:
    """Do not report statistical significance on too-small samples."""

    outcomes = (
        RunFailureOutcome("run_1", {"batch_size": 32}, True, ("failure_1",)),
        RunFailureOutcome("run_2", {"batch_size": 64}, False),
    )

    payload = config_failure_correlation_from_outcomes(
        outcomes,
        min_total_runs=10,
        generated_at=NOW,
    )

    assert payload["candidate_count"] == 0
    assert payload["candidates"] == []
    assert any("Insufficient data" in warning for warning in payload["warnings"])


def test_config_failure_redacts_unsafe_values_and_skips_sensitive_keys() -> None:
    """Config vocabulary in the report should avoid secrets and raw paths."""

    outcomes = tuple(
        [
            RunFailureOutcome(
                run_id=f"fail_{index}",
                config={
                    "dataset": "/private/data/train.csv",
                    "dataset/path": "train",
                    "api_token": "SECRET",
                    "nested": {"dropout": 0.2},
                },
                has_failure=True,
                failure_ids=(f"failure_{index}",),
            )
            for index in range(5)
        ]
        + [
            RunFailureOutcome(
                run_id=f"ok_{index}",
                config={
                    "dataset": "public_set",
                    "dataset/path": "eval",
                    "api_token": "SECRET",
                    "nested": {"dropout": 0.1},
                },
                has_failure=False,
            )
            for index in range(5)
        ]
    )

    payload = config_failure_correlation_from_outcomes(outcomes, generated_at=NOW)
    raw_json = json.dumps(payload, sort_keys=True)
    redacted_dataset_candidate = next(
        candidate
        for candidate in payload["candidates"]
        if candidate["feature"]["key"] == "dataset"
        and candidate["feature"]["value"].startswith("sha256:")
    )
    redacted_key_candidate = next(
        candidate
        for candidate in payload["candidates"]
        if candidate["feature"]["key"].startswith("sha256:")
        and candidate["feature"]["value"] == "train"
    )

    assert payload["skipped_counts"]["sensitive_config_keys"] == 10
    assert redacted_dataset_candidate["feature"]["value_redacted"] is True
    assert redacted_key_candidate["feature"]["key_redacted"] is True
    assert "/private/data/train.csv" not in raw_json
    assert "dataset/path" not in raw_json
    assert "SECRET" not in raw_json
    assert "api_token" not in raw_json


def test_config_failure_handles_scalar_privacy_edges_and_non_scalar_skips() -> None:
    """config-failure correlation should normalize scalar values and skip complex config values."""

    outcomes = tuple(
        [
            RunFailureOutcome(
                run_id=f"fail_{index}",
                config={
                    "enabled": True,
                    "nullable": None,
                    "fold": 1,
                    "temperature": float("nan"),
                    "layers": [1, 2],
                    "bad path": "model/checkpoint.bin",
                },
                has_failure=True,
                failure_ids=(f"failure_{index}",),
            )
            for index in range(5)
        ]
        + [
            RunFailureOutcome(
                run_id=f"ok_{index}",
                config={
                    "enabled": False,
                    "nullable": None,
                    "fold": 2,
                    "temperature": 0.0,
                    "layers": [3, 4],
                    "bad path": "safe_label",
                },
                has_failure=False,
            )
            for index in range(5)
        ]
    )

    payload = config_failure_correlation_from_outcomes(outcomes, generated_at=NOW)
    raw_json = json.dumps(payload, sort_keys=True)
    features, _ = config_failure._features_for_config(  # noqa: SLF001
        {"nullable": None, "fold": 1, "enabled": True}
    )
    feature_values = {(feature.key, feature.value) for feature in features}

    assert _candidate_for(payload, key="enabled", value="true")["feature"]["value"] == "true"
    assert ("nullable", "null") in feature_values
    assert ("fold", "1") in feature_values
    assert _candidate_for(payload, key="fold", value="1")["feature"]["value"] == "1"
    assert payload["skipped_counts"]["non_scalar_values"] == 10
    assert "model/checkpoint.bin" not in raw_json
    assert "non_finite_number" in raw_json


def test_config_failure_is_deterministic_for_same_input_ordering() -> None:
    """Run ordering should not change report IDs, ordering, or p-values."""

    first_order = (
        RunFailureOutcome("run_b", {"model": "large"}, True, ("failure_b",)),
        RunFailureOutcome("run_a", {"model": "large"}, True, ("failure_a",)),
        RunFailureOutcome("run_d", {"model": "small"}, False),
        RunFailureOutcome("run_c", {"model": "small"}, False),
        RunFailureOutcome("run_f", {"model": "large"}, True, ("failure_f",)),
        RunFailureOutcome("run_e", {"model": "small"}, False),
        RunFailureOutcome("run_h", {"model": "large"}, True, ("failure_h",)),
        RunFailureOutcome("run_g", {"model": "small"}, False),
        RunFailureOutcome("run_j", {"model": "large"}, True, ("failure_j",)),
        RunFailureOutcome("run_i", {"model": "small"}, False),
    )
    second_order = tuple(reversed(first_order))

    first = config_failure_correlation_from_outcomes(first_order, generated_at=NOW)
    second = config_failure_correlation_from_outcomes(second_order, generated_at=NOW)

    assert first == second


def test_config_failure_validates_parameters() -> None:
    """Invalid thresholds should fail before producing misleading output."""

    with pytest.raises(PmemValidationError, match="min_total_runs"):
        config_failure_correlation_from_outcomes((), min_total_runs=0)
    with pytest.raises(PmemValidationError, match="min_feature_group_runs"):
        config_failure_correlation_from_outcomes((), min_feature_group_runs=0)
    with pytest.raises(PmemValidationError, match="max_results"):
        config_failure_correlation_from_outcomes((), max_results=0)


def test_config_failure_payload_validates_parameters_before_project_lookup(tmp_path) -> None:
    """Project-level API should reject invalid thresholds before touching the DB."""

    with pytest.raises(PmemValidationError, match="min_total_runs"):
        config_failure.config_failure_correlation_payload(tmp_path, min_total_runs=0)
    with pytest.raises(PmemValidationError, match="min_feature_group_runs"):
        config_failure.config_failure_correlation_payload(tmp_path, min_feature_group_runs=0)
    with pytest.raises(PmemValidationError, match="max_results"):
        config_failure.config_failure_correlation_payload(tmp_path, max_results=0)


def test_config_failure_skips_underpowered_feature_groups() -> None:
    """Features with tiny exposed/unexposed groups should not emit p-values."""

    outcomes = tuple(
        [RunFailureOutcome("rare_fail", {"rare": "yes"}, True, ("failure_1",))]
        + [RunFailureOutcome(f"ok_{index}", {"common": "yes"}, False) for index in range(9)]
    )

    payload = config_failure_correlation_from_outcomes(
        outcomes,
        min_total_runs=10,
        min_feature_group_runs=2,
        generated_at=NOW,
    )

    assert all(candidate["feature"]["key"] != "rare" for candidate in payload["candidates"])


def test_config_failure_reports_no_comparison_runs() -> None:
    """All-failure datasets should degrade gracefully instead of reporting significance."""

    outcomes = tuple(
        RunFailureOutcome(f"fail_{index}", {"optimizer": "adam"}, True, (f"failure_{index}",))
        for index in range(10)
    )

    payload = config_failure_correlation_from_outcomes(outcomes, generated_at=NOW)

    assert payload["candidate_count"] == 0
    assert any("No non-failure comparison" in warning for warning in payload["warnings"])


def test_config_failure_private_helpers_fail_closed() -> None:
    """Defensive helpers should reject malformed JSON and keep math finite-safe."""

    with pytest.raises(PmemValidationError, match="could not be parsed"):
        config_failure._safe_json_object("{")  # noqa: SLF001
    with pytest.raises(PmemValidationError, match="must be an object"):
        config_failure._safe_json_object("[]")  # noqa: SLF001

    assert config_failure._log_comb(1, 2) == float("-inf")  # noqa: SLF001
    assert config_failure._round_float(float("inf")) == float("inf")  # noqa: SLF001


def _candidate_for(payload: dict[str, object], *, key: str, value: str) -> dict[str, Any]:
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert isinstance(candidate, dict)
        feature = candidate["feature"]
        assert isinstance(feature, dict)
        if feature["key"] == key and feature["value"] == value:
            return candidate
    raise AssertionError(f"missing candidate for {key}={value}")
