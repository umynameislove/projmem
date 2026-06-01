"""failure embeddings tests for deterministic local failure embeddings."""

from __future__ import annotations

import json
import sys

import pytest

from pmem.errors import PmemValidationError
from pmem.services.failure_embeddings import (
    FAILURE_EMBEDDING_METHOD,
    FAILURE_EMBEDDING_SCHEMA_VERSION,
    cosine_similarity,
    failure_embedding_payload,
)
from pmem.services.failure_logging import log_failure
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command


def test_failure_embedding_payload_handles_empty_project(tmp_path) -> None:
    """failure embeddings should have a stable zero-record contract."""

    init_project(tmp_path, project_name="demo")

    payload = failure_embedding_payload(
        tmp_path,
        generated_at="2026-01-01T00:00:00Z",
        dimension=32,
    )

    assert payload["schema_version"] == FAILURE_EMBEDDING_SCHEMA_VERSION
    assert payload["generated_at"] == "2026-01-01T00:00:00Z"
    assert payload["method"] == FAILURE_EMBEDDING_METHOD
    assert payload["dimension"] == 32
    assert payload["privacy_mode"] == "structured_only"
    assert payload["record_count"] == 0
    assert payload["records"] == []


def test_failure_embedding_payload_is_deterministic_and_excludes_raw_text(tmp_path) -> None:
    """Same DB and frozen timestamp should produce byte-stable JSON."""

    _create_failure(
        tmp_path,
        error_type="MetricRegression",
        description="SECRET accuracy dropped below target.",
        tag="data quality",
    )

    first = failure_embedding_payload(
        tmp_path,
        generated_at="2026-01-01T00:00:00Z",
        dimension=64,
    )
    second = failure_embedding_payload(
        tmp_path,
        generated_at="2026-01-01T00:00:00Z",
        dimension=64,
    )
    raw_json = json.dumps(first, sort_keys=True)
    record = first["records"][0]

    assert first == second
    assert first["include_text"] is False
    assert record["text_included"] is False
    assert len(record["vector"]) == 64
    assert record["vector_norm"] == pytest.approx(1.0)
    assert "description" not in record
    assert "SECRET" not in raw_json
    assert "accuracy dropped" not in raw_json


def test_failure_embedding_include_text_changes_vector_without_returning_text(tmp_path) -> None:
    """Confirmed free-text use should affect vectors but not echo text."""

    _create_failure(
        tmp_path,
        error_type="MetricRegression",
        description="SECRET optimizer diverged quickly.",
        tag="convergence",
    )

    structured = failure_embedding_payload(
        tmp_path,
        include_text=False,
        generated_at="2026-01-01T00:00:00Z",
        dimension=64,
    )
    text_derived = failure_embedding_payload(
        tmp_path,
        include_text=True,
        generated_at="2026-01-01T00:00:00Z",
        dimension=64,
    )
    raw_json = json.dumps(text_derived, sort_keys=True)

    assert text_derived["privacy_mode"] == "explicit_text_derived"
    assert structured["records"][0]["vector"] != text_derived["records"][0]["vector"]
    assert "derived_from_free_text" in raw_json
    assert "SECRET optimizer" not in raw_json


def test_failure_embedding_similarity_reflects_shared_failure_features(tmp_path) -> None:
    """Similar failure records should be closer than unrelated ones."""

    _create_failure(
        tmp_path,
        error_type="MetricRegression",
        description="Accuracy dropped after bad data split.",
        tag="data quality",
    )
    _create_failure(
        tmp_path,
        error_type="MetricRegression",
        description="Accuracy dropped again after noisy labels.",
        tag="data quality",
    )
    _create_failure(
        tmp_path,
        error_type="OOM",
        description="CUDA memory exhausted during batch step.",
        tag="oom",
    )

    payload = failure_embedding_payload(
        tmp_path,
        include_text=True,
        generated_at="2026-01-01T00:00:00Z",
        dimension=128,
    )
    metric_records = [
        record for record in payload["records"] if record["error_type"] == "MetricRegression"
    ]
    oom_record = next(record for record in payload["records"] if record["error_type"] == "OOM")

    similar_score = cosine_similarity(metric_records[0]["vector"], metric_records[1]["vector"])
    unrelated_score = max(
        cosine_similarity(record["vector"], oom_record["vector"]) for record in metric_records
    )

    assert payload["record_count"] == 3
    assert similar_score > unrelated_score


def test_failure_embedding_rejects_invalid_dimension(tmp_path) -> None:
    """Dimensions outside the validated range are not accepted."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemValidationError, match="dimension"):
        failure_embedding_payload(tmp_path, dimension=8)


def test_cosine_similarity_rejects_mismatched_dimensions() -> None:
    """L95: cosine_similarity must raise when vector lengths differ."""

    with pytest.raises(PmemValidationError, match="dimension"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine_similarity_returns_zero_for_zero_vector() -> None:
    """L99: cosine_similarity returns 0.0 when either vector has zero norm."""

    zero = [0.0, 0.0, 0.0]
    nonzero = [1.0, 0.0, 0.0]

    assert cosine_similarity(zero, nonzero) == 0.0
    assert cosine_similarity(nonzero, zero) == 0.0
    assert cosine_similarity(zero, zero) == 0.0


def test_failure_embedding_record_empty_fields_produce_zero_vector() -> None:
    """L148: _features_to_vector zero-norm fallback returns zero vector safely."""

    # All structured fields empty → every feature token is blank → filtered out
    # → empty feature list → zero vector, zero norm
    from pmem.services.failure_embeddings import failure_embedding_record

    record = {
        "id": "gap-test-id",
        "run_id": "gap-run-id",
        "error_type": "",
        "severity": "",
        "source": "",
        "tags": [],
        "created_at": "2026-01-01T00:00:00Z",
    }
    result = failure_embedding_record(record, include_text=False, dimension=32)

    assert result["vector_norm"] == 0.0
    assert all(v == 0.0 for v in result["vector"])
    assert len(result["vector"]) == 32


def _create_failure(tmp_path, *, error_type: str, description: str, tag: str) -> None:
    if not (tmp_path / ".pmem").exists():
        init_project(tmp_path, project_name="demo")
    run_result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])
    log_failure(
        tmp_path,
        run_id=run_result.record.run_id,
        error_type=error_type,
        description=description,
        root_cause="Synthetic test root cause",
        lesson="Synthetic test lesson",
        severity="high",
        tags=(tag,),
        source="user_confirmed",
    )
