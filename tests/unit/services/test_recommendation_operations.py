"""recommendation CLI recommendation CLI service tests."""

from __future__ import annotations

import json
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pmem.errors import (
    PmemNotFoundError,
    PmemPersistenceError,
    PmemSecurityError,
    PmemValidationError,
)
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services import recommendation_operations as operations
from pmem.services.project_init import init_project
from pmem.services.recommendation_operations import (
    export_recommendations,
    recommendation_detail_payload,
    recommendation_list_payload,
)
from pmem.utils.hashing import compute_text_hash

NOW = datetime(2026, 5, 31, 0, 0, tzinfo=timezone.utc)


def test_recommendation_list_handles_sparse_project_with_warning(tmp_path) -> None:
    """recommendation CLI should explain insufficient data without fabricating candidates."""

    init_project(tmp_path, project_name="empty-recommend-cli", primary_metric="accuracy")

    payload = recommendation_list_payload(tmp_path)

    assert payload["schema_version"] == "recommendation-list-result-v1"
    assert payload["recommendation_count"] == 0
    assert payload["recommendations"] == []
    assert payload["database_mutation"] is False
    assert any("Insufficient project evidence" in warning for warning in payload["warnings"])


def test_recommendation_list_returns_safe_verified_payload(tmp_path) -> None:
    """Generated recommendations should include counts and no private raw text."""

    _seed_recommendation_project(tmp_path)

    payload = recommendation_list_payload(tmp_path)
    raw_json = json.dumps(payload, sort_keys=True)
    recommendation_ids = [item["recommendation_id"] for item in payload["recommendations"]]

    assert payload["recommendation_count"] == 5
    assert payload["basis_counts"]["experiments"] == 3
    assert payload["basis_counts"]["runs"] >= 10
    assert payload["basis_counts"]["failures"] >= 5
    assert all(item.startswith("rec_d59_") for item in recommendation_ids)
    assert "rec_d59_try_next_001" not in recommendation_ids
    assert "evidence_counts" in payload["recommendations"][0]
    assert "PRIVATE" not in raw_json
    assert "python train.py" not in raw_json
    assert "SUPPORTS::" not in raw_json
    assert "CONTRADICTS::" not in raw_json
    assert "caused" not in raw_json.casefold()


def test_recommendation_list_surfaces_cd5_data_quality_warnings(tmp_path) -> None:
    """Cfile tracking should expose dataset, stale-metric, and possible mislabel risks concisely."""

    init_result = init_project(
        tmp_path,
        project_name="recommendation-quality-warnings",
        primary_metric="accuracy",
        metric_direction="max",
    )
    connection = connect_database(project_database_path(tmp_path))
    try:
        experiments = ExperimentRepository(connection)
        runs = RunRepository(connection)
        failures = FailureRepository(connection)
        experiments.create(
            experiment_id="exp_quality",
            project_id=init_result.project_id,
            name="quality",
            created_at=NOW.isoformat().replace("+00:00", "Z"),
            updated_at=NOW.isoformat().replace("+00:00", "Z"),
        )
        _create_run(
            runs,
            run_id="run_good_dataset_in_config",
            experiment_id="exp_quality",
            config={"dataset_id": "fashion_mnist", "lr": 0.001},
            metric=0.90,
            timestamp="2026-05-31T00:10:00Z",
        )
        _create_run(
            runs,
            run_id="run_failed_with_metric",
            experiment_id="exp_quality",
            config={"dataset_id": "fashion_mnist", "lr": 1.0},
            metric=0.20,
            timestamp="2026-05-31T00:11:00Z",
            status="failed",
            exit_code=1,
        )
        failures.create(
            failure_id="failure_failed_with_metric",
            run_id="run_failed_with_metric",
            error_type="config_error",
            description="PRIVATE failed metric text",
            root_cause="PRIVATE root cause",
            lesson="PRIVATE lesson",
            severity="high",
            tags=["config"],
            source="user_confirmed",
            created_at="2026-05-31T00:12:00Z",
        )
        _create_run(
            runs,
            run_id="run_strong_mislabeled",
            experiment_id="exp_quality",
            config={"dataset_id": "fashion_mnist", "lr": 0.1},
            metric=0.88,
            timestamp="2026-05-31T00:13:00Z",
        )
        failures.create(
            failure_id="failure_strong_mislabeled",
            run_id="run_strong_mislabeled",
            error_type="convergence",
            description="PRIVATE mislabeled text",
            root_cause="PRIVATE root cause",
            lesson="PRIVATE lesson",
            severity="medium",
            tags=["label"],
            source="user_confirmed",
            created_at="2026-05-31T00:14:00Z",
        )
    finally:
        connection.close()

    payload = recommendation_list_payload(tmp_path)
    warnings = "\n".join(payload["warnings"])

    assert "Dataset ids appear in run config metadata" in warnings
    assert "failed run(s) include primary metric values" in warnings
    assert "failure-labeled run(s) have strong primary metrics" in warnings
    assert "PRIVATE" not in json.dumps(payload, sort_keys=True)


def test_recommendation_detail_and_missing_id(tmp_path) -> None:
    """recommendation CLI detail lookup should reuse generated ids and fail cleanly."""

    _seed_recommendation_project(tmp_path)
    first_id = recommendation_list_payload(tmp_path)["recommendations"][0]["recommendation_id"]

    detail = recommendation_detail_payload(tmp_path, recommendation_id=first_id)

    assert detail["schema_version"] == "recommendation-detail-result-v1"
    assert detail["recommendation"]["recommendation_id"] == first_id
    with pytest.raises(PmemValidationError, match="Recommendation id cannot be blank"):
        recommendation_detail_payload(tmp_path, recommendation_id=" ")
    with pytest.raises(PmemNotFoundError, match="Recommendation candidate was not found"):
        recommendation_detail_payload(tmp_path, recommendation_id="rec_missing")


def test_recommendation_export_is_private_and_path_safe(tmp_path) -> None:
    """recommendation CLI export should write 0600 JSON and reject traversal."""

    _seed_recommendation_project(tmp_path)

    result = export_recommendations(tmp_path, output_path="exports/recommendations.json")
    raw_json = (tmp_path / "exports" / "recommendations.json").read_text(encoding="utf-8")

    assert result.payload["ok"] is True
    assert result.payload["recommendation_count"] == 5
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    assert "PRIVATE" not in raw_json
    assert "python train.py" not in raw_json
    with pytest.raises(PmemSecurityError):
        export_recommendations(tmp_path, output_path="../escape.json")


def test_recommendation_limit_and_export_path_validation(tmp_path) -> None:
    """recommendation CLI should reject unsafe limits and export paths before writing files."""

    init_project(tmp_path, project_name="recommend-path-policy", primary_metric="accuracy")
    (tmp_path / "existing_dir").mkdir()

    with pytest.raises(PmemValidationError, match="at least 1"):
        recommendation_list_payload(tmp_path, max_recommendations=0)
    with pytest.raises(PmemValidationError, match="50 or less"):
        recommendation_list_payload(tmp_path, max_recommendations=51)
    with pytest.raises(PmemValidationError, match="cannot be blank"):
        export_recommendations(tmp_path, output_path=" ")

    unsafe_paths = (
        "/tmp/recommendations.json",
        "C:\\temp\\recommendations.json",
        ".pmem/recommendations.json",
        "bad\x00name.json",
        "bad\\name.json",
        "existing_dir",
    )
    for unsafe_path in unsafe_paths:
        with pytest.raises(PmemSecurityError):
            export_recommendations(tmp_path, output_path=unsafe_path)


def test_recommendation_export_rejects_symlink_paths(tmp_path) -> None:
    """recommendation CLI export should reject symlink output and symlink parent paths."""

    init_project(tmp_path, project_name="recommend-symlink-policy", primary_metric="accuracy")
    target = tmp_path / "target.json"
    target.write_text("target", encoding="utf-8")
    output_link = tmp_path / "linked.json"
    output_link.symlink_to(target)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    linked_dir = tmp_path / "linked-dir"
    linked_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(PmemSecurityError, match="symlink"):
        export_recommendations(tmp_path, output_path="linked.json")
    with pytest.raises(PmemSecurityError, match="symlinks"):
        export_recommendations(tmp_path, output_path="linked-dir/out.json")


def test_private_json_write_cleans_temp_file_on_os_error(monkeypatch, tmp_path) -> None:
    """Atomic recommendation export writes should clean temp files on failure."""

    class FixedUuid:
        hex = "fixed"

    output = tmp_path / "recommendations.json"
    temp_path = tmp_path / ".recommendations.json.fixed.tmp"
    temp_path.write_text("partial", encoding="utf-8")
    monkeypatch.setattr(operations.uuid, "uuid4", lambda: FixedUuid())
    monkeypatch.setattr(
        operations.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(PmemPersistenceError, match="could not be written"):
        operations._write_private_json(output, {"ok": True})
    assert not temp_path.exists()


def test_display_path_falls_back_to_filename_for_external_paths(tmp_path) -> None:
    """Display paths should not leak external absolute directory structure."""

    assert operations._display_path(tmp_path, Path("/private/tmp/outside.json")) == "outside.json"


def _seed_recommendation_project(tmp_path) -> None:
    init_result = init_project(
        tmp_path,
        project_name="recommendation-cli-service",
        primary_metric="accuracy",
        metric_direction="max",
    )
    connection = connect_database(project_database_path(tmp_path))
    try:
        experiments = ExperimentRepository(connection)
        runs = RunRepository(connection)
        failures = FailureRepository(connection)
        for experiment_id in ("exp_a", "exp_b", "exp_c"):
            experiments.create(
                experiment_id=experiment_id,
                project_id=init_result.project_id,
                name=experiment_id,
                created_at=NOW.isoformat().replace("+00:00", "Z"),
                updated_at=NOW.isoformat().replace("+00:00", "Z"),
            )
        for index, value in enumerate((0.78, 0.79, 0.80, 0.81, 0.82, 0.79, 0.80, 0.81)):
            _create_run(
                runs,
                run_id=f"run_normal_{index}",
                experiment_id="exp_a",
                config={"family": "normal", "index": index},
                metric=value,
                timestamp=f"2026-05-31T00:00:{index:02d}Z",
            )
        _create_run(
            runs,
            run_id="run_outlier_high",
            experiment_id="exp_a",
            config={"family": "outlier"},
            metric=0.99,
            timestamp="2026-05-31T00:00:59Z",
        )
        for index, value in enumerate((0.45, 0.92, 0.50, 0.95)):
            _create_run(
                runs,
                run_id=f"run_var_{index}",
                experiment_id="exp_b",
                config={"family": "variance"},
                metric=value,
                timestamp=f"2026-05-31T00:01:{index:02d}Z",
            )
        for index in range(5):
            run_id = f"run_bad_{index}"
            _create_run(
                runs,
                run_id=run_id,
                experiment_id="exp_c",
                config={"family": "bad"},
                metric=0.30 + index * 0.01,
                timestamp=f"2026-05-31T00:02:{index:02d}Z",
                status="failed",
                exit_code=1,
            )
            failures.create(
                failure_id=f"failure_bad_{index}",
                run_id=run_id,
                error_type="MetricRegression",
                description="PRIVATE failure text",
                root_cause="PRIVATE root cause",
                lesson="PRIVATE lesson",
                severity="high",
                tags=["metric"],
                source="user_confirmed",
                created_at=f"2026-05-31T00:03:{index:02d}Z",
            )
        _create_run(
            runs,
            run_id="run_promote_c",
            experiment_id="exp_c",
            config={"family": "promote"},
            metric=0.90,
            timestamp="2026-05-31T00:04:00Z",
        )
    finally:
        connection.close()


def _create_run(
    runs: RunRepository,
    *,
    run_id: str,
    experiment_id: str,
    config: dict[str, object],
    metric: float,
    timestamp: str,
    status: str = "success",
    exit_code: int = 0,
) -> None:
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    runs.create(
        run_id=run_id,
        experiment_id=experiment_id,
        command="python train.py",
        cwd=".",
        exit_code=exit_code,
        status=status,
        config=config,
        config_hash=compute_text_hash(config_json),
        metrics={"accuracy": metric},
        timestamp=timestamp,
    )
