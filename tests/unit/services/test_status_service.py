"""Unit tests for the read-only status service (STS-002)."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from pmem.errors import PmemError, PmemPersistenceError, PmemValidationError
from pmem.services import status_service
from pmem.services.graph_operations import build_graph_artifact
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command
from pmem.services.status_service import (
    CollectedStatusState,
    assemble_status_payload,
    collect_status_state,
)
from pmem.status import (
    GraphState,
    RecommendationMode,
    StatusNextAction,
    StatusPayload,
    TargetStatus,
)
from pmem.summary import ProjectSummary

_NEXT_ACTION = StatusNextAction(
    action_id="rebuild_graph",
    reason="The evidence graph changed.",
    suggested_command="pmem graph build",
    related_entity_id=None,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _summary(**overrides: Any) -> ProjectSummary:
    base: dict[str, Any] = {
        "project_id": "proj_" + "0" * 32,
        "project_name": "demo-project",
        "objective": "Train a baseline",
        "primary_metric": "accuracy",
        "metric_direction": "max",
        "target_value": 0.9,
        "run_count": 5,
        "successful_run_count": 4,
        "failed_run_count": 1,
        "best_run_id": "run_" + "a" * 32,
        "best_metric_value": 0.95,
        "target_status": "met",
        "tracked_path_count": 2,
        "failure_count": 1,
        "decision_count": 0,
        "note_count": 0,
        "baseline_run_id": "run_" + "b" * 32,
        "timeline": (),
        "warnings": (),
    }
    base.update(overrides)
    return ProjectSummary(**base)


def _patch_summary(monkeypatch: pytest.MonkeyPatch, summary: ProjectSummary, root: Path) -> None:
    monkeypatch.setattr(status_service, "get_project_summary_readonly", lambda _root: summary)


def _init(root: Path, **kwargs: Any) -> None:
    init_project(root, project_name=kwargs.pop("project_name", "demo"), **kwargs)


def _run_ok(root: Path, marker: str = "ok") -> None:
    run_command(root, [sys.executable, "-c", f"print('{marker}')"])


def _make_schema_outdated(root: Path) -> None:
    db_path = root / ".pmem" / "pmem.db"
    connection = sqlite3.connect(db_path)
    try:
        versions = [
            row[0]
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        ]
        connection.execute("DELETE FROM schema_migrations WHERE version = ?", (versions[-1],))
        connection.commit()
    finally:
        connection.close()


def _fs_snapshot(root: Path) -> tuple[Any, ...]:
    db_path = root / ".pmem" / "pmem.db"
    stat = db_path.stat()
    return (
        sorted(os.listdir(root / ".pmem")),
        db_path.read_bytes(),
        stat.st_mtime_ns,
        stat.st_mode,
    )


# --------------------------------------------------------------------------- #
# API / seam                                                                   #
# --------------------------------------------------------------------------- #
def test_collect_returns_state_without_next_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_summary(monkeypatch, _summary(), tmp_path)
    state = collect_status_state(tmp_path)
    assert isinstance(state, CollectedStatusState)
    assert not hasattr(state, "next_action")


def test_assemble_requires_caller_supplied_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_summary(monkeypatch, _summary(), tmp_path)
    payload = assemble_status_payload(collect_status_state(tmp_path), next_action=_NEXT_ACTION)
    assert isinstance(payload, StatusPayload)
    assert payload.next_action.action_id == "rebuild_graph"
    assert payload.schema_version == "status-v1"
    assert payload.database_mutation is False
    assert payload.network is False
    assert payload.raw_text_in_output is False


def test_service_does_not_import_cli() -> None:
    source = Path(status_service.__file__).read_text(encoding="utf-8")
    assert "pmem.cli" not in source
    assert "import typer" not in source


# --------------------------------------------------------------------------- #
# Summary mapping                                                              #
# --------------------------------------------------------------------------- #
def test_summary_mapping_populates_all_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary = _summary()
    _patch_summary(monkeypatch, summary, tmp_path)
    state = collect_status_state(tmp_path)
    assert state.project.project_id == summary.project_id
    assert state.project.objective == "Train a baseline"
    assert state.metric.primary_metric == "accuracy"
    assert state.metric.target_status is TargetStatus.MET
    assert state.counts.run_count == 5
    assert state.best_run.run_id == summary.best_run_id
    assert state.best_run.metric_value == 0.95
    assert state.baseline.run_id == summary.baseline_run_id


_TARGET_CASES: dict[str, dict[str, Any]] = {
    "no_runs": {
        "run_count": 0,
        "successful_run_count": 0,
        "failed_run_count": 0,
        "tracked_path_count": 0,
        "primary_metric": None,
        "metric_direction": None,
        "target_value": None,
        "best_run_id": None,
        "best_metric_value": None,
        "baseline_run_id": None,
        "target_status": "no_runs",
    },
    "no_successful_runs": {
        "run_count": 4,
        "successful_run_count": 0,
        "failed_run_count": 3,
        "best_run_id": None,
        "best_metric_value": None,
        "target_status": "no_successful_runs",
    },
    "not_configured": {
        "primary_metric": None,
        "metric_direction": None,
        "target_value": None,
        "best_run_id": None,
        "best_metric_value": None,
        "target_status": "not_configured",
    },
    "no_metric": {
        "best_run_id": None,
        "best_metric_value": None,
        "target_status": "no_metric",
    },
    "not_met": {
        "best_metric_value": 0.5,
        "target_status": "not_met",
    },
}


@pytest.mark.parametrize("case", sorted(_TARGET_CASES))
def test_target_status_mapping_matches_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    summary = _summary(**_TARGET_CASES[case])
    _patch_summary(monkeypatch, summary, tmp_path)
    payload = assemble_status_payload(collect_status_state(tmp_path), next_action=_NEXT_ACTION)
    assert payload.metric.target_status.value == _TARGET_CASES[case]["target_status"]


# --------------------------------------------------------------------------- #
# Read-only: outdated schema must raise (never migrate)                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("evaluate", [False, True])
@pytest.mark.parametrize("build_graph", [False, True])
def test_outdated_schema_raises_without_migrating(
    tmp_path: Path, evaluate: bool, build_graph: bool
) -> None:
    _init(tmp_path, project_name="outdated")
    _run_ok(tmp_path)
    if build_graph:
        build_graph_artifact(tmp_path)
    _make_schema_outdated(tmp_path)
    before = _fs_snapshot(tmp_path)

    with pytest.raises(PmemError):
        collect_status_state(tmp_path, evaluate_recommendations=evaluate)

    after = _fs_snapshot(tmp_path)
    assert after == before
    assert not any(name.endswith(".bak") for name in after[0])


def test_uninitialized_project_raises(tmp_path: Path) -> None:
    with pytest.raises(PmemError):
        collect_status_state(tmp_path)


def _recursive_snapshot(root: Path) -> dict[str, tuple[str, bytes, int, int]]:
    snapshot: dict[str, tuple[str, bytes, int, int]] = {}
    for path in sorted((root / ".pmem").rglob("*")):
        stat = path.stat()
        if path.is_file():
            snapshot[str(path.relative_to(root))] = (
                "file",
                path.read_bytes(),
                stat.st_mtime_ns,
                stat.st_mode,
            )
        elif path.is_dir():
            snapshot[str(path.relative_to(root))] = (
                "directory",
                b"",
                stat.st_mtime_ns,
                stat.st_mode,
            )
    return snapshot


@pytest.mark.parametrize("evaluate", [False, True])
@pytest.mark.parametrize("build_graph", [False, True])
@pytest.mark.parametrize("start_mode", [0o600, 0o644])
def test_status_is_strictly_read_only(
    tmp_path: Path, evaluate: bool, build_graph: bool, start_mode: int
) -> None:
    _init(tmp_path, project_name="ro", current_objective="Train", primary_metric="accuracy")
    _run_ok(tmp_path)
    if build_graph:
        build_graph_artifact(tmp_path)
    os.chmod(tmp_path / ".pmem" / "pmem.db", start_mode)
    before = _recursive_snapshot(tmp_path)

    collect_status_state(tmp_path, evaluate_recommendations=evaluate)

    after = _recursive_snapshot(tmp_path)
    assert after == before
    # status must not "repair" a loose mode back to 0600
    assert (tmp_path / ".pmem" / "pmem.db").stat().st_mode & 0o777 == start_mode
    assert not any(name.endswith(".bak") for name in after)


@pytest.mark.parametrize("evaluate", [False, True])
@pytest.mark.parametrize("build_graph", [False, True])
def test_status_checkpointed_wal_creates_no_sidecars(
    tmp_path: Path, evaluate: bool, build_graph: bool
) -> None:
    _init(tmp_path, project_name="wal-checkpointed")
    _run_ok(tmp_path)
    if build_graph:
        build_graph_artifact(tmp_path)
    db_path = tmp_path / ".pmem" / "pmem.db"
    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    finally:
        connection.close()
    assert not list((tmp_path / ".pmem").glob("pmem.db-*"))
    before = _recursive_snapshot(tmp_path)

    collect_status_state(tmp_path, evaluate_recommendations=evaluate)

    assert _recursive_snapshot(tmp_path) == before
    assert not list((tmp_path / ".pmem").glob("pmem.db-*"))


def test_status_rejects_active_wal_without_touching_files(tmp_path: Path) -> None:
    _init(tmp_path, project_name="wal-active")
    db_path = tmp_path / ".pmem" / "pmem.db"
    writer = sqlite3.connect(db_path)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE wal_probe (value TEXT)")
        writer.execute("INSERT INTO wal_probe VALUES ('uncheckpointed')")
        writer.commit()
        assert list((tmp_path / ".pmem").glob("pmem.db-*"))
        before = _recursive_snapshot(tmp_path)

        with pytest.raises(PmemPersistenceError, match="active SQLite sidecar state"):
            collect_status_state(tmp_path)

        assert _recursive_snapshot(tmp_path) == before
    finally:
        writer.close()


def test_corrupt_database_raises_safe_pmem_error_without_mutation(tmp_path: Path) -> None:
    _init(tmp_path, project_name="corrupt")
    db_path = tmp_path / ".pmem" / "pmem.db"
    db_path.write_bytes(b"not sqlite at all")
    before = _recursive_snapshot(tmp_path)

    with pytest.raises(PmemPersistenceError) as exc_info:
        collect_status_state(tmp_path)

    assert _recursive_snapshot(tmp_path) == before
    assert "file is not a database" not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


def test_checksum_tamper_raises_without_touching_fs(tmp_path: Path) -> None:
    _init(tmp_path, project_name="tamper")
    _run_ok(tmp_path)
    db_path = tmp_path / ".pmem" / "pmem.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("UPDATE schema_migrations SET checksum = ?", ("0" * 64,))
        connection.commit()
    finally:
        connection.close()
    before = _recursive_snapshot(tmp_path)
    with pytest.raises(PmemError):
        collect_status_state(tmp_path)
    assert _recursive_snapshot(tmp_path) == before


def test_database_symlink_is_rejected(tmp_path: Path) -> None:
    _init(tmp_path, project_name="db-symlink")
    db_path = tmp_path / ".pmem" / "pmem.db"
    external = tmp_path.parent / "external.db"
    db_path.rename(external)
    try:
        os.symlink(external, db_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")
    with pytest.raises(PmemError):
        collect_status_state(tmp_path)


def test_config_symlink_is_rejected(tmp_path: Path) -> None:
    _init(tmp_path, project_name="config-symlink")
    config_path = tmp_path / ".pmem" / "config.yaml"
    external = tmp_path.parent / "external.yaml"
    config_path.rename(external)
    try:
        os.symlink(external, config_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")
    with pytest.raises(PmemError):
        collect_status_state(tmp_path)


def test_pmem_directory_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    alias = tmp_path / "alias"
    source.mkdir()
    alias.mkdir()
    _init(source, project_name="pmem-dir-symlink")
    (alias / ".pmem").symlink_to(source / ".pmem", target_is_directory=True)

    with pytest.raises(PmemError):
        collect_status_state(alias)


def test_project_root_symlink_is_allowed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    alias = tmp_path / "alias"
    source.mkdir()
    _init(source, project_name="root-symlink")
    alias.symlink_to(source, target_is_directory=True)

    state = collect_status_state(alias)

    assert state.project.project_name == "root-symlink"


def test_readonly_summary_matches_normal_summary(tmp_path: Path) -> None:
    from pmem.summary import get_project_summary, get_project_summary_readonly

    _init(
        tmp_path,
        project_name="equiv",
        current_objective="Train",
        primary_metric="accuracy",
        metric_direction="max",
        target_value=0.9,
    )
    _run_ok(tmp_path)
    normal = get_project_summary(tmp_path)
    readonly = get_project_summary_readonly(tmp_path)
    for field in (
        "project_id",
        "project_name",
        "objective",
        "primary_metric",
        "metric_direction",
        "target_value",
        "run_count",
        "successful_run_count",
        "failed_run_count",
        "best_run_id",
        "best_metric_value",
        "target_status",
        "tracked_path_count",
        "failure_count",
        "decision_count",
        "note_count",
        "baseline_run_id",
    ):
        assert getattr(normal, field) == getattr(readonly, field), field


# --------------------------------------------------------------------------- #
# Graph state (real projects; fingerprint-based)                              #
# --------------------------------------------------------------------------- #
def test_graph_missing(tmp_path: Path) -> None:
    _init(tmp_path, project_name="graph-missing")
    state = collect_status_state(tmp_path)
    assert state.graph.state is GraphState.MISSING
    assert state.graph.node_count is None
    assert state.graph.reason_code == "graph_not_built"
    assert any(w.code == "graph_missing" for w in state.warnings)


def test_graph_current(tmp_path: Path) -> None:
    _init(tmp_path, project_name="graph-current")
    _run_ok(tmp_path)
    build_graph_artifact(tmp_path)
    state = collect_status_state(tmp_path)
    assert state.graph.state is GraphState.CURRENT
    assert state.graph.node_count is not None
    assert state.graph.reason_code is None
    assert not any(w.source.value == "graph" for w in state.warnings)


def test_graph_stale_after_new_run(tmp_path: Path) -> None:
    _init(tmp_path, project_name="graph-stale")
    _run_ok(tmp_path)
    build_graph_artifact(tmp_path)
    _run_ok(tmp_path, "ok2")
    state = collect_status_state(tmp_path)
    assert state.graph.state is GraphState.STALE
    assert state.graph.reason_code == "graph_source_changed"
    assert any(w.code == "graph_stale" for w in state.warnings)


def test_graph_invalid_when_corrupt(tmp_path: Path) -> None:
    _init(tmp_path, project_name="graph-corrupt")
    _run_ok(tmp_path)
    build_graph_artifact(tmp_path)
    (tmp_path / ".pmem" / "graph.json").write_text("{ not valid json", encoding="utf-8")
    state = collect_status_state(tmp_path)
    assert state.graph.state is GraphState.INVALID
    assert state.graph.reason_code == "graph_unreadable"
    assert any(w.code == "graph_invalid" for w in state.warnings)


def test_graph_unknown_when_fingerprint_missing(tmp_path: Path) -> None:
    _init(tmp_path, project_name="graph-unknown")
    _run_ok(tmp_path)
    build_graph_artifact(tmp_path)
    graph_path = tmp_path / ".pmem" / "graph.json"
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    payload.get("metadata", {}).pop("source_fingerprint", None)
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    state = collect_status_state(tmp_path)
    assert state.graph.state is GraphState.UNKNOWN
    assert state.graph.reason_code == "graph_fingerprint_missing"


_MALFORMED_FINGERPRINTS = {
    "empty": "",
    "wrong_prefix": "md5:" + "a" * 64,
    "too_short": "sha256:" + "a" * 10,
    "non_hex": "sha256:" + "g" * 64,
}


@pytest.mark.parametrize("case", sorted(_MALFORMED_FINGERPRINTS))
def test_graph_malformed_fingerprint_is_invalid_or_unknown(tmp_path: Path, case: str) -> None:
    _init(tmp_path, project_name="graph-fp")
    _run_ok(tmp_path)
    build_graph_artifact(tmp_path)
    graph_path = tmp_path / ".pmem" / "graph.json"
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    payload["metadata"]["source_fingerprint"] = _MALFORMED_FINGERPRINTS[case]
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    state = collect_status_state(tmp_path)
    # empty -> unknown (missing); any non-empty malformed value -> invalid, never stale
    if case == "empty":
        assert state.graph.state is GraphState.UNKNOWN
    else:
        assert state.graph.state is GraphState.INVALID
        assert state.graph.reason_code == "graph_fingerprint_invalid"


def test_graph_count_mismatch_is_invalid(tmp_path: Path) -> None:
    _init(tmp_path, project_name="graph-count")
    _run_ok(tmp_path)
    build_graph_artifact(tmp_path)
    graph_path = tmp_path / ".pmem" / "graph.json"
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    payload["counts"]["nodes"] = 999
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    state = collect_status_state(tmp_path)
    assert state.graph.state is GraphState.INVALID
    assert state.graph.reason_code == "graph_count_mismatch"


def test_graph_symlink_is_invalid_and_unread(tmp_path: Path) -> None:
    _init(tmp_path, project_name="graph-symlink")
    outside = tmp_path.parent / "outside-graph.json"
    outside.write_text("{}", encoding="utf-8")
    graph_path = tmp_path / ".pmem" / "graph.json"
    try:
        os.symlink(outside, graph_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")
    state = collect_status_state(tmp_path)
    assert state.graph.state is GraphState.INVALID
    assert state.graph.reason_code == "graph_symlink"
    assert any(w.code == "graph_symlink" for w in state.warnings)


def test_database_error_in_fingerprint_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _init(tmp_path, project_name="graph-dberr")
    _run_ok(tmp_path)
    build_graph_artifact(tmp_path)

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise PmemPersistenceError("database is locked")

    monkeypatch.setattr(status_service, "compute_graph_source_fingerprint_readonly", _raise)
    with pytest.raises(PmemPersistenceError):
        collect_status_state(tmp_path)


def test_collect_does_not_mutate_current_db(tmp_path: Path) -> None:
    _init(tmp_path, project_name="no-mutate")
    _run_ok(tmp_path)
    build_graph_artifact(tmp_path)
    graph_path = tmp_path / ".pmem" / "graph.json"
    graph_before = (graph_path.stat().st_mtime_ns, graph_path.read_bytes())
    db_before = _fs_snapshot(tmp_path)

    collect_status_state(tmp_path)
    collect_status_state(tmp_path, evaluate_recommendations=True)

    assert (graph_path.stat().st_mtime_ns, graph_path.read_bytes()) == graph_before
    assert _fs_snapshot(tmp_path) == db_before


# --------------------------------------------------------------------------- #
# Recommendation policy                                                        #
# --------------------------------------------------------------------------- #
def test_default_does_not_call_generator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_summary(monkeypatch, _summary(), tmp_path)

    def _boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("recommendation generation must not run by default")

    monkeypatch.setattr(status_service, "recommendation_list_payload", _boom)
    state = collect_status_state(tmp_path)
    assert state.recommendations.mode is RecommendationMode.NOT_EVALUATED
    assert state.recommendations.candidate_count is None
    assert state.recommendations.active_count is None


def test_evaluate_maps_generated_on_demand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_summary(monkeypatch, _summary(), tmp_path)
    monkeypatch.setattr(
        status_service,
        "recommendation_list_payload",
        lambda _root, *, max_recommendations: {"recommendation_count": 4, "warnings": []},
    )
    state = collect_status_state(tmp_path, evaluate_recommendations=True)
    assert state.recommendations.mode is RecommendationMode.GENERATED_ON_DEMAND
    assert state.recommendations.candidate_count == 4
    assert state.recommendations.active_count is None


def test_sparse_real_project_generated_on_demand(tmp_path: Path) -> None:
    _init(tmp_path, project_name="sparse")
    _run_ok(tmp_path)
    state = collect_status_state(tmp_path, evaluate_recommendations=True)
    assert state.recommendations.mode is RecommendationMode.GENERATED_ON_DEMAND
    assert state.recommendations.candidate_count == 0
    assert state.recommendations.active_count is None


@pytest.mark.parametrize("bad_limit", [0, -1, 51, 100])
def test_invalid_recommendation_limit_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_limit: int
) -> None:
    _patch_summary(monkeypatch, _summary(), tmp_path)
    with pytest.raises(PmemValidationError):
        collect_status_state(tmp_path, max_recommendations=bad_limit)


_MALFORMED_RECO_PAYLOADS: dict[str, Any] = {
    "missing_key": {"warnings": []},
    "bool_count": {"recommendation_count": True},
    "string_count": {"recommendation_count": "3"},
    "negative_count": {"recommendation_count": -1},
    "over_limit": {"recommendation_count": 99},
    "not_a_dict": ["nope"],
}


@pytest.mark.parametrize("case", sorted(_MALFORMED_RECO_PAYLOADS))
def test_malformed_recommendation_payload_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    _patch_summary(monkeypatch, _summary(), tmp_path)
    monkeypatch.setattr(
        status_service,
        "recommendation_list_payload",
        lambda _root, *, max_recommendations: _MALFORMED_RECO_PAYLOADS[case],
    )
    with pytest.raises(PmemValidationError):
        collect_status_state(tmp_path, evaluate_recommendations=True, max_recommendations=5)


# --------------------------------------------------------------------------- #
# Warnings                                                                     #
# --------------------------------------------------------------------------- #
def test_empty_project_warning_codes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    summary = _summary(**_TARGET_CASES["no_runs"])
    _patch_summary(monkeypatch, summary, tmp_path)
    codes = {w.code for w in collect_status_state(tmp_path).warnings}
    assert {"no_tracked_paths", "no_runs", "graph_missing"} <= codes


def test_warnings_are_deterministic_and_sorted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary = _summary(target_status="not_met", best_metric_value=0.5, baseline_run_id=None)
    _patch_summary(monkeypatch, summary, tmp_path)
    first = collect_status_state(tmp_path).warnings
    second = collect_status_state(tmp_path).warnings
    assert first == second
    severity_rank = {"error": 0, "warning": 1, "info": 2}
    ranked = [(severity_rank[w.severity.value], w.source.value, w.code) for w in first]
    assert ranked == sorted(ranked)


def test_recommendation_warnings_are_typed_not_raw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_summary(monkeypatch, _summary(), tmp_path)
    raw = [
        "No opposing evidence was linked for this recommendation.",
        "No opposing evidence was linked for this recommendation.",
        "dataset metadata placement looks off",
    ]
    monkeypatch.setattr(
        status_service,
        "recommendation_list_payload",
        lambda _root, *, max_recommendations: {"recommendation_count": 2, "warnings": raw},
    )
    warnings = collect_status_state(tmp_path, evaluate_recommendations=True).warnings
    reco_codes = [w.code for w in warnings if w.source.value == "recommendation"]
    assert "recommendation_evidence_incomplete" in reco_codes
    assert "dataset_metadata_placement" in reco_codes
    assert reco_codes.count("recommendation_evidence_incomplete") == 1
    for warning in warnings:
        assert "opposing evidence" not in warning.message.lower()


# --------------------------------------------------------------------------- #
# Privacy / redaction                                                          #
# --------------------------------------------------------------------------- #
def test_path_like_project_text_is_redacted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary = _summary(objective="see /Users/private/secret", primary_metric="macro F1")
    _patch_summary(monkeypatch, summary, tmp_path)
    state = collect_status_state(tmp_path)
    assert state.metric.primary_metric == "macro F1"
    assert state.project.objective is not None
    assert state.project.objective.startswith("redacted_objective_")
    assert any(w.code == "status_text_redacted" for w in state.warnings)


def test_redaction_is_deterministic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    summary = _summary(objective="/etc/passwd leak")
    _patch_summary(monkeypatch, summary, tmp_path)
    first = collect_status_state(tmp_path).project.objective
    second = collect_status_state(tmp_path).project.objective
    assert first == second


_SENTINEL_PATHS = {
    "posix": "leak /Users/private/secret here",
    "windows": r"leak C:\private\secret here",
    "file_uri": "leak file:///Users/private/secret here",
    "prefix_colon": "path:/Users/private/secret",
    "prefix_equals": "source=/Users/private/secret",
    "prefix_colon_windows": r"path:C:\private\secret",
    "wrapped_parens": "(/Users/private/secret)",
}


@pytest.mark.parametrize("case", sorted(_SENTINEL_PATHS))
def test_real_project_path_text_never_serialized(tmp_path: Path, case: str) -> None:
    sentinel = _SENTINEL_PATHS[case]
    _init(tmp_path, project_name="real-privacy", current_objective=sentinel)
    _run_ok(tmp_path)
    state = collect_status_state(tmp_path)
    payload_json = assemble_status_payload(state, next_action=_NEXT_ACTION).model_dump_json()
    assert sentinel not in payload_json
    assert "/Users/private/secret" not in payload_json
    assert "C:\\private\\secret" not in payload_json
    assert str(tmp_path) not in payload_json
    assert any(w.code == "status_text_redacted" for w in state.warnings)


def test_safe_metric_with_space_preserved_real_project(tmp_path: Path) -> None:
    _init(tmp_path, project_name="safe-metric", primary_metric="macro F1")
    _run_ok(tmp_path)
    state = collect_status_state(tmp_path)
    assert state.metric.primary_metric == "macro F1"


@pytest.mark.parametrize(
    "safe_text",
    [
        "https://example.com/paper",
        "http://localhost:8000",
        "metric:accuracy",
        "ratio:1/2",
    ],
)
def test_safe_non_file_text_is_preserved_real_project(tmp_path: Path, safe_text: str) -> None:
    _init(tmp_path, project_name="safe-url", current_objective=safe_text)

    state = collect_status_state(tmp_path)

    assert state.project.objective == safe_text
    assert not any(w.code == "status_text_redacted" for w in state.warnings)
