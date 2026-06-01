"""Focused tests for summary project summary defensive branches."""

from __future__ import annotations

from pmem.repositories.runs import RunRecord
from pmem.summary.project_summary import _best_run, _run_metric, _target_value


def test_target_value_bool_returns_none() -> None:
    assert _target_value('{"target_value": true}') is None


def test_target_value_string_returns_none() -> None:
    assert _target_value('{"target_value": "high"}') is None


def test_best_run_no_metric_returns_none() -> None:
    assert _best_run((_run_with_metrics('{"accuracy": 0.9}'),), metric=None, direction="max") == (
        None,
        None,
    )


def test_run_metric_bool_returns_none() -> None:
    assert _run_metric(_run_with_metrics('{"f1": true}'), "f1") is None


def _run_with_metrics(metrics_json: str) -> RunRecord:
    return RunRecord(
        run_id="run_1",
        experiment_id="exp_1",
        name=None,
        command="python train.py",
        cwd="/project",
        exit_code=0,
        status="success",
        duration_sec=0.1,
        seed=None,
        stdout_path=None,
        stderr_path=None,
        stdout_preview=None,
        stderr_preview=None,
        env_json="{}",
        config_json="{}",
        config_hash=None,
        metrics_json=metrics_json,
        artifacts_json="[]",
        git_json="{}",
        evaluation_json="{}",
        failure_candidates_json="[]",
        timestamp="2026-05-18T00:00:00Z",
    )
