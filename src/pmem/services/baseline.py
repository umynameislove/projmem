"""baseline baseline tracking service workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.repositories.experiments import ExperimentRecord, ExperimentRepository
from pmem.repositories.runs import RunRecord, RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_context import require_project_context


@dataclass(frozen=True)
class BaselineSetResult:
    """User-facing result for marking a baseline run."""

    run_id: str
    experiment_id: str
    metric_count: int


@dataclass(frozen=True)
class BaselineComparisonResult:
    """User-facing result for comparing a run to the experiment baseline."""

    run_id: str
    experiment_id: str
    baseline_run_id: str
    metric_deltas: dict[str, float]


def set_baseline_run(project_root: str | Path, *, run_id: str) -> BaselineSetResult:
    """Mark an existing run as the baseline for its experiment."""

    context = require_project_context(project_root)
    connection = connect_database(project_database_path(context.root))
    try:
        runs = RunRepository(connection)
        experiments = ExperimentRepository(connection)
        run = _require_project_run(runs, experiments, context.project.id, run_id)
        experiment = _require_experiment(experiments, run.experiment_id)
        metrics = _numeric_metrics(json.loads(run.metrics_json))
        metadata = _metadata_object(experiment.metadata_json)
        metadata["baseline_run_id"] = run.run_id
        metadata["baseline_set_at"] = _utc_now_iso()
        metadata["baseline_metrics"] = metrics
        experiments.update_metadata(
            experiment_id=experiment.id,
            metadata=metadata,
            updated_at=_utc_now_iso(),
        )
    finally:
        connection.close()

    return BaselineSetResult(
        run_id=run.run_id,
        experiment_id=run.experiment_id,
        metric_count=len(metrics),
    )


def compare_run_to_baseline(
    project_root: str | Path,
    *,
    run_id: str,
) -> BaselineComparisonResult:
    """Compare one run's numeric metrics against the experiment baseline."""

    context = require_project_context(project_root)
    connection = connect_database(project_database_path(context.root))
    try:
        runs = RunRepository(connection)
        experiments = ExperimentRepository(connection)
        run = _require_project_run(runs, experiments, context.project.id, run_id)
        experiment = _require_experiment(experiments, run.experiment_id)
        metadata = _metadata_object(experiment.metadata_json)
        baseline_run_id = metadata.get("baseline_run_id")
        if not isinstance(baseline_run_id, str) or not baseline_run_id.strip():
            raise PmemNotFoundError("No baseline run is set for this experiment.")

        baseline = _require_project_run(runs, experiments, context.project.id, baseline_run_id)
        if baseline.experiment_id != run.experiment_id:
            raise PmemValidationError("Baseline run belongs to a different experiment.")

        baseline_metrics = _numeric_metrics(json.loads(baseline.metrics_json))
        current_metrics = _numeric_metrics(json.loads(run.metrics_json))
        metric_deltas = {
            metric: current_metrics[metric] - baseline_metrics[metric]
            for metric in sorted(current_metrics.keys() & baseline_metrics.keys())
        }
        evaluation = _metadata_object(run.evaluation_json)
        evaluation["baseline_comparison"] = {
            "baseline_run_id": baseline.run_id,
            "metric_deltas": metric_deltas,
        }
        runs.update_evaluation(run_id=run.run_id, evaluation=evaluation)
    finally:
        connection.close()

    return BaselineComparisonResult(
        run_id=run.run_id,
        experiment_id=run.experiment_id,
        baseline_run_id=baseline.run_id,
        metric_deltas=metric_deltas,
    )


def _require_project_run(
    runs: RunRepository,
    experiments: ExperimentRepository,
    project_id: str,
    run_id: str,
) -> RunRecord:
    """Return a run after checking that it belongs to the current project."""

    run = runs.get_by_id(run_id)
    if run is None:
        raise PmemNotFoundError("Run was not found.")
    experiment = _require_experiment(experiments, run.experiment_id)
    if experiment.project_id != project_id:
        raise PmemValidationError("Run does not belong to this project.")
    return run


def _require_experiment(
    experiments: ExperimentRepository,
    experiment_id: str,
) -> ExperimentRecord:
    """Return an experiment or raise a safe not-found error."""

    experiment = experiments.get_by_id(experiment_id)
    if experiment is None:
        raise PmemNotFoundError("Experiment was not found.")
    return experiment


def _metadata_object(raw_json: str) -> dict[str, Any]:
    """Decode a repository JSON object defensively."""

    data = json.loads(raw_json)
    if not isinstance(data, dict):
        raise PmemValidationError("Stored metadata must be a JSON object.")
    return data


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Return finite numeric metrics suitable for baseline comparison."""

    numeric_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            numeric_metrics[key] = float(value)
    return numeric_metrics


def _utc_now_iso() -> str:
    """Return compact UTC ISO timestamp for baseline metadata."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
