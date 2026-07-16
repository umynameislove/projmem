"""project summary project summary service.

The summary layer reads existing SQLite evidence and produces a small,
deterministic view for CLI output. It does not write database rows, read
artifacts, or expose free-text failure/decision/note bodies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pmem.repositories.decisions import DecisionRepository
from pmem.repositories.experiments import ExperimentRecord, ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.notes import NoteRepository
from pmem.repositories.runs import RunRecord, RunRepository
from pmem.repositories.sqlite import (
    connect_database,
    connect_database_readonly,
    project_database_path,
)
from pmem.repositories.tracked_paths import TrackedPathRepository
from pmem.services.project_context import (
    ProjectContext,
    require_project_context,
    require_project_context_readonly,
)


@dataclass(frozen=True)
class SummaryTimelineItem:
    """One high-level project progression item for timeline/status output."""

    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class ProjectSummary:
    """project summary project summary result rendered by the CLI."""

    project_id: str
    project_name: str
    objective: str | None
    primary_metric: str | None
    metric_direction: str | None
    target_value: float | None
    run_count: int
    successful_run_count: int
    failed_run_count: int
    best_run_id: str | None
    best_metric_value: float | None
    target_status: str
    tracked_path_count: int
    failure_count: int
    decision_count: int
    note_count: int
    baseline_run_id: str | None
    timeline: tuple[SummaryTimelineItem, ...]
    warnings: tuple[str, ...]


def get_project_summary(project_root: str | Path) -> ProjectSummary:
    """Return a deterministic summary for the initialized project."""

    context = require_project_context(project_root)
    connection = connect_database(project_database_path(context.root))
    try:
        return _summary_from_connection(context, connection)
    finally:
        connection.close()


def get_project_summary_readonly(project_root: str | Path) -> ProjectSummary:
    """Read-only variant of :func:`get_project_summary`.

    Shares the exact summary logic but resolves the project through a read-only
    context/connection: it never migrates, backs up, ``chmod``s, or creates the
    database, and it rejects a symlinked database/config.
    """

    context = require_project_context_readonly(project_root)
    connection = connect_database_readonly(project_database_path(context.root))
    try:
        return _summary_from_connection(context, connection)
    finally:
        connection.close()


def _summary_from_connection(context: ProjectContext, connection: Any) -> ProjectSummary:
    """Single source of summary construction shared by both entry points."""

    project_id = context.project.id
    experiments = ExperimentRepository(connection).list_for_project(project_id)
    runs = RunRepository(connection).list_for_project(project_id)
    tracked_paths = TrackedPathRepository(connection).list_for_project(project_id)
    failures = FailureRepository(connection).list_for_project(project_id)
    decisions = DecisionRepository(connection).list_for_project(project_id)
    notes = NoteRepository(connection).list_for_project(project_id)

    target_value = _target_value(context.project.target_json)
    best_run, best_metric_value = _best_run(
        runs,
        metric=context.project.primary_metric,
        direction=context.project.metric_direction,
    )
    successful_run_count = sum(1 for run in runs if run.status == "success")
    failed_run_count = sum(1 for run in runs if run.status == "failed")
    baseline_run_id = _baseline_run_id(experiments)
    target_status = _target_status(
        run_count=len(runs),
        successful_run_count=successful_run_count,
        target_value=target_value,
        metric=context.project.primary_metric,
        direction=context.project.metric_direction,
        best_metric_value=best_metric_value,
    )
    warnings = _warnings(
        target_status=target_status,
        tracked_path_count=len(tracked_paths),
        run_count=len(runs),
        successful_run_count=successful_run_count,
        baseline_run_id=baseline_run_id,
        metric=context.project.primary_metric,
    )
    timeline = _timeline(
        tracked_path_count=len(tracked_paths),
        run_count=len(runs),
        successful_run_count=successful_run_count,
        baseline_run_id=baseline_run_id,
        failure_count=len(failures),
        decision_count=len(decisions),
        note_count=len(notes),
        target_status=target_status,
    )

    return ProjectSummary(
        project_id=context.project.id,
        project_name=context.project.name,
        objective=context.project.current_objective,
        primary_metric=context.project.primary_metric,
        metric_direction=context.project.metric_direction,
        target_value=target_value,
        run_count=len(runs),
        successful_run_count=successful_run_count,
        failed_run_count=failed_run_count,
        best_run_id=best_run.run_id if best_run is not None else None,
        best_metric_value=best_metric_value,
        target_status=target_status,
        tracked_path_count=len(tracked_paths),
        failure_count=len(failures),
        decision_count=len(decisions),
        note_count=len(notes),
        baseline_run_id=baseline_run_id,
        timeline=timeline,
        warnings=warnings,
    )


def summary_json_payload(summary: ProjectSummary) -> dict[str, object]:
    """Return the stable summary machine-readable summary payload."""

    return {
        "project_id": summary.project_id,
        "project_name": summary.project_name,
        "objective": summary.objective,
        "primary_metric": summary.primary_metric,
        "metric_direction": summary.metric_direction,
        "target_value": summary.target_value,
        "run_count": summary.run_count,
        "successful_run_count": summary.successful_run_count,
        "failed_run_count": summary.failed_run_count,
        "best_run_id": summary.best_run_id,
        "best_metric_value": summary.best_metric_value,
        "target_status": summary.target_status,
        "tracked_path_count": summary.tracked_path_count,
        "failure_count": summary.failure_count,
        "decision_count": summary.decision_count,
        "note_count": summary.note_count,
        "baseline_run_id": summary.baseline_run_id,
        "timeline": [
            {"name": item.name, "status": item.status, "detail": item.detail}
            for item in summary.timeline
        ],
        "warnings": list(summary.warnings),
    }


def _target_value(target_json: str) -> float | None:
    """Extract the numeric project target value from schema-v1 JSON."""

    target = json.loads(target_json)
    value = target.get("target_value") if isinstance(target, dict) else None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _best_run(
    runs: tuple[RunRecord, ...],
    *,
    metric: str | None,
    direction: str | None,
) -> tuple[RunRecord | None, float | None]:
    """Return the best successful run by the configured primary metric."""

    if metric is None or direction not in {"max", "min"}:
        return None, None

    candidates: list[tuple[float, str, str, RunRecord]] = []
    for run in runs:
        if run.status != "success":
            continue
        value = _run_metric(run, metric)
        if value is not None:
            candidates.append((value, run.timestamp, run.run_id, run))

    if not candidates:
        return None, None

    selected = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    if direction == "max":
        selected = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return selected[3], selected[0]


def _run_metric(run: RunRecord, metric: str) -> float | None:
    """Extract one finite numeric metric from a run metrics JSON object."""

    metrics = json.loads(run.metrics_json)
    value = metrics.get(metric) if isinstance(metrics, dict) else None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _baseline_run_id(experiments: tuple[ExperimentRecord, ...]) -> str | None:
    """Return the first configured baseline baseline run id, if any."""

    for experiment in experiments:
        metadata = json.loads(experiment.metadata_json)
        if isinstance(metadata, dict):
            baseline_run_id = metadata.get("baseline_run_id")
            if isinstance(baseline_run_id, str) and baseline_run_id.strip():
                return baseline_run_id
    return None


def _target_status(
    *,
    run_count: int,
    successful_run_count: int,
    target_value: float | None,
    metric: str | None,
    direction: str | None,
    best_metric_value: float | None,
) -> str:
    """Compute the small summary target-status vocabulary."""

    if run_count == 0:
        return "no_runs"
    if successful_run_count == 0:
        return "no_successful_runs"
    if target_value is None or metric is None or direction is None:
        return "not_configured"
    if best_metric_value is None:
        return "no_metric"
    if direction == "max":
        return "met" if best_metric_value >= target_value else "not_met"
    return "met" if best_metric_value <= target_value else "not_met"


def _warnings(
    *,
    target_status: str,
    tracked_path_count: int,
    run_count: int,
    successful_run_count: int,
    baseline_run_id: str | None,
    metric: str | None,
) -> tuple[str, ...]:
    """Return concise status warnings without exposing raw run/user text."""

    warnings: list[str] = []
    if tracked_path_count == 0:
        warnings.append("No tracked files yet.")
    if run_count == 0:
        warnings.append("No runs captured yet.")
    elif successful_run_count == 0:
        warnings.append("No successful runs captured yet.")
    if run_count > 0 and baseline_run_id is None:
        warnings.append("No baseline run set.")
    if target_status == "not_met":
        warnings.append("Target is not met.")
    if target_status == "no_metric" and metric is not None:
        warnings.append(f"No successful run has metric {metric}.")
    return tuple(warnings)


def _timeline(
    *,
    tracked_path_count: int,
    run_count: int,
    successful_run_count: int,
    baseline_run_id: str | None,
    failure_count: int,
    decision_count: int,
    note_count: int,
    target_status: str,
) -> tuple[SummaryTimelineItem, ...]:
    """Return timeline high-level progression without scanning artifacts."""

    memory_count = failure_count + decision_count + note_count
    return (
        SummaryTimelineItem("init", "done", "project initialized"),
        SummaryTimelineItem(
            "track",
            "done" if tracked_path_count else "missing",
            f"{tracked_path_count} tracked file(s)",
        ),
        SummaryTimelineItem(
            "run",
            "done" if run_count else "missing",
            f"{run_count} run(s), {successful_run_count} successful",
        ),
        SummaryTimelineItem(
            "baseline",
            "done" if baseline_run_id is not None else "missing",
            baseline_run_id or "no baseline run set",
        ),
        SummaryTimelineItem(
            "memory",
            "done" if memory_count else "missing",
            f"{failure_count} failure(s), {decision_count} decision(s), {note_count} note(s)",
        ),
        SummaryTimelineItem("target", target_status, f"target status: {target_status}"),
    )
