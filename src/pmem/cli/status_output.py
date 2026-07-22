"""Deterministic, privacy-conscious text rendering for ``pmem status``.

The renderer is deliberately separate from the command orchestration in
``pmem.cli.app``. It accepts an already validated ``status-v1`` payload and a
Rich console, performs no project reads, and never mutates project state.

All dynamic values are printed with Rich markup and automatic highlighting
disabled. Project metadata and identifiers therefore cannot be interpreted as
Rich tags or terminal links by this layer.
"""

from __future__ import annotations

from rich.console import Console

from pmem.status import RecommendationMode, StatusPayload


def print_status_text(payload: StatusPayload, *, console: Console) -> None:
    """Render one concise human-readable status report."""

    _print(console, "Project status", style="bold")
    _print(console, f"Project: {payload.project.project_name}")
    _print(console, f"Project ID: {payload.project.project_id}")
    _print(console, f"Objective: {payload.project.objective or 'not set'}")

    metric = payload.metric
    direction = metric.direction.value if metric.direction is not None else "not set"
    _print(console, f"Primary metric: {metric.primary_metric or 'not set'}")
    _print(console, f"Metric direction: {direction}")
    _print(console, f"Target: {_number_or(metric.target_value, 'not set')}")
    _print(console, f"Best value: {_number_or(metric.best_value, 'none')}")
    _print(console, f"Target status: {metric.target_status.value}")
    _print(console, f"Best run: {payload.best_run.run_id or 'none'}")
    _print(console, f"Baseline: {payload.baseline.run_id or 'none'}")

    counts = payload.counts
    _print(
        console,
        "Runs: "
        f"total={counts.run_count} "
        f"successful={counts.successful_run_count} "
        f"failed={counts.failed_run_count}",
    )
    _print(
        console,
        "Memory: "
        f"tracked_paths={counts.tracked_path_count} "
        f"failures={counts.failure_count} "
        f"decisions={counts.decision_count} "
        f"notes={counts.note_count}",
    )

    graph = payload.graph
    graph_parts = [f"Graph: {graph.state.value}"]
    if graph.node_count is not None:
        graph_parts.append(f"nodes={graph.node_count}")
        graph_parts.append(f"edges={graph.edge_count}")
    if graph.reason_code is not None:
        graph_parts.append(f"reason={graph.reason_code}")
    _print(console, " ".join(graph_parts))

    _print(console, _recommendation_line(payload))

    _print(console, "Warnings", style="bold")
    if not payload.warnings:
        _print(console, "- none")
    else:
        for warning in payload.warnings:
            _print(
                console,
                f"- [{warning.severity.value.upper()}] "
                f"{warning.source.value}/{warning.code}: {warning.message}",
            )
            if warning.remediation is not None:
                _print(console, f"  Remediation: {warning.remediation}")

    action = payload.next_action
    _print(console, "Next action", style="bold")
    _print(console, f"Action: {action.action_id}")
    _print(console, f"Reason: {action.reason}")
    _print(console, f"Command: {action.suggested_command}")
    if action.related_entity_id is not None:
        _print(console, f"Related entity: {action.related_entity_id}")

    _print(
        console,
        "Safety: "
        f"database_mutation={_bool_text(payload.database_mutation)} "
        f"network={_bool_text(payload.network)} "
        f"raw_text_in_output={_bool_text(payload.raw_text_in_output)}",
    )


def _recommendation_line(payload: StatusPayload) -> str:
    recommendations = payload.recommendations
    parts = [f"Recommendations: {recommendations.mode.value}"]
    if recommendations.mode is not RecommendationMode.NOT_EVALUATED:
        parts.append(f"candidates={recommendations.candidate_count}")
    if recommendations.mode is RecommendationMode.PERSISTED_LIFECYCLE:
        parts.append(f"active={recommendations.active_count}")
    return " ".join(parts)


def _number_or(value: float | None, fallback: str) -> str:
    return fallback if value is None else format(value, ".12g")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _print(console: Console, value: str, *, style: str | None = None) -> None:
    console.print(value, style=style, markup=False, highlight=False)
