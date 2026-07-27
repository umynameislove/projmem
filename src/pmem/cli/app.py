"""Typer application for the `pmem` CLI.

The CLI layer parses arguments, calls service use cases, and renders safe
messages. Database writes, filesystem policy, and domain rules stay below this
layer so commands remain easy to test and extend.
"""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pmem import __version__
from pmem.cli.status_output import print_status_text, render_status_json
from pmem.domain.conflicts import ConflictCheckReport
from pmem.domain.import_bundle import ImportDryRunReport
from pmem.errors import PmemError, PmemValidationError
from pmem.repositories.failures import FailureRecord
from pmem.services.baseline import compare_run_to_baseline, set_baseline_run
from pmem.services.conflict_detection import (
    check_bundle_conflicts,
    conflict_check_report_json,
)
from pmem.services.conflict_resolution import (
    conflict_resolution_result_json,
    record_conflict_resolution,
)
from pmem.services.decision_logging import log_decision
from pmem.services.export_bundle import (
    export_bundle,
    export_bundle_result_json,
)
from pmem.services.failure_clustering import (
    DEFAULT_SIMILARITY_THRESHOLD,
    failure_cluster_payload,
)
from pmem.services.failure_embeddings import (
    DEFAULT_EMBEDDING_DIMENSION,
    failure_embedding_payload,
)
from pmem.services.failure_exports import (
    export_failure_records,
    failure_export_payload,
)
from pmem.services.failure_logging import log_failure
from pmem.services.failure_patterns import (
    failure_analysis_summary_payload,
    failure_pattern_report_payload,
)
from pmem.services.graph_operations import (
    build_graph_artifact,
    export_graph_artifact,
    graph_lineage_payload,
    graph_query_payload,
    graph_status_payload,
)
from pmem.services.import_apply import (
    apply_import_bundle,
    import_apply_result_json,
)
from pmem.services.import_dry_run import (
    dry_run_import_bundle,
    import_dry_run_report_json,
)
from pmem.services.mcp_operations import run_mcp_stdio
from pmem.services.note_logging import add_note
from pmem.services.pattern_operations import (
    anomalies_cli_payload,
    config_failure_cli_payload,
    dataset_failure_cli_payload,
    pattern_list_payload,
    recurring_failures_cli_payload,
    temporal_cli_payload,
)
from pmem.services.project_export import export_project
from pmem.services.project_init import init_project
from pmem.services.recommendation_operations import (
    export_recommendations,
    recommendation_detail_payload,
    recommendation_list_payload,
)
from pmem.services.run_capture import run_command
from pmem.services.shared_paths import (
    list_shared_path_statuses,
    register_shared_path,
    shared_path_registration_json,
    shared_path_statuses_json,
)
from pmem.services.status_service import build_status_payload, collect_status_state
from pmem.services.tracking import track_path
from pmem.summary import ProjectSummary, get_project_summary, summary_json_payload

console = Console()

app = typer.Typer(
    add_completion=False,
    help="Local-first long-horizon project memory for AI research.",
    no_args_is_help=True,
)

share_app = typer.Typer(
    add_completion=False,
    help="Register and inspect explicit local shared memory paths.",
    no_args_is_help=True,
)
app.add_typer(share_app, name="share")

failures_app = typer.Typer(
    add_completion=False,
    help="List and export confirmed failure records with privacy-safe defaults.",
    no_args_is_help=True,
)
app.add_typer(failures_app, name="failures")

graph_app = typer.Typer(
    add_completion=False,
    help="Build and inspect the private local evidence graph.",
    no_args_is_help=True,
)
app.add_typer(graph_app, name="graph")

patterns_app = typer.Typer(
    add_completion=False,
    help="Run local pattern screening reports with privacy-safe defaults.",
    no_args_is_help=True,
)
app.add_typer(patterns_app, name="patterns")

recommend_app = typer.Typer(
    add_completion=False,
    help="List and export evidence-backed recommendation candidates.",
    no_args_is_help=True,
)
app.add_typer(recommend_app, name="recommend")


def _version_callback(show_version: bool) -> None:
    """Print the installed CLI version and exit.

    Keeping version handling in one callback makes the smoke test tiny while
    leaving room for richer diagnostics in a future `pmem doctor` command.
    """

    if show_version:
        console.print(f"pmem {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed pmem version and exit.",
        ),
    ] = False,
) -> None:
    """Run the pmem command-line interface.

    The callback is intentionally light. It provides help/version behavior while
    keeping command implementation in focused subcommands.
    """


@app.command("init")
def init_command(
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Optional local project name. Defaults to the current directory name.",
        ),
    ] = None,
    goal: Annotated[
        str | None,
        typer.Option(
            "--goal",
            help="Optional project goal stored in the local project record.",
        ),
    ] = None,
    objective: Annotated[
        str | None,
        typer.Option(
            "--objective",
            help="Optional current objective stored in the local project record.",
        ),
    ] = None,
    metric: Annotated[
        str | None,
        typer.Option(
            "--metric",
            help="Optional primary metric name.",
        ),
    ] = None,
    metric_direction: Annotated[
        str | None,
        typer.Option(
            "--metric-direction",
            help="Metric direction when a metric or target is provided: max or min.",
        ),
    ] = None,
    target: Annotated[
        float | None,
        typer.Option(
            "--target",
            help="Optional numeric target value. Requires --metric and --metric-direction.",
        ),
    ] = None,
) -> None:
    """Initialize project-local `.pmem/` state."""

    try:
        result = init_project(
            project_root=Path.cwd(),
            project_name=name,
            goal=goal,
            current_objective=objective,
            primary_metric=metric,
            metric_direction=metric_direction,
            target_value=target,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if result.already_initialized:
        console.print("projmem is already initialized.")
    else:
        console.print("Initialized projmem at .pmem/")
    console.print("Database ready.")


@app.command("track")
def track_command(
    path: Annotated[
        str,
        typer.Argument(help="Project-relative file path to track."),
    ],
    update: Annotated[
        bool,
        typer.Option(
            "--update",
            help="Refresh the stored SHA-256 if the file is already tracked.",
        ),
    ] = False,
) -> None:
    """Track one project file by SHA-256."""

    try:
        result = track_path(project_root=Path.cwd(), user_path=path, update=update)
    except PmemError as exc:
        _exit_with_error(exc)

    if result.updated:
        console.print(f"Updated {result.path}")
    elif result.already_tracked:
        console.print(f"{result.path} is already tracked.")
    else:
        console.print(f"Tracked {result.path}")
    console.print(f"sha256: {result.sha256}")


@app.command(
    "run",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def run_cli_command(
    command: Annotated[
        list[str] | None,
        typer.Argument(help="Command argv to execute after `--`."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Optional human label for this run.",
        ),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help="Optional timeout in seconds.",
        ),
    ] = None,
    seed: Annotated[
        str | None,
        typer.Option(
            "--seed",
            help="Optional reproducibility seed recorded with the run.",
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            "--config",
            help="Optional project-relative JSON config file to hash and store with redaction.",
        ),
    ] = None,
    metrics: Annotated[
        str | None,
        typer.Option(
            "--metrics",
            help="Optional project-relative JSON metrics file to load after the command.",
        ),
    ] = None,
    artifact: Annotated[
        list[str] | None,
        typer.Option(
            "--artifact",
            help="Optional project-relative artifact file to hash after the command.",
        ),
    ] = None,
    dataset_id: Annotated[
        str | None,
        typer.Option(
            "--dataset-id",
            help="Optional dataset id metadata for dataset-failure screening.",
        ),
    ] = None,
    dataset_version: Annotated[
        str | None,
        typer.Option(
            "--dataset-version",
            help="Optional dataset version metadata; requires --dataset-id.",
        ),
    ] = None,
) -> None:
    """Run a command and capture local evidence."""

    try:
        result = run_command(
            project_root=Path.cwd(),
            command_args=command or [],
            name=name,
            timeout_seconds=timeout,
            seed=seed,
            config_path=config,
            metrics_path=metrics,
            artifact_paths=tuple(artifact or ()),
            dataset_id=dataset_id,
            dataset_version=dataset_version,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    console.print(f"Run {result.record.run_id} {result.record.status}")
    if result.record.exit_code is not None:
        console.print(f"exit_code: {result.record.exit_code}")
    console.print(f"stdout: {result.stdout_path}")
    console.print(f"stderr: {result.stderr_path}")
    if result.artifact_count:
        console.print(f"artifacts: {result.artifact_count}")


@app.command("log-failure")
def log_failure_command(
    run_id: Annotated[
        str,
        typer.Argument(help="Run id to attach the confirmed failure to."),
    ],
    error_type: Annotated[
        str,
        typer.Argument(help="Short error type or failure class."),
    ],
    description: Annotated[
        str,
        typer.Argument(help="Human-readable failure description."),
    ],
    severity: Annotated[
        str,
        typer.Option(
            "--severity",
            help="Failure severity: critical, high, medium, or low.",
        ),
    ] = "medium",
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Failure source: user_confirmed, auto_technical, or promoted_candidate.",
        ),
    ] = "user_confirmed",
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            help="Failure tag. Can be provided multiple times.",
        ),
    ] = None,
    root_cause: Annotated[
        str | None,
        typer.Option(
            "--root-cause",
            help="Optional concise root cause. Do not include secrets.",
        ),
    ] = None,
    lesson: Annotated[
        str | None,
        typer.Option(
            "--lesson",
            help="Optional lesson learned. Do not include secrets.",
        ),
    ] = None,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Output format: text or json.",
        ),
    ] = "text",
) -> None:
    """Record a confirmed failure for an existing run."""

    try:
        output_format = _validate_output_format(output)
        record = log_failure(
            project_root=Path.cwd(),
            run_id=run_id,
            error_type=error_type,
            description=description,
            root_cause=root_cause,
            lesson=lesson,
            severity=severity,
            tags=tuple(tag or ()),
            source=source,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if output_format == "json":
        console.print_json(data=_failure_record_json(record))
        return

    console.print(f"Logged failure {record.id}")
    console.print(f"run_id: {record.run_id}")
    console.print(f"severity: {record.severity}")
    if record.tags_json != "[]":
        console.print(f"tags: {record.tags_json}")


@app.command("log-decision")
def log_decision_command(
    description: Annotated[
        str,
        typer.Argument(help="Decision summary."),
    ],
    rationale: Annotated[
        str | None,
        typer.Option(
            "--rationale",
            help="Optional rationale for the decision.",
        ),
    ] = None,
    experiment_id: Annotated[
        str | None,
        typer.Option(
            "--experiment-id",
            help="Optional experiment id linked to this decision.",
        ),
    ] = None,
    related_experiment: Annotated[
        list[str] | None,
        typer.Option(
            "--related-experiment",
            help="Related experiment id. Can be provided multiple times.",
        ),
    ] = None,
    author: Annotated[
        str | None,
        typer.Option(
            "--author",
            help="Optional author label.",
        ),
    ] = None,
) -> None:
    """Record a durable project decision."""

    try:
        record = log_decision(
            project_root=Path.cwd(),
            description=description,
            rationale=rationale,
            experiment_id=experiment_id,
            related_experiments=tuple(related_experiment or ()),
            author=author,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    console.print(f"Logged decision {record.id}")
    console.print(f"project_id: {record.project_id}")
    if record.experiment_id:
        console.print(f"experiment_id: {record.experiment_id}")


@app.command("note")
def note_command(
    content: Annotated[
        str,
        typer.Argument(help="Note content."),
    ],
    experiment_id: Annotated[
        str | None,
        typer.Option(
            "--experiment-id",
            help="Optional experiment id linked to this note.",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Optional run id linked to this note.",
        ),
    ] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            help="Note tag. Can be provided multiple times.",
        ),
    ] = None,
    resolved: Annotated[
        bool,
        typer.Option(
            "--resolved",
            help="Mark the note as already resolved.",
        ),
    ] = False,
) -> None:
    """Record a lightweight project note."""

    try:
        record = add_note(
            project_root=Path.cwd(),
            content=content,
            experiment_id=experiment_id,
            run_id=run_id,
            tags=tuple(tag or ()),
            resolved=resolved,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    console.print(f"Logged note {record.id}")
    console.print(f"project_id: {record.project_id}")
    if record.run_id:
        console.print(f"run_id: {record.run_id}")


@app.command("baseline")
def baseline_command(
    run_id: Annotated[
        str,
        typer.Argument(help="Run id to mark as baseline or compare to baseline."),
    ],
    compare: Annotated[
        bool,
        typer.Option(
            "--compare",
            help="Compare the run to the experiment baseline instead of marking it.",
        ),
    ] = False,
) -> None:
    """Mark or compare an experiment baseline run."""

    try:
        if compare:
            comparison = compare_run_to_baseline(project_root=Path.cwd(), run_id=run_id)
            console.print(f"Compared {comparison.run_id} to baseline.")
            console.print(f"baseline_run_id: {comparison.baseline_run_id}")
            console.print(f"experiment_id: {comparison.experiment_id}")
            console.print(f"metrics: {len(comparison.metric_deltas)}")
            for metric, delta in comparison.metric_deltas.items():
                console.print(f"{metric}: {delta:+.6f}")
            return
        else:
            baseline = set_baseline_run(project_root=Path.cwd(), run_id=run_id)
    except PmemError as exc:
        _exit_with_error(exc)

    console.print(f"Marked {baseline.run_id} as baseline.")
    console.print(f"experiment_id: {baseline.experiment_id}")
    console.print(f"metrics: {baseline.metric_count}")


@app.command("summary")
def summary_command(
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Output format: text or json.",
        ),
    ] = "text",
) -> None:
    """Print a project summary with timeline and status information."""

    try:
        output_format = _validate_output_format(output)
        summary = get_project_summary(project_root=Path.cwd())
    except PmemError as exc:
        _exit_with_error(exc)

    if output_format == "json":
        console.print_json(data=summary_json_payload(summary))
        return

    _print_summary(summary)


STATUS_JSON_OPTION_HELP = "Emit the machine-readable status-v1 payload as JSON instead of text."
STATUS_COMMAND_HELP = (
    "Print concise read-only project status and exactly one next action. "
    "With --json, stdout is a single status-v1 document only when the exit "
    "code is 0, so check the exit code before parsing."
)


@app.command("status", help=STATUS_COMMAND_HELP)
def status_command(
    json_output: Annotated[
        bool,
        typer.Option("--json", help=STATUS_JSON_OPTION_HELP),
    ] = False,
) -> None:
    """Print concise read-only project status and exactly one next action.

    The user-facing help text lives in ``STATUS_COMMAND_HELP`` so that this
    docstring can record the rationale without leaking into ``--help`` output.

    Output contract for ``--json``: on success stdout is exactly one
    ``status-v1`` document followed by one newline. On failure the command
    follows the repository-wide CLI convention -- a human-readable
    ``Error: ...`` line and exit code 1 -- and emits no JSON at all, rather
    than a fake JSON envelope that would masquerade as a valid payload.

    Because that convention currently writes diagnostics to stdout (see
    ``_exit_with_error``), stdout is only guaranteed to be parseable JSON when
    the exit code is 0. Routing CLI diagnostics to stderr is a
    repository-wide change and is deliberately not done here, where it would
    make ``status`` inconsistent with the other 30+ commands.
    """

    try:
        state = collect_status_state(
            project_root=Path.cwd(),
            evaluate_recommendations=False,
        )
        payload = build_status_payload(state)
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        typer.echo(render_status_json(payload))
        return

    print_status_text(payload, console=console)


@app.command("mcp")
def mcp_command() -> None:
    """Start the local stdio MCP JSON-RPC server."""

    run_mcp_stdio(Path.cwd(), stdin=sys.stdin, stdout=sys.stdout)


@app.command("serve")
def serve_command(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="API bind host. Defaults to loopback; non-local binds require confirmation.",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            help="API bind port.",
        ),
    ] = 8765,
    confirm_non_local_bind: Annotated[
        bool,
        typer.Option(
            "--confirm-non-local-bind",
            help="Explicitly allow an API bind outside loopback. This may expose project metadata.",
        ),
    ] = False,
) -> None:
    """Start the localhost-first FastAPI project-state server."""

    from pmem.server import run_api_server

    try:
        run_api_server(
            Path.cwd(),
            host=host,
            port=port,
            confirm_non_local_bind=confirm_non_local_bind,
        )
    except PmemError as exc:
        _exit_with_error(exc)


@app.command("export")
def export_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write the local-memory project export as JSON.",
        ),
    ] = False,
) -> None:
    """Export project-local memory as JSON without copying artifact contents."""

    try:
        if not json_output:
            raise PmemValidationError("Export currently supports --json only.")
        payload = export_project(project_root=Path.cwd())
    except PmemError as exc:
        _exit_with_error(exc)

    console.print_json(data=payload)


@app.command("export-bundle")
def export_bundle_command(
    out: Annotated[
        str,
        typer.Option(
            "--out",
            help="Output path for the portability and failure-analysis export bundle JSON.",
        ),
    ],
    scope: Annotated[
        str,
        typer.Option(
            "--scope",
            help="Export scope. Portability bundles currently support project only.",
        ),
    ] = "project",
    include_artifacts: Annotated[
        bool,
        typer.Option(
            "--include-artifacts",
            help="Include explicit artifact bytes as base64 payloads.",
        ),
    ] = False,
    redact_fields: Annotated[
        str | None,
        typer.Option(
            "--redact-fields",
            help="Comma-separated privacy-sensitive field paths to replace with [REDACTED].",
        ),
    ] = None,
    freeze_timestamp: Annotated[
        str | None,
        typer.Option(
            "--freeze-timestamp",
            help="UTC ISO timestamp for deterministic bundle generation.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write the export-bundle result as JSON.",
        ),
    ] = False,
) -> None:
    """Write a deterministic portability and failure-analysis export bundle."""

    try:
        result = export_bundle(
            project_root=Path.cwd(),
            output_path=out,
            scope=scope,
            include_artifacts=include_artifacts,
            redact_fields=(redact_fields or "",),
            freeze_timestamp=freeze_timestamp,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=export_bundle_result_json(result))
        return

    console.print(f"Exported bundle: {result.display_path}")
    console.print(f"manifest_hash: {result.manifest_hash}")
    console.print(f"payload_hash: {result.payload_hash}")
    console.print(f"artifacts: {result.artifact_count}")


@app.command("import")
def import_command(
    bundle: Annotated[
        str,
        typer.Argument(help="Project-relative export bundle JSON path."),
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate the bundle and preview conflicts without writing to the database.",
        ),
    ] = False,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Apply a validated bundle into pending/quarantine state.",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm import apply after reviewing dry-run output.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write the import report as JSON.",
        ),
    ] = False,
) -> None:
    """Validate or quarantine-apply a portability and failure-analysis bundle."""

    try:
        if dry_run and apply:
            raise PmemValidationError("Choose either --dry-run or --apply, not both.")
        if not dry_run and not apply:
            raise PmemValidationError("Import requires --dry-run or --apply.")
        if apply:
            result = apply_import_bundle(
                project_root=Path.cwd(),
                bundle_path=bundle,
                confirm=confirm,
            )
            if json_output:
                console.print_json(data=import_apply_result_json(result))
            else:
                console.print("Import apply: PENDING")
                console.print(f"import_job_id: {result.job.id}")
                console.print("Database mutation: import_jobs/audit_events only")
                console.print(f"integrity_check: {result.integrity_check}")
            return
        report = dry_run_import_bundle(project_root=Path.cwd(), bundle_path=bundle)
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=import_dry_run_report_json(report))
    else:
        _print_import_dry_run_report(report)

    if not report.ok:
        raise typer.Exit(code=1)


@app.command("conflict-check")
def conflict_check_command(
    bundle: Annotated[
        str,
        typer.Argument(help="Project-relative export bundle JSON path."),
    ],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write the conflict report as JSON.",
        ),
    ] = False,
) -> None:
    """Detect portability and failure-analysis import conflicts without mutating local data."""

    try:
        report = check_bundle_conflicts(project_root=Path.cwd(), bundle_path=bundle)
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=conflict_check_report_json(report))
    else:
        _print_conflict_check_report(report)

    if not report.validation_ok:
        raise typer.Exit(code=1)


@app.command("resolve")
def resolve_command(
    conflict_id: Annotated[
        str,
        typer.Argument(help="Conflict id from `pmem conflict-check`."),
    ],
    action: Annotated[
        str,
        typer.Option(
            "--action",
            help=(
                "Resolution action: skip, keep-local, manual-required, duplicate, "
                "take-imported, overwrite."
            ),
        ),
    ],
    before_hash: Annotated[
        str | None,
        typer.Option(
            "--before-hash",
            help="Optional sha256 hash of local evidence before resolution.",
        ),
    ] = None,
    after_hash: Annotated[
        str | None,
        typer.Option(
            "--after-hash",
            help="Optional sha256 hash of chosen evidence after resolution.",
        ),
    ] = None,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm destructive-looking actions. Canonical data is still not overwritten.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write the resolution audit result as JSON.",
        ),
    ] = False,
) -> None:
    """Record a non-destructive conflict resolution audit event."""

    try:
        result = record_conflict_resolution(
            project_root=Path.cwd(),
            conflict_id=conflict_id,
            action=action,
            before_hash=before_hash,
            after_hash=after_hash,
            confirm=confirm,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=conflict_resolution_result_json(result))
        return

    console.print("Conflict resolution recorded.")
    console.print(f"conflict_id: {result.conflict_id}")
    console.print(f"action: {result.action}")
    console.print("Database mutation: audit_events only")
    console.print("Canonical data mutation: none")


@failures_app.command("list")
def failures_list_command(
    include_text: Annotated[
        bool,
        typer.Option(
            "--include-text",
            help="Include raw failure description/root-cause/lesson text.",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm that raw failure text may be printed.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write failure records as JSON.",
        ),
    ] = False,
) -> None:
    """List confirmed failures without exposing raw text by default."""

    try:
        _require_failure_text_confirmation(include_text=include_text, confirm=confirm)
        payload = failure_export_payload(Path.cwd(), include_text=include_text)
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    _print_failure_records(payload)


@failures_app.command("export")
def failures_export_command(
    out: Annotated[
        str,
        typer.Option(
            "--out",
            help="Project-relative JSON output path.",
        ),
    ],
    include_text: Annotated[
        bool,
        typer.Option(
            "--include-text",
            help="Include raw failure description/root-cause/lesson text.",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm that raw failure text may be written.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write export result metadata as JSON.",
        ),
    ] = False,
) -> None:
    """Export confirmed failures to reviewable JSON."""

    try:
        _require_failure_text_confirmation(include_text=include_text, confirm=confirm)
        result = export_failure_records(
            Path.cwd(),
            output_path=out,
            include_text=include_text,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    result_payload = {
        "ok": True,
        "path": result.display_path,
        "schema_version": result.payload["schema_version"],
        "privacy_mode": result.payload["privacy_mode"],
        "include_text": result.payload["include_text"],
        "record_count": result.payload["record_count"],
    }
    if json_output:
        console.print_json(data=result_payload)
        return

    console.print(f"Exported failures: {result.display_path}")
    console.print(f"records: {result.payload['record_count']}")
    console.print(f"privacy_mode: {result.payload['privacy_mode']}")


@failures_app.command("embed")
def failures_embed_command(
    dimension: Annotated[
        int,
        typer.Option(
            "--dimension",
            help="Hashing-vector dimension for local failure embeddings.",
        ),
    ] = DEFAULT_EMBEDDING_DIMENSION,
    include_text: Annotated[
        bool,
        typer.Option(
            "--include-text",
            help="Derive vectors from raw failure description/root-cause/lesson text.",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm that raw failure text may be used to derive embeddings.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write embedding records as JSON.",
        ),
    ] = False,
) -> None:
    """Compute deterministic local failure embeddings."""

    try:
        _require_failure_text_confirmation(include_text=include_text, confirm=confirm)
        payload = failure_embedding_payload(
            Path.cwd(),
            include_text=include_text,
            dimension=dimension,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    console.print(f"Failure embeddings: {payload['record_count']}")
    console.print(f"method: {payload['method']}")
    console.print(f"dimension: {payload['dimension']}")
    console.print(f"privacy_mode: {payload['privacy_mode']}")


@failures_app.command("cluster")
def failures_cluster_command(
    dimension: Annotated[
        int,
        typer.Option(
            "--dimension",
            help="Hashing-vector dimension for local failure embeddings.",
        ),
    ] = DEFAULT_EMBEDDING_DIMENSION,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Cosine similarity threshold from 0.0 to 1.0.",
        ),
    ] = DEFAULT_SIMILARITY_THRESHOLD,
    include_text: Annotated[
        bool,
        typer.Option(
            "--include-text",
            help="Derive clusters from raw failure description/root-cause/lesson text.",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm that raw failure text may be used to derive clusters.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write cluster report as JSON.",
        ),
    ] = False,
) -> None:
    """Cluster local failure embeddings without network or vector DB."""

    try:
        _require_failure_text_confirmation(include_text=include_text, confirm=confirm)
        payload = failure_cluster_payload(
            Path.cwd(),
            include_text=include_text,
            dimension=dimension,
            similarity_threshold=threshold,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    console.print(f"Failure clusters: {payload['cluster_count']}")
    console.print(f"records: {payload['record_count']}")
    console.print(f"threshold: {payload['similarity_threshold']}")
    console.print(f"privacy_mode: {payload['privacy_mode']}")
    for cluster in payload["clusters"]:
        console.print(
            f"- {cluster['cluster_id']}: size={cluster['size']} "
            f"prototype={cluster['prototype_failure_id']}"
        )


@failures_app.command("patterns")
def failures_patterns_command(
    dimension: Annotated[
        int,
        typer.Option(
            "--dimension",
            help="Hashing-vector dimension for local failure pattern analysis.",
        ),
    ] = DEFAULT_EMBEDDING_DIMENSION,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Cosine similarity threshold from 0.0 to 1.0.",
        ),
    ] = DEFAULT_SIMILARITY_THRESHOLD,
    include_text: Annotated[
        bool,
        typer.Option(
            "--include-text",
            help="Derive text terms from raw failure description/root-cause/lesson text.",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm that raw failure text may be used to derive pattern signals.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write pattern report as JSON.",
        ),
    ] = False,
) -> None:
    """Generate human-reviewable failure pattern candidates."""

    try:
        _require_failure_text_confirmation(include_text=include_text, confirm=confirm)
        payload = failure_pattern_report_payload(
            Path.cwd(),
            include_text=include_text,
            dimension=dimension,
            similarity_threshold=threshold,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    _print_failure_pattern_report(payload)


@failures_app.command("summary")
def failures_summary_command(
    dimension: Annotated[
        int,
        typer.Option(
            "--dimension",
            help="Hashing-vector dimension for local failure analysis summary.",
        ),
    ] = DEFAULT_EMBEDDING_DIMENSION,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Cosine similarity threshold from 0.0 to 1.0.",
        ),
    ] = DEFAULT_SIMILARITY_THRESHOLD,
    include_text: Annotated[
        bool,
        typer.Option(
            "--include-text",
            help="Derive summary signals from raw failure description/root-cause/lesson text.",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm that raw failure text may be used to derive summary signals.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write failure analysis summary as JSON.",
        ),
    ] = False,
) -> None:
    """Summarize failure analysis status and top pattern candidates."""

    try:
        _require_failure_text_confirmation(include_text=include_text, confirm=confirm)
        payload = failure_analysis_summary_payload(
            Path.cwd(),
            include_text=include_text,
            dimension=dimension,
            similarity_threshold=threshold,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    _print_failure_analysis_summary(payload)


@graph_app.command("build")
def graph_build_command(
    incremental: Annotated[
        bool,
        typer.Option(
            "--incremental",
            help=(
                "Reuse graph artifact when the source fingerprint is unchanged; "
                "fallback to full rebuild when a safe delta cannot be proven."
            ),
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write graph build result as JSON.",
        ),
    ] = False,
) -> None:
    """Build and persist the private `.pmem/graph.json` artifact."""

    try:
        payload = build_graph_artifact(Path.cwd(), incremental=incremental)
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    console.print("Graph built.")
    console.print(f"mode: {payload['mode']}")
    console.print(f"path: {payload['graph_path']}")
    counts = payload["counts"]
    if isinstance(counts, dict):
        console.print(f"nodes: {counts.get('nodes', 0)}")
        console.print(f"edges: {counts.get('edges', 0)}")
    console.print(f"source_changed: {payload['source_changed']}")
    console.print(f"graph_changed: {payload['graph_changed']}")
    console.print(f"file_mode: {payload['file_mode']}")


@graph_app.command("status")
def graph_status_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write graph status as JSON.",
        ),
    ] = False,
) -> None:
    """Inspect the private graph artifact without exposing graph contents."""

    try:
        payload = graph_status_payload(Path.cwd())
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    console.print(f"Graph artifact: {'present' if payload['exists'] else 'missing'}")
    console.print(f"path: {payload['graph_path']}")
    if not payload["exists"]:
        console.print(str(payload["message"]))
        return
    counts = payload["counts"]
    if isinstance(counts, dict):
        console.print(f"nodes: {counts.get('nodes', 0)}")
        console.print(f"edges: {counts.get('edges', 0)}")
    if payload.get("build_mode") is not None:
        console.print(f"build_mode: {payload['build_mode']}")
    if payload.get("source_fingerprint_prefix") is not None:
        console.print(f"source_fingerprint: {payload['source_fingerprint_prefix']}...")
    console.print(f"file_mode: {payload['file_mode']}")


@graph_app.command("query")
def graph_query_command(
    node: Annotated[
        str,
        typer.Option(
            "--node",
            help="Graph node id to inspect.",
        ),
    ],
    edge_type: Annotated[
        str | None,
        typer.Option(
            "--edge-type",
            help="Optional graph schema edge type filter.",
        ),
    ] = None,
    direction: Annotated[
        str,
        typer.Option(
            "--direction",
            help="Neighbor direction: in, out, or both.",
        ),
    ] = "both",
    depth: Annotated[
        int | None,
        typer.Option(
            "--depth",
            help="Optional bounded subgraph depth.",
        ),
    ] = None,
    path_to: Annotated[
        str | None,
        typer.Option(
            "--path-to",
            help="Optional target node id for deterministic path search.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write graph query result as JSON.",
        ),
    ] = False,
) -> None:
    """Query neighbors, optional path, and optional bounded subgraph."""

    try:
        payload = graph_query_payload(
            Path.cwd(),
            node_id=node,
            edge_type=edge_type,
            direction=direction,
            depth=depth,
            path_to=path_to,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    console.print(f"Graph query: {payload['node_id']}")
    console.print(f"found: {payload['found']}")
    console.print(f"neighbors: {payload['neighbor_count']}")
    for neighbor in payload["neighbors"]:
        if isinstance(neighbor, dict):
            console.print(
                "- "
                f"{neighbor.get('direction')} {neighbor.get('edge_type')} "
                f"{neighbor.get('node_id')} ({neighbor.get('node_type')})"
            )
    path = payload.get("path")
    if isinstance(path, dict):
        console.print(f"path_found: {path.get('found')}")
        console.print(f"path_nodes: {len(path.get('node_ids', []))}")
    subgraph = payload.get("subgraph")
    if isinstance(subgraph, dict):
        counts = subgraph.get("counts")
        if isinstance(counts, dict):
            console.print(f"subgraph_nodes: {counts.get('nodes', 0)}")
            console.print(f"subgraph_edges: {counts.get('edges', 0)}")


@graph_app.command("lineage")
def graph_lineage_command(
    run_id: Annotated[
        str,
        typer.Option(
            "--run-id",
            help="Run id or graph run node id to trace.",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write graph lineage result as JSON.",
        ),
    ] = False,
) -> None:
    """Trace run lineage through direct graph evidence links."""

    try:
        payload = graph_lineage_payload(Path.cwd(), run_id=run_id)
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    lineage = payload["lineage"]
    if not isinstance(lineage, dict):
        console.print("Graph lineage: unavailable")
        return
    console.print(f"Graph lineage: {lineage.get('run_node_id')}")
    counts = lineage.get("counts")
    if isinstance(counts, dict):
        console.print(f"hops: {counts.get('hops', 0)}")
    hops = lineage.get("hops")
    if isinstance(hops, list):
        for hop in hops:
            if not isinstance(hop, dict):
                continue
            console.print(
                "- "
                f"{hop.get('direction')} {hop.get('edge_type') or 'self'} "
                f"{hop.get('node_id')} ({hop.get('entity_type')})"
            )
    warnings = lineage.get("warnings")
    if isinstance(warnings, list) and warnings:
        console.print("Warnings:")
        for warning in warnings:
            console.print(f"- {warning}")


@graph_app.command("export")
def graph_export_command(
    out: Annotated[
        str,
        typer.Option(
            "--out",
            help="Project-relative output path for explicit graph export JSON.",
        ),
    ],
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm exporting the private derived graph artifact.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write graph export result as JSON.",
        ),
    ] = False,
) -> None:
    """Export the private graph artifact to an explicit review file."""

    try:
        result = export_graph_artifact(Path.cwd(), output_path=out, confirm=confirm)
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=result.payload)
        return

    console.print(f"Exported graph: {result.display_path}")
    console.print(f"records: {result.payload['counts']}")
    console.print(f"file_mode: {result.payload['file_mode']}")


@patterns_app.command("list")
def patterns_list_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write pattern report summaries as JSON.",
        ),
    ] = False,
) -> None:
    """List local pattern report status and candidate counts."""

    try:
        payload = pattern_list_payload(Path.cwd())
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    _print_patterns_list(payload)


@patterns_app.command("config-failure")
def patterns_config_failure_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write config-failure correlation report as JSON.",
        ),
    ] = False,
) -> None:
    """Run config/failure correlation screening."""

    try:
        payload = config_failure_cli_payload(Path.cwd())
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    _print_pattern_cli_payload(payload)


@patterns_app.command("dataset-failure")
def patterns_dataset_failure_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write dataset-failure correlation report as JSON.",
        ),
    ] = False,
) -> None:
    """Run explicit-dataset/failure screening."""

    try:
        payload = dataset_failure_cli_payload(Path.cwd())
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    _print_pattern_cli_payload(payload)


@patterns_app.command("recurring-failures")
def patterns_recurring_failures_command(
    include_text: Annotated[
        bool,
        typer.Option(
            "--include-text",
            help="Use raw failure text to derive local recurring-failure vectors.",
        ),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm that raw failure text may be used for this analysis.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write recurring failure report as JSON.",
        ),
    ] = False,
) -> None:
    """Run recurring failure screening."""

    try:
        _require_failure_text_confirmation(include_text=include_text, confirm=confirm)
        payload = recurring_failures_cli_payload(Path.cwd(), include_text=include_text)
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    _print_pattern_cli_payload(payload)


@patterns_app.command("temporal")
def patterns_temporal_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write temporal pattern report as JSON.",
        ),
    ] = False,
) -> None:
    """Run temporal metric drift and decision-shift screening."""

    try:
        payload = temporal_cli_payload(Path.cwd())
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    _print_pattern_cli_payload(payload)


@patterns_app.command("anomalies")
def patterns_anomalies_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write anomaly report as JSON.",
        ),
    ] = False,
) -> None:
    """Run anomaly detection metric outlier and reproducibility screening."""

    try:
        payload = anomalies_cli_payload(Path.cwd())
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    _print_pattern_cli_payload(payload)


@recommend_app.command("list")
def recommend_list_command(
    max_recommendations: Annotated[
        int,
        typer.Option(
            "--max",
            help="Maximum number of recommendation candidates to show.",
        ),
    ] = 5,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write recommendation candidates as JSON.",
        ),
    ] = False,
) -> None:
    """List local evidence-backed recommendation candidates."""

    try:
        payload = recommendation_list_payload(
            Path.cwd(),
            max_recommendations=max_recommendations,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    _print_recommendation_list(payload)


@recommend_app.command("run")
def recommend_run_command(
    recommendation_id: Annotated[
        str,
        typer.Argument(help="Recommendation candidate id from `pmem recommend list`."),
    ],
    max_recommendations: Annotated[
        int,
        typer.Option(
            "--max",
            help="Maximum generated candidates to search.",
        ),
    ] = 5,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write one recommendation candidate as JSON.",
        ),
    ] = False,
) -> None:
    """Show one recommendation candidate by id."""

    try:
        payload = recommendation_detail_payload(
            Path.cwd(),
            recommendation_id=recommendation_id,
            max_recommendations=max_recommendations,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=payload)
        return

    recommendation = payload["recommendation"]
    if isinstance(recommendation, dict):
        _print_one_recommendation(recommendation)
    warnings = payload.get("warnings")
    if isinstance(warnings, list) and warnings:
        console.print("Warnings:")
        for warning in warnings:
            console.print(f"- {warning}")


@recommend_app.command("export")
def recommend_export_command(
    out: Annotated[
        str,
        typer.Option(
            "--out",
            help="Project-relative output path for recommendation JSON.",
        ),
    ],
    max_recommendations: Annotated[
        int,
        typer.Option(
            "--max",
            help="Maximum number of recommendation candidates to export.",
        ),
    ] = 5,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write recommendation export result as JSON.",
        ),
    ] = False,
) -> None:
    """Export recommendation candidates to a private local JSON file."""

    try:
        result = export_recommendations(
            Path.cwd(),
            output_path=out,
            max_recommendations=max_recommendations,
        )
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=result.payload)
        return

    console.print(f"Exported recommendations: {result.display_path}")
    console.print(f"recommendations: {result.payload['recommendation_count']}")
    console.print(f"file_mode: {result.payload['file_mode']}")


@share_app.command("init")
def share_init_command(
    path: Annotated[
        str,
        typer.Argument(help="Local directory to register as an explicit shared memory path."),
    ],
    alias: Annotated[
        str | None,
        typer.Option(
            "--alias",
            help="Unique local alias for the shared memory path.",
        ),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Shared path mode: read, write, or read_write.",
        ),
    ] = "read_write",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write the registration result as JSON.",
        ),
    ] = False,
) -> None:
    """Register a local shared memory path without starting sync."""

    try:
        result = register_shared_path(Path.cwd(), path, alias=alias, mode=mode)
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=shared_path_registration_json(result))
        return

    console.print("Shared path registered.")
    console.print(f"alias: {result.record.alias}")
    console.print(f"mode: {result.record.mode}")
    console.print(f"status: {result.status.status}")
    console.print(f"path: {result.status.path_display}")
    console.print("Sync: none")


@share_app.command("status")
def share_status_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write shared path status as JSON.",
        ),
    ] = False,
) -> None:
    """Inspect registered shared memory paths."""

    try:
        statuses = list_shared_path_statuses(Path.cwd())
    except PmemError as exc:
        _exit_with_error(exc)

    if json_output:
        console.print_json(data=shared_path_statuses_json(statuses))
        return

    console.print("Shared paths:")
    if not statuses:
        console.print("- none")
        return
    for status in statuses:
        console.print(f"- {status.alias}: {status.status} ({status.mode}) {status.path_display}")


def _validate_output_format(value: str) -> str:
    """Validate machine-readable output mode flags."""

    cleaned = value.strip().lower()
    if cleaned not in {"text", "json"}:
        raise PmemValidationError("Output format must be text or json.")
    return cleaned


def _require_failure_text_confirmation(*, include_text: bool, confirm: bool) -> None:
    """Require explicit confirmation before exposing raw failure free text."""

    if include_text and not confirm:
        raise PmemValidationError("Failure text export requires --confirm with --include-text.")


def _failure_record_json(record: FailureRecord) -> dict[str, object]:
    """Return the failure JSON JSON payload for one confirmed failure.

    description, root_cause, and lesson are intentionally omitted: they are
    free-text fields that may contain sensitive data (see SECURITY.md).
    """

    return {
        "id": record.id,
        "run_id": record.run_id,
        "error_type": record.error_type,
        "severity": record.severity,
        "tags": json.loads(record.tags_json),
        "source": record.source,
        "created_at": record.created_at,
    }


def _print_failure_records(payload: dict[str, object]) -> None:
    """Render failure export failure records without raw free text by default."""

    records = payload["records"]
    if not isinstance(records, list):
        records = []
    console.print(f"Failures: {payload['record_count']}")
    console.print(f"privacy_mode: {payload['privacy_mode']}")
    if not records:
        console.print("- none")
        return
    for record in records:
        if not isinstance(record, dict):
            continue
        tags = record.get("tags")
        tags_text = ",".join(str(tag) for tag in tags) if isinstance(tags, list) else ""
        console.print(
            "- "
            f"{record.get('id')}: {record.get('severity')} "
            f"{record.get('source')} {record.get('error_type')} "
            f"run={record.get('run_id')} tags={tags_text}"
        )
        if record.get("text_included") is True:
            console.print(f"  description: {record.get('description')}")
            if record.get("root_cause") is not None:
                console.print(f"  root_cause: {record.get('root_cause')}")
            if record.get("lesson") is not None:
                console.print(f"  lesson: {record.get('lesson')}")


def _print_summary(summary: ProjectSummary) -> None:
    """Render the human-readable project summary project summary."""

    console.print(f"Project: {summary.project_name}")
    console.print(f"Objective: {summary.objective or 'not set'}")
    console.print(f"Primary metric: {summary.primary_metric or 'not set'}")
    console.print(f"Metric direction: {summary.metric_direction or 'not set'}")
    console.print(
        "Target: "
        + (f"{summary.target_value:.6g}" if summary.target_value is not None else "not set")
    )
    console.print(f"Run count: {summary.run_count}")
    console.print(f"Best run: {summary.best_run_id or 'none'}")
    console.print(
        "Best metric value: "
        + (f"{summary.best_metric_value:.6g}" if summary.best_metric_value is not None else "none")
    )
    console.print(f"Target status: {summary.target_status}")
    console.print("Timeline:")
    for item in summary.timeline:
        console.print(f"- {item.name}: {item.status} - {item.detail}")
    console.print("Warnings:")
    if summary.warnings:
        for warning in summary.warnings:
            console.print(f"- {warning}")
    else:
        console.print("- none")


def _print_failure_pattern_report(payload: dict[str, object]) -> None:
    """Render failure pattern report pattern candidates without raw failure text."""

    console.print(f"Failure pattern candidates: {payload['pattern_count']}")
    console.print(f"records: {payload['record_count']}")
    console.print(f"clusters: {payload['cluster_count']}")
    console.print(f"privacy_mode: {payload['privacy_mode']}")
    console.print("heuristic: metadata-first, human review required")
    patterns = payload["patterns"]
    if not isinstance(patterns, list) or not patterns:
        console.print("- none")
        return
    for pattern in patterns:
        if not isinstance(pattern, dict):
            continue
        console.print(
            "- "
            f"{pattern.get('pattern_id')}: {pattern.get('heuristic_label')} "
            f"cluster={pattern.get('cluster_id')} evidence={pattern.get('evidence_count')} "
            f"score={pattern.get('heuristic_score')}"
        )
        console.print(f"  why: {pattern.get('explanation')}")
        console.print(f"  next: {pattern.get('review_recommendation')}")


def _print_failure_analysis_summary(payload: dict[str, object]) -> None:
    """Render failure analysis summary failure analysis summary for the CLI."""

    console.print("Failure analysis summary")
    console.print(f"status: {payload['status']}")
    console.print(f"records: {payload['record_count']}")
    console.print(f"clusters: {payload['cluster_count']}")
    console.print(f"pattern_candidates: {payload['pattern_count']}")
    console.print(f"privacy_mode: {payload['privacy_mode']}")
    console.print("Top pattern candidates:")
    patterns = payload["top_patterns"]
    if not isinstance(patterns, list) or not patterns:
        console.print("- none")
    else:
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            console.print(
                "- "
                f"{pattern.get('pattern_id')}: {pattern.get('heuristic_label')} "
                f"evidence={pattern.get('evidence_count')}"
            )
    console.print("Next actions:")
    actions = payload["next_actions"]
    if isinstance(actions, list):
        for action in actions:
            console.print(f"- {action}")


def _print_patterns_list(payload: dict[str, object]) -> None:
    """Render pattern CLI pattern report summaries without raw analysis attributes."""

    console.print("Pattern reports")
    console.print(f"patterns: {payload['pattern_count']}")
    console.print(f"candidates: {payload['candidate_count']}")
    console.print(f"privacy_mode: {payload['privacy_mode']}")
    patterns = payload["patterns"]
    if not isinstance(patterns, list) or not patterns:
        console.print("- none")
        return
    for item in patterns:
        if not isinstance(item, dict):
            continue
        console.print(
            "- "
            f"{item.get('pattern')}: {item.get('status')} "
            f"candidates={item.get('candidate_count')} warnings={item.get('warning_count')}"
        )
        warnings = item.get("warnings")
        if isinstance(warnings, list) and warnings:
            console.print(f"  first_warning: {warnings[0]}")


def _print_pattern_cli_payload(payload: dict[str, object]) -> None:
    """Render one pattern CLI pattern report wrapper in concise text form."""

    summary = payload["summary"]
    if not isinstance(summary, dict):
        console.print("Pattern report unavailable")
        return
    console.print(f"Pattern: {payload['pattern']}")
    console.print(f"status: {summary.get('status')}")
    console.print(f"candidates: {summary.get('candidate_count')}")
    console.print(f"privacy_mode: {payload['privacy_mode']}")
    warnings = summary.get("warnings")
    if isinstance(warnings, list) and warnings:
        console.print("Warnings:")
        for warning in warnings:
            console.print(f"- {warning}")


def _print_recommendation_list(payload: dict[str, object]) -> None:
    """Render recommendation CLI recommendation candidates without raw project text."""

    console.print("Recommendation candidates")
    console.print(f"recommendations: {payload['recommendation_count']}")
    console.print(f"privacy_mode: {payload['privacy_mode']}")
    console.print(str(payload["scope_message"]))
    recommendations = payload["recommendations"]
    if not isinstance(recommendations, list) or not recommendations:
        console.print("- none")
    else:
        for recommendation in recommendations:
            if isinstance(recommendation, dict):
                _print_one_recommendation(recommendation)
    warnings = payload["warnings"]
    if isinstance(warnings, list) and warnings:
        console.print("Warnings:")
        for warning in warnings:
            console.print(f"- {warning}")


def _print_one_recommendation(recommendation: dict[str, object]) -> None:
    """Render one recommendation candidate in concise audit wording."""

    supporting = recommendation.get("supporting_evidence")
    opposing = recommendation.get("opposing_evidence")
    failures = recommendation.get("related_failures")
    evidence_count = (
        (len(supporting) if isinstance(supporting, list) else 0)
        + (len(opposing) if isinstance(opposing, list) else 0)
        + (len(failures) if isinstance(failures, list) else 0)
    )
    console.print(
        "- "
        f"{recommendation.get('recommendation_id')}: "
        f"{recommendation.get('type')} "
        f"confidence={recommendation.get('confidence')} "
        f"evidence={evidence_count}"
    )
    console.print(f"  title: {_short_cli_text(recommendation.get('title'))}")
    console.print(f"  why: {_short_cli_text(recommendation.get('description'))}")
    console.print(f"  next: {_short_cli_text(recommendation.get('suggested_action'))}")


def _short_cli_text(value: object, *, max_chars: int = 180) -> str:
    """Return compact CLI text so users do not need to inspect full JSON."""

    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _print_import_dry_run_report(report: ImportDryRunReport) -> None:
    """Render the human-readable import dry-run import dry-run report."""

    console.print(f"Import dry-run: {'PASS' if report.ok else 'FAIL'}")
    console.print(f"Bundle: {report.bundle_path}")
    console.print(f"export_format_version: {report.export_format_version or 'unknown'}")
    console.print(f"schema_version: {report.schema_version or 'unknown'}")
    console.print("Database mutation: none")
    console.print("Entities:")
    for entity_type, count in sorted(report.entity_counts.items()):
        console.print(f"- {entity_type}: {count}")

    console.print("PRIVACY REVIEW:")
    if report.privacy_review:
        for item in report.privacy_review:
            console.print(f"- {item.field}: {item.count} - {item.message}")
    else:
        console.print("- no free-text or artifact metadata warnings detected")

    console.print("Conflicts:")
    if report.conflicts:
        for item in report.conflicts:
            console.print(
                f"- {item.conflict_type}: {item.entity_type} {item.entity_id} - {item.message}"
            )
    else:
        console.print("- none")

    if report.warnings:
        console.print("Warnings:")
        for warning in report.warnings:
            console.print(f"- {warning.code}: {warning.message}")

    if report.errors:
        console.print("Errors:")
        for error in report.errors:
            field = f"{error.field}: " if error.field else ""
            console.print(f"- {error.code}: {field}{error.message}")


def _print_conflict_check_report(report: ConflictCheckReport) -> None:
    """Render the human-readable conflict detection conflict-check report."""

    data = conflict_check_report_json(report)
    console.print(f"Conflict check: {'PASS' if data['validation_ok'] else 'FAIL'}")
    console.print(f"Bundle: {data['bundle_path']}")
    console.print("Database mutation: none")
    console.print(f"Conflicts: {data['conflict_count']}")
    conflicts = data["conflicts"]
    if conflicts:
        for item in conflicts:
            console.print(
                "- "
                f"{item['conflict_type']}: {item['entity_type']} {item['entity_id']} "
                f"({item['severity']})"
            )
    else:
        console.print("- none")
    validation_errors = data["validation_errors"]
    if validation_errors:
        console.print("Validation errors:")
        for error in validation_errors:
            field = f"{error['field']}: " if error["field"] else ""
            console.print(f"- {error['code']}: {field}{error['message']}")


def _exit_with_error(exc: PmemError) -> None:
    """Render expected errors without exposing tracebacks or raw internals."""

    console.print(f"Error: {exc}", style="red", markup=False)
    raise typer.Exit(code=1)
