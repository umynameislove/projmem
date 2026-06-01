"""`pmem note` service workflow."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from pmem.domain.note import Note
from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.notes import NoteRecord, NoteRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_context import require_project_context


def add_note(
    project_root: str | Path,
    *,
    content: str,
    experiment_id: str | None = None,
    run_id: str | None = None,
    tags: tuple[str, ...] = (),
    resolved: bool = False,
) -> NoteRecord:
    """Validate and persist one project note."""

    context = require_project_context(project_root)
    created_at = _utc_now_iso()
    try:
        note = Note(
            id=f"note_{uuid.uuid4().hex}",
            project_id=context.project.id,
            experiment_id=experiment_id,
            run_id=run_id,
            content=content,
            tags=list(tags),
            context={},
            resolved=resolved,
            created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00")),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise PmemValidationError(_note_validation_message(str(exc))) from exc

    connection = connect_database(project_database_path(context.root))
    try:
        experiments = ExperimentRepository(connection)
        resolved_experiment_id = _resolve_note_experiment_id(
            runs=RunRepository(connection),
            experiments=experiments,
            project_id=context.project.id,
            experiment_id=note.experiment_id,
            run_id=note.run_id,
        )
        return NoteRepository(connection).create(
            note_id=note.id,
            project_id=note.project_id,
            experiment_id=resolved_experiment_id,
            run_id=note.run_id,
            content=note.content,
            tags=note.tags,
            context=note.context,
            resolved=note.resolved,
            created_at=created_at,
        )
    finally:
        connection.close()


def _resolve_note_experiment_id(
    *,
    runs: RunRepository,
    experiments: ExperimentRepository,
    project_id: str,
    experiment_id: str | None,
    run_id: str | None,
) -> str | None:
    """Validate note links and infer experiment_id from run_id when possible."""

    linked_experiment_id = experiment_id
    if run_id is not None:
        run = runs.get_by_id(run_id)
        if run is None:
            raise PmemNotFoundError("Run was not found.")
        if linked_experiment_id is not None and linked_experiment_id != run.experiment_id:
            raise PmemValidationError("Run does not belong to the provided experiment.")
        linked_experiment_id = run.experiment_id

    if linked_experiment_id is not None:
        experiment = experiments.get_by_id(linked_experiment_id)
        if experiment is None:
            raise PmemNotFoundError("Experiment was not found.")
        if experiment.project_id != project_id:
            raise PmemValidationError("Experiment does not belong to this project.")
    return linked_experiment_id


def _note_validation_message(raw_message: str) -> str:
    """Return a concise public validation message."""

    if "tags" in raw_message:
        return "Note tags cannot be blank."
    if "cannot be blank" in raw_message:
        return "Note text fields cannot be blank."
    return "Note input is invalid."


def _utc_now_iso() -> str:
    """Return compact UTC ISO timestamp for memory records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
