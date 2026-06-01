"""`pmem log-decision` service workflow."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from pmem.domain.decision import Decision
from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.repositories.decisions import DecisionRecord, DecisionRepository
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.services.project_context import require_project_context


def log_decision(
    project_root: str | Path,
    *,
    description: str,
    rationale: str | None = None,
    experiment_id: str | None = None,
    related_experiments: tuple[str, ...] = (),
    author: str | None = None,
) -> DecisionRecord:
    """Validate and persist one durable project decision."""

    context = require_project_context(project_root)
    created_at = _utc_now_iso()
    try:
        decision = Decision(
            id=f"dec_{uuid.uuid4().hex}",
            project_id=context.project.id,
            experiment_id=experiment_id,
            description=description,
            rationale=rationale,
            related_experiments=list(related_experiments),
            created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00")),
            author=author,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise PmemValidationError(_decision_validation_message(str(exc))) from exc

    connection = connect_database(project_database_path(context.root))
    try:
        experiments = ExperimentRepository(connection)
        _assert_project_experiment(experiments, context.project.id, decision.experiment_id)
        for related_experiment in decision.related_experiments:
            _assert_project_experiment(experiments, context.project.id, related_experiment)
        return DecisionRepository(connection).create(
            decision_id=decision.id,
            project_id=decision.project_id,
            experiment_id=decision.experiment_id,
            description=decision.description,
            rationale=decision.rationale,
            related_experiments=decision.related_experiments,
            created_at=created_at,
            author=decision.author,
        )
    finally:
        connection.close()


def _assert_project_experiment(
    repository: ExperimentRepository,
    project_id: str,
    experiment_id: str | None,
) -> None:
    """Ensure an optional experiment link belongs to the current project."""

    if experiment_id is None:
        return
    experiment = repository.get_by_id(experiment_id)
    if experiment is None:
        raise PmemNotFoundError("Experiment was not found.")
    if experiment.project_id != project_id:
        raise PmemValidationError("Experiment does not belong to this project.")


def _decision_validation_message(raw_message: str) -> str:
    """Return a concise public validation message."""

    if "cannot be blank" in raw_message:
        return "Decision text fields cannot be blank."
    return "Decision input is invalid."


def _utc_now_iso() -> str:
    """Return compact UTC ISO timestamp for memory records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
