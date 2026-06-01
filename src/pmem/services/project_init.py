"""`pmem init` service workflow.

The service owns the init use case: create local state, run migrations, create
one project row, and preserve existing config/database state on repeated runs.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pmem.domain.common import MetricDirection
from pmem.errors import PmemConflictError, PmemValidationError
from pmem.migrations.runner import MigrationResult
from pmem.repositories.projects import ProjectRecord, ProjectRepository
from pmem.repositories.sqlite import PMEM_DIRNAME, connect_database, project_database_path
from pmem.services.config import (
    ProjectConfig,
    project_config_path,
    read_project_config,
    write_project_config_if_missing,
)
from pmem.services.database import ensure_database

ARTIFACTS_DIRNAME = "artifacts"
SNAPSHOTS_DIRNAME = "snapshots"
MAX_PROJECT_NAME_LENGTH = 120
MAX_PROJECT_TEXT_LENGTH = 512


@dataclass(frozen=True)
class InitProjectResult:
    """User-facing result of `pmem init`."""

    already_initialized: bool
    project_id: str
    project_name: str
    pmem_dir: Path
    db_path: Path
    config_path: Path
    artifacts_dir: Path
    snapshots_dir: Path
    migration_result: MigrationResult


@dataclass(frozen=True)
class InitProjectMetadata:
    """Optional project context accepted by `pmem init` flags."""

    goal: str | None = None
    current_objective: str | None = None
    primary_metric: str | None = None
    metric_direction: str | None = None
    target_value: float | None = None


def init_project(
    project_root: str | Path,
    project_name: str | None = None,
    *,
    goal: str | None = None,
    current_objective: str | None = None,
    primary_metric: str | None = None,
    metric_direction: str | None = None,
    target_value: float | None = None,
) -> InitProjectResult:
    """Initialize local `.pmem/` state without overwriting existing state."""

    root = Path(project_root)
    metadata = validate_init_metadata(
        goal=goal,
        current_objective=current_objective,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
        target_value=target_value,
    )
    pmem_dir = root / PMEM_DIRNAME
    artifacts_dir = pmem_dir / ARTIFACTS_DIRNAME
    snapshots_dir = pmem_dir / SNAPSHOTS_DIRNAME
    config_path = project_config_path(root)
    db_path = project_database_path(root)

    config_exists_before = config_path.exists()
    db_exists_before = db_path.exists()

    pmem_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    migration_result = ensure_database(root)
    project_record, config_created = _ensure_project_record(
        root,
        config_path,
        project_name,
        metadata,
    )

    return InitProjectResult(
        already_initialized=config_exists_before and db_exists_before and not config_created,
        project_id=project_record.id,
        project_name=project_record.name,
        pmem_dir=pmem_dir,
        db_path=db_path,
        config_path=config_path,
        artifacts_dir=artifacts_dir,
        snapshots_dir=snapshots_dir,
        migration_result=migration_result,
    )


def _ensure_project_record(
    project_root: Path,
    config_path: Path,
    project_name: str | None,
    metadata: InitProjectMetadata,
) -> tuple[ProjectRecord, bool]:
    """Synchronize config identity and the SQLite project row."""

    db_path = project_database_path(project_root)
    connection = connect_database(db_path)
    try:
        repository = ProjectRepository(connection)
        if config_path.exists():
            config = read_project_config(config_path)
            if (
                project_name is not None
                and validate_project_name(project_name) != config.project_name
            ):
                raise PmemConflictError("projmem is already initialized with a different name.")
            record = repository.get_by_id(config.project_id)
            if record is None:
                record = repository.create(
                    project_id=config.project_id,
                    name=config.project_name,
                    created_at=config.created_at,
                    updated_at=_utc_now_iso(),
                    goal=metadata.goal,
                    current_objective=metadata.current_objective,
                    primary_metric=metadata.primary_metric,
                    metric_direction=metadata.metric_direction,
                    target=_target_json(metadata),
                )
            _assert_metadata_compatible(record, metadata)
            return record, False

        record = _find_or_create_single_project(repository, project_root, project_name, metadata)
        config = ProjectConfig(
            version=1,
            project_id=record.id,
            project_name=record.name,
            created_at=record.created_at,
        )
        config_created = write_project_config_if_missing(config_path, config)
        return record, config_created
    finally:
        connection.close()


def _find_or_create_single_project(
    repository: ProjectRepository,
    project_root: Path,
    project_name: str | None,
    metadata: InitProjectMetadata,
) -> ProjectRecord:
    """Use an existing single project row or create the first one."""

    records = repository.list_projects()
    if len(records) > 1:
        raise PmemConflictError("Multiple project records exist. Resolve the database first.")
    if len(records) == 1:
        record = records[0]
        if project_name is not None and validate_project_name(project_name) != record.name:
            raise PmemConflictError("projmem is already initialized with a different name.")
        _assert_metadata_compatible(record, metadata)
        return record

    name = validate_project_name(project_name or project_root.name or "projmem-project")
    timestamp = _utc_now_iso()
    return repository.create(
        project_id=f"proj_{uuid.uuid4().hex}",
        name=name,
        created_at=timestamp,
        updated_at=timestamp,
        goal=metadata.goal,
        current_objective=metadata.current_objective,
        primary_metric=metadata.primary_metric,
        metric_direction=metadata.metric_direction,
        target=_target_json(metadata),
    )


def validate_project_name(name: str) -> str:
    """Validate the project name accepted by service/CLI."""

    cleaned = name.strip()
    if not cleaned:
        raise PmemValidationError("Project name cannot be blank.")
    if len(cleaned) > MAX_PROJECT_NAME_LENGTH:
        raise PmemValidationError("Project name is too long.")
    if any(ord(character) < 32 for character in cleaned):
        raise PmemValidationError("Project name contains unsupported control characters.")
    return cleaned


def validate_init_metadata(
    *,
    goal: str | None,
    current_objective: str | None,
    primary_metric: str | None,
    metric_direction: str | None,
    target_value: float | None,
) -> InitProjectMetadata:
    """Validate optional context flags before project creation."""

    cleaned_goal = _validate_optional_project_text(goal, "Project goal")
    cleaned_objective = _validate_optional_project_text(
        current_objective,
        "Project objective",
    )
    cleaned_metric = _validate_optional_project_text(primary_metric, "Project metric")
    cleaned_direction = _validate_metric_direction(metric_direction)
    if target_value is not None and not math.isfinite(target_value):
        raise PmemValidationError("Project target must be finite.")
    if target_value is not None and (cleaned_metric is None or cleaned_direction is None):
        raise PmemValidationError("Project target requires metric and metric direction.")
    if cleaned_direction is not None and cleaned_metric is None:
        raise PmemValidationError("Metric direction requires project metric.")
    return InitProjectMetadata(
        goal=cleaned_goal,
        current_objective=cleaned_objective,
        primary_metric=cleaned_metric,
        metric_direction=cleaned_direction,
        target_value=target_value,
    )


def _validate_optional_project_text(value: str | None, label: str) -> str | None:
    """Validate optional project context text."""

    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        raise PmemValidationError(f"{label} cannot be blank.")
    if len(cleaned) > MAX_PROJECT_TEXT_LENGTH:
        raise PmemValidationError(f"{label} is too long.")
    if any(ord(character) < 32 for character in cleaned):
        raise PmemValidationError(f"{label} contains unsupported control characters.")
    return cleaned


def _validate_metric_direction(metric_direction: str | None) -> str | None:
    """Normalize and validate metric direction flag."""

    if metric_direction is None:
        return None
    cleaned = metric_direction.strip().lower()
    if cleaned not in {MetricDirection.MAX.value, MetricDirection.MIN.value}:
        raise PmemValidationError("Metric direction must be 'max' or 'min'.")
    return cleaned


def _target_json(metadata: InitProjectMetadata) -> dict[str, object]:
    """Build the project init project target JSON payload."""

    if metadata.target_value is None:
        return {}
    return {
        "target_value": metadata.target_value,
        "baseline": None,
        "constraints": {},
        "owner_notes": None,
        "history": [],
    }


def _assert_metadata_compatible(record: ProjectRecord, metadata: InitProjectMetadata) -> None:
    """Prevent repeated init from silently changing project context."""

    expected_pairs = (
        ("goal", record.goal, metadata.goal),
        ("objective", record.current_objective, metadata.current_objective),
        ("metric", record.primary_metric, metadata.primary_metric),
        ("metric direction", record.metric_direction, metadata.metric_direction),
    )
    for label, existing, requested in expected_pairs:
        if requested is not None and existing != requested:
            raise PmemConflictError(f"projmem is already initialized with a different {label}.")

    if metadata.target_value is not None:
        existing_target = json.loads(record.target_json)
        if existing_target.get("target_value") != metadata.target_value:
            raise PmemConflictError("projmem is already initialized with a different target.")


def _utc_now_iso() -> str:
    """Return compact UTC ISO timestamp for file tracking records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
