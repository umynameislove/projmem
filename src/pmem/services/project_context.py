"""Shared project-context helpers for local-memory memory commands.

Commands after `pmem init` all need the same initialization checks. Keeping the
check here prevents each service from inventing a different missing-init error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pmem.errors import PmemNotFoundError
from pmem.repositories.projects import ProjectRecord, ProjectRepository
from pmem.repositories.sqlite import project_database_path
from pmem.services.config import ProjectConfig, project_config_path, read_project_config
from pmem.services.database import ensure_database


@dataclass(frozen=True)
class ProjectContext:
    """Resolved project identity for commands that require `pmem init`."""

    root: Path
    config: ProjectConfig
    project: ProjectRecord


def require_project_context(project_root: str | Path) -> ProjectContext:
    """Return the initialized project context or raise a safe not-found error."""

    root = Path(project_root)
    config_path = project_config_path(root)
    db_path = project_database_path(root)
    if not config_path.exists() or not db_path.exists():
        raise PmemNotFoundError("projmem is not initialized. Run `pmem init` first.")

    ensure_database(root)
    config = read_project_config(config_path)
    from pmem.repositories.sqlite import connect_database

    connection = connect_database(db_path)
    try:
        project = ProjectRepository(connection).get_by_id(config.project_id)
    finally:
        connection.close()

    if project is None:
        raise PmemNotFoundError("projmem is not initialized. Run `pmem init` first.")
    return ProjectContext(root=root, config=config, project=project)
