"""Shared project-context helpers for local-memory memory commands.

Commands after `pmem init` all need the same initialization checks. Keeping the
check here prevents each service from inventing a different missing-init error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pmem.errors import PmemNotFoundError, PmemSecurityError
from pmem.migrations.runner import verify_schema_current
from pmem.repositories.projects import ProjectRecord, ProjectRepository
from pmem.repositories.sqlite import (
    PMEM_DIRNAME,
    connect_database_readonly,
    project_database_path,
)
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


def require_project_context_readonly(project_root: str | Path) -> ProjectContext:
    """Resolve the project context without ever writing to disk.

    Read-only counterpart of :func:`require_project_context`: it does **not**
    call ``ensure_database`` (no migration, backup, ``mkdir`` or ``chmod``). It
    refuses a symlinked ``.pmem`` directory or config/database file, verifies
    the schema is current (version + checksum), and reads through the shared
    immutable read-only connection policy.
    An out-of-date or tampered schema raises a safe error instead of migrating.
    """

    root = Path(project_root)
    pmem_dir = root / PMEM_DIRNAME
    config_path = project_config_path(root)
    db_path = project_database_path(root)
    if pmem_dir.is_symlink() or config_path.is_symlink() or db_path.is_symlink():
        raise PmemSecurityError("projmem state paths must not be symlinks.")
    if not config_path.exists() or not db_path.exists():
        raise PmemNotFoundError("projmem is not initialized. Run `pmem init` first.")

    config = read_project_config(config_path)
    connection = connect_database_readonly(db_path)
    try:
        verify_schema_current(connection)
        project = ProjectRepository(connection).get_by_id(config.project_id)
    finally:
        connection.close()

    if project is None:
        raise PmemNotFoundError("projmem is not initialized. Run `pmem init` first.")
    return ProjectContext(root=root, config=config, project=project)
