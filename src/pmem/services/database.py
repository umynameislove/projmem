"""Database service use cases."""

from __future__ import annotations

from pathlib import Path

from pmem.migrations.runner import MigrationResult, apply_migrations
from pmem.repositories.sqlite import project_database_path


def ensure_database(project_root: str | Path) -> MigrationResult:
    """Create or migrate the project-local database for a project root."""

    return apply_migrations(project_database_path(project_root))
