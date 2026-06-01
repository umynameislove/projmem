"""Application-level errors with safe public messages.

Database and filesystem exceptions often contain local paths, SQL fragments, or
raw driver wording. CLI/service code should raise these errors instead of
showing infrastructure exceptions directly to users.
"""

from __future__ import annotations


class PmemError(Exception):
    """Base error for expected pmem failures."""

    public_message = "pmem operation failed."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.public_message)


class PmemValidationError(PmemError):
    """Input or shape validation failed before persistence."""

    public_message = "Input validation failed."


class PmemDomainError(PmemError):
    """A domain rule was violated."""

    public_message = "Project memory rule failed."


class PmemNotFoundError(PmemError):
    """A requested project-memory record does not exist."""

    public_message = "Requested record was not found."


class PmemConflictError(PmemError):
    """A write conflicts with existing project-memory state."""

    public_message = "Project memory state conflict."


class PmemPersistenceError(PmemError):
    """Persistence failed without exposing raw database details."""

    public_message = "Database operation failed."


class PmemSecurityError(PmemError):
    """A path, config, or operation failed a safety check."""

    public_message = "Security check failed."
