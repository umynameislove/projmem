"""Tests for safe application error messages."""

from pmem.errors import (
    PmemConflictError,
    PmemDomainError,
    PmemError,
    PmemNotFoundError,
    PmemPersistenceError,
    PmemSecurityError,
    PmemValidationError,
)


def test_errors_have_safe_default_messages() -> None:
    """Default messages should not expose infrastructure details."""

    errors = [
        PmemError(),
        PmemValidationError(),
        PmemDomainError(),
        PmemNotFoundError(),
        PmemConflictError(),
        PmemPersistenceError(),
        PmemSecurityError(),
    ]

    assert [str(error) for error in errors] == [
        "pmem operation failed.",
        "Input validation failed.",
        "Project memory rule failed.",
        "Requested record was not found.",
        "Project memory state conflict.",
        "Database operation failed.",
        "Security check failed.",
    ]


def test_error_can_use_specific_safe_message() -> None:
    """Callers may replace the message with another public-safe string."""

    error = PmemPersistenceError("Migration failed.")

    assert str(error) == "Migration failed."
