"""SQL-safety tests for portability repositories."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.errors import PmemValidationError
from pmem.repositories.portability import AuditEventRepository, SharedPathRepository
from pmem.repositories.sqlite import connect_database
from pmem.services.conflict_resolution import _validate_conflict_id

runner = CliRunner()
HASH_A = "sha256:" + "a" * 64


def test_shared_path_repository_parameterizes_alias_and_path(monkeypatch, tmp_path) -> None:
    """SQL-like text should be stored as data, not executed as SQL."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "sql-demo"]).exit_code == 0
    malicious_alias = "team'); DROP TABLE shared_paths; --"
    malicious_path = "/tmp/shared'); DROP TABLE projects; --"
    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        record = SharedPathRepository(connection).create(
            shared_path_id="share_sql",
            alias=malicious_alias,
            path=malicious_path,
            mode="read",
            policy={"note": "parameterized"},
            created_at="2026-05-22T00:00:00Z",
        )
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        connection.close()

    assert record.alias == malicious_alias
    assert {"shared_paths", "projects"}.issubset(tables)


def test_audit_event_repository_parameterizes_metadata(monkeypatch, tmp_path) -> None:
    """Audit metadata can contain SQL-like text without damaging the schema."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "audit-sql-demo"]).exit_code == 0
    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        event = AuditEventRepository(connection).insert(
            event_id="audit_sql",
            event_type="conflict.resolution_recorded",
            entity_type="conflict",
            entity_id="conflict_sql",
            before_hash=HASH_A,
            after_hash=HASH_A,
            actor="local",
            timestamp="2026-05-22T00:00:00Z",
            metadata={"comment": "x'); DROP TABLE audit_events; --"},
        )
        connection.commit()
        row = connection.execute(
            "SELECT metadata_json FROM audit_events WHERE id = ?",
            (event.id,),
        ).fetchone()
        count = int(connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0])
    finally:
        connection.close()

    assert count == 1
    assert json.loads(str(row["metadata_json"])) == {"comment": "x'); DROP TABLE audit_events; --"}


def test_validate_conflict_id_blocks_sql_injection_characters() -> None:
    """Conflict IDs containing SQL injection characters must be rejected before DB insertion."""

    for malicious in (
        "'); DROP TABLE audit_events; --",
        "conflict_x; DROP TABLE audit_events; --",
        "1 OR 1=1",
        "conflict\x00_id",
        "id with spaces",
        "",
        "  ",
    ):
        with pytest.raises(PmemValidationError):
            _validate_conflict_id(malicious)


def test_validate_conflict_id_accepts_safe_identifiers() -> None:
    """Conflict IDs that match the safe character set must pass validation."""

    for safe in (
        "conflict_abc-123",
        "CONFLICT.ID:v2",
        "abc",
        "A" * 128,
    ):
        result = _validate_conflict_id(safe)
        assert result == safe.strip()
