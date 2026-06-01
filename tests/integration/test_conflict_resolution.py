"""conflict resolution non-destructive conflict resolution audit tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.errors import PmemPersistenceError
from pmem.repositories.sqlite import connect_database
from pmem.services.conflict_resolution import record_conflict_resolution

runner = CliRunner()
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def test_resolve_records_audit_event_without_overwriting_data(monkeypatch, tmp_path) -> None:
    """conflict resolution should record operator intent and leave canonical memory untouched."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "resolve-demo"]).exit_code == 0
    before = _row_counts(tmp_path)

    result = runner.invoke(
        app,
        [
            "resolve",
            "conflict_demo",
            "--action",
            "keep-local",
            "--before-hash",
            HASH_A,
            "--after-hash",
            HASH_A,
            "--json",
        ],
    )
    payload = json.loads(result.stdout)
    after = _row_counts(tmp_path)
    audit_row = _latest_audit_event(tmp_path)

    assert result.exit_code == 0
    assert payload["ok"] is True
    assert payload["database_mutation"] == "audit_event_only"
    assert payload["hash_evidence_complete"] is True
    assert after["audit_events"] == before["audit_events"] + 1
    for table in ("projects", "experiments", "runs", "failures", "decisions", "notes"):
        assert after[table] == before[table]
    assert audit_row["event_type"] == "conflict.resolution_recorded"
    assert audit_row["before_hash"] == HASH_A
    assert audit_row["after_hash"] == HASH_A
    assert "Traceback" not in result.stdout


def test_resolve_text_output_and_optional_hash_gap(monkeypatch, tmp_path) -> None:
    """Human output should clarify that resolution is audit-only."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "resolve-demo"]).exit_code == 0

    result = runner.invoke(app, ["resolve", "conflict_text", "--action", "skip"])

    assert result.exit_code == 0
    assert "Conflict resolution recorded." in result.stdout
    assert "Database mutation: audit_events only" in result.stdout
    assert "Canonical data mutation: none" in result.stdout
    assert _latest_audit_event(tmp_path)["before_hash"] is None


def test_resolve_destructive_action_requires_confirm(monkeypatch, tmp_path) -> None:
    """Overwrite-like policies must not record silently."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "resolve-demo"]).exit_code == 0
    before = _row_counts(tmp_path)

    rejected = runner.invoke(
        app,
        [
            "resolve",
            "conflict_demo",
            "--action",
            "take-imported",
            "--before-hash",
            HASH_A,
            "--after-hash",
            HASH_B,
            "--json",
        ],
    )
    after_rejected = _row_counts(tmp_path)
    accepted = runner.invoke(
        app,
        [
            "resolve",
            "conflict_demo",
            "--action",
            "take-imported",
            "--before-hash",
            HASH_A,
            "--after-hash",
            HASH_B,
            "--confirm",
            "--json",
        ],
    )
    after_accepted = _row_counts(tmp_path)
    payload = json.loads(accepted.stdout)

    assert rejected.exit_code == 1
    assert "require --confirm" in rejected.stdout
    assert after_rejected == before
    assert accepted.exit_code == 0
    assert payload["destructive_confirmed"] is True
    assert payload["database_mutation"] == "audit_event_only"
    assert after_accepted["audit_events"] == before["audit_events"] + 1


def test_resolve_rejects_invalid_action_and_hash_safely(monkeypatch, tmp_path) -> None:
    """CLI errors should be actionable and traceback-free."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "resolve-demo"]).exit_code == 0

    bad_action = runner.invoke(app, ["resolve", "conflict_demo", "--action", "merge-now"])
    blank_conflict = runner.invoke(app, ["resolve", "", "--action", "skip"])
    bad_hash = runner.invoke(
        app,
        [
            "resolve",
            "conflict_demo",
            "--action",
            "skip",
            "--before-hash",
            "not-a-hash",
        ],
    )

    assert bad_action.exit_code == 1
    assert "Resolution action must be one of" in bad_action.stdout
    assert blank_conflict.exit_code == 1
    assert "Conflict id cannot be blank" in blank_conflict.stdout
    assert bad_hash.exit_code == 1
    assert "before_hash must use sha256" in bad_hash.stdout
    assert "Traceback" not in bad_action.stdout
    assert "Traceback" not in bad_hash.stdout


def test_resolve_rolls_back_audit_event_on_insert_failure(monkeypatch, tmp_path) -> None:
    """A failed conflict resolution audit write should leave the database unchanged."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "resolve-demo"]).exit_code == 0
    before = _row_counts(tmp_path)

    def fail_insert(*args: Any, **kwargs: Any) -> None:
        raise PmemPersistenceError("forced audit failure")

    monkeypatch.setattr(
        "pmem.services.conflict_resolution.AuditEventRepository.insert",
        fail_insert,
    )

    with pytest.raises(PmemPersistenceError):
        record_conflict_resolution(
            tmp_path,
            conflict_id="conflict_demo",
            action="keep-local",
            before_hash=HASH_A,
            after_hash=HASH_A,
        )

    assert _row_counts(tmp_path) == before


def _row_counts(project_root: Path) -> dict[str, int]:
    connection = connect_database(project_root / ".pmem" / "pmem.db")
    try:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "projects",
                "experiments",
                "runs",
                "failures",
                "decisions",
                "notes",
                "audit_events",
            )
        }
    finally:
        connection.close()


def _latest_audit_event(project_root: Path) -> dict[str, Any]:
    connection = connect_database(project_root / ".pmem" / "pmem.db")
    try:
        row = connection.execute(
            """
            SELECT event_type, before_hash, after_hash, metadata_json
            FROM audit_events
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).fetchone()
        return dict(row)
    finally:
        connection.close()
