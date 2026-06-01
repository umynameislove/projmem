"""export-bundle provenance evidence bundle and replay-oriented provenance tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database

runner = CliRunner()
FROZEN_AT = "2026-05-22T00:00:00Z"


def test_evidence_bundle_provenance_and_replay_audit_chain(monkeypatch, tmp_path) -> None:
    """A bundle should carry safe provenance and be traceable through apply/resolve audit."""

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    monkeypatch.chdir(source)
    _seed_project(source)
    export_result = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "bundle.json",
            "--freeze-timestamp",
            FROZEN_AT,
            "--json",
        ],
    )
    bundle_path = source / "bundle.json"
    bundle_text = bundle_path.read_text(encoding="utf-8")
    bundle = json.loads(bundle_text)
    source_hash = "sha256:" + hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    assert export_result.exit_code == 0
    assert bundle["manifest"]["generated_at"] == FROZEN_AT
    assert bundle["manifest"]["freeze_timestamp"] is True
    assert bundle["provenance"]["tool"] == "projmem"
    assert bundle["provenance"]["source"] == "local-export"
    assert bundle["provenance"]["tool_version"]
    assert "remote_url" not in bundle_text
    assert str(tmp_path) not in bundle_text

    monkeypatch.chdir(target)
    assert runner.invoke(app, ["init", "--name", "target"]).exit_code == 0
    (target / "incoming.json").write_text(bundle_text, encoding="utf-8")

    dry_run = runner.invoke(app, ["import", "--dry-run", "incoming.json", "--json"])
    apply_result = runner.invoke(
        app,
        ["import", "--apply", "incoming.json", "--confirm", "--json"],
    )
    conflict_result = runner.invoke(app, ["conflict-check", "incoming.json", "--json"])
    conflict_payload = json.loads(conflict_result.stdout)
    already_applied = next(
        item
        for item in conflict_payload["conflicts"]
        if item["conflict_type"] == "already_applied_package"
    )
    resolve_result = runner.invoke(
        app,
        ["resolve", already_applied["conflict_id"], "--action", "skip", "--json"],
    )

    assert dry_run.exit_code == 0
    assert apply_result.exit_code == 0
    assert json.loads(apply_result.stdout)["source_hash"] == source_hash
    assert conflict_result.exit_code == 0
    assert resolve_result.exit_code == 0
    assert _pragma_status(target) == {"integrity_check": "ok", "foreign_key_check": []}
    assert _import_jobs(target)[0]["source_hash"] == source_hash
    assert {event["event_type"] for event in _audit_events(target)} == {
        "import.apply_quarantined",
        "conflict.resolution_recorded",
    }


def test_evidence_bundle_is_byte_stable_with_frozen_timestamp(monkeypatch, tmp_path) -> None:
    """Frozen timestamps are the portability layer reproducibility switch for bundle bytes."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)

    first = runner.invoke(
        app,
        ["export-bundle", "--out", "a.json", "--freeze-timestamp", FROZEN_AT],
    )
    second = runner.invoke(
        app,
        ["export-bundle", "--out", "b.json", "--freeze-timestamp", FROZEN_AT],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def _seed_project(project_root: Path) -> str:
    (project_root / "README.md").write_text("demo\n", encoding="utf-8")
    assert runner.invoke(app, ["init", "--name", "evidence-demo"]).exit_code == 0
    assert runner.invoke(app, ["track", "README.md"]).exit_code == 0
    run_id = _run_with_artifact(project_root)
    assert (
        runner.invoke(
            app,
            [
                "log-failure",
                run_id,
                "MetricRegression",
                "Accuracy dropped once.",
                "--root-cause",
                "Bad split.",
                "--lesson",
                "Pin seed.",
                "--tag",
                "reproducibility",
            ],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app,
            ["log-decision", "Keep baseline.", "--rationale", "It is reproducible."],
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["note", "Review bundle.", "--run-id", run_id]).exit_code == 0
    return run_id


def _run_with_artifact(project_root: Path) -> str:
    script = (
        "from pathlib import Path; import json; "
        "Path('metrics.json').write_text(json.dumps({'accuracy': 0.91}), encoding='utf-8'); "
        "Path('model.txt').write_text('weights', encoding='utf-8'); "
        "print('ok')"
    )
    result = runner.invoke(
        app,
        [
            "run",
            "--metrics",
            "metrics.json",
            "--artifact",
            "model.txt",
            "--",
            sys.executable,
            "-c",
            script,
        ],
    )
    assert result.exit_code == 0
    assert (project_root / "model.txt").is_file()
    return result.stdout.split()[1]


def _pragma_status(project_root: Path) -> dict[str, Any]:
    connection = connect_database(project_root / ".pmem" / "pmem.db")
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        fk_rows = [dict(row) for row in connection.execute("PRAGMA foreign_key_check").fetchall()]
        return {"integrity_check": integrity, "foreign_key_check": fk_rows}
    finally:
        connection.close()


def _import_jobs(project_root: Path) -> list[dict[str, Any]]:
    connection = connect_database(project_root / ".pmem" / "pmem.db")
    try:
        rows = connection.execute(
            "SELECT source_hash, status, provenance_source FROM import_jobs ORDER BY created_at"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def _audit_events(project_root: Path) -> list[dict[str, Any]]:
    connection = connect_database(project_root / ".pmem" / "pmem.db")
    try:
        rows = connection.execute(
            "SELECT event_type, entity_type, before_hash, after_hash FROM audit_events "
            "ORDER BY timestamp"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
