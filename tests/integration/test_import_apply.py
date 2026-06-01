"""import apply import apply/quarantine workflow tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.errors import PmemPersistenceError, PmemSecurityError, PmemValidationError
from pmem.repositories.sqlite import connect_database
from pmem.services.import_apply import (
    _provenance_source,
    _resolve_bundle_for_hash,
    apply_import_bundle,
)

runner = CliRunner()


def test_import_apply_refuses_without_confirm(monkeypatch, tmp_path) -> None:
    """import apply apply must require explicit confirmation after dry-run review."""

    bundle_text = _export_source_bundle(monkeypatch, tmp_path / "source")
    target = tmp_path / "target"
    _init_target_with_bundle(monkeypatch, target, bundle_text)
    before = _row_counts(target)

    result = runner.invoke(app, ["import", "--apply", "incoming.json", "--json"])
    after = _row_counts(target)

    assert result.exit_code == 1
    assert before == after
    assert "requires --confirm" in result.stdout
    assert "Traceback" not in result.stdout
    assert _import_job_count(target) == 0


def test_import_apply_writes_pending_job_and_audit_only(monkeypatch, tmp_path) -> None:
    """import apply apply should quarantine metadata without overwriting canonical memory."""

    bundle_text = _export_source_bundle(monkeypatch, tmp_path / "source")
    target = tmp_path / "target"
    _init_target_with_bundle(monkeypatch, target, bundle_text)
    before = _row_counts(target)

    result = runner.invoke(app, ["import", "--apply", "incoming.json", "--confirm", "--json"])
    payload = json.loads(result.stdout)
    after = _row_counts(target)
    pragmas = _pragma_status(target)

    assert result.exit_code == 0
    assert payload["ok"] is True
    assert payload["status"] == "pending"
    assert payload["database_mutation"] == "quarantine_pending_import_job"
    assert payload["row_counts_before"] == before
    assert payload["row_counts_after"] == after
    assert payload["integrity_check"] == "ok"
    assert payload["foreign_key_check"] == []
    assert after["import_jobs"] == before["import_jobs"] + 1
    assert after["audit_events"] == before["audit_events"] + 1
    for table in ("projects", "experiments", "runs", "failures", "decisions", "notes"):
        assert after[table] == before[table]
    assert pragmas["integrity_check"] == "ok"
    assert pragmas["foreign_key_check"] == []
    assert "Traceback" not in result.stdout


def test_import_apply_text_output_is_safe(monkeypatch, tmp_path) -> None:
    """Human output should report quarantine status without traceback internals."""

    bundle_text = _export_source_bundle(monkeypatch, tmp_path / "source")
    target = tmp_path / "target"
    _init_target_with_bundle(monkeypatch, target, bundle_text)

    result = runner.invoke(app, ["import", "--apply", "incoming.json", "--confirm"])

    assert result.exit_code == 0
    assert "Import apply: PENDING" in result.stdout
    assert "Database mutation: import_jobs/audit_events only" in result.stdout
    assert "integrity_check: ok" in result.stdout
    assert "Traceback" not in result.stdout


def test_import_apply_rolls_back_if_transaction_fails(monkeypatch, tmp_path) -> None:
    """A partial failure after import_jobs insert must leave SQLite unchanged."""

    bundle_text = _export_source_bundle(monkeypatch, tmp_path / "source")
    target = tmp_path / "target"
    _init_target_with_bundle(monkeypatch, target, bundle_text)
    before = _row_counts(target)

    def fail_after_import_job(*args: Any, **kwargs: Any) -> None:
        raise PmemPersistenceError("forced transaction failure")

    monkeypatch.setattr("pmem.services.import_apply._insert_audit_event", fail_after_import_job)

    with pytest.raises(PmemPersistenceError):
        apply_import_bundle(target, "incoming.json", confirm=True)

    after = _row_counts(target)
    assert after == before
    assert _import_job_count(target) == 0


def test_import_command_rejects_missing_or_conflicting_modes(monkeypatch, tmp_path) -> None:
    """The import command should require one explicit mode."""

    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.chdir(target)
    assert runner.invoke(app, ["init", "--name", "target"]).exit_code == 0
    (target / "incoming.json").write_text("{}", encoding="utf-8")

    missing_mode = runner.invoke(app, ["import", "incoming.json"])
    conflicting_modes = runner.invoke(app, ["import", "--dry-run", "--apply", "incoming.json"])

    assert missing_mode.exit_code == 1
    assert "requires --dry-run or --apply" in missing_mode.stdout
    assert conflicting_modes.exit_code == 1
    assert "either --dry-run or --apply" in conflicting_modes.stdout
    assert "Traceback" not in missing_mode.stdout
    assert "Traceback" not in conflicting_modes.stdout


def test_import_apply_bundle_hash_path_safety(tmp_path) -> None:
    """import apply hash resolution should keep bundle paths project-relative and safe."""

    root = tmp_path / "project"
    root.mkdir()
    (root / "bundle.json").write_text("{}", encoding="utf-8")
    (root / "bundle-dir").mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    assert _resolve_bundle_for_hash(root, "bundle.json") == root / "bundle.json"
    with pytest.raises(PmemValidationError):
        _resolve_bundle_for_hash(root, "")
    with pytest.raises(PmemSecurityError):
        _resolve_bundle_for_hash(root, "/tmp/bundle.json")
    with pytest.raises(PmemSecurityError):
        _resolve_bundle_for_hash(root, ".PMEM/bundle.json")
    with pytest.raises(PmemSecurityError):
        _resolve_bundle_for_hash(root, "../outside.json")
    with pytest.raises(PmemSecurityError):
        _resolve_bundle_for_hash(root, "bundle-dir")
    assert _provenance_source({}) is None


def test_import_apply_rejects_invalid_bundle_without_mutation(monkeypatch, tmp_path) -> None:
    """Invalid bundles should fail dry-run before any import_jobs mutation."""

    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.chdir(target)
    assert runner.invoke(app, ["init", "--name", "target"]).exit_code == 0
    (target / "incoming.json").write_text("{}", encoding="utf-8")
    before = _row_counts(target)

    result = runner.invoke(app, ["import", "--apply", "incoming.json", "--confirm", "--json"])
    after = _row_counts(target)

    assert result.exit_code == 1
    assert before == after
    assert "failed dry-run validation" in result.stdout
    assert "Traceback" not in result.stdout
    assert _import_job_count(target) == 0


def test_import_apply_help_is_available() -> None:
    """import apply CLI help should render cleanly across terminal widths."""

    result = runner.invoke(app, ["import", "--help"])

    assert result.exit_code == 0
    assert result.stdout.strip()
    assert "Traceback" not in result.stdout


def _export_source_bundle(monkeypatch: pytest.MonkeyPatch, source: Path) -> str:
    source.mkdir()
    monkeypatch.chdir(source)
    _seed_project(source)
    result = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "bundle.json",
            "--freeze-timestamp",
            "2026-01-01T00:00:00Z",
            "--json",
        ],
    )
    assert result.exit_code == 0
    return (source / "bundle.json").read_text(encoding="utf-8")


def _init_target_with_bundle(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    bundle_text: str,
) -> None:
    target.mkdir()
    monkeypatch.chdir(target)
    assert runner.invoke(app, ["init", "--name", "target"]).exit_code == 0
    (target / "incoming.json").write_text(bundle_text, encoding="utf-8")


def _seed_project(project_root: Path) -> str:
    (project_root / "README.md").write_text("demo\n", encoding="utf-8")
    assert runner.invoke(app, ["init", "--name", "source"]).exit_code == 0
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
            ],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app, ["log-decision", "Keep baseline.", "--rationale", "It is reproducible."]
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


def _row_counts(project_root: Path) -> dict[str, int]:
    tables = (
        "projects",
        "experiments",
        "runs",
        "failures",
        "decisions",
        "notes",
        "tracked_paths",
        "export_packages",
        "import_jobs",
        "shared_paths",
        "audit_events",
    )
    connection = connect_database(project_root / ".pmem" / "pmem.db")
    try:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
    finally:
        connection.close()


def _import_job_count(project_root: Path) -> int:
    return _row_counts(project_root)["import_jobs"]


def _pragma_status(project_root: Path) -> dict[str, Any]:
    connection = connect_database(project_root / ".pmem" / "pmem.db")
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        fk_rows = [dict(row) for row in connection.execute("PRAGMA foreign_key_check").fetchall()]
        return {"integrity_check": integrity, "foreign_key_check": fk_rows}
    finally:
        connection.close()
