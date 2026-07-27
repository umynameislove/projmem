"""Security regressions for the read-only ``pmem status`` text boundary."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pmem.cli.app import app

runner = CliRunner()


def test_status_cli_does_not_mutate_private_project_state(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _init_project("read-only")
    _run_python("print('ok')")
    before = _private_snapshot(tmp_path)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert _private_snapshot(tmp_path) == before
    assert not list((tmp_path / ".pmem").glob("pmem.db-*"))
    assert not any(path.name.endswith(".bak") for path in (tmp_path / ".pmem").rglob("*"))


def test_status_cli_does_not_expose_command_config_or_failure_text(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _init_project("privacy")
    command_sensitive_marker = "COMMAND_SECRET_4f4d"
    config_sensitive_marker = "CONFIG_SECRET_83b9"
    failure_sensitive_marker = "FAILURE_SECRET_1ac7"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"api_key": config_sensitive_marker, "learning_rate": 0.1}),
        encoding="utf-8",
    )
    failed = runner.invoke(
        app,
        [
            "run",
            "--config",
            "config.json",
            "--",
            sys.executable,
            "-c",
            f"import sys; marker={command_sensitive_marker!r}; sys.exit(2)",
        ],
    )
    run_id = failed.stdout.split()[1]
    logged = runner.invoke(
        app,
        ["log-failure", run_id, "RuntimeError", failure_sensitive_marker],
    )

    result = runner.invoke(app, ["status"])

    assert failed.exit_code == 0
    assert logged.exit_code == 0
    assert result.exit_code == 0
    assert command_sensitive_marker not in result.stdout
    assert config_sensitive_marker not in result.stdout
    assert failure_sensitive_marker not in result.stdout
    assert str(tmp_path) not in result.stdout
    assert "raw_text_in_output=false" in result.stdout


def test_status_cli_redacts_absolute_path_project_text(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    private_objective = str(tmp_path / "private" / "research-notes.txt")
    init_result = runner.invoke(
        app,
        ["init", "--name", "path-redaction", "--objective", private_objective],
    )

    result = runner.invoke(app, ["status"])

    assert init_result.exit_code == 0
    assert result.exit_code == 0
    assert private_objective not in result.stdout
    assert str(tmp_path) not in result.stdout
    assert "Objective: redacted_objective_" in result.stdout
    assert "data_quality/status_text_redacted" in result.stdout


def test_status_cli_rejects_corrupt_database_without_internal_leak_or_repair(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _init_project("corrupt")
    db_path = tmp_path / ".pmem" / "pmem.db"
    db_path.write_bytes(b"not sqlite and must remain unchanged")
    before = db_path.read_bytes()

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    assert db_path.read_bytes() == before
    assert "Traceback" not in result.stdout
    assert "file is not a database" not in result.stdout.lower()
    assert "SELECT " not in result.stdout
    assert str(tmp_path) not in result.stdout


def test_status_cli_rejects_active_wal_without_touching_sidecars(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _init_project("active-wal")
    db_path = tmp_path / ".pmem" / "pmem.db"
    writer = sqlite3.connect(db_path)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE wal_probe (value TEXT)")
        writer.execute("INSERT INTO wal_probe VALUES ('private')")
        writer.commit()
        before = _private_snapshot(tmp_path)

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 1
        assert "active SQLite sidecar state" in result.stdout
        assert _private_snapshot(tmp_path) == before
        assert str(tmp_path) not in result.stdout
    finally:
        writer.close()


def test_status_cli_does_not_follow_graph_symlink(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _init_project("graph-symlink")
    _run_python("print('ok')")
    outside_sensitive_marker = "OUTSIDE_GRAPH_SECRET_7d2e"
    outside = tmp_path.parent / "outside-status-graph.json"
    outside.write_text(outside_sensitive_marker, encoding="utf-8")
    graph_path = tmp_path / ".pmem" / "graph.json"
    try:
        os.symlink(outside, graph_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Graph: invalid reason=graph_symlink" in result.stdout
    assert "Action: resolve_graph_symlink" in result.stdout
    assert "Command: pmem graph --help" in result.stdout
    assert outside_sensitive_marker not in result.stdout
    assert outside.read_text(encoding="utf-8") == outside_sensitive_marker


def test_status_json_does_not_mutate_private_project_state(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _init_project("json-read-only")
    _run_python("print('ok')")
    before = _private_snapshot(tmp_path)

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0
    assert _private_snapshot(tmp_path) == before
    assert not list((tmp_path / ".pmem").glob("pmem.db-*"))
    assert not any(path.name.endswith(".bak") for path in (tmp_path / ".pmem").rglob("*"))


def test_status_json_does_not_expose_sensitive_markers_in_raw_stdout(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _init_project("json-privacy")
    command_sensitive_marker = "COMMAND_SECRET_4f4d"
    config_sensitive_marker = "CONFIG_SECRET_83b9"
    failure_sensitive_marker = "FAILURE_SECRET_1ac7"
    (tmp_path / "config.json").write_text(
        json.dumps({"api_key": config_sensitive_marker, "learning_rate": 0.1}),
        encoding="utf-8",
    )
    failed = runner.invoke(
        app,
        [
            "run",
            "--config",
            "config.json",
            "--",
            sys.executable,
            "-c",
            f"import sys; marker={command_sensitive_marker!r}; sys.exit(2)",
        ],
    )
    run_id = failed.stdout.split()[1]
    logged = runner.invoke(app, ["log-failure", run_id, "RuntimeError", failure_sensitive_marker])

    result = runner.invoke(app, ["status", "--json"])

    assert failed.exit_code == 0
    assert logged.exit_code == 0
    assert result.exit_code == 0
    raw = result.stdout
    for marker in (command_sensitive_marker, config_sensitive_marker, failure_sensitive_marker):
        assert marker not in raw
    assert str(tmp_path) not in raw
    document = json.loads(raw)
    flat = json.dumps(document)
    for marker in (command_sensitive_marker, config_sensitive_marker, failure_sensitive_marker):
        assert marker not in flat
    assert document["raw_text_in_output"] is False


def test_status_json_does_not_follow_graph_symlink(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _init_project("json-graph-symlink")
    _run_python("print('ok')")
    outside_sensitive_marker = "OUTSIDE_GRAPH_SECRET_7d2e"
    outside = tmp_path.parent / "outside-json-graph.json"
    outside.write_text(outside_sensitive_marker, encoding="utf-8")
    graph_path = tmp_path / ".pmem" / "graph.json"
    try:
        os.symlink(outside, graph_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0
    assert outside_sensitive_marker not in result.stdout
    document = json.loads(result.stdout)
    assert document["graph"]["state"] == "invalid"
    assert document["graph"]["reason_code"] == "graph_symlink"
    assert document["next_action"]["action_id"] == "resolve_graph_symlink"
    assert outside.read_text(encoding="utf-8") == outside_sensitive_marker


def test_status_json_rejects_corrupt_database_without_leak(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _init_project("json-corrupt")
    db_path = tmp_path / ".pmem" / "pmem.db"
    db_path.write_bytes(b"not sqlite and must remain unchanged")
    before = db_path.read_bytes()

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 1
    assert db_path.read_bytes() == before
    assert '"schema_version"' not in result.stdout
    assert "Traceback" not in result.stdout
    assert "file is not a database" not in result.stdout.lower()
    assert "SELECT " not in result.stdout
    assert str(tmp_path) not in result.stdout


def test_status_json_rejects_active_wal_without_touching_sidecars(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _init_project("json-active-wal")
    db_path = tmp_path / ".pmem" / "pmem.db"
    writer = sqlite3.connect(db_path)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE wal_probe (value TEXT)")
        writer.execute("INSERT INTO wal_probe VALUES ('private')")
        writer.commit()
        before = _private_snapshot(tmp_path)

        result = runner.invoke(app, ["status", "--json"])

        assert result.exit_code == 1
        assert "active SQLite sidecar state" in result.stdout
        assert '"schema_version"' not in result.stdout
        assert _private_snapshot(tmp_path) == before
    finally:
        writer.close()


def test_status_json_escapes_injection_and_redacts_unsafe_text(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    # Safe but tricky objective: quotes and JSON-looking braces (no path/control).
    spoof_objective = 'literal "quote" and {"schema_version": "spoofed"}'
    assert (
        runner.invoke(
            app, ["init", "--name", "json-injection", "--objective", spoof_objective]
        ).exit_code
        == 0
    )

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0
    document = json.loads(result.stdout)  # structure survives the tricky string
    assert document["schema_version"] == "status-v1"  # not the spoofed value
    assert document["project"]["objective"] == spoof_objective  # kept but JSON-escaped
    assert set(document) == {
        "schema_version",
        "project",
        "metric",
        "counts",
        "best_run",
        "baseline",
        "graph",
        "recommendations",
        "warnings",
        "next_action",
        "database_mutation",
        "network",
        "raw_text_in_output",
    }


def test_status_json_redacts_absolute_path_project_text(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    private_objective = str(tmp_path / "private" / "notes.txt")
    assert (
        runner.invoke(
            app, ["init", "--name", "json-path-redaction", "--objective", private_objective]
        ).exit_code
        == 0
    )

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0
    assert private_objective not in result.stdout
    assert str(tmp_path) not in result.stdout
    assert "\x1b" not in result.stdout  # JSON never emits raw control bytes
    document = json.loads(result.stdout)
    assert document["project"]["objective"].startswith("redacted_objective_")


@pytest.mark.parametrize(
    "unsafe_objective",
    [
        "line one\nline two",
        "terminal\x1b[31mred",
    ],
)
def test_status_json_redacts_control_character_text_without_mutation(
    monkeypatch,
    tmp_path,
    unsafe_objective,
) -> None:
    monkeypatch.chdir(tmp_path)
    _init_project("json-control-redaction")
    db_path = tmp_path / ".pmem" / "pmem.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE projects SET current_objective = ?",
            (unsafe_objective,),
        )
        connection.commit()
    before = _private_snapshot(tmp_path)

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0
    assert unsafe_objective not in result.stdout
    assert "\x1b" not in result.stdout
    document = json.loads(result.stdout)
    assert document["project"]["objective"].startswith("redacted_objective_")
    assert _private_snapshot(tmp_path) == before


def _init_project(name: str) -> None:
    result = runner.invoke(app, ["init", "--name", name])
    assert result.exit_code == 0


def _run_python(script: str):
    result = runner.invoke(app, ["run", "--", sys.executable, "-c", script])
    assert result.exit_code == 0
    return result


def _private_snapshot(root: Path) -> dict[str, tuple[str, bytes, int, int]]:
    snapshot: dict[str, tuple[str, bytes, int, int]] = {}
    private_root = root / ".pmem"
    for path in sorted(private_root.rglob("*")):
        stat = path.lstat()
        relative = str(path.relative_to(root))
        if path.is_symlink():
            snapshot[relative] = (
                "symlink",
                os.readlink(path).encode(),
                stat.st_mtime_ns,
                stat.st_mode,
            )
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes(), stat.st_mtime_ns, stat.st_mode)
        elif path.is_dir():
            snapshot[relative] = ("directory", b"", stat.st_mtime_ns, stat.st_mode)
    return snapshot
