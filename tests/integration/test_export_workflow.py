"""project export export workflow tests."""

from __future__ import annotations

import json
import sys

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database

runner = CliRunner()


def test_export_json_round_trip_includes_local_memory_entities(monkeypatch, tmp_path) -> None:
    """project export export should serialize current project evidence without artifact contents."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    init_result = runner.invoke(
        app,
        [
            "init",
            "--name",
            "export-demo",
            "--objective",
            "Train classifier",
            "--metric",
            "accuracy",
            "--metric-direction",
            "max",
            "--target",
            "0.9",
        ],
    )
    track_result = runner.invoke(app, ["track", "README.md"])
    run_id = _run_with_metric(tmp_path, "accuracy", 0.91)
    failure_result = runner.invoke(
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
    )
    decision_result = runner.invoke(
        app,
        [
            "log-decision",
            "Keep baseline.",
            "--rationale",
            "It is reproducible.",
            "--author",
            "qa",
        ],
    )
    note_result = runner.invoke(app, ["note", "Review export.", "--run-id", run_id])

    result = runner.invoke(app, ["export", "--json"])
    payload = json.loads(result.stdout)
    entities = payload["entities"]

    assert init_result.exit_code == 0
    assert track_result.exit_code == 0
    assert failure_result.exit_code == 0
    assert decision_result.exit_code == 0
    assert note_result.exit_code == 0
    assert result.exit_code == 0
    assert payload["schema_version"] == "schema-v1"
    assert payload["export_at"].endswith("Z")
    assert entities["projects"][0]["name"] == "export-demo"
    assert entities["projects"][0]["target"]["target_value"] == 0.9
    assert len(entities["tracked_paths"]) == 1
    assert entities["tracked_paths"][0]["path"] == "README.md"
    assert len(entities["experiments"]) == 1
    assert len(entities["runs"]) == 1
    assert entities["runs"][0]["run_id"] == run_id
    assert entities["runs"][0]["metrics"] == {"accuracy": 0.91}
    assert "metrics_json" not in entities["runs"][0]
    assert len(entities["failures"]) == 1
    assert entities["failures"][0]["tags"] == ["reproducibility"]
    assert len(entities["decisions"]) == 1
    assert entities["decisions"][0]["author"] == "qa"
    assert len(entities["notes"]) == 1
    assert entities["notes"][0]["run_id"] == run_id


def test_export_requires_init_clean_error(monkeypatch, tmp_path) -> None:
    """project export missing-init export errors should stay traceback-free."""

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["export", "--json"])

    assert result.exit_code == 1
    assert "projmem is not initialized. Run `pmem init` first." in result.stdout
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()


def test_export_requires_json_flag(monkeypatch, tmp_path) -> None:
    """project export export should avoid ambiguous human output until a text contract exists."""

    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "--name", "export-demo"])

    result = runner.invoke(app, ["export"])

    assert init_result.exit_code == 0
    assert result.exit_code == 1
    assert "Export currently supports --json only." in result.stdout
    assert "Traceback" not in result.stdout


def test_export_workflow_keeps_database_integrity(monkeypatch, tmp_path) -> None:
    """project export export is read-only and should leave SQLite integrity checks clean."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "export-demo"])
    _run_with_metric(tmp_path, "accuracy", 0.91)

    export_result = runner.invoke(app, ["export", "--json"])
    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()

    assert export_result.exit_code == 0
    assert integrity == "ok"
    assert foreign_key_rows == []


def _run_with_metric(tmp_path, metric: str, value: float) -> str:
    script = (
        "from pathlib import Path; import json; "
        f"Path('metrics.json').write_text(json.dumps({{{metric!r}: {value}}}), "
        "encoding='utf-8'); print('ok')"
    )
    result = runner.invoke(
        app,
        ["run", "--metrics", "metrics.json", "--", sys.executable, "-c", script],
    )
    assert result.exit_code == 0
    assert (tmp_path / "metrics.json").is_file()
    return result.stdout.split()[1]
