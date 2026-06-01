"""baseline CLI-to-SQLite baseline workflow tests."""

import json
import sys

import pytest
from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database

runner = CliRunner()


def _run_with_metric(metric_value: float) -> str:
    """Run a command that writes a metric file and return the captured run id."""

    script = (
        "from pathlib import Path; import json; "
        f"Path('metrics.json').write_text(json.dumps({{'accuracy': {metric_value}}}), "
        "encoding='utf-8')"
    )
    result = runner.invoke(
        app,
        ["run", "--metrics", "metrics.json", "--", sys.executable, "-c", script],
    )
    assert result.exit_code == 0
    return result.stdout.split()[1]


def test_cli_baseline_and_compare_keep_database_integrity(monkeypatch, tmp_path) -> None:
    """baseline should mark and compare baseline runs through the real CLI."""

    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "--name", "demo"])
    baseline_run_id = _run_with_metric(0.8)
    new_run_id = _run_with_metric(0.85)

    baseline_result = runner.invoke(app, ["baseline", baseline_run_id])
    compare_result = runner.invoke(app, ["baseline", new_run_id, "--compare"])

    assert init_result.exit_code == 0
    assert baseline_result.exit_code == 0
    assert f"Marked {baseline_run_id} as baseline." in baseline_result.stdout
    assert compare_result.exit_code == 0
    assert f"Compared {new_run_id} to baseline." in compare_result.stdout
    assert f"baseline_run_id: {baseline_run_id}" in compare_result.stdout
    assert "accuracy: +0.050000" in compare_result.stdout

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        experiment_metadata = json.loads(
            connection.execute("SELECT metadata_json FROM experiments").fetchone()[0]
        )
        evaluation = json.loads(
            connection.execute(
                "SELECT evaluation_json FROM runs WHERE run_id = ?",
                (new_run_id,),
            ).fetchone()[0]
        )
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()

    assert experiment_metadata["baseline_run_id"] == baseline_run_id
    assert evaluation["baseline_comparison"]["baseline_run_id"] == baseline_run_id
    assert evaluation["baseline_comparison"]["metric_deltas"] == pytest.approx({"accuracy": 0.05})
    assert foreign_key_rows == []
    assert integrity == "ok"
