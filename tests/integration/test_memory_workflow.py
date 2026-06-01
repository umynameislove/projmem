"""memory logging CLI-to-SQLite memory workflow tests."""

import sys

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database

runner = CliRunner()


def test_failure_decision_note_workflow_keeps_database_integrity(monkeypatch, tmp_path) -> None:
    """memory logging commands should persist rows without breaking SQLite integrity."""

    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "--name", "demo"])
    run_result = runner.invoke(app, ["run", "--", sys.executable, "-c", "print('ok')"])
    run_id = run_result.stdout.split()[1]

    failure_result = runner.invoke(
        app,
        [
            "log-failure",
            run_id,
            "MetricRegression",
            "Accuracy dropped below target.",
            "--severity",
            "medium",
            "--source",
            "user_confirmed",
            "--tag",
            "data quality",
        ],
    )
    decision_result = runner.invoke(
        app,
        ["log-decision", "Keep the first baseline.", "--rationale", "It is reproducible."],
    )
    note_result = runner.invoke(
        app,
        ["note", "Check macro F1 next.", "--run-id", run_id, "--tag", "follow up"],
    )

    assert init_result.exit_code == 0
    assert run_result.exit_code == 0
    assert failure_result.exit_code == 0
    assert decision_result.exit_code == 0
    assert note_result.exit_code == 0

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        counts = {
            "failures": connection.execute("SELECT count(*) FROM failures").fetchone()[0],
            "decisions": connection.execute("SELECT count(*) FROM decisions").fetchone()[0],
            "notes": connection.execute("SELECT count(*) FROM notes").fetchone()[0],
        }
        failure_row = connection.execute(
            "SELECT run_id, severity, tags_json, source FROM failures"
        ).fetchone()
        note_row = connection.execute("SELECT run_id, tags_json FROM notes").fetchone()
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()

    assert counts == {"failures": 1, "decisions": 1, "notes": 1}
    assert failure_row["run_id"] == run_id
    assert failure_row["severity"] == "medium"
    assert failure_row["tags_json"] == '["data_quality"]'
    assert failure_row["source"] == "user_confirmed"
    assert note_row["run_id"] == run_id
    assert note_row["tags_json"] == '["follow_up"]'
    assert foreign_key_rows == []
    assert integrity == "ok"
