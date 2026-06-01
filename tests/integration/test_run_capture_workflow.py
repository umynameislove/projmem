"""run capture/reproducibility metadata end-to-end run capture tests."""

import json
import sys

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database
from pmem.utils.hashing import compute_file_hash

runner = CliRunner()


def test_cli_init_run_with_metadata_keeps_database_integrity(monkeypatch, tmp_path) -> None:
    """The real run capture/reproducibility metadata CLI workflow should leave SQLite consistent."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"epochs": 1, "password": "private"}),
        encoding="utf-8",
    )
    script = (
        "from pathlib import Path; import json; "
        "Path('metrics.json').write_text(json.dumps({'accuracy': 0.88}), encoding='utf-8'); "
        "Path('artifact.txt').write_text('artifact-data', encoding='utf-8'); "
        "print('ok')"
    )

    init_result = runner.invoke(app, ["init", "--name", "demo"])
    run_result = runner.invoke(
        app,
        [
            "run",
            "--seed",
            "42",
            "--config",
            "config.json",
            "--metrics",
            "metrics.json",
            "--artifact",
            "artifact.txt",
            "--",
            sys.executable,
            "-c",
            script,
        ],
    )

    assert init_result.exit_code == 0
    assert run_result.exit_code == 0
    assert "success" in run_result.stdout

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        run_row = connection.execute(
            """
            SELECT seed, stdout_path, stderr_path, stdout_preview, config_json,
                   config_hash, metrics_json, artifacts_json
            FROM runs
            """
        ).fetchone()
        experiment_count = connection.execute("SELECT count(*) FROM experiments").fetchone()[0]
        run_count = connection.execute("SELECT count(*) FROM runs").fetchone()[0]
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()

    assert experiment_count == 1
    assert run_count == 1
    assert run_row["seed"] == "42"
    assert run_row["stdout_preview"] == "ok\n"
    assert (tmp_path / run_row["stdout_path"]).read_text(encoding="utf-8") == "ok\n"
    assert (tmp_path / run_row["stderr_path"]).read_text(encoding="utf-8") == ""
    assert json.loads(run_row["config_json"]) == {
        "epochs": 1,
        "password": "***REDACTED***",
    }
    assert run_row["config_hash"] == compute_file_hash(tmp_path / "config.json")
    assert json.loads(run_row["metrics_json"]) == {"accuracy": 0.88}
    assert json.loads(run_row["artifacts_json"]) == [
        {
            "path": "artifact.txt",
            "sha256": compute_file_hash(tmp_path / "artifact.txt"),
            "size_bytes": 13,
        }
    ]
    assert foreign_key_rows == []
    assert integrity == "ok"
