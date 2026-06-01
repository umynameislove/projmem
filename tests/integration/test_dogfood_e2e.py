"""dogfood end-to-end dogfood smoke tests for `pmem run`."""

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database
from pmem.utils.hashing import compute_file_hash

runner = CliRunner()

DOGFOOD_SCRIPT = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "dogfood" / "ag_news_smoke.py"
)


def test_ag_news_dogfood_smoke_captures_reproducible_run_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    """dogfood should prove the CLI can capture dogfood metrics and artifacts."""

    _copy_dogfood_script(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"dataset": "ag_news_smoke", "split": "synthetic-smoke"}, sort_keys=True),
        encoding="utf-8",
    )
    _create_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    init_result = runner.invoke(app, ["init", "--name", "ag-news-dogfood"])
    run_result = runner.invoke(
        app,
        [
            "run",
            "--name",
            "ag-news-smoke",
            "--seed",
            "7",
            "--config",
            "config.json",
            "--metrics",
            "metrics.json",
            "--artifact",
            "dogfood_report.txt",
            "--",
            sys.executable,
            "ag_news_smoke.py",
            "--metrics",
            "metrics.json",
            "--artifact",
            "dogfood_report.txt",
            "--seed",
            "7",
        ],
    )

    assert init_result.exit_code == 0
    assert run_result.exit_code == 0
    assert "Traceback" not in run_result.output

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        row = connection.execute(
            """
            SELECT status, seed, config_hash, metrics_json, artifacts_json, git_json
            FROM runs
            """
        ).fetchone()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()

    metrics = json.loads(row["metrics_json"])
    artifacts = json.loads(row["artifacts_json"])
    git_json = json.loads(row["git_json"])

    assert row["status"] == "success"
    assert row["seed"] == "7"
    assert len(row["config_hash"]) == 64
    assert metrics["dataset_subset_size"] == 4
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert artifacts == [
        {
            "path": "dogfood_report.txt",
            "sha256": compute_file_hash(tmp_path / "dogfood_report.txt"),
            "size_bytes": (tmp_path / "dogfood_report.txt").stat().st_size,
        }
    ]
    assert len(git_json["commit"]) == 40
    assert git_json["detached"] is False
    assert integrity == "ok"


def _copy_dogfood_script(project_path: Path) -> None:
    """Copy the dogfood script so captured command paths stay project-relative."""

    (project_path / "ag_news_smoke.py").write_text(
        DOGFOOD_SCRIPT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _create_git_repo(repo_path: Path) -> None:
    """Create a local Git repo while ignoring generated dogfood outputs."""

    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "projmem test"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    (repo_path / ".gitignore").write_text(
        "\n".join([".pmem/", "metrics.json", "dogfood_report.txt", ""]),
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", ".gitignore", "ag_news_smoke.py", "config.json"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "dogfood smoke"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
