"""Git metadata CLI integration tests for safe Git metadata capture."""

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database

runner = CliRunner()


def test_cli_run_in_non_git_project_stores_empty_git_metadata(monkeypatch, tmp_path) -> None:
    """Non-Git projects should run successfully and store `{}` for git_json."""

    monkeypatch.chdir(tmp_path)

    init_result = runner.invoke(app, ["init", "--name", "demo"])
    run_result = runner.invoke(app, ["run", "--", sys.executable, "-c", "print('ok')"])

    assert init_result.exit_code == 0
    assert run_result.exit_code == 0
    assert "Traceback" not in run_result.output

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        row = connection.execute("SELECT git_json FROM runs").fetchone()
    finally:
        connection.close()

    assert json.loads(row["git_json"]) == {}


def test_cli_run_in_git_project_stores_safe_git_metadata(monkeypatch, tmp_path) -> None:
    """Git metadata should include commit evidence without storing remote URLs."""

    _create_git_repo(tmp_path)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/private.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    monkeypatch.chdir(tmp_path)

    init_result = runner.invoke(app, ["init", "--name", "demo"])
    run_result = runner.invoke(app, ["run", "--", sys.executable, "-c", "print('ok')"])

    assert init_result.exit_code == 0
    assert run_result.exit_code == 0

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        row = connection.execute("SELECT git_json FROM runs").fetchone()
    finally:
        connection.close()

    git_json = json.loads(row["git_json"])
    assert git_json["branch"]
    assert len(git_json["commit"]) == 40
    assert git_json["detached"] is False
    assert git_json["dirty"] is False
    assert git_json["has_remote"] is True
    assert "https://example.invalid/private.git" not in str(git_json)


def _create_git_repo(repo_path: Path) -> None:
    """Create a deterministic local Git repository for CLI smoke tests."""

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
    (repo_path / ".gitignore").write_text(".pmem/\n", encoding="utf-8")
    (repo_path / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore", "README.md"], cwd=repo_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
