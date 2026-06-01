"""CLI tests for `pmem run`."""

import json
import sys

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database

runner = CliRunner()


def test_cli_run_success_path(monkeypatch, tmp_path) -> None:
    """`pmem run -- ...` should capture a successful command."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "demo"])

    result = runner.invoke(
        app, ["run", "--name", "baseline", "--", sys.executable, "-c", "print('ok')"]
    )

    assert result.exit_code == 0
    assert "Run run_" in result.stdout
    assert "success" in result.stdout
    assert "exit_code: 0" in result.stdout
    assert "stdout: .pmem/artifacts/runs/" in result.stdout
    assert "ok" not in result.stdout


def test_cli_run_requires_init(monkeypatch, tmp_path) -> None:
    """Running before init should fail with the project init-approved message."""

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run", "--", sys.executable, "-c", "print('ok')"])

    assert result.exit_code == 1
    assert "projmem is not initialized. Run `pmem init` first." in result.stdout
    assert "Traceback" not in result.stdout


def test_cli_run_failed_command_is_recorded(monkeypatch, tmp_path) -> None:
    """A nonzero child command should still produce a captured run."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "demo"])

    result = runner.invoke(app, ["run", "--", sys.executable, "-c", "import sys; sys.exit(2)"])

    assert result.exit_code == 0
    assert "failed" in result.stdout
    assert "exit_code: 2" in result.stdout


def test_cli_run_empty_command_error_is_clean(monkeypatch, tmp_path) -> None:
    """Missing command argv should not expose Typer or subprocess internals."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "demo"])

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    assert "Run command cannot be empty." in result.stdout
    assert "Traceback" not in result.stdout


def test_cli_run_rejects_pmem_metadata_path(monkeypatch, tmp_path) -> None:
    """CLI should render safe errors for internal metadata paths."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "demo"])

    result = runner.invoke(
        app,
        ["run", "--config", ".pmem/config.yaml", "--", sys.executable, "-c", "print('ok')"],
    )

    assert result.exit_code == 1
    assert "projmem internal files cannot be used as run metadata." in result.stdout
    assert "sqlite" not in result.stdout.lower()


def test_cli_run_accepts_dataset_metadata(monkeypatch, tmp_path) -> None:
    """`pmem run` should expose dataset metadata for dataset-failure screening."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "demo"])

    result = runner.invoke(
        app,
        [
            "run",
            "--dataset-id",
            "fashion_mnist_full",
            "--dataset-version",
            "v1",
            "--",
            sys.executable,
            "-c",
            "print('ok')",
        ],
    )

    assert result.exit_code == 0
    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        row = connection.execute("SELECT artifacts_json FROM runs").fetchone()
    finally:
        connection.close()
    assert json.loads(row["artifacts_json"]) == [
        {
            "dataset_id": "fashion_mnist_full",
            "metadata_kind": "dataset",
            "version": "v1",
        }
    ]
