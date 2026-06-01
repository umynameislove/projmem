"""CLI tests for `pmem init`."""

from typer.testing import CliRunner

from pmem.cli.app import app

runner = CliRunner()


def test_cli_init_success_path(monkeypatch, tmp_path) -> None:
    """`pmem init` should create local project state and print safe output."""

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", "--name", "demo"])

    assert result.exit_code == 0
    assert "Initialized projmem at .pmem/" in result.stdout
    assert "Database ready." in result.stdout
    assert (tmp_path / ".pmem" / "pmem.db").is_file()
    assert (tmp_path / ".pmem" / "config.yaml").is_file()
    assert (tmp_path / ".pmem" / "artifacts").is_dir()
    assert (tmp_path / ".pmem" / "snapshots").is_dir()


def test_cli_init_idempotent_path(monkeypatch, tmp_path) -> None:
    """Running `pmem init` twice should report existing state without duplicate rows."""

    monkeypatch.chdir(tmp_path)

    first = runner.invoke(app, ["init", "--name", "demo"])
    second = runner.invoke(app, ["init"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "projmem is already initialized." in second.stdout
    assert "Database ready." in second.stdout


def test_cli_init_accepts_project_context_flags(monkeypatch, tmp_path) -> None:
    """CLI should expose the project init init context flags from the audit."""

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "init",
            "--name",
            "demo",
            "--objective",
            "Train baseline",
            "--metric",
            "accuracy",
            "--metric-direction",
            "max",
            "--target",
            "0.9",
        ],
    )

    assert result.exit_code == 0
    assert "Initialized projmem at .pmem/" in result.stdout


def test_cli_init_error_output_is_clean(monkeypatch, tmp_path) -> None:
    """CLI errors should not expose tracebacks, SQL, or local absolute paths."""

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", "--name", "bad\nname"])

    assert result.exit_code == 1
    assert "Error: Project name contains unsupported control characters." in result.stdout
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()
    assert str(tmp_path) not in result.stdout
