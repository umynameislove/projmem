"""FastAPI adapter FastAPI CLI integration tests."""

from __future__ import annotations

from click import Group
from typer.main import get_command
from typer.testing import CliRunner

from pmem import server
from pmem.cli.app import app

runner = CliRunner()


def test_serve_help_documents_localhost_first_options() -> None:
    """FastAPI adapter serve options should be discoverable from CLI help."""

    help_env = {"COLUMNS": "160"}
    root_help = runner.invoke(app, ["--help"], env=help_env)
    serve_help = runner.invoke(app, ["serve", "--help"], env=help_env)
    root_command = get_command(app)
    assert isinstance(root_command, Group)
    serve_command = root_command.commands["serve"]
    option_names = {option for param in serve_command.params for option in param.opts}

    assert root_help.exit_code == 0
    assert "serve" in root_help.stdout
    assert serve_help.exit_code == 0
    assert {"--host", "--port", "--confirm-non-local-bind"} <= option_names
    assert "localhost-first" in serve_help.stdout


def test_serve_cli_uses_loopback_default(monkeypatch, tmp_path) -> None:
    """FastAPI adapter CLI should pass the safe default bind to its server adapter."""

    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(project_root, *, host, port, confirm_non_local_bind):
        captured.update(
            project_root=project_root,
            host=host,
            port=port,
            confirm_non_local_bind=confirm_non_local_bind,
        )

    monkeypatch.setattr(server, "run_api_server", fake_run)

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["confirm_non_local_bind"] is False


def test_serve_cli_rejects_non_local_bind_without_confirmation(monkeypatch, tmp_path) -> None:
    """FastAPI adapter CLI must fail closed for accidental network exposure."""

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])

    assert result.exit_code == 1
    assert "confirm-non-local-bind" in result.stdout
    assert "Traceback" not in result.stdout
