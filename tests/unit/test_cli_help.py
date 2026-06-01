"""Smoke tests for the CLI tooling CLI shell.

These tests verify that the package exposes a runnable command before real
local-memory behavior is implemented. Deeper command tests should be added alongside
each WBS gate.
"""

from typer.testing import CliRunner

from pmem import __version__
from pmem.cli.app import app

runner = CliRunner()


def test_cli_help_renders() -> None:
    """`pmem --help` should work as soon as CLI tooling tooling exists."""

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Local-first long-horizon project memory" in result.stdout


def test_cli_version_renders() -> None:
    """`pmem --version` should report the installed package version."""

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert f"pmem {__version__}" in result.stdout
