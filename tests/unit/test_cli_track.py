"""CLI tests for `pmem track`."""

from typer.testing import CliRunner

from pmem.cli.app import app

runner = CliRunner()


def test_cli_track_success_path(monkeypatch, tmp_path) -> None:
    """`pmem track <path>` should track a file after init."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")

    init_result = runner.invoke(app, ["init", "--name", "demo"])
    track_result = runner.invoke(app, ["track", "README.md"])

    assert init_result.exit_code == 0
    assert track_result.exit_code == 0
    assert "Tracked README.md" in track_result.stdout
    assert "sha256:" in track_result.stdout


def test_cli_track_requires_init(monkeypatch, tmp_path) -> None:
    """Tracking before init should fail with a clear next action."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")

    result = runner.invoke(app, ["track", "README.md"])

    assert result.exit_code == 1
    assert "projmem is not initialized. Run `pmem init` first." in result.stdout
    assert "Traceback" not in result.stdout


def test_cli_track_duplicate_output(monkeypatch, tmp_path) -> None:
    """Duplicate tracking should report existing state without creating a new row."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    runner.invoke(app, ["init", "--name", "demo"])
    first = runner.invoke(app, ["track", "README.md"])
    second = runner.invoke(app, ["track", "README.md"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "README.md is already tracked." in second.stdout
    assert "sha256:" in second.stdout


def test_cli_track_update_refreshes_hash(monkeypatch, tmp_path) -> None:
    """`pmem track --update` should report a refreshed tracked file."""

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "README.md"
    target.write_text("first\n", encoding="utf-8")
    runner.invoke(app, ["init", "--name", "demo"])
    first = runner.invoke(app, ["track", "README.md"])
    target.write_text("changed\n", encoding="utf-8")
    second = runner.invoke(app, ["track", "README.md", "--update"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "Updated README.md" in second.stdout
    assert first.stdout.split("sha256: ")[1].strip() != second.stdout.split("sha256: ")[1].strip()


def test_cli_track_rejects_missing_path_and_pmem(monkeypatch, tmp_path) -> None:
    """CLI should render safe errors for missing files and internal files."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "demo"])

    missing = runner.invoke(app, ["track", "missing.txt"])
    internal = runner.invoke(app, ["track", ".pmem/pmem.db"])
    internal_case_variant = runner.invoke(app, ["track", ".PMEM/pmem.db"])

    assert missing.exit_code == 1
    assert "Tracked path does not exist." in missing.stdout
    assert internal.exit_code == 1
    assert "projmem internal files cannot be tracked." in internal.stdout
    assert internal_case_variant.exit_code == 1
    assert "projmem internal files cannot be tracked." in internal_case_variant.stdout
    assert "Traceback" not in missing.stdout
    assert "sqlite" not in internal.stdout.lower()
