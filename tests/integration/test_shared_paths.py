"""shared-path shared memory path foundation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.errors import PmemSecurityError
from pmem.repositories.sqlite import connect_database
from pmem.services.shared_paths import register_shared_path

runner = CliRunner()


def test_share_init_registers_local_directory_without_path_leak(monkeypatch, tmp_path) -> None:
    """shared-path registration should store a path but keep CLI output privacy-preserving."""

    project = tmp_path / "project"
    project.mkdir()
    shared = tmp_path / "team-shared"
    shared.mkdir()
    monkeypatch.chdir(project)
    assert runner.invoke(app, ["init", "--name", "share-demo"]).exit_code == 0

    result = runner.invoke(
        app,
        ["share", "init", str(shared), "--alias", "team", "--json"],
    )
    payload = json.loads(result.stdout)
    rows = _shared_path_rows(project)

    assert result.exit_code == 0
    assert payload["ok"] is True
    assert payload["alias"] == "team"
    assert payload["status"] == "ok"
    assert str(tmp_path) not in result.stdout
    assert payload["path_display"] == "<external:team-shared>"
    assert len(rows) == 1
    assert rows[0]["alias"] == "team"
    assert rows[0]["path"] == shared.resolve().as_posix()


def test_share_status_validates_registered_paths_and_updates_timestamp(
    monkeypatch, tmp_path
) -> None:
    """Status should be local-only validation, not background sync."""

    monkeypatch.chdir(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    assert runner.invoke(app, ["init", "--name", "share-demo"]).exit_code == 0
    assert runner.invoke(app, ["share", "init", "shared", "--alias", "local"]).exit_code == 0

    result = runner.invoke(app, ["share", "status", "--json"])
    payload = json.loads(result.stdout)
    rows = _shared_path_rows(tmp_path)

    assert result.exit_code == 0
    assert payload["ok"] is True
    assert payload["database_mutation"] == "shared_paths_last_checked_update"
    assert payload["shared_paths"][0]["alias"] == "local"
    assert payload["shared_paths"][0]["path_display"] == "shared"
    assert rows[0]["last_checked_at"] is not None
    assert "Sync" not in result.stdout


def test_share_text_outputs_include_no_sync_boundary(monkeypatch, tmp_path) -> None:
    """Text output should not imply sync or leak external absolute paths."""

    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external-share"
    external.mkdir()
    monkeypatch.chdir(project)
    assert runner.invoke(app, ["init", "--name", "share-demo"]).exit_code == 0

    empty_status = runner.invoke(app, ["share", "status"])
    init_result = runner.invoke(app, ["share", "init", str(external), "--alias", "team"])
    status_result = runner.invoke(app, ["share", "status"])

    assert empty_status.exit_code == 0
    assert "- none" in empty_status.stdout
    assert init_result.exit_code == 0
    assert "Shared path registered." in init_result.stdout
    assert "Sync: none" in init_result.stdout
    assert str(tmp_path) not in init_result.stdout
    assert status_result.exit_code == 0
    assert "team: ok" in status_result.stdout
    assert str(tmp_path) not in status_result.stdout


def test_share_init_rejects_unsafe_paths(monkeypatch, tmp_path) -> None:
    """Traversal, .pmem, files, symlinks, and missing directories are unsafe."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "share-demo"]).exit_code == 0
    (tmp_path / "file.txt").write_text("not a dir", encoding="utf-8")
    (tmp_path / "safe").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "safe", target_is_directory=True)

    cases = [
        "../outside",
        ".PMEM",
        "missing",
        "file.txt",
        "link",
        "bad\\path",
    ]
    for case in cases:
        result = runner.invoke(app, ["share", "init", case, "--alias", f"a{len(case)}"])
        assert result.exit_code == 1
        assert "Traceback" not in result.stdout


def test_share_init_rejects_bad_alias_and_mode(monkeypatch, tmp_path) -> None:
    """Alias and mode are part of the shared path trust boundary."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "shared").mkdir()
    assert runner.invoke(app, ["init", "--name", "share-demo"]).exit_code == 0

    blank_alias = runner.invoke(app, ["share", "init", "shared", "--alias", " "])
    slash_alias = runner.invoke(app, ["share", "init", "shared", "--alias", "bad/name"])
    invalid_alias = runner.invoke(app, ["share", "init", "shared", "--alias", "bad name"])
    invalid_mode = runner.invoke(
        app, ["share", "init", "shared", "--alias", "team", "--mode", "sync"]
    )

    assert blank_alias.exit_code == 1
    assert "alias cannot be blank" in blank_alias.stdout
    assert slash_alias.exit_code == 1
    assert "alias contains unsafe" in slash_alias.stdout
    assert invalid_alias.exit_code == 1
    assert "alias must use letters" in invalid_alias.stdout
    assert invalid_mode.exit_code == 1
    assert "mode must be read" in invalid_mode.stdout


def test_share_init_rejects_duplicate_alias_and_path(monkeypatch, tmp_path) -> None:
    """shared-path should keep shared path registration deterministic."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "shared").mkdir()
    assert runner.invoke(app, ["init", "--name", "share-demo"]).exit_code == 0
    assert runner.invoke(app, ["share", "init", "shared", "--alias", "team"]).exit_code == 0

    duplicate_alias = runner.invoke(app, ["share", "init", "shared", "--alias", "team"])
    duplicate_path = runner.invoke(app, ["share", "init", "shared", "--alias", "team2"])

    assert duplicate_alias.exit_code == 1
    assert "alias already exists" in duplicate_alias.stdout
    assert duplicate_path.exit_code == 1
    assert "already registered" in duplicate_path.stdout
    assert len(_shared_path_rows(tmp_path)) == 1


def test_shared_path_service_rejects_symlink_parts(monkeypatch, tmp_path) -> None:
    """Service-level safety should reject symlink ancestors too."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "share-demo"]).exit_code == 0
    real = tmp_path / "real"
    real.mkdir()
    (real / "child").mkdir()
    link = tmp_path / "link-parent"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(PmemSecurityError):
        register_shared_path(tmp_path, "link-parent/child", alias="linked")


def test_share_help_is_available() -> None:
    """shared-path commands should appear in CLI help."""

    root_help = runner.invoke(app, ["--help"])
    share_help = runner.invoke(app, ["share", "--help"])
    init_help = runner.invoke(app, ["share", "init", "--help"])

    assert root_help.exit_code == 0
    assert "share" in root_help.stdout
    assert share_help.exit_code == 0
    assert "init" in share_help.stdout
    assert init_help.exit_code == 0
    assert init_help.stdout.strip()
    assert "Register a local shared memory path" in init_help.stdout


def _shared_path_rows(project_root: Path) -> list[dict[str, object]]:
    connection = connect_database(project_root / ".pmem" / "pmem.db")
    try:
        rows = connection.execute(
            """
            SELECT alias, path, mode, last_checked_at
            FROM shared_paths
            ORDER BY alias
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
