"""Tests for best-effort Git metadata capture."""

import subprocess

import pytest

import pmem.integrations.git as git_module
from pmem.integrations.git import collect_git_metadata


def _create_git_commit(repo_path) -> None:
    """Create a local Git repository with one commit and no remote URL."""

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
    (repo_path / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )


def test_collect_git_metadata_returns_empty_outside_git(tmp_path) -> None:
    """Non-Git directories should not fail run capture."""

    assert collect_git_metadata(tmp_path) == {}


def test_collect_git_metadata_omits_remote_urls(tmp_path) -> None:
    """Git capture should store safe metadata without remote URLs."""

    _create_git_commit(tmp_path)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/projmem.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    metadata = collect_git_metadata(tmp_path)

    assert metadata["branch"]
    assert len(metadata["commit"]) == 40
    assert metadata["detached"] is False
    assert metadata["dirty"] is False
    assert metadata["has_remote"] is True
    assert "https://example.invalid/projmem.git" not in str(metadata)


def test_collect_git_metadata_reports_dirty_worktree(tmp_path) -> None:
    """Dirty flag should be true when tracked project files change."""

    _create_git_commit(tmp_path)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")

    metadata = collect_git_metadata(tmp_path)

    assert metadata["dirty"] is True


def test_collect_git_metadata_marks_detached_head(tmp_path) -> None:
    """Detached HEAD should not be stored as a misleading branch name."""

    _create_git_commit(tmp_path)
    commit = collect_git_metadata(tmp_path)["commit"]
    subprocess.run(
        ["git", "checkout", "--detach", commit], cwd=tmp_path, check=True, capture_output=True
    )

    metadata = collect_git_metadata(tmp_path)

    assert metadata["commit"] == commit
    assert metadata["branch"] is None
    assert metadata["detached"] is True


def test_collect_git_metadata_degrades_when_optional_probe_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional Git probes should not erase the required commit evidence."""

    _create_git_commit(tmp_path)
    original_git_text = git_module._git_text

    def flaky_git_text(cwd, *args: str) -> str:
        if args in {("status", "--porcelain"), ("remote",)}:
            raise RuntimeError("optional git metadata unavailable")
        return original_git_text(cwd, *args)

    monkeypatch.setattr(git_module, "_git_text", flaky_git_text)

    metadata = collect_git_metadata(tmp_path)

    assert len(metadata["commit"]) == 40
    assert metadata["dirty"] is False
    assert metadata["has_remote"] is False
