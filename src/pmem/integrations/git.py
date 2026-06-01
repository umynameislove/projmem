"""Best-effort Git metadata capture for run evidence.

Git is an optional local integration. Failures must never fail `pmem run`,
and remote URLs are intentionally not stored because they can contain private
hostnames or credentials.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

GIT_TIMEOUT_SECONDS = 2.0


def collect_git_metadata(cwd: str | Path) -> dict[str, Any]:
    """Return safe Git metadata for `cwd`, or `{}` when Git is unavailable."""

    working_dir = Path(cwd)
    try:
        commit = _git_text(working_dir, "rev-parse", "HEAD")
    except Exception:
        return {}

    if not commit:
        return {}

    branch = _optional_git_text(working_dir, "rev-parse", "--abbrev-ref", "HEAD")
    dirty_output = _optional_git_text(working_dir, "status", "--porcelain")
    remotes = _optional_git_text(working_dir, "remote").splitlines()
    detached = branch == "HEAD"

    return {
        "branch": None if detached else branch or None,
        "commit": commit,
        "detached": detached,
        "dirty": bool(dirty_output.strip()),
        "has_remote": bool([remote for remote in remotes if remote.strip()]),
    }


def _git_text(cwd: Path, *args: str) -> str:
    """Run one local Git command and return stripped stdout."""

    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        check=False,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError("git metadata unavailable")
    return completed.stdout.strip()


def _optional_git_text(cwd: Path, *args: str) -> str:
    """Return optional Git metadata without making run capture fail."""

    try:
        return _git_text(cwd, *args)
    except Exception:
        return ""
