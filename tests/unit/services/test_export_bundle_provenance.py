"""Unit tests for export-bundle provenance export bundle provenance helpers."""

from __future__ import annotations

from pmem.services import export_bundle


def test_provenance_uses_runtime_tool_version(monkeypatch) -> None:
    """Provenance should read the package version instead of hardcoding release text."""

    monkeypatch.setattr(export_bundle, "__version__", "9.9.9-test")

    payload = export_bundle._provenance({"projects": [{"name": "demo"}]})

    assert payload == {
        "tool": "projmem",
        "tool_version": "9.9.9-test",
        "source": "local-export",
        "project_name": "demo",
        "git_commit_hash": None,
        "git_dirty": None,
    }


def test_provenance_handles_missing_project_without_private_path() -> None:
    """Empty project arrays should not force absolute paths or host identifiers."""

    payload = export_bundle._provenance({"projects": []})

    assert payload["tool"] == "projmem"
    assert payload["project_name"] is None
    assert payload["git_commit_hash"] is None
    assert payload["git_dirty"] is None


def test_safe_git_metadata_removes_remote_url_like_keys() -> None:
    """Exported Git metadata should not carry remote URLs into bundles."""

    safe = export_bundle._safe_git_metadata(
        {
            "commit": "abc123",
            "branch": "main",
            "remote_url": "git@example.invalid:private/repo.git",
            "upstreamUrl": "https://example.invalid/private",
        }
    )

    assert safe == {"commit": "abc123", "branch": "main"}
