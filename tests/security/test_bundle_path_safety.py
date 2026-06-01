"""portability security path-safety regression tests for bundle and artifact boundaries."""

from __future__ import annotations

import os

import pytest

from pmem.errors import PmemNotFoundError, PmemSecurityError
from pmem.services.export_bundle import _resolve_artifact_path
from pmem.services.import_dry_run import _resolve_bundle_path


def test_import_bundle_path_rejects_absolute_traversal_pmem_symlink_and_fifo(tmp_path) -> None:
    """Bundle import paths must be project-relative regular files outside .pmem."""

    project = tmp_path / "project"
    project.mkdir()
    bundle = project / "bundle.json"
    bundle.write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (project / "bundle-link.json").symlink_to(bundle)

    assert _resolve_bundle_path(project, "bundle.json") == bundle
    with pytest.raises(PmemSecurityError):
        _resolve_bundle_path(project, str(bundle))
    with pytest.raises(PmemSecurityError):
        _resolve_bundle_path(project, "../outside.json")
    with pytest.raises(PmemSecurityError):
        _resolve_bundle_path(project, ".PmEm/bundle.json")
    with pytest.raises(PmemSecurityError):
        _resolve_bundle_path(project, "bundle-link.json")
    with pytest.raises(PmemNotFoundError):
        _resolve_bundle_path(project, "missing.json")

    fifo = project / "bundle.fifo"
    os.mkfifo(fifo)
    with pytest.raises(PmemSecurityError):
        _resolve_bundle_path(project, "bundle.fifo")


def test_import_bundle_path_rejects_null_byte_and_control_characters(tmp_path) -> None:
    """Null bytes and control characters in bundle path must raise PmemSecurityError."""

    project = tmp_path / "project"
    project.mkdir()

    for unsafe in (
        "bundle\x00.json",
        "bun\x00dle.json",
        "\x00bundle.json",
        "bundle\x01.json",
        "bundle\x1f.json",
    ):
        with pytest.raises(PmemSecurityError, match="unsafe control characters"):
            _resolve_bundle_path(project, unsafe)


def test_artifact_path_resolution_rejects_private_or_non_regular_sources(tmp_path) -> None:
    """Opt-in artifact byte inclusion must not follow unsafe metadata paths."""

    (tmp_path / "artifact.txt").write_text("artifact", encoding="utf-8")
    (tmp_path / "artifact-link.txt").symlink_to(tmp_path / "artifact.txt")
    os.mkfifo(tmp_path / "artifact.fifo")

    assert _resolve_artifact_path(tmp_path, "artifact.txt") == tmp_path / "artifact.txt"
    for unsafe_path in (
        "/tmp/private.txt",
        "C:/private.txt",
        "../private.txt",
        "nested/../../private.txt",
        ".pMeM/pmem.db",
        "artifact-link.txt",
        "artifact.fifo",
    ):
        with pytest.raises(PmemSecurityError):
            _resolve_artifact_path(tmp_path, unsafe_path)


def test_artifact_path_rejects_null_byte_and_control_characters(tmp_path) -> None:
    """Null bytes and control characters in artifact path must raise PmemSecurityError."""

    for unsafe in (
        "artifact\x00.txt",
        "arti\x00fact.txt",
        "\x00artifact.txt",
        "artifact\x01.txt",
        "artifact\x1f.txt",
    ):
        with pytest.raises(PmemSecurityError, match="unsafe control characters"):
            _resolve_artifact_path(tmp_path, unsafe)
