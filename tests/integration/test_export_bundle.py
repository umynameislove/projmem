"""export bundle export-bundle workflow tests."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.errors import PmemNotFoundError, PmemSecurityError, PmemValidationError
from pmem.repositories.sqlite import connect_database
from pmem.services.export_bundle import (
    _artifact_index,
    _display_path,
    _prepare_entities,
    _privacy_flags,
    _resolve_artifact_path,
    _resolve_output_path,
)

runner = CliRunner()


def test_export_bundle_is_deterministic_and_dry_run_accepts_it(monkeypatch, tmp_path) -> None:
    """Two frozen exports from the same DB should be byte-identical and valid."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)

    first = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "bundle-a.json",
            "--freeze-timestamp",
            "2026-01-01T00:00:00Z",
            "--json",
        ],
    )
    second = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "bundle-b.json",
            "--freeze-timestamp",
            "2026-01-01T00:00:00Z",
            "--json",
        ],
    )
    dry_run = runner.invoke(app, ["import", "--dry-run", "bundle-a.json", "--json"])
    bundle = json.loads((tmp_path / "bundle-a.json").read_text(encoding="utf-8"))

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert (tmp_path / "bundle-a.json").read_bytes() == (tmp_path / "bundle-b.json").read_bytes()
    assert dry_run.exit_code == 0
    assert json.loads(dry_run.stdout)["ok"] is True
    assert bundle["manifest"]["payload_hash"] == _payload_hash(bundle)
    assert bundle["manifest"]["manifest_hash"] == _manifest_hash(bundle["manifest"])
    assert bundle["manifest"]["freeze_timestamp"] is True
    assert bundle["artifact_index"]
    assert all("content_base64" not in item for item in bundle["artifact_index"])
    assert _export_package_count(tmp_path) == 1


def test_export_bundle_redacts_free_text_and_omits_git_remote(monkeypatch, tmp_path) -> None:
    """Redaction should happen before hashing and Git remote URLs must not appear."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)

    result = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "redacted.json",
            "--redact-fields",
            "notes.content,decisions.rationale",
            "--freeze-timestamp",
            "2026-01-01T00:00:00Z",
            "--json",
        ],
    )
    payload = json.loads(result.stdout)
    bundle_text = (tmp_path / "redacted.json").read_text(encoding="utf-8")
    bundle = json.loads(bundle_text)

    assert result.exit_code == 0
    assert payload["ok"] is True
    assert bundle["entities"]["notes"][0]["content"] == "[REDACTED]"
    assert bundle["entities"]["decisions"][0]["rationale"] == "[REDACTED]"
    assert any(flag["code"] == "redacted_field" for flag in bundle["privacy_flags"])
    assert "remote_url" not in bundle_text
    assert "github.com" not in bundle_text


def test_export_bundle_artifact_bytes_are_opt_in(monkeypatch, tmp_path) -> None:
    """Artifact metadata is default; base64 content appears only with --include-artifacts."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)

    default_result = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "metadata-only.json",
            "--freeze-timestamp",
            "2026-01-01T00:00:00Z",
        ],
    )
    include_result = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "with-artifacts.json",
            "--include-artifacts",
            "--freeze-timestamp",
            "2026-01-01T00:00:00Z",
        ],
    )
    metadata_only = json.loads((tmp_path / "metadata-only.json").read_text(encoding="utf-8"))
    with_artifacts = json.loads((tmp_path / "with-artifacts.json").read_text(encoding="utf-8"))
    dry_run = runner.invoke(app, ["import", "--dry-run", "with-artifacts.json", "--json"])

    assert default_result.exit_code == 0
    assert include_result.exit_code == 0
    assert "content_base64" not in metadata_only["artifact_index"][0]
    assert with_artifacts["artifact_index"][0]["content_encoding"] == "base64"
    assert with_artifacts["artifact_index"][0]["content_base64"]
    assert dry_run.exit_code == 0
    assert json.loads(dry_run.stdout)["ok"] is True


def test_export_bundle_rejects_unknown_redaction_field(monkeypatch, tmp_path) -> None:
    """Unknown redact fields should fail before writing a bundle."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)

    result = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "bad.json",
            "--redact-fields",
            "projects.name",
        ],
    )

    assert result.exit_code == 1
    assert "Unsupported redact field" in result.stdout
    assert not (tmp_path / "bad.json").exists()
    assert "Traceback" not in result.stdout


def test_export_bundle_rejects_scope_and_bad_freeze_timestamp(monkeypatch, tmp_path) -> None:
    """export bundle should fail safely for unsupported scope and invalid freeze timestamps."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)

    unsupported_scope = runner.invoke(
        app,
        ["export-bundle", "--out", "scope.json", "--scope", "run"],
    )
    blank_timestamp = runner.invoke(
        app,
        ["export-bundle", "--out", "blank.json", "--freeze-timestamp", ""],
    )
    invalid_timestamp = runner.invoke(
        app,
        ["export-bundle", "--out", "invalid.json", "--freeze-timestamp", "not-a-date"],
    )
    missing_timezone = runner.invoke(
        app,
        ["export-bundle", "--out", "missing-tz.json", "--freeze-timestamp", "2026-01-01T00:00:00"],
    )

    assert unsupported_scope.exit_code == 1
    assert "Only project scope" in unsupported_scope.stdout
    assert blank_timestamp.exit_code == 1
    assert "cannot be blank" in blank_timestamp.stdout
    assert invalid_timestamp.exit_code == 1
    assert "ISO-8601 UTC" in invalid_timestamp.stdout
    assert missing_timezone.exit_code == 1
    assert "UTC timezone" in missing_timezone.stdout
    for output in (
        unsupported_scope.stdout,
        blank_timestamp.stdout,
        invalid_timestamp.stdout,
        missing_timezone.stdout,
    ):
        assert "Traceback" not in output


def test_export_bundle_rejects_unsafe_output_paths(monkeypatch, tmp_path) -> None:
    """Bundle output should not be written inside .pmem, directories, or symlinks."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    (tmp_path / "bundle-dir").mkdir()
    (tmp_path / "real-bundle.json").write_text("{}", encoding="utf-8")
    (tmp_path / "linked-bundle.json").symlink_to(tmp_path / "real-bundle.json")

    inside_pmem = runner.invoke(app, ["export-bundle", "--out", ".PMEM/leak.json"])
    directory = runner.invoke(app, ["export-bundle", "--out", "bundle-dir"])
    symlink = runner.invoke(app, ["export-bundle", "--out", "linked-bundle.json"])

    assert inside_pmem.exit_code == 1
    assert "cannot point inside .pmem" in inside_pmem.stdout
    assert directory.exit_code == 1
    assert "must point to a file" in directory.stdout
    assert symlink.exit_code == 1
    assert "cannot be a symlink" in symlink.stdout
    assert "Traceback" not in inside_pmem.stdout
    assert "Traceback" not in directory.stdout
    assert "Traceback" not in symlink.stdout


def test_export_bundle_rejects_unsafe_artifact_paths(tmp_path) -> None:
    """Artifact byte inclusion should reject unsafe metadata paths."""

    (tmp_path / "real.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "link-inside.txt").symlink_to(tmp_path / "real.txt")
    outside = tmp_path.parent / "outside-artifact.txt"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "link-outside.txt").symlink_to(outside)

    for unsafe_path in ("/tmp/secret.txt", "C:/secret.txt", "../secret.txt", ".PMEM/pmem.db"):
        with pytest.raises(PmemSecurityError):
            _resolve_artifact_path(tmp_path, unsafe_path)

    with pytest.raises(PmemNotFoundError):
        _resolve_artifact_path(tmp_path, "missing.txt")
    with pytest.raises(PmemSecurityError):
        _resolve_artifact_path(tmp_path, "link-inside.txt")
    with pytest.raises(PmemSecurityError):
        _resolve_artifact_path(tmp_path, "link-outside.txt")


def test_export_bundle_normalizes_runs_without_git_dict() -> None:
    """Runs with malformed Git metadata should not leak raw values into bundles."""

    entities = {
        "projects": [],
        "experiments": [],
        "runs": [{"run_id": "run_a", "git": "not-a-dict"}],
        "failures": [],
        "decisions": [],
        "notes": [],
        "tracked_paths": [],
    }

    prepared = _prepare_entities(entities)

    assert prepared["runs"][0]["git"] == {}
    assert prepared["runs"][0]["git_commit_hash"] is None


def test_export_bundle_ignores_malformed_artifact_metadata(tmp_path) -> None:
    """Malformed run artifact metadata should not crash bundle construction."""

    index = _artifact_index(
        tmp_path,
        [
            {"run_id": "run_a", "artifacts": "not-a-list"},
            {"run_id": "run_b", "artifacts": [None, {"path": "model.txt"}]},
            {"run_id": "run_c", "artifacts": [{"path": "model.txt", "sha256": 123}]},
        ],
        include_artifacts=False,
    )

    assert index == []


def test_export_bundle_helpers_handle_empty_privacy_and_display_path(tmp_path) -> None:
    """Small helper branches should stay deterministic and safe."""

    flags = _privacy_flags(
        {
            "runs": [],
            "failures": [],
            "decisions": [],
            "notes": [],
        },
        [],
        {},
    )

    assert flags == []
    assert _display_path(tmp_path, tmp_path.parent / "bundle.json") == "bundle.json"


def test_export_bundle_rejects_blank_output_path(tmp_path) -> None:
    """Blank output path should fail before writing a bundle."""

    with pytest.raises(PmemValidationError) as exc_info:
        _resolve_output_path(tmp_path, "")

    assert "cannot be blank" in str(exc_info.value)


def test_export_bundle_help_is_available() -> None:
    """export bundle CLI help should render cleanly across terminal widths."""

    result = runner.invoke(app, ["export-bundle", "--help"])

    assert result.exit_code == 0
    assert result.stdout.strip()
    assert "Traceback" not in result.stdout


def _seed_project(tmp_path: Path) -> str:
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    init_result = runner.invoke(app, ["init", "--name", "bundle-demo"])
    track_result = runner.invoke(app, ["track", "README.md"])
    run_id = _run_with_artifact(tmp_path)
    failure_result = runner.invoke(
        app,
        [
            "log-failure",
            run_id,
            "MetricRegression",
            "Accuracy dropped once.",
            "--root-cause",
            "Bad split.",
            "--lesson",
            "Pin seed.",
            "--tag",
            "reproducibility",
        ],
    )
    decision_result = runner.invoke(
        app,
        ["log-decision", "Keep baseline.", "--rationale", "It is reproducible."],
    )
    note_result = runner.invoke(app, ["note", "Review bundle.", "--run-id", run_id])

    assert init_result.exit_code == 0
    assert track_result.exit_code == 0
    assert failure_result.exit_code == 0
    assert decision_result.exit_code == 0
    assert note_result.exit_code == 0
    return run_id


def _run_with_artifact(tmp_path: Path) -> str:
    script = (
        "from pathlib import Path; import json; "
        "Path('metrics.json').write_text(json.dumps({'accuracy': 0.91}), encoding='utf-8'); "
        "Path('model.txt').write_text('weights', encoding='utf-8'); "
        "print('ok')"
    )
    result = runner.invoke(
        app,
        [
            "run",
            "--metrics",
            "metrics.json",
            "--artifact",
            "model.txt",
            "--",
            sys.executable,
            "-c",
            script,
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "model.txt").is_file()
    return result.stdout.split()[1]


def _manifest_hash(manifest: dict[str, Any]) -> str:
    candidate = copy.deepcopy(manifest)
    candidate["manifest_hash"] = None
    return _sha256_tag(candidate)


def _payload_hash(bundle: dict[str, Any]) -> str:
    candidate = copy.deepcopy(bundle)
    candidate["manifest"]["manifest_hash"] = None
    candidate["manifest"]["payload_hash"] = None
    return _sha256_tag(candidate)


def _sha256_tag(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _export_package_count(project_root: Path) -> int:
    connection = connect_database(project_root / ".pmem" / "pmem.db")
    try:
        return int(connection.execute("SELECT count(*) FROM export_packages").fetchone()[0])
    finally:
        connection.close()
