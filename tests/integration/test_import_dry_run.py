"""import dry-run import dry-run validation workflow tests."""

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
from pmem.errors import PmemValidationError
from pmem.repositories.sqlite import connect_database
from pmem.services.import_dry_run import dry_run_import_bundle
from pmem.services.project_export import export_project

runner = CliRunner()


def test_import_dry_run_accepts_valid_bundle_without_mutating_db(monkeypatch, tmp_path) -> None:
    """import dry-run dry-run should validate a bundle and leave SQLite rows unchanged."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle_path = _write_bundle(tmp_path, _bundle_from_project(tmp_path))
    before = _row_counts(tmp_path)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json"])
    after = _row_counts(tmp_path)

    assert result.exit_code == 0
    assert before == after
    assert "Import dry-run: PASS" in result.stdout
    assert "Database mutation: none" in result.stdout
    assert "PRIVACY REVIEW:" in result.stdout
    assert "Conflicts:" in result.stdout
    assert "Traceback" not in result.stdout
    assert bundle_path.name in result.stdout


def test_import_dry_run_json_report_is_machine_readable(monkeypatch, tmp_path) -> None:
    """import dry-run --json output should expose validation, privacy, and conflict previews."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    _write_bundle(tmp_path, _bundle_from_project(tmp_path))

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["database_mutation"] is False
    assert payload["export_format_version"] == "1.0"
    assert payload["schema_version"] == "schema-v1"
    assert payload["entity_counts"]["projects"] == 1
    assert payload["privacy_review"]
    assert payload["conflicts"]


def test_import_dry_run_rejects_tampered_payload_hash(monkeypatch, tmp_path) -> None:
    """import dry-run should catch bundle tampering with an actionable, traceback-free error."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _bundle_from_project(tmp_path)
    bundle["entities"]["projects"][0]["name"] = "tampered"
    _write_bundle(tmp_path, bundle, refresh_hashes=False)
    before = _row_counts(tmp_path)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json"])
    after = _row_counts(tmp_path)

    assert result.exit_code == 1
    assert before == after
    assert "payload_hash_mismatch" in result.stdout
    assert "bundle payload may have been edited" in result.stdout
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()


def test_import_dry_run_reports_missing_fk_without_mutating_db(monkeypatch, tmp_path) -> None:
    """import dry-run should report missing dependencies before any import apply exists."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _bundle_from_project(tmp_path)
    bundle["entities"]["runs"][0]["experiment_id"] = "exp_missing"
    _write_bundle(tmp_path, bundle, refresh_hashes=True)
    before = _row_counts(tmp_path)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json"])
    after = _row_counts(tmp_path)

    assert result.exit_code == 1
    assert before == after
    assert "missing_dependency" in result.stdout
    assert "exp_missing" in result.stdout
    assert "Traceback" not in result.stdout


def test_import_dry_run_rejects_malformed_json_cleanly(monkeypatch, tmp_path) -> None:
    """Malformed JSON should produce a validation report, not a traceback."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "import-demo"])
    (tmp_path / "bundle.json").write_text("{", encoding="utf-8")
    before = _row_counts(tmp_path)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json"])
    after = _row_counts(tmp_path)

    assert result.exit_code == 1
    assert before == after
    assert "invalid_json" in result.stdout
    assert "Traceback" not in result.stdout


def test_import_dry_run_rejects_non_object_bundle(monkeypatch, tmp_path) -> None:
    """Bundle root must be an object, not a raw array or scalar."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "import-demo"])
    (tmp_path / "bundle.json").write_text("[]", encoding="utf-8")

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json"])

    assert result.exit_code == 1
    assert "invalid_bundle_shape" in result.stdout
    assert "Traceback" not in result.stdout


def test_import_dry_run_reports_missing_top_level_keys(monkeypatch, tmp_path) -> None:
    """Missing top-level keys should be reported as validation errors."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "import-demo"])
    (tmp_path / "bundle.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json"])

    assert result.exit_code == 1
    assert "missing_top_level_key" in result.stdout
    assert "invalid_manifest" in result.stdout
    assert "invalid_entities" in result.stdout
    assert "Traceback" not in result.stdout


def test_import_dry_run_reports_bundle_shape_matrix(monkeypatch, tmp_path) -> None:
    """import dry-run should report manifest, entity, privacy, and artifact shape errors."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "import-demo"])
    bundle = {
        "manifest": {
            "export_format_version": "9.0",
            "schema_version": "schema-unknown",
            "entity_counts": [],
            "manifest_hash": "bad",
            "payload_hash": "bad",
        },
        "entities": {
            "projects": [{"id": ""}, {"id": "proj_a"}, {"id": "proj_a"}],
            "runs": "not-an-array",
            "extra_entities": [],
        },
        "artifact_index": {},
        "privacy_flags": {},
        "provenance": {},
        "extra": True,
    }
    _write_raw_json(tmp_path, bundle)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json"])

    assert result.exit_code == 1
    for expected in (
        "unknown_top_level_key",
        "missing_manifest_key",
        "invalid_manifest_hash",
        "invalid_payload_hash",
        "missing_entity_array",
        "unknown_entity_array",
        "invalid_entity_array",
        "missing_entity_id",
        "duplicate_entity_id",
        "invalid_entity_counts",
        "invalid_artifact_index",
        "invalid_privacy_flags",
        "unsupported_export_format_version",
        "unsupported_schema_version",
    ):
        assert expected in result.stdout
    assert "Traceback" not in result.stdout


def test_import_dry_run_reports_count_mismatch_and_invalid_related_experiments(
    monkeypatch,
    tmp_path,
) -> None:
    """import dry-run should validate manifest counts and related experiment shape."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _bundle_from_project(tmp_path)
    bundle["manifest"]["entity_counts"]["projects"] = 99
    bundle["entities"]["decisions"][0]["related_experiments"] = "not-a-list"
    _write_bundle(tmp_path, bundle, refresh_hashes=True)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json"])

    assert result.exit_code == 1
    assert "entity_count_mismatch" in result.stdout
    assert "invalid_related_experiments" in result.stdout
    assert "Traceback" not in result.stdout


def test_import_dry_run_reports_missing_required_reference(monkeypatch, tmp_path) -> None:
    """Blank required references should be rejected before apply exists."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _bundle_from_project(tmp_path)
    bundle["entities"]["decisions"][0]["project_id"] = ""
    _write_bundle(tmp_path, bundle, refresh_hashes=True)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json"])

    assert result.exit_code == 1
    assert "missing_reference" in result.stdout
    assert "must reference an existing record" in result.stdout
    assert "Traceback" not in result.stdout


def test_import_dry_run_reviews_artifact_metadata(monkeypatch, tmp_path) -> None:
    """Artifact index metadata should show up in the privacy review."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _bundle_from_project(tmp_path)
    bundle["artifact_index"].append(_artifact_entry(bundle))
    bundle["manifest"]["artifact_count"] = 1
    _write_bundle(tmp_path, bundle, refresh_hashes=True)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert any(item["field"] == "artifact_index" for item in payload["privacy_review"])


@pytest.mark.parametrize(
    ("path_value", "expected_message"),
    [
        ("../secret.txt", "path traversal"),
        ("/tmp/secret.txt", "project-relative"),
        (".PMEM/pmem.db", "inside .pmem"),
        ("safe/../../evil.txt", "path traversal"),
        ("outputs\\model.bin", "POSIX"),
        ("outputs/\x00model.bin", "control characters"),
        ("outputs/\x1fmodel.bin", "control characters"),
    ],
)
def test_import_dry_run_rejects_malicious_artifact_paths(
    monkeypatch,
    tmp_path,
    path_value,
    expected_message,
) -> None:
    """Artifact metadata paths must not bypass bundle path safety."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _bundle_from_project(tmp_path)
    bundle["artifact_index"].append(_artifact_entry(bundle, path=path_value))
    bundle["manifest"]["artifact_count"] = 1
    _write_bundle(tmp_path, bundle, refresh_hashes=True)
    before = _row_counts(tmp_path)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json", "--json"])
    after = _row_counts(tmp_path)
    payload = json.loads(result.stdout)

    assert result.exit_code == 1
    assert before == after
    assert any(error["code"] == "invalid_artifact_path" for error in payload["errors"])
    assert expected_message in result.stdout
    assert "Traceback" not in result.stdout
    assert str(tmp_path) not in result.stdout


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"sha256": "sha256:" + "0" * 64}, "invalid_artifact_hash"),
        ({"sha256": "A" * 64}, "invalid_artifact_hash"),
        ({"hash_algorithm": "md5"}, "unsupported_artifact_hash_algorithm"),
        ({"size_bytes": -1}, "invalid_artifact_size"),
        ({"size_bytes": "1"}, "invalid_artifact_size"),
        ({"run_id": "run_missing"}, "missing_dependency"),
    ],
)
def test_import_dry_run_rejects_invalid_artifact_metadata_fields(
    monkeypatch,
    tmp_path,
    overrides,
    expected_code,
) -> None:
    """Artifact metadata should have valid hash, size, algorithm, and run refs."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _bundle_from_project(tmp_path)
    bundle["artifact_index"].append(_artifact_entry(bundle, **overrides))
    bundle["manifest"]["artifact_count"] = 1
    _write_bundle(tmp_path, bundle, refresh_hashes=True)
    before = _row_counts(tmp_path)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json", "--json"])
    after = _row_counts(tmp_path)
    payload = json.loads(result.stdout)

    assert result.exit_code == 1
    assert before == after
    assert any(error["code"] == expected_code for error in payload["errors"])
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()


def test_import_dry_run_rejects_missing_artifact_required_fields(monkeypatch, tmp_path) -> None:
    """Artifact index entries should contain the required metadata fields."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _bundle_from_project(tmp_path)
    bundle["artifact_index"].append({})
    bundle["manifest"]["artifact_count"] = 1
    _write_bundle(tmp_path, bundle, refresh_hashes=True)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json", "--json"])
    payload = json.loads(result.stdout)
    codes = {error["code"] for error in payload["errors"]}

    assert result.exit_code == 1
    assert {"invalid_artifact_path", "missing_artifact_hash", "invalid_artifact_size"}.issubset(
        codes
    )
    assert "missing_reference" in codes
    assert "Traceback" not in result.stdout


def test_import_dry_run_rejects_duplicate_artifact_paths_after_normalization(
    monkeypatch,
    tmp_path,
) -> None:
    """Artifact paths should be unique after POSIX normalization and case-folding."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _bundle_from_project(tmp_path)
    bundle["artifact_index"].append(_artifact_entry(bundle, path="Outputs/Model.bin"))
    bundle["artifact_index"].append(_artifact_entry(bundle, path="outputs/./model.bin"))
    bundle["manifest"]["artifact_count"] = 2
    _write_bundle(tmp_path, bundle, refresh_hashes=True)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 1
    assert any(error["code"] == "duplicate_artifact_path" for error in payload["errors"])
    assert "Traceback" not in result.stdout


def test_import_dry_run_rejects_artifact_count_mismatch(monkeypatch, tmp_path) -> None:
    """Manifest artifact_count should match artifact_index length."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _bundle_from_project(tmp_path)
    bundle["artifact_index"].append(_artifact_entry(bundle))
    bundle["manifest"]["artifact_count"] = 2
    _write_bundle(tmp_path, bundle, refresh_hashes=True)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 1
    assert any(error["code"] == "artifact_count_mismatch" for error in payload["errors"])
    assert "Traceback" not in result.stdout


def test_import_dry_run_rejects_non_object_artifact_entry(monkeypatch, tmp_path) -> None:
    """Artifact index entries should be objects."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _bundle_from_project(tmp_path)
    bundle["artifact_index"].append("outputs/model.bin")
    bundle["manifest"]["artifact_count"] = 1
    _write_bundle(tmp_path, bundle, refresh_hashes=True)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 1
    assert any(error["code"] == "invalid_artifact_entry" for error in payload["errors"])
    assert "Traceback" not in result.stdout


def test_import_dry_run_requires_initialized_project(monkeypatch, tmp_path) -> None:
    """import dry-run missing-init errors should stay safe and clear."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "bundle.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json"])

    assert result.exit_code == 1
    assert "projmem is not initialized. Run `pmem init` first." in result.stdout
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()


def test_import_dry_run_rejects_path_traversal_without_leaking_absolute_path(
    monkeypatch,
    tmp_path,
) -> None:
    """Bundle paths must stay inside the project until shared_paths exists."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "import-demo"])
    outside = tmp_path.parent / f"{tmp_path.name}_bundle.json"
    outside.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["import", "--dry-run", f"../{outside.name}"])

    assert result.exit_code == 1
    assert "Bundle path must stay inside the project." in result.stdout
    assert str(tmp_path.parent) not in result.stdout
    assert "Traceback" not in result.stdout


def test_import_dry_run_rejects_pmem_bundle_path_case_insensitive(monkeypatch, tmp_path) -> None:
    """import dry-run should inherit the local-memory .pmem privacy boundary for bundle paths."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "import-demo"])

    result = runner.invoke(app, ["import", "--dry-run", ".PMEM/bundle.json"])

    assert result.exit_code == 1
    assert "Bundle path cannot point inside .pmem." in result.stdout
    assert "Traceback" not in result.stdout


def test_import_dry_run_rejects_blank_bundle_path_after_init(monkeypatch, tmp_path) -> None:
    """Service-level blank path validation should remain explicit."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "import-demo"], catch_exceptions=False)

    with pytest.raises(PmemValidationError, match="Bundle path cannot be blank"):
        dry_run_import_bundle(tmp_path, " ")


def test_import_dry_run_rejects_absolute_missing_directory_and_symlink_paths(
    monkeypatch,
    tmp_path,
) -> None:
    """import dry-run path validation should avoid unsafe bundle path classes."""

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "import-demo"])
    (tmp_path / "bundle.json").write_text("{}", encoding="utf-8")
    (tmp_path / "bundle_dir").mkdir()
    try:
        (tmp_path / "bundle_link.json").symlink_to(tmp_path / "bundle.json")
    except OSError:
        pytest.skip("symlink creation is not available on this filesystem")

    absolute_result = runner.invoke(app, ["import", "--dry-run", str(tmp_path / "bundle.json")])
    missing_result = runner.invoke(app, ["import", "--dry-run", "missing.json"])
    directory_result = runner.invoke(app, ["import", "--dry-run", "bundle_dir"])
    symlink_result = runner.invoke(app, ["import", "--dry-run", "bundle_link.json"])

    assert absolute_result.exit_code == 1
    assert "Bundle path must be project-relative." in absolute_result.stdout
    assert str(tmp_path) not in absolute_result.stdout
    assert missing_result.exit_code == 1
    assert "Bundle file was not found." in missing_result.stdout
    assert directory_result.exit_code == 1
    assert "Bundle path must point to a file, not a directory." in directory_result.stdout
    assert symlink_result.exit_code == 1
    assert "Bundle path cannot contain symlinks." in symlink_result.stdout


def _seed_project(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    init_result = runner.invoke(app, ["init", "--name", "import-demo"])
    track_result = runner.invoke(app, ["track", "README.md"])
    run_id = _run_with_metric(tmp_path, "accuracy", 0.91)
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
    note_result = runner.invoke(app, ["note", "Review import.", "--run-id", run_id])

    assert init_result.exit_code == 0
    assert track_result.exit_code == 0
    assert failure_result.exit_code == 0
    assert decision_result.exit_code == 0
    assert note_result.exit_code == 0


def _run_with_metric(tmp_path: Path, metric: str, value: float) -> str:
    script = (
        "from pathlib import Path; import json; "
        f"Path('metrics.json').write_text(json.dumps({{{metric!r}: {value}}}), "
        "encoding='utf-8'); print('ok')"
    )
    result = runner.invoke(
        app,
        ["run", "--metrics", "metrics.json", "--", sys.executable, "-c", script],
    )
    assert result.exit_code == 0
    assert (tmp_path / "metrics.json").is_file()
    return result.stdout.split()[1]


def _bundle_from_project(project_root: Path) -> dict[str, Any]:
    export_payload = export_project(project_root)
    entities = copy.deepcopy(export_payload["entities"])
    for run in entities["runs"]:
        git = run.get("git", {})
        run["git_commit_hash"] = git.get("commit") if isinstance(git, dict) else None

    bundle = {
        "manifest": {
            "export_format_version": "1.0",
            "schema_version": export_payload["schema_version"],
            "generated_at": "2026-05-21T00:00:00Z",
            "freeze_timestamp": True,
            "project_id": export_payload["project_id"],
            "entity_counts": {
                entity_type: len(records) for entity_type, records in entities.items()
            },
            "artifact_count": 0,
            "canonical_json": {
                "encoding": "utf-8",
                "sort_keys": True,
                "separators": [",", ":"],
                "line_ending": "lf",
            },
            "manifest_hash": None,
            "payload_hash": None,
        },
        "entities": entities,
        "artifact_index": [],
        "privacy_flags": [
            {
                "code": "free_text_present",
                "severity": "warning",
                "field": "notes.content",
                "count": 1,
                "message": "Free-text memory may contain sensitive information.",
            }
        ],
        "provenance": {
            "tool": "projmem",
            "tool_version": "0.1.0a0",
            "source": "test-fixture",
            "project_name": "import-demo",
            "git_commit_hash": None,
            "git_dirty": False,
        },
    }
    _refresh_hashes(bundle)
    return bundle


def _artifact_entry(bundle: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    entry = {
        "path": "outputs/model.bin",
        "sha256": "0" * 64,
        "size_bytes": 1,
        "run_id": bundle["entities"]["runs"][0]["run_id"],
        "hash_algorithm": "sha256",
    }
    entry.update(overrides)
    return entry


def _write_bundle(
    project_root: Path,
    bundle: dict[str, Any],
    *,
    refresh_hashes: bool = True,
) -> Path:
    if refresh_hashes:
        _refresh_hashes(bundle)
    path = project_root / "bundle.json"
    path.write_text(
        json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_raw_json(project_root: Path, payload: Any) -> Path:
    path = project_root / "bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _refresh_hashes(bundle: dict[str, Any]) -> None:
    bundle["manifest"]["manifest_hash"] = None
    bundle["manifest"]["payload_hash"] = None
    bundle["manifest"]["payload_hash"] = _payload_hash(bundle)
    bundle["manifest"]["manifest_hash"] = _manifest_hash(bundle["manifest"])


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


def _row_counts(project_root: Path) -> dict[str, int]:
    connection = connect_database(project_root / ".pmem" / "pmem.db")
    try:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "projects",
                "experiments",
                "runs",
                "failures",
                "decisions",
                "notes",
                "tracked_paths",
            )
        }
    finally:
        connection.close()
