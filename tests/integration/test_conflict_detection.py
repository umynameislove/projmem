"""conflict detection conflict detection foundation tests."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database
from pmem.services.conflict_detection import check_bundle_conflicts

runner = CliRunner()


def test_conflict_check_detects_same_id_same_hash_without_mutation(monkeypatch, tmp_path) -> None:
    """Checking a bundle against its source project should preview duplicate ids only."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    _export_bundle("bundle.json")
    before = _row_counts(tmp_path)

    result = runner.invoke(app, ["conflict-check", "bundle.json", "--json"])
    payload = json.loads(result.stdout)
    after = _row_counts(tmp_path)

    assert result.exit_code == 0
    assert before == after
    assert payload["database_mutation"] is False
    assert any(item["conflict_type"] == "same_id_same_hash" for item in payload["conflicts"])
    assert "Traceback" not in result.stdout


def test_conflict_check_text_output_and_validation_errors(monkeypatch, tmp_path) -> None:
    """Human output should be safe for both valid and invalid bundles."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    _export_bundle("bundle.json")
    valid = runner.invoke(app, ["conflict-check", "bundle.json"])
    (tmp_path / "bad.json").write_text("[]", encoding="utf-8")
    invalid = runner.invoke(app, ["conflict-check", "bad.json"])

    assert valid.exit_code == 0
    assert "Conflict check: PASS" in valid.stdout
    assert "Database mutation: none" in valid.stdout
    assert invalid.exit_code == 1
    assert "Conflict check: FAIL" in invalid.stdout
    assert "invalid_bundle_shape" in invalid.stdout
    assert "Traceback" not in invalid.stdout


def test_conflict_check_detects_divergent_and_semantic_duplicate(monkeypatch, tmp_path) -> None:
    """conflict detection should distinguish hash divergence from semantic duplicate runs."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _exported_bundle("bundle.json")
    bundle["entities"]["runs"][0]["stdout_preview"] = "changed output"
    semantic_duplicate = copy.deepcopy(bundle["entities"]["runs"][0])
    semantic_duplicate["run_id"] = "run_semantic_duplicate"
    bundle["entities"]["runs"].append(semantic_duplicate)
    bundle["manifest"]["entity_counts"]["runs"] = len(bundle["entities"]["runs"])
    _write_bundle(tmp_path, bundle)

    result = runner.invoke(app, ["conflict-check", "bundle.json", "--json"])
    payload = json.loads(result.stdout)
    conflict_types = {item["conflict_type"] for item in payload["conflicts"]}

    assert result.exit_code == 0
    assert "same_id_different_hash" in conflict_types
    assert "unsafe_overwrite_risk" in conflict_types
    assert "semantic_duplicate" in conflict_types
    assert "changed output" not in result.stdout


def test_conflict_check_detects_stale_baseline(monkeypatch, tmp_path) -> None:
    """Baseline drift should be previewed before any import resolution."""

    monkeypatch.chdir(tmp_path)
    run_id = _seed_project(tmp_path)
    assert runner.invoke(app, ["baseline", run_id]).exit_code == 0
    bundle = _exported_bundle("bundle.json")
    experiment = bundle["entities"]["experiments"][0]
    experiment["metadata"]["baseline_run_id"] = "run_old_baseline"
    experiment["updated_at"] = "2020-01-01T00:00:00Z"
    _write_bundle(tmp_path, bundle)

    report = check_bundle_conflicts(tmp_path, "bundle.json")

    assert any(item.conflict_type == "stale_baseline" for item in report.conflicts)
    assert report.database_mutation is False


def test_conflict_check_detects_artifact_collisions_and_missing_artifacts(
    monkeypatch, tmp_path
) -> None:
    """conflict detection artifact checks should cover path/hash collision and index gaps."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _exported_bundle("bundle.json")
    bundle["artifact_index"][0]["sha256"] = "1" * 64
    bundle["artifact_index"][0]["content_encoding"] = "base64"
    run_artifact_path = bundle["entities"]["runs"][0]["artifacts"][0]["path"]
    bundle["entities"]["runs"][0]["artifacts"].append(
        {"path": "outputs/missing.bin", "sha256": "2" * 64, "size_bytes": 7}
    )
    assert run_artifact_path
    _write_bundle(tmp_path, bundle)

    result = runner.invoke(app, ["conflict-check", "bundle.json", "--json"])
    payload = json.loads(result.stdout)
    conflict_types = {item["conflict_type"] for item in payload["conflicts"]}

    assert result.exit_code == 0
    assert "artifact_hash_mismatch" in conflict_types
    assert "missing_artifact" in conflict_types


def test_conflict_check_reports_validation_version_and_dependency_conflicts(
    monkeypatch, tmp_path
) -> None:
    """Version mismatch and missing dependency should be surfaced as actionable conflicts."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _exported_bundle("bundle.json")
    bundle["manifest"]["schema_version"] = "schema-v999"
    bundle["entities"]["runs"][0]["experiment_id"] = "exp_missing"
    _write_bundle(tmp_path, bundle)

    result = runner.invoke(app, ["conflict-check", "bundle.json", "--json"])
    payload = json.loads(result.stdout)
    conflict_types = {item["conflict_type"] for item in payload["conflicts"]}

    assert result.exit_code == 1
    assert payload["validation_ok"] is False
    assert "schema_version_mismatch" in conflict_types
    assert "missing_dependency" in conflict_types
    assert "Traceback" not in result.stdout


def test_conflict_check_detects_already_applied_package(monkeypatch, tmp_path) -> None:
    """A second check after import apply apply should flag the exact bundle hash."""

    bundle_text = _source_bundle(monkeypatch, tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.chdir(target)
    assert runner.invoke(app, ["init", "--name", "target"]).exit_code == 0
    (target / "incoming.json").write_text(bundle_text, encoding="utf-8")
    assert runner.invoke(app, ["import", "--apply", "incoming.json", "--confirm"]).exit_code == 0

    result = runner.invoke(app, ["conflict-check", "incoming.json", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert any(item["conflict_type"] == "already_applied_package" for item in payload["conflicts"])


def test_conflict_check_no_conflict_path_for_new_bundle(monkeypatch, tmp_path) -> None:
    """A new source bundle imported into an empty target should have no conflicts."""

    bundle_text = _source_bundle(monkeypatch, tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.chdir(target)
    assert runner.invoke(app, ["init", "--name", "target"]).exit_code == 0
    (target / "incoming.json").write_text(bundle_text, encoding="utf-8")

    result = runner.invoke(app, ["conflict-check", "incoming.json", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["conflict_count"] == 0
    assert payload["conflicts"] == []


def _source_bundle(monkeypatch, source: Path) -> str:
    source.mkdir()
    monkeypatch.chdir(source)
    _seed_project(source)
    _export_bundle("bundle.json")
    return (source / "bundle.json").read_text(encoding="utf-8")


def _exported_bundle(path: str) -> dict[str, Any]:
    _export_bundle(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _export_bundle(path: str) -> None:
    result = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            path,
            "--freeze-timestamp",
            "2026-01-01T00:00:00Z",
            "--json",
        ],
    )
    assert result.exit_code == 0


def _seed_project(project_root: Path) -> str:
    (project_root / "README.md").write_text("demo\n", encoding="utf-8")
    assert runner.invoke(app, ["init", "--name", "conflict-demo"]).exit_code == 0
    assert runner.invoke(app, ["track", "README.md"]).exit_code == 0
    run_id = _run_with_artifact(project_root)
    assert runner.invoke(app, ["note", "Review conflict.", "--run-id", run_id]).exit_code == 0
    return run_id


def _run_with_artifact(project_root: Path) -> str:
    script = (
        "from pathlib import Path; import json; "
        "Path('metrics.json').write_text(json.dumps({'accuracy': 0.91}), encoding='utf-8'); "
        "Path('outputs').mkdir(exist_ok=True); "
        "Path('outputs/model.txt').write_text('weights', encoding='utf-8'); "
        "print('ok')"
    )
    result = runner.invoke(
        app,
        [
            "run",
            "--metrics",
            "metrics.json",
            "--artifact",
            "outputs/model.txt",
            "--",
            sys.executable,
            "-c",
            script,
        ],
    )
    assert result.exit_code == 0
    assert (project_root / "outputs" / "model.txt").is_file()
    return result.stdout.split()[1]


def _write_bundle(project_root: Path, bundle: dict[str, Any]) -> Path:
    _refresh_hashes(bundle)
    path = project_root / "bundle.json"
    path.write_text(
        json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
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
                "import_jobs",
                "audit_events",
            )
        }
    finally:
        connection.close()
