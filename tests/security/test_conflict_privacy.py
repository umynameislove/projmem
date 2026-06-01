"""portability security conflict-report privacy regression tests."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from pmem.cli.app import app

runner = CliRunner()


def test_conflict_check_does_not_echo_incoming_free_text(monkeypatch, tmp_path) -> None:
    """Conflict reports should expose ids/hashes, not raw note or failure content."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _exported_bundle(tmp_path)
    bundle["entities"]["notes"][0]["content"] = "PRIVATE_NOTE_SHOULD_NOT_APPEAR"
    bundle["entities"]["failures"][0]["description"] = "PRIVATE_FAILURE_SHOULD_NOT_APPEAR"
    _write_bundle(tmp_path, bundle)

    result = runner.invoke(app, ["conflict-check", "bundle.json", "--json"])
    payload = json.loads(result.stdout)
    conflict_types = {item["conflict_type"] for item in payload["conflicts"]}

    assert result.exit_code == 0
    assert "same_id_different_hash" in conflict_types
    assert "PRIVATE_NOTE_SHOULD_NOT_APPEAR" not in result.stdout
    assert "PRIVATE_FAILURE_SHOULD_NOT_APPEAR" not in result.stdout
    assert "Traceback" not in result.stdout


def _seed_project(project_root: Path) -> str:
    (project_root / "README.md").write_text("demo\n", encoding="utf-8")
    assert runner.invoke(app, ["init", "--name", "conflict-privacy-demo"]).exit_code == 0
    assert runner.invoke(app, ["track", "README.md"]).exit_code == 0
    script = "from pathlib import Path; Path('metrics.json').write_text('{\"acc\": 1}')"
    run_result = runner.invoke(
        app, ["run", "--metrics", "metrics.json", "--", sys.executable, "-c", script]
    )
    assert run_result.exit_code == 0
    run_id = run_result.stdout.split()[1]
    assert runner.invoke(app, ["note", "Local note.", "--run-id", run_id]).exit_code == 0
    assert (
        runner.invoke(
            app,
            ["log-failure", run_id, "MetricRegression", "Local failure."],
        ).exit_code
        == 0
    )
    return run_id


def _exported_bundle(project_root: Path) -> dict[str, Any]:
    result = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "bundle.json",
            "--freeze-timestamp",
            "2026-05-22T00:00:00Z",
        ],
    )
    assert result.exit_code == 0
    return json.loads((project_root / "bundle.json").read_text(encoding="utf-8"))


def _write_bundle(project_root: Path, bundle: dict[str, Any]) -> None:
    bundle["manifest"]["manifest_hash"] = None
    bundle["manifest"]["payload_hash"] = None
    bundle["manifest"]["payload_hash"] = _payload_hash(bundle)
    bundle["manifest"]["manifest_hash"] = _manifest_hash(bundle["manifest"])
    (project_root / "bundle.json").write_text(
        json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
