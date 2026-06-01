"""portability security malformed bundle payload and provenance validation tests."""

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


def test_import_rejects_non_utf8_bundle_without_traceback(monkeypatch, tmp_path) -> None:
    """Non-UTF-8 payloads should produce a controlled validation error."""

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--name", "payload-demo"]).exit_code == 0
    (tmp_path / "bundle.json").write_bytes(b"\xff\xfe\x00")

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json"])

    assert result.exit_code == 1
    assert "UTF-8 JSON" in result.stdout
    assert "Traceback" not in result.stdout
    assert str(tmp_path) not in result.stdout


def test_import_rejects_unknown_provenance_remote_url(monkeypatch, tmp_path) -> None:
    """Provenance must not silently accept remote URL metadata."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _exported_bundle(tmp_path)
    bundle["provenance"]["remote_url"] = "https://example.invalid/private/repo.git"
    _write_bundle(tmp_path, bundle)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 1
    assert any(error["code"] == "unknown_provenance_key" for error in payload["errors"])
    assert "example.invalid" not in result.stdout
    assert "Traceback" not in result.stdout


def test_import_rejects_invalid_provenance_shape(monkeypatch, tmp_path) -> None:
    """A bundle must carry object-shaped provenance for export-bundle provenance replay evidence."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)
    bundle = _exported_bundle(tmp_path)
    bundle["provenance"] = ["not", "an", "object"]
    _write_bundle(tmp_path, bundle)

    result = runner.invoke(app, ["import", "--dry-run", "bundle.json", "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 1
    assert any(error["code"] == "invalid_provenance" for error in payload["errors"])
    assert "Traceback" not in result.stdout


def _seed_project(project_root: Path) -> None:
    (project_root / "README.md").write_text("demo\n", encoding="utf-8")
    assert runner.invoke(app, ["init", "--name", "payload-demo"]).exit_code == 0
    assert runner.invoke(app, ["track", "README.md"]).exit_code == 0
    script = "from pathlib import Path; Path('metric.txt').write_text('ok', encoding='utf-8')"
    run_result = runner.invoke(app, ["run", "--", sys.executable, "-c", script])
    assert run_result.exit_code == 0


def _exported_bundle(project_root: Path) -> dict[str, Any]:
    result = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "bundle.json",
            "--freeze-timestamp",
            "2026-05-22T00:00:00Z",
            "--json",
        ],
    )
    assert result.exit_code == 0
    return json.loads((project_root / "bundle.json").read_text(encoding="utf-8"))


def _write_bundle(project_root: Path, bundle: dict[str, Any]) -> None:
    candidate = copy.deepcopy(bundle)
    candidate["manifest"]["manifest_hash"] = None
    candidate["manifest"]["payload_hash"] = None
    bundle["manifest"]["manifest_hash"] = None
    bundle["manifest"]["payload_hash"] = _sha256_tag(candidate)
    manifest = copy.deepcopy(bundle["manifest"])
    manifest["manifest_hash"] = None
    bundle["manifest"]["manifest_hash"] = _sha256_tag(manifest)
    (project_root / "bundle.json").write_text(
        json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256_tag(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
