"""portability security privacy regression tests for export bundle contents."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database

runner = CliRunner()


def test_export_bundle_strips_git_remote_url_and_private_project_path(
    monkeypatch, tmp_path
) -> None:
    """Safe Git metadata should survive export while remote URLs and project paths do not."""

    monkeypatch.chdir(tmp_path)
    run_id = _seed_project(tmp_path)
    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        connection.execute(
            "UPDATE runs SET git_json = ? WHERE run_id = ?",
            (
                json.dumps(
                    {
                        "commit": "abc123",
                        "branch": "main",
                        "remote_url": "https://example.invalid/private/repo.git",
                    }
                ),
                run_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()

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
    bundle_text = (tmp_path / "bundle.json").read_text(encoding="utf-8")
    bundle = json.loads(bundle_text)

    assert result.exit_code == 0
    assert bundle["entities"]["runs"][0]["git"] == {"branch": "main", "commit": "abc123"}
    assert "remote_url" not in bundle_text
    assert "example.invalid" not in bundle_text
    assert str(tmp_path) not in bundle_text


def test_export_bundle_redaction_removes_selected_free_text_before_hashing(
    monkeypatch, tmp_path
) -> None:
    """Explicit redaction should remove selected plaintext memory from shared bundles."""

    monkeypatch.chdir(tmp_path)
    _seed_project(tmp_path)

    result = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "redacted.json",
            "--redact-fields",
            "failures.description,decisions.rationale,notes.content",
            "--freeze-timestamp",
            "2026-05-22T00:00:00Z",
        ],
    )
    bundle_text = (tmp_path / "redacted.json").read_text(encoding="utf-8")
    bundle = json.loads(bundle_text)
    redacted_fields = {
        flag["field"] for flag in bundle["privacy_flags"] if flag["code"] == "redacted_field"
    }

    assert result.exit_code == 0
    assert "PRIVATE_DATASET_SAMPLE_001" not in bundle_text
    assert bundle["entities"]["failures"][0]["description"] == "[REDACTED]"
    assert bundle["entities"]["decisions"][0]["rationale"] == "[REDACTED]"
    assert bundle["entities"]["notes"][0]["content"] == "[REDACTED]"
    assert {
        "failures.description",
        "decisions.rationale",
        "notes.content",
    }.issubset(redacted_fields)


def test_export_bundle_privacy_flags_include_stdout_and_stderr_preview(
    monkeypatch, tmp_path
) -> None:
    """runs.stdout_preview and runs.stderr_preview must appear in privacy_flags when non-empty."""

    from pmem.repositories.sqlite import connect_database as _connect_database

    monkeypatch.chdir(tmp_path)
    run_id = _seed_project(tmp_path)

    connection = _connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        connection.execute(
            "UPDATE runs SET stdout_preview = ?, stderr_preview = ? WHERE run_id = ?",
            ("PRIVATE_STDOUT_CONTENT", "PRIVATE_STDERR_CONTENT", run_id),
        )
        connection.commit()
    finally:
        connection.close()

    result = runner.invoke(
        app,
        [
            "export-bundle",
            "--out",
            "bundle_flags.json",
            "--freeze-timestamp",
            "2026-05-22T00:00:00Z",
        ],
    )
    bundle = json.loads((tmp_path / "bundle_flags.json").read_text(encoding="utf-8"))
    flagged_fields = {flag["field"] for flag in bundle["privacy_flags"]}

    assert result.exit_code == 0
    assert "runs.stdout_preview" in flagged_fields, (
        "stdout_preview must be flagged as free_text_present in privacy_flags"
    )
    assert "runs.stderr_preview" in flagged_fields, (
        "stderr_preview must be flagged as free_text_present in privacy_flags"
    )
    # Content must appear in the bundle body (not redacted by default)
    # but privacy_flags warns the recipient
    assert any(
        f["code"] == "free_text_present" and f["field"] == "runs.stdout_preview"
        for f in bundle["privacy_flags"]
    )
    assert any(
        f["code"] == "free_text_present" and f["field"] == "runs.stderr_preview"
        for f in bundle["privacy_flags"]
    )


def _seed_project(project_root: Path) -> str:
    (project_root / "README.md").write_text("demo\n", encoding="utf-8")
    assert runner.invoke(app, ["init", "--name", "privacy-demo"]).exit_code == 0
    assert runner.invoke(app, ["track", "README.md"]).exit_code == 0
    script = "from pathlib import Path; Path('metrics.json').write_text('{\"acc\": 1}')"
    run_result = runner.invoke(
        app, ["run", "--metrics", "metrics.json", "--", sys.executable, "-c", script]
    )
    assert run_result.exit_code == 0
    run_id = run_result.stdout.split()[1]
    assert (
        runner.invoke(
            app,
            [
                "log-failure",
                run_id,
                "MetricRegression",
                "PRIVATE_DATASET_SAMPLE_001",
                "--root-cause",
                "Bad split.",
            ],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app,
            ["log-decision", "Keep baseline.", "--rationale", "PRIVATE_DATASET_SAMPLE_001"],
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["note", "PRIVATE_DATASET_SAMPLE_001"]).exit_code == 0
    return run_id
