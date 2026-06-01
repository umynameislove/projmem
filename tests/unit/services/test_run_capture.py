"""Service tests for the run capture/reproducibility metadata `pmem run` workflow."""

import json
import os
import subprocess
import sys

import pytest

import pmem.services.run_capture as run_capture
from pmem.errors import PmemNotFoundError, PmemSecurityError, PmemValidationError
from pmem.repositories.sqlite import connect_database
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command
from pmem.utils.hashing import compute_file_hash


def test_run_command_requires_initialized_project(tmp_path) -> None:
    """Run capture should not create `.pmem/` implicitly."""

    with pytest.raises(PmemNotFoundError, match="pmem init"):
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    assert not (tmp_path / ".pmem").exists()


def test_run_command_success_creates_default_experiment_and_artifacts(tmp_path) -> None:
    """run capture success path should persist one run and full stdout/stderr artifacts."""

    init_project(tmp_path, project_name="demo")

    result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"], name="baseline")

    assert result.record.status == "success"
    assert result.record.exit_code == 0
    assert result.record.name == "baseline"
    assert result.record.stdout_preview == "ok\n"
    assert (tmp_path / result.stdout_path).read_text(encoding="utf-8") == "ok\n"
    assert (tmp_path / result.stderr_path).read_text(encoding="utf-8") == ""

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        experiment_count = connection.execute("SELECT count(*) FROM experiments").fetchone()[0]
        run_count = connection.execute("SELECT count(*) FROM runs").fetchone()[0]
        row = connection.execute(
            "SELECT env_json, git_json, cwd FROM runs WHERE run_id = ?",
            (result.record.run_id,),
        ).fetchone()
    finally:
        connection.close()

    assert experiment_count == 1
    assert run_count == 1
    assert row["cwd"] == "."
    env = json.loads(row["env_json"])
    assert sorted(env) == ["platform", "pmem_version", "python_version"]
    assert json.loads(row["git_json"]) == {}


def test_run_command_is_idempotent_for_default_experiment(tmp_path) -> None:
    """Repeated runs should reuse the default experiment without duplicates."""

    init_project(tmp_path, project_name="demo")

    first = run_command(tmp_path, [sys.executable, "-c", "print('one')"])
    second = run_command(tmp_path, [sys.executable, "-c", "print('two')"])

    assert first.record.experiment_id == second.record.experiment_id

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        experiment_count = connection.execute("SELECT count(*) FROM experiments").fetchone()[0]
        run_count = connection.execute("SELECT count(*) FROM runs").fetchone()[0]
    finally:
        connection.close()

    assert experiment_count == 1
    assert run_count == 2


def test_run_command_failed_exit_is_recorded_without_throwing(tmp_path) -> None:
    """A nonzero command exit should create a failed run row."""

    init_project(tmp_path, project_name="demo")

    result = run_command(tmp_path, [sys.executable, "-c", "import sys; sys.exit(7)"])

    assert result.record.status == "failed"
    assert result.record.exit_code == 7


def test_run_command_caps_preview_but_keeps_full_stdout_artifact(tmp_path) -> None:
    """SQLite previews should stay capped while full stream artifacts remain."""

    init_project(tmp_path, project_name="demo")
    script = "print('x' * 3000)"

    result = run_command(tmp_path, [sys.executable, "-c", script])

    assert len(result.record.stdout_preview or "") == 2048
    assert len((tmp_path / result.stdout_path).read_text(encoding="utf-8")) == 3001


def test_run_command_captures_d8_metadata_with_redaction(tmp_path) -> None:
    """reproducibility metadata should store seed, redacted config, metrics, and artifact hashes."""

    init_project(tmp_path, project_name="demo")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "lr": 0.01,
                "api_token": "do-not-store",
                "nested": [{"secret": "redact-me"}],
            }
        ),
        encoding="utf-8",
    )
    script = (
        "from pathlib import Path; import json; "
        "Path('metrics.json').write_text(json.dumps({'accuracy': 0.91}), encoding='utf-8'); "
        "Path('model.txt').write_text('weights', encoding='utf-8')"
    )

    result = run_command(
        tmp_path,
        [sys.executable, "-c", script],
        seed="13",
        config_path="config.json",
        metrics_path="metrics.json",
        artifact_paths=("model.txt",),
    )

    config = json.loads(result.record.config_json)
    metrics = json.loads(result.record.metrics_json)
    artifacts = json.loads(result.record.artifacts_json)

    assert result.record.seed == "13"
    assert config == {
        "api_token": "***REDACTED***",
        "lr": 0.01,
        "nested": [{"secret": "***REDACTED***"}],
    }
    assert result.record.config_hash == compute_file_hash(config_path)
    assert metrics == {"accuracy": 0.91}
    assert artifacts == [
        {
            "path": "model.txt",
            "sha256": compute_file_hash(tmp_path / "model.txt"),
            "size_bytes": 7,
        }
    ]


def test_run_command_skips_stale_metrics_when_command_does_not_update_file(tmp_path) -> None:
    """Failed commands must not attach a previous run's metrics file by accident."""

    init_project(tmp_path, project_name="demo")
    (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 0.91}), encoding="utf-8")

    result = run_command(
        tmp_path,
        [sys.executable, "-c", "import sys; sys.exit(9)"],
        metrics_path="metrics.json",
    )

    assert result.record.status == "failed"
    assert json.loads(result.record.metrics_json) == {}


def test_run_command_keeps_failed_run_metrics_when_file_is_written_by_run(tmp_path) -> None:
    """Failed commands may still emit valid metrics when the file is fresh."""

    init_project(tmp_path, project_name="demo")
    script = (
        "from pathlib import Path; import json, sys; "
        "Path('metrics.json').write_text(json.dumps({'loss': 9.5}), encoding='utf-8'); "
        "sys.exit(3)"
    )

    result = run_command(
        tmp_path,
        [sys.executable, "-c", script],
        metrics_path="metrics.json",
    )

    assert result.record.status == "failed"
    assert json.loads(result.record.metrics_json) == {"loss": 9.5}


def test_run_command_records_explicit_dataset_metadata_for_pattern_mining(tmp_path) -> None:
    """Dataset-failure screening should have a normal `pmem run` metadata source."""

    init_project(tmp_path, project_name="demo")

    result = run_command(
        tmp_path,
        [sys.executable, "-c", "print('ok')"],
        dataset_id="fashion_mnist_full",
        dataset_version="v1",
    )

    assert json.loads(result.record.artifacts_json) == [
        {
            "dataset_id": "fashion_mnist_full",
            "metadata_kind": "dataset",
            "version": "v1",
        }
    ]


def test_run_command_rejects_invalid_dataset_metadata(tmp_path) -> None:
    """Dataset labels should be explicit metadata, not paths or secret-looking text."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemValidationError, match="Dataset version requires"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            dataset_version="v1",
        )
    with pytest.raises(PmemValidationError, match="Dataset id is unsupported"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            dataset_id="datasets/private/raw.csv",
        )
    with pytest.raises(PmemValidationError, match="Dataset id is unsupported"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            dataset_id="api_token",
        )


def test_run_command_rejects_empty_command_and_bad_timeout(tmp_path) -> None:
    """Invalid run inputs should fail before subprocess execution."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemValidationError, match="empty"):
        run_command(tmp_path, [])
    with pytest.raises(PmemValidationError, match="empty arguments"):
        run_command(tmp_path, [sys.executable, ""])
    with pytest.raises(PmemValidationError, match="empty"):
        run_command(tmp_path, [None])  # type: ignore[list-item]
    with pytest.raises(PmemValidationError, match="control"):
        run_command(tmp_path, [sys.executable, "bad\narg"])
    with pytest.raises(PmemValidationError, match="positive finite"):
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"], timeout_seconds=0)


def test_run_command_rejects_invalid_optional_text(tmp_path) -> None:
    """Run name and seed should be bounded, nonblank, and printable."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemValidationError, match="Run name cannot be blank"):
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"], name=" ")
    with pytest.raises(PmemValidationError, match="Run name is too long"):
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"], name="x" * 121)
    with pytest.raises(PmemValidationError, match="Run seed contains unsupported"):
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"], seed="bad\nseed")


def test_run_command_rejects_unsafe_metadata_paths(tmp_path) -> None:
    """Run metadata should not read internal or outside-project files."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemSecurityError, match="internal files"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            config_path=".pmem/config.yaml",
        )
    with pytest.raises(PmemSecurityError, match="internal files"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            config_path=".PMEM/config.yaml",
        )
    with pytest.raises(PmemSecurityError, match="inside"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            metrics_path="../metrics.json",
        )
    with pytest.raises(PmemValidationError, match="blank"):
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"], metrics_path=" ")
    with pytest.raises(PmemValidationError, match="too long"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            metrics_path=f"{'x' * 513}.json",
        )
    with pytest.raises(PmemValidationError, match="control"):
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"], metrics_path="bad\n.json")
    with pytest.raises(PmemSecurityError, match="relative"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            metrics_path=str(tmp_path / "metrics.json"),
        )
    with pytest.raises(PmemSecurityError, match="internal files"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            metrics_path="docs/../.pmem/config.yaml",
        )
    with pytest.raises(PmemSecurityError, match="internal files"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            metrics_path="docs/../.pMeM/config.yaml",
        )


def test_run_command_rejects_symlink_metadata_path(tmp_path) -> None:
    """reproducibility metadata metadata should not follow symlinks."""

    init_project(tmp_path, project_name="demo")
    target = tmp_path / "config.json"
    target.write_text("{}", encoding="utf-8")
    os.symlink(target, tmp_path / "config-link.json")

    with pytest.raises(PmemSecurityError, match="Symlink"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            config_path="config-link.json",
        )


def test_run_command_rejects_invalid_metrics_json(tmp_path) -> None:
    """Metrics must be a flat object with primitive finite values."""

    init_project(tmp_path, project_name="demo")
    script = "from pathlib import Path; Path('metrics.json').write_text('{\"accuracy\": [0.9]}')"

    with pytest.raises(PmemValidationError, match="primitive"):
        run_command(
            tmp_path,
            [sys.executable, "-c", script],
            metrics_path="metrics.json",
        )


def test_run_command_rejects_bad_json_metadata_shapes(monkeypatch, tmp_path) -> None:
    """Config/metrics JSON files should be small UTF-8 JSON objects."""

    init_project(tmp_path, project_name="demo")
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    (tmp_path / "array.json").write_text("[]", encoding="utf-8")
    (tmp_path / "large.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PmemValidationError, match="JSON object"):
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"], config_path="bad.json")
    with pytest.raises(PmemValidationError, match="JSON object"):
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"], config_path="array.json")

    monkeypatch.setattr(run_capture, "MAX_JSON_METADATA_BYTES", 1)
    with pytest.raises(PmemValidationError, match="too large"):
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"], config_path="large.json")


def test_run_command_validates_metric_edge_cases(tmp_path) -> None:
    """Metrics should reject unsafe names and values but allow primitives."""

    init_project(tmp_path, project_name="demo")
    write_valid = (
        "from pathlib import Path; import json; "
        "Path('valid.json').write_text("
        "json.dumps({'ok': True, 'count': 3, 'note': None, 'label': 'baseline'}), "
        "encoding='utf-8')"
    )

    result = run_command(
        tmp_path,
        [sys.executable, "-c", write_valid],
        metrics_path="./valid.json",
    )
    assert json.loads(result.record.metrics_json) == {
        "count": 3,
        "label": "baseline",
        "note": None,
        "ok": True,
    }

    with pytest.raises(PmemValidationError, match="Metric names"):
        run_command(
            tmp_path,
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('blank-key.json').write_text('{\"\": 1}')",
            ],
            metrics_path="blank-key.json",
        )
    with pytest.raises(PmemValidationError, match="Metric name is unsupported"):
        run_command(
            tmp_path,
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import json; "
                "Path('long-key.json').write_text(json.dumps({'%s': 1}))" % ("x" * 121),
            ],
            metrics_path="long-key.json",
        )
    with pytest.raises(PmemValidationError, match="finite"):
        run_command(
            tmp_path,
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('nan.json').write_text('{\"loss\": NaN}')",
            ],
            metrics_path="nan.json",
        )
    with pytest.raises(PmemValidationError, match="unsupported"):
        run_command(
            tmp_path,
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('bad-string.json').write_text("
                '\'{"note": "bad\\\\nvalue"}\')',
            ],
            metrics_path="bad-string.json",
        )


def test_run_command_rejects_missing_and_directory_artifacts(tmp_path) -> None:
    """Explicit artifacts must exist as regular files after the command."""

    init_project(tmp_path, project_name="demo")
    (tmp_path / "artifact-dir").mkdir()

    with pytest.raises(PmemNotFoundError, match="Run artifact does not exist"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            artifact_paths=("missing.txt",),
        )
    with pytest.raises(PmemValidationError, match="must be a file"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            artifact_paths=("artifact-dir",),
        )


def test_run_command_rejects_runtime_symlink_and_fifo_metadata(tmp_path) -> None:
    """Metadata created after validation is still checked before storage."""

    init_project(tmp_path, project_name="demo")
    os.mkfifo(tmp_path / "metrics.fifo")
    script = (
        "import os; from pathlib import Path; "
        "Path('real.json').write_text('{}', encoding='utf-8'); "
        "os.symlink('real.json', 'runtime-link.json')"
    )

    with pytest.raises(PmemSecurityError, match="Symlink"):
        run_command(
            tmp_path,
            [sys.executable, "-c", script],
            metrics_path="runtime-link.json",
        )
    with pytest.raises(PmemValidationError, match="regular file"):
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            metrics_path="metrics.fifo",
        )


def test_run_command_artifact_hash_error_is_safe(monkeypatch, tmp_path) -> None:
    """Artifact read failures should not leak raw filesystem details."""

    init_project(tmp_path, project_name="demo")
    (tmp_path / "artifact.txt").write_text("data", encoding="utf-8")

    def fail_hash(_path):
        raise OSError("raw path leak")

    monkeypatch.setattr(run_capture, "compute_file_hash", fail_hash)

    with pytest.raises(PmemValidationError) as exc_info:
        run_command(
            tmp_path,
            [sys.executable, "-c", "print('ok')"],
            artifact_paths=("artifact.txt",),
        )

    assert str(exc_info.value) == "Run artifact could not be read."
    assert "raw path leak" not in str(exc_info.value)


def test_run_command_fails_cleanly_when_project_row_is_missing(tmp_path) -> None:
    """Inconsistent config/database state should not leak SQLite details."""

    init_project(tmp_path, project_name="demo")
    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        connection.execute("DELETE FROM projects")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PmemNotFoundError, match="pmem init"):
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"])


def test_run_command_maps_subprocess_start_failure(monkeypatch, tmp_path) -> None:
    """Subprocess start failures should become safe validation errors."""

    init_project(tmp_path, project_name="demo")

    def fail_run(*_args, **_kwargs):
        raise OSError("raw executable path")

    monkeypatch.setattr(run_capture.subprocess, "run", fail_run)

    with pytest.raises(PmemValidationError) as exc_info:
        run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    assert str(exc_info.value) == "Run command could not be started."
    assert "raw executable path" not in str(exc_info.value)


def test_run_command_handles_keyboard_interrupt(monkeypatch, tmp_path) -> None:
    """KeyboardInterrupt should be captured as interrupted status."""

    init_project(tmp_path, project_name="demo")

    def interrupt_run(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_capture.subprocess, "run", interrupt_run)
    monkeypatch.setattr(run_capture, "collect_git_metadata", lambda _root: {})

    result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    assert result.record.status == "interrupted"
    assert result.record.exit_code is None


def test_run_command_timeout_normalizes_text_streams(monkeypatch, tmp_path) -> None:
    """TimeoutExpired streams may be text; service should store bytes safely."""

    init_project(tmp_path, project_name="demo")

    def timeout_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["demo"], timeout=1, output="partial", stderr="err")

    monkeypatch.setattr(run_capture.subprocess, "run", timeout_run)

    result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    assert result.record.status == "timeout"
    assert (tmp_path / result.stdout_path).read_text(encoding="utf-8") == "partial"
    assert (tmp_path / result.stderr_path).read_text(encoding="utf-8") == "err"


def test_run_command_timeout_preserves_byte_streams(monkeypatch, tmp_path) -> None:
    """TimeoutExpired byte streams should be stored unchanged."""

    init_project(tmp_path, project_name="demo")

    def timeout_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["demo"], timeout=1, output=b"partial")

    monkeypatch.setattr(run_capture.subprocess, "run", timeout_run)

    result = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    assert (tmp_path / result.stdout_path).read_bytes() == b"partial"


def test_run_command_timeout_records_timeout_status(tmp_path) -> None:
    """Timeouts should be captured as run rows instead of raw subprocess errors."""

    init_project(tmp_path, project_name="demo")

    result = run_command(
        tmp_path,
        [sys.executable, "-c", "import time; time.sleep(1)"],
        timeout_seconds=0.01,
    )

    assert result.record.status == "timeout"
    assert result.record.exit_code is None
