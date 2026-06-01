"""`pmem run` service workflow.

Run capture records command execution evidence and reproducibility metadata.
The service owns subprocess execution, artifact paths, privacy filtering, and
the orchestration between project config, default experiment, and run storage.
"""

from __future__ import annotations

import json
import math
import platform
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem import __version__
from pmem.domain.common import RunStatus
from pmem.errors import PmemNotFoundError, PmemSecurityError, PmemValidationError
from pmem.integrations.git import collect_git_metadata
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.projects import ProjectRepository
from pmem.repositories.runs import RunRecord, RunRepository
from pmem.repositories.sqlite import PMEM_DIRNAME, connect_database, project_database_path
from pmem.services.config import project_config_path, read_project_config
from pmem.services.database import ensure_database
from pmem.services.project_init import ARTIFACTS_DIRNAME
from pmem.utils.hashing import compute_bytes_hash, compute_file_hash

RUNS_DIRNAME = "runs"
STDOUT_FILENAME = "stdout.txt"
STDERR_FILENAME = "stderr.txt"
STDIO_PREVIEW_CHAR_LIMIT = 2048
MAX_RUN_NAME_LENGTH = 120
MAX_SEED_LENGTH = 128
MAX_DATASET_LABEL_LENGTH = 120
MAX_METADATA_PATH_LENGTH = 512
MAX_JSON_METADATA_BYTES = 1024 * 1024
DATASET_LABEL_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "auth",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
)


@dataclass(frozen=True)
class RunCaptureResult:
    """User-facing result of `pmem run`."""

    record: RunRecord
    stdout_path: str
    stderr_path: str
    artifact_count: int


@dataclass(frozen=True)
class MetadataPath:
    """A project-relative metadata file path after safety validation."""

    absolute_path: Path
    relative_path: str


@dataclass(frozen=True)
class MetadataSnapshot:
    """Pre-run file identity used to reject stale metadata files."""

    exists: bool
    mtime_ns: int | None
    size_bytes: int | None


@dataclass(frozen=True)
class CapturedCommand:
    """Raw command result before SQLite persistence."""

    status: RunStatus
    exit_code: int | None
    duration_sec: float
    stdout: bytes
    stderr: bytes


def run_command(
    project_root: str | Path,
    command_args: list[str],
    *,
    name: str | None = None,
    timeout_seconds: float | None = None,
    seed: str | None = None,
    config_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    artifact_paths: tuple[str | Path, ...] = (),
    dataset_id: str | None = None,
    dataset_version: str | None = None,
) -> RunCaptureResult:
    """Run one command and persist execution evidence in the local project DB."""

    root = Path(project_root)
    _assert_initialized(root)
    command = _validate_command(command_args)
    run_name = _validate_optional_text(name, "Run name", MAX_RUN_NAME_LENGTH)
    run_seed = _validate_optional_text(seed, "Run seed", MAX_SEED_LENGTH)
    timeout = _validate_timeout(timeout_seconds)
    config_metadata_path = _validate_metadata_path(root, config_path) if config_path else None
    metrics_metadata_path = _validate_metadata_path(root, metrics_path) if metrics_path else None
    artifact_metadata_paths = tuple(
        _validate_metadata_path(root, artifact_path) for artifact_path in artifact_paths
    )
    dataset_metadata = _dataset_metadata(dataset_id, dataset_version)
    metrics_snapshot = _metadata_snapshot(metrics_metadata_path)

    config_payload, config_hash = _load_config_metadata(config_metadata_path)
    ensure_database(root)
    config = read_project_config(project_config_path(root))
    timestamp = _utc_now_iso()

    connection = connect_database(project_database_path(root))
    try:
        project = ProjectRepository(connection).get_by_id(config.project_id)
        if project is None:
            raise PmemNotFoundError("projmem is not initialized. Run `pmem init` first.")
        experiment = ExperimentRepository(connection).get_or_create_default(
            project_id=project.id,
            timestamp=timestamp,
        )
    finally:
        connection.close()

    run_id = f"run_{uuid.uuid4().hex}"
    run_dir_relative = f"{PMEM_DIRNAME}/{ARTIFACTS_DIRNAME}/{RUNS_DIRNAME}/{run_id}"
    run_dir = root / run_dir_relative
    _create_private_directory(run_dir)

    captured = _execute_command(root, command, timeout)
    stdout_relative = f"{run_dir_relative}/{STDOUT_FILENAME}"
    stderr_relative = f"{run_dir_relative}/{STDERR_FILENAME}"
    _write_private_file(root / stdout_relative, captured.stdout)
    _write_private_file(root / stderr_relative, captured.stderr)

    metrics = _load_metrics_metadata_if_fresh(metrics_metadata_path, metrics_snapshot)
    artifacts = _load_artifact_metadata(artifact_metadata_paths, dataset_metadata)
    env = _safe_runtime_metadata()
    git = collect_git_metadata(root)

    connection = connect_database(project_database_path(root))
    try:
        record = RunRepository(connection).create(
            run_id=run_id,
            experiment_id=experiment.id,
            name=run_name,
            command=shlex.join(command),
            cwd=".",
            exit_code=captured.exit_code,
            status=captured.status.value,
            duration_sec=captured.duration_sec,
            seed=run_seed,
            stdout_path=stdout_relative,
            stderr_path=stderr_relative,
            stdout_preview=_preview_bytes(captured.stdout),
            stderr_preview=_preview_bytes(captured.stderr),
            env=env,
            config=config_payload,
            config_hash=config_hash,
            metrics=metrics,
            artifacts=artifacts,
            git=git,
            evaluation={},
            failure_candidates=[],
            timestamp=timestamp,
        )
    finally:
        connection.close()

    return RunCaptureResult(
        record=record,
        stdout_path=stdout_relative,
        stderr_path=stderr_relative,
        artifact_count=len(artifacts),
    )


def _assert_initialized(project_root: Path) -> None:
    """Require explicit `pmem init` before run capture starts."""

    if (
        not project_config_path(project_root).exists()
        or not project_database_path(project_root).exists()
    ):
        raise PmemNotFoundError("projmem is not initialized. Run `pmem init` first.")


def _validate_command(command_args: list[str]) -> list[str]:
    """Validate command argv before subprocess execution."""

    if not command_args:
        raise PmemValidationError("Run command cannot be empty.")
    cleaned: list[str] = []
    for argument in command_args:
        if argument is None:
            raise PmemValidationError("Run command cannot be empty.")
        text = str(argument)
        if text == "":
            raise PmemValidationError("Run command cannot contain empty arguments.")
        if any(ord(character) < 32 for character in text):
            raise PmemValidationError("Run command contains unsupported control characters.")
        cleaned.append(text)
    return cleaned


def _validate_optional_text(value: str | None, label: str, max_length: int) -> str | None:
    """Validate optional run metadata text."""

    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        raise PmemValidationError(f"{label} cannot be blank.")
    if len(cleaned) > max_length:
        raise PmemValidationError(f"{label} is too long.")
    if any(ord(character) < 32 for character in cleaned):
        raise PmemValidationError(f"{label} contains unsupported control characters.")
    return cleaned


def _validate_timeout(timeout_seconds: float | None) -> float | None:
    """Validate optional timeout seconds."""

    if timeout_seconds is None:
        return None
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise PmemValidationError("Run timeout must be a positive finite number.")
    return timeout_seconds


def _validate_dataset_label(value: str | None, label: str) -> str | None:
    """Validate dataset identity metadata before storing it in artifacts_json."""

    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        raise PmemValidationError(f"{label} cannot be blank.")
    if len(cleaned) > MAX_DATASET_LABEL_LENGTH or not DATASET_LABEL_RE.fullmatch(cleaned):
        raise PmemValidationError(f"{label} is unsupported.")
    if any(part in cleaned.casefold() for part in SECRET_KEY_PARTS):
        raise PmemValidationError(f"{label} is unsupported.")
    return cleaned


def _dataset_metadata(dataset_id: str | None, version: str | None) -> dict[str, str] | None:
    """Return explicit dataset metadata for dataset-failure screening."""

    clean_id = _validate_dataset_label(dataset_id, "Dataset id")
    clean_version = _validate_dataset_label(version, "Dataset version")
    if clean_version is not None and clean_id is None:
        raise PmemValidationError("Dataset version requires --dataset-id.")
    if clean_id is None:
        return None
    return {
        "dataset_id": clean_id,
        "version": clean_version or "unknown",
        "metadata_kind": "dataset",
    }


def _validate_metadata_path(project_root: Path, user_path: str | Path) -> MetadataPath:
    """Validate a config, metrics, or artifact path without reading it yet."""

    raw_text = str(user_path)
    if not raw_text.strip():
        raise PmemValidationError("Run metadata path cannot be blank.")
    if len(raw_text) > MAX_METADATA_PATH_LENGTH:
        raise PmemValidationError("Run metadata path is too long.")
    if any(ord(character) < 32 for character in raw_text):
        raise PmemValidationError("Run metadata path contains unsupported control characters.")

    raw_path = Path(raw_text)
    if raw_path.is_absolute():
        raise PmemSecurityError("Run metadata path must be relative to the project root.")
    if _is_pmem_internal_path(raw_path):
        raise PmemSecurityError("projmem internal files cannot be used as run metadata.")

    root = project_root.resolve()
    candidate = project_root / raw_path
    if _has_symlink_component(project_root, raw_path):
        raise PmemSecurityError("Symlink metadata paths are not supported.")

    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise PmemSecurityError("Run metadata path must stay inside the project root.") from exc

    if _is_pmem_internal_path(relative):
        raise PmemSecurityError("projmem internal files cannot be used as run metadata.")
    return MetadataPath(absolute_path=candidate, relative_path=relative.as_posix())


def _has_symlink_component(project_root: Path, raw_path: Path) -> bool:
    """Return whether any existing user-provided path component is a symlink."""

    current = project_root
    for part in raw_path.parts:
        if part in ("", "."):
            continue
        current = current / part
        if current.is_symlink():
            return True
    return False


def _is_pmem_internal_path(path: Path) -> bool:
    """Return whether a path targets `.pmem/`, including case variants."""

    return bool(path.parts and path.parts[0].casefold() == PMEM_DIRNAME.casefold())


def _load_config_metadata(path: MetadataPath | None) -> tuple[dict[str, Any], str | None]:
    """Load explicit config metadata with secret-like keys redacted."""

    if path is None:
        return {}, None
    raw = _read_limited_json_file(path, "Run config file")
    data = _decode_json_object(raw, "Run config file")
    return _redact_sensitive_values(data), compute_bytes_hash(raw)


def _load_metrics_metadata(path: MetadataPath | None) -> dict[str, Any]:
    """Load explicit metrics metadata after the command finishes."""

    if path is None:
        return {}
    raw = _read_limited_json_file(path, "Run metrics file")
    data = _decode_json_object(raw, "Run metrics file")
    return _validate_metrics(data)


def _load_metrics_metadata_if_fresh(
    path: MetadataPath | None,
    before: MetadataSnapshot,
) -> dict[str, Any]:
    """Load metrics only when the command created or updated the metrics file."""

    if path is None:
        return {}
    after = _metadata_snapshot(path)
    if not after.exists or after == before:
        return {}
    return _load_metrics_metadata(path)


def _metadata_snapshot(path: MetadataPath | None) -> MetadataSnapshot:
    """Capture metadata file state without following unsafe paths into `.pmem`."""

    if path is None or not path.absolute_path.exists():
        return MetadataSnapshot(exists=False, mtime_ns=None, size_bytes=None)
    _assert_regular_file(path, "Run metrics file")
    try:
        stat_result = path.absolute_path.stat()
    except OSError:
        return MetadataSnapshot(exists=False, mtime_ns=None, size_bytes=None)
    return MetadataSnapshot(
        exists=True,
        mtime_ns=stat_result.st_mtime_ns,
        size_bytes=stat_result.st_size,
    )


def _load_artifact_metadata(
    paths: tuple[MetadataPath, ...],
    dataset_metadata: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Hash explicit run artifact files after the command finishes."""

    artifacts: list[dict[str, Any]] = []
    if dataset_metadata is not None:
        artifacts.append(dataset_metadata)
    for path in paths:
        _assert_regular_file(path, "Run artifact")
        try:
            size_bytes = path.absolute_path.stat().st_size
            sha256 = compute_file_hash(path.absolute_path)
        except OSError as exc:
            raise PmemValidationError("Run artifact could not be read.") from exc
        artifacts.append(
            {
                "path": path.relative_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    return artifacts


def _read_limited_json_file(path: MetadataPath, label: str) -> bytes:
    """Read a small JSON metadata file with safe public errors."""

    _assert_regular_file(path, label)
    try:
        size_bytes = path.absolute_path.stat().st_size
        if size_bytes > MAX_JSON_METADATA_BYTES:
            raise PmemValidationError(f"{label} is too large.")
        return path.absolute_path.read_bytes()
    except PmemValidationError:
        raise
    except OSError as exc:
        raise PmemValidationError(f"{label} could not be read.") from exc


def _assert_regular_file(path: MetadataPath, label: str) -> None:
    """Require metadata files to be regular files inside the project."""

    if not path.absolute_path.exists():
        raise PmemNotFoundError(f"{label} does not exist.")
    if path.absolute_path.is_symlink():
        raise PmemSecurityError("Symlink metadata paths are not supported.")
    if path.absolute_path.is_dir():
        raise PmemValidationError(f"{label} must be a file.")
    if not path.absolute_path.is_file():
        raise PmemValidationError(f"{label} must be a regular file.")


def _decode_json_object(raw: bytes, label: str) -> dict[str, Any]:
    """Decode one JSON object from bytes."""

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PmemValidationError(f"{label} must contain a JSON object.") from exc
    if not isinstance(data, dict):
        raise PmemValidationError(f"{label} must contain a JSON object.")
    return data


def _validate_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Validate metrics as a flat JSON object with primitive values."""

    validated: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key.strip():
            raise PmemValidationError("Metric names must be non-blank strings.")
        if len(key) > 120 or any(ord(character) < 32 for character in key):
            raise PmemValidationError("Metric name is unsupported.")
        if isinstance(value, bool) or value is None:
            validated[key] = value
        elif isinstance(value, int):
            validated[key] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise PmemValidationError("Metric value must be finite.")
            validated[key] = value
        elif isinstance(value, str):
            if len(value) > 512 or any(ord(character) < 32 for character in value):
                raise PmemValidationError("Metric value is unsupported.")
            validated[key] = value
        else:
            raise PmemValidationError("Metric values must be primitive JSON values.")
    return validated


def _redact_sensitive_values(value: Any) -> Any:
    """Return JSON-compatible config with secret-like keys redacted."""

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                redacted[key_text] = "***REDACTED***"
            else:
                redacted[key_text] = _redact_sensitive_values(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_values(item) for item in value]
    return value


def _is_sensitive_key(key: str) -> bool:
    """Return whether a config key should be redacted before SQLite storage."""

    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SECRET_KEY_PARTS)


def _execute_command(
    project_root: Path,
    command: list[str],
    timeout_seconds: float | None,
) -> CapturedCommand:
    """Execute a command without shell interpolation and capture raw streams."""

    started_at = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        duration = time.perf_counter() - started_at
        status = RunStatus.SUCCESS if completed.returncode == 0 else RunStatus.FAILED
        return CapturedCommand(
            status=status,
            exit_code=completed.returncode,
            duration_sec=duration,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started_at
        return CapturedCommand(
            status=RunStatus.TIMEOUT,
            exit_code=None,
            duration_sec=duration,
            stdout=_timeout_stream(exc.stdout),
            stderr=_timeout_stream(exc.stderr),
        )
    except KeyboardInterrupt:
        duration = time.perf_counter() - started_at
        return CapturedCommand(
            status=RunStatus.INTERRUPTED,
            exit_code=None,
            duration_sec=duration,
            stdout=b"",
            stderr=b"",
        )
    except OSError as exc:
        raise PmemValidationError("Run command could not be started.") from exc


def _timeout_stream(value: bytes | str | None) -> bytes:
    """Normalize subprocess timeout streams to bytes."""

    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="replace")


def _preview_bytes(content: bytes) -> str:
    """Decode and cap SQLite previews without changing full artifact files."""

    return content.decode("utf-8", errors="replace")[:STDIO_PREVIEW_CHAR_LIMIT]


def _create_private_directory(path: Path) -> None:
    """Create a project artifact directory restricted to the current OS user."""

    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as exc:
        raise PmemValidationError("Run artifact directory could not be created.") from exc


def _write_private_file(path: Path, content: bytes) -> None:
    """Write a run artifact file restricted to the current OS user."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o600)
    except OSError as exc:
        raise PmemValidationError("Run artifact file could not be written.") from exc


def _safe_runtime_metadata() -> dict[str, str]:
    """Return non-sensitive runtime metadata instead of a full env dump."""

    return {
        "platform": platform.system(),
        "pmem_version": __version__,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
    }


def _utc_now_iso() -> str:
    """Return compact UTC ISO timestamp for run records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
