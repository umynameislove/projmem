"""Project-local config file helpers.

file tracking uses a tiny YAML-compatible file at `.pmem/config.yaml` to keep the stable
project id outside the database. The parser is deliberately narrow so local-memory
does not need a YAML dependency for four scalar fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pmem.errors import PmemValidationError
from pmem.repositories.sqlite import PMEM_DIRNAME

CONFIG_FILENAME = "config.yaml"
CONFIG_VERSION = 1


@dataclass(frozen=True)
class ProjectConfig:
    """Stable project identity stored in `.pmem/config.yaml`."""

    version: int
    project_id: str
    project_name: str
    created_at: str


def project_config_path(project_root: str | Path) -> Path:
    """Return the project-local config path."""

    return Path(project_root) / PMEM_DIRNAME / CONFIG_FILENAME


def read_project_config(config_path: str | Path) -> ProjectConfig:
    """Read the narrow file tracking config format."""

    raw_values: dict[str, object] = {}
    try:
        lines = Path(config_path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PmemValidationError("Project config could not be read.") from exc

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise PmemValidationError("Project config is invalid.")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            raw_values[key] = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise PmemValidationError("Project config is invalid.") from exc

    version = raw_values.get("version")
    project_id = raw_values.get("project_id")
    project_name = raw_values.get("project_name")
    created_at = raw_values.get("created_at")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or not isinstance(project_id, str)
        or not isinstance(project_name, str)
        or not isinstance(created_at, str)
    ):
        raise PmemValidationError("Project config is invalid.")

    config = ProjectConfig(
        version=version,
        project_id=project_id,
        project_name=project_name,
        created_at=created_at,
    )

    if config.version != CONFIG_VERSION:
        raise PmemValidationError("Project config version is unsupported.")
    if not config.project_id.strip() or not config.project_name.strip():
        raise PmemValidationError("Project config is invalid.")
    return config


def write_project_config_if_missing(config_path: str | Path, config: ProjectConfig) -> bool:
    """Create the config file without overwriting an existing one."""

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(format_project_config(config))
        path.chmod(0o600)
    except FileExistsError:
        return False
    except OSError as exc:
        raise PmemValidationError("Project config could not be written.") from exc
    return True


def format_project_config(config: ProjectConfig) -> str:
    """Render config as strict YAML-compatible scalar lines."""

    return (
        "# projmem local project config\n"
        f"version: {json.dumps(config.version)}\n"
        f"project_id: {json.dumps(config.project_id)}\n"
        f"project_name: {json.dumps(config.project_name)}\n"
        f"created_at: {json.dumps(config.created_at)}\n"
    )
