"""Tests for the narrow `.pmem/config.yaml` helper."""

import pytest

from pmem.errors import PmemValidationError
from pmem.services.config import (
    ProjectConfig,
    format_project_config,
    project_config_path,
    read_project_config,
    write_project_config_if_missing,
)

NOW = "2026-05-15T00:00:00Z"


def test_project_config_path_is_project_local(tmp_path) -> None:
    """Config should live under project-local `.pmem/`."""

    assert project_config_path(tmp_path) == tmp_path / ".pmem" / "config.yaml"


def test_write_project_config_does_not_overwrite_existing_file(tmp_path) -> None:
    """Repeated init must not overwrite user-local config."""

    config_path = project_config_path(tmp_path)
    first = ProjectConfig(version=1, project_id="proj_1", project_name="demo", created_at=NOW)
    second = ProjectConfig(version=1, project_id="proj_2", project_name="other", created_at=NOW)

    assert write_project_config_if_missing(config_path, first) is True
    original_text = config_path.read_text(encoding="utf-8")
    assert write_project_config_if_missing(config_path, second) is False

    assert config_path.read_text(encoding="utf-8") == original_text
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert read_project_config(config_path) == first


def test_config_format_round_trips_values_that_look_like_sql(tmp_path) -> None:
    """Dangerous-looking scalar values should stay inert config data."""

    config_path = project_config_path(tmp_path)
    config = ProjectConfig(
        version=1,
        project_id="proj_1",
        project_name="'; DROP TABLE projects; --",
        created_at=NOW,
    )

    config_path.parent.mkdir(parents=True)
    config_path.write_text(format_project_config(config), encoding="utf-8")

    assert read_project_config(config_path) == config


def test_invalid_config_fails_with_safe_message(tmp_path) -> None:
    """Invalid config should not leak parser details."""

    config_path = project_config_path(tmp_path)
    config_path.parent.mkdir(parents=True)
    config_path.write_text("project_id: not-json\n", encoding="utf-8")

    with pytest.raises(PmemValidationError) as exc_info:
        read_project_config(config_path)

    assert str(exc_info.value) == "Project config is invalid."


def test_missing_config_file_fails_with_safe_message(tmp_path) -> None:
    """Missing config read errors should not expose local paths."""

    with pytest.raises(PmemValidationError) as exc_info:
        read_project_config(project_config_path(tmp_path))

    assert str(exc_info.value) == "Project config could not be read."
    assert str(tmp_path) not in str(exc_info.value)


def test_config_rejects_unsupported_version_and_blank_identity(tmp_path) -> None:
    """Config version and identity fields are part of init safety."""

    config_path = project_config_path(tmp_path)
    config_path.parent.mkdir(parents=True)

    config_path.write_text(
        'version: 2\nproject_id: "proj_1"\nproject_name: "demo"\ncreated_at: "now"\n',
        encoding="utf-8",
    )
    with pytest.raises(PmemValidationError, match="unsupported"):
        read_project_config(config_path)

    config_path.write_text(
        'version: 1\nproject_id: " "\nproject_name: "demo"\ncreated_at: "now"\n',
        encoding="utf-8",
    )
    with pytest.raises(PmemValidationError, match="invalid"):
        read_project_config(config_path)


def test_config_rejects_lines_without_key_value_separator(tmp_path) -> None:
    """Malformed config lines should be rejected by the narrow parser."""

    config_path = project_config_path(tmp_path)
    config_path.parent.mkdir(parents=True)
    config_path.write_text("not a key value line\n", encoding="utf-8")

    with pytest.raises(PmemValidationError, match="invalid"):
        read_project_config(config_path)
