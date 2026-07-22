"""Unit tests for the ``pmem status`` command and text renderer (STS-004)."""

from __future__ import annotations

import importlib
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from pmem.cli.status_output import print_status_text
from pmem.errors import PmemPersistenceError
from pmem.status import RecommendationMode, StatusPayload, StatusProject, StatusRecommendations

cli_module = importlib.import_module("pmem.cli.app")
status_output_module = importlib.import_module("pmem.cli.status_output")
runner = CliRunner()
_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures/status/status_payload_v1.json"


def _payload() -> StatusPayload:
    return StatusPayload.model_validate(json.loads(_FIXTURE_PATH.read_text(encoding="utf-8")))


def _render(payload: StatusPayload) -> str:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=240)
    print_status_text(payload, console=console)
    return stream.getvalue()


def test_status_command_uses_readonly_collection_without_recommendation_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_state = object()
    payload = _payload()
    calls: list[tuple[Path, bool]] = []

    def _collect(project_root: Path, *, evaluate_recommendations: bool) -> object:
        calls.append((project_root, evaluate_recommendations))
        return sentinel_state

    def _build(state: object) -> StatusPayload:
        assert state is sentinel_state
        return payload

    monkeypatch.setattr(cli_module, "collect_status_state", _collect)
    monkeypatch.setattr(cli_module, "build_status_payload", _build)

    result = runner.invoke(cli_module.app, ["status"])

    assert result.exit_code == 0
    assert calls == [(Path.cwd(), False)]
    assert "Action: rebuild_graph" in result.stdout
    assert result.stdout.count("Next action") == 1


def test_status_command_maps_service_errors_without_traceback_or_internals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise PmemPersistenceError("Project status could not be read safely.")

    monkeypatch.setattr(cli_module, "collect_status_state", _raise)

    result = runner.invoke(cli_module.app, ["status"])

    assert result.exit_code == 1
    assert "Error: Project status could not be read safely." in result.stdout
    assert "Traceback" not in result.stdout
    assert "sqlite" not in result.stdout.lower()
    assert "SELECT " not in result.stdout


def test_status_text_renders_all_sections_and_exact_next_action() -> None:
    text = _render(_payload())

    expected_lines = {
        "Project status",
        "Project: AG News baseline",
        "Primary metric: accuracy",
        "Target: 0.9",
        "Best value: 0.87",
        "Target status: not_met",
        "Runs: total=12 successful=8 failed=3",
        "Memory: tracked_paths=2 failures=5 decisions=1 notes=4",
        "Graph: stale nodes=40 edges=63 reason=graph_source_changed",
        "Recommendations: generated_on_demand candidates=3",
        (
            "- [WARNING] graph/graph_stale: Evidence graph is stale; "
            "rebuild before relying on lineage."
        ),
        "Action: rebuild_graph",
        "Reason: New runs were captured after the last graph build.",
        "Command: pmem graph build",
        "Safety: database_mutation=false network=false raw_text_in_output=false",
    }
    assert expected_lines <= set(text.splitlines())
    assert text.count("Next action") == 1
    assert text.count("Action: ") == 1
    assert text.count("Command: ") == 1


@pytest.mark.parametrize(
    ("recommendations", "expected"),
    [
        (
            StatusRecommendations(
                mode=RecommendationMode.NOT_EVALUATED,
                candidate_count=None,
                active_count=None,
            ),
            "Recommendations: not_evaluated",
        ),
        (
            StatusRecommendations(
                mode=RecommendationMode.GENERATED_ON_DEMAND,
                candidate_count=0,
                active_count=None,
            ),
            "Recommendations: generated_on_demand candidates=0",
        ),
        (
            StatusRecommendations(
                mode=RecommendationMode.PERSISTED_LIFECYCLE,
                candidate_count=4,
                active_count=2,
            ),
            "Recommendations: persisted_lifecycle candidates=4 active=2",
        ),
    ],
)
def test_status_text_renders_each_recommendation_mode(
    recommendations: StatusRecommendations,
    expected: str,
) -> None:
    payload = _payload().model_copy(update={"recommendations": recommendations})
    assert expected in _render(payload)


def test_status_text_disables_rich_markup_and_terminal_highlighting() -> None:
    payload = _payload().model_copy(
        update={
            "project": StatusProject(
                project_id="proj_" + "1" * 32,
                project_name="[bold red]literal project",
                objective="[underline]literal objective",
            )
        }
    )

    text = _render(payload)

    assert "[bold red]literal project" in text
    assert "[underline]literal objective" in text
    assert "\x1b" not in text


def test_status_help_is_registered_as_text_only_command() -> None:
    root_help = runner.invoke(cli_module.app, ["--help"])
    status_help = runner.invoke(cli_module.app, ["status", "--help"])

    assert root_help.exit_code == 0
    assert "status" in root_help.stdout
    assert status_help.exit_code == 0
    assert "Print concise read-only project status" in status_help.stdout
    assert "--json" not in status_help.stdout


def test_status_renderer_layer_has_no_service_or_filesystem_dependency() -> None:
    module_file = status_output_module.__file__
    assert module_file is not None
    source = Path(module_file).read_text(encoding="utf-8")
    assert "pmem.services" not in source
    assert "pathlib" not in source
    assert "Path(" not in source
