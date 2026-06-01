"""Service tests for `pmem note`."""

import json
import sys

import pytest

from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.repositories.sqlite import connect_database
from pmem.services.note_logging import add_note
from pmem.services.project_init import init_project
from pmem.services.run_capture import run_command


def test_add_note_persists_project_note(tmp_path) -> None:
    """note logging should store project notes with normalized tags."""

    init_project(tmp_path, project_name="demo")
    record = add_note(
        tmp_path,
        content="Try smaller batch size next.",
        tags=("Open Question",),
    )

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        row = connection.execute("SELECT content, tags_json, resolved FROM notes").fetchone()
    finally:
        connection.close()

    assert record.id.startswith("note_")
    assert row["content"] == "Try smaller batch size next."
    assert json.loads(row["tags_json"]) == ["open_question"]
    assert row["resolved"] == 0


def test_add_note_links_to_run_and_infers_experiment(tmp_path) -> None:
    """Run-linked notes should keep consistent run/experiment context."""

    init_project(tmp_path, project_name="demo")
    run = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    record = add_note(
        tmp_path,
        content="This run is the first smoke test.",
        run_id=run.record.run_id,
        resolved=True,
    )

    assert record.run_id == run.record.run_id
    assert record.experiment_id == run.record.experiment_id
    assert record.resolved is True


def test_add_note_requires_initialized_project(tmp_path) -> None:
    """Note logging must not create implicit project state."""

    with pytest.raises(PmemNotFoundError, match="pmem init"):
        add_note(tmp_path, content="hello")


def test_add_note_rejects_missing_run(tmp_path) -> None:
    """Run-linked notes should fail cleanly when run_id does not exist."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemNotFoundError, match="Run was not found"):
        add_note(tmp_path, content="hello", run_id="run_missing")


def test_add_note_rejects_mismatched_run_and_experiment(tmp_path) -> None:
    """A note cannot link a run to a different experiment id."""

    init_project(tmp_path, project_name="demo")
    run = run_command(tmp_path, [sys.executable, "-c", "print('ok')"])

    with pytest.raises(PmemValidationError, match="provided experiment"):
        add_note(
            tmp_path,
            content="hello",
            run_id=run.record.run_id,
            experiment_id="exp_other",
        )


def test_add_note_rejects_blank_tag(tmp_path) -> None:
    """Blank note tags should not create useless JSON search keys."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemValidationError, match="tags cannot be blank"):
        add_note(tmp_path, content="hello", tags=("   ",))


def test_add_note_rejects_blank_content(tmp_path) -> None:
    """Note content is required memory, not optional whitespace."""

    init_project(tmp_path, project_name="demo")

    with pytest.raises(PmemValidationError, match="cannot be blank"):
        add_note(tmp_path, content="  ")
