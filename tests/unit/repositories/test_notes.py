"""Tests for the note repository."""

import json

import pytest

from pmem.errors import PmemPersistenceError
from pmem.migrations.runner import apply_migrations
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.notes import NoteRepository
from pmem.repositories.projects import ProjectRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database

NOW = "2026-05-18T00:00:00Z"


@pytest.fixture()
def repository(tmp_path):
    """Return a note repository backed by one migrated run."""

    db_path = tmp_path / "pmem.db"
    apply_migrations(db_path)
    connection = connect_database(db_path)
    try:
        ProjectRepository(connection).create(
            project_id="proj_1",
            name="demo",
            created_at=NOW,
            updated_at=NOW,
        )
        ExperimentRepository(connection).create(
            experiment_id="exp_1",
            project_id="proj_1",
            name="default",
            created_at=NOW,
            updated_at=NOW,
        )
        RunRepository(connection).create(
            run_id="run_1",
            experiment_id="exp_1",
            command="python train.py",
            cwd=".",
            exit_code=0,
            status="success",
            timestamp=NOW,
        )
        yield NoteRepository(connection)
    finally:
        connection.close()


def test_create_and_read_note(repository: NoteRepository) -> None:
    """A note should round-trip with deterministic tag and context JSON."""

    record = repository.create(
        note_id="note_1",
        project_id="proj_1",
        experiment_id="exp_1",
        run_id="run_1",
        content="Try lower learning rate next.",
        tags=["follow_up"],
        context={"kind": "question"},
        resolved=False,
        created_at=NOW,
    )

    assert repository.get_by_id("note_1") == record
    assert repository.list_for_project("proj_1") == (record,)
    assert json.loads(record.tags_json) == ["follow_up"]
    assert json.loads(record.context_json) == {"kind": "question"}


def test_orphan_run_is_rejected(repository: NoteRepository) -> None:
    """Notes cannot attach to missing runs."""

    with pytest.raises(PmemPersistenceError, match="constraint violation"):
        repository.create(
            note_id="note_bad",
            project_id="proj_1",
            experiment_id=None,
            run_id="missing",
            content="bad",
            tags=[],
            context={},
            resolved=False,
            created_at=NOW,
        )
