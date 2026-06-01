"""file tracking integration workflow tests."""

from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.repositories.sqlite import connect_database

runner = CliRunner()


def test_cli_init_then_track_keeps_database_integrity(monkeypatch, tmp_path) -> None:
    """The real file tracking CLI workflow should leave SQLite consistent."""

    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs" / "specs"
    docs.mkdir(parents=True)
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")
    (docs / "schema-v1.md").write_text("schema\n", encoding="utf-8")

    first_init = runner.invoke(app, ["init", "--name", "demo"])
    second_init = runner.invoke(app, ["init"])
    first_track = runner.invoke(app, ["track", "README.md"])
    second_track = runner.invoke(app, ["track", "docs/specs/schema-v1.md"])

    assert first_init.exit_code == 0
    assert second_init.exit_code == 0
    assert first_track.exit_code == 0
    assert second_track.exit_code == 0

    connection = connect_database(tmp_path / ".pmem" / "pmem.db")
    try:
        project_count = connection.execute("SELECT count(*) FROM projects").fetchone()[0]
        tracked_count = connection.execute("SELECT count(*) FROM tracked_paths").fetchone()[0]
        migration_versions = [
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()

    assert project_count == 1
    assert tracked_count == 2
    assert migration_versions == ["0001_schema_v1", "0002_phase2_portability"]
    assert foreign_key_rows == []
    assert integrity == "ok"
