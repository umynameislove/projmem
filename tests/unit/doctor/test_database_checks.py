"""Database doctor check tests (DOC-002).

Every case below uses a real SQLite file on disk. Nothing about SQLite is
mocked, so a production path that stopped issuing its query would turn these
tests red rather than leaving them green against a fake.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

import pmem.doctor.database_checks as database_checks_module
from pmem.doctor import (
    DoctorCategory,
    DoctorCheckContext,
    DoctorCheckOutcome,
    DoctorCheckResult,
    DoctorOverallOutcome,
    DoctorProject,
    DoctorSeverity,
    build_doctor_report,
    render_doctor_report_json,
)
from pmem.doctor.database_checks import (
    DATABASE_CHECK_IDS,
    DatabaseInspection,
    PathState,
    ProbeState,
    ReadState,
    database_check_definitions,
    inspect_database,
    run_database_checks,
)
from pmem.migrations.runner import CURRENT_MIGRATIONS
from pmem.repositories.sqlite import PMEM_DIRNAME, project_database_path
from pmem.services.project_init import init_project

_PROJECT = DoctorProject(
    project_id="proj_9f2c1a7b4d6e40f2a1b3c5d7e9f00112",
    project_name="AG News baseline",
)


# --------------------------------------------------------------------------- #
# Fixtures that build real project state                                       #
# --------------------------------------------------------------------------- #
def _healthy_project(tmp_path: Path) -> Path:
    init_project(tmp_path, project_name="doctor-db", primary_metric="accuracy")
    return tmp_path


def _run_all(project_root: Path) -> dict[str, DoctorCheckResult]:
    """Execute one invocation-scoped snapshot and key results by id."""

    context = DoctorCheckContext(project_root=project_root)
    return {result.check_id: result for result in run_database_checks(context)}


def _outcomes(results: dict[str, DoctorCheckResult]) -> dict[str, tuple[str, str]]:
    return {
        check_id: (result.outcome.value, result.severity.value)
        for check_id, result in results.items()
    }


def _corrupt_pages(db_path: Path) -> None:
    """Damage an interior page so SQLite reports a malformed image."""

    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE doctor_probe(id INTEGER PRIMARY KEY, value TEXT)")
    connection.execute("CREATE INDEX doctor_probe_value ON doctor_probe(value)")
    connection.executemany(
        "INSERT INTO doctor_probe(value) VALUES(?)",
        [(f"value-{index}" * 20,) for index in range(400)],
    )
    connection.commit()
    connection.close()

    raw = bytearray(db_path.read_bytes())
    for offset in range(4096 * 2, min(4096 * 3, len(raw))):
        raw[offset] = 0x00
    db_path.write_bytes(bytes(raw))


def _break_foreign_keys(db_path: Path) -> None:
    """Insert an orphan row with enforcement disabled, as tampering would."""

    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute(
        "INSERT INTO experiments"
        "(id, project_id, name, hypothesis, status, created_at, updated_at) "
        "VALUES('exp_orphan', 'proj_missing', 'orphan', 'h', 'active', "
        "'2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
    )
    connection.commit()
    connection.close()


def _drop_a_migration(db_path: Path) -> None:
    """Remove the newest recorded migration, using the real version string."""

    connection = sqlite3.connect(db_path)
    connection.execute(
        "DELETE FROM schema_migrations WHERE version = ?", (CURRENT_MIGRATIONS[-1].version,)
    )
    connection.commit()
    connection.close()


def _tamper_checksum(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE schema_migrations SET checksum = ? WHERE version = ?",
        ("a" * 64, CURRENT_MIGRATIONS[0].version),
    )
    connection.commit()
    connection.close()


# --------------------------------------------------------------------------- #
# Factory / registry contract                                                  #
# --------------------------------------------------------------------------- #
def test_factory_returns_the_six_stable_ids_in_canonical_order() -> None:
    definitions = database_check_definitions()

    assert tuple(definition.check_id for definition in definitions) == DATABASE_CHECK_IDS
    assert DATABASE_CHECK_IDS == tuple(sorted(DATABASE_CHECK_IDS))
    assert set(DATABASE_CHECK_IDS) == {
        "database.exists",
        "database.readable",
        "database.integrity",
        "database.foreign_keys",
        "database.migrations_current",
        "database.migration_checksums",
    }


def test_every_definition_is_a_database_category_definition() -> None:
    for definition in database_check_definitions():
        assert definition.category is DoctorCategory.DATABASE
        assert definition.check_id.startswith("database.")


def test_factory_is_deterministic_across_calls() -> None:
    first = tuple(definition.check_id for definition in database_check_definitions())
    second = tuple(definition.check_id for definition in database_check_definitions())

    assert first == second


def test_factory_runs_no_check_and_touches_no_disk(tmp_path: Path) -> None:
    before = sorted(path.name for path in tmp_path.iterdir())

    database_check_definitions()

    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_every_result_matches_its_definition(tmp_path: Path) -> None:
    """``execute`` refuses a result whose id/category drifted from the wiring."""

    context = DoctorCheckContext(project_root=_healthy_project(tmp_path))

    for definition in database_check_definitions():
        result = definition.execute(context)
        assert result.check_id == definition.check_id
        assert result.category is definition.category


def test_checks_share_one_snapshot_per_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Six checks must observe one snapshot, not six independent opens."""

    root = _healthy_project(tmp_path)
    opens: list[str] = []
    real_connect = sqlite3.connect

    def _counting_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        opens.append(str(args[0]))
        return real_connect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", _counting_connect)
    _run_all(root)

    assert len(opens) == 1


def test_retained_definitions_refresh_after_project_state_changes(tmp_path: Path) -> None:
    definitions = database_check_definitions()
    context = DoctorCheckContext(project_root=tmp_path)

    missing = {definition.check_id: definition.execute(context) for definition in definitions}
    init_project(tmp_path, project_name="cache-regression")
    initialized = {definition.check_id: definition.execute(context) for definition in definitions}

    assert missing["database.exists"].outcome is DoctorCheckOutcome.FAIL
    assert initialized["database.exists"].outcome is DoctorCheckOutcome.PASS
    assert initialized["database.integrity"].outcome is DoctorCheckOutcome.PASS


def test_execution_order_does_not_change_results(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    context = DoctorCheckContext(project_root=root)

    forward = database_check_definitions()
    backward = tuple(reversed(database_check_definitions()))

    ordered = {d.check_id: d.execute(context) for d in forward}
    reverse_ordered = {d.check_id: d.execute(context) for d in backward}

    assert ordered == reverse_ordered


# --------------------------------------------------------------------------- #
# Decision table: healthy database                                             #
# --------------------------------------------------------------------------- #
def test_healthy_database_passes_every_check(tmp_path: Path) -> None:
    results = _run_all(_healthy_project(tmp_path))

    assert set(results) == set(DATABASE_CHECK_IDS)
    for check_id, result in results.items():
        assert result.outcome is DoctorCheckOutcome.PASS, check_id
        assert result.severity is DoctorSeverity.INFO, check_id
        assert result.remediation is None, check_id


def test_healthy_database_report_is_healthy(tmp_path: Path) -> None:
    results = _run_all(_healthy_project(tmp_path))
    report = build_doctor_report(project=_PROJECT, checks=tuple(results.values()))

    assert report.overall_outcome is DoctorOverallOutcome.HEALTHY


def test_repeated_runs_render_identical_json(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)

    first = build_doctor_report(project=_PROJECT, checks=tuple(_run_all(root).values()))
    second = build_doctor_report(project=_PROJECT, checks=tuple(_run_all(root).values()))

    assert render_doctor_report_json(first) == render_doctor_report_json(second)


# --------------------------------------------------------------------------- #
# Decision table: path-level failures                                          #
# --------------------------------------------------------------------------- #
def _assert_only_exists_failed(results: dict[str, DoctorCheckResult]) -> None:
    assert results["database.exists"].outcome is DoctorCheckOutcome.FAIL
    assert results["database.exists"].severity is DoctorSeverity.ERROR
    assert results["database.exists"].remediation is not None
    for check_id in DATABASE_CHECK_IDS:
        if check_id == "database.exists":
            continue
        assert results[check_id].outcome is DoctorCheckOutcome.SKIPPED, check_id
        assert results[check_id].outcome is not DoctorCheckOutcome.PASS, check_id


def test_missing_database_fails_exists_and_skips_the_rest(tmp_path: Path) -> None:
    results = _run_all(tmp_path)

    _assert_only_exists_failed(results)
    assert not (tmp_path / PMEM_DIRNAME).exists()


def test_missing_project_creates_nothing(tmp_path: Path) -> None:
    before = sorted(path.name for path in tmp_path.iterdir())

    _run_all(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == before
    assert not (tmp_path / PMEM_DIRNAME).exists()


def test_directory_in_place_of_database_fails_closed(tmp_path: Path) -> None:
    project_database_path(tmp_path).mkdir(parents=True)

    results = _run_all(tmp_path)

    _assert_only_exists_failed(results)
    assert inspect_database(tmp_path).path_state is PathState.DIRECTORY


def test_symlinked_database_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-doctor.db"
    connection = sqlite3.connect(outside)
    connection.execute("CREATE TABLE t(id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    db_path = project_database_path(tmp_path)
    db_path.parent.mkdir(parents=True)
    try:
        os.symlink(outside, db_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")
    before = outside.read_bytes()

    results = _run_all(tmp_path)

    _assert_only_exists_failed(results)
    assert inspect_database(tmp_path).path_state is PathState.SYMLINK
    assert outside.read_bytes() == before
    assert db_path.is_symlink()


def test_symlinked_pmem_directory_is_not_followed(tmp_path: Path) -> None:
    outside_dir = tmp_path.parent / "outside-pmem"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "pmem.db").write_bytes(b"outside")
    try:
        os.symlink(outside_dir, tmp_path / PMEM_DIRNAME)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    results = _run_all(tmp_path)

    _assert_only_exists_failed(results)
    assert inspect_database(tmp_path).path_state is PathState.PARENT_SYMLINK
    assert (outside_dir / "pmem.db").read_bytes() == b"outside"


def test_symlinked_project_root_is_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside-project"
    outside.mkdir()
    _healthy_project(outside)
    linked_root = tmp_path / "linked-project"
    try:
        os.symlink(outside, linked_root)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    def _forbidden_connect(path: object) -> sqlite3.Connection:
        raise AssertionError("a symlinked root must be rejected before database access")

    monkeypatch.setattr(database_checks_module, "connect_database_readonly", _forbidden_connect)

    results = _run_all(linked_root)

    _assert_only_exists_failed(results)
    assert inspect_database(linked_root).path_state is PathState.ROOT_SYMLINK


# --------------------------------------------------------------------------- #
# Decision table: unreadable / corrupt content                                 #
# --------------------------------------------------------------------------- #
def test_random_bytes_fail_readable_not_exists(tmp_path: Path) -> None:
    """Opening succeeds on garbage; only the probe query reveals the truth."""

    db_path = project_database_path(tmp_path)
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(os.urandom(4096))

    results = _run_all(tmp_path)

    assert results["database.exists"].outcome is DoctorCheckOutcome.PASS
    assert results["database.readable"].outcome is DoctorCheckOutcome.FAIL
    assert results["database.readable"].severity is DoctorSeverity.ERROR
    for check_id in ("database.integrity", "database.foreign_keys"):
        assert results[check_id].outcome is DoctorCheckOutcome.SKIPPED, check_id


def test_empty_file_is_readable_but_has_no_migrations(tmp_path: Path) -> None:
    """A zero-byte file is a valid empty SQLite database, verified on disk."""

    db_path = project_database_path(tmp_path)
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"")

    results = _run_all(tmp_path)

    assert results["database.exists"].outcome is DoctorCheckOutcome.PASS
    assert results["database.readable"].outcome is DoctorCheckOutcome.PASS
    assert results["database.integrity"].outcome is DoctorCheckOutcome.PASS
    assert results["database.foreign_keys"].outcome is DoctorCheckOutcome.PASS
    assert results["database.migrations_current"].outcome is DoctorCheckOutcome.FAIL
    assert results["database.migrations_current"].severity is DoctorSeverity.ERROR
    # nothing recorded means there is no checksum to verify -- never a pass
    assert results["database.migration_checksums"].outcome is DoctorCheckOutcome.SKIPPED


def test_corrupted_pages_fail_integrity(tmp_path: Path) -> None:
    db_path = project_database_path(tmp_path)
    db_path.parent.mkdir(parents=True)
    _corrupt_pages(db_path)

    results = _run_all(tmp_path)

    assert results["database.integrity"].outcome is DoctorCheckOutcome.FAIL
    assert results["database.integrity"].severity is DoctorSeverity.ERROR
    assert results["database.integrity"].remediation is not None


def test_corrupt_database_report_is_unhealthy(tmp_path: Path) -> None:
    db_path = project_database_path(tmp_path)
    db_path.parent.mkdir(parents=True)
    _corrupt_pages(db_path)

    results = _run_all(tmp_path)
    report = build_doctor_report(project=None, checks=tuple(results.values()))

    assert report.overall_outcome is DoctorOverallOutcome.UNHEALTHY


# --------------------------------------------------------------------------- #
# Decision table: foreign keys, migrations, checksums                          #
# --------------------------------------------------------------------------- #
def test_foreign_key_violation_fails_with_error(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    _break_foreign_keys(project_database_path(root))

    results = _run_all(root)

    assert results["database.foreign_keys"].outcome is DoctorCheckOutcome.FAIL
    assert results["database.foreign_keys"].severity is DoctorSeverity.ERROR
    assert results["database.integrity"].outcome is DoctorCheckOutcome.PASS


def test_missing_migration_fails_migrations_current(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    _drop_a_migration(project_database_path(root))

    results = _run_all(root)

    assert results["database.migrations_current"].outcome is DoctorCheckOutcome.FAIL
    assert results["database.migrations_current"].severity is DoctorSeverity.ERROR
    # v1 is still recorded and still matches, so the checksum check is honest
    assert results["database.migration_checksums"].outcome is DoctorCheckOutcome.PASS


def test_missing_schema_migrations_table_fails_migrations_current(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    connection = sqlite3.connect(project_database_path(root))
    connection.execute("DROP TABLE schema_migrations")
    connection.commit()
    connection.close()

    results = _run_all(root)

    assert results["database.migrations_current"].outcome is DoctorCheckOutcome.FAIL
    assert results["database.migration_checksums"].outcome is DoctorCheckOutcome.SKIPPED


def test_tampered_checksum_fails_migration_checksums(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    _tamper_checksum(project_database_path(root))

    results = _run_all(root)

    assert results["database.migration_checksums"].outcome is DoctorCheckOutcome.FAIL
    assert results["database.migration_checksums"].severity is DoctorSeverity.ERROR
    assert results["database.migrations_current"].outcome is DoctorCheckOutcome.PASS


def test_only_unknown_migrations_cannot_produce_checksum_pass(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    connection = sqlite3.connect(project_database_path(root))
    connection.execute("DELETE FROM schema_migrations")
    connection.execute(
        "INSERT INTO schema_migrations(version, applied_at, checksum) VALUES (?, ?, ?)",
        ("999", "2026-01-01T00:00:00Z", "a" * 64),
    )
    connection.commit()
    connection.close()

    results = _run_all(root)

    assert results["database.migrations_current"].outcome is DoctorCheckOutcome.FAIL
    assert results["database.migration_checksums"].outcome is DoctorCheckOutcome.SKIPPED


# --------------------------------------------------------------------------- #
# Decision table: active SQLite sidecars                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal", "-mj0123ABCD"])
def test_active_sidecar_skips_readable_without_claiming_failure(
    tmp_path: Path, suffix: str
) -> None:
    root = _healthy_project(tmp_path)
    db_path = project_database_path(root)
    sidecar = db_path.with_name(db_path.name + suffix)
    sidecar.write_bytes(b"")

    results = _run_all(root)

    readable = results["database.readable"]
    assert readable.outcome is DoctorCheckOutcome.SKIPPED
    assert readable.severity is DoctorSeverity.WARNING
    assert readable.remediation is not None
    assert results["database.exists"].outcome is DoctorCheckOutcome.PASS
    for check_id in ("database.integrity", "database.foreign_keys"):
        assert results[check_id].outcome is DoctorCheckOutcome.SKIPPED, check_id
    assert sidecar.exists()  # never removed or checkpointed


def test_active_sidecar_report_is_incomplete_not_unhealthy(tmp_path: Path) -> None:
    root = _healthy_project(tmp_path)
    db_path = project_database_path(root)
    db_path.with_name(db_path.name + "-wal").write_bytes(b"")

    results = _run_all(root)
    report = build_doctor_report(project=_PROJECT, checks=tuple(results.values()))

    assert report.overall_outcome is DoctorOverallOutcome.INCOMPLETE


def test_sidecar_detector_agrees_with_the_connection_policy(tmp_path: Path) -> None:
    """The doctor must not fork the sidecar naming policy it relies on."""

    from pmem.errors import PmemPersistenceError
    from pmem.repositories.sqlite import connect_database_readonly

    root = _healthy_project(tmp_path)
    db_path = project_database_path(root)

    for suffix in ("-wal", "-shm", "-journal", "-mjXYZ"):
        sidecar = db_path.with_name(db_path.name + suffix)
        sidecar.write_bytes(b"")
        assert inspect_database(root).read_state is ReadState.SIDECAR_ACTIVE, suffix
        with pytest.raises(PmemPersistenceError):
            connect_database_readonly(db_path)
        sidecar.unlink()

    assert inspect_database(root).read_state is ReadState.OK


# --------------------------------------------------------------------------- #
# No downstream check may fake a pass                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "break_it",
    [
        pytest.param(lambda root: None, id="missing_database"),
        pytest.param(
            lambda root: (
                project_database_path(root).parent.mkdir(parents=True, exist_ok=True)
                or project_database_path(root).write_bytes(os.urandom(2048))
            ),
            id="random_bytes",
        ),
    ],
)
def test_a_broken_prerequisite_never_yields_a_downstream_pass(
    tmp_path: Path, break_it: Callable[[Path], None]
) -> None:
    break_it(tmp_path)

    results = _run_all(tmp_path)

    downstream = ("database.integrity", "database.foreign_keys", "database.migrations_current")
    for check_id in downstream:
        assert results[check_id].outcome is not DoctorCheckOutcome.PASS, check_id


def test_inspection_snapshot_is_immutable(tmp_path: Path) -> None:
    inspection = inspect_database(_healthy_project(tmp_path))

    assert isinstance(inspection, DatabaseInspection)
    assert inspection.read_state is ReadState.OK
    assert inspection.integrity_state is ProbeState.OK
    with pytest.raises(AttributeError):
        inspection.read_state = ReadState.UNREADABLE  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Defensive branches                                                           #
# --------------------------------------------------------------------------- #
def test_non_regular_file_in_place_of_database_fails_closed(tmp_path: Path) -> None:
    """A FIFO is neither a directory nor a symlink, but is still not a database."""

    db_path = project_database_path(tmp_path)
    db_path.parent.mkdir(parents=True)
    try:
        os.mkfifo(db_path)
    except (AttributeError, OSError, NotImplementedError):
        pytest.skip("named pipes are not supported on this platform")

    inspection = inspect_database(tmp_path)
    results = _run_all(tmp_path)

    assert inspection.path_state is PathState.NOT_REGULAR_FILE
    _assert_only_exists_failed(results)


def test_unstatable_path_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An OSError while classifying the path must not crash the diagnostic."""

    def _raise(self: Path) -> bool:
        raise OSError("classification failed")

    monkeypatch.setattr(Path, "is_symlink", _raise)

    inspection = inspect_database(tmp_path)

    assert inspection.path_state is PathState.NOT_REGULAR_FILE
    assert inspection.read_state is ReadState.NOT_INSPECTED


def test_unlistable_directory_is_not_misreported_as_an_active_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If sidecar state cannot be listed, fail closed rather than read anyway."""

    root = _healthy_project(tmp_path)

    def _raise(db_path: object) -> bool:
        raise OSError("cannot list directory")

    monkeypatch.setattr(database_checks_module, "has_active_sqlite_sidecars", _raise)

    inspection = inspect_database(root)

    assert inspection.read_state is ReadState.SIDECAR_STATE_UNREADABLE
    assert inspection.integrity_state is ProbeState.NOT_INSPECTED

    results = _run_all(root)
    readable = results["database.readable"]
    assert readable.outcome is DoctorCheckOutcome.FAIL
    assert "another projmem command" not in readable.message


def test_integrity_check_returning_a_non_ok_row_is_a_failure() -> None:
    """Cover the branch where SQLite *reports* damage instead of raising.

    The corruptions reproducible on this platform all make the pragma raise
    (see ``test_corrupted_pages_fail_integrity``, which uses a real file), so
    this narrow stub covers the other documented SQLite behaviour without
    replacing any real-database case.
    """

    class _ReportingConnection:
        def execute(self, sql: str) -> _ReportingConnection:
            assert sql == "PRAGMA integrity_check"
            return self

        def fetchall(self) -> list[tuple[str]]:
            return [("*** in database main ***",), ("row 1 missing from index ix",)]

    state = database_checks_module._integrity_state(_ReportingConnection())  # type: ignore[arg-type]

    assert state is ProbeState.FAILED


def test_integrity_failure_blocks_untrusted_downstream_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _healthy_project(tmp_path)
    monkeypatch.setattr(
        database_checks_module,
        "_integrity_state",
        lambda connection: ProbeState.FAILED,
    )

    results = _run_all(root)

    assert results["database.integrity"].outcome is DoctorCheckOutcome.FAIL
    for check_id in (
        "database.foreign_keys",
        "database.migrations_current",
        "database.migration_checksums",
    ):
        assert results[check_id].outcome is DoctorCheckOutcome.SKIPPED, check_id


def test_foreign_key_pragma_error_is_a_failure() -> None:
    """A pragma that raises must fail, never silently pass."""

    class _RaisingConnection:
        def execute(self, sql: str) -> object:
            raise sqlite3.DatabaseError("database disk image is malformed")

    state = database_checks_module._foreign_key_state(_RaisingConnection())  # type: ignore[arg-type]

    assert state is ProbeState.FAILED


def test_schema_inspection_failure_blocks_migration_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable migration table must skip, never pass, the schema checks."""

    from pmem.errors import PmemPersistenceError

    root = _healthy_project(tmp_path)

    def _raise(connection: object, *args: object, **kwargs: object) -> object:
        raise PmemPersistenceError("The project database could not be read.")

    monkeypatch.setattr(database_checks_module, "inspect_schema", _raise)

    results = _run_all(root)

    assert results["database.migrations_current"].outcome is DoctorCheckOutcome.SKIPPED
    assert results["database.migration_checksums"].outcome is DoctorCheckOutcome.SKIPPED
    assert results["database.integrity"].outcome is DoctorCheckOutcome.PASS
