"""Registry contract tests (DOC-001).

Every check used here is a pure fake: no database, no filesystem, no network.
The registry is only responsible for holding definitions and returning them in
a deterministic order, and these tests are what prove it stays that way.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from pmem.doctor import (
    DoctorCategory,
    DoctorCheckContext,
    DoctorCheckDefinition,
    DoctorCheckOutcome,
    DoctorCheckRegistry,
    DoctorCheckResult,
    DoctorSeverity,
)

_RAN: list[str] = []


def _fake_result(check_id: str, category: DoctorCategory) -> DoctorCheckResult:
    return DoctorCheckResult(
        check_id=check_id,
        category=category,
        outcome=DoctorCheckOutcome.PASS,
        severity=DoctorSeverity.INFO,
        message="A pure fake check used only to exercise the registry contract.",
        remediation=None,
        related_entity_id=None,
    )


def _definition(
    check_id: str,
    category: DoctorCategory = DoctorCategory.DATABASE,
) -> DoctorCheckDefinition:
    def _run(context: DoctorCheckContext) -> DoctorCheckResult:
        _RAN.append(check_id)
        assert isinstance(context.project_root, Path)
        return _fake_result(check_id, category)

    return DoctorCheckDefinition(check_id=check_id, category=category, run=_run)


# --------------------------------------------------------------------------- #
# Registration                                                                 #
# --------------------------------------------------------------------------- #
def test_register_a_single_check() -> None:
    registry = DoctorCheckRegistry()
    registry.register(_definition("database.exists"))

    assert len(registry) == 1
    assert "database.exists" in registry
    assert registry.registered_check_ids() == ("database.exists",)


def test_register_many_checks() -> None:
    registry = DoctorCheckRegistry(
        [
            _definition("database.exists"),
            _definition("permissions.private_directory", DoctorCategory.PERMISSIONS),
            _definition("tracked_paths.symlink", DoctorCategory.TRACKED_PATHS),
        ]
    )

    assert len(registry) == 3
    assert registry.registered_check_ids() == (
        "database.exists",
        "permissions.private_directory",
        "tracked_paths.symlink",
    )


def test_duplicate_check_id_is_rejected() -> None:
    registry = DoctorCheckRegistry([_definition("database.exists")])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_definition("database.exists"))


def test_duplicate_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="already registered"):
        DoctorCheckRegistry([_definition("database.exists"), _definition("database.exists")])


def test_registry_rejects_non_definition_values() -> None:
    registry = DoctorCheckRegistry()

    with pytest.raises(ValueError, match="DoctorCheckDefinition"):
        registry.register("database.exists")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Definition validation                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "check_id",
    ["database", "Database.exists", "database exists", "database.exists ", "/etc/passwd", ""],
)
def test_definition_rejects_a_malformed_check_id(check_id: str) -> None:
    with pytest.raises(ValueError):
        _definition(check_id)


def test_definition_rejects_a_category_mismatch() -> None:
    with pytest.raises(ValueError, match="must equal category"):
        _definition("database.exists", DoctorCategory.PERMISSIONS)


def test_definition_rejects_a_non_enum_category() -> None:
    with pytest.raises(ValueError, match="DoctorCategory member"):
        DoctorCheckDefinition(
            check_id="database.exists",
            category="database",  # type: ignore[arg-type]
            run=lambda context: _fake_result("database.exists", DoctorCategory.DATABASE),
        )


def test_definition_rejects_a_non_callable_run() -> None:
    with pytest.raises(ValueError, match="callable"):
        DoctorCheckDefinition(
            check_id="database.exists",
            category=DoctorCategory.DATABASE,
            run="not-callable",  # type: ignore[arg-type]
        )


def test_definition_is_immutable() -> None:
    definition = _definition("database.exists")

    with pytest.raises(AttributeError):
        definition.check_id = "database.other"  # type: ignore[misc]


def test_context_is_immutable() -> None:
    context = DoctorCheckContext(project_root=Path("."))

    with pytest.raises(AttributeError):
        context.project_root = Path("/tmp")  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Deterministic ordering                                                       #
# --------------------------------------------------------------------------- #
def test_registration_order_does_not_affect_returned_order() -> None:
    ids = ("tracked_paths.symlink", "database.exists", "permissions.private_directory")
    categories = {
        "tracked_paths.symlink": DoctorCategory.TRACKED_PATHS,
        "database.exists": DoctorCategory.DATABASE,
        "permissions.private_directory": DoctorCategory.PERMISSIONS,
    }

    orders = {
        DoctorCheckRegistry(
            [_definition(check_id, categories[check_id]) for check_id in permutation]
        ).registered_check_ids()
        for permutation in itertools.permutations(ids)
    }

    assert orders == {tuple(sorted(ids))}


def test_ordering_is_stable_across_repeated_calls() -> None:
    registry = DoctorCheckRegistry(
        [_definition("database.integrity"), _definition("database.exists")]
    )

    assert registry.registered_checks() == registry.registered_checks()
    assert registry.registered_check_ids() == ("database.exists", "database.integrity")


def test_registered_checks_returns_an_immutable_tuple() -> None:
    registry = DoctorCheckRegistry([_definition("database.exists")])
    checks = registry.registered_checks()

    assert isinstance(checks, tuple)
    with pytest.raises(AttributeError):
        checks.append(checks[0])  # type: ignore[attr-defined]


def test_returned_tuple_cannot_mutate_the_registry() -> None:
    registry = DoctorCheckRegistry([_definition("database.exists")])
    first = registry.registered_checks()

    assert first is not registry.registered_checks() or len(registry) == 1
    assert len(registry) == 1


# --------------------------------------------------------------------------- #
# Inertness: no side effects, no repair hook                                   #
# --------------------------------------------------------------------------- #
def test_registration_does_not_run_the_check() -> None:
    _RAN.clear()
    registry = DoctorCheckRegistry([_definition("database.exists")])
    registry.registered_checks()

    assert _RAN == []


def test_a_check_only_runs_when_explicitly_invoked() -> None:
    _RAN.clear()
    registry = DoctorCheckRegistry([_definition("database.exists")])
    definition = registry.registered_checks()[0]

    result = definition.execute(DoctorCheckContext(project_root=Path(".")))

    assert _RAN == ["database.exists"]
    assert result.check_id == "database.exists"


def test_execute_rejects_a_non_result() -> None:
    definition = DoctorCheckDefinition(
        check_id="database.exists",
        category=DoctorCategory.DATABASE,
        run=lambda context: "not-a-result",  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(ValueError, match="must return a DoctorCheckResult"):
        definition.execute(DoctorCheckContext(project_root=Path(".")))


def test_execute_rejects_a_result_with_another_check_id() -> None:
    definition = DoctorCheckDefinition(
        check_id="database.exists",
        category=DoctorCategory.DATABASE,
        run=lambda context: _fake_result("database.integrity", DoctorCategory.DATABASE),
    )

    with pytest.raises(ValueError, match="check_id must match"):
        definition.execute(DoctorCheckContext(project_root=Path(".")))


def test_execute_rejects_a_result_with_another_category() -> None:
    mismatched = _fake_result("database.exists", DoctorCategory.DATABASE).model_copy(
        update={"category": DoctorCategory.PERMISSIONS}
    )
    definition = DoctorCheckDefinition(
        check_id="database.exists",
        category=DoctorCategory.DATABASE,
        run=lambda context: mismatched,
    )

    with pytest.raises(ValueError, match="category must match"):
        definition.execute(DoctorCheckContext(project_root=Path(".")))


def test_execute_does_not_hide_an_unexpected_exception() -> None:
    def _raise(context: DoctorCheckContext) -> DoctorCheckResult:
        raise RuntimeError("producer bug")

    definition = DoctorCheckDefinition(
        check_id="database.exists",
        category=DoctorCategory.DATABASE,
        run=_raise,
    )

    with pytest.raises(RuntimeError, match="producer bug"):
        definition.execute(DoctorCheckContext(project_root=Path(".")))


def test_registry_exposes_no_repair_hook() -> None:
    registry = DoctorCheckRegistry([_definition("database.exists")])
    definition = registry.registered_checks()[0]

    for forbidden in ("fix", "repair", "apply", "remediate", "heal", "autofix"):
        assert not hasattr(registry, forbidden), forbidden
        assert not hasattr(definition, forbidden), forbidden


def test_importing_the_package_touches_nothing(tmp_path: Path) -> None:
    """A fresh interpreter importing ``pmem.doctor`` must create no file at all."""

    src = Path(__file__).resolve().parents[3] / "src"
    before = sorted(path.name for path in tmp_path.iterdir())

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pmem.doctor as d; assert d.DOCTOR_SCHEMA_VERSION == 'doctor-v1'",
        ],
        cwd=tmp_path,
        env={"PYTHONPATH": str(src), "PATH": "", "SYSTEMROOT": ""},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == before
    assert not (tmp_path / ".pmem").exists()


def test_importing_the_package_opens_no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sqlite3

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("importing pmem.doctor must not open a database")

    monkeypatch.setattr(sqlite3, "connect", _forbidden)
    for name in [key for key in list(sys.modules) if key.startswith("pmem.doctor")]:
        del sys.modules[name]

    module = importlib.import_module("pmem.doctor")

    assert module.DOCTOR_SCHEMA_VERSION == "doctor-v1"
