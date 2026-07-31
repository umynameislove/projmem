"""Typed, deterministic registry of doctor checks (DOC-001).

The registry is the seam DOC-002..DOC-005 register their checks through, and
DOC-006 iterates to build a report. It is deliberately inert:

- importing this module registers nothing, opens no database, reads no file and
  runs no check;
- there is no module-level singleton and no decorator, so a test can build an
  isolated registry and no import order can mutate shared state;
- registration is explicit and validated, so a duplicate or malformed
  ``check_id`` fails at registration time rather than producing a broken report;
- :meth:`DoctorCheckRegistry.registered_checks` returns an immutable tuple in
  canonical ``check_id`` order, so registration order cannot influence output.

Repair is out of scope by construction. Neither the definition nor the registry
exposes a fix/repair/apply hook: a check may only *describe* what it found.
Confirmed repair is designed separately in DOC-007.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pmem.doctor.model import DoctorCategory, DoctorCheckResult, validate_check_id


@dataclass(frozen=True, slots=True)
class DoctorCheckContext:
    """Everything a check is allowed to know about the project.

    Only the project root is provided. A check that needs project data must go
    through the existing read-only seam (``connect_database_readonly``,
    ``require_project_context_readonly``) so the read-only guarantee proved for
    ``pmem status`` extends to ``pmem doctor``. The context itself performs no
    I/O and is frozen, so a check cannot smuggle state to the next check
    through it.
    """

    project_root: Path


class DoctorCheck(Protocol):
    """Callable interface every doctor check implements.

    A check inspects through approved read-only seams, then returns exactly one
    :class:`DoctorCheckResult`. It must not mutate the project, and it must not
    raise for an expected diagnostic condition -- a broken database is a
    ``fail`` result, not an exception.
    """

    def __call__(self, context: DoctorCheckContext) -> DoctorCheckResult: ...


@dataclass(frozen=True, slots=True)
class DoctorCheckDefinition:
    """Immutable registration record for one check.

    ``check_id`` and ``category`` are validated here rather than only on the
    produced result, so a misnamed check is rejected at wiring time instead of
    surfacing later as a confusing report.
    """

    check_id: str
    category: DoctorCategory
    run: DoctorCheck

    def __post_init__(self) -> None:
        cleaned = validate_check_id(self.check_id)
        if not isinstance(self.category, DoctorCategory):
            msg = "doctor check category must be a DoctorCategory member"
            raise ValueError(msg)
        namespace = cleaned.split(".", 1)[0]
        if namespace != self.category.value:
            msg = f"check_id namespace '{namespace}' must equal category '{self.category.value}'"
            raise ValueError(msg)
        if not callable(self.run):
            msg = "doctor check definition requires a callable run function"
            raise ValueError(msg)

    def execute(self, context: DoctorCheckContext) -> DoctorCheckResult:
        """Run the check and prove the result belongs to this definition.

        Expected diagnostic conditions belong in a typed result. Unexpected
        callable exceptions are deliberately left for the future service layer
        to map onto a safe internal-error result without leaking raw details.
        """

        result = self.run(context)
        if not isinstance(result, DoctorCheckResult):
            msg = "doctor check must return a DoctorCheckResult"
            raise ValueError(msg)
        if result.check_id != self.check_id:
            msg = "doctor check result check_id must match its registered definition"
            raise ValueError(msg)
        if result.category is not self.category:
            msg = "doctor check result category must match its registered definition"
            raise ValueError(msg)
        return result


class DoctorCheckRegistry:
    """Explicit, order-independent collection of check definitions."""

    __slots__ = ("_definitions",)

    def __init__(self, definitions: Iterable[DoctorCheckDefinition] = ()) -> None:
        self._definitions: dict[str, DoctorCheckDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: DoctorCheckDefinition) -> None:
        """Add one definition. Rejects a duplicate ``check_id``."""

        if not isinstance(definition, DoctorCheckDefinition):
            msg = "doctor registry accepts DoctorCheckDefinition values only"
            raise ValueError(msg)
        if definition.check_id in self._definitions:
            msg = f"doctor check_id '{definition.check_id}' is already registered"
            raise ValueError(msg)
        self._definitions[definition.check_id] = definition

    def registered_checks(self) -> tuple[DoctorCheckDefinition, ...]:
        """Return every definition in canonical ascending ``check_id`` order.

        ``check_id`` is unique, so the sort is total and needs no tie-break;
        the result never depends on registration order or on dict iteration
        order.
        """

        return tuple(self._definitions[check_id] for check_id in sorted(self._definitions))

    def registered_check_ids(self) -> tuple[str, ...]:
        """Return the registered ids in the same canonical order."""

        return tuple(sorted(self._definitions))

    def __contains__(self, check_id: object) -> bool:
        return check_id in self._definitions

    def __len__(self) -> int:
        return len(self._definitions)
