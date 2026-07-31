"""doctor contract ``doctor-v1`` payload models (DOC-001).

This module is a **pure schema contract** for ``pmem doctor``. It defines a
versioned, strict, typed and privacy-safe report so later tasks build on a
stable shape:

- DOC-002..DOC-005 implement the real database/permission/tracked-path/evidence
  checks and produce :class:`DoctorCheckResult` values.
- DOC-006 assembles the service and the CLI.
- DOC-007 designs the confirmed repair mode.

Scope of the safety guarantees (kept honest): the model enforces **shape**,
strict typing, closed vocabularies, cross-field consistency, safe identifiers,
absolute-path/control-character rejection, and length limits. It **cannot**
prove that arbitrary human-authored text is free of secrets.

Producer obligations that the contract cannot verify, and that DOC-002..DOC-005
must therefore honour in every check they write:

- never place raw SQL, a query plan, or a table/column dump into ``message`` or
  ``remediation``;
- never place a traceback, an ``Exception`` repr, or a raw driver error string
  into any field -- map failures onto a stable ``check_id`` and a
  human-written message instead;
- never place raw config values, secret-like config keys, or raw
  failure/note/decision bodies into the report;
- never place a filesystem path (absolute *or* project-relative) into the
  report -- the absolute-path validator here rejects the obvious cases, but a
  relative path such as ``src/train.py`` is still project content and must not
  be emitted;
- keep ``remediation`` a normalized, human-written instruction or a controlled
  ``pmem`` command, never an interpolated shell string.

Determinism: given the same validated report, serialization is byte-stable. The
contract carries no timestamp, duration, generated report id, or other volatile
metadata. Producers remain responsible for deriving stable messages and results
from the same project state; stable project/entity identifiers are allowed.

Read-only: nothing in this module opens SQLite, touches the filesystem, or
performs I/O of any kind. Importing it is side-effect free.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pmem.domain.common import PmemStrEnum
from pmem.status.textsafety import contains_absolute_path, contains_control_chars

_EnumT = TypeVar("_EnumT", bound=Enum)

DOCTOR_SCHEMA_VERSION = "doctor-v1"

# Strict + closed + immutable everywhere, matching the ``status-v1`` house
# style. ``strict=True`` blocks silent coercion such as ``"1" -> 1`` and
# ``True -> 1``; ``extra="forbid"`` blocks unknown keys. No field carries a
# default, so a producer that forgets a field (including the schema version or
# a safety flag) fails validation instead of silently defaulting. Nullable data
# must be sent explicitly as ``null``.
_MODEL_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)

# Length limits (local to the doctor package; no public constants are shared
# with ``status-v1`` so the two contracts can evolve independently).
_MAX_CHECK_ID_LENGTH = 64
_MAX_IDENTIFIER_LENGTH = 128
_MAX_PROJECT_NAME_LENGTH = 120
_MAX_MESSAGE_LENGTH = 512
_MAX_REMEDIATION_LENGTH = 256
_MAX_CHECKS = 200

# A ``check_id`` is a dotted namespace: ``<category>.<name>[.<name>...]``.
# Requiring at least two segments forces every check to declare its family, and
# the character class excludes whitespace, path separators, shell syntax and
# every control character by construction.
#
# Both patterns anchor with ``\Z`` rather than ``$``. In Python ``$`` also
# matches immediately before a trailing newline, so ``"database.exists\n"``
# would satisfy a ``$``-anchored pattern and smuggle a line break into an
# identifier that later renders into text output.
_CHECK_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\Z")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")


# --------------------------------------------------------------------------- #
# Reusable validation helpers                                                  #
# --------------------------------------------------------------------------- #
def _reject_unsafe_text(value: str) -> None:
    """Reject control characters (this covers ANSI ``ESC``) and absolute paths.

    The detectors are imported from :mod:`pmem.status.textsafety`, a leaf module
    with no dependencies of its own, so the doctor and status contracts cannot
    drift apart on path policy and the regex is never duplicated.
    """

    if contains_control_chars(value):
        msg = "doctor text fields must not contain control characters"
        raise ValueError(msg)
    if contains_absolute_path(value):
        msg = "doctor text fields must not contain an absolute filesystem path"
        raise ValueError(msg)


def _clean_text(value: str, *, max_length: int) -> str:
    """Return a stripped, non-blank, control-free, path-free, bounded string."""

    cleaned = value.strip()
    if not cleaned:
        msg = "doctor required text fields must not be blank"
        raise ValueError(msg)
    if len(cleaned) > max_length:
        msg = f"doctor text exceeds the maximum length of {max_length}"
        raise ValueError(msg)
    _reject_unsafe_text(cleaned)
    return cleaned


def _clean_optional_text(value: str | None, *, max_length: int) -> str | None:
    if value is None:
        return None
    return _clean_text(value, max_length=max_length)


def validate_check_id(value: str) -> str:
    """Return ``value`` unchanged if it is a well-formed ``check_id``, else raise.

    Shared internally because the registry validates ids at registration time
    using the same rule the contract enforces at serialization time; a second
    copy of the pattern would be free to drift. It is intentionally not
    re-exported from :mod:`pmem.doctor`.

    Unlike the free-text fields, this deliberately does **not** strip
    surrounding whitespace. A ``check_id`` is a stable machine key that appears
    in registries, reports and user scripts; normalizing it silently would let
    ``"database.exists"`` and ``"database.exists "`` register as one id while
    reading as two, which would quietly undermine the duplicate-id guarantee.
    Malformed input is rejected rather than repaired.
    """

    if len(value) > _MAX_CHECK_ID_LENGTH:
        msg = f"doctor check_id exceeds the maximum length of {_MAX_CHECK_ID_LENGTH}"
        raise ValueError(msg)
    if not _CHECK_ID_RE.match(value):
        msg = "doctor check_id must match ^[a-z][a-z0-9_]*(\\.[a-z][a-z0-9_]*)+\\Z"
        raise ValueError(msg)
    return value


def _clean_identifier(value: str, *, max_length: int = _MAX_IDENTIFIER_LENGTH) -> str:
    cleaned = value.strip()
    if not _SAFE_IDENTIFIER_RE.match(cleaned):
        msg = "doctor identifiers must match ^[A-Za-z0-9][A-Za-z0-9_.:-]*\\Z"
        raise ValueError(msg)
    if len(cleaned) > max_length:
        msg = f"doctor identifiers exceed the maximum length of {max_length}"
        raise ValueError(msg)
    return cleaned


def _clean_optional_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    return _clean_identifier(value)


def _coerce_enum_value(enum_cls: type[_EnumT], value: Any) -> _EnumT:
    """Accept an enum member or its canonical string value; reject anything else."""

    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str) and not isinstance(value, bool):
        try:
            return enum_cls(value)
        except ValueError as exc:
            msg = f"invalid {enum_cls.__name__} value"
            raise ValueError(msg) from exc
    msg = f"{enum_cls.__name__} must be provided as its string value"
    raise ValueError(msg)


# --------------------------------------------------------------------------- #
# Vocabularies                                                                 #
# --------------------------------------------------------------------------- #
class DoctorCategory(PmemStrEnum):
    """Closed check-family vocabulary; the first ``check_id`` segment must match.

    The families mirror the diagnostics recorded for ``pmem doctor`` in the
    product plan: database integrity, filesystem permissions, tracked-path
    state, evidence freshness, and the surrounding environment (for example Git
    availability). Implementations land in DOC-002..DOC-005.
    """

    DATABASE = "database"
    PERMISSIONS = "permissions"
    TRACKED_PATHS = "tracked_paths"
    EVIDENCE = "evidence"
    ENVIRONMENT = "environment"


class DoctorCheckOutcome(PmemStrEnum):
    """What happened when the check ran.

    Deliberately separate from :class:`DoctorSeverity`: this answers *"did the
    check run and did it pass"*, never *"how much does it matter"*.
    """

    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"


class DoctorSeverity(PmemStrEnum):
    """How much attention **this result** demands.

    Note the semantics carefully: severity qualifies the result that was
    actually produced, not the hypothetical impact of a failure that did not
    happen. That is what makes the report skimmable -- a reader can trust that
    nothing carrying ``error`` is a passing check -- and it is what the
    aggregation rules in :func:`resolve_overall_outcome` rely on.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DoctorOverallOutcome(PmemStrEnum):
    """Report-level conclusion derived from the individual check results."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    INCOMPLETE = "incomplete"


# --------------------------------------------------------------------------- #
# Nested models                                                                #
# --------------------------------------------------------------------------- #
class DoctorProject(BaseModel):
    """Minimal, safe project identity.

    Intentionally smaller than ``status-v1``'s project block: doctor reports on
    health, not on goals, so no objective or metric is carried here.
    """

    model_config = _MODEL_CONFIG

    project_id: str
    project_name: str

    @field_validator("project_id")
    @classmethod
    def _validate_project_id(cls, value: str) -> str:
        return _clean_identifier(value)

    @field_validator("project_name")
    @classmethod
    def _validate_project_name(cls, value: str) -> str:
        return _clean_text(value, max_length=_MAX_PROJECT_NAME_LENGTH)


class DoctorCheckResult(BaseModel):
    """One diagnostic result.

    Outcome/severity/remediation are locked together so a report cannot claim
    something incoherent, such as a passing check that demands remediation or a
    failure with no way to act on it.
    """

    model_config = _MODEL_CONFIG

    check_id: str
    category: DoctorCategory
    outcome: DoctorCheckOutcome
    severity: DoctorSeverity
    message: str
    remediation: str | None
    related_entity_id: str | None

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value: Any) -> Any:
        return _coerce_enum_value(DoctorCategory, value)

    @field_validator("outcome", mode="before")
    @classmethod
    def _coerce_outcome(cls, value: Any) -> Any:
        return _coerce_enum_value(DoctorCheckOutcome, value)

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, value: Any) -> Any:
        return _coerce_enum_value(DoctorSeverity, value)

    @field_validator("check_id")
    @classmethod
    def _validate_check_id(cls, value: str) -> str:
        return validate_check_id(value)

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        return _clean_text(value, max_length=_MAX_MESSAGE_LENGTH)

    @field_validator("remediation")
    @classmethod
    def _validate_remediation(cls, value: str | None) -> str | None:
        return _clean_optional_text(value, max_length=_MAX_REMEDIATION_LENGTH)

    @field_validator("related_entity_id")
    @classmethod
    def _validate_related_entity_id(cls, value: str | None) -> str | None:
        return _clean_optional_identifier(value)

    @model_validator(mode="after")
    def _validate_consistency(self) -> DoctorCheckResult:
        namespace = self.check_id.split(".", 1)[0]
        if namespace != self.category.value:
            msg = f"check_id namespace '{namespace}' must equal category '{self.category.value}'"
            raise ValueError(msg)

        if self.outcome is DoctorCheckOutcome.PASS:
            if self.severity is not DoctorSeverity.INFO:
                msg = "a passing check must carry severity 'info'"
                raise ValueError(msg)
            if self.remediation is not None:
                msg = "a passing check must not carry remediation"
                raise ValueError(msg)
        elif self.outcome is DoctorCheckOutcome.FAIL:
            if self.severity is DoctorSeverity.INFO:
                msg = "a failing check must carry severity 'warning' or 'error'"
                raise ValueError(msg)
            if self.remediation is None:
                msg = "a failing check must carry remediation"
                raise ValueError(msg)
        elif self.outcome is DoctorCheckOutcome.SKIPPED:
            if self.severity is DoctorSeverity.ERROR:
                msg = "a skipped check is a coverage gap and must not carry severity 'error'"
                raise ValueError(msg)
        else:  # NOT_APPLICABLE
            if self.severity is not DoctorSeverity.INFO:
                msg = "a not-applicable check must carry severity 'info'"
                raise ValueError(msg)
            if self.remediation is not None:
                msg = "a not-applicable check must not carry remediation"
                raise ValueError(msg)
        return self


class DoctorSummary(BaseModel):
    """Counts derived from ``checks``; the root validator proves they match."""

    model_config = _MODEL_CONFIG

    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    not_applicable: int = Field(ge=0)
    info: int = Field(ge=0)
    warning: int = Field(ge=0)
    error: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_internal_totals(self) -> DoctorSummary:
        if self.passed + self.failed + self.skipped + self.not_applicable != self.total:
            msg = "summary outcome counts must sum to total"
            raise ValueError(msg)
        if self.info + self.warning + self.error != self.total:
            msg = "summary severity counts must sum to total"
            raise ValueError(msg)
        return self


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #
def summarize_checks(checks: tuple[DoctorCheckResult, ...]) -> DoctorSummary:
    """Return the exact counts implied by ``checks``. Pure and total."""

    outcomes = [check.outcome for check in checks]
    severities = [check.severity for check in checks]
    return DoctorSummary(
        total=len(checks),
        passed=outcomes.count(DoctorCheckOutcome.PASS),
        failed=outcomes.count(DoctorCheckOutcome.FAIL),
        skipped=outcomes.count(DoctorCheckOutcome.SKIPPED),
        not_applicable=outcomes.count(DoctorCheckOutcome.NOT_APPLICABLE),
        info=severities.count(DoctorSeverity.INFO),
        warning=severities.count(DoctorSeverity.WARNING),
        error=severities.count(DoctorSeverity.ERROR),
    )


def resolve_overall_outcome(checks: tuple[DoctorCheckResult, ...]) -> DoctorOverallOutcome:
    """Return the single report-level conclusion. Pure, total and deterministic.

    Decision table (first matching row wins; ``*`` means "any"):

    ==================  ====================  ===========  ==========  ==============
    any fail + error    any fail + warning    any skipped  applicable  overall
    ==================  ====================  ===========  ==========  ==============
    yes                 *                     *            *           ``unhealthy``
    no                  yes                   *            *           ``degraded``
    no                  no                    yes          *           ``incomplete``
    no                  no                    no           no          ``incomplete``
    no                  no                    no           yes         ``healthy``
    ==================  ====================  ===========  ==========  ==============

    Two precedence choices are deliberate and are locked by tests:

    - A definite failure outranks unknown coverage, so ``degraded`` beats
      ``incomplete``. A problem the user can act on now is more useful than a
      report that leads with "some checks did not run".
    - ``not_applicable`` is neutral when at least one check ran. It is distinct
      from ``skipped`` so a platform-specific check does not make a supported
      healthy conclusion incomplete. If every check is not applicable, however,
      the report has established nothing and is ``incomplete``.
    - ``healthy`` is only reachable when every applicable check ran and passed.

    ``fail`` + ``info`` cannot occur: :class:`DoctorCheckResult` rejects it.
    """

    failures = [check for check in checks if check.outcome is DoctorCheckOutcome.FAIL]
    if any(check.severity is DoctorSeverity.ERROR for check in failures):
        return DoctorOverallOutcome.UNHEALTHY
    if failures:
        return DoctorOverallOutcome.DEGRADED
    if any(check.outcome is DoctorCheckOutcome.SKIPPED for check in checks):
        return DoctorOverallOutcome.INCOMPLETE
    if not any(
        check.outcome in (DoctorCheckOutcome.PASS, DoctorCheckOutcome.FAIL) for check in checks
    ):
        return DoctorOverallOutcome.INCOMPLETE
    return DoctorOverallOutcome.HEALTHY


def canonical_check_order(
    checks: tuple[DoctorCheckResult, ...],
) -> tuple[DoctorCheckResult, ...]:
    """Return ``checks`` in canonical order: ascending ``check_id``.

    ``check_id`` is unique within a report, so this sort is total and needs no
    tie-break. Producer/registration order therefore cannot influence the
    serialized report.
    """

    return tuple(sorted(checks, key=lambda check: check.check_id))


# --------------------------------------------------------------------------- #
# Root report                                                                  #
# --------------------------------------------------------------------------- #
class DoctorReport(BaseModel):
    """Root ``doctor-v1`` report.

    Every field is required (no defaults): a producer that omits the schema
    version, a safety flag or a nullable field fails validation. ``checks`` is
    stored as a tuple so the validated report is deeply immutable while still
    serialising to a JSON array.

    The report is self-proving: ``summary`` and ``overall_outcome`` are
    recomputed from ``checks`` during validation and a mismatch is rejected, so
    a hand-written or tampered report cannot understate its own findings.
    """

    model_config = _MODEL_CONFIG

    schema_version: Literal["doctor-v1"]
    project: DoctorProject | None
    overall_outcome: DoctorOverallOutcome
    summary: DoctorSummary
    checks: tuple[DoctorCheckResult, ...] = Field(min_length=1, max_length=_MAX_CHECKS)
    database_mutation: Literal[False]
    network: Literal[False]
    automatic_repair: Literal[False]
    raw_text_in_output: Literal[False]

    @field_validator("overall_outcome", mode="before")
    @classmethod
    def _coerce_overall_outcome(cls, value: Any) -> Any:
        return _coerce_enum_value(DoctorOverallOutcome, value)

    @field_validator("checks", mode="before")
    @classmethod
    def _coerce_checks(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_report_consistency(self) -> DoctorReport:
        check_ids = [check.check_id for check in self.checks]
        if len(set(check_ids)) != len(check_ids):
            msg = "doctor report must not contain duplicate check_id values"
            raise ValueError(msg)
        if check_ids != sorted(check_ids):
            msg = "doctor checks must be serialized in ascending check_id order"
            raise ValueError(msg)

        expected_summary = summarize_checks(self.checks)
        if self.summary != expected_summary:
            msg = "doctor summary counts must match the checks they describe"
            raise ValueError(msg)

        expected_outcome = resolve_overall_outcome(self.checks)
        if self.overall_outcome is not expected_outcome:
            msg = f"overall_outcome must be '{expected_outcome.value}' for the given check results"
            raise ValueError(msg)

        if self.project is None and self.overall_outcome is DoctorOverallOutcome.HEALTHY:
            msg = "a report without project identity cannot conclude 'healthy'"
            raise ValueError(msg)
        return self


def build_doctor_report(
    *,
    project: DoctorProject | None,
    checks: tuple[DoctorCheckResult, ...],
) -> DoctorReport:
    """Assemble a validated ``doctor-v1`` report from raw check results.

    The single supported way to produce a report. Checks are placed in
    canonical order and the summary and overall outcome are derived, so callers
    cannot introduce an ordering or an aggregation that disagrees with the
    results, and two callers that collected the same results in different
    orders produce byte-identical reports.
    """

    ordered = canonical_check_order(checks)
    return DoctorReport(
        schema_version=DOCTOR_SCHEMA_VERSION,
        project=project,
        overall_outcome=resolve_overall_outcome(ordered),
        summary=summarize_checks(ordered),
        checks=ordered,
        database_mutation=False,
        network=False,
        automatic_repair=False,
        raw_text_in_output=False,
    )


def render_doctor_report_json(report: DoctorReport) -> str:
    """Serialize a validated report to deterministic ``doctor-v1`` JSON text.

    Deterministic, not canonical: no JSON canonicalization standard (JCS or
    similar) is implemented. What is guaranteed is that Pydantic emits fields in
    declaration order with stable 2-space indentation, that the contract carries
    no runtime-dependent value, and that the caller owns the trailing newline.
    """

    return report.model_dump_json(indent=2)
