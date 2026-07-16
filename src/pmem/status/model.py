"""status contract ``status-v1`` payload models.

This module is a **pure schema contract** for ``pmem status`` (micro-task
STS-001). It defines a versioned, strict, typed and privacy-safe JSON payload
so later tasks build on a stable shape:

- STS-002 assembles the payload from project state.
- STS-003 owns the next-action ordering rules.
- STS-004/005 render CLI text / JSON.
- STS-006 adds integration tests.

The contract deliberately does **not** read SQLite, build the graph, generate
recommendations, or expose any CLI. It only validates and serialises a payload.

Scope of the safety guarantees (kept honest): the model enforces **shape**,
strict typing, closed vocabularies, cross-field consistency, safe identifiers,
absolute-path/control-character rejection, and length limits. It **cannot**
prove that arbitrary human-authored text is free of secrets; producers must not
place raw failure/decision/note text into this payload.

Design FACTs mirrored here:

- ``target_status`` vocabulary and its logic match
  ``summary/project_summary.py::_target_status``.
- The graph today only reports ``exists`` (``services/graph_operations.py``);
  the closed missing/current/stale/invalid/unknown vocabulary is defined here
  but detection is left to STS-002+.
- Recommendations are generated on demand with **no** persisted lifecycle
  (no ``recommendations`` table), so ``active_count`` is only meaningful with a
  persisted lifecycle.
- Golden identifiers use the real formats ``proj_<hex>`` / ``run_<hex>``
  (``services/project_init.py``, ``services/run_capture.py``).
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pmem.domain.common import MetricDirection, PmemStrEnum

_EnumT = TypeVar("_EnumT", bound=Enum)

STATUS_SCHEMA_VERSION = "status-v1"

# Strict + closed + immutable everywhere. ``strict=True`` blocks silent coercion
# such as ``"1" -> 1`` and ``True -> 1``; ``extra="forbid"`` blocks unknown keys.
# No field carries a default, so a producer that forgets a field (including the
# schema version or a privacy flag) fails validation instead of silently
# defaulting. Nullable data must be sent explicitly as ``null``.
_MODEL_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)

# Length limits (local to the status package; no public constants are changed).
_MAX_IDENTIFIER_LENGTH = 128
_MAX_CODE_LENGTH = 64
_MAX_METRIC_NAME_LENGTH = 512
_MAX_PROJECT_NAME_LENGTH = 120
_MAX_OBJECTIVE_LENGTH = 512
_MAX_MESSAGE_LENGTH = 512
_MAX_REASON_LENGTH = 512
_MAX_REMEDIATION_LENGTH = 256
_MAX_COMMAND_LENGTH = 256
_MAX_WARNINGS = 100

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_ABS_PATH_SPLIT_RE = re.compile(r"[\s()\[\]{}\"'<>,=]+")
_SHELL_METACHARACTERS: tuple[str, ...] = (
    "&&",
    "||",
    ";",
    "|",
    "`",
    "$(",
    "${",
    "&",
    ">",
    "<",
    "\n",
    "\r",
)
_SUGGESTED_COMMAND_PREFIX = "pmem "


# --------------------------------------------------------------------------- #
# Reusable validation helpers                                                  #
# --------------------------------------------------------------------------- #
def _reject_control_chars(value: str) -> None:
    if any(ord(char) < 32 for char in value):
        msg = "status text fields must not contain control characters"
        raise ValueError(msg)


def _reject_absolute_path(value: str) -> None:
    """Reject absolute paths, including when wrapped in quotes/parens or a URL."""

    if "file://" in value.lower():
        msg = "status text fields must not contain a file:// path"
        raise ValueError(msg)
    for raw_token in _ABS_PATH_SPLIT_RE.split(value):
        token = raw_token.strip()
        if not token:
            continue
        normalized = token.replace("\\", "/")
        if normalized.startswith("/"):
            msg = "status text fields must not contain an absolute filesystem path"
            raise ValueError(msg)
        if _WINDOWS_DRIVE_RE.match(token) or _WINDOWS_DRIVE_RE.match(normalized):
            msg = "status text fields must not contain an absolute filesystem path"
            raise ValueError(msg)


def _clean_text(value: str, *, max_length: int) -> str:
    """Return a stripped, non-blank, control-free, path-free, bounded string."""

    cleaned = value.strip()
    if not cleaned:
        msg = "status required text fields must not be blank"
        raise ValueError(msg)
    if len(cleaned) > max_length:
        msg = f"status text exceeds the maximum length of {max_length}"
        raise ValueError(msg)
    _reject_control_chars(cleaned)
    _reject_absolute_path(cleaned)
    return cleaned


def _clean_optional_text(value: str | None, *, max_length: int) -> str | None:
    if value is None:
        return None
    return _clean_text(value, max_length=max_length)


def _clean_code(value: str) -> str:
    cleaned = value.strip()
    if not _SAFE_CODE_RE.match(cleaned):
        msg = "status codes must match ^[a-z][a-z0-9_]*$"
        raise ValueError(msg)
    if len(cleaned) > _MAX_CODE_LENGTH:
        msg = f"status codes exceed the maximum length of {_MAX_CODE_LENGTH}"
        raise ValueError(msg)
    return cleaned


def _clean_optional_code(value: str | None) -> str | None:
    if value is None:
        return None
    return _clean_code(value)


def _clean_identifier(value: str, *, max_length: int = _MAX_IDENTIFIER_LENGTH) -> str:
    cleaned = value.strip()
    if not _SAFE_IDENTIFIER_RE.match(cleaned):
        msg = "status identifiers must match ^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
        raise ValueError(msg)
    if len(cleaned) > max_length:
        msg = f"status identifiers exceed the maximum length of {max_length}"
        raise ValueError(msg)
    return cleaned


def _clean_optional_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    return _clean_identifier(value)


def _finite_number_or_none(value: Any) -> float | None:
    """Accept a finite int/float (as float) or None; reject bool/str/NaN/inf."""

    if value is None:
        return None
    if isinstance(value, bool):
        msg = "status numeric fields must not be booleans"
        raise ValueError(msg)
    if isinstance(value, int):
        value = float(value)
    if not isinstance(value, float):
        msg = "status numeric fields must be real numbers"
        raise ValueError(msg)
    if not math.isfinite(value):
        msg = "status numeric fields must be finite"
        raise ValueError(msg)
    return value


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


def _validate_suggested_command(value: str) -> str:
    cleaned = _clean_text(value, max_length=_MAX_COMMAND_LENGTH)
    if not cleaned.startswith(_SUGGESTED_COMMAND_PREFIX):
        msg = "status suggested_command must be a pmem command starting with 'pmem '"
        raise ValueError(msg)
    for token in _SHELL_METACHARACTERS:
        if token in cleaned:
            msg = "status suggested_command must not contain shell metacharacters"
            raise ValueError(msg)
    return cleaned


# --------------------------------------------------------------------------- #
# Vocabularies                                                                 #
# --------------------------------------------------------------------------- #
class TargetStatus(PmemStrEnum):
    """Target evaluation vocabulary; mirrors ``project_summary._target_status``."""

    NO_RUNS = "no_runs"
    NO_SUCCESSFUL_RUNS = "no_successful_runs"
    NOT_CONFIGURED = "not_configured"
    NO_METRIC = "no_metric"
    MET = "met"
    NOT_MET = "not_met"


class GraphState(PmemStrEnum):
    """Closed graph-freshness vocabulary. Detection is defined in STS-002+."""

    MISSING = "missing"
    CURRENT = "current"
    STALE = "stale"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class RecommendationMode(PmemStrEnum):
    """How recommendation state is sourced. On-demand is not a persisted lifecycle."""

    NOT_EVALUATED = "not_evaluated"
    GENERATED_ON_DEMAND = "generated_on_demand"
    PERSISTED_LIFECYCLE = "persisted_lifecycle"


class WarningSeverity(PmemStrEnum):
    """Small severity vocabulary for status warnings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class WarningSource(PmemStrEnum):
    """Which subsystem raised a status warning."""

    SUMMARY = "summary"
    GRAPH = "graph"
    RECOMMENDATION = "recommendation"
    DATA_QUALITY = "data_quality"


# --------------------------------------------------------------------------- #
# Nested models                                                                #
# --------------------------------------------------------------------------- #
class StatusProject(BaseModel):
    """Project identity block. ``objective`` is nullable but must be sent."""

    model_config = _MODEL_CONFIG

    project_id: str
    project_name: str
    objective: str | None

    @field_validator("project_id")
    @classmethod
    def _validate_project_id(cls, value: str) -> str:
        return _clean_identifier(value)

    @field_validator("project_name")
    @classmethod
    def _validate_project_name(cls, value: str) -> str:
        return _clean_text(value, max_length=_MAX_PROJECT_NAME_LENGTH)

    @field_validator("objective")
    @classmethod
    def _validate_objective(cls, value: str | None) -> str | None:
        return _clean_optional_text(value, max_length=_MAX_OBJECTIVE_LENGTH)


class StatusMetric(BaseModel):
    """Primary metric + target evaluation state. Values are finite or null."""

    model_config = _MODEL_CONFIG

    primary_metric: str | None
    direction: MetricDirection | None
    target_value: float | None
    best_value: float | None
    target_status: TargetStatus

    @field_validator("primary_metric")
    @classmethod
    def _validate_metric_name(cls, value: str | None) -> str | None:
        # ``pmem init`` accepts primary-metric text (up to 512 chars), not an
        # identifier token. Preserve that compatibility while applying the
        # status payload's path/control safety policy.
        return _clean_optional_text(value, max_length=_MAX_METRIC_NAME_LENGTH)

    @field_validator("target_value", "best_value", mode="before")
    @classmethod
    def _validate_numbers(cls, value: Any) -> float | None:
        return _finite_number_or_none(value)

    @field_validator("direction", mode="before")
    @classmethod
    def _coerce_direction(cls, value: Any) -> Any:
        if value is None:
            return None
        return _coerce_enum_value(MetricDirection, value)

    @field_validator("target_status", mode="before")
    @classmethod
    def _coerce_target_status(cls, value: Any) -> Any:
        return _coerce_enum_value(TargetStatus, value)


class StatusCounts(BaseModel):
    """Non-negative integer counters. Totals cannot exceed the run count."""

    model_config = _MODEL_CONFIG

    run_count: int = Field(ge=0)
    successful_run_count: int = Field(ge=0)
    failed_run_count: int = Field(ge=0)
    tracked_path_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    note_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_run_totals(self) -> StatusCounts:
        # Runs may also be unknown/interrupted/timeout, so equality is NOT
        # required; only the impossible over-count is rejected.
        if self.successful_run_count + self.failed_run_count > self.run_count:
            msg = "successful + failed runs cannot exceed total run_count"
            raise ValueError(msg)
        return self


class StatusBestRun(BaseModel):
    """Best observed run, kept distinct from the baseline block."""

    model_config = _MODEL_CONFIG

    run_id: str | None
    metric_value: float | None

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str | None) -> str | None:
        return _clean_optional_identifier(value)

    @field_validator("metric_value", mode="before")
    @classmethod
    def _validate_metric_value(cls, value: Any) -> float | None:
        return _finite_number_or_none(value)

    @model_validator(mode="after")
    def _validate_pairing(self) -> StatusBestRun:
        if (self.run_id is None) != (self.metric_value is None):
            msg = "best_run run_id and metric_value must both be present or both null"
            raise ValueError(msg)
        return self


class StatusBaseline(BaseModel):
    """Baseline run reference only. No fabricated baseline metric value."""

    model_config = _MODEL_CONFIG

    run_id: str | None

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str | None) -> str | None:
        return _clean_optional_identifier(value)


class StatusGraph(BaseModel):
    """Graph-freshness block with a single, closed count/reason convention.

    - ``current`` -> node_count and edge_count present (>=0); reason_code null.
    - ``stale``   -> node_count and edge_count present (>=0); reason_code set.
    - ``missing`` / ``invalid`` / ``unknown`` -> both counts null; reason_code set.
    """

    model_config = _MODEL_CONFIG

    state: GraphState
    node_count: int | None = Field(ge=0)
    edge_count: int | None = Field(ge=0)
    reason_code: str | None

    @field_validator("state", mode="before")
    @classmethod
    def _coerce_state(cls, value: Any) -> Any:
        return _coerce_enum_value(GraphState, value)

    @field_validator("reason_code")
    @classmethod
    def _validate_reason_code(cls, value: str | None) -> str | None:
        return _clean_optional_code(value)

    @model_validator(mode="after")
    def _validate_consistency(self) -> StatusGraph:
        both_present = self.node_count is not None and self.edge_count is not None
        both_absent = self.node_count is None and self.edge_count is None
        if not (both_present or both_absent):
            msg = "graph node_count and edge_count must both be present or both null"
            raise ValueError(msg)
        if self.state in (GraphState.CURRENT, GraphState.STALE):
            if not both_present:
                msg = f"graph state '{self.state.value}' requires node_count and edge_count"
                raise ValueError(msg)
        elif not both_absent:
            msg = f"graph state '{self.state.value}' requires null node_count and edge_count"
            raise ValueError(msg)
        if self.state is GraphState.CURRENT:
            if self.reason_code is not None:
                msg = "graph state 'current' must not carry a reason_code"
                raise ValueError(msg)
        elif self.reason_code is None:
            msg = f"graph state '{self.state.value}' requires a reason_code"
            raise ValueError(msg)
        return self


class StatusRecommendations(BaseModel):
    """Recommendation availability with closed per-mode invariants."""

    model_config = _MODEL_CONFIG

    mode: RecommendationMode
    candidate_count: int | None = Field(ge=0)
    active_count: int | None = Field(ge=0)

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, value: Any) -> Any:
        return _coerce_enum_value(RecommendationMode, value)

    @model_validator(mode="after")
    def _validate_consistency(self) -> StatusRecommendations:
        if self.mode is RecommendationMode.NOT_EVALUATED:
            if self.candidate_count is not None or self.active_count is not None:
                msg = "not_evaluated requires null candidate_count and active_count"
                raise ValueError(msg)
        elif self.mode is RecommendationMode.GENERATED_ON_DEMAND:
            if self.candidate_count is None:
                msg = "generated_on_demand requires a candidate_count"
                raise ValueError(msg)
            if self.active_count is not None:
                msg = "generated_on_demand has no persisted active_count"
                raise ValueError(msg)
        else:  # PERSISTED_LIFECYCLE
            if self.candidate_count is None or self.active_count is None:
                msg = "persisted_lifecycle requires candidate_count and active_count"
                raise ValueError(msg)
            if self.active_count > self.candidate_count:
                msg = "active_count cannot exceed candidate_count"
                raise ValueError(msg)
        return self


class StatusWarning(BaseModel):
    """One typed, privacy-safe warning. Never carries raw project free text."""

    model_config = _MODEL_CONFIG

    code: str
    severity: WarningSeverity
    message: str
    source: WarningSource
    remediation: str | None

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, value: Any) -> Any:
        return _coerce_enum_value(WarningSeverity, value)

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_source(cls, value: Any) -> Any:
        return _coerce_enum_value(WarningSource, value)

    @field_validator("code")
    @classmethod
    def _validate_code(cls, value: str) -> str:
        return _clean_code(value)

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        return _clean_text(value, max_length=_MAX_MESSAGE_LENGTH)

    @field_validator("remediation")
    @classmethod
    def _validate_remediation(cls, value: str | None) -> str | None:
        return _clean_optional_text(value, max_length=_MAX_REMEDIATION_LENGTH)


class StatusNextAction(BaseModel):
    """The single next action. Ordering rules are owned by STS-003, not here."""

    model_config = _MODEL_CONFIG

    action_id: str
    reason: str
    suggested_command: str
    related_entity_id: str | None

    @field_validator("action_id")
    @classmethod
    def _validate_action_id(cls, value: str) -> str:
        return _clean_code(value)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        return _clean_text(value, max_length=_MAX_REASON_LENGTH)

    @field_validator("suggested_command")
    @classmethod
    def _validate_command(cls, value: str) -> str:
        return _validate_suggested_command(value)

    @field_validator("related_entity_id")
    @classmethod
    def _validate_related_entity_id(cls, value: str | None) -> str | None:
        return _clean_optional_identifier(value)


# --------------------------------------------------------------------------- #
# Root payload                                                                 #
# --------------------------------------------------------------------------- #
class StatusPayload(BaseModel):
    """Root ``status-v1`` payload.

    Every field is required (no defaults): a producer that omits the schema
    version, a privacy flag, ``warnings`` or any nullable field fails
    validation. ``warnings`` is stored as a tuple so the validated payload is
    deeply immutable while still serialising to a JSON array. Warning order is
    preserved (STS-002 is responsible for deterministic ordering; the model
    never re-sorts). A root validator ties the target/best-run blocks to the
    ``_target_status`` logic so self-contradictory payloads are rejected.
    """

    model_config = _MODEL_CONFIG

    schema_version: Literal["status-v1"]
    project: StatusProject
    metric: StatusMetric
    counts: StatusCounts
    best_run: StatusBestRun
    baseline: StatusBaseline
    graph: StatusGraph
    recommendations: StatusRecommendations
    warnings: tuple[StatusWarning, ...] = Field(max_length=_MAX_WARNINGS)
    next_action: StatusNextAction
    database_mutation: Literal[False]
    network: Literal[False]
    raw_text_in_output: Literal[False]

    @field_validator("warnings", mode="before")
    @classmethod
    def _coerce_warnings(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_target_consistency(self) -> StatusPayload:
        metric = self.metric
        counts = self.counts
        best = self.best_run

        # Universal best-run/best-value coupling (independent of target_status).
        if metric.best_value != best.metric_value:
            msg = "metric.best_value must equal best_run.metric_value"
            raise ValueError(msg)

        status = metric.target_status
        best_run_empty = best.run_id is None and best.metric_value is None

        if counts.run_count == 0:
            if status is not TargetStatus.NO_RUNS:
                msg = "target_status must be 'no_runs' when run_count is 0"
                raise ValueError(msg)
            if not best_run_empty or metric.best_value is not None:
                msg = "no_runs requires an empty best_run and null best_value"
                raise ValueError(msg)
            return self

        if counts.successful_run_count == 0:
            if status is not TargetStatus.NO_SUCCESSFUL_RUNS:
                msg = "target_status must be 'no_successful_runs' when there are no successful runs"
                raise ValueError(msg)
            if not best_run_empty or metric.best_value is not None:
                msg = "no_successful_runs requires an empty best_run and null best_value"
                raise ValueError(msg)
            return self

        if metric.primary_metric is None or metric.direction is None or metric.target_value is None:
            if status is not TargetStatus.NOT_CONFIGURED:
                msg = "target_status must be 'not_configured' when metric/direction/target is null"
                raise ValueError(msg)
            # ``_best_run`` cannot select a run without both a metric name and
            # direction. A missing target alone may still have a best run.
            if metric.primary_metric is None or metric.direction is None:
                if not best_run_empty or metric.best_value is not None:
                    msg = "missing metric or direction requires an empty best_run"
                    raise ValueError(msg)
            return self

        if metric.best_value is None:
            if status is not TargetStatus.NO_METRIC:
                msg = "target_status must be 'no_metric' when there is no best metric value"
                raise ValueError(msg)
            return self

        if best.run_id is None:
            msg = "a configured project with a best metric requires a non-null best_run.run_id"
            raise ValueError(msg)
        if metric.direction is MetricDirection.MAX:
            expected = (
                TargetStatus.MET
                if metric.best_value >= metric.target_value
                else TargetStatus.NOT_MET
            )
        else:
            expected = (
                TargetStatus.MET
                if metric.best_value <= metric.target_value
                else TargetStatus.NOT_MET
            )
        if status is not expected:
            msg = f"target_status must be '{expected.value}' for the given best/target values"
            raise ValueError(msg)
        return self
