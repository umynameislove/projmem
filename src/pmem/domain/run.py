"""Run entity model.

Run captures command execution evidence plus the DOCX's new evaluation layer:
technical status, metrics, target/baseline comparison, and unresolved failure
candidates. Confirmed failures live in `Failure`, not directly on `Run`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pmem.domain.common import RunStatus
from pmem.domain.target import FailureCandidate, RunEvaluation


class Run(BaseModel):
    """A single command execution recorded by `pmem run`."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    experiment_id: str
    command: str
    cwd: str
    name: str | None = None
    exit_code: int | None = None
    status: RunStatus = RunStatus.UNKNOWN
    duration_sec: float | None = None
    seed: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    stdout_preview: str | None = None
    stderr_preview: str | None = None
    env: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str | None = None
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    git: dict[str, Any] = Field(default_factory=dict)
    evaluation: RunEvaluation = Field(default_factory=RunEvaluation)
    failure_candidates: list[FailureCandidate] = Field(default_factory=list)
    timestamp: datetime | None = None

    @field_validator("run_id", "experiment_id", "command", "cwd")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        """Required run fields must be non-empty strings."""

        cleaned = value.strip()
        if not cleaned:
            msg = "run_id/experiment_id/command/cwd cannot be blank"
            raise ValueError(msg)
        return cleaned

    @field_validator("duration_sec")
    @classmethod
    def duration_must_not_be_negative(cls, value: float | None) -> float | None:
        """Duration can be unknown, but cannot be negative."""

        if value is not None and value < 0:
            msg = "duration_sec cannot be negative"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def successful_run_should_not_have_nonzero_exit(self) -> Run:
        """Prevent an obviously contradictory technical status."""

        if self.status == RunStatus.SUCCESS and self.exit_code not in (0, None):
            msg = "status=success requires exit_code 0 or unknown"
            raise ValueError(msg)
        return self
