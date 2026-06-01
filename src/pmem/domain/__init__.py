"""Domain layer for durable project-memory concepts.

The domain package owns typed, persistence-agnostic models. CLI code should not
reach into SQLite directly; it should validate user intent here, then pass clean
objects to services/repositories.
"""

from pmem.domain.common import (
    EvaluationFlag,
    ExperimentStatus,
    FailureCandidateKind,
    FailureSeverity,
    FailureSource,
    MetricDirection,
    ProjectStatus,
    RunStatus,
)
from pmem.domain.decision import Decision
from pmem.domain.experiment import Experiment
from pmem.domain.failure import Failure
from pmem.domain.failure_taxonomy import DEFAULT_FAILURE_TAGS
from pmem.domain.note import Note
from pmem.domain.project import Project
from pmem.domain.run import Run
from pmem.domain.target import (
    BaselineReference,
    FailureCandidate,
    FailureCriterion,
    ProjectTarget,
    RunEvaluation,
    TargetSpec,
)
from pmem.domain.tracked_path import TrackedPath

__all__ = [
    "DEFAULT_FAILURE_TAGS",
    "BaselineReference",
    "Decision",
    "EvaluationFlag",
    "Experiment",
    "ExperimentStatus",
    "Failure",
    "FailureCandidate",
    "FailureCandidateKind",
    "FailureCriterion",
    "FailureSeverity",
    "FailureSource",
    "MetricDirection",
    "Note",
    "Project",
    "ProjectStatus",
    "ProjectTarget",
    "Run",
    "RunEvaluation",
    "RunStatus",
    "TargetSpec",
    "TrackedPath",
]
