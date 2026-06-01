"""Graph schema contracts.

This module defines stable identifiers, node/edge vocabulary, classification,
and a deterministic JSON schema prototype. It deliberately does not read
SQLite, build a NetworkX graph, or expose CLI commands.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import PurePosixPath
from typing import Any

from pmem.domain.common import PmemStrEnum
from pmem.errors import PmemSecurityError, PmemValidationError
from pmem.graph.privacy import GRAPH_CONFIG_DEFAULTS, graph_artifact_policy
from pmem.utils.hashing import compute_text_hash

GRAPH_SCHEMA_VERSION = "graph-schema-v1"
NODE_TYPES: tuple[NodeType, ...]
EDGE_SPECS: tuple[EdgeSpec, ...]
_EDGE_ID_SEPARATOR = "::"
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


class NodeType(PmemStrEnum):
    """Graph node types supported by the local evidence graph."""

    PROJECT = "project"
    EXPERIMENT = "experiment"
    RUN = "run"
    CONFIG = "config"
    DATASET = "dataset"
    METRIC = "metric"
    ARTIFACT = "artifact"
    FAILURE = "failure"
    DECISION = "decision"
    NOTE = "note"
    CODE_MODULE = "code_module"


class EdgeClass(PmemStrEnum):
    """Edge confidence and build timing classes."""

    DIRECT = "direct"
    CONDITIONAL_DIRECT = "conditional_direct"
    DERIVED = "derived"
    OPTIONAL = "optional"
    DEFERRED = "deferred"


class EdgeType(PmemStrEnum):
    """Graph edge types.

    `OBSERVED_IN` is intentionally observational, not causal. Do not introduce
    a causal failure edge without explicit human-confirmed evidence in a future
    task.
    """

    BELONGS_TO = "BELONGS_TO"
    OBSERVED_IN = "OBSERVED_IN"
    NOTE_ON = "NOTE_ON"
    NOTE_IN_EXPERIMENT = "NOTE_IN_EXPERIMENT"
    DECISION_IN_PROJECT = "DECISION_IN_PROJECT"
    DECISION_IN_EXPERIMENT = "DECISION_IN_EXPERIMENT"
    USES_CONFIG = "USES_CONFIG"
    PRODUCES_METRIC = "PRODUCES_METRIC"
    PRODUCES_ARTIFACT = "PRODUCES_ARTIFACT"
    TRACKS_CODE = "TRACKS_CODE"
    TRAINED_ON = "TRAINED_ON"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    BASED_ON = "BASED_ON"
    SUPERSEDES = "SUPERSEDES"


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """Static schema metadata for an allowed edge type."""

    edge_type: EdgeType
    source: NodeType
    target: NodeType
    edge_class: EdgeClass
    creation_rule: str
    availability: str

    def to_dict(self) -> dict[str, str]:
        """Return a stable JSON-ready edge spec."""

        return {
            "edge_type": self.edge_type.value,
            "source": self.source.value,
            "target": self.target.value,
            "edge_class": self.edge_class.value,
            "creation_rule": self.creation_rule,
            "availability": self.availability,
        }


NODE_TYPES = (
    NodeType.PROJECT,
    NodeType.EXPERIMENT,
    NodeType.RUN,
    NodeType.CONFIG,
    NodeType.DATASET,
    NodeType.METRIC,
    NodeType.ARTIFACT,
    NodeType.FAILURE,
    NodeType.DECISION,
    NodeType.NOTE,
    NodeType.CODE_MODULE,
)

EDGE_SPECS = (
    EdgeSpec(
        EdgeType.BELONGS_TO,
        NodeType.RUN,
        NodeType.EXPERIMENT,
        EdgeClass.DIRECT,
        "runs.experiment_id foreign key",
        "direct_ingestion",
    ),
    EdgeSpec(
        EdgeType.OBSERVED_IN,
        NodeType.FAILURE,
        NodeType.RUN,
        EdgeClass.DIRECT,
        "failures.run_id foreign key; observational, not causal",
        "direct_ingestion",
    ),
    EdgeSpec(
        EdgeType.NOTE_ON,
        NodeType.NOTE,
        NodeType.RUN,
        EdgeClass.CONDITIONAL_DIRECT,
        "notes.run_id present",
        "direct_ingestion",
    ),
    EdgeSpec(
        EdgeType.NOTE_IN_EXPERIMENT,
        NodeType.NOTE,
        NodeType.EXPERIMENT,
        EdgeClass.CONDITIONAL_DIRECT,
        "notes.experiment_id present",
        "direct_ingestion",
    ),
    EdgeSpec(
        EdgeType.DECISION_IN_PROJECT,
        NodeType.DECISION,
        NodeType.PROJECT,
        EdgeClass.DIRECT,
        "decisions.project_id foreign key",
        "direct_ingestion",
    ),
    EdgeSpec(
        EdgeType.DECISION_IN_EXPERIMENT,
        NodeType.DECISION,
        NodeType.EXPERIMENT,
        EdgeClass.CONDITIONAL_DIRECT,
        "decisions.experiment_id present",
        "direct_ingestion",
    ),
    EdgeSpec(
        EdgeType.USES_CONFIG,
        NodeType.RUN,
        NodeType.CONFIG,
        EdgeClass.CONDITIONAL_DIRECT,
        "runs.config_hash present",
        "direct_ingestion",
    ),
    EdgeSpec(
        EdgeType.PRODUCES_METRIC,
        NodeType.RUN,
        NodeType.METRIC,
        EdgeClass.CONDITIONAL_DIRECT,
        "runs.metrics_json contains a finite numeric metric",
        "direct_ingestion",
    ),
    EdgeSpec(
        EdgeType.PRODUCES_ARTIFACT,
        NodeType.RUN,
        NodeType.ARTIFACT,
        EdgeClass.CONDITIONAL_DIRECT,
        "runs.artifacts_json contains artifact path/hash metadata",
        "direct_ingestion",
    ),
    EdgeSpec(
        EdgeType.TRACKS_CODE,
        NodeType.PROJECT,
        NodeType.CODE_MODULE,
        EdgeClass.CONDITIONAL_DIRECT,
        "tracked_paths row exists",
        "direct_ingestion",
    ),
    EdgeSpec(
        EdgeType.TRAINED_ON,
        NodeType.RUN,
        NodeType.DATASET,
        EdgeClass.DEFERRED,
        "dataset identity is not guaranteed by current artifact metadata",
        "deferred",
    ),
    EdgeSpec(
        EdgeType.SUPPORTS,
        NodeType.RUN,
        NodeType.DECISION,
        EdgeClass.DERIVED,
        "metric comparison rule with direct evidence",
        "derived_analysis",
    ),
    EdgeSpec(
        EdgeType.CONTRADICTS,
        NodeType.RUN,
        NodeType.DECISION,
        EdgeClass.DERIVED,
        "metric comparison rule with direct evidence",
        "derived_analysis",
    ),
    EdgeSpec(
        EdgeType.BASED_ON,
        NodeType.RUN,
        NodeType.RUN,
        EdgeClass.OPTIONAL,
        "explicit user-logged run linkage",
        "optional_future",
    ),
    EdgeSpec(
        EdgeType.SUPERSEDES,
        NodeType.DECISION,
        NodeType.DECISION,
        EdgeClass.OPTIONAL,
        "explicit user-logged decision linkage",
        "optional_future",
    ),
)


def project_node_id(project_id: str) -> str:
    return f"project:{_clean_id_part('project_id', project_id)}"


def experiment_node_id(experiment_id: str) -> str:
    return f"experiment:{_clean_id_part('experiment_id', experiment_id)}"


def run_node_id(run_id: str) -> str:
    return f"run:{_clean_id_part('run_id', run_id)}"


def config_node_id(config_hash: str) -> str:
    return f"config:{_clean_id_part('config_hash', config_hash)}"


def dataset_node_id(dataset_id: str) -> str:
    return f"dataset:{_clean_id_part('dataset_id', dataset_id)}"


def metric_node_id(run_id: str, metric_name: str) -> str:
    return f"metric:{_clean_id_part('run_id', run_id)}:{_clean_id_part('metric_name', metric_name)}"


def artifact_node_id(run_id: str, artifact_path: str) -> str:
    normalized = normalize_project_relative_path(artifact_path)
    return f"artifact:{_clean_id_part('run_id', run_id)}:{compute_text_hash(normalized)}"


def failure_node_id(failure_id: str) -> str:
    return f"failure:{_clean_id_part('failure_id', failure_id)}"


def decision_node_id(decision_id: str) -> str:
    return f"decision:{_clean_id_part('decision_id', decision_id)}"


def note_node_id(note_id: str) -> str:
    return f"note:{_clean_id_part('note_id', note_id)}"


def code_module_node_id(project_id: str, project_relative_path: str) -> str:
    normalized = normalize_project_relative_path(project_relative_path)
    return f"code:{_clean_id_part('project_id', project_id)}:{compute_text_hash(normalized)}"


def edge_id(edge_type: EdgeType | str, source_node_id: str, target_node_id: str) -> str:
    """Return the canonical deterministic edge id."""

    edge_value = edge_type.value if isinstance(edge_type, EdgeType) else str(edge_type)
    _validate_edge_type(edge_value)
    source = _clean_node_id("source_node_id", source_node_id)
    target = _clean_node_id("target_node_id", target_node_id)
    return f"{edge_value}{_EDGE_ID_SEPARATOR}{source}{_EDGE_ID_SEPARATOR}{target}"


def is_supported_metric_value(value: object) -> bool:
    """Return whether a JSON metric value should become a Metric node."""

    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    return math.isfinite(float(value))


def metric_payload(
    *,
    run_id: str,
    metric_name: str,
    value: int | float,
    primary_metric: str | None = None,
) -> dict[str, Any]:
    """Return stable Metric node attributes for a finite numeric metric."""

    if not is_supported_metric_value(value):
        raise PmemValidationError("Metric nodes require finite numeric metric values.")
    return {
        "node_id": metric_node_id(run_id, metric_name),
        "node_type": NodeType.METRIC.value,
        "run_id": _clean_id_part("run_id", run_id),
        "metric_name": _clean_id_part("metric_name", metric_name),
        "value": float(value),
        "is_primary_metric": primary_metric == metric_name,
    }


def normalize_project_relative_path(path: str) -> str:
    """Normalize a safe project-relative path for graph IDs."""

    if not isinstance(path, str) or not path.strip():
        raise PmemValidationError("Graph paths must be non-empty project-relative strings.")
    if any(ord(char) < 32 for char in path):
        raise PmemSecurityError("Graph paths contain unsafe characters.")
    if _WINDOWS_DRIVE_RE.match(path) or path.startswith(("/", "\\")):
        raise PmemSecurityError("Graph paths must be project-relative.")

    normalized = path.replace("\\", "/")
    parts: list[str] = []
    for part in PurePosixPath(normalized).parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise PmemSecurityError("Graph paths must not traverse outside the project.")
        if part.casefold() == ".pmem":
            raise PmemSecurityError("Graph paths must not reference .pmem internals.")
        parts.append(part)
    if not parts:
        raise PmemValidationError("Graph paths must identify a project file.")
    return "/".join(parts)


def graph_schema_document() -> dict[str, Any]:
    """Return a deterministic JSON-ready schema document."""

    edge_specs = sorted(EDGE_SPECS, key=lambda spec: spec.edge_type.value)
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "node_types": [node_type.value for node_type in NODE_TYPES],
        "node_count": len(NODE_TYPES),
        "edge_specs": [spec.to_dict() for spec in edge_specs],
        "edge_count": len(edge_specs),
        "edge_id_format": "{edge_type}::{source_node_id}::{target_node_id}",
        "causal_failure_edge": False,
        "failure_edge": EdgeType.OBSERVED_IN.value,
        "graph_engine": {
            "prototype": "networkx",
            "local_first": True,
            "server_required": False,
            "neo4j_migration_gate": "synthetic benchmark only",
        },
        "artifact_policy": graph_artifact_policy(),
        "config_defaults": GRAPH_CONFIG_DEFAULTS,
        "privacy": {
            "raw_text_default": False,
            "context_pack_omits_free_text_by_default": True,
            "absolute_paths_default": False,
        },
    }


def canonical_schema_json() -> str:
    """Return canonical JSON for schema snapshot tests and ADR evidence."""

    return json.dumps(graph_schema_document(), sort_keys=True, separators=(",", ":"))


def _clean_id_part(label: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PmemValidationError(f"Graph {label} must be a non-empty string.")
    cleaned = value.strip()
    if any(ord(char) < 32 for char in cleaned):
        raise PmemValidationError(f"Graph {label} contains unsafe characters.")
    if _EDGE_ID_SEPARATOR in cleaned:
        raise PmemValidationError(f"Graph {label} must not contain '{_EDGE_ID_SEPARATOR}'.")
    return cleaned


def _clean_node_id(label: str, value: str) -> str:
    cleaned = _clean_id_part(label, value)
    if ":" not in cleaned:
        raise PmemValidationError(f"Graph {label} must be a typed node id.")
    return cleaned


def _validate_edge_type(edge_value: str) -> None:
    allowed = {edge_type.value for edge_type in EdgeType}
    if edge_value not in allowed:
        raise PmemValidationError("Graph edge type is not part of the graph schema.")
