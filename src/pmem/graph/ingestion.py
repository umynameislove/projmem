"""graph ingestion SQLite-to-graph ingestion.

This module builds an in-memory graph document from existing SQLite records.
It does not persist `.pmem/graph.json`, expose CLI commands, or create derived
relationships. Persistence and NetworkX CRUD wrappers belong to later evidence graph layer
tasks.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pmem.errors import PmemNotFoundError, PmemSecurityError, PmemValidationError
from pmem.graph.provenance import GraphProvenance, provenance
from pmem.graph.schema import (
    GRAPH_SCHEMA_VERSION,
    EdgeClass,
    EdgeType,
    NodeType,
    artifact_node_id,
    code_module_node_id,
    config_node_id,
    decision_node_id,
    edge_id,
    experiment_node_id,
    failure_node_id,
    is_supported_metric_value,
    metric_payload,
    normalize_project_relative_path,
    note_node_id,
    project_node_id,
    run_node_id,
)
from pmem.repositories.decisions import DecisionRecord, DecisionRepository
from pmem.repositories.experiments import ExperimentRecord, ExperimentRepository
from pmem.repositories.failures import FailureRecord, FailureRepository
from pmem.repositories.notes import NoteRecord, NoteRepository
from pmem.repositories.projects import ProjectRecord, ProjectRepository
from pmem.repositories.runs import RunRecord, RunRepository
from pmem.repositories.sqlite import (
    connect_database,
    connect_database_readonly,
    project_database_path,
)
from pmem.repositories.tracked_paths import TrackedPathRecord, TrackedPathRepository
from pmem.utils.hashing import compute_text_hash

GRAPH_INGESTION_METHOD = "sqlite-direct-ingestion-v1"
DIRECT_INGESTION_EDGE_CLASSES = frozenset({EdgeClass.DIRECT, EdgeClass.CONDITIONAL_DIRECT})


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One privacy-safe in-memory graph node."""

    node_id: str
    node_type: NodeType
    attributes: dict[str, Any]
    provenance: tuple[GraphProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready node."""

        return {
            "id": self.node_id,
            "type": self.node_type.value,
            "attributes": _stable_data(self.attributes),
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """One evidence-backed in-memory graph edge."""

    edge_id: str
    edge_type: EdgeType
    source: str
    target: str
    edge_class: EdgeClass
    attributes: dict[str, Any]
    provenance: tuple[GraphProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready edge."""

        return {
            "id": self.edge_id,
            "type": self.edge_type.value,
            "source": self.source,
            "target": self.target,
            "edge_class": self.edge_class.value,
            "attributes": _stable_data(self.attributes),
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(frozen=True, slots=True)
class GraphDocument:
    """graph ingestion in-memory graph document.

    The document is JSON-ready but not the persisted `.pmem/graph.json` format.
    graph engine owns graph persistence and round-trip serialization.
    """

    schema_version: str
    method: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    counts: dict[str, Any]
    warnings: tuple[str, ...]
    skipped_counts: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready graph document."""

        ordered_nodes = sorted(self.nodes, key=lambda node: node.node_id)
        ordered_edges = sorted(self.edges, key=lambda edge: edge.edge_id)
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "metadata": _stable_data(self.metadata),
            "counts": _stable_data(self.counts),
            "skipped_counts": dict(sorted(self.skipped_counts.items())),
            "warnings": list(self.warnings),
            "nodes": [node.to_dict() for node in ordered_nodes],
            "edges": [edge.to_dict() for edge in ordered_edges],
        }


def build_graph_from_project(project_root: str | Path) -> GraphDocument:
    """Build an in-memory graph document from a project root."""

    return build_graph_from_database(project_database_path(project_root))


def build_graph_from_database(db_path: str | Path) -> GraphDocument:
    """Build an in-memory graph document from an existing SQLite database."""

    path = Path(db_path)
    if not path.exists():
        raise PmemNotFoundError("Project database was not found.")
    connection = connect_database(path)
    try:
        return _GraphIngestion(connection).build()
    finally:
        connection.close()


def build_graph_from_database_readonly(db_path: str | Path) -> GraphDocument:
    """Build an in-memory graph document using a strictly read-only connection."""

    connection = connect_database_readonly(db_path)
    try:
        return _GraphIngestion(connection).build()
    finally:
        connection.close()


class _GraphIngestion:
    """Small scoped graph ingestion ingestion coordinator."""

    def __init__(self, connection: Any) -> None:
        self._projects = ProjectRepository(connection)
        self._experiments = ExperimentRepository(connection)
        self._runs = RunRepository(connection)
        self._failures = FailureRepository(connection)
        self._decisions = DecisionRepository(connection)
        self._notes = NoteRepository(connection)
        self._tracked_paths = TrackedPathRepository(connection)
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._warnings: list[str] = []
        self._skipped: Counter[str] = Counter()

    def build(self) -> GraphDocument:
        """Read existing rows and create only direct/conditional-direct graph data."""

        projects = self._projects.list_projects()
        if not projects:
            self._warnings.append("No project rows found; graph is empty.")

        for project in projects:
            self._add_project(project)
            for experiment in self._experiments.list_for_project(project.id):
                self._add_experiment(experiment)
            for run in self._runs.list_for_project(project.id):
                self._add_run(run, project=project)
            for failure in self._failures.list_for_project(project.id):
                self._add_failure(failure)
            for decision in self._decisions.list_for_project(project.id):
                self._add_decision(decision)
            for note in self._notes.list_for_project(project.id):
                self._add_note(note)
            for tracked_path in self._tracked_paths.list_for_project(project.id):
                self._add_code_module(project, tracked_path)

        return GraphDocument(
            schema_version=GRAPH_SCHEMA_VERSION,
            method=GRAPH_INGESTION_METHOD,
            nodes=tuple(sorted(self._nodes.values(), key=lambda node: node.node_id)),
            edges=tuple(sorted(self._edges.values(), key=lambda edge: edge.edge_id)),
            counts=self._counts(),
            warnings=tuple(self._warnings),
            skipped_counts=dict(sorted(self._skipped.items())),
            metadata={
                "database_mutation": False,
                "artifact_persistence": False,
                "direct_edge_classes": sorted(
                    edge_class.value for edge_class in DIRECT_INGESTION_EDGE_CLASSES
                ),
                "deferred_edge_types": [
                    EdgeType.TRAINED_ON.value,
                    EdgeType.SUPPORTS.value,
                    EdgeType.CONTRADICTS.value,
                    EdgeType.BASED_ON.value,
                    EdgeType.SUPERSEDES.value,
                ],
            },
        )

    def _add_project(self, project: ProjectRecord) -> None:
        node_id = project_node_id(project.id)
        self._put_node(
            GraphNode(
                node_id=node_id,
                node_type=NodeType.PROJECT,
                attributes={
                    "primary_metric": project.primary_metric,
                    "metric_direction": project.metric_direction,
                    "has_target": project.target_json != "{}",
                    "created_at": project.created_at,
                    "updated_at": project.updated_at,
                },
                provenance=(
                    provenance(
                        source_table="projects",
                        source_pk=project.id,
                        source_field="id",
                        creation_rule="projects row",
                    ),
                ),
            )
        )

    def _add_experiment(self, experiment: ExperimentRecord) -> None:
        node_id = experiment_node_id(experiment.id)
        self._put_node(
            GraphNode(
                node_id=node_id,
                node_type=NodeType.EXPERIMENT,
                attributes={
                    "project_id": project_node_id(experiment.project_id),
                    "status": experiment.status,
                    "is_baseline": experiment.is_baseline,
                    "primary_metric": experiment.primary_metric,
                    "has_target": experiment.target_json is not None,
                    "created_at": experiment.created_at,
                    "updated_at": experiment.updated_at,
                },
                provenance=(
                    provenance(
                        source_table="experiments",
                        source_pk=experiment.id,
                        source_field="id",
                        creation_rule="experiments row",
                    ),
                ),
            )
        )

    def _add_run(self, run: RunRecord, *, project: ProjectRecord) -> None:
        run_id_value = run_node_id(run.run_id)
        self._put_node(
            GraphNode(
                node_id=run_id_value,
                node_type=NodeType.RUN,
                attributes={
                    "experiment_id": experiment_node_id(run.experiment_id),
                    "status": run.status,
                    "exit_code": run.exit_code,
                    "duration_sec": run.duration_sec,
                    "seed_present": run.seed is not None,
                    "timestamp": run.timestamp,
                    "has_config": run.config_hash is not None,
                },
                provenance=(
                    provenance(
                        source_table="runs",
                        source_pk=run.run_id,
                        source_field="run_id",
                        creation_rule="runs row",
                    ),
                ),
            )
        )
        self._put_edge(
            EdgeType.BELONGS_TO,
            run_id_value,
            experiment_node_id(run.experiment_id),
            EdgeClass.DIRECT,
            provenance(
                source_table="runs",
                source_pk=run.run_id,
                source_field="experiment_id",
                creation_rule="runs.experiment_id foreign key",
            ),
        )

        if run.config_hash:
            self._add_config(run)
            self._put_edge(
                EdgeType.USES_CONFIG,
                run_id_value,
                config_node_id(run.config_hash),
                EdgeClass.CONDITIONAL_DIRECT,
                provenance(
                    source_table="runs",
                    source_pk=run.run_id,
                    source_field="config_hash",
                    creation_rule="runs.config_hash present",
                ),
            )
        else:
            self._skipped["config_missing"] += 1

        self._add_metrics(run, project=project)
        self._add_artifacts(run)

    def _add_config(self, run: RunRecord) -> None:
        if run.config_hash is None:
            return
        self._put_node(
            GraphNode(
                node_id=config_node_id(run.config_hash),
                node_type=NodeType.CONFIG,
                attributes={
                    "config_hash": run.config_hash,
                    "raw_config_included": False,
                },
                provenance=(
                    provenance(
                        source_table="runs",
                        source_pk=run.run_id,
                        source_field="config_hash",
                        creation_rule="runs.config_hash present",
                    ),
                ),
            )
        )

    def _add_metrics(self, run: RunRecord, *, project: ProjectRecord) -> None:
        metrics = _load_json_object(
            run.metrics_json, table="runs", pk=run.run_id, field="metrics_json"
        )
        if not metrics:
            if metrics == {}:
                self._skipped["metrics_empty"] += 1
            return
        for metric_name in sorted(metrics):
            value = metrics[metric_name]
            if not is_supported_metric_value(value):
                self._skipped["metric_not_numeric"] += 1
                continue
            payload = metric_payload(
                run_id=run.run_id,
                metric_name=metric_name,
                value=value,
                primary_metric=project.primary_metric,
            )
            node = GraphNode(
                node_id=str(payload["node_id"]),
                node_type=NodeType.METRIC,
                attributes={
                    "run_id": run_node_id(run.run_id),
                    "metric_name": str(payload["metric_name"]),
                    "value": payload["value"],
                    "is_primary_metric": payload["is_primary_metric"],
                },
                provenance=(
                    provenance(
                        source_table="runs",
                        source_pk=run.run_id,
                        source_field=f"metrics_json.{metric_name}",
                        creation_rule="runs.metrics_json finite numeric metric",
                    ),
                ),
            )
            self._put_node(node)
            self._put_edge(
                EdgeType.PRODUCES_METRIC,
                run_node_id(run.run_id),
                node.node_id,
                EdgeClass.CONDITIONAL_DIRECT,
                provenance(
                    source_table="runs",
                    source_pk=run.run_id,
                    source_field=f"metrics_json.{metric_name}",
                    creation_rule="runs.metrics_json finite numeric metric",
                ),
            )

    def _add_artifacts(self, run: RunRecord) -> None:
        artifacts = _load_json_array(
            run.artifacts_json,
            table="runs",
            pk=run.run_id,
            field="artifacts_json",
        )
        if not artifacts:
            if artifacts == []:
                self._skipped["artifacts_empty"] += 1
            return
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                self._skipped["artifact_malformed"] += 1
                continue
            path = artifact.get("path")
            sha256 = artifact.get("sha256")
            size_bytes = artifact.get("size_bytes")
            if not isinstance(path, str) or not isinstance(sha256, str):
                self._skipped["artifact_missing_path_or_hash"] += 1
                continue
            try:
                normalized_path = normalize_project_relative_path(path)
                node_id = artifact_node_id(run.run_id, normalized_path)
            except (PmemSecurityError, PmemValidationError):
                self._skipped["artifact_unsafe_path"] += 1
                self._warnings.append(f"Skipped unsafe artifact path for run {run.run_id}.")
                continue
            path_hash = compute_text_hash(normalized_path)
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.ARTIFACT,
                attributes={
                    "run_id": run_node_id(run.run_id),
                    "path_hash": path_hash,
                    "sha256": sha256,
                    "size_bytes": size_bytes if isinstance(size_bytes, int) else None,
                    "raw_path_included": False,
                },
                provenance=(
                    provenance(
                        source_table="runs",
                        source_pk=run.run_id,
                        source_field=f"artifacts_json[{index}]",
                        creation_rule="runs.artifacts_json artifact metadata",
                    ),
                ),
            )
            self._put_node(node)
            self._put_edge(
                EdgeType.PRODUCES_ARTIFACT,
                run_node_id(run.run_id),
                node.node_id,
                EdgeClass.CONDITIONAL_DIRECT,
                provenance(
                    source_table="runs",
                    source_pk=run.run_id,
                    source_field=f"artifacts_json[{index}]",
                    creation_rule="runs.artifacts_json artifact metadata",
                ),
            )
            if "dataset_id" in artifact:
                self._skipped["dataset_metadata_deferred"] += 1

    def _add_failure(self, failure: FailureRecord) -> None:
        node_id = failure_node_id(failure.id)
        self._put_node(
            GraphNode(
                node_id=node_id,
                node_type=NodeType.FAILURE,
                attributes={
                    "run_id": run_node_id(failure.run_id),
                    "error_type": failure.error_type,
                    "severity": failure.severity,
                    "source": failure.source,
                    "tags": _load_json_array(
                        failure.tags_json,
                        table="failures",
                        pk=failure.id,
                        field="tags_json",
                    ),
                    "created_at": failure.created_at,
                    "raw_text_included": False,
                },
                provenance=(
                    provenance(
                        source_table="failures",
                        source_pk=failure.id,
                        source_field="id",
                        creation_rule="failures row",
                    ),
                ),
            )
        )
        self._put_edge(
            EdgeType.OBSERVED_IN,
            node_id,
            run_node_id(failure.run_id),
            EdgeClass.DIRECT,
            provenance(
                source_table="failures",
                source_pk=failure.id,
                source_field="run_id",
                creation_rule="failures.run_id foreign key; observational not causal",
            ),
        )

    def _add_decision(self, decision: DecisionRecord) -> None:
        node_id = decision_node_id(decision.id)
        self._put_node(
            GraphNode(
                node_id=node_id,
                node_type=NodeType.DECISION,
                attributes={
                    "project_id": project_node_id(decision.project_id),
                    "experiment_id": (
                        experiment_node_id(decision.experiment_id)
                        if decision.experiment_id
                        else None
                    ),
                    "has_rationale": decision.rationale is not None,
                    "related_experiment_count": len(
                        _load_json_array(
                            decision.related_experiments_json,
                            table="decisions",
                            pk=decision.id,
                            field="related_experiments_json",
                        )
                    ),
                    "created_at": decision.created_at,
                    "raw_text_included": False,
                },
                provenance=(
                    provenance(
                        source_table="decisions",
                        source_pk=decision.id,
                        source_field="id",
                        creation_rule="decisions row",
                    ),
                ),
            )
        )
        self._put_edge(
            EdgeType.DECISION_IN_PROJECT,
            node_id,
            project_node_id(decision.project_id),
            EdgeClass.DIRECT,
            provenance(
                source_table="decisions",
                source_pk=decision.id,
                source_field="project_id",
                creation_rule="decisions.project_id foreign key",
            ),
        )
        if decision.experiment_id:
            self._put_edge(
                EdgeType.DECISION_IN_EXPERIMENT,
                node_id,
                experiment_node_id(decision.experiment_id),
                EdgeClass.CONDITIONAL_DIRECT,
                provenance(
                    source_table="decisions",
                    source_pk=decision.id,
                    source_field="experiment_id",
                    creation_rule="decisions.experiment_id present",
                ),
            )

    def _add_note(self, note: NoteRecord) -> None:
        node_id = note_node_id(note.id)
        tags = _load_json_array(note.tags_json, table="notes", pk=note.id, field="tags_json")
        self._put_node(
            GraphNode(
                node_id=node_id,
                node_type=NodeType.NOTE,
                attributes={
                    "project_id": project_node_id(note.project_id),
                    "experiment_id": (
                        experiment_node_id(note.experiment_id) if note.experiment_id else None
                    ),
                    "run_id": run_node_id(note.run_id) if note.run_id else None,
                    "tag_count": len(tags),
                    "resolved": note.resolved,
                    "created_at": note.created_at,
                    "raw_text_included": False,
                },
                provenance=(
                    provenance(
                        source_table="notes",
                        source_pk=note.id,
                        source_field="id",
                        creation_rule="notes row",
                    ),
                ),
            )
        )
        if note.run_id:
            self._put_edge(
                EdgeType.NOTE_ON,
                node_id,
                run_node_id(note.run_id),
                EdgeClass.CONDITIONAL_DIRECT,
                provenance(
                    source_table="notes",
                    source_pk=note.id,
                    source_field="run_id",
                    creation_rule="notes.run_id present",
                ),
            )
        if note.experiment_id:
            self._put_edge(
                EdgeType.NOTE_IN_EXPERIMENT,
                node_id,
                experiment_node_id(note.experiment_id),
                EdgeClass.CONDITIONAL_DIRECT,
                provenance(
                    source_table="notes",
                    source_pk=note.id,
                    source_field="experiment_id",
                    creation_rule="notes.experiment_id present",
                ),
            )

    def _add_code_module(self, project: ProjectRecord, tracked_path: TrackedPathRecord) -> None:
        try:
            normalized_path = normalize_project_relative_path(tracked_path.path)
            node_id = code_module_node_id(project.id, normalized_path)
        except (PmemSecurityError, PmemValidationError):
            self._skipped["tracked_path_unsafe"] += 1
            self._warnings.append(f"Skipped unsafe tracked path {tracked_path.id}.")
            return
        path_hash = compute_text_hash(normalized_path)
        self._put_node(
            GraphNode(
                node_id=node_id,
                node_type=NodeType.CODE_MODULE,
                attributes={
                    "project_id": project_node_id(project.id),
                    "path_hash": path_hash,
                    "sha256": tracked_path.sha256,
                    "size_bytes": tracked_path.size_bytes,
                    "tag_present": tracked_path.tag is not None,
                    "raw_path_included": False,
                    "last_checked": tracked_path.last_checked,
                },
                provenance=(
                    provenance(
                        source_table="tracked_paths",
                        source_pk=tracked_path.id,
                        source_field="path",
                        creation_rule="tracked_paths row",
                    ),
                ),
            )
        )
        self._put_edge(
            EdgeType.TRACKS_CODE,
            project_node_id(project.id),
            node_id,
            EdgeClass.CONDITIONAL_DIRECT,
            provenance(
                source_table="tracked_paths",
                source_pk=tracked_path.id,
                source_field="path",
                creation_rule="tracked_paths row",
            ),
        )

    def _put_node(self, node: GraphNode) -> None:
        if not node.provenance:
            raise PmemValidationError("Graph nodes require provenance.")
        self._nodes.setdefault(node.node_id, node)

    def _put_edge(
        self,
        edge_type: EdgeType,
        source: str,
        target: str,
        edge_class: EdgeClass,
        edge_provenance: GraphProvenance,
    ) -> None:
        if edge_class not in DIRECT_INGESTION_EDGE_CLASSES:
            self._skipped[f"edge_class_{edge_class.value}"] += 1
            return
        edge = GraphEdge(
            edge_id=edge_id(edge_type, source, target),
            edge_type=edge_type,
            source=source,
            target=target,
            edge_class=edge_class,
            attributes={"direct_ingestion": True},
            provenance=(edge_provenance,),
        )
        self._edges.setdefault(edge.edge_id, edge)

    def _counts(self) -> dict[str, Any]:
        node_type_counts = Counter(node.node_type.value for node in self._nodes.values())
        edge_type_counts = Counter(edge.edge_type.value for edge in self._edges.values())
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "node_types": dict(sorted(node_type_counts.items())),
            "edge_types": dict(sorted(edge_type_counts.items())),
        }


def _load_json_object(value: str, *, table: str, pk: str, field: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PmemValidationError(f"{table}.{field} for {pk} is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise PmemValidationError(f"{table}.{field} for {pk} must be a JSON object.")
    return parsed


def _load_json_array(value: str, *, table: str, pk: str, field: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PmemValidationError(f"{table}.{field} for {pk} is not valid JSON.") from exc
    if not isinstance(parsed, list):
        raise PmemValidationError(f"{table}.{field} for {pk} must be a JSON array.")
    return parsed


def _stable_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_data(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_stable_data(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_data(item) for item in value]
    return value
