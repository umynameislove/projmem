"""incremental graph build conservative graph incremental build support.

The current SQLite schema does not expose reliable row-level update/delete
metadata across every graph source table. incremental graph build therefore implements a safe
incremental wrapper: unchanged source fingerprints become no-ops; changed or
unreadable graph artifacts fall back to a full rebuild with explicit metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.errors import PmemError, PmemNotFoundError
from pmem.graph.engine import GraphEngine
from pmem.graph.ingestion import GraphDocument, build_graph_from_project
from pmem.graph.persistence import default_graph_artifact_path, read_graph_document
from pmem.graph.schema import GRAPH_SCHEMA_VERSION
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.utils.hashing import compute_text_hash

GRAPH_INCREMENTAL_METHOD = "conservative-source-fingerprint-v1"

BUILD_MODE_FULL = "full"
BUILD_MODE_INCREMENTAL_NOOP = "incremental_noop"
BUILD_MODE_INCREMENTAL_FALLBACK_FULL = "incremental_fallback_full"

SOURCE_DB_DISPLAY_PATH = ".pmem/pmem.db"

_SOURCE_TABLES: tuple[str, ...] = (
    "projects",
    "experiments",
    "runs",
    "failures",
    "decisions",
    "notes",
    "tracked_paths",
)

_TABLE_ORDER_COLUMNS: dict[str, str] = {
    "projects": "id",
    "experiments": "id",
    "runs": "run_id",
    "failures": "id",
    "decisions": "id",
    "notes": "id",
    "tracked_paths": "id",
}

_SENSITIVE_COLUMNS = frozenset(
    {
        # Store hashed placeholders for any column that can carry user wording,
        # project vocabulary, paths, command context, or free-form JSON.
        "goal",
        "current_objective",
        "name",
        "hypothesis",
        "command",
        "cwd",
        "stdout_path",
        "stderr_path",
        "stdout_preview",
        "stderr_preview",
        "env_json",
        "config_json",
        "metrics_json",
        "artifacts_json",
        "git_json",
        "evaluation_json",
        "failure_candidates_json",
        "error_type",
        "description",
        "root_cause",
        "lesson",
        "rationale",
        "related_experiments_json",
        "author",
        "content",
        "context_json",
        "tags_json",
        "path",
        "tag",
        "metadata_json",
        "target_json",
        "failure_criteria_json",
    }
)


@dataclass(frozen=True, slots=True)
class GraphSourceFingerprint:
    """Privacy-safe fingerprint summary for graph source rows."""

    value: str
    table_counts: dict[str, int]
    computed_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "table_counts": dict(sorted(self.table_counts.items())),
            "computed_at": self.computed_at,
        }


@dataclass(frozen=True, slots=True)
class GraphBuildResult:
    """Result of a full or conservative incremental graph build."""

    document: GraphDocument
    mode: str
    source_changed: bool
    graph_changed: bool
    previous_fingerprint: str | None
    current_fingerprint: str
    table_counts: dict[str, int]
    warnings: tuple[str, ...]
    artifact_path: Path
    should_persist: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "source_changed": self.source_changed,
            "graph_changed": self.graph_changed,
            "previous_fingerprint": self.previous_fingerprint,
            "current_fingerprint": self.current_fingerprint,
            "table_counts": dict(sorted(self.table_counts.items())),
            "warnings": list(self.warnings),
            "should_persist": self.should_persist,
        }


def compute_graph_source_fingerprint(db_path: str | Path) -> GraphSourceFingerprint:
    """Compute a deterministic privacy-preserving fingerprint of graph source tables."""

    path = Path(db_path)
    if not path.exists():
        raise PmemNotFoundError("Project database was not found.")

    connection = connect_database(path)
    try:
        canonical_tables: dict[str, list[dict[str, object]]] = {}
        table_counts: dict[str, int] = {}
        for table in _SOURCE_TABLES:
            rows = _read_source_rows(connection, table)
            canonical_tables[table] = rows
            table_counts[table] = len(rows)
    finally:
        connection.close()

    payload = {
        "method": GRAPH_INCREMENTAL_METHOD,
        "schema_version": GRAPH_SCHEMA_VERSION,
        "tables": canonical_tables,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return GraphSourceFingerprint(
        value=f"sha256:{compute_text_hash(serialized)}",
        table_counts=dict(sorted(table_counts.items())),
        computed_at=_utc_now_iso(),
    )


def build_graph_full(project_root: str | Path) -> GraphBuildResult:
    """Build a fresh graph document from SQLite and mark it as a full rebuild."""

    root = Path(project_root)
    fingerprint = compute_graph_source_fingerprint(project_database_path(root))
    now = _utc_now_iso()
    document = _document_with_metadata(
        _fresh_engine_document(root),
        mode=BUILD_MODE_FULL,
        fingerprint=fingerprint,
        previous_fingerprint=None,
        source_changed=True,
        graph_changed=True,
        incremental_requested=False,
        incremental_applied=False,
        fallback_reason=None,
        now=now,
        warnings=(),
    )
    return GraphBuildResult(
        document=document,
        mode=BUILD_MODE_FULL,
        source_changed=True,
        graph_changed=True,
        previous_fingerprint=None,
        current_fingerprint=fingerprint.value,
        table_counts=fingerprint.table_counts,
        warnings=(),
        artifact_path=default_graph_artifact_path(root),
        should_persist=True,
    )


def build_graph_incremental(project_root: str | Path) -> GraphBuildResult:
    """Build or reuse the graph using conservative source fingerprint checks."""

    root = Path(project_root)
    graph_path = default_graph_artifact_path(root)
    fingerprint = compute_graph_source_fingerprint(project_database_path(root))
    now = _utc_now_iso()

    if not graph_path.exists():
        return _fallback_full_rebuild(
            root,
            fingerprint=fingerprint,
            previous_fingerprint=None,
            fallback_reason="graph artifact missing; full rebuild required",
            now=now,
        )

    try:
        existing = read_graph_document(graph_path)
    except PmemError:
        return _fallback_full_rebuild(
            root,
            fingerprint=fingerprint,
            previous_fingerprint=None,
            fallback_reason="graph artifact unreadable; full rebuild required",
            now=now,
        )

    previous_fingerprint = _metadata_source_fingerprint(existing)
    if existing.schema_version != GRAPH_SCHEMA_VERSION:
        return _fallback_full_rebuild(
            root,
            fingerprint=fingerprint,
            previous_fingerprint=previous_fingerprint,
            fallback_reason="graph schema changed; full rebuild required",
            now=now,
        )
    if previous_fingerprint != fingerprint.value:
        return _fallback_full_rebuild(
            root,
            fingerprint=fingerprint,
            previous_fingerprint=previous_fingerprint,
            fallback_reason=(
                "source fingerprint changed; full rebuild required because row-level "
                "delta safety is not available"
            ),
            now=now,
        )

    warnings = ("Source fingerprint unchanged; graph artifact reused without rewrite.",)
    return GraphBuildResult(
        document=existing,
        mode=BUILD_MODE_INCREMENTAL_NOOP,
        source_changed=False,
        graph_changed=False,
        previous_fingerprint=previous_fingerprint,
        current_fingerprint=fingerprint.value,
        table_counts=fingerprint.table_counts,
        warnings=warnings,
        artifact_path=graph_path,
        should_persist=False,
    )


def _fallback_full_rebuild(
    project_root: Path,
    *,
    fingerprint: GraphSourceFingerprint,
    previous_fingerprint: str | None,
    fallback_reason: str,
    now: str,
) -> GraphBuildResult:
    warnings = (fallback_reason,)
    document = _document_with_metadata(
        _fresh_engine_document(project_root),
        mode=BUILD_MODE_INCREMENTAL_FALLBACK_FULL,
        fingerprint=fingerprint,
        previous_fingerprint=previous_fingerprint,
        source_changed=True,
        graph_changed=True,
        incremental_requested=True,
        incremental_applied=False,
        fallback_reason=fallback_reason,
        now=now,
        warnings=warnings,
    )
    return GraphBuildResult(
        document=document,
        mode=BUILD_MODE_INCREMENTAL_FALLBACK_FULL,
        source_changed=True,
        graph_changed=True,
        previous_fingerprint=previous_fingerprint,
        current_fingerprint=fingerprint.value,
        table_counts=fingerprint.table_counts,
        warnings=warnings,
        artifact_path=default_graph_artifact_path(project_root),
        should_persist=True,
    )


def _fresh_engine_document(project_root: Path) -> GraphDocument:
    ingested = build_graph_from_project(project_root)
    return GraphEngine.from_document(ingested).to_document()


def _document_with_metadata(
    document: GraphDocument,
    *,
    mode: str,
    fingerprint: GraphSourceFingerprint,
    previous_fingerprint: str | None,
    source_changed: bool,
    graph_changed: bool,
    incremental_requested: bool,
    incremental_applied: bool,
    fallback_reason: str | None,
    now: str,
    warnings: tuple[str, ...],
) -> GraphDocument:
    metadata = dict(document.metadata)
    created_at = str(metadata.get("created_at") or now)
    metadata.update(
        {
            "artifact_persistence": True,
            "database_mutation": False,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "build_id": f"graph-build-{compute_text_hash(f'{fingerprint.value}:{now}')[:16]}",
            "build_method": GRAPH_INCREMENTAL_METHOD,
            "build_mode": mode,
            "source_db": SOURCE_DB_DISPLAY_PATH,
            "source_fingerprint": fingerprint.value,
            "source_fingerprint_computed_at": fingerprint.computed_at,
            "previous_source_fingerprint": previous_fingerprint,
            "source_changed": source_changed,
            "graph_changed": graph_changed,
            "source_table_counts": dict(sorted(fingerprint.table_counts.items())),
            "node_count": document.counts.get("nodes", len(document.nodes)),
            "edge_count": document.counts.get("edges", len(document.edges)),
            "ingestion_warnings": list(document.warnings),
            "created_at": created_at,
            "updated_at": now,
            "full_rebuild_at": now,
            "incremental_since": previous_fingerprint,
            "incremental_requested": incremental_requested,
            "incremental_applied": incremental_applied,
            "fallback_reason": fallback_reason,
        }
    )
    combined_warnings = tuple(dict.fromkeys((*document.warnings, *warnings)))
    return GraphDocument(
        schema_version=document.schema_version,
        method=document.method,
        nodes=document.nodes,
        edges=document.edges,
        counts=document.counts,
        warnings=combined_warnings,
        skipped_counts=document.skipped_counts,
        metadata=metadata,
    )


def _metadata_source_fingerprint(document: GraphDocument) -> str | None:
    value = document.metadata.get("source_fingerprint")
    return value if isinstance(value, str) and value.startswith("sha256:") else None


def _read_source_rows(connection: Any, table: str) -> list[dict[str, object]]:
    order_column = _TABLE_ORDER_COLUMNS[table]
    cursor = connection.execute(f"SELECT * FROM {table} ORDER BY {order_column}")  # noqa: S608
    rows: list[dict[str, object]] = []
    for row in cursor.fetchall():
        rows.append(_canonical_row(table, dict(row)))
    return rows


def _canonical_row(table: str, row: dict[str, object]) -> dict[str, object]:
    return {key: _canonical_value(table, key, row[key]) for key in sorted(row)}


def _canonical_value(table: str, column: str, value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    if column in _SENSITIVE_COLUMNS:
        return {"sha256": compute_text_hash(text), "stored": False}
    return text


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
