"""recommendation evidence recommendation evidence linking.

This module verifies that recommendation evidence points to real local graph
nodes backed by SQLite rows. It does not generate recommendations, create graph
edges, expose CLI commands, or read raw failure/decision/note text.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pmem.errors import PmemNotFoundError, PmemValidationError
from pmem.graph.ingestion import GraphDocument, GraphNode, build_graph_from_database_readonly
from pmem.graph.provenance import GraphProvenance
from pmem.graph.schema import EdgeType, NodeType
from pmem.recommendations.model import EvidenceItem, Recommendation
from pmem.repositories.sqlite import connect_database_readonly, execute, project_database_path
from pmem.services.project_context import require_project_context_readonly

RECOMMENDATION_EVIDENCE_LINK_SCHEMA_VERSION = "recommendation-evidence-link-v1"
RECOMMENDATION_EVIDENCE_LINK_METHOD = "graph_sqlite_entity_verification_v1"

_SQLITE_EXISTENCE_QUERIES = {
    "projects": "SELECT 1 FROM projects WHERE id = ? LIMIT 1",
    "experiments": "SELECT 1 FROM experiments WHERE id = ? LIMIT 1",
    "runs": "SELECT 1 FROM runs WHERE run_id = ? LIMIT 1",
    "failures": "SELECT 1 FROM failures WHERE id = ? LIMIT 1",
    "decisions": "SELECT 1 FROM decisions WHERE id = ? LIMIT 1",
    "notes": "SELECT 1 FROM notes WHERE id = ? LIMIT 1",
    "tracked_paths": "SELECT 1 FROM tracked_paths WHERE id = ? LIMIT 1",
}


@dataclass(frozen=True, slots=True)
class LinkedEvidence:
    """One recommendation evidence item verified against graph and SQLite."""

    evidence: EvidenceItem
    graph_provenance: tuple[GraphProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready linked evidence without graph node attributes."""

        return {
            "entity_id": self.evidence.entity_id,
            "entity_type": self.evidence.entity_type.value,
            "source": self.evidence.source.value,
            "summary": self.evidence.summary,
            "graph_verified": True,
            "sqlite_verified": True,
            "sqlite_sources": [
                {
                    "source_table": item.source_table,
                    "source_pk": item.source_pk,
                    "source_field": item.source_field,
                    "creation_rule": item.creation_rule,
                }
                for item in self.graph_provenance
            ],
        }


@dataclass(frozen=True, slots=True)
class RecommendationEvidenceLinks:
    """Verified evidence buckets for one recommendation model recommendation."""

    recommendation_id: str
    supporting_evidence: tuple[LinkedEvidence, ...]
    opposing_evidence: tuple[LinkedEvidence, ...]
    related_failures: tuple[LinkedEvidence, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready evidence-linking data."""

        return {
            "schema_version": RECOMMENDATION_EVIDENCE_LINK_SCHEMA_VERSION,
            "method": RECOMMENDATION_EVIDENCE_LINK_METHOD,
            "recommendation_id": self.recommendation_id,
            "supporting_evidence": [item.to_dict() for item in self.supporting_evidence],
            "opposing_evidence": [item.to_dict() for item in self.opposing_evidence],
            "related_failures": [item.to_dict() for item in self.related_failures],
            "counts": {
                "supporting_evidence": len(self.supporting_evidence),
                "opposing_evidence": len(self.opposing_evidence),
                "related_failures": len(self.related_failures),
                "total_evidence": (
                    len(self.supporting_evidence)
                    + len(self.opposing_evidence)
                    + len(self.related_failures)
                ),
            },
            "warnings": list(self.warnings),
            "database_mutation": False,
            "raw_text_in_output": False,
            "derived_graph_edges": False,
        }


def link_recommendation_evidence(
    project_root: str | Path,
    recommendation: Recommendation,
) -> RecommendationEvidenceLinks:
    """Verify recommendation evidence against a fresh project graph and SQLite."""

    context = require_project_context_readonly(project_root)
    document = build_graph_from_database_readonly(project_database_path(context.root))
    return link_recommendation_evidence_from_document(
        project_database_path(context.root),
        document,
        recommendation,
    )


def link_recommendation_evidence_from_document(
    db_path: str | Path,
    document: GraphDocument,
    recommendation: Recommendation,
) -> RecommendationEvidenceLinks:
    """Verify recommendation evidence against an explicit graph document and DB."""

    database = Path(db_path)
    if not database.exists():
        raise PmemNotFoundError("Project database was not found.")

    nodes = {node.node_id: node for node in document.nodes}
    observed_failure_nodes = {
        edge.source for edge in document.edges if edge.edge_type is EdgeType.OBSERVED_IN
    }
    _reject_duplicate_evidence("supporting_evidence", recommendation.supporting_evidence)
    _reject_duplicate_evidence("opposing_evidence", recommendation.opposing_evidence)
    _reject_duplicate_evidence("related_failures", recommendation.related_failures)

    connection = connect_database_readonly(database)
    try:
        supporting = _link_bucket(connection, nodes, recommendation.supporting_evidence)
        opposing = _link_bucket(connection, nodes, recommendation.opposing_evidence)
        related_failures = _link_bucket(connection, nodes, recommendation.related_failures)
    finally:
        connection.close()

    for item in related_failures:
        if item.evidence.entity_id not in observed_failure_nodes:
            raise PmemValidationError(
                "Recommendation related failure evidence must be observed in a run."
            )

    warnings = _relationship_warnings(supporting, opposing, related_failures)
    return RecommendationEvidenceLinks(
        recommendation_id=recommendation.recommendation_id,
        supporting_evidence=supporting,
        opposing_evidence=opposing,
        related_failures=related_failures,
        warnings=warnings,
    )


def _link_bucket(
    connection: sqlite3.Connection,
    nodes: dict[str, GraphNode],
    evidence_items: Sequence[EvidenceItem],
) -> tuple[LinkedEvidence, ...]:
    linked: list[LinkedEvidence] = []
    for item in evidence_items:
        node = nodes.get(item.entity_id)
        if node is None:
            raise PmemValidationError("Recommendation evidence entity_id was not found in graph.")
        if node.node_type is not item.entity_type:
            raise PmemValidationError(
                "Recommendation evidence entity_type does not match the graph node."
            )
        sqlite_provenance = tuple(
            provenance_item
            for provenance_item in node.provenance
            if _provenance_exists_in_sqlite(connection, provenance_item)
        )
        if not sqlite_provenance:
            raise PmemValidationError(
                "Recommendation evidence entity_id was not verified in SQLite."
            )
        linked.append(LinkedEvidence(evidence=item, graph_provenance=sqlite_provenance))
    return tuple(linked)


def _provenance_exists_in_sqlite(
    connection: sqlite3.Connection,
    graph_provenance: GraphProvenance,
) -> bool:
    query = _SQLITE_EXISTENCE_QUERIES.get(graph_provenance.source_table)
    if query is None:
        raise PmemValidationError("Recommendation evidence uses unsupported provenance table.")
    return execute(connection, query, (graph_provenance.source_pk,)).fetchone() is not None


def _reject_duplicate_evidence(label: str, evidence_items: Sequence[EvidenceItem]) -> None:
    seen: set[tuple[str, NodeType]] = set()
    for item in evidence_items:
        key = (item.entity_id, item.entity_type)
        if key in seen:
            raise PmemValidationError(f"Recommendation {label} contains duplicate evidence.")
        seen.add(key)


def _relationship_warnings(
    supporting: tuple[LinkedEvidence, ...],
    opposing: tuple[LinkedEvidence, ...],
    related_failures: tuple[LinkedEvidence, ...],
) -> tuple[str, ...]:
    warnings: list[str] = []
    if not opposing:
        warnings.append("No opposing evidence was linked for this recommendation.")
    if not related_failures:
        warnings.append("No related failure evidence was linked for this recommendation.")
    supporting_run_count = sum(
        1 for item in supporting if item.evidence.entity_type is NodeType.RUN
    )
    if supporting_run_count == 0:
        warnings.append("No supporting run evidence was linked for this recommendation.")
    return tuple(warnings)
