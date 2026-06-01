"""Graph provenance helpers for graph schema.

Every graph node and edge must point back to observed local evidence. These
helpers intentionally store source table/row metadata only; they do not read
SQLite and they do not infer relationships.
"""

from __future__ import annotations

from dataclasses import dataclass

from pmem.errors import PmemValidationError


@dataclass(frozen=True, slots=True)
class GraphProvenance:
    """Audit metadata for a graph node or edge."""

    source_table: str
    source_pk: str
    source_field: str
    creation_rule: str

    def __post_init__(self) -> None:
        for label, value in (
            ("source_table", self.source_table),
            ("source_pk", self.source_pk),
            ("source_field", self.source_field),
            ("creation_rule", self.creation_rule),
        ):
            _validate_public_token(label, value)

    def to_dict(self) -> dict[str, str]:
        """Return a stable JSON-ready provenance object."""

        return {
            "source_table": self.source_table,
            "source_pk": self.source_pk,
            "source_field": self.source_field,
            "creation_rule": self.creation_rule,
        }


def provenance(
    *,
    source_table: str,
    source_pk: str,
    source_field: str,
    creation_rule: str,
) -> GraphProvenance:
    """Create a validated provenance record."""

    return GraphProvenance(
        source_table=source_table,
        source_pk=source_pk,
        source_field=source_field,
        creation_rule=creation_rule,
    )


def _validate_public_token(label: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PmemValidationError(f"Graph provenance {label} must be a non-empty string.")
    if any(ord(char) < 32 for char in value):
        raise PmemValidationError(f"Graph provenance {label} contains unsafe characters.")
