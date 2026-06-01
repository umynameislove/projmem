"""graph schema graph provenance tests."""

from __future__ import annotations

import pytest

from pmem.errors import PmemValidationError
from pmem.graph.provenance import GraphProvenance, provenance


def test_graph_provenance_is_stable_json_ready() -> None:
    record = provenance(
        source_table="failures",
        source_pk="failure_1",
        source_field="run_id",
        creation_rule="failures.run_id foreign key",
    )

    assert record.to_dict() == {
        "source_table": "failures",
        "source_pk": "failure_1",
        "source_field": "run_id",
        "creation_rule": "failures.run_id foreign key",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "source_table": "",
            "source_pk": "failure_1",
            "source_field": "run_id",
            "creation_rule": "rule",
        },
        {
            "source_table": "failures",
            "source_pk": "failure_1",
            "source_field": "run_id\x00",
            "creation_rule": "rule",
        },
    ],
)
def test_graph_provenance_rejects_missing_or_control_character_fields(
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(PmemValidationError):
        GraphProvenance(**kwargs)
