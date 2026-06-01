"""mock Claude integration mock Claude + MCP integration tests.

mock Claude integration intentionally does not call a real Claude/API endpoint. The CI contract is:
a mock MCP client requests ``get_context_pack`` over the real stdio JSON-RPC
surface, a deterministic mock Claude response cites project-local entity ids,
and a validator rejects any citation that was not present in the context pack
or backed by SQLite.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from pmem.cli.app import app
from pmem.errors import PmemValidationError
from pmem.graph.schema import run_node_id
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, execute, project_database_path
from pmem.services.graph_operations import build_graph_artifact
from pmem.services.project_init import init_project
from pmem.utils.hashing import compute_text_hash

runner = CliRunner()
NOW_TEXT = "2026-05-31T15:00:00Z"


def test_mock_claude_uses_mcp_context_and_cites_real_project_runs(
    monkeypatch,
    tmp_path,
) -> None:
    """Mock Claude should cite only run ids that MCP supplied from real project data."""

    monkeypatch.chdir(tmp_path)
    _seed_d63_project(tmp_path)
    build_graph_artifact(tmp_path)

    context_pack = _mock_mcp_client_context_pack(tmp_path)
    response = _mock_claude_response(context_pack)
    validated = _validate_mock_claude_response(
        tmp_path,
        response,
        context_pack,
    )
    combined = json.dumps({"context": context_pack, "response": validated}, sort_keys=True)

    assert context_pack["schema_version"] == "mcp-context-pack-v1"
    assert context_pack["recommendations"]["recommendation_count"] == 5
    assert context_pack["database_mutation"] is False
    assert context_pack["network"] is False
    assert validated["schema_version"] == "mock-claude-recommendation-v1"
    assert validated["model_provider"] == "mock"
    assert validated["network"] is False
    assert validated["validated_against_context"] is True
    assert validated["cited_run_ids"]
    assert all(run_id.startswith("run_") for run_id in validated["cited_run_ids"])
    assert "PRIVATE" not in combined
    assert "python train.py" not in combined
    assert "SUPPORTS::" not in combined
    assert "CONTRADICTS::" not in combined
    assert "caused" not in combined.casefold()


def test_validator_rejects_mock_claude_hallucinated_entity_id(
    monkeypatch,
    tmp_path,
) -> None:
    """mock Claude integration should fail closed when a model response invents evidence ids."""

    monkeypatch.chdir(tmp_path)
    _seed_d63_project(tmp_path)
    context_pack = _mock_mcp_client_context_pack(tmp_path)
    response = _mock_claude_response(context_pack)
    hallucinated = dict(response)
    hallucinated["cited_entity_ids"] = [*response["cited_entity_ids"], run_node_id("missing")]
    hallucinated["cited_run_ids"] = [*response["cited_run_ids"], "missing"]

    with pytest.raises(PmemValidationError, match="not present in the MCP context pack"):
        _validate_mock_claude_response(tmp_path, hallucinated, context_pack)


def test_validator_rejects_context_entity_that_is_missing_from_sqlite(
    monkeypatch,
    tmp_path,
) -> None:
    """Verify context citations against SQLite, not just JSON shape."""

    monkeypatch.chdir(tmp_path)
    _seed_d63_project(tmp_path)
    context_pack = _mock_mcp_client_context_pack(tmp_path)
    response = _mock_claude_response(context_pack)
    connection = connect_database(project_database_path(tmp_path))
    try:
        execute(
            connection,
            "DELETE FROM failures WHERE run_id = ?",
            (response["cited_run_ids"][0],),
        )
        execute(connection, "DELETE FROM runs WHERE run_id = ?", (response["cited_run_ids"][0],))
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PmemValidationError, match="was not found in SQLite"):
        _validate_mock_claude_response(tmp_path, response, context_pack)


def test_entity_id_index_includes_future_evidence_buckets() -> None:
    """The mock Claude integration hallucination gate should not hard-code only current buckets."""

    recommendation = {
        "supporting_evidence": [{"entity_id": run_node_id("run_1")}],
        "neutral_evidence": [{"entity_id": run_node_id("run_future")}],
        "warnings": ["not evidence"],
    }

    assert _recommendation_entity_ids(recommendation) == [
        run_node_id("run_1"),
        run_node_id("run_future"),
    ]


def _mock_mcp_client_context_pack(tmp_path) -> dict[str, Any]:
    request = {
        "jsonrpc": "2.0",
        "id": "d63-context",
        "method": "tools/call",
        "params": {
            "name": "get_context_pack",
            "arguments": {"max_items": 10, "token_budget": 100_000},
        },
    }
    result = runner.invoke(app, ["mcp"], input=json.dumps(request) + "\n")
    assert result.exit_code == 0, result.stdout
    response = json.loads(result.stdout)
    assert response["id"] == "d63-context"
    assert response["result"]["isError"] is False
    return json.loads(response["result"]["content"][0]["text"])


def _mock_claude_response(context_pack: dict[str, Any]) -> dict[str, Any]:
    recommendations = context_pack["recommendations"]["recommendations"]
    for recommendation in recommendations:
        cited_entity_ids = _recommendation_entity_ids(recommendation)
        cited_run_ids = [
            entity_id.removeprefix("run:")
            for entity_id in cited_entity_ids
            if entity_id.startswith("run:")
        ]
        if cited_run_ids:
            return {
                "schema_version": "mock-claude-recommendation-v1",
                "model_provider": "mock",
                "network": False,
                "recommendation_id": recommendation["recommendation_id"],
                "answer": (
                    "Review the cited project-local run evidence before choosing the "
                    "next experiment. This is a mock CI response, not a model claim."
                ),
                "cited_entity_ids": cited_entity_ids[:5],
                "cited_run_ids": cited_run_ids[:5],
            }
    raise AssertionError(
        "Seeded mock Claude integration fixture did not produce run-backed recommendations."
    )


def _validate_mock_claude_response(
    tmp_path,
    response: dict[str, Any],
    context_pack: dict[str, Any],
) -> dict[str, Any]:
    cited_entity_ids = _string_list(response, "cited_entity_ids")
    cited_run_ids = _string_list(response, "cited_run_ids")
    if not cited_entity_ids or not cited_run_ids:
        raise PmemValidationError("Mock Claude response must cite at least one run.")
    context_entity_ids = _context_entity_ids(context_pack)
    for entity_id in cited_entity_ids:
        if entity_id not in context_entity_ids:
            raise PmemValidationError(
                "Mock Claude cited an entity_id that was not present in the MCP context pack."
            )
    connection = connect_database(project_database_path(tmp_path))
    try:
        for run_id in cited_run_ids:
            if (
                execute(
                    connection,
                    "SELECT 1 FROM runs WHERE run_id = ? LIMIT 1",
                    (run_id,),
                ).fetchone()
                is None
            ):
                raise PmemValidationError(
                    "Mock Claude cited a run_id that was not found in SQLite."
                )
        for entity_id in cited_entity_ids:
            if entity_id.startswith("failure:"):
                failure_id = entity_id.removeprefix("failure:")
                if (
                    execute(
                        connection,
                        "SELECT 1 FROM failures WHERE id = ? LIMIT 1",
                        (failure_id,),
                    ).fetchone()
                    is None
                ):
                    raise PmemValidationError(
                        "Mock Claude cited a failure_id that was not found in SQLite."
                    )
    finally:
        connection.close()
    return {
        **response,
        "validated_against_context": True,
        "database_mutation": False,
        "raw_text_in_output": False,
    }


def _context_entity_ids(context_pack: dict[str, Any]) -> set[str]:
    recommendations = context_pack["recommendations"]["recommendations"]
    entity_ids: set[str] = set()
    for recommendation in recommendations:
        entity_ids.update(_recommendation_entity_ids(recommendation))
    return entity_ids


def _recommendation_entity_ids(recommendation: dict[str, Any]) -> list[str]:
    entity_ids: list[str] = []
    for value in recommendation.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            entity_id = item.get("entity_id")
            if isinstance(entity_id, str):
                entity_ids.append(entity_id)
    return sorted(set(entity_ids))


def _string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PmemValidationError(f"Mock Claude response {key} must be a list of strings.")
    return value


def _seed_d63_project(tmp_path) -> None:
    init_result = init_project(
        tmp_path,
        project_name="PRIVATE d63 project",
        primary_metric="accuracy",
        metric_direction="max",
    )
    connection = connect_database(project_database_path(tmp_path))
    try:
        experiments = ExperimentRepository(connection)
        runs = RunRepository(connection)
        failures = FailureRepository(connection)
        for experiment_id in ("exp_a", "exp_b", "exp_c"):
            experiments.create(
                experiment_id=experiment_id,
                project_id=init_result.project_id,
                name=f"PRIVATE {experiment_id}",
                created_at=NOW_TEXT,
                updated_at=NOW_TEXT,
            )
        for index, value in enumerate((0.78, 0.79, 0.80, 0.81, 0.82, 0.79, 0.80, 0.81)):
            _create_run(
                runs,
                run_id=f"run_normal_{index}",
                experiment_id="exp_a",
                config={"family": "normal", "index": index},
                metric=value,
                timestamp=f"2026-05-31T15:00:{index:02d}Z",
            )
        _create_run(
            runs,
            run_id="run_outlier_high",
            experiment_id="exp_a",
            config={"family": "outlier"},
            metric=0.99,
            timestamp="2026-05-31T15:00:59Z",
        )
        for index, value in enumerate((0.45, 0.92, 0.50, 0.95)):
            _create_run(
                runs,
                run_id=f"run_var_{index}",
                experiment_id="exp_b",
                config={"family": "variance"},
                metric=value,
                timestamp=f"2026-05-31T15:01:{index:02d}Z",
            )
        for index in range(5):
            run_id = f"run_bad_{index}"
            _create_run(
                runs,
                run_id=run_id,
                experiment_id="exp_c",
                config={"family": "bad"},
                metric=0.30 + index * 0.01,
                timestamp=f"2026-05-31T15:02:{index:02d}Z",
                status="failed",
                exit_code=1,
            )
            failures.create(
                failure_id=f"failure_bad_{index}",
                run_id=run_id,
                error_type="MetricRegression",
                description="PRIVATE failure text",
                root_cause="PRIVATE root cause",
                lesson="PRIVATE lesson",
                severity="high",
                tags=["metric"],
                source="user_confirmed",
                created_at=f"2026-05-31T15:03:{index:02d}Z",
            )
        _create_run(
            runs,
            run_id="run_promote_c",
            experiment_id="exp_c",
            config={"family": "promote"},
            metric=0.90,
            timestamp="2026-05-31T15:04:00Z",
        )
    finally:
        connection.close()


def _create_run(
    runs: RunRepository,
    *,
    run_id: str,
    experiment_id: str,
    config: dict[str, object],
    metric: float,
    timestamp: str,
    status: str = "success",
    exit_code: int = 0,
) -> None:
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    runs.create(
        run_id=run_id,
        experiment_id=experiment_id,
        command="python train.py --PRIVATE",
        cwd=".",
        exit_code=exit_code,
        status=status,
        config=config,
        config_hash=compute_text_hash(config_json),
        metrics={"accuracy": metric},
        artifacts=[
            {
                "path": "outputs/PRIVATE-artifact.txt",
                "sha256": compute_text_hash(run_id),
                "size_bytes": 10,
            }
        ],
        timestamp=timestamp,
    )
