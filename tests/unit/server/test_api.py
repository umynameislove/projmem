"""FastAPI adapter localhost-first FastAPI adapter tests."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from pmem.errors import PmemError, PmemSecurityError, PmemValidationError
from pmem.graph.schema import run_node_id
from pmem.repositories.experiments import ExperimentRepository
from pmem.repositories.failures import FailureRepository
from pmem.repositories.runs import RunRepository
from pmem.repositories.sqlite import connect_database, project_database_path
from pmem.server import api
from pmem.server.api import create_api_app, run_api_server, serve_configuration_payload
from pmem.services.graph_operations import build_graph_artifact
from pmem.services.project_init import init_project
from pmem.utils.hashing import compute_text_hash

NOW = "2026-05-31T14:00:00Z"


def test_default_serve_configuration_is_loopback_only() -> None:
    """FastAPI adapter must bind localhost by default."""

    payload = serve_configuration_payload()

    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 8765
    assert payload["loopback_only"] is True
    assert payload["warnings"] == []
    assert payload["authentication"] == "none"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
def test_non_local_bind_requires_explicit_confirmation(host) -> None:
    """FastAPI adapter should fail closed before exposing project metadata on a network."""

    with pytest.raises(PmemSecurityError, match="confirm-non-local-bind"):
        serve_configuration_payload(host=host)

    payload = serve_configuration_payload(host=host, confirm_non_local_bind=True)

    assert payload["loopback_only"] is False
    assert payload["warnings"]


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost", "127.0.0.2"],
)
def test_loopback_variants_do_not_require_confirmation(host) -> None:
    """Loopback binds should remain usable without a network-exposure override."""

    assert serve_configuration_payload(host=host)["loopback_only"] is True


def test_bind_configuration_rejects_invalid_hosts_and_ports() -> None:
    """FastAPI adapter bind validation should reject unsafe or malformed values."""

    with pytest.raises(PmemValidationError, match="cannot be blank"):
        serve_configuration_payload(host=" ")
    with pytest.raises(PmemSecurityError, match="unsafe"):
        serve_configuration_payload(host="bad\x00host")
    with pytest.raises(PmemValidationError, match="integer"):
        serve_configuration_payload(port=True)
    with pytest.raises(PmemValidationError, match="between 1 and 65535"):
        serve_configuration_payload(port=0)
    with pytest.raises(PmemValidationError, match="between 1 and 65535"):
        serve_configuration_payload(port=65_536)


def test_api_endpoints_return_metadata_only_payloads(tmp_path) -> None:
    """REST endpoints should reuse metadata-only services."""

    _seed_project(tmp_path)
    build_graph_artifact(tmp_path)
    client = TestClient(create_api_app(tmp_path))

    summary = client.get("/summary")
    current = client.get("/current-state")
    recommendations = client.get("/recommendations")
    failures = client.get("/failures")
    neighbors = client.get(
        "/graph/neighbors",
        params={"node_id": run_node_id("run_api"), "edge_type": "USES_CONFIG", "depth": 1},
    )
    lineage = client.get("/graph/lineage", params={"run_id": "run_api"})
    combined = "\n".join(
        response.text
        for response in [summary, current, recommendations, failures, neighbors, lineage]
    )

    assert all(
        response.status_code == 200
        for response in [summary, current, recommendations, failures, neighbors, lineage]
    )
    assert summary.json()["endpoint"] == "summary"
    assert failures.json()["payload"]["records"][0]["text_included"] is False
    assert neighbors.json()["payload"]["found"] is True
    assert lineage.json()["payload"]["lineage"]["run_node_id"] == run_node_id("run_api")
    assert "PRIVATE" not in combined
    assert "python train.py" not in combined
    assert "attributes" not in neighbors.text


def test_api_disables_interactive_docs(tmp_path) -> None:
    """FastAPI adapter should not expose interactive schema surfaces by default."""

    client = TestClient(create_api_app(tmp_path))

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_graph_neighbors_degrades_before_graph_build(tmp_path) -> None:
    """FastAPI adapter graph neighbor REST endpoint should explain a missing graph artifact."""

    _seed_project(tmp_path)
    client = TestClient(create_api_app(tmp_path))

    response = client.get("/graph/neighbors", params={"node_id": run_node_id("run_api")})

    assert response.status_code == 200
    assert response.json()["payload"]["available"] is False
    assert "pmem graph build" in response.text


def test_graph_lineage_missing_artifact_maps_to_safe_404(tmp_path) -> None:
    """FastAPI adapter should translate expected graph errors without tracebacks."""

    _seed_project(tmp_path)
    client = TestClient(create_api_app(tmp_path), raise_server_exceptions=False)

    response = client.get("/graph/lineage", params={"run_id": "run_api"})

    assert response.status_code == 404
    assert response.json()["ok"] is False
    assert "Graph artifact was not found" in response.text
    assert "Traceback" not in response.text


def test_fastapi_query_validation_rejects_out_of_range_inputs(tmp_path) -> None:
    """FastAPI should reject invalid bounded query parameters before dispatch."""

    _seed_project(tmp_path)
    client = TestClient(create_api_app(tmp_path))

    assert client.get("/failures", params={"max_items": 0}).status_code == 422
    assert client.get("/recommendations", params={"max_recommendations": 51}).status_code == 422
    assert (
        client.get(
            "/graph/neighbors",
            params={"node_id": run_node_id("run_api"), "direction": "sideways"},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/graph/neighbors",
            params={"node_id": run_node_id("run_api"), "edge_type": ""},
        ).status_code
        == 422
    )


def test_run_api_server_passes_validated_bind_to_uvicorn(monkeypatch, tmp_path) -> None:
    """The FastAPI adapter runner should use the validated host and port."""

    captured: dict[str, object] = {}

    def fake_run(app, *, host, port, log_level):
        captured.update(app=app, host=host, port=port, log_level=log_level)

    monkeypatch.setattr(api.uvicorn, "run", fake_run)

    payload = run_api_server(tmp_path, host="127.0.0.1", port=9876)

    assert payload["loopback_only"] is True
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9876
    assert captured["log_level"] == "info"


def test_api_status_code_mapping_is_explicit() -> None:
    """FastAPI adapter safe application errors should map to stable HTTP classes."""

    assert api._status_code(PmemSecurityError()) == 403
    assert api._status_code(PmemValidationError()) == 422
    assert api._status_code(PmemError()) == 400


def _seed_project(tmp_path) -> None:
    init_result = init_project(
        tmp_path,
        project_name="PRIVATE api project",
        primary_metric="accuracy",
        metric_direction="max",
    )
    connection = connect_database(project_database_path(tmp_path))
    try:
        experiments = ExperimentRepository(connection)
        runs = RunRepository(connection)
        failures = FailureRepository(connection)
        experiments.create(
            experiment_id="exp_api",
            project_id=init_result.project_id,
            name="PRIVATE experiment",
            hypothesis="PRIVATE hypothesis",
            created_at=NOW,
            updated_at=NOW,
        )
        config_json = json.dumps({"family": "api"}, sort_keys=True, separators=(",", ":"))
        runs.create(
            run_id="run_api",
            experiment_id="exp_api",
            command="python train.py --PRIVATE",
            cwd=".",
            exit_code=1,
            status="failed",
            config={"family": "api"},
            config_hash=compute_text_hash(config_json),
            metrics={"accuracy": 0.4},
            timestamp=NOW,
        )
        failures.create(
            failure_id="failure_api",
            run_id="run_api",
            error_type="ValueError",
            description="PRIVATE failure",
            root_cause="PRIVATE root cause",
            lesson="PRIVATE lesson",
            severity="high",
            tags=["metric"],
            source="user_confirmed",
            created_at=NOW,
        )
    finally:
        connection.close()
