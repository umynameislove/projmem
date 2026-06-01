"""graph schema graph privacy policy tests."""

from __future__ import annotations

from pmem.graph.privacy import (
    DEFAULT_CONTEXT_OMITTED_FIELDS,
    FORBIDDEN_ADAPTER_IMPORT_PREFIXES,
    GRAPH_CONFIG_DEFAULTS,
    GRAPH_JSON_FILE_MODE,
    GRAPH_JSON_RELATIVE_PATH,
    graph_artifact_policy,
    is_context_field_allowed,
    sanitize_untrusted_context_text,
)


def test_graph_artifact_policy_is_private_and_excluded_from_export_bundle() -> None:
    policy = graph_artifact_policy()

    assert policy["path"] == GRAPH_JSON_RELATIVE_PATH
    assert policy["file_mode_octal"] == "0o600"
    assert policy["file_mode_decimal"] == GRAPH_JSON_FILE_MODE
    assert policy["derived_artifact"] is True
    assert policy["export_bundle_default"] == "exclude"
    assert policy["warning_size_bytes"] == 20 * 1024 * 1024


def test_context_policy_omits_free_text_by_default() -> None:
    for field in DEFAULT_CONTEXT_OMITTED_FIELDS:
        assert not is_context_field_allowed(field)

    assert is_context_field_allowed("failure.description", include_text=True)
    assert is_context_field_allowed("run.exit_code")
    assert is_context_field_allowed("failure.id")


def test_context_text_sanitizer_removes_control_characters() -> None:
    assert sanitize_untrusted_context_text("safe\x00text\x1f\nnext") == "safetext\nnext"


def test_graph_config_defaults_lock_d41_thresholds() -> None:
    assert GRAPH_CONFIG_DEFAULTS["graph"]["max_depth"] == 3
    assert GRAPH_CONFIG_DEFAULTS["graph"]["graph_json_warning_mb"] == 20
    assert GRAPH_CONFIG_DEFAULTS["patterns"]["min_total_runs_for_correlation"] == 20
    assert GRAPH_CONFIG_DEFAULTS["patterns"]["min_runs_per_config_group"] == 10
    assert GRAPH_CONFIG_DEFAULTS["patterns"]["min_failures_for_correlation"] == 5
    assert GRAPH_CONFIG_DEFAULTS["mcp"]["context_token_budget"] == 100000


def test_mcp_fastapi_adapter_policy_forbids_cli_imports() -> None:
    assert "pmem.cli" in FORBIDDEN_ADAPTER_IMPORT_PREFIXES
