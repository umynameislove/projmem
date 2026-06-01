"""Privacy policy constants for evidence graph artifacts and context packs."""

from __future__ import annotations

from typing import Any

GRAPH_JSON_RELATIVE_PATH = ".pmem/graph.json"
GRAPH_JSON_FILE_MODE = 0o600
GRAPH_JSON_WARNING_BYTES = 20 * 1024 * 1024
GRAPH_EXPORT_BUNDLE_DEFAULT = "exclude"
GRAPH_TEXT_INCLUDE_POLICY = "explicit_include_text_confirm_only"
FORBIDDEN_ADAPTER_IMPORT_PREFIXES: tuple[str, ...] = ("pmem.cli",)

DEFAULT_CONTEXT_OMITTED_FIELDS: tuple[str, ...] = (
    "failure.description",
    "failure.root_cause",
    "failure.lesson",
    "decision.description",
    "decision.rationale",
    "note.content",
    "run.command",
    "run.stdout_preview",
    "run.stderr_preview",
    "artifact.path",
)

GRAPH_CONFIG_DEFAULTS: dict[str, dict[str, int]] = {
    "graph": {
        "max_depth": 3,
        "graph_json_warning_mb": 20,
    },
    "patterns": {
        "min_total_runs_for_correlation": 20,
        "min_runs_per_config_group": 10,
        "min_failures_for_correlation": 5,
    },
    "mcp": {
        "context_token_budget": 100000,
    },
}


def graph_artifact_policy() -> dict[str, Any]:
    """Return the graph schema graph artifact policy as JSON-ready data."""

    return {
        "path": GRAPH_JSON_RELATIVE_PATH,
        "file_mode_octal": "0o600",
        "file_mode_decimal": GRAPH_JSON_FILE_MODE,
        "write_policy": "restricted-temp-file-then-atomic-replace",
        "export_bundle_default": GRAPH_EXPORT_BUNDLE_DEFAULT,
        "derived_artifact": True,
        "warning_size_bytes": GRAPH_JSON_WARNING_BYTES,
    }


def is_context_field_allowed(field_path: str, *, include_text: bool = False) -> bool:
    """Return whether a field may enter graph/MCP context by default."""

    normalized = field_path.strip().lower()
    if include_text:
        return bool(normalized)
    return normalized not in DEFAULT_CONTEXT_OMITTED_FIELDS


def sanitize_untrusted_context_text(value: str) -> str:
    """Strip unsafe control characters before any future explicit text context."""

    return "".join(char for char in value if char == "\n" or ord(char) >= 32)
