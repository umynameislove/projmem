"""Deterministic local failure embeddings for failure-analysis layer failure embeddings.

This module intentionally avoids heavyweight NLP dependencies. It uses a stable
hashing-vectorizer style representation so failure-analysis layer can build clustering and
pattern-report foundations while preserving the small core install.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmem.errors import PmemValidationError
from pmem.services.failure_exports import list_failure_records

FAILURE_EMBEDDING_SCHEMA_VERSION = "failure-embedding-v1"
FAILURE_EMBEDDING_METHOD = "local_hashing_tf_l2"
DEFAULT_EMBEDDING_DIMENSION = 128
MIN_EMBEDDING_DIMENSION = 16
MAX_EMBEDDING_DIMENSION = 4096

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def failure_embedding_payload(
    project_root: str | Path,
    *,
    include_text: bool = False,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return deterministic local embeddings for confirmed failures."""

    clean_dimension = validate_embedding_dimension(dimension)
    records = list_failure_records(project_root, include_text=include_text)
    embedded_records = [
        failure_embedding_record(record, include_text=include_text, dimension=clean_dimension)
        for record in records
    ]
    return {
        "schema_version": FAILURE_EMBEDDING_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "method": FAILURE_EMBEDDING_METHOD,
        "dimension": clean_dimension,
        "privacy_mode": "explicit_text_derived" if include_text else "structured_only",
        "include_text": include_text,
        "record_count": len(embedded_records),
        "records": embedded_records,
        "privacy_flags": _privacy_flags(include_text=include_text, record_count=len(records)),
        "algorithm": {
            "tokenization": "lowercase_ascii_word_tokens",
            "features": "structured_tokens_plus_optional_free_text_unigrams_bigrams",
            "hash": "sha256_signed_feature_hashing",
            "normalization": "l2",
            "raw_text_in_output": False,
            "network": False,
        },
    }


def failure_embedding_record(
    record: dict[str, Any],
    *,
    include_text: bool,
    dimension: int,
) -> dict[str, Any]:
    """Embed one failure export failure record without returning raw free text."""

    clean_dimension = validate_embedding_dimension(dimension)
    features = _failure_features(record, include_text=include_text)
    vector = _features_to_vector(features, clean_dimension)
    return {
        "failure_id": str(record["id"]),
        "run_id": str(record["run_id"]),
        "error_type": str(record["error_type"]),
        "severity": str(record["severity"]),
        "source": str(record["source"]),
        "tags": [str(tag) for tag in record.get("tags", []) if str(tag).strip()],
        "created_at": str(record["created_at"]),
        "vector": vector,
        "vector_norm": _vector_norm(vector),
        "source_fields": ("structured", "free_text") if include_text else ("structured",),
        "text_included": False,
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two vectors."""

    if len(left) != len(right):
        raise PmemValidationError("Embedding vectors must have the same dimension.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    score = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return round(score, 10)


def validate_embedding_dimension(value: int) -> int:
    """Validate failure embeddings embedding vector dimension."""

    if value < MIN_EMBEDDING_DIMENSION or value > MAX_EMBEDDING_DIMENSION:
        raise PmemValidationError(
            f"Embedding dimension must be between {MIN_EMBEDDING_DIMENSION} "
            f"and {MAX_EMBEDDING_DIMENSION}."
        )
    return value


def _failure_features(record: dict[str, Any], *, include_text: bool) -> list[tuple[str, float]]:
    features: list[tuple[str, float]] = []
    features.append((f"error_type:{_normalize_token(str(record.get('error_type', '')))}", 2.0))
    features.append((f"severity:{_normalize_token(str(record.get('severity', '')))}", 1.5))
    features.append((f"source:{_normalize_token(str(record.get('source', '')))}", 1.0))
    for tag in record.get("tags", []):
        token = _normalize_token(str(tag))
        if token:
            features.append((f"tag:{token}", 1.8))

    if include_text:
        text = " ".join(
            str(record.get(field) or "")
            for field in ("description", "root_cause", "lesson")
            if record.get(field) is not None
        )
        tokens = _text_tokens(text)
        features.extend((f"text:{token}", 1.0) for token in tokens)
        features.extend(
            (f"bigram:{left}_{right}", 1.2) for left, right in zip(tokens, tokens[1:], strict=False)
        )
    return [(feature, weight) for feature, weight in features if feature.split(":", 1)[1]]


def _features_to_vector(features: list[tuple[str, float]], dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for feature, weight in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 10) for value in vector]


def _text_tokens(value: str) -> list[str]:
    return [_normalize_token(match.group(0)) for match in _TOKEN_RE.finditer(value.casefold())]


def _normalize_token(value: str) -> str:
    return "_".join(token for token in _TOKEN_RE.findall(value.casefold()) if token)


def _vector_norm(vector: list[float]) -> float:
    return round(math.sqrt(sum(value * value for value in vector)), 10)


def _privacy_flags(*, include_text: bool, record_count: int) -> list[dict[str, Any]]:
    if not record_count:
        return []
    if include_text:
        return [
            {
                "code": "derived_from_free_text",
                "severity": "warning",
                "message": "Vectors are derived from failure free text; treat them as private.",
            }
        ]
    return [
        {
            "code": "raw_text_excluded",
            "severity": "info",
            "message": "Vectors use structured failure metadata only by default.",
        }
    ]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
