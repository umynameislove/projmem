"""Default failure taxonomy and tag normalization.

The taxonomy is intentionally a service-layer suggestion, not a database enum:
users may add their own tags, but local-memory should always normalize tags and
prefer these defaults so portability and failure-analysis clustering receives cleaner data.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict


class FailureTagDefinition(BaseModel):
    """Human-readable metadata for a suggested failure tag."""

    model_config = ConfigDict(frozen=True)

    description: str
    example: str


DEFAULT_FAILURE_TAGS: dict[str, FailureTagDefinition] = {
    "data_quality": FailureTagDefinition(
        description="Label errors, noisy data, missing values, or preprocessing bugs.",
        example="Training set has mislabeled AG News examples.",
    ),
    "config_error": FailureTagDefinition(
        description="Hyperparameter mistake, config mismatch, or typo.",
        example="Learning rate 0.1 was used instead of 0.001.",
    ),
    "oom": FailureTagDefinition(
        description="Out of memory in GPU or RAM.",
        example="Batch size is too large for available memory.",
    ),
    "gradient_issue": FailureTagDefinition(
        description="NaN loss, exploding or vanishing gradients, or instability.",
        example="Loss becomes NaN after epoch 3.",
    ),
    "convergence": FailureTagDefinition(
        description="Model does not converge, oscillates, or plateaus too early.",
        example="Validation loss does not improve after many epochs.",
    ),
    "reproducibility": FailureTagDefinition(
        description="Same config produces meaningfully different results.",
        example="Macro F1 varies widely across seeds.",
    ),
    "environment": FailureTagDefinition(
        description="Dependency, CUDA, OS, or library issue.",
        example="Torch and Transformers versions conflict.",
    ),
    "logic_error": FailureTagDefinition(
        description="Bug in code, formula, or metric computation.",
        example="Accuracy is computed with an off-by-one batch bug.",
    ),
    "timeout": FailureTagDefinition(
        description="Training is too slow or killed by a time/resource quota.",
        example="Job is killed after exceeding runtime budget.",
    ),
    "data_pipeline": FailureTagDefinition(
        description="Data loading, preprocessing, I/O, or corrupt-file failure.",
        example="Data loader crashes on a corrupt row.",
    ),
}

_NON_TAG_CHAR = re.compile(r"[^a-z0-9]+")
_REPEATED_UNDERSCORE = re.compile(r"_+")


def normalize_tag(tag: str) -> str:
    """Normalize a user/default tag to snake_case.

    Unknown tags are still allowed after normalization. Empty tags are rejected
    because they create useless search/filter keys in `tags_json`.
    """

    normalized = tag.strip().lower()
    normalized = _NON_TAG_CHAR.sub("_", normalized)
    normalized = _REPEATED_UNDERSCORE.sub("_", normalized).strip("_")
    if not normalized:
        msg = "failure tags cannot be empty"
        raise ValueError(msg)
    return normalized


def normalize_tags(tags: list[str] | tuple[str, ...]) -> list[str]:
    """Normalize tags and remove duplicates while preserving first-seen order."""

    seen: set[str] = set()
    normalized_tags: list[str] = []
    for tag in tags:
        normalized = normalize_tag(tag)
        if normalized not in seen:
            seen.add(normalized)
            normalized_tags.append(normalized)
    return normalized_tags


def is_default_tag(tag: str) -> bool:
    """Return whether a tag belongs to the built-in taxonomy."""

    return normalize_tag(tag) in DEFAULT_FAILURE_TAGS
