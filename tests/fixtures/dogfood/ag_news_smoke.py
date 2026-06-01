"""Deterministic AG News-style smoke task for `pmem run` dogfooding.

This script is not a model-quality benchmark. It exists to verify that the CLI
captures seed, config, metrics, artifact metadata, stdout/stderr, and Git
metadata without network access or private data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

LABELS = ("World", "Sports", "Business", "Sci/Tech")

EXAMPLES = (
    ("World", "government summit discusses international policy"),
    ("Sports", "team wins final after late goal"),
    ("Business", "market shares rise after earnings report"),
    ("Sci/Tech", "software researchers publish new system design"),
)

KEYWORDS = {
    "World": ("government", "international", "policy"),
    "Sports": ("team", "final", "goal"),
    "Business": ("market", "shares", "earnings"),
    "Sci/Tech": ("software", "researchers", "system"),
}


def main() -> int:
    """Run the deterministic smoke task and write metrics plus one artifact."""

    parser = argparse.ArgumentParser(description="AG News-style dogfood smoke task")
    parser.add_argument("--metrics", required=True, help="Project-relative metrics JSON path")
    parser.add_argument("--artifact", required=True, help="Project-relative report artifact path")
    parser.add_argument("--seed", required=True, help="Seed recorded by the surrounding run")
    args = parser.parse_args()

    start = time.perf_counter()
    predictions = [(label, _predict(text)) for label, text in EXAMPLES]
    accuracy = _accuracy(predictions)
    macro_f1 = _macro_f1(predictions)
    runtime_seconds = time.perf_counter() - start

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "validation_loss": round(1.0 - accuracy, 6),
        "runtime_seconds": round(runtime_seconds, 6),
        "dataset_subset_size": len(EXAMPLES),
    }
    Path(args.metrics).write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")

    report = "\n".join(
        [
            "AG News smoke report",
            "purpose: verify projmem run metadata capture",
            "benchmark: false",
            f"seed: {args.seed}",
            f"examples: {len(EXAMPLES)}",
            f"accuracy: {accuracy:.3f}",
            f"macro_f1: {macro_f1:.3f}",
            "",
        ]
    )
    Path(args.artifact).write_text(report, encoding="utf-8")
    print("ag-news-smoke complete")
    return 0


def _predict(text: str) -> str:
    """Classify a synthetic headline by deterministic keyword match."""

    lowered = text.lower()
    for label in LABELS:
        if any(keyword in lowered for keyword in KEYWORDS[label]):
            return label
    return "World"


def _accuracy(predictions: list[tuple[str, str]]) -> float:
    """Return exact-match accuracy for the synthetic examples."""

    correct = sum(1 for expected, predicted in predictions if expected == predicted)
    return correct / len(predictions)


def _macro_f1(predictions: list[tuple[str, str]]) -> float:
    """Return macro F1 over the four AG News labels."""

    scores = []
    for label in LABELS:
        true_positive = sum(
            1 for expected, predicted in predictions if expected == label and predicted == label
        )
        false_positive = sum(
            1 for expected, predicted in predictions if expected != label and predicted == label
        )
        false_negative = sum(
            1 for expected, predicted in predictions if expected == label and predicted != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


if __name__ == "__main__":
    raise SystemExit(main())
