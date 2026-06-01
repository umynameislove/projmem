"""NLP capability tests for optional NLP capability gates."""

from __future__ import annotations

import sys

import pytest

from pmem.errors import PmemValidationError
from pmem.services import nlp_capability


def test_nlp_capability_reports_available_when_optional_modules_exist(monkeypatch) -> None:
    """The gate should pass without importing heavy NLP modules."""

    monkeypatch.setattr(nlp_capability, "_module_available", lambda name: True)

    assert nlp_capability.is_nlp_available() is True
    assert nlp_capability.missing_nlp_modules() == ()
    nlp_capability.require_nlp_extra("failure embeddings")
    assert "sentence_transformers" not in sys.modules


def test_nlp_capability_reports_missing_dependency_without_core_failure(monkeypatch) -> None:
    """Core projmem should remain usable when the optional NLP extra is absent."""

    monkeypatch.setattr(nlp_capability, "_module_available", lambda name: False)

    assert nlp_capability.is_nlp_available() is False
    assert nlp_capability.missing_nlp_modules() == ("sentence_transformers",)
    with pytest.raises(PmemValidationError) as excinfo:
        nlp_capability.require_nlp_extra("failure embeddings")

    message = str(excinfo.value)
    assert "failure embeddings requires projmem[nlp]" in message
    assert "offline/local-first" in message
    assert "sentence_transformers" in message


def test_nlp_capability_uses_actionable_default_feature_name(monkeypatch) -> None:
    """Blank feature names should still produce a clear public message."""

    monkeypatch.setattr(nlp_capability, "_module_available", lambda name: False)

    with pytest.raises(PmemValidationError, match="This feature requires projmem\\[nlp\\]"):
        nlp_capability.require_nlp_extra("   ")
