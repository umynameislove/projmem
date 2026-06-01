"""Optional NLP capability gate for failure-analysis layer.

The core CLI must stay small and offline. NLP capability therefore exposes a capability
check that later failure analysis analysis modules can call before importing heavy NLP
packages or loading local models.
"""

from __future__ import annotations

from importlib.util import find_spec

from pmem.errors import PmemValidationError

NLP_EXTRA_MODULES: tuple[str, ...] = ("sentence_transformers",)
NLP_EXTRA_INSTALL_HINT = "projmem[nlp]"


def is_nlp_available() -> bool:
    """Return whether the optional local NLP dependency set is importable."""

    return not missing_nlp_modules()


def missing_nlp_modules() -> tuple[str, ...]:
    """Return optional NLP module import names that are not installed."""

    return tuple(module for module in NLP_EXTRA_MODULES if not _module_available(module))


def require_nlp_extra(feature_name: str) -> None:
    """Raise a safe actionable error when optional NLP support is unavailable."""

    clean_feature = feature_name.strip() or "This feature"
    missing = missing_nlp_modules()
    if not missing:
        return
    modules = ", ".join(missing)
    raise PmemValidationError(
        f"{clean_feature} requires {NLP_EXTRA_INSTALL_HINT}. "
        "Core projmem remains offline/local-first and does not load NLP packages "
        f"by default. Missing module(s): {modules}."
    )


def _module_available(module_name: str) -> bool:
    """Check module availability without importing the module."""

    return find_spec(module_name) is not None
