"""Shared, dependency-free text-safety helpers for the status payload.

Both the contract validator (``pmem.status.model``) and the producer redaction
(``pmem.services.status_service``) use this single implementation so the
absolute-path policy cannot drift between consumer and producer.

Guarantee (kept honest): these helpers detect **absolute filesystem paths** and
control characters. They do NOT attempt to detect arbitrary secrets in free
text; producers must not place raw failure/decision/note bodies into the
status payload.
"""

from __future__ import annotations

import re

# Split on punctuation that can precede an embedded path. Non-file URI tokens
# are removed first so ``https://...`` is not mistaken for a local POSIX path.
_PATH_SPLIT_RE = re.compile(r"""[\s()\[\]{}"'<>,;:=|]+""")
_NON_FILE_URI_RE = re.compile(r"(?i)\b(?!file:)[a-z][a-z0-9+.-]*://[^\s\]\[(){}\"'<>]+")
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")


def contains_control_chars(value: str) -> bool:
    """Return whether the string contains any ASCII control character."""

    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def contains_absolute_path(value: str) -> bool:
    """Return whether the string embeds an absolute POSIX/Windows/URI path.

    Detects the path even when it follows a prefix or punctuation, e.g.
    ``/Users/x``, ``see /Users/x``, ``path:/Users/x``, ``src=/Users/x``,
    ``(/Users/x)``, ``"file:///Users/x"``, ``C:\\x`` and ``path:C:\\x``.
    """

    if "file://" in value.lower():
        return True
    if _WINDOWS_ABSOLUTE_RE.search(value):
        return True
    without_remote_uris = _NON_FILE_URI_RE.sub("", value)
    for raw_token in _PATH_SPLIT_RE.split(without_remote_uris):
        token = raw_token.strip()
        if not token:
            continue
        normalized = token.replace("\\", "/")
        if normalized.startswith("/"):
            return True
    return False
