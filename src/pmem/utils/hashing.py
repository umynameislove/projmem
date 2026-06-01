"""SHA-256 hashing helpers.

entity schema locks SHA-256 as the only hash algorithm for configs, artifacts, and tracked
paths. All future hashing code should call this module rather than using
`hashlib` directly in services.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_SIZE = 1024 * 1024


def compute_bytes_hash(content: bytes) -> str:
    """Return lowercase SHA-256 hex for raw bytes."""

    return hashlib.sha256(content).hexdigest()


def compute_text_hash(content: str) -> str:
    """Return lowercase SHA-256 hex for UTF-8 text."""

    return compute_bytes_hash(content.encode("utf-8"))


def compute_file_hash(path: str | Path) -> str:
    """Return SHA-256 hex for a file, reading in chunks for large artifacts."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_directory_hash(path: str | Path) -> str:
    """Return a deterministic SHA-256 hash for a directory tree.

    The digest includes each relative file path and its content hash in sorted
    order. This is good enough for local-memory tracked-path change detection.
    """

    root = Path(path)
    digest = hashlib.sha256()
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative_path = file_path.relative_to(root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(compute_file_hash(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def compute_path_hash(path: str | Path) -> str:
    """Hash either a file or directory with the local-memory SHA-256 policy."""

    target = Path(path)
    if target.is_dir():
        return compute_directory_hash(target)
    return compute_file_hash(target)
