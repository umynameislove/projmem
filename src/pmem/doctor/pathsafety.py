"""Descriptor-anchored filesystem primitives shared by doctor diagnostics.

Every doctor check that touches the filesystem must resist the same attack: a
path component being replaced by a symlink between the moment it is inspected
and the moment it is opened. The defence is always identical -- never resolve a
path, open each component relative to an already anchored descriptor with
``O_NOFOLLOW``, and prove the opened descriptor still corresponds to the name
that was inspected.

These primitives live in one module so that policy exists exactly once. A second
copy would be free to drift, and a drifted copy of a security policy is worse
than no policy at all: it reads as if it were enforced.

Nothing here opens regular-file content, mutates anything, or retains identity
metadata. Callers own every descriptor they are handed and must close it.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def anchored_traversal_supported() -> bool:
    """Return whether this platform supports descriptor-anchored traversal.

    Windows has neither ``O_NOFOLLOW`` nor ``dir_fd`` support, so a check that
    silently degraded to path-based access there would claim a guarantee it
    cannot provide. Callers must report ``not_applicable`` instead.
    """

    return (
        os.name == "posix"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
    )


def directory_open_flags() -> int:
    """Flags that open a directory itself and reject a symlink at the final component."""

    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def file_open_flags() -> int:
    """Flags that open a regular file for reading without following a link.

    ``O_NONBLOCK`` is included so that a component raced into a FIFO cannot
    block the diagnostic forever on ``open``; the caller still verifies with
    ``fstat`` that what it opened is a regular file.
    """

    flags = os.O_RDONLY | os.O_NOFOLLOW
    return flags | getattr(os, "O_NONBLOCK", 0)


def open_directory(path: Path) -> int:
    """Open an externally supplied directory path, refusing a symlink."""

    return os.open(path, directory_open_flags())


def open_directory_at(parent_fd: int, name: str) -> int:
    """Open a direct child directory relative to an anchored parent descriptor."""

    return os.open(name, directory_open_flags(), dir_fd=parent_fd)


def open_file_at(parent_fd: int, name: str) -> int:
    """Open a direct child file relative to an anchored parent descriptor."""

    return os.open(name, file_open_flags(), dir_fd=parent_fd)


def same_directory_binding(parent_fd: int, name: str, child_fd: int) -> bool:
    """Prove ``name`` still identifies the directory held by ``child_fd``."""

    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(child_fd)
    except OSError:
        return False
    return (
        stat.S_ISDIR(named.st_mode)
        and named.st_dev == opened.st_dev
        and named.st_ino == opened.st_ino
    )


def same_file_binding(parent_fd: int, name: str, child_fd: int) -> bool:
    """Prove ``name`` still identifies the regular file held by ``child_fd``."""

    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(child_fd)
    except OSError:
        return False
    return (
        stat.S_ISREG(named.st_mode)
        and stat.S_ISREG(opened.st_mode)
        and named.st_dev == opened.st_dev
        and named.st_ino == opened.st_ino
    )


def same_path_binding(path: Path, directory_fd: int) -> bool:
    """Prove an externally supplied root path still names its opened directory."""

    try:
        named = path.lstat()
        opened = os.fstat(directory_fd)
    except OSError:
        return False
    return (
        stat.S_ISDIR(named.st_mode)
        and named.st_dev == opened.st_dev
        and named.st_ino == opened.st_ino
    )


def close_quietly(descriptor: int) -> None:
    """Close a descriptor, ignoring a close-time failure.

    A failed ``close`` cannot be acted on by a read-only diagnostic and must not
    mask the result the caller already computed.
    """

    try:
        os.close(descriptor)
    except OSError:
        pass
