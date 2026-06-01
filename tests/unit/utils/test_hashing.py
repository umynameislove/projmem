"""Tests for SHA-256 hashing helpers."""

from pathlib import Path

from pmem.utils.hashing import (
    compute_bytes_hash,
    compute_directory_hash,
    compute_file_hash,
    compute_path_hash,
    compute_text_hash,
)


def test_bytes_and_text_hash_use_sha256() -> None:
    """Known SHA-256 output keeps the algorithm contract explicit."""

    expected = (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9c"  # pragma: allowlist secret
        "b410ff61f20015ad"  # pragma: allowlist secret
    )

    assert compute_bytes_hash(b"abc") == expected
    assert compute_text_hash("abc") == expected


def test_file_hash_changes_with_content(tmp_path: Path) -> None:
    """Tracked file hashes should change when file content changes."""

    target = tmp_path / "config.yaml"
    target.write_text("lr: 0.001\n", encoding="utf-8")
    first_hash = compute_file_hash(target)

    target.write_text("lr: 0.01\n", encoding="utf-8")
    second_hash = compute_file_hash(target)

    assert first_hash != second_hash
    assert compute_path_hash(target) == second_hash


def test_directory_hash_is_deterministic(tmp_path: Path) -> None:
    """Directory hashes include sorted relative file paths and content hashes."""

    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")

    first_hash = compute_directory_hash(tmp_path)
    second_hash = compute_directory_hash(tmp_path)

    assert first_hash == second_hash
    assert len(first_hash) == 64
