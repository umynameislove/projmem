"""Focused tests for the shared status text-safety policy."""

import pytest

from pmem.status.textsafety import contains_absolute_path, contains_control_chars


@pytest.mark.parametrize(
    "value",
    [
        "/Users/private/x",
        "path:/Users/private/x",
        r"C:\private\x",
        r"path:C:\private\x",
        "file:///Users/private/x",
        "source=/etc/passwd",
        "(/Users/private/x)",
    ],
)
def test_absolute_filesystem_paths_are_unsafe(value: str) -> None:
    assert contains_absolute_path(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/paper",
        "http://localhost:8000",
        "s3://research-bucket/model",
        "metric:accuracy",
        "ratio:1/2",
        "macro F1",
    ],
)
def test_non_file_text_and_remote_uris_are_safe(value: str) -> None:
    assert contains_absolute_path(value) is False


@pytest.mark.parametrize("value", ["line\nfeed", "tab\tvalue", "delete\x7fvalue"])
def test_ascii_control_characters_are_unsafe(value: str) -> None:
    assert contains_control_chars(value) is True


def test_regular_unicode_text_has_no_control_characters() -> None:
    assert contains_control_chars("Thử nghiệm mô hình") is False
