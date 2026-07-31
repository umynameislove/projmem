"""Privacy and text-safety tests for the ``doctor-v1`` contract (DOC-001).

The contract cannot prove that free text is secret-free. What it *can* prove is
that a producer cannot smuggle a filesystem path, a control character or a
terminal escape sequence into a public field, and that is what these tests
lock. The unsafe-content tests below all assert rejection, so no sensitive
marker ever reaches a serialized report through them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import pmem.doctor.model as doctor_model
from pmem.doctor import (
    DoctorCategory,
    DoctorCheckOutcome,
    DoctorCheckResult,
    DoctorProject,
    DoctorSeverity,
    build_doctor_report,
    render_doctor_report_json,
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "doctor" / "doctor_report_v1.json"
)

_UNSAFE_TEXT = (
    "/etc/passwd",
    "/Users/alice/secret/train.py",
    "see /var/log/pmem.log for details",
    "C:\\Users\\alice\\config.json",
    "path:C:/Users/alice",
    "file:///Users/alice/.pmem/pmem.db",
    "line one\nline two",
    "carriage\rreturn",
    "tab\tseparated",
    "\x1b[31mred text\x1b[0m",
    "bell\x07character",
    "delete\x7fcharacter",
)


def _check(**overrides: object) -> DoctorCheckResult:
    payload: dict[str, object] = {
        "check_id": "database.exists",
        "category": DoctorCategory.DATABASE,
        "outcome": DoctorCheckOutcome.PASS,
        "severity": DoctorSeverity.INFO,
        "message": "The project database is present and readable.",
        "remediation": None,
        "related_entity_id": None,
    }
    payload.update(overrides)
    return DoctorCheckResult.model_validate(payload)


# --------------------------------------------------------------------------- #
# Unsafe text is rejected in every public text field                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", _UNSAFE_TEXT)
def test_unsafe_message_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        _check(message=value)


@pytest.mark.parametrize("value", _UNSAFE_TEXT)
def test_unsafe_remediation_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        _check(
            outcome=DoctorCheckOutcome.FAIL,
            severity=DoctorSeverity.ERROR,
            remediation=value,
        )


@pytest.mark.parametrize("value", _UNSAFE_TEXT)
def test_unsafe_project_name_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        DoctorProject(project_id="proj_abc123", project_name=value)


@pytest.mark.parametrize(
    "value",
    ["/etc/passwd", "run id", "run;rm -rf", "run$(id)", "run|tee", "../run_1", "run\x1b[0m"],
)
def test_unsafe_identifier_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        _check(related_entity_id=value)

    with pytest.raises(ValidationError):
        DoctorProject(project_id=value, project_name="AG News baseline")


def test_blank_required_text_is_rejected() -> None:
    for blank in ("", "   ", "\u00a0"):
        with pytest.raises(ValidationError):
            _check(message=blank)


def test_overlong_text_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _check(message="a" * 513)

    with pytest.raises(ValidationError):
        _check(
            outcome=DoctorCheckOutcome.FAIL,
            severity=DoctorSeverity.ERROR,
            remediation="a" * 257,
        )

    with pytest.raises(ValidationError):
        DoctorProject(project_id="proj_abc123", project_name="a" * 121)


def test_overlong_identifier_is_rejected() -> None:
    """Identifiers are bounded too, not only free text."""

    overlong = "run_" + "a" * 130

    with pytest.raises(ValidationError):
        _check(related_entity_id=overlong)

    with pytest.raises(ValidationError):
        DoctorProject(project_id=overlong, project_name="AG News baseline")


# --------------------------------------------------------------------------- #
# Safe-but-tricky text survives without breaking the document                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value",
    [
        'a literal "quote" inside the message',
        'JSON-looking text {"schema_version": "spoofed"}',
        "an https://example.com reference is not a local path",
        "a relative mention of pmem.db without a leading slash",
        "unicode is fine: dự án thử nghiệm",
    ],
)
def test_safe_text_is_accepted(value: str) -> None:
    assert _check(message=value).message == value


def test_json_looking_injection_cannot_spoof_the_document() -> None:
    spoof = 'nested {"schema_version": "doctor-v2", "overall_outcome": "healthy"}'
    report = build_doctor_report(
        project=DoctorProject(project_id="proj_abc123", project_name="AG News baseline"),
        checks=(_check(message=spoof),),
    )

    raw = render_doctor_report_json(report)
    document = json.loads(raw)

    assert document["schema_version"] == "doctor-v1"
    assert document["overall_outcome"] == "healthy"
    assert document["checks"][0]["message"] == spoof
    assert set(document) == {
        "schema_version",
        "project",
        "overall_outcome",
        "summary",
        "checks",
        "database_mutation",
        "network",
        "automatic_repair",
        "raw_text_in_output",
    }


# --------------------------------------------------------------------------- #
# Producer-obligation markers never reach a serialized report                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "leaky",
    [
        "sqlite3.DatabaseError: file is not a database at /Users/alice/.pmem/pmem.db",
        'SELECT * FROM runs WHERE id = "run_1"; -- /tmp/dump.sql',
        'Traceback (most recent call last):\n  File "/src/pmem/app.py", line 1',
        "api_key=sk-live-1234 loaded from /Users/alice/config.json",
    ],
)
def test_leaky_producer_text_never_reaches_the_report(leaky: str) -> None:
    """Raw exception/SQL/config text carries a path or newline, so it is rejected.

    This is exactly the guarantee the contract offers -- and no more. Text that
    leaks a secret *without* a path or control character would pass validation,
    which is why the module docstring places that duty on the producer.
    """

    with pytest.raises(ValidationError):
        _check(message=leaky)


def test_serialized_report_contains_no_control_or_ansi_bytes() -> None:
    report = build_doctor_report(
        project=DoctorProject(project_id="proj_abc123", project_name="AG News baseline"),
        checks=(_check(),),
    )

    raw = render_doctor_report_json(report)

    assert "\x1b" not in raw
    assert not any(ord(char) < 32 and char != "\n" for char in raw)
    assert "\n" in raw  # indentation only; the payload text itself carries none


def test_render_does_not_fall_back_to_str_coercion() -> None:
    """``model_dump_json`` must serialize typed values, never ``default=str``."""

    report = build_doctor_report(
        project=DoctorProject(project_id="proj_abc123", project_name="AG News baseline"),
        checks=(_check(),),
    )
    document = json.loads(render_doctor_report_json(report))

    assert document["project"]["project_id"] == "proj_abc123"
    assert document["checks"][0]["remediation"] is None
    assert document["checks"][0]["related_entity_id"] is None
    assert document["summary"]["total"] == 1
    assert isinstance(document["summary"]["total"], int)


# --------------------------------------------------------------------------- #
# The contract states its own limits honestly                                  #
# --------------------------------------------------------------------------- #
def test_module_docstring_does_not_over_claim_secret_detection() -> None:
    doc = doctor_model.__doc__ or ""

    assert "cannot" in doc
    assert "prove that arbitrary human-authored text is free of secrets" in doc
    for over_claim in (
        "detects every secret",
        "guarantees no secrets",
        "prevents all leaks",
        "redacts all secrets",
    ):
        assert over_claim not in doc


def test_module_docstring_lists_the_producer_obligations() -> None:
    doc = doctor_model.__doc__ or ""

    for obligation in ("raw SQL", "traceback", "config values", "filesystem path"):
        assert obligation in doc, obligation


def test_module_docstring_scopes_determinism_to_validated_input() -> None:
    doc = doctor_model.__doc__ or ""

    assert "same validated report" in doc
    assert "Producers remain responsible" in doc
    assert "no run id" not in doc


def test_golden_fixture_honours_relative_path_producer_obligation() -> None:
    """The public example must not normalize project-relative paths as safe text."""

    raw = _FIXTURE_PATH.read_text(encoding="utf-8")

    for project_content_marker in (".pmem", "src/", "tests/", "artifacts/"):
        assert project_content_marker not in raw


def test_path_detection_is_shared_not_copied() -> None:
    """Doctor must reuse the status text-safety leaf module, not fork its regex."""

    from pmem.status import textsafety

    assert doctor_model.contains_absolute_path is textsafety.contains_absolute_path
    assert doctor_model.contains_control_chars is textsafety.contains_control_chars

    source = Path(doctor_model.__file__).read_text(encoding="utf-8")
    assert "_PATH_SPLIT_RE" not in source
    assert "_WINDOWS_ABSOLUTE_RE" not in source
    assert "file://" not in source
