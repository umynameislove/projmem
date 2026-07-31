"""Public ``pmem doctor`` contract and check registry (DOC-001).

The schema/model layer and the inert, typed check registry live here. Both are
dependency-light and side-effect free: importing this package opens no
database, reads no project file, creates nothing, and runs no check. Service
assembly and the CLI land in DOC-006 and are not imported from this package.

Real diagnostics (database integrity, permissions, tracked paths, evidence
freshness) are implemented in DOC-002..DOC-005 against this contract.
"""

from __future__ import annotations

from pmem.doctor.model import (
    DOCTOR_SCHEMA_VERSION,
    DoctorCategory,
    DoctorCheckOutcome,
    DoctorCheckResult,
    DoctorOverallOutcome,
    DoctorProject,
    DoctorReport,
    DoctorSeverity,
    DoctorSummary,
    build_doctor_report,
    render_doctor_report_json,
)
from pmem.doctor.registry import (
    DoctorCheck,
    DoctorCheckContext,
    DoctorCheckDefinition,
    DoctorCheckRegistry,
)

__all__ = [
    "DOCTOR_SCHEMA_VERSION",
    "DoctorCategory",
    "DoctorCheck",
    "DoctorCheckContext",
    "DoctorCheckDefinition",
    "DoctorCheckOutcome",
    "DoctorCheckRegistry",
    "DoctorCheckResult",
    "DoctorOverallOutcome",
    "DoctorProject",
    "DoctorReport",
    "DoctorSeverity",
    "DoctorSummary",
    "build_doctor_report",
    "render_doctor_report_json",
]
