"""Schema v2 portability migration for portability layer."""

from __future__ import annotations

from pmem.migrations.schema_v1 import Migration
from pmem.utils.hashing import compute_text_hash

SCHEMA_V2_SQL = """
CREATE TABLE IF NOT EXISTS export_packages (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(id)) > 0),
    version TEXT NOT NULL CHECK (length(trim(version)) > 0),
    scope_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(scope_json) AND json_type(scope_json) = 'object'),
    manifest_hash TEXT NOT NULL UNIQUE CHECK (
        length(manifest_hash) = 71
        AND substr(manifest_hash, 1, 7) = 'sha256:'
        AND substr(manifest_hash, 8) NOT GLOB '*[^0-9a-f]*'
    ),
    payload_hash TEXT NOT NULL CHECK (
        length(payload_hash) = 71
        AND substr(payload_hash, 1, 7) = 'sha256:'
        AND substr(payload_hash, 8) NOT GLOB '*[^0-9a-f]*'
    ),
    artifact_count INTEGER NOT NULL DEFAULT 0 CHECK (artifact_count >= 0),
    path TEXT NOT NULL CHECK (length(trim(path)) > 0),
    status TEXT NOT NULL DEFAULT 'written'
        CHECK (status IN ('written', 'failed')),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_export_packages_manifest_hash
    ON export_packages(manifest_hash);
CREATE INDEX IF NOT EXISTS idx_export_packages_status_created
    ON export_packages(status, created_at DESC);

CREATE TABLE IF NOT EXISTS import_jobs (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(id)) > 0),
    source_hash TEXT NOT NULL CHECK (
        length(source_hash) = 71
        AND substr(source_hash, 1, 7) = 'sha256:'
        AND substr(source_hash, 8) NOT GLOB '*[^0-9a-f]*'
    ),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'applied', 'failed', 'rolled_back')),
    conflict_count INTEGER NOT NULL DEFAULT 0 CHECK (conflict_count >= 0),
    report_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(report_json) AND json_type(report_json) = 'object'),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) > 0),
    applied_at TEXT,
    provenance_source TEXT
);

CREATE INDEX IF NOT EXISTS idx_import_jobs_source_hash
    ON import_jobs(source_hash);
CREATE INDEX IF NOT EXISTS idx_import_jobs_status
    ON import_jobs(status);
CREATE INDEX IF NOT EXISTS idx_import_jobs_created
    ON import_jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS shared_paths (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(id)) > 0),
    alias TEXT NOT NULL UNIQUE CHECK (length(trim(alias)) > 0),
    path TEXT NOT NULL CHECK (length(trim(path)) > 0),
    mode TEXT NOT NULL DEFAULT 'read_write'
        CHECK (mode IN ('read', 'write', 'read_write')),
    policy_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(policy_json) AND json_type(policy_json) = 'object'),
    last_checked_at TEXT,
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_shared_paths_alias
    ON shared_paths(alias);
CREATE INDEX IF NOT EXISTS idx_shared_paths_mode
    ON shared_paths(mode);

CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(id)) > 0),
    event_type TEXT NOT NULL CHECK (length(trim(event_type)) > 0),
    entity_type TEXT,
    entity_id TEXT,
    before_hash TEXT CHECK (
        before_hash IS NULL
        OR (
            length(before_hash) = 71
            AND substr(before_hash, 1, 7) = 'sha256:'
            AND substr(before_hash, 8) NOT GLOB '*[^0-9a-f]*'
        )
    ),
    after_hash TEXT CHECK (
        after_hash IS NULL
        OR (
            length(after_hash) = 71
            AND substr(after_hash, 1, 7) = 'sha256:'
            AND substr(after_hash, 8) NOT GLOB '*[^0-9a-f]*'
        )
    ),
    actor TEXT,
    timestamp TEXT NOT NULL CHECK (length(trim(timestamp)) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_audit_events_event_type_timestamp
    ON audit_events(event_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_entity
    ON audit_events(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp
    ON audit_events(timestamp DESC);
"""

SCHEMA_V2 = Migration(
    version="0002_phase2_portability",
    sql=SCHEMA_V2_SQL,
    checksum=compute_text_hash(SCHEMA_V2_SQL),
)
