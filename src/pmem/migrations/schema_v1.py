"""Schema v1 migration contract."""

from __future__ import annotations

from dataclasses import dataclass

from pmem.utils.hashing import compute_text_hash


@dataclass(frozen=True)
class Migration:
    """A versioned SQL migration."""

    version: str
    sql: str
    checksum: str


SCHEMA_V1_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY NOT NULL CHECK (length(trim(version)) > 0),
    applied_at TEXT NOT NULL CHECK (length(trim(applied_at)) > 0),
    checksum TEXT NOT NULL CHECK (
        length(checksum) = 64 AND checksum NOT GLOB '*[^0-9a-f]*'
    )
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(id)) > 0),
    name TEXT NOT NULL UNIQUE CHECK (length(trim(name)) > 0),
    goal TEXT,
    current_objective TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'archived')),
    primary_metric TEXT,
    metric_direction TEXT
        CHECK (metric_direction IS NULL OR metric_direction IN ('max', 'min')),
    target_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(target_json) AND json_type(target_json) = 'object'),
    failure_criteria_json TEXT NOT NULL DEFAULT '[]'
        CHECK (
            json_valid(failure_criteria_json)
            AND json_type(failure_criteria_json) = 'array'
        ),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) > 0),
    updated_at TEXT NOT NULL CHECK (length(trim(updated_at)) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_projects_status
    ON projects(status);
CREATE INDEX IF NOT EXISTS idx_projects_primary_metric
    ON projects(primary_metric);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(id)) > 0),
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    hypothesis TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'completed', 'abandoned')),
    is_baseline INTEGER NOT NULL DEFAULT 0 CHECK (is_baseline IN (0, 1)),
    primary_metric TEXT,
    target_json TEXT
        CHECK (
            target_json IS NULL
            OR (json_valid(target_json) AND json_type(target_json) = 'object')
        ),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) > 0),
    updated_at TEXT NOT NULL CHECK (length(trim(updated_at)) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object'),
    UNIQUE (project_id, name)
);

CREATE INDEX IF NOT EXISTS idx_experiments_project_status
    ON experiments(project_id, status);
CREATE INDEX IF NOT EXISTS idx_experiments_project_baseline
    ON experiments(project_id)
    WHERE is_baseline = 1 AND status = 'active';
CREATE UNIQUE INDEX IF NOT EXISTS ux_experiments_one_active_baseline
    ON experiments(project_id)
    WHERE is_baseline = 1 AND status = 'active';

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(run_id)) > 0),
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE RESTRICT,
    name TEXT,
    command TEXT NOT NULL CHECK (length(trim(command)) > 0),
    cwd TEXT NOT NULL CHECK (length(trim(cwd)) > 0),
    exit_code INTEGER,
    status TEXT NOT NULL DEFAULT 'unknown'
        CHECK (status IN ('unknown', 'success', 'failed', 'interrupted', 'timeout')),
    duration_sec REAL CHECK (duration_sec IS NULL OR duration_sec >= 0),
    seed TEXT,
    stdout_path TEXT,
    stderr_path TEXT,
    stdout_preview TEXT CHECK (
        stdout_preview IS NULL OR length(stdout_preview) <= 2048
    ),
    stderr_preview TEXT CHECK (
        stderr_preview IS NULL OR length(stderr_preview) <= 2048
    ),
    env_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(env_json) AND json_type(env_json) = 'object'),
    config_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(config_json) AND json_type(config_json) = 'object'),
    config_hash TEXT CHECK (
        config_hash IS NULL
        OR (length(config_hash) = 64 AND config_hash NOT GLOB '*[^0-9a-f]*')
    ),
    metrics_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metrics_json) AND json_type(metrics_json) = 'object'),
    artifacts_json TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(artifacts_json) AND json_type(artifacts_json) = 'array'),
    git_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(git_json) AND json_type(git_json) = 'object'),
    evaluation_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(evaluation_json) AND json_type(evaluation_json) = 'object'),
    failure_candidates_json TEXT NOT NULL DEFAULT '[]'
        CHECK (
            json_valid(failure_candidates_json)
            AND json_type(failure_candidates_json) = 'array'
        ),
    timestamp TEXT NOT NULL CHECK (length(trim(timestamp)) > 0),
    CHECK (status != 'success' OR exit_code IS NULL OR exit_code = 0)
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment_timestamp
    ON runs(experiment_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status_timestamp
    ON runs(status, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp
    ON runs(timestamp DESC);

CREATE TABLE IF NOT EXISTS failures (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(id)) > 0),
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
    error_type TEXT NOT NULL CHECK (length(trim(error_type)) > 0),
    description TEXT NOT NULL CHECK (length(trim(description)) > 0),
    root_cause TEXT,
    lesson TEXT,
    severity TEXT NOT NULL DEFAULT 'medium'
        CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    tags_json TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(tags_json) AND json_type(tags_json) = 'array'),
    source TEXT NOT NULL DEFAULT 'user_confirmed'
        CHECK (source IN ('user_confirmed', 'auto_technical', 'promoted_candidate')),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_failures_run
    ON failures(run_id);
CREATE INDEX IF NOT EXISTS idx_failures_severity_created
    ON failures(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_failures_source_created
    ON failures(source, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_failures_error_type
    ON failures(error_type);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(id)) > 0),
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    experiment_id TEXT REFERENCES experiments(id) ON DELETE RESTRICT,
    description TEXT NOT NULL CHECK (length(trim(description)) > 0),
    rationale TEXT,
    related_experiments_json TEXT NOT NULL DEFAULT '[]'
        CHECK (
            json_valid(related_experiments_json)
            AND json_type(related_experiments_json) = 'array'
        ),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) > 0),
    author TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_project_created
    ON decisions(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_experiment
    ON decisions(experiment_id);

CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(id)) > 0),
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    experiment_id TEXT REFERENCES experiments(id) ON DELETE RESTRICT,
    run_id TEXT REFERENCES runs(run_id) ON DELETE RESTRICT,
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    tags_json TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(tags_json) AND json_type(tags_json) = 'array'),
    context_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(context_json) AND json_type(context_json) = 'object'),
    resolved INTEGER NOT NULL DEFAULT 0 CHECK (resolved IN (0, 1)),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_notes_project_resolved_created
    ON notes(project_id, resolved, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notes_experiment
    ON notes(experiment_id);
CREATE INDEX IF NOT EXISTS idx_notes_run
    ON notes(run_id);

CREATE TABLE IF NOT EXISTS tracked_paths (
    id TEXT PRIMARY KEY NOT NULL CHECK (length(trim(id)) > 0),
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
    path TEXT NOT NULL CHECK (length(trim(path)) > 0),
    tag TEXT,
    hash TEXT NOT NULL CHECK (
        length(hash) = 64 AND hash NOT GLOB '*[^0-9a-f]*'
    ),
    size_bytes INTEGER CHECK (size_bytes IS NULL OR size_bytes >= 0),
    last_checked TEXT NOT NULL CHECK (length(trim(last_checked)) > 0),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) > 0),
    UNIQUE (project_id, path)
);

CREATE INDEX IF NOT EXISTS idx_tracked_paths_project_path
    ON tracked_paths(project_id, path);
CREATE INDEX IF NOT EXISTS idx_tracked_paths_tag
    ON tracked_paths(tag);
"""

SCHEMA_V1 = Migration(
    version="0001_schema_v1",
    sql=SCHEMA_V1_SQL,
    checksum=compute_text_hash(SCHEMA_V1_SQL),
)
