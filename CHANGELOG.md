# Changelog

## Unreleased

### Added

- `pmem status` prints a concise read-only project status on one screen —
  objective and primary metric, run/failure/decision/tracked-path counts, best
  observed run, baseline, evidence-graph freshness, and data-quality warnings —
  followed by exactly one next action with its reason and a suggested command.
  The command is strictly read-only: it never migrates, backs up, `chmod`s, or
  creates the database, it refuses to follow a symlinked database or graph
  artifact, and it reports a stale or checksum-tampered schema as a safe error
  instead of upgrading it.
- `pmem status --json` emits the versioned `status-v1` payload for scripts,
  editors, and assistant integrations. Text and JSON are rendered from the same
  validated model, reducing the risk of semantic drift between the two views.
  The document is deterministic (no timestamps, stable field order, 2-space
  indentation), carries no ANSI decoration, and is followed by exactly one
  newline.
- Recommendation candidates are not generated while rendering status, so
  `recommendations.mode` currently reports `not_evaluated`. A persisted
  recommendation lifecycle is required before status can report active
  recommendation counts.

### Notes

- On failure, `pmem status --json` follows the existing CLI convention — a
  human-readable `Error: ...` line and exit code 1 — and emits no JSON rather
  than a fake envelope that would masquerade as a valid `status-v1` payload.
  Diagnostics currently go to stdout like every other command, so callers must
  check the exit code before parsing stdout.

## 0.4.0a1 (2026-06-02)

### Changed

- Recommendation generation now emits avoid candidates from strong
  config-feature failure signals while suppressing noisy avoid candidates when
  strong successful counter-evidence exists for the same feature.
- `pmem recommend list` now surfaces concise data-quality warnings for dataset
  metadata placement, stale failed-run metrics, and possible failure-label
  mislabels. Text output includes a short `why:` line so users do not need to
  inspect full JSON for the main rationale.
- `pmem run` now avoids stale metrics by attaching `--metrics` JSON only when
  the command creates or updates the file.
- `pmem run` now accepts `--dataset-id` and `--dataset-version` so
  dataset-failure screening has a normal CLI data source.
- Public documentation was refreshed for a clean alpha snapshot with
  feature-based wording and current install metadata.

## 0.3.0a0 (2026-05-31)

### Added

- Local evidence graph ingestion builds an in-memory `GraphDocument` from
  SQLite records with stable node and edge ids, provenance on every graph item,
  and metadata-first payloads.
- Graph engine and persistence add a NetworkX `MultiDiGraph` wrapper,
  deterministic JSON round-trips, atomic `.pmem/graph.json` writes, and private
  file permissions.
- Graph query and lineage APIs provide deterministic neighbors, path search,
  bounded subgraphs, and direct-evidence run lineage.
- `pmem graph` adds build, incremental build, status, query, lineage, and
  confirmed export commands for the private local evidence graph.
- Pattern-mining modules add config-failure correlation, dataset-failure
  screening, recurring failure candidates, temporal analysis, and conservative
  metric anomaly detection.
- `pmem patterns` exposes local pattern reports through privacy-safe text and
  JSON commands.
- Recommendation models, evidence linking, generator logic, and `pmem recommend`
  commands add five evidence-scoped recommendation types: `try_next`, `avoid`,
  `verify`, `promote`, and `investigate`.
- `pmem mcp` adds a local stdio JSON-RPC context provider for mock MCP clients.
- `pmem serve` adds a localhost-first FastAPI adapter over metadata-only project
  state, recommendation, failure, graph-neighbor, and lineage services.
- Mock Claude integration tests exercise the real local MCP stdio context-pack
  path and reject hallucinated or stale cited entity ids.
- The NetworkX-vs-Neo4j migration benchmark uses synthetic 1K/5K/10K node
  graphs. The 5K-node P99 query time was below the documented migration
  threshold, so the project stays on local NetworkX.

### Security

- Graph ingestion and persistence keep graph payloads metadata-first, omit raw
  failure/decision/note text, hash or redact artifact/code paths, validate
  graph shape and provenance, and write `.pmem/graph.json` with private file
  permissions.
- Graph query, lineage, and CLI output stay metadata-only by default and do not
  emit derived causal edges or export graph data without explicit confirmation.
- Incremental graph builds avoid unsafe partial-delta claims: unchanged source
  fingerprints no-op, while changed or unreadable artifacts fall back to full
  rebuilds.
- Pattern-mining reports are local, metadata-first screening candidates. They
  do not mutate SQLite, create `SUPPORTS`/`CONTRADICTS` graph edges, call
  network services, or claim causality/root cause.
- Recommendation generation validates evidence before returning candidates and
  keeps CLI output local, metadata-first, and scoped to project evidence.
- MCP stdio and FastAPI surfaces are metadata-only. MCP uses stdin/stdout only;
  FastAPI binds `127.0.0.1` by default and requires explicit confirmation for
  non-loopback hosts.
- Mock Claude integration treats model output as untrusted and accepts it only
  when cited entity ids are present in the context pack and backed by SQLite.
- Migration benchmark data is synthetic and metadata-only; it does not inspect
  project `.pmem` data, mutate SQLite, call network services, add Neo4j as a
  dependency, or create hosted graph storage.

## 0.2.0a0 (2026-05-23)

### Added

- Portability bundles with dry-run validation, deterministic export hashing,
  opt-in artifact bytes, privacy flags, and field redaction.
- Quarantine import flow with explicit confirmation, import job provenance,
  audit events, and rollback tests.
- Conflict detection and non-destructive conflict-resolution audit records.
- Explicit local shared-memory path registration without a server, cloud sync,
  daemon, or remote URL storage.
- Privacy and SQL-safety regression tests for bundle paths, payload shape,
  provenance, conflict reports, and portability repositories.
- Optional NLP capability gate that keeps heavy NLP imports out of core CLI
  startup.
- Privacy-safe failure list/export commands with explicit
  `--include-text --confirm` handling for raw failure text.
- Deterministic local failure embeddings, cosine-threshold clustering,
  human-reviewable pattern reports, and compact failure-analysis summaries.

### Security

- Import dry-run validation rejects malformed provenance objects and unknown
  provenance keys such as remote URL fields.
- Failure exports remain plaintext and hide raw failure free text by default.
- NLP remains local/optional and must not call remote model APIs by default.
- Failure embeddings and clusters use structured metadata by default; raw text
  analysis requires explicit `--include-text --confirm`.
- Failure pattern reports and summaries use conservative wording and treat
  labels as human-reviewable candidates, not confirmed root causes.

## 0.1.0a0 (2026-05-18)

### Added

- Initial local-first CLI, project initialization, file tracking, run capture,
  failure taxonomy, project memory entities, JSON output contracts, SHA-256
  hashing helpers, SQLite migration runner, database constraints, and CI gates.
