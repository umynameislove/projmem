# Security

## Database Safety

- SQLite database path is project-local: `.pmem/pmem.db`.
- No production credential or connection string is used by projmem.
- Foreign keys are enabled on every managed connection.
- Migration applies in a transaction.
- Existing DB files are backed up before pending migrations.
- `PRAGMA foreign_key_check` and `PRAGMA integrity_check` run after migration.

## Query Safety

- Repository helpers use parameterized queries.
- `ProjectRepository` and `TrackedPathRepository` write user input with SQL
  parameters, not string interpolation.
- Raw SQLite errors are mapped to app-level errors.
- Public error messages must not include SQL, local paths, tokens, or input
  payloads.

## Path Safety

- `pmem track` accepts project-relative regular files only.
- `pmem track --update` refreshes tracked hashes without creating duplicate
  rows.
- `.pmem/`, `.pmem/pmem.db`, and `.pmem/config.yaml` cannot be tracked.
- Case variants such as `.PMEM/`, `.PmEm/`, and `.pMeM/` are also rejected for
  tracking and run metadata.
- Absolute paths and paths resolving outside the project root are rejected.
- Symlink tracking is rejected until a safe follow/no-follow policy exists.
- Directory tracking is rejected until recursive snapshot policy exists.

## Data Integrity

Schema v1 enforces:

- enum/status `CHECK` constraints;
- required JSON validity;
- SHA-256 hash shape;
- FK integrity;
- duplicate prevention for experiment names and tracked paths;
- one active baseline per project.

## Secret Handling

- `.env`, `.pmem/`, build artifacts, coverage files, and editor swap files are
  ignored.
- Word temporary lock files are ignored.
- `.pmem/pmem.db` and `.pmem/config.yaml` are restricted to mode `600` after
  creation on supported POSIX filesystems.
- Pre-commit runs `detect-secrets` with `.secrets.baseline`.
- Current baseline has two audited false positives, both dummy redaction
  fixtures in tests. No real secret is recorded in the baseline.
- Demo artifacts need privacy review before public release.

## Run Capture Privacy

The run-capture privacy boundary is:

- Do not store full environment variables by default.
- Store stdout/stderr previews in SQLite, not full streams.
- Store full streams in project-local artifacts only.
- Git metadata capture must not fail the run and must not store remote URLs.
- Detached HEAD is stored as `detached=true` with `branch=null`.
- `pmem run` executes argv directly and does not use shell interpolation.
- `cwd` is stored as `.` to avoid unnecessary absolute path capture.
- `--config` is explicit opt-in. Secret-like config keys such as `token`,
  `password`, `secret`, `credential`, and `api_key` are redacted before SQLite
  storage.
- `--metrics` accepts only a flat JSON object with primitive values.
- `--artifact` stores path, SHA-256, and `size_bytes`; it does not copy artifact
  content into SQLite.
- Run metadata paths must be project-relative, stay outside `.pmem/`, and avoid
  symlinks.

## Failure Logging Privacy

- Failure `description`, `root_cause`, and `lesson` are free-text fields.
- These fields are stored as plaintext in `.pmem/pmem.db`.
- Do not paste tokens, credentials, API keys, passwords, secrets, private
  dataset samples, or sensitive logs into failure text.
- Record enough technical detail to explain the failure without storing raw
  secrets or private data.
- `pmem failures list` and `pmem failures export` hide raw failure free text by
  default. Raw text requires the explicit `--include-text --confirm` gate.
- Failure exports are plaintext JSON files. Review them before sharing, and keep
  them outside `.pmem/`.

## Optional NLP Privacy

- NLP, embedding, clustering, and pattern analysis are optional local analysis
  features.
- Core projmem must remain usable without NLP packages installed.
- NLP-related code must use lazy capability checks before importing heavy local
  NLP dependencies.
- `pmem failures embed`, `pmem failures cluster`, `pmem failures patterns`, and
  `pmem failures summary` use deterministic local analysis by default, with no
  network call and no vector database.
- The default embedding input is structured failure metadata only. Raw failure
  free text requires `--include-text --confirm`.
- Embeddings, cluster reports, pattern reports, and failure-analysis summaries
  do not include raw text by default, but they are derived project artifacts and
  should still be treated as private.
- Pattern labels are heuristic audit candidates. They are not confirmed root
  causes and should not be published or shared as causal findings without human
  review.
- Remote model APIs, automatic uploads, hosted vector databases, cloud sync, and
  daemonized analysis are outside the default local analysis boundary unless a future security review defines the
  threat model and data boundary.
- Failure text is privacy-sensitive input for any future NLP feature. It must
  not be sent to a network service by default.

## Evidence Graph Privacy

- Evidence graph work starts from local schema contracts and NetworkX. It does
  not require a server, daemon, cloud service, remote model API, or graph
  database.
- `.pmem/graph.json` is a derived private artifact. When graph persistence is
  implemented, it should be written with mode `600`, via a restricted temporary
  file followed by atomic replace.
- `.pmem/graph.json` and future `.pmem/graph/` caches are excluded from
  `pmem export-bundle` by default because canonical memory remains SQLite plus
  explicitly captured artifacts.
- Incremental graph builds are conservative. When the SQLite source fingerprint
  is unchanged, the existing graph artifact is reused; when source rows change
  and row-level delta safety cannot be proven, projmem performs a full rebuild
  rather than applying a partial merge.
- Graph source fingerprints store hashes and table counts only. They must not
  store raw failure, decision, note, command, stdout/stderr, artifact path, code
  path, or absolute local path text.
- Failure, decision, note, stdout/stderr preview, command, and artifact path
  text must be omitted from default graph/MCP context packs. Any future raw-text
  inclusion needs an explicit opt-in and confirmation gate.
- Failure-to-run graph edges use `OBSERVED_IN` wording. Graph outputs must not
  claim causality or root cause unless a future task records explicit
  human-confirmed causal evidence.
- Future MCP/FastAPI adapters must import graph/services/domain contracts, not
  Typer/Rich CLI modules, to avoid circular dependencies and accidental CLI
  rendering leakage.

## Pattern-Mining Privacy

- Config-failure correlation is a local metadata-only analysis over
  `.pmem/pmem.db`.
- Config-failure correlation reads run `config_json` and confirmed failure ids.
  It does not read or
  emit failure descriptions, root causes, lessons, stdout/stderr previews, or
  command strings.
- Sensitive config keys such as token, password, secret, credential, auth, key,
  API, or private are skipped.
- Unsafe config keys or string values, including path-like labels or strings
  containing sensitive substrings, are represented with short SHA-256 labels
  instead of raw text.
- Config-failure correlation reports screening candidates only. It must not claim
  causation, root cause, user recommendation, or confirmed pattern truth.
- Config-failure correlation does not mutate SQLite, create graph
  `SUPPORTS`/`CONTRADICTS` edges, call network services, import cloud APIs, or
  add model/vector database dependencies.
- Dataset-failure correlation uses only explicit artifact `dataset_id`
  metadata, confirmed failure ids, and finite numeric metrics. It does not infer
  dataset identity from artifact paths because paths can expose private data and
  do not prove dataset semantics.
- Dataset-failure correlation redacts unsafe dataset ids, dataset versions,
  and metric names into short SHA-256 labels. It does not emit raw artifact
  paths, failure descriptions, root causes, lessons, command strings,
  stdout/stderr previews, or graph derived edges.
- Recurring failure detection combines local hashing embeddings, seeded failure
  clusters, and graph context. By default it uses structured metadata only,
  emits no raw failure descriptions/root causes/lessons, and does not expose raw
  artifact or code paths.
- Recurring failure outputs are candidates for human review. They must not
  be treated as shared root-cause claims, confirmed causal explanations, or
  recommendations.
- Temporal analysis uses only run timestamps, finite numeric primary metric
  values, and decision ids/timestamps. Decision descriptions and rationales are
  intentionally not read into the report.
- Temporal analysis reports metric drift and before/after decision-shift
  screening candidates only. It must not be treated as causal impact evidence,
  a root-cause finding, or a recommendation.
- Temporal analysis redacts unsafe metric names into short SHA-256 labels and
  emits no raw command strings, stdout/stderr previews, artifact paths, failure
  text, note content, decision rationale, graph derived edges, network calls,
  or database writes.
- Anomaly detection uses only finite numeric metrics, experiment/run ids,
  timestamps, and config hashes. It never emits raw config JSON or config
  values; config identity is represented as a short SHA-256 fingerprint.
- Metric outlier and same-config variance outputs are screening candidates
  for human review. They must not be treated as causal findings,
  recommendations, or proof that a result is unreproducible.
- `pmem patterns` CLI commands wrap pattern reports with metadata-only
  summaries by default. They do not add recommendations, graph
  `SUPPORTS`/`CONTRADICTS` edges, network access, database writes, or causal
  labels.
- `pmem patterns recurring-failures` keeps raw failure text disabled by
  default. Using raw failure text for local vector derivation requires the
  explicit `--include-text --confirm` gate.
- Recommendation models require typed graph entity ids on every evidence item
  and reject edge ids or mismatched entity-type prefixes. The model layer is a
  schema layer only: it does not generate recommendations, verify evidence
  against SQLite, create graph edges, call network services, or expose raw
  text.
- Recommendation evidence linking verifies supporting, opposing, and
  related-failure evidence against both graph nodes and SQLite provenance rows.
  It rejects fabricated evidence ids, unsupported provenance tables, duplicate
  evidence within a bucket, and related failures without an `OBSERVED_IN` graph
  edge. It does not generate recommendations, create graph edges, add CLI/MCP/API
  surfaces, call network services, or read raw failure/decision/note text.
- Recommendation generation must pass every generated candidate through evidence
  linking before returning it. The generator uses project-local
  metadata, finite numeric metrics, config fingerprints, run ids, and failure
  ids only. It does not create graph `SUPPORTS`/`CONTRADICTS` edges, add
  CLI/MCP/API surfaces, call network services, write to SQLite, or read raw
  failure/decision/note text.
- `pmem recommend` renders and exports only metadata-first recommendation
  candidates produced by the generator and rechecked through evidence
  linking. `pmem recommend` output omits raw failure/decision/note text, omits
  command/stdout/stderr content, writes exports with private `0600` file mode,
  and does not create graph edges, call network services, write to SQLite, or
  expose MCP/API surfaces.
- MCP stdio uses stdin/stdout JSON-RPC only. It does not bind a socket,
  start FastAPI, call cloud APIs, write SQLite, or create graph edges. Default
  context packs omit raw project objective/name, failure descriptions/root
  causes/lessons, decision descriptions/rationales, note content, command text,
  stdout/stderr previews, and raw artifact/code paths. Context packs are bounded
  by a conservative token-budget check before they are returned.
- FastAPI is a secondary local adapter over metadata-only services.
  `pmem serve` binds `127.0.0.1` by default, disables interactive API docs, and
  rejects non-loopback hosts unless the user passes
  `--confirm-non-local-bind`. The API has no authentication layer: non-loopback
  overrides can expose project metadata and should be used only with deliberate
  network controls. Localhost is not an authentication boundary: another local
  process can call the API. The REST surface intentionally excludes the
  MCP-only context-pack tool. REST requests do not write SQLite or create graph
  edges.
- Claude integration coverage is mock-only in CI. The test path uses the
  real local `pmem mcp` stdio transport but does not call Claude, Anthropic, or
  any remote model API. Any model response is treated as untrusted until cited
  `entity_id`/`run_id` values are proven present in the MCP context pack and
  backed by SQLite rows. Hallucinated ids must fail closed before a response can
  be accepted for human review.
- Neo4j migration benchmarking uses synthetic metadata-only graph
  documents. It does not inspect project `.pmem/` data, mutate SQLite, call
  network services, add Neo4j as a dependency, or create hosted graph storage.
  The current migration decision is to stay on local NetworkX unless a future
  benchmark exceeds the documented migration threshold.
- Release metadata, tag, and publish steps require explicit owner confirmation
  and must remain tied to the verified source snapshot.

## Decision and Note Privacy

- Decision `description` and `rationale` are free-text fields stored as
  plaintext in `.pmem/pmem.db`.
- Note `content` is a free-text field stored as plaintext in `.pmem/pmem.db`.
- Do not paste tokens, credentials, API keys, passwords, secrets, private
  dataset samples, or sensitive data into these fields.

## Export Privacy

- `pmem export --json` can include plaintext failure, decision, and note fields.
- Export can also include command strings, stdout/stderr previews, and artifact
  metadata already stored in SQLite.
- Export does not copy artifact file contents, but the JSON output should still
  be reviewed before sharing outside the local project.

## Bundle Privacy

- `pmem export-bundle` writes a plaintext JSON bundle. Review it before sharing.
- Artifact bytes are excluded by default. `--include-artifacts` is explicit
  opt-in and can copy project artifact contents into the bundle as base64.
- `--redact-fields` can redact supported free-text fields before bundle hashing
  and writing, but it does not guarantee removal of secrets from unsupported
  fields, filenames, artifact bytes, command arguments, or previews.
- Export bundle Git metadata must not include remote URLs.
- Bundle files should stay project-relative and outside `.pmem/`.

## Import Safety

- `pmem import --dry-run` validates a bundle and previews privacy/conflict
  information without writing to SQLite.
- `pmem import --apply` requires `--confirm` and first passes dry-run
  validation.
- Import apply writes a pending/quarantine `import_jobs` row and an
  `audit_events` row. It does not overwrite trusted local project records by
  default.
- Import apply is transactional. On failure, the database should remain at the
  pre-apply row counts.

## Conflict and Shared Path Safety

- `pmem conflict-check` is read-only. Reports include ids and hashes, not raw
  free-text memory content.
- `pmem resolve` records an audit event only. It does not overwrite trusted
  local project records or apply destructive merge policies.
- Destructive-looking resolution actions such as `take-imported` and
  `overwrite` require `--confirm` and still remain audit-only.
- Shared memory paths are explicit local directories. They are not a server,
  cloud sync, background daemon, watcher, or remote collaboration service.
- Shared path registration rejects traversal segments, `.pmem` path segments,
  symlinks, missing paths, non-directories, and unsafe control characters.
- Normal shared path CLI/JSON output uses privacy-preserving display paths
  rather than absolute local paths.
