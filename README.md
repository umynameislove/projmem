# projmem

[![CI](https://github.com/umynameislove/projmem/actions/workflows/ci.yml/badge.svg)](https://github.com/umynameislove/projmem/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://github.com/umynameislove/projmem)
[![Coverage](https://img.shields.io/badge/coverage-95%25%2B-brightgreen)](https://github.com/umynameislove/projmem)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Local-first long-horizon project memory for AI/ML research — no account, no cloud, no telemetry.**

`pmem` is a CLI tool that records *what you did and why*: which files you tracked, every command you ran, configs/metrics/artifacts from each run, Git state at the time, and failure taxonomy for experiments that went wrong. Everything lives in a local SQLite database inside `.pmem/`.

![projmem demo](docs/demo.gif)

> **It doesn't just log experiments — it tells you what to avoid.** After a sweep, `pmem recommend` surfaces evidence-backed candidates like *"avoid `lr=1`"* and screens for config/dataset failure patterns — using **metadata only**, so your raw configs, commands, and datasets never leave your machine.

---

## Why projmem?

| | MLflow / W&B | projmem |
|---|---|---|
| Needs account / server | ✅ | ❌ |
| Works fully offline | ❌ | ✅ |
| Records failure taxonomy | ❌ | ✅ |
| Privacy: no remote URL stored | ❌ | ✅ |
| Secret redaction in config | ❌ | ✅ |
| Dependency footprint | Heavy | `pydantic` + `rich` + `typer` + `networkx` |

projmem is built for researchers and engineers who want experiment memory *without* giving up privacy or requiring infra.

---

## Demo

```
$ pmem init --objective "Train AG News baseline" --metric accuracy \
            --metric-direction max --target 0.9
✓ Initialized project 'my-experiment' in .pmem/

$ pmem track train.py
✓ Tracked train.py (sha256: a3f8c1…)

$ pmem run --name smoke --seed 42 --config config.json \
           --metrics metrics.json -- python train.py
● Running: python train.py
✓ Run completed  exit=0  run_id=r-01j…
  stdout → .pmem/artifacts/runs/r-01j…/stdout.txt
  metrics → .pmem/artifacts/runs/r-01j…/metrics.json

$ pmem run --name failed-run -- python train.py --bad-flag
✗ Run failed  exit=1  run_id=r-02j…
  stderr → .pmem/artifacts/runs/r-02j…/stderr.txt
```

After enough runs, ask projmem what the evidence says — output is metadata-only and never causal-claiming:

```
$ pmem recommend list
rec_… type=avoid confidence=medium evidence=8
  title: Avoid broad reuse of config feature lr=1
  why:   4 confirmed failures share config feature lr=1; no strong successful
         counter-evidence. Treat as an avoid candidate for review, not causal proof.
  next:  Do not reuse lr=1 broadly until the linked runs are reviewed.

$ pmem patterns list          # config-failure, dataset-failure, temporal, anomaly screening
```

## Real usage

projmem is dogfooded on real experiments. In a 30-run Fashion-MNIST
hyperparameter sweep — varying learning rate, optimizer, batch size, and seed,
with deliberate out-of-memory, too-high-learning-rate, and corrupted-dataset
runs — `pmem` recorded every run, captured 16 confirmed failures across the
built-in taxonomy, and then:

- **`pmem recommend`** surfaced an evidence-backed `avoid lr=1` candidate, linked
  to the failed runs that shared that config feature — and *withheld* the signal
  for `lr=0.1`, because strong successful runs were counter-evidence.
- **`pmem patterns`** flagged config-failure and dataset-failure correlations,
  and correctly stayed silent on a low-variance reproducibility group instead of
  inventing an anomaly.
- Every public payload stayed **metadata-only**: scanning the JSON output for
  dataset names, commands, and raw config values returned nothing.

This is a self-test, not a benchmark against other tools. The point is that the
recommendation and privacy guarantees hold on real, messy experiment data.

## Installation

Public PyPI install for the latest published alpha:

```bash
pip install projmem
pmem --help
```

TestPyPI is used only for release rehearsal when preparing a new alpha. The
normal user install path is PyPI:

```bash
pip install -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  projmem==0.4.0a1
pmem --help
```

Development install from source:

```bash
# Requires uv — https://docs.astral.sh/uv/
uv sync --all-groups --no-editable
uv run --no-sync pmem --help
```

If a local editable environment becomes stale, run the CLI through the source
tree while you refresh the environment:

```bash
PYTHONPATH=src uv run python -m pmem --help
```

---

## Quickstart

Initialize a project, track the code that matters, run an experiment, then
review the local memory:

```bash
pmem init --objective "Train a baseline" --metric accuracy \
  --metric-direction max --target 0.9

pmem track train.py
pmem run --name baseline --seed 42 --metrics metrics.json -- python train.py
pmem summary
```

When a run fails, record the confirmed failure explicitly. projmem treats
failure records as evidence for later review, not as automatic root-cause
claims:

```bash
pmem log-failure <run-id> config_error "Wrong label mapping in config"
pmem failures list
```

---

## Commands

| Command | What it does |
|---|---|
| `pmem init` | Create `.pmem/` and record project goal, metric, target |
| `pmem track <path>` | Hash and register a file; `--update` refreshes the hash |
| `pmem run -- <cmd>` | Execute a command and capture stdout/stderr, Git state, exit code |
| `pmem run --seed --config --metrics --artifact --dataset-id` | Capture seed, config, fresh metrics, artifacts, and explicit dataset metadata |
| `pmem log-failure <run-id> <type> <description>` | Store a confirmed failure with the built-in failure taxonomy; supports `--output json` |
| `pmem log-decision <description>` | Store a durable project decision and optional rationale |
| `pmem note <content>` | Store a lightweight project note |
| `pmem baseline <run-id>` | Mark a run as experiment baseline; `--compare` compares another run |
| `pmem summary` | Print project objective, target status, best run, timeline, and warnings |
| `pmem export --json` | Export project memory as deterministic JSON |
| `pmem export-bundle --out <path> [--json]` | Write a deterministic portability bundle; artifact bytes are opt-in |
| `pmem import --dry-run <bundle> [--json]` | Validate a portability bundle without mutating SQLite |
| `pmem import --apply <bundle> --confirm [--json]` | Quarantine a validated bundle as a pending import job |
| `pmem conflict-check <bundle> [--json]` | Detect bundle conflicts before merge or destructive write |
| `pmem resolve <conflict-id> --action <action> [--json]` | Record a non-destructive conflict resolution audit event |
| `pmem share init <path> [--alias] [--json]` | Register an explicit local shared memory path |
| `pmem share status [--json]` | Validate registered shared memory paths |
| `pmem failures list [--json]` | List confirmed failures without raw free text by default |
| `pmem failures export --out <path> [--json]` | Export failure records as deterministic JSON; raw text requires `--include-text --confirm` |
| `pmem failures embed [--json]` | Compute deterministic local failure embeddings without network or vector DB |
| `pmem failures cluster [--json]` | Cluster local failure embeddings and emit a deterministic 2D projection |
| `pmem failures patterns [--json]` | Generate human-reviewable pattern candidates with heuristic labels |
| `pmem failures summary [--json]` | Summarize failure analysis status and top pattern candidates |
| `pmem graph build [--incremental] [--json]` | Build or safely reuse the private local evidence graph at `.pmem/graph.json` |
| `pmem graph status [--json]` | Inspect graph artifact status and counts without exposing graph contents |
| `pmem graph query --node <id> [--json]` | Query graph neighbors, optional paths, and bounded subgraphs |
| `pmem graph lineage --run-id <id> [--json]` | Trace run lineage through direct graph evidence links |
| `pmem graph export --out <path> --confirm [--json]` | Explicitly export the private derived graph artifact for review |
| `pmem patterns list [--json]` | Summarize local pattern reports and candidate counts |
| `pmem patterns config-failure [--json]` | Screen config features associated with confirmed failures; non-causal |
| `pmem patterns dataset-failure [--json]` | Screen explicit dataset metadata associated with failures or metric anomalies |
| `pmem patterns recurring-failures [--json]` | Screen recurring failure candidates with structured metadata by default |
| `pmem patterns temporal [--json]` | Screen metric drift and decision-adjacent metric shifts; non-causal |
| `pmem patterns anomalies [--json]` | Screen metric outliers and same-config high-variance results |
| `pmem recommend list/run/export [--json]` | Review evidence-backed recommendation candidates without raw text |
| `pmem mcp` | Start the local stdio context provider for MCP-style clients |
| `pmem serve [--host 127.0.0.1] [--port 8765]` | Start the localhost-first metadata-only REST adapter |

Most review-oriented commands support `--json` for machine-readable output.

---

## Team Workflow: Bundle-Based Collaboration

projmem currently supports collaboration through explicit export/import bundles,
not realtime sync. Each teammate keeps their own local `.pmem/` database. Do
not put one shared `.pmem/pmem.db` on Google Drive, Dropbox, OneDrive, a network
drive, or a shared folder for multiple people to write at the same time.

Recommended three-person workflow:

1. Each member works locally and records runs, failures, decisions, and notes.
2. Each member exports a reviewable bundle:

   ```bash
   mkdir -p shared
   pmem export-bundle --out shared/member-a-2026-05-26.json --json
   ```

3. The project lead reviews before writing anything:

   ```bash
   pmem import --dry-run shared/member-a-2026-05-26.json --json
   pmem conflict-check shared/member-a-2026-05-26.json --json
   ```

4. Only after review, quarantine-apply the bundle into the lead's local
   project memory:

   ```bash
   pmem import --apply shared/member-a-2026-05-26.json --confirm --json
   ```

5. Use `pmem resolve` to record non-destructive conflict-resolution audit
   events when a bundle has conflicts that need human review.

`pmem share init` registers an explicit local shared path so projmem can inspect
that path later. It does not start a sync daemon, does not create accounts, and
does not make SQLite safe for concurrent writes by multiple users.

```bash
pmem share init ./shared --alias lab-dropbox --mode read_write --json
pmem share status --json
```

## Failure Analysis Workflow

Failure analysis is local and privacy-first. Raw failure text is not exposed by
default. Pattern labels are heuristic audit candidates, not confirmed root
causes.

Suggested flow:

```bash
pmem failures list
pmem failures export --out failures.json --json
pmem failures embed --json
pmem failures cluster --json
pmem failures patterns --json
pmem failures summary --json
```

Use raw text only when you intentionally opt in:

```bash
pmem failures export --out failures-with-text.json --include-text --confirm --json
pmem failures summary --include-text --confirm --json
```

The embedding and clustering path uses deterministic local hashing vectors and
cosine-threshold grouping. It is designed to make repeated failure review easier
without adding a model download, vector database, cloud API, or network call to
the core runtime.

## Pattern Detection Workflow

Pattern commands turn the local memory into reviewable screening reports. They
do not create recommendations, graph `SUPPORTS`/`CONTRADICTS` edges,
root-cause claims, or causal proof.

```bash
pmem patterns list --json
pmem patterns config-failure --json
pmem patterns dataset-failure --json
pmem patterns recurring-failures --json
pmem patterns temporal --json
pmem patterns anomalies --json
```

All pattern commands are local and metadata-first. `recurring-failures` uses
structured failure metadata by default; raw failure text remains opt-in and
requires `--include-text --confirm`.

## Local Evidence Graph Workflow

The evidence graph layer builds a local graph over the existing
`.pmem/pmem.db` records. The graph is a private, rebuildable artifact for audit
and navigation. It does not replace SQLite as canonical memory, does not infer
root causes, and does not provide recommendations.

Suggested flow:

```bash
pmem graph build --json
pmem graph build --incremental --json
pmem graph status --json
pmem graph query --node <node-id> --json
pmem graph lineage --run-id <run-id> --json
```

Use export only when you intentionally want a review copy of the private derived
graph:

```bash
pmem graph export --out graph-review.json --confirm --json
```

Graph edges are evidence links. `OBSERVED_IN` means a failure was recorded for a
run; it is not a causal claim. Derived, optional, and deferred edge classes such
as `SUPPORTS`, `CONTRADICTS`, and `TRAINED_ON` are schema-reserved future work,
not emitted by the current ingestion pipeline.

## Local Demo Plan

The local demo records a small project, builds the evidence graph, runs pattern
and recommendation surfaces, and checks MCP/REST entry points without calling
any cloud service.

```bash
mkdir projmem-demo && cd projmem-demo
printf 'print("demo run")\n' > train.py
printf '{"lr": 0.001, "batch_size": 32}\n' > config.json
printf '{"accuracy": 0.82}\n' > metrics.json

pmem init --objective "projmem demo" --metric accuracy --metric-direction max --target 0.9
pmem track train.py
pmem run --name baseline --seed 1 --config config.json --metrics metrics.json -- python train.py

# Copy the printed run_id into the commands below.
pmem log-failure <run-id> config_error "Demo configuration failure" --tag config
pmem log-decision "Keep lr=0.001 as the demo baseline" --rationale "Lineage demo"
pmem note "Review the demo failure after the next run." --run-id <run-id>

pmem graph build --incremental --json
pmem graph status --json
pmem graph lineage --run-id <run-id> --json
pmem patterns list --json
pmem recommend list --json
pmem mcp --help
pmem serve --help
```

Sparse demo projects may return no recommendation candidates. That is expected:
recommendations require enough project-local evidence and must degrade with an
insufficient-data message rather than fabricate advice.

---

## Architecture

```
pmem run -- python train.py
      │
      ▼
  cli/           ← parse args, render output         (Typer + Rich)
  services/      ← use-case orchestration, transactions
  domain/        ← entities, enums, Pydantic v2 validation
  repositories/  ← SQLite read/write (parameterized queries only)
  migrations/    ← schema versioning, backup, integrity check
  integrations/  ← Git metadata capture (best-effort, no remote URL)
  graph/         ← local evidence graph contracts, engine, query, lineage
  utils/         ← SHA-256 hashing, helpers
```

Dependency direction is strictly one-way: `cli → services → domain / repositories / integrations`. Domain never imports CLI, service, or migration code.

## Quality

- **700+ tests** — unit + integration + security, 95%+ coverage enforced at CI
- **Matrix CI** — Python 3.10 / 3.11 / 3.12 × Ubuntu 22.04 / macOS 14
- **Static analysis** — `ruff` (lint + format) + `pyright` (strict type checking)
- **Pre-commit hooks** — `detect-secrets`, ruff, pyright on every commit
- **Security hardening** — path traversal guard, secret redaction, no remote URL stored

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov=pmem --cov-report=term-missing
uv run pre-commit run --all-files
```

---

## Data Model

`.pmem/pmem.db` — local SQLite, never leaves the machine:

| Table | Stores |
|---|---|
| `projects` | name, goal, objective, metric, target |
| `tracked_paths` | file path, SHA-256 hash, timestamps |
| `experiments` | name, description |
| `runs` | command, exit code, seed, config hash, stdout/stderr preview, Git metadata |
| `failures` | confirmed failure records with severity/source/tags |
| `decisions` | durable decisions with rationale |
| `notes` | lightweight project notes and run/experiment links |
| `experiments.metadata_json` | baseline run id and baseline metrics |
| `export_packages` | bundle manifest/payload hash and local export audit metadata |
| `import_jobs` | pending/quarantined import jobs and dry-run validation reports |
| `shared_paths` | explicit local shared memory path registrations |
| `audit_events` | portability and import/export audit evidence |

Derived graph state is stored separately as `.pmem/graph.json` after
`pmem graph build`. It is a private rebuildable artifact, not canonical memory,
and is excluded from export bundles by default. Incremental build metadata uses
source fingerprints and table counts; project vocabulary from metrics, tags,
failure types, commands, free text, and paths is hashed or omitted rather than
stored as readable metadata.

## Security & Privacy

- All data stays local — `.pmem/` is project-local by design
- Git remote URLs are **never** stored
- Config keys matching `token / password / secret / key` patterns are redacted before DB insert
- Path traversal and command injection are actively guarded against
- `.pmem/` itself cannot be tracked (prevents recursive capture)
- Export bundles are plaintext review artifacts. Share them only with trusted
  collaborators and inspect them before import.
- stdout/stderr previews, failure descriptions, decisions, and notes may contain
  private project context. Keep raw-text exports behind explicit review.

→ [SECURITY.md](SECURITY.md) for full policy.

## Current Limitations

- No realtime sync, cloud sync, account system, team roles, or remote server.
- No owner approval queue or graphical accept/reject UI yet.
- `pmem import --apply` writes to pending/quarantine state and avoids trusted
  local overwrite by default, but final merge policy remains human-reviewed.
- `pmem share` registers and inspects local shared paths; it is not a sync
  engine.
- Failure pattern labels are metadata-first heuristics for audit support, not
  human-confirmed root causes or causal explanations.
- The graph CLI exposes local graph build/status/query/lineage/export
  only. Incremental graph builds no-op on unchanged source fingerprints and
  fall back to full rebuilds when safe row-level deltas cannot be proven. The
  graph CLI does not provide recommendations, MCP, FastAPI, remote graph
  storage, or derived causal edges.
- The graph migration decision keeps projmem on local NetworkX for now: the synthetic
  5K-node P99 query benchmark was below the 2-second Neo4j migration threshold.
  Neo4j remains deferred until benchmark evidence justifies the maintenance
  cost.
- The recommendation engine exposes local recommendation candidates through
  `pmem recommend list`, `pmem recommend run <id>`, and `pmem recommend export`.
  Recommendations remain evidence-scoped audit candidates, not automatic fixes.
- `pmem mcp` exposes a local stdio JSON-RPC context provider for
  MCP-style clients. It is metadata-only by default and does not start HTTP,
  bind a port, call network services, or expose raw failure/decision/note text.
  `pmem serve` adds a secondary FastAPI REST adapter that binds `127.0.0.1` by
  default. It is auth-free for local use only: any process with local access may
  call it. Binding outside loopback requires the explicit
  `--confirm-non-local-bind` override and may expose project metadata on the
  network. The REST surface is intentionally smaller than MCP and does not
  expose the MCP-only context-pack tool.

The current alpha line (`0.4.0a1`) covers the evidence graph, pattern reports,
recommendations, MCP stdio, localhost REST, and mock Claude integration checks.
The implementation is local-first, metadata-first, and still excludes hosted
graph storage and graph UI. Detailed planning, ADR, spec, audit, and tracking
materials are kept outside the public package repo so this repository stays
focused on installable source, tests, CI, and user-facing policy.

---

## Local-First Guarantee

Core commands run **offline** and write only to `.pmem/`:

```
.pmem/
  config.yaml          ← project metadata
  pmem.db              ← SQLite database
  artifacts/runs/      ← stdout, stderr, metrics, artifact files per run
  snapshots/           ← (planned) file snapshots
  graph.json           ← private derived evidence graph, rebuildable
```
