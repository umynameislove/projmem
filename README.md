# projmem

[![CI](https://github.com/umynameislove/projmem/actions/workflows/ci.yml/badge.svg)](https://github.com/umynameislove/projmem/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/projmem)](https://pypi.org/project/projmem/)
[![Downloads](https://img.shields.io/pypi/dm/projmem)](https://pypi.org/project/projmem/)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://github.com/umynameislove/projmem)
[![Coverage](https://img.shields.io/badge/coverage-95%25%2B-brightgreen)](https://github.com/umynameislove/projmem)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Local-first experiment memory for ML — no account, no cloud, no telemetry.**

`pmem` is a CLI that records *what you did and why*: tracked files, every command you
ran, configs/metrics/artifacts per run, Git state, and a failure taxonomy for runs that
went wrong. Everything lives in a local SQLite database inside `.pmem/`.

![projmem demo](docs/demo.gif)

> **It doesn't just log experiments — it tells you what to avoid.** After a sweep,
> `pmem recommend` surfaces evidence-backed candidates like *"avoid `lr=1`"* — using
> **metadata only**, so your raw configs, commands, and datasets never leave your machine.

---

## Why projmem?

| | MLflow / W&B | projmem |
|---|---|---|
| Needs account / server | ✅ | ❌ |
| Works fully offline | ❌ | ✅ |
| Records failure taxonomy | ❌ | ✅ |
| Recommends what to avoid | ❌ | ✅ |
| Metadata-only / privacy-first | ❌ | ✅ |

Built for researchers who want experiment memory *without* giving up privacy or running infra.

## Install

```bash
pip install projmem
pmem --help
```

## Quickstart

```bash
pmem init --objective "Train a baseline" --metric accuracy --metric-direction max --target 0.9
pmem track train.py
pmem run --name baseline --seed 42 --metrics metrics.json -- python train.py
pmem summary

# record a confirmed failure (evidence for later review, not an auto root-cause claim)
pmem log-failure <run-id> config_error "Wrong label mapping in config"
```

After enough runs, ask projmem what the evidence says — output is metadata-only and
never causal-claiming:

```
$ pmem recommend list
rec_… type=avoid confidence=medium evidence=8
  title: Avoid broad reuse of config feature lr=1
  why:   4 confirmed failures share config feature lr=1; no strong successful
         counter-evidence. An avoid candidate for review, not causal proof.
  next:  Do not reuse lr=1 broadly until the linked runs are reviewed.
```

## Real usage

Dogfooded on a 30-run Fashion-MNIST sweep with deliberate OOM, too-high-`lr`, and
corrupted-dataset runs. projmem recorded every run, captured 16 confirmed failures, then:

- **`pmem recommend`** surfaced `avoid lr=1` linked to the failed runs — and *withheld*
  the signal for `lr=0.1`, because successful runs were counter-evidence.
- **`pmem patterns`** flagged config/dataset-failure correlations and stayed silent on a
  low-variance group instead of inventing an anomaly.
- Every public payload stayed **metadata-only** — no dataset names, commands, or raw
  config values leaked.

## Core commands

| Command | What it does |
|---|---|
| `pmem init` | Create `.pmem/` and record goal, metric, target |
| `pmem track <path>` | Hash and register a file |
| `pmem run -- <cmd>` | Run a command; capture stdout/stderr, Git state, exit code, metrics |
| `pmem log-failure <run-id> <type> <desc>` | Store a confirmed failure (built-in taxonomy) |
| `pmem summary` | Project status, best run, timeline, warnings |
| `pmem patterns list` | Config / dataset / temporal / anomaly failure screening |
| `pmem recommend list` | Evidence-backed recommendation candidates |
| `pmem graph build / lineage` | Build and trace the local evidence graph |
| `pmem mcp` / `pmem serve` | Local stdio (MCP) / localhost-only REST adapter |

Run `pmem --help` for the full command surface (export/import bundles, sharing, failure
embeddings, etc.). Most review commands support `--json`.

## Architecture

```
cli/           ← parse args, render output (Typer + Rich)
services/      ← use-case orchestration, transactions
domain/        ← entities, enums, Pydantic v2 validation
repositories/  ← SQLite (parameterized queries only)
graph/         ← local evidence graph: engine, query, lineage
```

Dependency direction is strictly one-way: `cli → services → domain / repositories`.

## Quality

- **700+ tests** — unit + integration + security, 95%+ coverage enforced at CI
- **Matrix CI** — Python 3.10 / 3.11 / 3.12 × Ubuntu / macOS
- **Static analysis** — `ruff` + `pyright` (strict), `detect-secrets` pre-commit
- **Security hardening** — path-traversal guard, secret redaction, no remote URL stored

## Security & privacy

All data stays local in `.pmem/`. Git remote URLs are never stored; secret-like config
keys are redacted before insert; `.pmem/` cannot track itself. Public output
(patterns / recommendations / MCP / REST) is metadata-only — IDs, hashes, counts, scores,
never raw failure text, commands, or config values.

→ See [SECURITY.md](SECURITY.md) for the full policy.

## Limitations

Alpha (`0.4.0a1`). Single-user, local-only: no realtime/cloud sync, accounts, or remote
server. Recommendations and pattern labels are **evidence-scoped review candidates, not
causal proof or automatic fixes**. `pmem serve` is auth-free and loopback-only by design.

## Development

```bash
# requires uv — https://docs.astral.sh/uv/
uv sync --all-groups --no-editable
uv run pytest --cov=pmem
```

MIT licensed. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
