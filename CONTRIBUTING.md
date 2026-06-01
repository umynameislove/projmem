# Contributing

## Setup

```bash
uv sync --all-groups --no-editable
```

## Quality Gate

Before committing:

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest --cov=pmem --cov-report=term-missing
uv run pre-commit run --all-files
```

## Coding Rules

- Keep the domain layer independent from SQLite and CLI code.
- CLI code only parses input and renders output.
- Services own multi-step use cases and transaction boundaries.
- Repositories do not contain business rules.
- Migrations must have fresh-database and idempotency tests.
- Do not add heavy core dependencies unless a dogfood task clearly needs them.

## Documentation Rules

- Keep public documentation compact and user-facing.
- README explains install, quickstart, commands, architecture, data model, and quality.
- SECURITY covers privacy and vulnerability handling.
- CHANGELOG records release history.
- Avoid adding planning, audit, roadmap, or AI-prompt notes to the public tree.

## Commit Rule

Each commit should close one gate or one independent part of a gate. Do not mix
schema, CLI, docs, and release config changes unless they are directly related.
