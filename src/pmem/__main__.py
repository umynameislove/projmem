"""Executable module entry point for `python -m pmem`.

The console script declared in `pyproject.toml` points directly at
`pmem.cli.app:app`, but this file keeps module execution available for local
debugging and clean-machine smoke tests.
"""

from pmem.cli.app import app

if __name__ == "__main__":
    app()
