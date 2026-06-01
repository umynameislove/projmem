"""Service layer for user-facing use cases.

Services coordinate CLI requests, domain validation, repositories, integrations,
and summary logic. The database database service creates or migrates `.pmem/pmem.db`;
future commands should continue to call services instead of writing SQL in CLI.
"""
