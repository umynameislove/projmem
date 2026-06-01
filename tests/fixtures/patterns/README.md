# Pattern Fixtures

This directory records the synthetic fixture families used by the pattern
detection test suite.

The integration tests still create temporary SQLite projects at runtime so they
can exercise the real repositories and CLI without shipping `.pmem/` databases.
`known_pattern_fixtures.json` is the committed fixture manifest that maps each
synthetic fixture family to its test file, expected signal, and claim boundary.

All fixture entries are metadata-only:

- no raw failure text
- no raw config values
- no raw artifact paths
- no network dependency
- no causal or root-cause truth claim
