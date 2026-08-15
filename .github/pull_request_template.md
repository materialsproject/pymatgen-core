## Summary

<!-- Briefly describe what this PR does and why. Link any related issues/PRs. -->

## Checklist

- [ ] Tests for the affected code pass locally (e.g. `uv run pytest tests/core -k lattice`
      for a lattice change — the full suite runs in CI)
- [ ] Lint passes: `uv run ruff check . && uv run ruff format --check .`

## Breaking changes?

- [ ] This PR contains **breaking changes** (removals/renames, signature or
      behavior changes, changed defaults/output, dropped Python versions, ...).
      If so, prefix the title with `[breaking]` and describe the change and
      migration steps under `## Breaking Changes` below.

## Breaking Changes

<!-- Optional. Copied verbatim into COMPATIBILITY.md at release time by
     `invoke update-changelog`. Be specific: what changed, what is affected,
     how to migrate. -->
