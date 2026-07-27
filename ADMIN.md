# Introduction

This documentation provides a guide for `pymatgen` administrators.

## Releases

The general procedure for releasing `pymatgen` comprises the following steps:

1. Make sure all CI checks are green. We don't want to release known bugs.
2. Generate and commit the changelog.
3. Tag and push the clean release commit.
4. Publish the GitHub release from that tag.
5. Make sure the release action publishes the new version to PyPI and conda-forge runs to completion.

## Doing the release

First generate and commit the changelog:

```sh
uv run invoke update-changelog
git add CHANGES.md
git commit -m "Update changelog"
```

Then create and push the release tag, and create its GitHub Release:

```sh
uv run invoke release
```

The package version is derived from the tag, so the tagged commit must have a clean working
tree. The GitHub Release workflow builds and publishes the artifacts. Double check that the
releases are properly done on PyPI.
