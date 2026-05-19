# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository scope

This is `pymatgen-core` — the trimmed core of the pymatgen ecosystem (the larger umbrella `pymatgen` package depends on this). It implements the foundational data structures (`Element`, `Site`, `Molecule`, `Structure`, `Lattice`, `Composition`), symmetry/lattice operations, the I/O layer for many DFT/MD codes (VASP, ABINIT, CIF, Gaussian, LAMMPS, CP2K, Q-Chem, …), and analysis modules. Python ≥ 3.11, src layout under `src/pymatgen/`.

## Setup & common commands

Dependency management is **`uv`-based** (see `[tool.uv]` in `pyproject.toml`; default groups are `dev`, `test`, `lint`). Docs are built downstream in the umbrella `pymatgen` repo and there is no `docs/` folder here.

```sh
uv sync                       # install all default groups + project (editable)
uv run pytest tests           # run full test suite (pytest-xdist auto-parallel via addopts)
uv run pytest tests/core/test_structure.py::TestStructure::test_init   # single test
uv run pytest tests/core -k "lattice"   # subset by keyword
uv run ruff check . && uv run ruff format --check .   # lint (matches CI)
uv run mypy -p pymatgen        # type check
uv run pyright src             # secondary type check (CI runs this on src only)
uv run pre-commit run --all-files
```

Build / release helpers live in `tasks.py` (pyinvoke):

```sh
uv run invoke lint             # ruff + mypy + ruff format
uv run invoke update-changelog # generate CHANGES.md from PRs (uses OPENAPI_KEY for GPT summarization)
uv run invoke release          # full PyPI release flow (maintainer-only); release triggers docs rebuild in materialsproject/pymatgen via repository_dispatch
```

## Cython extensions — important

`setup.py` builds two Cython extensions:
- `pymatgen.util.coord_cython` (`src/pymatgen/util/coord_cython.pyx`) — periodic-BC coordinate ops
- `pymatgen.optimization.neighbors` (`src/pymatgen/optimization/neighbors.pyx`) — neighbor finding

After editing any `.pyx`, rebuild before tests see the change:

```sh
uv pip install -e . --no-build-isolation   # or: uv sync --reinstall-package pymatgen-core
```

Worth knowing about how the rebuild works: **every invocation triggers a full rebuild, not just after `.pyx` edits**. uv copies the project into a temp build dir and invokes `setuptools.build_meta`, which runs `build_ext` (Cython → `.c` → `.so`) in that temp dir; the resulting `.so` files are then dropped next to the `.pyx` sources via the editable wheel install. Because the temp build dir starts empty, setuptools' incremental "only recompile if `.pyx` is newer than `.so`" check always misses — you pay a ~2.5 s rebuild every time. `--no-build-isolation` only skips uv's tmp-venv-with-build-deps setup, not the rebuild itself. (If you really want a true no-op when nothing changed, invoke `python setup.py build_ext --inplace` directly — but it's rarely worth it.)

Compiled `.so` files live next to the `.pyx` sources (e.g. `src/pymatgen/optimization/neighbors.cpython-*.so`). If imports look stale after a `.pyx` edit, you forgot the rebuild — the `.so` is what gets imported.

## Tests — non-obvious bits

- `tests/conftest.py` runs every test inside a fresh `tempfile.TemporaryDirectory` (`autouse=True`). Tests should not assume CWD is the repo root.
- The same conftest exposes a `test_files_dir` fixture pointing at `../pymatgen-test-files` (a sibling repo, **not** the in-tree `test-files/` directory). CI sets `PMG_TEST_FILES_DIR=$GITHUB_WORKSPACE/test-files` to redirect to the in-tree copy. Locally, either clone `pymatgen-test-files` as a sibling, symlink it, or set `PMG_TEST_FILES_DIR` to the in-tree `test-files/` path.
- `pyproject.toml` sets `addopts = "-n auto --import-mode=importlib"`. Tests must work under xdist (no shared mutable global state) and importlib mode (no implicit-namespace tricks).
- For matplotlib-using tests in headless contexts, set `MPLBACKEND=Agg` (CI does this).
- **Coverage caveat for optional-dependency files.** `uv sync` only installs the default groups (`dev, test, lint`); it does *not* pull `[project.optional-dependencies]` extras (ase, phonopy, hiphive, netcdf4, …). Test files for those modules begin with `pytest.importorskip(...)` and silently skip when the dep is missing, so a plain local `pytest --cov` reports misleadingly low numbers (e.g. `io/ase.py` shows 18% without `ase`, 95% with it). CI installs `extras: optional` (`uv pip install <wheel>[optional] --group test`), so its coverage numbers are the authoritative ones. To reproduce CI coverage locally, run `uv pip install -e ".[optional]"` before measuring. Note: a few hard-to-install deps (OpenBabel, BoltzTraP, GULP, …) aren't even in `optional` — code gated on those is uniformly untested across all environments and won't be reachable without adding them to a conda-forge install step.

## Architecture — how the pieces fit

The package is intentionally a flat-ish namespace under `pymatgen.*`. The dependency direction generally goes:

```
core  →  symmetry / util  →  io / analysis / electronic_structure / phonon / transformations / alchemy / command_line
```

Key entry points re-exported from `pymatgen.core.__init__` (these are the *stable* public API surface — prefer importing from `pymatgen.core` rather than submodules):
`Structure`, `IStructure`, `Molecule`, `IMolecule`, `Lattice`, `Site`, `PeriodicSite`, `Composition`, `Element`, `Species`, `DummySpecies`, `SymmOp`, `FloatWithUnit`, `ArrayWithUnit`, `Unit`, `get_el_sp`.

Conventions enforced across the codebase:
- **MSONable serialization.** Almost every public class implements `as_dict()` / `from_dict()` (via `monty.json.MSONable`). When adding a new class that holds materials data, follow this pattern — many downstream tools (atomate, FireWorks, Materials Project DB) rely on it.
- **`Structure.from_file()` / `Structure.to(filename)`** are the canonical I/O entry points; format-specific classes live under `pymatgen.io.<format>` and are discovered by extension.
- **Immutable vs mutable variants:** `IStructure`/`IMolecule` are immutable (hashable, suitable for dict keys / sets); `Structure`/`Molecule` are mutable subclasses. New analysis code that doesn't mutate should accept the `I*` form.
- **Settings file:** `pymatgen.core.__init__` loads `~/.config/.pmgrc.yaml` (or `~/.pmgrc.yaml`, or `$PMG_CONFIG_FILE`). Settings like `PMG_VASP_PSP_DIR`, `PMG_MAPI_KEY`, `PMG_DEFAULT_FUNCTIONAL` come from there or from env vars of the same name.
- **CLI entry point:** `pmg = "pymatgen.cli.pmg:main"` (declared in `pyproject.toml`). Wait — note: this CLI module is not in `pymatgen-core` itself; it's referenced for forward compatibility.

## Style & lint

- Ruff config in `pyproject.toml` is **`select = ["ALL"]`** with a long ignore list. Don't fight the included rules (especially `D` pydocstyle Google convention, `I` import sorting). The required `from __future__ import annotations` import is enforced by `isort.required-imports`.
- Line length 120.
- Tests, `src/pymatgen/analysis/*`, `src/pymatgen/io/*`, and `dev_scripts/*` have relaxed docstring rules (see `tool.ruff.lint.per-file-ignores`).
- `mypy` excludes `src/pymatgen/io/cp2k` and `src/pymatgen/io/lammps` — don't waste time chasing type errors there.
- Pre-commit hooks include `codespell` (with a curated `ignore-words-list` for materials-science false positives like `Nd`, `Te`, `CoO`) and `cython-lint` for `.pyx` files.

## Versioning & release

Version lives in `pyproject.toml` as a calver string (`YYYY.M.D`), kept in sync with the latest tag (`v<version>`). The `invoke set-ver`, `invoke update-changelog`, and `invoke release` tasks automate the bump → changelog → tag → PyPI flow. See `ADMIN.md` for the maintainer playbook.
