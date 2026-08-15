# Compatibility

This file documents **breaking changes** to `pymatgen-core`: removals, renames,
signature changes, changed defaults or output, dropped Python versions, and any
other change that may require action when upgrading.

It is the `pymatgen-core` counterpart of the retired
[`docs/compatibility.md`](https://github.com/materialsproject/pymatgen/blob/main/docs/compatibility.md)
from the umbrella `pymatgen` repository.

## How this file is maintained

Entries are **not** hand-written with placeholder versions. The workflow is:

1. **At PR time.** A PR that breaks something is titled `[breaking] ...` and
   describes the change (plus migration steps) under `## Breaking Changes` in
   its body — see the [pull request template](.github/pull_request_template.md).
2. **At release time.** `invoke update-changelog` (see `tasks.py`) collects every
   PR merged since the last release whose title starts with `[breaking]` and
   writes them into this file under the **actual** release version header.

So the version header is always the real release version, never a placeholder
like `v2025.?.?`.

## What counts as a breaking change?

- Removal or rename of a public class, function, method, attribute, or argument.
- Change of a signature, default value, or return type.
- Change of output format or behavior that existing code may rely on.
- Dropping support for a Python or dependency version.

Non-breaking **deprecations** (still functional, emitting a `DeprecationWarning`
with a deadline) should be announced in `CHANGES.md` instead; the eventual
removal PR must reference the deprecation.

## Recent Breaking Changes

### v2026.8.13

- `Locpot`/`Elfcar` now store collinear spin-polarized data under `spin_up`/
  `spin_down` keys instead of `total`/`diff` (matching what VASP actually
  writes). Direct access via the legacy keys still works but emits a
  `DeprecationWarning`; code that iterates over `.data` (`keys()`/`items()`,
  `**data` unpacking, dict equality, or `as_dict()` serialization) now sees the
  new keys, and `write_file` emits the new block labels. `is_soc` is now
  detected from the `diff_x`/`diff_y`/`diff_z` keys instead of `len(data) >= 4`.
  See [#99](https://github.com/materialsproject/pymatgen-core/pull/99).

- `hkl_transformation` now uses the "first nonzero index positive" sign
  convention instead of "at most one negative index", so the returned hkl may
  differ between equivalent representations (e.g. `(1, -1, -2)` where
  `(-1, 1, 2)` was returned before). New `uvw_transformation` transforms
  direct-lattice `[uvw]` vectors using the same machinery. See
  [#114](https://github.com/materialsproject/pymatgen-core/pull/114).

### v2026.7.31

- `SpacegroupAnalyzer` R-centering handling changed: the conventional-to-primitive
  transformation matrix is now positive-determinant (spglib-style) and primitive
  rhombohedral cells may come out with a different (equivalent) orientation. See
  [#110](https://github.com/materialsproject/pymatgen-core/pull/110).

### v2026.7.27

- `get_symmetrically_distinct_miller_indices` no longer implicitly primitivizes
  trigonal cells (the new `cell` argument makes this explicit), and the
  representative index chosen for a symmetry-equivalent set may differ. See
  [#101](https://github.com/materialsproject/pymatgen-core/pull/101).
- Removed APIs whose deprecation deadlines fell in 2025:
  `StructureGraph`/`MoleculeGraph` `with_empty_graph`/`with_edges`/`with_local_env_strategy`,
  `BrunnerNN_reciprocal`/`BrunnerNN_relative`/`BrunnerNN_real`, `IStructure.ntypesp`,
  the CP2K aliases (`V_Hartree_Cube`, `MO_Cubes`, `E_Density_Cube`, `Xc_Functional`,
  `Kpoint_Set`, `Band_Structure`), JDFTx `to_dict()`, and `DictSet` (use
  `VaspInputSet`). See
  [#103](https://github.com/materialsproject/pymatgen-core/pull/103).

---

Older history (pre-`pymatgen-core` split) remains in the umbrella
[`pymatgen` compatibility docs](https://github.com/materialsproject/pymatgen/blob/main/docs/compatibility.md).
