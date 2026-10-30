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

### v2026.10.30

* [breaking] Default Structure.relax/calculate/calc_property to TensorNet-PES-MatPES-r2SCAN-2025.2 by @shyuep in [#260](https://github.com/materialsproject/pymatgen-core/pull/260)

  - `Structure.relax()`, `Structure.calculate()` and `Structure.calc_property()` now default to the r2SCAN foundation potential `TensorNet-PES-MatPES-r2SCAN-2025.2` instead of `TensorNet-MatPES-PBE-v2025.1-PES`. Energies, forces, stresses and relaxed geometries change accordingly; for example, r2SCAN-level energies are not on the PBE scale.
    - To get PBE-level results (e.g. for Materials Project compatibility), pass `calculator="TensorNet-PES-MatPES-PBE-2025.2"`.
  - The default `stress_weight` in `Structure.relax()` changes from `0.01` to `1 / 160.21766208` (GPa -> eV/A^3). Pass `stress_weight=0.01` explicitly to keep the old weighting on matgl < 4.0.2; newer matgl rejects it.
  - The `matcalc` and `mlp` extras now need `matgl>=3.0.0`, and the `matcalc` extra needs `matcalc>=0.5.0`.


### v2026.8.30

* [breaking] Do not record the identity as a rotation axis in PointGroupAnalyzer by @HiroYokoyama in [#134](https://github.com/materialsproject/pymatgen-core/pull/134)

  `sch_symbol` changes for molecules whose unique axis of inertia carries no
  rotation of its own but which have one perpendicular C2 axis: they were
  reported as `D2` and are now reported as `C2`. `get_symmetry_operations()`
  and `get_pointgroup()` are unaffected — they already returned the
  2-operation group. Code that compared `sch_symbol` against the previous
  (incorrect) value must be updated; nothing else is affected.

* [Breaking] Deprecate `Orbital.dx2` with `Orbital.dx2_y2` by @DanielYang59 in [#94](https://github.com/materialsproject/pymatgen-core/pull/94)

  `Orbital.dx2` is renamed to `Orbital.dx2_y2` (enum value `8` unchanged). To
  avoid breaking consumers of serialized data, this is a two-phase transition:

  - **Until 2027-08-17:** `Orbital.dx2` / `Orbital["dx2"]` still resolve (with a
    `DeprecationWarning`); `CompleteCohp.as_dict()` / `CompleteDos.as_dict()`
    still write the legacy `dx2` key; readers accept both `dx2` and `dx2_y2`.
  - **From 2027-08-17:** canonical `dx2_y2` is serialized and the legacy
    alias/read paths are removed.

  **Migration:** use `Orbital.dx2_y2` / `"dx2_y2"`; existing serialized data
  needs no conversion.

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
