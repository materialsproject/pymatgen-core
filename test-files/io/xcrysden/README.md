# XCrySDen XSF fixtures

These fixtures are tiny hand-written samples derived from the official XCrySDen
XSF specification. They are intended to exercise parser behavior without
copying large generated outputs from external programs.

Primary specification:

- [XCrySDen XSF specification](http://www.xcrysden.org/doc/XSF.html)

Mirror used when the primary site is unavailable:

- [XCrySDen XSF specification mirror](https://web.mit.edu/xcrysden_v1.5.60/www/XCRYSDEN/doc/XSF.html)

Fixture provenance:

- `crystal_primvec_primcoord.xsf`: hand-written from the `CRYSTAL` / `PRIMVEC` / `PRIMCOORD` grammar in the official spec.
- `crystal_primcoord_forces.xsf`: hand-written variant of the same official grammar with optional force columns.
- `crystal_convvec.xsf`: hand-written from the `CONVVEC` example shape described by the official spec.
- `molecule_atoms.xsf`: hand-written from the `ATOMS` section described by the official spec.
- `axsf_fixed_cell.axsf`: hand-written from the AXSF fixed-cell animation grammar in the official spec.
- `axsf_variable_cell.axsf`: hand-written from the AXSF variable-cell animation grammar in the official spec.
- `datagrid_2d.xsf`: hand-written from the `BEGIN_BLOCK_DATAGRID_2D` grammar in the official spec.
- `datagrid_3d.xsf`: hand-written from the `BEGIN_BLOCK_DATAGRID_3D` grammar in the official spec.
- `bandgrid_3d.bxsf`: hand-written from the `BEGIN_INFO` / `BEGIN_BLOCK_BANDGRID_3D` grammar in the official spec.

All files in this directory are original test fixtures for pymatgen.
