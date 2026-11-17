"""Writing a structure or molecule and reading it back must not silently lose sites or species.

Inputs a format can't represent (listed in REFUSED) must raise ValueError or NotImplementedError.
All others must come back with the same elements and occupancies, and with the same oxidation
states in formats that store them.
"""

from __future__ import annotations

import re

import pytest

from pymatgen.core import DummySpecies, Element, Lattice, Molecule, Species, Structure
from pymatgen.io.adf import AdfInput, AdfTask
from pymatgen.io.gaussian import GaussianInput
from pymatgen.io.qchem.inputs import QCInput

LATTICE = Lattice.cubic(5)
STRUCTURES = {
    "charged": Structure(
        LATTICE, ["Fe2+", "O2-", "Na+", "Cl-"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
    ),
    "mixed": Structure(
        LATTICE,
        [Species("Fe", 2), Element("Fe"), Species("O", -2), Element("O")],
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    ),
    "disordered": Structure(
        LATTICE,
        [{Species("Fe", 2): 0.5, Species("Fe", 3): 0.5}, Element("O"), Species("Na", 1)],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    ),
    "dummy": Structure(
        LATTICE,
        [DummySpecies("X"), Species("Fe", 2), Species("O", -2)],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    ),
}
MOLECULES = {
    "charged": Molecule(
        [Species("O", -2), Species("H", 1), Species("H", 1)], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]
    ),
    "mixed": Molecule([Species("Fe", 2), Element("Fe"), Element("O")], [[0, 0, 0], [1.9, 0, 0], [0, 1.9, 0]]),
    "dummy": Molecule([DummySpecies("X"), Element("O"), Element("H")], [[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    "disordered": Molecule([{"Fe": 0.5, "Ni": 0.5}, Element("O")], [[0, 0, 0], [1.9, 0, 0]]),
}

STRUCTURE_FORMATS = ["cif", "mcif", "poscar", "cssr", "xsf", "mcsqs", "res", "pwmat", "json", "yaml"]
MOLECULE_FORMATS = ["xyz", "gjf", "qchem", "json", "yaml", "mol", "sdf", "pdb", "cml"]
OPENBABEL_FORMATS = {"mol", "sdf", "pdb", "cml"}
# Formats that store oxidation states
STRUCTURE_FORMATS_WITH_OXI = {"cif", "mcif", "cssr", "mcsqs", "json", "yaml"}
MOLECULE_FORMATS_WITH_OXI = {"json", "yaml"}
# Inputs a format can't represent, which must raise ValueError or NotImplementedError
# (when writing, or for mcif when reading)
REFUSED = {
    *((fmt, "disordered") for fmt in ("mcif", "poscar", "cssr", "xsf", "res", "pwmat")),
    ("pwmat", "dummy"),
    *((fmt, "disordered") for fmt in ("xyz", "gjf", "qchem", *OPENBABEL_FORMATS)),
    *((fmt, "dummy") for fmt in OPENBABEL_FORMATS),
}


def _elements(site):
    return sorted((sp.symbol, round(occu, 3)) for sp, occu in site.species.items())


def _oxi_states(site):
    # A plain element has no oxidation state, which is the same as 0
    return sorted((sp.symbol, getattr(sp, "oxi_state", None) or 0) for sp in site.species)


def _write_and_read(obj, fmt):
    if fmt == "qchem":  # not a registered format
        return QCInput.from_str(str(QCInput(obj, rem={"job_type": "sp", "method": "hf", "basis": "sto-3g"}))).molecule
    return type(obj).from_str(obj.to(fmt=fmt), fmt=fmt)


@pytest.mark.parametrize("case", STRUCTURES)
@pytest.mark.parametrize("fmt", STRUCTURE_FORMATS)
def test_structure_round_trip(fmt, case):
    struct = STRUCTURES[case]
    if (fmt, case) in REFUSED:
        with pytest.raises((ValueError, NotImplementedError)):
            _write_and_read(struct, fmt)
        return
    read = _write_and_read(struct, fmt)
    assert sorted(map(_elements, read)) == sorted(map(_elements, struct))
    if fmt in STRUCTURE_FORMATS_WITH_OXI:
        assert sorted(map(_oxi_states, read)) == sorted(map(_oxi_states, struct))


@pytest.mark.parametrize("case", MOLECULES)
@pytest.mark.parametrize("fmt", MOLECULE_FORMATS)
def test_molecule_round_trip(fmt, case):
    if fmt in OPENBABEL_FORMATS:
        pytest.importorskip("openbabel")
    mol = MOLECULES[case]
    if (fmt, case) in REFUSED:
        with pytest.raises((ValueError, NotImplementedError)):
            _write_and_read(mol, fmt)
        return
    read = _write_and_read(mol, fmt)
    assert list(map(_elements, read)) == list(map(_elements, mol))
    if fmt in MOLECULE_FORMATS_WITH_OXI:
        assert list(map(_oxi_states, read)) == list(map(_oxi_states, mol))


def test_gaussian_coordinates_refuse_disordered_molecules():
    gau = GaussianInput(MOLECULES["disordered"])
    for get_coords in (gau.get_cart_coords, gau.get_zmatrix):
        with pytest.raises(ValueError, match="ordered molecule"):
            get_coords()


@pytest.mark.parametrize("case", ["charged", "mixed"])
def test_adf_atom_types_are_element_symbols(case, tmp_path):
    # There is no reader for ADF input. An atom type is an element symbol, optionally
    # followed by ".suffix"; the charge goes in its own block.
    path = tmp_path / "adf.inp"
    AdfInput(AdfTask()).write_file(MOLECULES[case], str(path))
    lines = path.read_text().splitlines()
    start = next(idx for idx, line in enumerate(lines) if line.upper().startswith("ATOMS")) + 1
    atom_types = [line.split()[0] for line in lines[start : lines.index("END", start)]]
    assert len(atom_types) == len(MOLECULES[case])
    assert all(re.fullmatch(r"[A-Z][a-z]?(\.\w+)?", atom_type) for atom_type in atom_types), atom_types
