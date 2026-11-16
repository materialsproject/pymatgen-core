"""Writing a structure or molecule and reading it back must not silently lose sites or species.

A writer may refuse what its format can't represent with ValueError or NotImplementedError, and a
reader with NotImplementedError. Otherwise the elements and occupancies must come back, and so must
the oxidation states in formats that store them.
"""

from __future__ import annotations

import re

import pytest

from pymatgen.core import DummySpecies, Element, Lattice, Molecule, Species, Structure
from pymatgen.io.adf import AdfInput, AdfTask
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
}

STRUCTURE_FORMATS = ["cif", "mcif", "poscar", "cssr", "xsf", "mcsqs", "res", "pwmat", "json", "yaml"]
MOLECULE_FORMATS = ["xyz", "gjf", "json", "yaml", "mol", "sdf", "pdb", "cml"]
OPENBABEL_FORMATS = {"mol", "sdf", "pdb", "cml"}
# Formats that store oxidation states
STRUCTURE_FORMATS_WITH_OXI = {"cif", "mcif", "cssr", "mcsqs", "json", "yaml"}
MOLECULE_FORMATS_WITH_OXI = {"json", "yaml"}


def _elements(site):
    return sorted((sp.symbol, round(occu, 3)) for sp, occu in site.species.items())


def _oxi_states(site):
    # A plain element has no oxidation state, which is the same as 0
    return sorted((sp.symbol, getattr(sp, "oxi_state", None) or 0) for sp in site.species)


@pytest.mark.parametrize("case", STRUCTURES)
@pytest.mark.parametrize("fmt", STRUCTURE_FORMATS)
def test_structure_round_trip(fmt, case):
    struct = STRUCTURES[case]
    try:
        text = struct.to(fmt=fmt)
    except (ValueError, NotImplementedError):
        return
    try:
        read = Structure.from_str(text, fmt=fmt)
    except NotImplementedError:
        return
    assert sorted(map(_elements, read)) == sorted(map(_elements, struct))
    if fmt in STRUCTURE_FORMATS_WITH_OXI:
        assert sorted(map(_oxi_states, read)) == sorted(map(_oxi_states, struct))


@pytest.mark.parametrize("case", MOLECULES)
@pytest.mark.parametrize("fmt", MOLECULE_FORMATS)
def test_molecule_round_trip(fmt, case):
    if fmt in OPENBABEL_FORMATS:
        pytest.importorskip("openbabel")
    mol = MOLECULES[case]
    try:
        text = mol.to(fmt=fmt)
    except (ValueError, NotImplementedError):
        return
    read = Molecule.from_str(text, fmt=fmt)
    assert [site.specie.symbol for site in read] == [site.specie.symbol for site in mol]
    if fmt in MOLECULE_FORMATS_WITH_OXI:
        assert list(map(_oxi_states, read)) == list(map(_oxi_states, mol))


@pytest.mark.parametrize("case", MOLECULES)
def test_qchem_round_trip(case):
    mol = MOLECULES[case]
    try:
        text = str(QCInput(mol, rem={"job_type": "sp", "method": "hf", "basis": "sto-3g"}))
    except (ValueError, NotImplementedError):
        return
    read = QCInput.from_str(text).molecule
    assert [site.specie.symbol for site in read] == [site.specie.symbol for site in mol]


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
