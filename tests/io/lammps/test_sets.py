from __future__ import annotations

from pymatgen.core import Lattice, Structure
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.sets import LammpsInputSet


def test_from_directory_roundtrip(tmp_path):
    # write_input names the data file "input.data" and the input script may not set atom_style
    struct = Structure(Lattice.cubic(3.6), ["Cu"], [[0, 0, 0]])
    data = LammpsData.from_structure(struct, atom_style="full")
    input_set = LammpsInputSet("units metal\nread_data input.data\nrun 0", data)
    input_set.write_input(tmp_path)

    input_set2 = LammpsInputSet.from_directory(tmp_path)
    assert input_set2.data.atom_style == "full"
    assert len(input_set2.data.atoms) == 1
    assert input_set2.inputfile.get_args("run") == "0"


def test_from_directory_atom_style_hint(tmp_path):
    # Without atom_style in the input script, the "Atoms # <style>" hint written by LammpsData is used
    struct = Structure(Lattice.cubic(3.6), ["Cu"], [[0, 0, 0]])
    data = LammpsData.from_structure(struct, atom_style="charge")
    assert "Atoms # charge" in data.get_str()
    LammpsInputSet("units metal\nread_data input.data\nrun 0", data).write_input(tmp_path)
    assert LammpsInputSet.from_directory(tmp_path).data.atom_style == "charge"


def test_from_directory_defaults_to_atomic(tmp_path):
    # Without atom_style in the input script or a style hint, use the LAMMPS default "atomic"
    (tmp_path / "in.lammps").write_text("units metal\nread_data system.data\nrun 0\n")
    (tmp_path / "system.data").write_text(
        "Cu atomic data\n\n1 atoms\n1 atom types\n\n0.0 3.6 xlo xhi\n0.0 3.6 ylo yhi\n0.0 3.6 zlo zhi\n\n"
        "Masses\n\n1 63.546\n\nAtoms\n\n1 1 0.0 0.0 0.0\n"
    )
    input_set = LammpsInputSet.from_directory(tmp_path)
    assert input_set.data.atom_style == "atomic"
    assert len(input_set.data.atoms) == 1
