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
