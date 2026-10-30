from __future__ import annotations

import json

from monty.json import MontyDecoder, MontyEncoder

from pymatgen.core import Lattice, Structure
from pymatgen.core.structure_analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.testing import MatSciTest


class TestSymmetrizedStructure(MatSciTest):
    def setup_method(self):
        self.structure = Structure(
            lattice=Lattice.cubic(3),
            species=("Fe", "Fe"),
            coords=((0, 0, 0), (0.5, 0.5, 0.5)),
        )

        self.symm_structure = SpacegroupAnalyzer(self.structure).get_symmetrized_structure()

    def test_str_repr(self):
        assert str(self.symm_structure) == repr(self.symm_structure)
        assert "Reduced Formula: Fe" in str(self.symm_structure)

    def test_dict(self):
        dct = self.symm_structure.as_dict()

        assert isinstance(SymmetrizedStructure.from_dict(dct), SymmetrizedStructure)

    def test_serialize(self):
        assert isinstance(self.symm_structure, SymmetrizedStructure)
        self.assert_msonable(self.symm_structure)

        self.symm_structure.to(fmt="json")

    def test_json_round_trip_keeps_spacegroup(self):
        json_str = json.dumps(self.symm_structure.as_dict(), cls=MontyEncoder)
        symm_structure = json.loads(json_str, cls=MontyDecoder)
        assert isinstance(symm_structure, SymmetrizedStructure)
        assert symm_structure.spacegroup.int_symbol == self.symm_structure.spacegroup.int_symbol
        assert symm_structure.spacegroup.int_number == self.symm_structure.spacegroup.int_number
        assert len(symm_structure.spacegroup) == len(self.symm_structure.spacegroup)
        assert str(symm_structure) == str(self.symm_structure)

    def test_str_site_properties(self):
        # equivalence class 1 (Cl) starts at site index 2, so its row must show the Cl property
        structure = Structure(
            Lattice.tetragonal(4, 6),
            ["Na", "Na", "Cl", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0.5]],
            site_properties={"charge": [1, 1, -1, -1]},
        )
        symm_structure = SpacegroupAnalyzer(structure).get_symmetrized_structure()
        assert symm_structure.equivalent_indices == [[0, 1], [2, 3]]
        rows = str(symm_structure).splitlines()[-2:]
        assert rows[0].split()[1] == "Na"
        assert rows[0].split()[-1] == "1"
        assert rows[1].split()[1] == "Cl"
        assert rows[1].split()[-1] == "-1"

    def test_find_equivalent_sites(self):
        site = self.symm_structure.sites[0]
        assert self.symm_structure.find_equivalent_sites(site) == self.symm_structure.sites
