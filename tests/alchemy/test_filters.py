from __future__ import annotations

import json

import pytest
from monty.json import MontyDecoder

from pymatgen.alchemy.filters import (
    ContainsSpecieFilter,
    RemoveDuplicatesFilter,
    RemoveExistingFilter,
    SpecieProximityFilter,
)
from pymatgen.alchemy.transmuters import StandardTransmuter
from pymatgen.core import Lattice, Species, Structure
from pymatgen.core.structure_matcher import StructureMatcher
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest


class TestContainsSpecieFilter(MatSciTest):
    def test_filtering(self):
        coords = [[0, 0, 0], [0.75, 0.75, 0.75], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
        lattice = Lattice([[3.0, 0.0, 0.0], [1.0, 3.0, 0], [0, -2.0, 3.0]])
        struct = Structure(lattice, [{"Si4+": 0.5, "O2-": 0.25, "P5+": 0.25}] * 4, coords)

        species1 = [Species("Si5+"), Species("Mg2+")]
        f1 = ContainsSpecieFilter(species1, strict_compare=True, AND=False)
        assert not f1.test(struct), "Incorrect filter"
        f2 = ContainsSpecieFilter(species1, strict_compare=False, AND=False)
        assert f2.test(struct), "Incorrect filter"
        species2 = [Species("Si4+"), Species("Mg2+")]
        f3 = ContainsSpecieFilter(species2, strict_compare=True, AND=False)
        assert f3.test(struct), "Incorrect filter"
        f4 = ContainsSpecieFilter(species2, strict_compare=False, AND=False)
        assert f4.test(struct), "Incorrect filter"

        species3 = [Species("Si5+"), Species("O2-")]
        f5 = ContainsSpecieFilter(species3, strict_compare=True, AND=True)
        assert not f5.test(struct), "Incorrect filter"
        f6 = ContainsSpecieFilter(species3, strict_compare=False, AND=True)
        assert f6.test(struct), "Incorrect filter"
        species4 = [Species("Si4+"), Species("Mg2+")]
        f7 = ContainsSpecieFilter(species4, strict_compare=True, AND=True)
        assert not f7.test(struct), "Incorrect filter"
        f8 = ContainsSpecieFilter(species4, strict_compare=False, AND=True)
        assert not f8.test(struct), "Incorrect filter"

    def test_as_from_dict(self):
        species1 = ["Si5+", "Mg2+"]
        f1 = ContainsSpecieFilter(species1, strict_compare=True, AND=False)
        dct = f1.as_dict()
        assert isinstance(ContainsSpecieFilter.from_dict(dct), ContainsSpecieFilter)


class TestSpecieProximityFilter(MatSciTest):
    def test_filter(self):
        struct = self.get_structure("Li10GeP2S12")
        sf = SpecieProximityFilter({"Li": 1})
        assert sf.test(struct)
        sf = SpecieProximityFilter({"Li": 2})
        assert not sf.test(struct)
        sf = SpecieProximityFilter({"P": 1})
        assert sf.test(struct)
        sf = SpecieProximityFilter({"P": 5})
        assert not sf.test(struct)

    @pytest.mark.parametrize(
        ("min_dists", "expected"),
        [
            ({"Na": 2.9}, True),  # nearest Na-Na is the periodic image at 3 A
            ({"Na": 3.1}, False),
            ({"K": 10}, True),  # species not in structure
            ({"Li": 1.5, "Na": 1.5}, True),  # Li-Na pairs are not checked, only like pairs
            ({"Li": 2.1}, False),  # Li on the disordered site is 2 A from the pure Li
            ({"Li": 1.9, "Na": 2.9}, True),
            ({"Li": 1.9, "Na": 3.1}, False),
        ],
    )
    def test_filter_pairs(self, min_dists, expected):
        # Na at z = 0 A, Li at z = 1 A, (Li, Mg) at z = 3 A; periodic images are 3 A apart along a and b
        struct = Structure(
            Lattice.tetragonal(3, 6),
            ["Na", "Li", {"Li": 0.5, "Mg": 0.5}],
            [[0, 0, 0], [0, 0, 1 / 6], [0, 0, 0.5]],
        )
        assert SpecieProximityFilter(min_dists).test(struct) is expected

    def test_as_from_dict(self):
        sf = SpecieProximityFilter({"Li": 1})
        dct = sf.as_dict()
        assert isinstance(SpecieProximityFilter.from_dict(dct), SpecieProximityFilter)


class TestRemoveDuplicatesFilter:
    def setup_method(self):
        with open(f"{TEST_FILES_DIR}/entries/TiO2_entries.json", encoding="utf-8") as file:
            entries = json.load(file, cls=MontyDecoder)
        self._struct_list = [entry.structure for entry in entries]
        self._sm = StructureMatcher()

    def test_filter(self):
        transmuter = StandardTransmuter.from_structures(self._struct_list)
        dup_filter = RemoveDuplicatesFilter()
        transmuter.apply_filter(dup_filter)
        assert len(transmuter.transformed_structures) == 11

    def test_as_from_dict(self):
        fil = RemoveDuplicatesFilter()
        dct = fil.as_dict()
        assert isinstance(RemoveDuplicatesFilter().from_dict(dct), RemoveDuplicatesFilter)


class TestRemoveExistingFilter:
    def setup_method(self):
        with open(f"{TEST_FILES_DIR}/entries/TiO2_entries.json", encoding="utf-8") as file:
            entries = json.load(file, cls=MontyDecoder)
        self._struct_list = [entry.structure for entry in entries]
        self._sm = StructureMatcher()
        self._existing_structures = self._struct_list[:-1]

    def test_filter(self):
        fil = RemoveExistingFilter(self._existing_structures)
        transmuter = StandardTransmuter.from_structures(self._struct_list)
        transmuter.apply_filter(fil)
        assert len(transmuter.transformed_structures) == 1
        assert self._sm.fit(
            self._struct_list[-1],
            transmuter.transformed_structures[-1].final_structure,
        )

    @pytest.mark.parametrize("symprec", [None, 1e-3])
    def test_different_composition(self, symprec):
        # with symprec=None a structure of a different composition used to fall through to a
        # SpacegroupAnalyzer(symprec=None) call, raising SymmetryUndeterminedError
        nacl = Structure(Lattice.cubic(5.6), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        kbr = Structure(Lattice.cubic(6.6), ["K", "Br"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        fil = RemoveExistingFilter([nacl], symprec=symprec)
        assert fil.test(kbr)
        assert not fil.test(nacl.copy())

    def test_as_from_dict(self):
        # as_dict used to drop existing_structures/symprec and from_dict raised TypeError on init_args
        fil = RemoveExistingFilter(self._existing_structures, symprec=1e-3)
        fil2 = MontyDecoder().process_decoded(json.loads(json.dumps(fil.as_dict())))
        assert isinstance(fil2, RemoveExistingFilter)
        assert fil2.symprec == 1e-3
        assert fil2.existing_structures == self._existing_structures
        assert isinstance(fil2.structure_matcher, StructureMatcher)
        assert not fil2.test(self._existing_structures[0])


class TestContainSpecieRepr:
    def test_repr(self):
        f = ContainsSpecieFilter(["Si4+"], strict_compare=True, AND=False, exclude=True)
        s = repr(f)
        assert "ContainsSpecieFilter" in s
        assert "strict_compare = True" in s
        assert "exclude = True" in s


class TestChargeBalanceFilter:
    def test_filter(self):
        from pymatgen.alchemy.filters import ChargeBalanceFilter
        from pymatgen.core import Lattice

        lattice = Lattice.cubic(4.0)
        # neutral structure: LiFePO4 is charge-balanced
        neutral = MatSciTest.get_structure("LiFePO4")
        f = ChargeBalanceFilter()
        assert f.test(neutral) is True
        # a composition with a genuine net charge fails
        charged = Structure(
            lattice,
            [Species("Si4+"), Species("O2-")],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        assert f.test(charged) is False


class TestSpeciesMaxDistFilter:
    def test_filter(self):
        from pymatgen.alchemy.filters import SpeciesMaxDistFilter

        struct = MatSciTest.get_structure("LiFePO4")
        f = SpeciesMaxDistFilter("Li", "Fe", max_dist=50.0)
        assert f.test(struct) is True


class TestRemoveDuplicatesFilterExtra:
    def test_typeerror(self):
        with pytest.raises(TypeError, match="must be a dict"):
            RemoveDuplicatesFilter(structure_matcher=42)  # type: ignore[arg-type]

    def test_from_dict_structure_matcher(self):
        f = RemoveDuplicatesFilter(structure_matcher=StructureMatcher().as_dict())
        assert isinstance(f.structure_matcher, StructureMatcher)

    def test_symprec_filter(self):
        struct = MatSciTest.get_structure("LiFePO4")
        f = RemoveDuplicatesFilter(symprec=1e-4)
        assert f.test(struct) is True


class TestRemoveExistingFilterExtra:
    def test_from_dict_structure_matcher(self):
        struct = MatSciTest.get_structure("LiFePO4")
        f = RemoveExistingFilter([struct], structure_matcher=StructureMatcher().as_dict())
        assert isinstance(f.structure_matcher, StructureMatcher)

    def test_symprec_reject_duplicate(self):
        struct = MatSciTest.get_structure("LiFePO4")
        f = RemoveExistingFilter([struct], symprec=1e-4)
        assert f.test(struct) is False
        # a similar-structure with identical space group and similar lattice should also be filtered
        dup = struct.copy()
        assert f.test(dup) is False


class TestRemoveDuplicatesFilterSymprec:
    def test_symprec(self):
        struct = MatSciTest.get_structure("LiFePO4")
        f = RemoveDuplicatesFilter(symprec=1e-4)
        first = f.test(struct)
        second = f.test(struct.copy())
        assert first is True
        assert second is False
