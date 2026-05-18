from __future__ import annotations

import re

import pytest
from pytest import approx

from pymatgen.core import Composition, Species, Structure
from pymatgen.core.bond_valence import (
    BVAnalyzer,
    add_oxidation_state_by_site_fraction,
    calculate_bv_sum,
    calculate_bv_sum_unordered,
    get_z_ordered_elmap,
)
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/core/bond_valence"


class TestBVAnalyzer(MatSciTest):
    def setup_method(self):
        self.analyzer = BVAnalyzer()

    def test_get_valences(self):
        struct = Structure.from_file(f"{TEST_DIR}/LiMn2O4.json")
        valences = [1, 1, 3, 3, 4, 4, -2, -2, -2, -2, -2, -2, -2, -2]
        assert self.analyzer.get_valences(struct) == valences
        struct = self.get_structure("LiFePO4")
        valences = [1] * 4 + [2] * 4 + [5] * 4 + [-2] * 16
        assert self.analyzer.get_valences(struct) == valences
        struct = self.get_structure("Li3V2(PO4)3")
        valences = [1] * 6 + [3] * 4 + [5] * 6 + [-2] * 24
        assert self.analyzer.get_valences(struct) == valences
        struct = Structure.from_file(f"{TEST_DIR}/Li4Fe3Mn1(PO4)4.json")
        valences = [1] * 4 + [2] * 4 + [5] * 4 + [-2] * 16
        assert self.analyzer.get_valences(struct) == valences
        struct = self.get_structure("NaFePO4")
        assert self.analyzer.get_valences(struct) == valences

        # trigger ValueError Structure contains elements not in set of BV parameters
        with pytest.raises(
            ValueError,
            match=re.escape("Structure contains elements not in set of BV parameters: {Element Xe}"),
        ):
            self.analyzer.get_valences(self.get_structure("Li10GeP2S12").replace_species({"Li": "Xe"}, in_place=False))

    def test_get_oxi_state_structure(self):
        struct = Structure.from_file(f"{TEST_DIR}/LiMn2O4.json")
        oxi_struct = self.analyzer.get_oxi_state_decorated_structure(struct)
        assert Species("Mn3+") in oxi_struct.composition.elements
        assert Species("Mn4+") in oxi_struct.composition.elements

    def _disordered_LiMn2O4(self):
        # Mix Li/Na on one site so the structure is unordered but still
        # solvable by `BVAnalyzer`.
        struct = Structure.from_file(f"{TEST_DIR}/LiMn2O4.json")
        struct[0].species = Composition("Li0.5Na0.5")
        return struct

    def test_get_valences_unordered(self):
        # Exercises `_calc_site_probabilities_unordered` and the unordered
        # recursion branch in `get_valences`.
        struct = self._disordered_LiMn2O4()
        valences = self.analyzer.get_valences(struct)
        # 14 sites with per-site lists; the mixed Li/Na site has two entries.
        assert len(valences) == len(struct)
        assert valences[0] == [1, 1]  # Li+ and Na+ both +1
        # Mn sites carry +3 and +4, O sites carry -2.
        flat = {v for site in valences for v in site}
        assert flat <= {1, 3, 4, -2}

    def test_get_oxi_state_structure_unordered(self):
        struct = self._disordered_LiMn2O4()
        oxi_struct = self.analyzer.get_oxi_state_decorated_structure(struct)
        # First site is still mixed but both species now carry +1.
        first = oxi_struct[0].species
        assert {Species("Li+"), Species("Na+")} == set(first)

    def test_forbidden_species(self):
        # `forbidden_species` filters ICSD_BV_DATA — exercises the
        # non-default branch of `BVAnalyzer.__init__`.
        analyzer = BVAnalyzer(forbidden_species=["O-"])
        assert Species("O-") not in analyzer.icsd_bv_data
        # baseline still has the default data, so this is a strict subset.
        from pymatgen.core.bond_valence import ICSD_BV_DATA

        assert len(analyzer.icsd_bv_data) < len(ICSD_BV_DATA)


class TestBondValenceSum(MatSciTest):
    def test_calculate_bv_sum(self):
        struct = Structure.from_file(f"{TEST_DIR}/LiMn2O4.json")
        neighbors = struct.get_neighbors(struct[0], 3.0)
        bv_sum = calculate_bv_sum(struct[0], neighbors)
        assert bv_sum == approx(0.7723402182087497)

    def test_calculate_bv_sum_unordered(self):
        struct = Structure.from_file(f"{TEST_DIR}/LiMn2O4.json")
        struct[0].species = Composition("Li0.5Na0.5")
        neighbors = struct.get_neighbors(struct[0], 3.0)
        bv_sum = calculate_bv_sum_unordered(struct[0], neighbors)
        assert bv_sum == approx(1.5494662306918852)


class TestModuleHelpers(MatSciTest):
    def test_get_z_ordered_elmap(self):
        comp = Composition(
            {
                Species("Ni", 3): 0.2,
                Species("Ni", 4): 0.2,
                Species("Cr", 3): 0.15,
                Species("Zn", 2): 0.34,
                Species("Cr", 4): 0.11,
            }
        )
        ordered = get_z_ordered_elmap(comp)
        # The function returns a sorted list of (Species, occupation) tuples.
        # Verify that all input species are present, with their occupations
        # preserved, and that ordering is deterministic (input order independent).
        assert dict(ordered) == dict(comp.items())
        assert sum(occ for _, occ in ordered) == approx(1.0)
        # Re-permuting the input dict should not change the output ordering.
        comp2 = Composition(dict(reversed(list(comp.items()))))  # reverse insertion order
        assert get_z_ordered_elmap(comp2) == ordered
        # Within each element, oxidation states sort ascending.
        cr_oxi = [sp.oxi_state for sp, _ in ordered if sp.symbol == "Cr"]
        assert cr_oxi == sorted(cr_oxi)

    def test_add_oxidation_state_by_site_fraction(self):
        struct = Structure.from_file(f"{TEST_DIR}/LiMn2O4.json")
        struct[0].species = Composition("Li0.5Na0.5")
        # 14 sites: index 0 has 2 species (Li, Na), rest have 1 each.
        # `get_z_ordered_elmap` sorts Li (Z=3) before Na (Z=11); we mirror that.
        oxi_states = [[1, 1]] + [[3]] * 2 + [[4]] * 2 + [[-2]] * 8 + [[-2]]  # 14 entries
        oxi_states = [[1, 1]] + [[3], [3], [4], [4]] + [[-2]] * 8 + [[-2]]
        result = add_oxidation_state_by_site_fraction(struct, oxi_states)
        # Site 0 has Li+ and Na+
        assert {sp.symbol for sp in result[0].species} == {"Li", "Na"}
        assert all(sp.oxi_state == 1 for sp in result[0].species)

    def test_add_oxidation_state_by_site_fraction_too_few(self):
        struct = Structure.from_file(f"{TEST_DIR}/LiMn2O4.json")
        struct[0].species = Composition("Li0.5Na0.5")
        with pytest.raises(ValueError, match="Oxidation state of all sites must be specified"):
            add_oxidation_state_by_site_fraction(struct, [[1, 1]])
