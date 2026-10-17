from __future__ import annotations

import collections
import re
from io import StringIO
from itertools import combinations
from numbers import Number
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.colors import Normalize, same_color
from monty.json import MontyDecoder
from monty.serialization import dumpfn, loadfn
from numpy.testing import assert_allclose
from pytest import approx

from pymatgen.analysis import phase_diagram
from pymatgen.analysis.phase_diagram import (
    CompoundPhaseDiagram,
    GrandPotentialPhaseDiagram,
    GrandPotPDEntry,
    PatchedPhaseDiagram,
    PDEntry,
    PDPlotter,
    PhaseDiagram,
    ReactionDiagram,
    TransformedPDEntry,
    TransformedPDEntryError,
    order_phase_diagram,
    tet_coord,
    triangular_coord,
    uniquelines,
)
from pymatgen.core import Composition, DummySpecies, Element
from pymatgen.core.entries import ComputedEntry, ConstantEnergyAdjustment, EntrySet
from pymatgen.util.coord import Simplex
from pymatgen.util.string import htmlify
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/analysis"


class TestPDEntry:
    def setup_method(self):
        comp = Composition("LiFeO2")
        self.entry = PDEntry(comp, 53, name="mp-757614")
        self.gp_entry = GrandPotPDEntry(self.entry, {Element("O"): 1.5})

    def test_get_energy(self):
        assert self.entry.energy == 53, "Wrong energy!"
        assert self.gp_entry.energy == 50, "Wrong energy!"

    def test_get_chemical_energy(self):
        assert self.gp_entry.chemical_energy == 3, "Wrong energy!"

    def test_get_energy_per_atom(self):
        assert self.entry.energy_per_atom == approx(53.0 / 4), "Wrong energy per atom!"
        assert self.gp_entry.energy_per_atom == approx(50.0 / 2), "Wrong energy per atom!"

    def test_get_name(self):
        assert self.entry.name == "mp-757614"
        assert self.gp_entry.name == "mp-757614"

    def test_composition(self):
        comp = self.entry.composition
        expected_comp = Composition("LiFeO2")
        assert comp == expected_comp
        comp = self.gp_entry.composition
        expected_comp = Composition("LiFe")
        assert comp == expected_comp

    def test_formula(self):
        assert self.entry.formula == "Li1 Fe1 O2"
        assert self.gp_entry.formula == "Li1 Fe1 O2"

    def test_reduced_formula(self):
        assert self.entry.reduced_formula == "LiFeO2"
        assert self.gp_entry.reduced_formula == "LiFeO2"

    def test_elements(self):
        expected_elements = list(map(Element, ["Li", "Fe", "O"]))
        assert self.entry.elements == expected_elements

    def test_is_element(self):
        assert not self.entry.is_element
        assert not self.gp_entry.is_element

    def test_as_from_dict(self):
        dct = self.entry.as_dict()
        gpd = self.gp_entry.as_dict()
        entry = PDEntry.from_dict(dct)

        assert entry.name == "mp-757614"
        assert entry.energy_per_atom == approx(53.0 / 4)
        gp_entry = GrandPotPDEntry.from_dict(gpd)
        assert gp_entry.name == "mp-757614"
        assert gp_entry.energy_per_atom == approx(50.0 / 2)

        d_anon = dct.copy()
        del d_anon["name"]
        try:
            entry = PDEntry.from_dict(d_anon)
        except KeyError:
            self.fail("Should not need to supply name!")

    def test_str(self):
        assert str(self.entry) == "PDEntry : Li1 Fe1 O2 (mp-757614) with energy = 53.0000"
        pde = self.entry.as_dict()
        del pde["name"]
        pde = PDEntry.from_dict(pde)
        assert str(pde) == "PDEntry : Li1 Fe1 O2 with energy = 53.0000"

    def test_read_csv(self):
        entries = EntrySet.from_csv(f"{TEST_DIR}/pd_entries_test.csv")
        assert entries.chemsys == {"Li", "Fe", "O"}, "Wrong elements!"
        assert len(entries) == 490, "Wrong number of entries!"


class TestTransformedPDEntry:
    def setup_method(self):
        comp = Composition("LiFeO2")
        entry = PDEntry(comp, 53)

        terminal_compositions = [*map(Composition, ("Li2O", "FeO", "LiO8"))]

        sp_mapping = {}
        for idx, comp in enumerate(terminal_compositions):
            sp_mapping[comp] = DummySpecies(f"X{chr(102 + idx)}")

        self.transformed_entry = TransformedPDEntry(entry, sp_mapping)

    def test_get_energy(self):
        assert self.transformed_entry.energy == 53, "Wrong energy!"
        assert self.transformed_entry.original_entry.energy == approx(53.0)

    def test_get_energy_per_atom(self):
        assert self.transformed_entry.energy_per_atom == approx(53.0 / (23 / 15))

    def test_get_name(self):
        assert self.transformed_entry.name == "LiFeO2"

    def test_composition(self):
        comp = self.transformed_entry.composition
        expected_comp = Composition(
            {
                DummySpecies("Xf"): 14 / 30,
                DummySpecies("Xg"): 1.0,
                DummySpecies("Xh"): 2 / 30,
            }
        )
        assert comp == expected_comp

    def test_elements(self):
        expected_elements = list(map(Element, ["Li", "Fe", "O"]))
        assert self.transformed_entry.elements == expected_elements

    def test_is_element(self):
        assert self.transformed_entry.is_element is False
        assert self.transformed_entry.original_entry.is_element is False
        iron = Composition("Fe")
        assert TransformedPDEntry(PDEntry(iron, 0), {iron: iron}).is_element is True

    def test_as_from_dict(self):
        dct = self.transformed_entry.as_dict()
        entry = TransformedPDEntry.from_dict(dct)
        assert entry.name == "LiFeO2" == self.transformed_entry.name
        assert entry.energy_per_atom == approx(53.0 / (23 / 15)) == self.transformed_entry.energy_per_atom

    def test_str(self):
        assert str(self.transformed_entry).startswith("TransformedPDEntry Xf0+0.46666667 Xg")
        assert str(self.transformed_entry).endswith("with original composition Li1 Fe1 O2, energy = 53.0000")

    def test_normalize(self):
        norm_entry = self.transformed_entry.normalize(mode="atom")
        expected_comp = Composition(
            {
                DummySpecies("Xf"): 7 / 23,
                DummySpecies("Xg"): 15 / 23,
                DummySpecies("Xh"): 1 / 23,
            }
        )
        assert norm_entry.composition == expected_comp


class TestPhaseDiagram(MatSciTest):
    def setup_method(self):
        self.entries = EntrySet.from_csv(f"{TEST_DIR}/pd_entries_test.csv")
        self.pd = PhaseDiagram(self.entries)

    def test_init(self):
        # Ensure that a bad set of entries raises a PD error. Remove all Li
        # from self.entries.
        entries = filter(
            lambda e: (not e.composition.is_element) or e.elements[0] != Element("Li"),
            self.entries,
        )
        with pytest.raises(
            ValueError,
            match=r"Missing terminal entries for elements \['Fe', 'Li', 'O'\]",
        ):
            PhaseDiagram(entries)

    def test_repr(self):
        assert (
            str(self.pd) == "Li-Fe-O phase diagram\n11 stable phases: \nFe, FeO, "
            "Fe2O3, Fe3O4, LiFeO2, Li, Li2O, LiO, Li5FeO4, Li2FeO3, O"
        )

    def test_dim1(self):
        # Ensure that dim 1 PDs can be generated.
        for el in ("Li", "Fe", "O2"):
            entries = [entry for entry in self.entries if entry.reduced_formula == el]
            pd = PhaseDiagram(entries)
            assert len(pd.stable_entries) == 1

            for entry in entries:
                e_hull = pd.get_e_above_hull(entry)
                assert e_hull >= 0

            plotter = PDPlotter(pd)
            lines, *_ = plotter.pd_plot_data
            assert lines[0][1] == [0, 0]

    def test_ordering(self):
        # Test sorting of elements
        entries = [ComputedEntry(Composition(formula), 0) for formula in ["O", "N", "Fe"]]
        pd = PhaseDiagram(entries)
        sorted_elements = (Element("Fe"), Element("N"), Element("O"))
        assert tuple(pd.elements) == sorted_elements

        entries.reverse()
        pd = PhaseDiagram(entries)
        assert tuple(pd.elements) == sorted_elements

        # Test manual specification of order
        ordering = [Element(elt_string) for elt_string in ["O", "N", "Fe"]]
        pd = PhaseDiagram(entries, elements=ordering)
        assert tuple(pd.elements) == tuple(ordering)

    def test_stable_entries(self):
        stable_formulas = [ent.reduced_formula for ent in self.pd.stable_entries]
        expected_stable = "Fe2O3 Li5FeO4 LiFeO2 Fe3O4 Li Fe Li2O O2 FeO".split()
        for formula in expected_stable:
            assert formula in stable_formulas, f"{formula} not in stable entries!"

    def test_get_form_energy(self):
        expected_formation_energies = {
            "Li5FeO4": -164.8117344,
            "Li2O2": -14.119232793,
            "Fe2O3": -16.57416433,
            "FeO": -5.71415199,
            "Li": 0.0,
            "LiFeO2": -7.73275231,
            "Li2O": -6.229303868,
            "Fe": 0.0,
            "Fe3O4": -22.56571445,
            "Li2FeO3": -45.67166036,
            "O2": 0.0,
        }

        for entry in self.pd.stable_entries:
            formula = entry.reduced_formula
            expected = expected_formation_energies[formula]
            n_atoms = entry.composition.num_atoms
            # test get_form_energy
            assert self.pd.get_form_energy(entry) == approx(expected), formula
            # test get_form_energy_per_atom
            assert self.pd.get_form_energy_per_atom(entry) == approx(expected / n_atoms), formula

    def test_get_reference_energy(self):
        expected_ref_energies = {
            "Li5FeO4": -265.5546721,
            "Li2O2": -24.685172046,
            "Fe2O3": -51.93425725,
            "FeO": -21.70885048,
            "Li": -1.91301487,
            "LiFeO2": -17.02571825,
            "Li2O": -8.084307881,
            "Fe": -6.5961471,
            "Fe3O4": -73.6431077,
            "Li2FeO3": -92.78804506,
            "O2": -25.54966885,
        }
        for entry in self.pd.stable_entries:
            formula = entry.reduced_formula
            actual = self.pd.get_reference_energy(entry.composition)
            expected = expected_ref_energies[formula]
            assert actual == approx(expected), formula

    def test_all_entries_hulldata(self):
        assert len(self.pd.all_entries_hulldata) == 490

    def test_planar_inputs(self):
        elems = ["H", "He", "Li", "Be", "B", "Rb"]
        e1, e2, e3, e4, e5, e6 = (PDEntry(elem, 0) for elem in elems)

        pd = PhaseDiagram([e1, e2, e3, e4, e5, e6], map(Element, elems))

        assert len(pd.facets) == 1

    def test_str(self):
        assert (
            str(self.pd) == "Li-Fe-O phase diagram\n11 stable phases: \nFe, FeO, Fe2O3, Fe3O4,"
            " LiFeO2, Li, Li2O, LiO, Li5FeO4, Li2FeO3, O"
        )

    def test_get_e_above_hull(self):
        for entry in self.pd.all_entries:
            for entry in self.pd.stable_entries:
                decomp, e_hull = self.pd.get_decomp_and_e_above_hull(entry)
                assert e_hull < 1e-11, "Stable entries should have e_above_hull of zero!"
                assert decomp[entry] == 1, "Decomposition of stable entry should be itself."

            e_ah = self.pd.get_e_above_hull(entry)
            assert isinstance(e_ah, Number)
            assert e_ah >= 0

    def test_get_decomp_and_e_above_hull_on_error(self):
        for method, expected in (
            (self.pd.get_e_above_hull, None),
            (self.pd.get_decomp_and_e_above_hull, (None, None)),
        ):
            # test raises ValueError on entry with element not in the phase diagram
            U_entry = PDEntry("U", 0)
            with pytest.raises(
                ValueError,
                match="Unable to get decomposition for PDEntry : U1 with energy",
            ):
                method(U_entry)

            # test raises ValueError on entry with very negative energy
            too_neg_entry = PDEntry("Li", -1e6)
            match_msg = "No valid decomposition found for PDEntry : Li1 with energy"
            with pytest.raises(ValueError, match=match_msg):
                method(too_neg_entry)

            with pytest.warns(UserWarning, match=match_msg):
                out = method(too_neg_entry, on_error="warn")
            assert out == expected

            out = method(too_neg_entry, on_error="ignore")
            assert out == expected

    def test_downstream_methods_can_also_ignore_errors(self):
        # test that downstream methods get_e_above_hull() and get_phase_separation_energy()
        # can also ignore errors
        too_neg_entry = PDEntry("Li", -1e6)
        exotic_entry = PDEntry("U W", -1e6)

        # get_e_above_hull
        with pytest.raises(ValueError, match="No valid decomposition found for PDEntry "):
            self.pd.get_e_above_hull(too_neg_entry, on_error="raise")
        assert self.pd.get_e_above_hull(too_neg_entry, on_error="ignore") is None

        with pytest.raises(ValueError, match="Unable to get decomposition for PDEntry"):
            self.pd.get_e_above_hull(exotic_entry, on_error="raise")
        assert self.pd.get_e_above_hull(exotic_entry, on_error="ignore") is None

        # get_phase_separation_energy
        with pytest.raises(ValueError, match="Unable to get decomposition for PDEntry"):
            self.pd.get_phase_separation_energy(exotic_entry, on_error="raise")
        assert self.pd.get_phase_separation_energy(exotic_entry, on_error="ignore") is None

    def test_get_equilibrium_reaction_energy(self):
        for entry in self.pd.stable_entries:
            assert self.pd.get_equilibrium_reaction_energy(entry) <= 0, (
                "Stable entries should have negative equilibrium reaction energy!"
            )

    def test_get_phase_separation_energy(self):
        for entry in self.pd.unstable_entries:
            if entry.composition.fractional_composition not in [
                entry.composition.fractional_composition for entry in self.pd.stable_entries
            ]:
                assert self.pd.get_phase_separation_energy(entry) >= 0, (
                    "Unstable entries should have positive decomposition energy!"
                )
            elif entry.is_element:
                el_ref = self.pd.el_refs[entry.elements[0]]
                e_d = entry.energy_per_atom - el_ref.energy_per_atom
                assert self.pd.get_phase_separation_energy(entry) == approx(e_d)
            # NOTE the remaining materials would require explicit tests as they
            # could be either positive or negative

        for entry in self.pd.stable_entries:
            if entry.composition.is_element:
                assert self.pd.get_phase_separation_energy(entry) == 0, (
                    "Stable elemental entries should have decomposition energy of zero!"
                )
            else:
                assert self.pd.get_phase_separation_energy(entry) <= 0, (
                    "Stable entries should have negative decomposition energy!"
                )
                assert self.pd.get_phase_separation_energy(entry, stable_only=True) == approx(
                    self.pd.get_equilibrium_reaction_energy(entry)
                ), "Using `stable_only=True` should give decomposition energy equal to equilibrium reaction energy!"

        # Test that we get correct behavior with a polymorph
        toy_entries = {"Li": 0.0, "Li2O": -5, "LiO2": -4, "O2": 0.0}

        toy_pd = PhaseDiagram([PDEntry(c, e) for c, e in toy_entries.items()])

        # stable entry
        assert toy_pd.get_phase_separation_energy(PDEntry("Li2O", -5)) == approx(-1.0)
        # polymorph
        assert toy_pd.get_phase_separation_energy(PDEntry("Li2O", -4)) == approx(-2.0 / 3.0)

        # Test that the method works for novel entries
        novel_stable_entry = PDEntry("Li5FeO4", -999)
        assert self.pd.get_phase_separation_energy(novel_stable_entry) < 0, (
            "Novel stable entries should have negative decomposition energy!"
        )

        novel_unstable_entry = PDEntry("Li5FeO4", 999)
        assert self.pd.get_phase_separation_energy(novel_unstable_entry) > 0, (
            "Novel unstable entries should have positive decomposition energy!"
        )

        duplicate_entry = PDEntry("Li2O", -14.31361175)
        scaled_dup_entry = PDEntry("Li4O2", -14.31361175 * 2)
        stable_entry = next(entry for entry in self.pd.stable_entries if entry.name == "Li2O")

        assert self.pd.get_phase_separation_energy(duplicate_entry) == self.pd.get_phase_separation_energy(
            stable_entry
        ), "Novel duplicates of stable entries should have same decomposition energy!"

        assert self.pd.get_phase_separation_energy(scaled_dup_entry) == self.pd.get_phase_separation_energy(
            stable_entry
        ), "Novel scaled duplicates of stable entries should have same decomposition energy!"

    def test_inner_hull_reduction(self):
        """Regression test for two related bugs in get_decomp_and_phase_separation_energy:

        Bug 1: Line 997 passed _get_stable_entries_in_space(entry_elems) unfiltered into
        competing_entries, which re-introduced the same-composition stable entry that was
        explicitly excluded at line 980 via same_comp_mem_ids. The SLSQP would then
        trivially decompose the target into itself, producing a wrong (too large) energy.

        Bug 2: Line 999 overwrote the inner hull result with the original full set,
        making PhaseDiagram(reduced_space) dead code and defeating the space_limit
        optimization.

        The fix: filter same_comp_mem_ids when adding stable entries back at line 997,
        then remove the overwrite at line 999.
        """
        # Pick a stable non-elemental entry as the polymorph target.
        # The gate at line 976 requires a stable same-composition entry to exist,
        # which is satisfied since we're using a stable entry itself.
        entry = next(e for e in self.pd.stable_entries if not e.is_element)

        # space_limit=1 forces the inner hull branch for any non-trivial entry.
        result_reduced = self.pd.get_decomp_and_phase_separation_energy(entry, space_limit=1)
        result_full = self.pd.get_decomp_and_phase_separation_energy(entry, space_limit=10_000)

        # Both paths must produce the same phase separation energy.
        assert result_reduced[1] == approx(result_full[1], abs=1e-5), (
            "Inner hull reduction produced a different phase separation energy. "
            "Likely cause: same-composition stable entry leaked into competing_entries."
        )

        # The same-composition entry must NOT appear in the decomposition from either path.
        entry_frac = entry.composition.fractional_composition
        for decomp, _ in (result_reduced, result_full):
            for competing in decomp:
                assert competing.composition.fractional_composition != entry_frac, (
                    "Same-composition entry appeared in decomposition — should have been excluded."
                )

    def test_get_decomposition(self):
        for entry in self.pd.stable_entries:
            assert len(self.pd.get_decomposition(entry.composition)) == 1, (
                "Stable composition should have only 1 decomposition!"
            )
        dim = len(self.pd.elements)
        for entry in self.pd.all_entries:
            n_decomp = len(self.pd.get_decomposition(entry.composition))
            assert n_decomp > 0
            assert n_decomp <= dim, (
                "The number of decomposition phases can at most be equal to the number of components."
            )

        # Just to test decomposition for a fictitious composition
        actual = {entry.formula: amt for entry, amt in self.pd.get_decomposition(Composition("Li3Fe7O11")).items()}
        expected = {
            "Fe2 O2": 0.0952380952380949,
            "Li1 Fe1 O2": 0.5714285714285714,
            "Fe6 O8": 0.33333333333333393,
        }
        assert actual == approx(expected)

    def test_get_transition_chempots(self):
        for el in self.pd.elements:
            assert len(self.pd.get_transition_chempots(el)) <= len(self.pd.facets)

    def test_get_element_profile(self):
        for el in self.pd.elements:
            for entry in self.pd.stable_entries:
                if not entry.composition.is_element:
                    assert len(self.pd.get_element_profile(el, entry.composition)) <= len(self.pd.facets)

        expected = [
            {
                "evolution": 1.0,
                "chempot": -4.2582781416666666,
                "reaction": "Li2O + 0.5 O2 -> Li2O2",
            },
            {"evolution": 0, "chempot": -5.08859066, "reaction": "Li2O -> Li2O"},
            {
                "evolution": -1.0,
                "chempot": -10.48758201,
                "reaction": "Li2O -> 2 Li + 0.5 O2",
            },
        ]
        result = self.pd.get_element_profile(Element("O"), Composition("Li2O"))
        for d1, d2 in zip(expected, result, strict=True):
            assert d1["evolution"] == approx(d2["evolution"])
            assert d1["chempot"] == approx(d2["chempot"])
            assert d1["reaction"] == str(d2["reaction"])

    def test_get_get_chempot_range_map(self):
        elements = [el for el in self.pd.elements if el.symbol != "Fe"]
        assert len(self.pd.get_chempot_range_map(elements)) == 10

    def test_getmu_vertices_stability_phase(self):
        results = self.pd.getmu_vertices_stability_phase(Composition("LiFeO2"), Element("O"))
        assert len(results) == approx(6)
        test_equality = False
        for c in results:
            if (
                abs(c[Element("O")] + 7.115) < 1e-2
                and abs(c[Element("Fe")] + 6.596) < 1e-2
                and abs(c[Element("Li")] + 3.931) < 1e-2
            ):
                test_equality = True
        assert test_equality, "there is an expected vertex missing in the list"

    def test_getmu_range_stability_phase(self):
        results = self.pd.get_chempot_range_stability_phase(Composition("LiFeO2"), Element("O"))
        assert results[Element("O")][1] == approx(-4.450181224)
        assert results[Element("Fe")][0] == approx(-6.5961470)
        assert results[Element("Li")][0] == approx(-3.6250022625)

    def test_get_hull_energy(self):
        for entry in self.pd.stable_entries:
            h_e = self.pd.get_hull_energy(entry.composition)
            assert h_e == approx(entry.energy)
            n_h_e = self.pd.get_hull_energy(entry.composition.fractional_composition)
            assert n_h_e == approx(entry.energy_per_atom)

    def test_get_hull_energy_per_atom(self):
        for entry in self.pd.stable_entries:
            h_e = self.pd.get_hull_energy_per_atom(entry.composition)
            assert h_e == approx(entry.energy_per_atom)

    def test_1d_pd(self):
        entry = PDEntry("H", 0)
        pd = PhaseDiagram([entry])
        decomp, e_above_hull = pd.get_decomp_and_e_above_hull(PDEntry("H", 1))
        assert e_above_hull == 1
        assert decomp[entry] == approx(1.0)

    def test_get_critical_compositions_fractional(self):
        c1 = Composition("Fe2O3").fractional_composition
        c2 = Composition("Li3FeO4").fractional_composition
        c3 = Composition("Li2O").fractional_composition

        comps = self.pd.get_critical_compositions(c1, c2)
        expected = [
            Composition("Fe2O3").fractional_composition,
            Composition("Li0.3243244Fe0.1621621O0.51351349"),
            Composition("Li3FeO4").fractional_composition,
        ]
        for crit, exp in zip(comps, expected, strict=True):
            assert crit.almost_equals(exp, rtol=0, atol=1e-5)

        comps = self.pd.get_critical_compositions(c1, c3)
        expected = [
            Composition("Fe0.4O0.6"),
            Composition("LiFeO2").fractional_composition,
            Composition("Li5FeO4").fractional_composition,
            Composition("Li2O").fractional_composition,
        ]
        for crit, exp in zip(comps, expected, strict=True):
            assert crit.almost_equals(exp, rtol=0, atol=1e-5)

    def test_get_critical_compositions(self):
        c1 = Composition("Fe2O3")
        c2 = Composition("Li3FeO4")
        c3 = Composition("Li2O")

        comps = self.pd.get_critical_compositions(c1, c2)
        expected = [
            Composition("Fe2O3"),
            Composition("Li0.3243244Fe0.1621621O0.51351349") * 7.4,
            Composition("Li3FeO4"),
        ]
        for crit, exp in zip(comps, expected, strict=True):
            assert crit.almost_equals(exp, rtol=0, atol=1e-5)

        comps = self.pd.get_critical_compositions(c1, c3)
        expected = [
            Composition("Fe2O3"),
            Composition("LiFeO2"),
            Composition("Li5FeO4") / 3,
            Composition("Li2O"),
        ]
        for crit, exp in zip(comps, expected, strict=True):
            assert crit.almost_equals(exp, rtol=0, atol=1e-5)

        # Don't fail silently if input compositions aren't in phase diagram
        # Can be very confusing if you're working with a GrandPotentialPD
        with pytest.raises(ValueError, match="Xe1 has elements not in the phase diagram Li, Fe, O"):
            self.pd.get_critical_compositions(Composition("Xe"), Composition("Mn"))

        # For the moment, should also fail even if compositions are in the gppd
        # because it isn't handled properly
        gppd = GrandPotentialPhaseDiagram(self.pd.all_entries, {"Xe": 1}, [*self.pd.elements, Element("Xe")])
        with pytest.raises(
            ValueError,
            match="Li3 Fe1 O4 Xe1 has elements not in the phase diagram",
        ):
            gppd.get_critical_compositions(Composition("Fe2O3"), Composition("Li3FeO4Xe"))

        # check that the function still works though
        comps = gppd.get_critical_compositions(c1, c2)
        expected = [
            Composition("Fe2O3"),
            Composition("Li0.3243244Fe0.1621621O0.51351349") * 7.4,
            Composition("Li3FeO4"),
        ]
        for crit, exp in zip(comps, expected, strict=True):
            assert crit.almost_equals(exp, rtol=0, atol=1e-5)

        # case where the endpoints are identical
        assert self.pd.get_critical_compositions(c1, c1 * 2) == [c1, c1 * 2]

    def test_get_composition_chempots(self):
        c1 = Composition("Fe3.1O4")
        c2 = Composition("Fe3.2O4.1Li0.01")

        e1 = self.pd.get_hull_energy(c1)
        e2 = self.pd.get_hull_energy(c2)

        cp = self.pd.get_composition_chempots(c1)
        calc_e2 = e1 + sum(cp[k] * v for k, v in (c2 - c1).items())
        assert e2 == approx(calc_e2)

    def test_get_all_chempots(self):
        c1 = Composition("Fe3.1O4")
        c2 = Composition("FeO")

        cp1 = self.pd.get_all_chempots(c1)
        cp_result = {
            Element("Li"): -4.077061954,
            Element("Fe"): -6.741593864,
            Element("O"): -6.969907375,
        }

        for elem, energy in cp_result.items():
            assert cp1["Fe3O4-FeO-LiFeO2"][elem] == approx(energy)

        cp2 = self.pd.get_all_chempots(c2)
        cp_result = {
            Element("O"): -7.11535414,
            Element("Fe"): -6.5961471,
            Element("Li"): -3.93161518,
        }

        for elem, energy in cp_result.items():
            assert cp2["FeO-LiFeO2-Fe"][elem] == approx(energy)

    def test_get_plot(self):
        self.pd.get_plot()  # PDPlotter functionality is tested separately

    def test_as_from_dict(self):
        # test round-trip for other entry types such as ComputedEntry
        entry = ComputedEntry("H", 0.0, 0.0, entry_id="test")
        pd = PhaseDiagram([entry])
        pd_dict = pd.as_dict()
        pd_roundtrip = PhaseDiagram.from_dict(pd_dict)
        assert pd.all_entries[0].entry_id == pd_roundtrip.all_entries[0].entry_id

        pd_dict = self.pd.as_dict()
        reconstructed_pd = PhaseDiagram.from_dict(pd_dict)
        reconstructed_pd_dict = reconstructed_pd.as_dict()
        assert reconstructed_pd_dict == pd_dict
        assert isinstance(pd.to_json(), str)

        assert MontyDecoder().process_decoded(pd_dict).as_dict() == pd_dict

        for entry in self.pd.all_entries:
            # NOTE: allow_negative=True is necessary due to fp errors we see after serialization
            decomp_rpd, e_above_hull_rppd = reconstructed_pd.get_decomp_and_e_above_hull(entry, allow_negative=True)
            decomp_pd, e_above_hull_ppd = self.pd.get_decomp_and_e_above_hull(entry)

            # Check that decompositions sum to the original composition
            for decomp, name in [(decomp_pd, "pd"), (decomp_rpd, "rpd")]:
                decomp_comp = Composition({})
                for e, amount in decomp.items():
                    comp_scaled = e.composition.fractional_composition * amount
                    decomp_comp += comp_scaled
                assert decomp_comp.almost_equals(
                    entry.composition.fractional_composition, rtol=0, atol=Composition.amount_tolerance
                ), (
                    f"Decomposition for {entry} from {name} does not sum to original composition! "
                    f"Expected {entry.composition}, got {decomp_comp}"
                )

            # Compare decompositions by matching entries by composition
            # since serialization creates new entry objects with potentially slightly different energies
            decomp_pd_by_comp = {e.composition: amount for e, amount in decomp_pd.items()}
            decomp_rpd_by_comp = {e.composition: amount for e, amount in decomp_rpd.items()}
            assert decomp_pd_by_comp == approx(decomp_rpd_by_comp), f"Decomposition for {entry} is not correct!"
            assert np.isclose(e_above_hull_rppd, e_above_hull_ppd, rtol=1e-4, atol=1e-4)

    def test_read_json(self):
        dumpfn(self.pd, f"{self.tmp_path}/pd.json")
        pd = loadfn(f"{self.tmp_path}/pd.json")
        assert isinstance(pd, PhaseDiagram)
        assert pd.elements == self.pd.elements
        assert {*pd.as_dict()} == {*self.pd.as_dict()}

    def test_el_refs(self):
        # Create an imitation of pre_computed phase diagram with el_refs keys being
        # tuple[str, PDEntry] instead of tuple[Element, PDEntry].
        mock_el_refs = [(str(el), entry) for el, entry in self.pd.el_refs.items()]
        mock_computed_data = {**self.pd.computed_data, "el_refs": mock_el_refs}
        pd = PhaseDiagram(self.entries, computed_data=mock_computed_data)
        # Check the keys in el_refs dict have been updated to Element object via PhaseDiagram class.
        assert all(isinstance(el, Element) for el in pd.el_refs)

    def test_val_err_on_no_entries(self):
        # check that PhaseDiagram raises ValueError when building phase diagram with no entries
        for entries in [None, [], set(), ()]:
            with pytest.raises(ValueError, match=re.escape("Unable to build phase diagram without entries.")):
                PhaseDiagram(entries=entries)


class TestGrandPotentialPhaseDiagram:
    def setup_method(self):
        self.entries = EntrySet.from_csv(f"{TEST_DIR}/pd_entries_test.csv")
        self.pd = GrandPotentialPhaseDiagram(self.entries, {Element("O"): -5})
        self.pd6 = GrandPotentialPhaseDiagram(self.entries, {Element("O"): -6})

    def test_stable_entries(self):
        stable_formulas = [ent.original_entry.reduced_formula for ent in self.pd.stable_entries]
        expected_stable = ["Li5FeO4", "Li2FeO3", "LiFeO2", "Fe2O3", "Li2O2"]
        for formula in expected_stable:
            assert formula in stable_formulas, f"{formula} not in stable entries!"
        assert len(self.pd6.stable_entries) == 4

    def test_get_formation_energy(self):
        stable_formation_energies = {
            ent.original_entry.reduced_formula: self.pd.get_form_energy(ent) for ent in self.pd.stable_entries
        }
        expected_formation_energies = {
            "Fe2O3": 0.0,
            "Li5FeO4": -5.30551504,
            "Li2FeO3": -2.34247415,
            "LiFeO2": -0.4302639625,
            "Li2O2": 0.0,
        }
        for formula, energy in expected_formation_energies.items():
            assert energy == approx(stable_formation_energies[formula]), (
                f"Calculated formation for {formula} is not correct!"
            )

    def test_str(self):
        # using startswith since order of stable phases is random
        assert str(self.pd).startswith(
            "Fe-Li GrandPotentialPhaseDiagram with chempots = 'mu_O = -5.0000'5 stable phases: "
        )


def test_get_equilibrium_reaction_energy_open_element_in_original_composition():
    entries = [
        PDEntry(Composition("Li"), -2.0),
        PDEntry(Composition("Fe"), -8.0),
        PDEntry(Composition("O2"), -10.0),
        PDEntry(Composition("LiFe3O"), -40.0),
    ]
    gpd = GrandPotentialPhaseDiagram(entries, {"Li": -2.0})

    # Stable GP entry has projected composition Fe3O but original composition LiFe3O.
    gp_entry = next(
        entry
        for entry in gpd.stable_entries
        if entry.composition == Composition("Fe3O") and entry.original_comp == Composition("LiFe3O")
    )
    assert gp_entry in gpd.stable_entries

    ere = gpd.get_equilibrium_reaction_energy(gp_entry)
    assert isinstance(ere, float)
    assert ere <= 0


class TestCompoundPhaseDiagram:
    def setup_method(self):
        self.entries = EntrySet.from_csv(f"{TEST_DIR}/pd_entries_test.csv")
        self.pd = CompoundPhaseDiagram(self.entries, [Composition("Li2O"), Composition("Fe2O3")])

    def test_stable_entries(self):
        stable_formulas = [ent.name for ent in self.pd.stable_entries]
        expected_stable = ["Fe2O3", "Li5FeO4", "LiFeO2", "Li2O"]
        for formula in expected_stable:
            assert formula in stable_formulas

    def test_get_formation_energy(self):
        stable_formation_energies = {ent.name: self.pd.get_form_energy(ent) for ent in self.pd.stable_entries}
        expected_formation_energies = {
            "Li5FeO4": -7.0773284399999739,
            "Fe2O3": 0,
            "LiFeO2": -0.4745592975,
            "Li2O": 0,
        }
        for formula, energy in expected_formation_energies.items():
            assert energy == approx(stable_formation_energies[formula])

    def test_str(self):
        assert str(self.pd) == "Xf-Xg phase diagram\n4 stable phases: \nLiFeO2, Li2O, Li5FeO4, Fe2O3"

    def test_num2str(self):
        ret = set()

        num = 100
        for idx in range(num):
            val = CompoundPhaseDiagram.num2str(idx)
            assert val.isalpha()
            assert ord(val[0]) >= ord("f")
            ret.add(val)

        assert len(ret) == num


class TestPatchedPhaseDiagram:
    def setup_method(self):
        self.entries = EntrySet.from_csv(f"{TEST_DIR}/reaction_entries_test.csv")
        # NOTE add He to test for correct behavior despite no patches involving He
        self.no_patch_entry = he_entry = PDEntry("He", -1.23)
        self.entries.add(he_entry)

        self.pd = PhaseDiagram(entries=self.entries)
        self.ppd = PatchedPhaseDiagram(entries=self.entries)
        self.ppd_all = PatchedPhaseDiagram(entries=self.entries, keep_all_spaces=True)

        # novel entries not in any of the patches
        self.novel_comps = [Composition("H5C2OP"), Composition("V2PH4C")]
        for c in self.novel_comps:
            assert c.chemical_system not in self.ppd.spaces

        self.novel_entries = [PDEntry(c, -39.8) for c in self.novel_comps]

    def test_get_stable_entries(self):
        assert self.pd.stable_entries == self.ppd.stable_entries

    def test_get_qhull_entries(self):
        # NOTE qhull_entry is an specially sorted list due to it's construction, we
        # can't mimic this in ppd therefore just test if sorted versions are equal.
        assert sorted(self.pd.qhull_entries, key=lambda e: e.composition) == sorted(
            self.ppd.qhull_entries, key=lambda e: e.composition
        )

    def test_get_decomposition(self):
        for comp in self.novel_comps:
            decomp_pd = self.pd.get_decomposition(comp)
            decomp_ppd = self.ppd.get_decomposition(comp)
            assert decomp_pd == approx(decomp_ppd)

    def test_get_phase_separation_energy(self):
        for entry in self.novel_entries:
            e_phase_sep_pd = self.pd.get_phase_separation_energy(entry)
            e_phase_sep_ppd = self.ppd.get_phase_separation_energy(entry)
            assert np.isclose(e_phase_sep_pd, e_phase_sep_ppd)

    def test_get_equilibrium_reaction_energy(self):
        for entry in self.pd.stable_entries:
            e_equi_rxn_pd = self.pd.get_equilibrium_reaction_energy(entry)
            e_equi_rxn_pdd = self.ppd.get_equilibrium_reaction_energy(entry)
            assert np.isclose(e_equi_rxn_pd, e_equi_rxn_pdd)

    def test_get_form_energy(self):
        for entry in self.pd.stable_entries:
            e_form_pd = self.pd.get_form_energy(entry)
            e_form_ppd = self.ppd.get_form_energy(entry)
            assert np.isclose(e_form_pd, e_form_ppd)

    def test_dimensionality(self):
        assert self.pd.dim == self.ppd.dim

        # test dims of sub PDs
        dim_counts = collections.Counter(pd.dim for pd in self.ppd.pds.values())
        assert dim_counts == {4: 2, 3: 2}

        # test dims of sub PDs
        dim_counts = collections.Counter(pd.dim for pd in self.ppd_all.pds.values())
        assert dim_counts == {2: 8, 3: 7, 4: 2}

    def test_get_hull_energy(self):
        for comp in self.novel_comps:
            e_hull_pd = self.pd.get_hull_energy(comp)
            e_hull_ppd = self.ppd.get_hull_energy(comp)
            assert np.isclose(e_hull_pd, e_hull_ppd)

    def test_get_decomp_and_e_above_hull(self):
        for entry in self.pd.all_entries:
            decomp_pd, e_above_hull_pd = self.pd.get_decomp_and_e_above_hull(entry)
            decomp_ppd, e_above_hull_ppd = self.ppd.get_decomp_and_e_above_hull(entry, check_stable=True)

            # Check that decompositions sum to the original composition
            for decomp, name in [(decomp_pd, "pd"), (decomp_ppd, "ppd")]:
                decomp_comp = Composition({})
                for e, amount in decomp.items():
                    comp_scaled = e.composition.fractional_composition * amount
                    decomp_comp += comp_scaled
                assert decomp_comp.almost_equals(
                    entry.composition.fractional_composition, rtol=0, atol=Composition.amount_tolerance
                ), (
                    f"Decomposition for {entry} from {name} does not sum to original composition! "
                    f"Expected {entry.composition}, got {decomp_comp}"
                )

            assert decomp_pd == approx(decomp_ppd)
            assert np.isclose(e_above_hull_pd, e_above_hull_ppd)

    def test_repr(self):
        assert repr(self.ppd) == str(self.ppd) == "PatchedPhaseDiagram covering 4 sub-spaces"

    def test_as_from_dict(self):
        ppd_dict = self.ppd.as_dict()
        assert ppd_dict["@module"] == type(self.ppd).__module__
        assert ppd_dict["@class"] == type(self.ppd).__name__

        # Check format with computed_data
        assert "computed_data" in ppd_dict
        computed_data = ppd_dict["computed_data"]
        assert "all_entries" in computed_data
        assert isinstance(computed_data["all_entries"], list)
        assert all(isinstance(entry, dict) for entry in computed_data["all_entries"])

        # Verify entries can be reconstructed
        reconstructed_entries = [MontyDecoder().process_decoded(entry) for entry in computed_data["all_entries"]]
        assert len(reconstructed_entries) == len(self.ppd.all_entries)
        assert ppd_dict["elements"] == [elem.symbol for elem in self.ppd.elements]

        reconstructed_ppd = PatchedPhaseDiagram.from_dict(ppd_dict)
        reconstructed_dict = reconstructed_ppd.as_dict()
        assert ppd_dict == reconstructed_dict

        assert MontyDecoder().process_decoded(ppd_dict).as_dict() == ppd_dict

        for entry in self.ppd.all_entries:
            decomp_pd, e_above_hull_pd = self.pd.get_decomp_and_e_above_hull(entry)
            for pd, name in [
                (self.ppd, "ppd"),
                (reconstructed_ppd, "rppd"),
            ]:
                # NOTE: allow_negative=True is necessary due to fp errors we see after serialization
                decomp, e_above_hull = pd.get_decomp_and_e_above_hull(entry, check_stable=True, allow_negative=True)
                decomp_comp = Composition({})
                for e, amount in decomp.items():
                    comp_scaled = e.composition.fractional_composition * amount
                    decomp_comp += comp_scaled
                assert decomp_comp.almost_equals(
                    entry.composition.fractional_composition, rtol=0, atol=Composition.amount_tolerance
                ), (
                    f"Decomposition for {entry} from {name} does not sum to original composition! "
                    f"Expected {entry.composition}, got {decomp_comp}"
                )

                assert decomp == approx(decomp_pd), f"Decomposition for {entry} is not correct!"
                assert np.isclose(e_above_hull, e_above_hull_pd)

    def test_read_json(self, tmp_path):
        dumpfn(self.ppd, f"{tmp_path}/ppd.json")
        ppd = loadfn(f"{tmp_path}/ppd.json")
        assert isinstance(ppd, PatchedPhaseDiagram)
        assert ppd.elements == self.ppd.elements
        assert {*ppd.as_dict()} == {*self.ppd.as_dict()}

    def test_get_pd_for_entry(self):
        for entry in self.ppd.all_entries:
            if entry == self.no_patch_entry:
                continue
            pd = self.ppd.get_pd_for_entry(entry)
            # test that entry is in pd and pd can return valid decomp
            assert isinstance(pd.get_decomposition(entry.composition), dict)

        with pytest.raises(ValueError, match="No suitable PhaseDiagrams found for PDEntry"):
            self.ppd.get_pd_for_entry(self.no_patch_entry)

    def test_raises_on_missing_terminal_entries(self):
        entry = PDEntry("FeO", -1.23)
        with pytest.raises(ValueError, match=r"Missing terminal entries for elements \['Fe', 'O'\]"):
            PatchedPhaseDiagram(entries=[entry])

    def test_contains(self):
        for space in self.ppd.spaces:
            assert space in self.ppd
        unlikely_chem_space = frozenset(map(Element, "HBCNOFPS"))
        assert unlikely_chem_space not in self.ppd

    def test_getitem(self):
        chem_space = self.ppd.spaces[0]
        pd = self.ppd[chem_space]
        assert isinstance(pd, PhaseDiagram)
        assert chem_space in pd._qhull_spaces

        with pytest.raises(KeyError, match="frozenset"):
            self.ppd[frozenset(map(Element, "HBCNOFPS"))]

    def test_iter(self):
        for pd in self.ppd:
            assert isinstance(pd, PhaseDiagram)
        assert len(self.ppd) == len(self.ppd.pds)

    def test_len(self):
        assert len(self.ppd) == len(self.ppd.pds)

    def test_setitem_and_delitem(self):
        unlikely_chem_space = frozenset(map(Element, "HBCNOFPS"))
        self.ppd[unlikely_chem_space] = self.pd
        assert unlikely_chem_space in self.ppd
        assert self.ppd[unlikely_chem_space] == self.pd
        del self.ppd[unlikely_chem_space]  # test __delitem__() and restore original state

    def test_remove_redundant_spaces(self):
        spaces = tuple(frozenset(entry.elements) for entry in self.ppd.qhull_entries)
        # NOTE this is 5 not 4 as "He" is a non redundant space that gets dropped for other reasons
        assert len(self.ppd.remove_redundant_spaces(spaces)) == 5

        test = (
            list(combinations(range(1, 7), 4))
            + list(combinations(range(1, 10), 2))
            + list(combinations([1, 4, 7, 9, 2], 5))
        )
        test = [frozenset(t) for t in test]
        assert len(self.ppd.remove_redundant_spaces(test)) == 30

    def test_update_returns_new_instance(self):
        """update() must return a new PatchedPhaseDiagram, not mutate self."""
        original_stable = self.ppd.stable_entries.copy()
        ppd_new = self.ppd.update([PDEntry("V2O5", -500)])
        assert ppd_new is not self.ppd
        assert isinstance(ppd_new, PatchedPhaseDiagram)
        assert self.ppd.stable_entries == original_stable

    def test_update_with_empty_entries(self):
        ppd_new = self.ppd.update([])
        assert ppd_new is self.ppd

    def test_update_entry_in_existing_space(self):
        """Add a very low energy entry in an existing chemical space and verify
        it becomes stable and the patched PD matches a full rebuild."""
        new_entry = PDEntry("V2O5", -500)
        ppd_new, info = self.ppd.update([new_entry], return_info=True)
        assert new_entry in info["new_stable_entries"]
        assert len(info["new_spaces"]) == 0
        assert any(frozenset(new_entry.elements).issubset(s) for s in info["updated_spaces"])
        assert new_entry in ppd_new.stable_entries
        ppd_rebuilt = PatchedPhaseDiagram(entries=[*list(self.entries), new_entry])
        assert ppd_new.stable_entries == ppd_rebuilt.stable_entries

    def test_update_new_chemical_space(self):
        """Add entries that span a chemical space not present in any existing patch."""
        new_entries = [PDEntry("FeO", -10), PDEntry("Fe", -1), PDEntry("O", -2)]
        existing_elements = {el for entry in self.entries for el in entry.elements}
        fe_el = Element("Fe")
        o_el = Element("O")
        if fe_el in existing_elements and o_el in existing_elements:
            ppd = PatchedPhaseDiagram(entries=self.entries)
            fe_o_space = frozenset([fe_el, o_el])
            if not any(s.issuperset(fe_o_space) for s in ppd.spaces):
                ppd_new, info = ppd.update(new_entries, return_info=True)
                assert len(info["new_spaces"]) > 0
            else:
                ppd_full = PatchedPhaseDiagram(entries=list(self.entries) + new_entries)
                ppd_new = ppd.update(new_entries)
                assert ppd_new.stable_entries == ppd_full.stable_entries

    def test_update_matches_full_rebuild(self):
        """The updated PPD should give the same stable entries, decompositions,
        and hull energies as a PPD built from scratch with all entries."""
        new_entries = [PDEntry("V2O3", -200), PDEntry("PH3", -50)]
        ppd_updated = PatchedPhaseDiagram(entries=self.entries).update(new_entries)
        all_entries = list(self.entries) + new_entries
        ppd_full = PatchedPhaseDiagram(entries=all_entries)
        assert ppd_updated.stable_entries == ppd_full.stable_entries
        for comp in self.novel_comps:
            e_hull_updated = ppd_updated.get_hull_energy(comp)
            e_hull_full = ppd_full.get_hull_energy(comp)
            assert np.isclose(e_hull_updated, e_hull_full)

    def test_update_unstable_entry_no_change(self):
        """Adding a high-energy entry should not change stable entries."""
        stable_before = self.ppd.stable_entries.copy()
        ppd_new, info = self.ppd.update([PDEntry("V2O5", 1000)], return_info=True)
        assert info["new_stable_entries"] == set()
        assert info["removed_stable_entries"] == set()
        assert ppd_new.stable_entries == stable_before

    def test_update_multiple_batches(self):
        """Simulate a discovery campaign with multiple sequential updates."""
        ppd = PatchedPhaseDiagram(entries=self.entries)
        batch1 = [PDEntry("V2O5", -500)]
        ppd = ppd.update(batch1)
        batch2 = [PDEntry("PH3", -50)]
        ppd = ppd.update(batch2)
        all_entries = list(self.entries) + batch1 + batch2
        ppd_full = PatchedPhaseDiagram(entries=all_entries)
        assert ppd.stable_entries == ppd_full.stable_entries

    def test_update_rejects_new_elements(self):
        """Entries with elements not in the original PPD should raise."""
        with pytest.raises(ValueError, match="does not support adding new elements"):
            self.ppd.update([PDEntry("LiF", -10)])

    def test_update_returns_removed_stable(self):
        """When a new entry displaces an old stable entry, it should appear
        in removed_stable_entries."""
        stable_formulas = {e.reduced_formula for e in self.ppd.stable_entries}
        if "V2O5" in stable_formulas:
            target_comp = "V2O5"
        else:
            target_comp = next(e.reduced_formula for e in self.ppd.stable_entries if not e.composition.is_element)
        old_entry = next(e for e in self.ppd.stable_entries if e.reduced_formula == target_comp)
        crusher = PDEntry(target_comp, old_entry.energy * 10)
        _, info = self.ppd.update([crusher], return_info=True)
        assert crusher in info["new_stable_entries"]
        assert old_entry in info["removed_stable_entries"]

    def test_update_preserves_keep_all_spaces(self):
        """The keep_all_spaces setting should be stored and respected on update."""
        ppd_all = PatchedPhaseDiagram(entries=self.entries, keep_all_spaces=True)
        assert ppd_all.keep_all_spaces is True
        n_spaces_before = len(ppd_all.spaces)
        ppd_new = ppd_all.update([PDEntry("V2O5", -500)])
        assert ppd_new.keep_all_spaces is True
        assert len(ppd_new.spaces) >= n_spaces_before

    def test_update_shares_entry_objects(self):
        """Entry objects should be shared (same id) between old and new PPD."""
        original_entry_ids = {id(e) for e in self.ppd.all_entries}
        ppd_new = self.ppd.update([PDEntry("V2O5", -500)])
        carried_ids = {id(e) for e in ppd_new.all_entries} & original_entry_ids
        assert carried_ids == original_entry_ids

    def test_update_same_comp_lower_energy_replaces(self):
        """A new entry with the same composition but lower energy should
        replace the existing qhull entry and become stable."""
        target = next(
            e for e in self.ppd.stable_entries if not e.composition.is_element and len(e.composition.elements) >= 2
        )
        better = PDEntry(target.composition, target.energy * 10)
        ppd_new, info = self.ppd.update([better], return_info=True)
        assert better in ppd_new.stable_entries
        assert better in info["new_stable_entries"]
        assert target in info["removed_stable_entries"]
        rc = target.composition.reduced_composition
        qhull_for_comp = [e for e in ppd_new.qhull_entries if e.composition.reduced_composition == rc]
        assert len(qhull_for_comp) == 1
        assert qhull_for_comp[0] is better

    def test_update_same_comp_higher_energy_noop(self):
        """A new entry with the same composition but higher energy should
        not replace the existing qhull entry or change stability."""
        target = next(
            e for e in self.ppd.stable_entries if not e.composition.is_element and len(e.composition.elements) >= 2
        )
        worse = PDEntry(target.composition, target.energy / 10)
        ppd_new, info = self.ppd.update([worse], return_info=True)
        assert target in ppd_new.stable_entries
        assert worse not in ppd_new.stable_entries
        assert info["new_stable_entries"] == set()
        assert info["removed_stable_entries"] == set()
        rc = target.composition.reduced_composition
        qhull_for_comp = [e for e in ppd_new.qhull_entries if e.composition.reduced_composition == rc]
        assert len(qhull_for_comp) == 1
        assert qhull_for_comp[0] is target

    def test_update_lower_elemental_ref_raises(self):
        """By default, a new elemental entry with lower energy should raise."""
        el = next(iter(self.ppd.el_refs))
        old_ref = self.ppd.el_refs[el]
        new_ref = PDEntry(str(el), old_ref.energy_per_atom * 2)
        with pytest.raises(ValueError, match="lower than current el_ref"):
            self.ppd.update([new_ref])

    def test_update_lower_elemental_ref_ignore(self):
        """With on_new_el_ref='ignore', el_refs should stay unchanged."""
        el = next(iter(self.ppd.el_refs))
        old_ref = self.ppd.el_refs[el]
        new_ref = PDEntry(str(el), old_ref.energy_per_atom * 2)
        ppd_new = self.ppd.update([new_ref], on_new_el_ref="ignore")
        assert ppd_new.el_refs[el] is old_ref

    def test_update_lower_elemental_ref_recalculate(self):
        """With on_new_el_ref='recalculate', el_refs should update and
        affected spaces should be rebuilt."""
        el = next(iter(self.ppd.el_refs))
        old_ref = self.ppd.el_refs[el]
        new_ref = PDEntry(str(el), old_ref.energy_per_atom * 2)
        ppd_new, info = self.ppd.update([new_ref], return_info=True, on_new_el_ref="recalculate")
        assert ppd_new.el_refs[el] is new_ref
        affected_els = {el}
        for space in info["updated_spaces"]:
            assert space & affected_els
        ppd_full = PatchedPhaseDiagram(entries=[*list(self.entries), new_ref])
        assert ppd_new.stable_entries == ppd_full.stable_entries

    def test_update_qhull_dedup_across_batches(self):
        """After multiple updates adding entries for the same composition,
        qhull_entries should contain only one entry per reduced composition."""
        comp = "V2O5"
        ppd = PatchedPhaseDiagram(entries=self.entries)
        ppd = ppd.update([PDEntry(comp, -100)])
        ppd = ppd.update([PDEntry(comp, -500)])
        rc = Composition(comp).reduced_composition
        qhull_for_comp = [e for e in ppd.qhull_entries if e.composition.reduced_composition == rc]
        assert len(qhull_for_comp) == 1
        assert np.isclose(qhull_for_comp[0].energy, -500)

    def test_update_all_positive_formation_energy(self):
        """When all new entries have positive formation energy, the PPD
        should still return a new instance with all_entries updated but
        stable entries unchanged."""
        stable_before = self.ppd.stable_entries.copy()
        n_entries_before = len(self.ppd.all_entries)
        ppd_new = self.ppd.update([PDEntry("V2O5", 9999)])
        assert ppd_new.stable_entries == stable_before
        assert len(ppd_new.all_entries) == n_entries_before + 1


class TestReactionDiagram:
    def setup_method(self):
        self.entries = list(EntrySet.from_csv(f"{TEST_DIR}/reaction_entries_test.csv").entries)
        for entry in self.entries:
            if entry.reduced_formula == "VPO5":
                entry1 = entry
            elif entry.reduced_formula == "H4(CO)3":
                entry2 = entry
        self.rd = ReactionDiagram(entry1=entry1, entry2=entry2, all_entries=self.entries[2:])

    def test_get_compound_pd(self):
        self.rd.get_compound_pd()

    def test_formula(self):
        for entry in self.rd.rxn_entries:
            assert Element.V in entry.composition
            assert Element.O in entry.composition
            assert Element.C in entry.composition
            assert Element.P in entry.composition
            assert Element.H in entry.composition
        # formed_formula = [entry.reduced_formula for entry in self.rd.rxn_entries]
        # expected_formula = [
        #     "V0.12707182P0.12707182H0.0441989C0.03314917O0.66850829",
        #     "V0.125P0.125H0.05C0.0375O0.6625",
        #     "V0.12230216P0.12230216H0.05755396C0.04316547O0.65467626",
        #     "V0.11340206P0.11340206H0.08247423C0.06185567O0.62886598",
        #     "V0.11267606P0.11267606H0.08450704C0.06338028O0.62676056",
        #     "V0.11229947P0.11229947H0.0855615C0.06417112O0.62566845",
        #     "V0.09677419P0.09677419H0.12903226C0.09677419O0.58064516",
        #     "V0.05882353P0.05882353H0.23529412C0.17647059O0.47058824",
        #     "V0.04225352P0.04225352H0.28169014C0.21126761O0.42253521",
        # ]

        # # Please do not uncomment this test. This test is far too fragile because of numerical precision errors.
        # # Unless someone wants to make an effort to write a PROPER test which do not fail with changes in
        # # OS or numpy versions, DO NOT UNCOMMENT!
        # for formula in expected_formula:
        #     assert formula in formed_formula, f"{formed_formula=} not in {expected_formula=}"


class TestPDPlotter:
    def setup_method(self):
        entries = list(EntrySet.from_csv(f"{TEST_DIR}/pd_entries_test.csv"))

        elemental_entries = [entry for entry in entries if entry.elements == [Element("Li")]]
        self.pd_unary = PhaseDiagram(elemental_entries)
        self.plotter_unary_plotly = PDPlotter(self.pd_unary, backend="plotly")

        entries_LiO = [entry for entry in entries if "Fe" not in entry.composition]
        self.pd_binary = PhaseDiagram(entries_LiO)
        self.plotter_binary_mpl = PDPlotter(self.pd_binary, backend="matplotlib")
        self.plotter_binary_plotly = PDPlotter(self.pd_binary, backend="plotly")

        self.pd_ternary = PhaseDiagram(entries)
        self.plotter_ternary_mpl = PDPlotter(self.pd_ternary, backend="matplotlib")
        self.plotter_ternary_plotly_2d = PDPlotter(self.pd_ternary, backend="plotly", ternary_style="2d")
        self.plotter_ternary_plotly_3d = PDPlotter(self.pd_ternary, backend="plotly", ternary_style="3d")

        entries.append(PDEntry("C", 0))
        self.pd_quaternary = PhaseDiagram(entries)
        self.plotter_quaternary_mpl = PDPlotter(self.pd_quaternary, backend="matplotlib")
        self.plotter_quaternary_plotly = PDPlotter(self.pd_quaternary, backend="plotly")

    def test_plot_pd_with_no_unstable(self):
        # https://github.com/materialsproject/pymatgen/issues/2885
        pd_entries = [PDEntry(comp, 0) for comp in ["Li", "Co", "O"]]
        pd = PhaseDiagram(pd_entries)
        plotter = PDPlotter(pd, backend="plotly", show_unstable=False)
        ax = plotter.get_plot()
        assert isinstance(ax, go.Figure)
        assert len(ax.data) == 4

    def test_pd_plot_data(self):
        lines, labels, unstable_entries = self.plotter_ternary_mpl.pd_plot_data
        assert len(lines) == 22
        assert len(labels) == len(self.pd_ternary.stable_entries), "Incorrect number of lines generated!"
        assert len(unstable_entries) == len(self.pd_ternary.all_entries) - len(self.pd_ternary.stable_entries), (
            "Incorrect number of lines generated!"
        )
        lines, labels, unstable_entries = self.plotter_quaternary_mpl.pd_plot_data
        assert len(lines) == 33
        assert len(labels) == len(self.pd_quaternary.stable_entries)
        assert len(unstable_entries) == len(self.pd_quaternary.all_entries) - len(self.pd_quaternary.stable_entries)
        lines, labels, unstable_entries = self.plotter_binary_mpl.pd_plot_data
        assert len(lines) == 3
        assert len(labels) == len(self.pd_binary.stable_entries)

    def test_matplotlib_plots(self):
        # Some very basic ("non")-tests. Just to make sure the methods are callable.
        for plotter in (
            self.plotter_binary_mpl,
            self.plotter_ternary_mpl,
            self.plotter_quaternary_mpl,
        ):
            ax = plotter.get_plot()
            assert isinstance(ax, plt.Axes)

        self.plotter_ternary_mpl.get_contour_pd_plot()
        self.plotter_ternary_mpl.get_chempot_range_map_plot([Element("Li"), Element("O")])
        self.plotter_ternary_mpl.plot_element_profile(Element("O"), Composition("Li2O"))

        # test show()
        assert self.plotter_ternary_mpl.show() is None

    def test_plotly_plots(self):
        # Also very basic tests. Ensures callability and 2D vs 3D properties.
        for plotter in (
            self.plotter_unary_plotly,
            self.plotter_binary_plotly,
            self.plotter_ternary_plotly_2d,
            self.plotter_ternary_plotly_3d,
            self.plotter_quaternary_plotly,
        ):
            fig = plotter.get_plot()
            assert isinstance(fig, go.Figure)

        # test show()
        # suppress default plotly behavior of opening figure in browser by patching plotly.io.show to noop
        with patch("plotly.io.show") as mock_show:
            assert self.plotter_ternary_plotly_2d.show() is None
            mock_show.assert_called_once()


class TestUtilityFunction:
    def test_unique_lines(self):
        testdata = [
            [5, 53, 353],
            [399, 20, 52],
            [399, 400, 20],
            [13, 399, 52],
            [21, 400, 353],
            [393, 5, 353],
            [400, 393, 353],
            [393, 400, 399],
            [393, 13, 5],
            [13, 393, 399],
            [400, 17, 20],
            [21, 17, 400],
        ]
        expected = {
            (5, 393),
            (21, 353),
            (353, 400),
            (5, 13),
            (17, 20),
            (21, 400),
            (17, 400),
            (52, 399),
            (393, 399),
            (20, 52),
            (353, 393),
            (5, 353),
            (5, 53),
            (13, 399),
            (393, 400),
            (13, 52),
            (53, 353),
            (17, 21),
            (13, 393),
            (20, 399),
            (399, 400),
            (20, 400),
        }
        assert uniquelines(testdata) == expected

    def test_triangular_coord(self):
        coord = [0.5, 0.5]
        coord = triangular_coord(coord)
        assert_allclose(coord, [0.75, 0.4330127])

    def test_tet_coord(self):
        coord = [0.5, 0.5, 0.5]
        coord = tet_coord(coord)
        assert_allclose(coord, [1.0, 0.57735027, 0.40824829])

    @pytest.mark.parametrize(
        "ordering",
        [
            ["O", "Li", "Fe"],
            ["O", "Fe", "Li"],
            ["Li", "Fe", "O"],
            ["Li", "O", "Fe"],
            ["Fe", "O", "Li"],
            ["Fe", "Li", "O"],
        ],
    )
    def test_order_phase_diagram(self, ordering):
        entries = EntrySet.from_csv(f"{TEST_DIR}/pd_entries_test.csv")
        lines, stable, unstable = PDPlotter(PhaseDiagram(entries), backend="matplotlib").pd_plot_data
        new_lines, new_stable, new_unstable = order_phase_diagram(lines, stable, unstable, ordering)
        assert len(new_lines) == len(lines)
        assert len(new_stable) == len(stable)
        assert set(new_unstable) == set(unstable)
        # The requested Up/Left/Right element now sits on the corresponding triangle vertex
        targets = {ordering[0]: (0.5, np.sqrt(3) / 2), ordering[1]: (0, 0), ordering[2]: (1, 0)}
        for coord, entry in new_stable.items():
            if entry.name in targets:
                assert_allclose(coord, targets[entry.name], atol=1e-8)
        # Rotations and reflections are isometries; pairwise distances are preserved
        stable_entries = list(stable.values())
        old_coords = {entry: np.array(c) for c, entry in stable.items()}
        new_coords = {entry: np.array(c) for c, entry in new_stable.items()}
        for e1, e2 in combinations(stable_entries[:5], 2):
            assert np.linalg.norm(new_coords[e1] - new_coords[e2]) == approx(
                np.linalg.norm(old_coords[e1] - old_coords[e2])
            )

    def test_order_phase_diagram_invalid(self):
        entries = EntrySet.from_csv(f"{TEST_DIR}/pd_entries_test.csv")
        lines, stable, unstable = PDPlotter(PhaseDiagram(entries), backend="matplotlib").pd_plot_data
        with pytest.raises(ValueError, match="should be in ordering"):
            order_phase_diagram(lines, stable, unstable, ["Up", "Left", "Right"])


class TestPhaseDiagramEdgeCases:
    def setup_method(self):
        self.entries = EntrySet.from_csv(f"{TEST_DIR}/pd_entries_test.csv")
        self.pd = PhaseDiagram(self.entries)

    def test_computed_data_type_error(self):
        with pytest.raises(TypeError, match="computed_data should be dict, got list"):
            PhaseDiagram(self.entries, computed_data=[1, 2])

    def test_extra_terminal_elements(self):
        with pytest.raises(ValueError, match=r"more terminal elements than dimensions: \['Fe'\]"):
            PhaseDiagram(self.entries, elements=[Element("Li"), Element("O")])

    def test_from_dict_old_top_level_all_entries(self):
        dct = {
            "elements": [el.symbol for el in self.pd.elements],
            "all_entries": [entry.as_dict() for entry in self.pd.all_entries],
        }
        pd = PhaseDiagram.from_dict(dct)
        assert {e.composition.reduced_formula for e in pd.stable_entries} == {
            e.composition.reduced_formula for e in self.pd.stable_entries
        }

    def test_from_dict_legacy_fallback(self):
        dct = self.pd.as_dict()
        # Legacy format: elements as MSON dicts and el_refs as (Element dict, Entry dict) pairs
        dct["elements"] = [el.as_dict() for el in self.pd.elements]
        dct["computed_data"]["el_refs"] = [(el.as_dict(), entry.as_dict()) for el, entry in self.pd.el_refs.items()]
        with pytest.warns(FutureWarning, match="Deprecated on 2025-02-03"):
            pd = PhaseDiagram.from_dict(dct)
        assert len(pd.stable_entries) == len(self.pd.stable_entries)
        for el, ref in self.pd.el_refs.items():
            assert pd.el_refs[el].energy_per_atom == approx(ref.energy_per_atom)
        li2o = Composition("Li2O")
        assert pd.get_hull_energy_per_atom(li2o) == approx(self.pd.get_hull_energy_per_atom(li2o))

    def test_get_reference_energy_per_atom(self):
        comp = Composition("Li2O")
        el_refs = self.pd.el_refs
        expected = (2 * el_refs[Element("Li")].energy_per_atom + el_refs[Element("O")].energy_per_atom) / 3
        assert self.pd.get_reference_energy_per_atom(comp) == approx(expected)
        assert self.pd.get_reference_energy_per_atom(comp) == approx(-2.694769293888889)

    def test_no_facet_found(self):
        # Simulate a composition that lies in no simplex (e.g. due to numerical issues)
        comp = Composition("Li2O")
        assert len(self.pd._get_all_facets_and_simplexes(comp)) >= 1
        with patch.object(Simplex, "in_simplex", return_value=False):
            with pytest.raises(RuntimeError, match="No facet found for comp"):
                self.pd._get_facet_and_simplex(comp)
            with pytest.raises(RuntimeError, match="No facets found for comp"):
                self.pd._get_all_facets_and_simplexes(comp)

    def test_get_decomp_and_e_above_hull_warn_on_missing_element(self):
        with pytest.warns(UserWarning, match="Unable to get decomposition for PDEntry : U1"):
            assert self.pd.get_decomp_and_e_above_hull(PDEntry("U", 0), on_error="warn") == (None, None)

    def test_get_equilibrium_reaction_energy_unstable_raises(self):
        unstable = next(e for e in self.pd.unstable_entries if not e.is_element)
        with pytest.raises(ValueError, match="is unstable, the equilibrium reaction energy is available only"):
            self.pd.get_equilibrium_reaction_energy(unstable)

    def test_phase_separation_slsqp_fallback(self):
        entry = next(e for e in self.pd.stable_entries if not e.is_element)
        expected = self.pd.get_phase_separation_energy(entry, space_limit=10_000)

        real = phase_diagram._get_slsqp_decomp
        calls = []

        def flaky(comp, competing, tols, maxiter):
            calls.append(len(competing))
            if len(calls) == 1:
                raise ValueError("infeasible reduced system")
            return real(comp, competing, tols, maxiter)

        with (
            patch.object(phase_diagram, "_get_slsqp_decomp", side_effect=flaky),
            pytest.warns(UserWarning, match="Calculating inner hull"),
        ):
            result = self.pd.get_phase_separation_energy(entry, space_limit=1)
        assert len(calls) == 2
        # The retry uses the full (larger) set of competing entries
        assert calls[1] >= calls[0]
        assert result == approx(expected, abs=1e-5)

        # Without inner hull reduction, the ValueError propagates
        with (
            patch.object(phase_diagram, "_get_slsqp_decomp", side_effect=ValueError("boom")),
            pytest.raises(ValueError, match="boom"),
        ):
            self.pd.get_phase_separation_energy(entry, space_limit=10_000)

    def test_get_slsqp_decomp_no_valid_decomp(self):
        # Competing entries contain no Li or O, so the molar constraint cannot be satisfied.
        competing = [PDEntry("Fe", -1), PDEntry("Fe2", -3)]
        with pytest.raises(ValueError, match="No valid decomp found for"):
            phase_diagram._get_slsqp_decomp(Composition("Li2O"), competing)

    def test_chempot_methods_invalid_element(self):
        with pytest.raises(ValueError, match="get_transition_chempots can only be called with elements"):
            self.pd.get_transition_chempots(Element("U"))
        with pytest.raises(ValueError, match="get_transition_chempots can only be called with elements"):
            self.pd.get_element_profile(Element("U"), Composition("Li2O"))

    def test_get_chempot_range_map_unreferenced(self):
        elems = [Element("Li"), Element("O")]
        ref = self.pd.get_chempot_range_map(elems)
        unref = self.pd.get_chempot_range_map(elems, referenced=False)
        assert {e.name for e in ref} == {e.name for e in unref}
        offset = np.array([self.pd.el_refs[el].energy_per_atom for el in elems])
        for entry, simplices in ref.items():
            for s_ref, s_unref in zip(simplices, unref[entry], strict=True):
                assert_allclose(s_unref.coords, s_ref.coords + offset, atol=1e-8)

    def test_getmu_vertices_and_range_missing_element(self):
        # Li2O has no Fe, which exercises the zero-amount padding branch
        li2o = Composition("Li2O")
        li, o = Element("Li"), Element("O")
        vertices = self.pd.getmu_vertices_stability_phase(li2o, o)
        assert len(vertices) >= 2
        li2o_entry = next(e for e in self.pd.stable_entries if e.reduced_formula == "Li2O")
        e_li2o = li2o_entry.energy / li2o_entry.composition.get_reduced_composition_and_factor()[1]
        for vertex in vertices:
            # Absolute chemical potentials consistent with Li2O energy: 2 mu_Li + mu_O = E(Li2O)
            assert 2 * vertex[li] + vertex[o] == approx(e_li2o, abs=1e-6)
            assert vertex[li] <= self.pd.el_refs[li].energy_per_atom + 1e-8
            assert vertex[o] <= self.pd.el_refs[o].energy_per_atom + 1e-8

        mu_range = self.pd.get_chempot_range_stability_phase(li2o, o)
        o_min, o_max = mu_range[o]
        assert o_min <= o_max
        assert o_max == approx(max(v[o] for v in vertices))
        assert o_min == approx(min(v[o] for v in vertices))

        # A composition that is not stable returns None
        assert self.pd.getmu_vertices_stability_phase(Composition("FeO3"), o) is None


class TestGrandPotentialEdgeCases:
    def setup_method(self):
        self.entries = EntrySet.from_csv(f"{TEST_DIR}/pd_entries_test.csv")
        self.gppd = GrandPotentialPhaseDiagram(self.entries, {Element("O"): -5})

    def test_grand_pot_entry_repr(self):
        entry = next(e for e in self.gppd.stable_entries if e.original_entry.reduced_formula == "Li2O2")
        rep = repr(entry)
        assert rep.startswith(f"GrandPotPDEntry with original composition {entry.original_entry.composition}")
        assert f"energy = {entry.original_entry.energy:.4f}" in rep
        assert rep.endswith("chempots = mu_O = -5.0000")

    def test_as_from_dict(self):
        dct = self.gppd.as_dict()
        assert dct["@class"] == "GrandPotentialPhaseDiagram"
        assert dct["chempots"] == {Element("O"): -5}
        gppd = GrandPotentialPhaseDiagram.from_dict(dct)
        assert gppd.chempots == self.gppd.chempots
        assert gppd.elements == self.gppd.elements
        expected = {e.original_entry.reduced_formula: self.gppd.get_form_energy(e) for e in self.gppd.stable_entries}
        actual = {e.original_entry.reduced_formula: gppd.get_form_energy(e) for e in gppd.stable_entries}
        assert actual == approx(expected)


class TestCompoundPhaseDiagramEdgeCases:
    def setup_method(self):
        self.entries = EntrySet.from_csv(f"{TEST_DIR}/pd_entries_test.csv")
        self.terminals = [Composition("Li2O"), Composition("Fe2O3")]
        self.cpd = CompoundPhaseDiagram(self.entries, self.terminals)

    def test_as_dict(self):
        dct = self.cpd.as_dict()
        assert dct["@class"] == "CompoundPhaseDiagram"
        assert dct["normalize_terminal_compositions"] is True
        assert len(dct["original_entries"]) == len(self.entries)
        assert [Composition(c) for c in dct["terminal_compositions"]] == self.terminals

    def test_from_dict_round_trip(self):
        cpd = CompoundPhaseDiagram.from_dict(self.cpd.as_dict())
        assert sorted(e.name for e in cpd.stable_entries) == sorted(e.name for e in self.cpd.stable_entries)

    def test_transform_entries_skips_negative_reactants(self):
        entries = [PDEntry("Li", -2), PDEntry("O2", -8), PDEntry("Li2O2", -20), PDEntry("LiO8", -40)]
        cpd = CompoundPhaseDiagram(entries, [Composition("Li"), Composition("Li2O2")])
        # O2 and LiO8 lie outside the Li-Li2O2 segment (they need a negative amount of Li)
        assert sorted(e.original_entry.reduced_formula for e in cpd.all_entries) == ["Li", "Li2O2"]

    def test_transformed_entry_negative_reactant_raises(self):
        sp_mapping = {Composition("Li"): DummySpecies("Xf"), Composition("Li2O2"): DummySpecies("Xg")}
        with pytest.raises(TransformedPDEntryError, match="Only reactions with positive amounts"):
            TransformedPDEntry(PDEntry("LiO8", -40), sp_mapping)


class TestPatchedPhaseDiagramEdgeCases:
    def setup_method(self):
        self.entries = EntrySet.from_csv(f"{TEST_DIR}/reaction_entries_test.csv")
        self.entries.add(PDEntry("He", -1.23))
        self.ppd = PatchedPhaseDiagram(entries=self.entries)

    def test_more_terminal_elements_than_dims(self):
        entries = [PDEntry("Li", -1), PDEntry("O", -2), PDEntry("Fe", -3), PDEntry("Li2O", -10)]
        with pytest.raises(ValueError, match="more terminal elements than dimensions"):
            PatchedPhaseDiagram(entries, elements=[Element("Li"), Element("O")])

    def test_update_uncovered_space(self):
        he = Element("He")
        other = next(el for el in self.ppd.elements if el != he)
        assert not any(he in space and other in space for space in self.ppd.spaces)
        new_entry = PDEntry(Composition({he: 1, other: 1}), -1000)
        ppd_new, info = self.ppd.update([new_entry], return_info=True)
        new_space = frozenset([he, other])
        assert new_space in info["new_spaces"]
        assert new_space in ppd_new.spaces
        assert new_entry in ppd_new.stable_entries
        ppd_full = PatchedPhaseDiagram(entries=[*self.entries, new_entry])
        assert ppd_new.stable_entries == ppd_full.stable_entries


class TestPDPlotterExtra:
    def setup_method(self):
        self.entries = list(EntrySet.from_csv(f"{TEST_DIR}/pd_entries_test.csv"))
        self.pd_ternary = PhaseDiagram(self.entries)
        self.pd_binary = PhaseDiagram([e for e in self.entries if "Fe" not in e.composition])
        self.pd_unary = PhaseDiagram([e for e in self.entries if e.elements == [Element("Li")]])
        self.pd_quaternary = PhaseDiagram([*self.entries, PDEntry("C", 0)])

    def test_too_many_components(self):
        pd = PhaseDiagram([*self.entries, PDEntry("C", 0), PDEntry("N", 0)])
        with pytest.raises(ValueError, match="Only 1-4 components supported"):
            PDPlotter(pd)

    def test_get_plot_dim_guard_and_unknown_backend(self):
        plotter = PDPlotter(self.pd_binary, backend="plotly")
        plotter._dim = 5
        with pytest.raises(ValueError, match="Plotting is only supported for unary/binary/ternary/quaternary"):
            plotter.get_plot()
        assert PDPlotter(self.pd_binary, backend="bokeh").get_plot() is None

    @pytest.mark.parametrize(
        ("pd_attr", "ternary_style", "trace_type"),
        [
            ("pd_unary", "2d", go.Scatter),
            ("pd_binary", "2d", go.Scatter),
            ("pd_ternary", "2d", go.Scatterternary),
            ("pd_ternary", "3d", go.Scatter3d),
            ("pd_quaternary", "2d", go.Scatter3d),
        ],
    )
    def test_plotly_highlight_entries(self, pd_attr, ternary_style, trace_type):
        pd = getattr(self, pd_attr)
        stable = next(e for e in pd.stable_entries if e.is_element)
        plotter = PDPlotter(pd, backend="plotly", ternary_style=ternary_style)
        fig = plotter.get_plot(highlight_entries=[stable])
        assert isinstance(fig, go.Figure)
        highlighted = [trace for trace in fig.data if trace.name == "Highlighted"]
        assert len(highlighted) == 1
        assert isinstance(highlighted[0], trace_type)
        assert len(highlighted[0].hovertext) == 1
        assert highlighted[0].hovertext[0].startswith(str(stable.elements[0]))
        assert "(Stable)" in highlighted[0].hovertext[0]
        # Highlighted entries are removed from the stable trace
        stable_trace = next(trace for trace in fig.data if trace.name == "Stable")
        assert len(stable_trace.hovertext) == len(plotter.pd_plot_data[1]) - 1

    @staticmethod
    def _small_binary():
        # LiO sits 0.1 eV/atom above the Li2O-O2 tie line
        entries = [PDEntry("Li", -2), PDEntry("O2", -10), PDEntry("Li2O", -20), PDEntry("LiO", -12.3)]
        pd = PhaseDiagram(entries)
        stable = next(e for e in pd.stable_entries if e.reduced_formula == "Li2O")
        unstable = next(e for e in pd.unstable_entries if e.composition == Composition("LiO"))
        assert pd.get_e_above_hull(unstable) == approx(0.1)
        return pd, stable, unstable

    def test_plotly_highlight_unstable_entry(self):
        pd, stable, unstable = self._small_binary()
        fig = PDPlotter(pd, backend="plotly").get_plot(highlight_entries=[stable, unstable])
        highlighted = next(trace for trace in fig.data if trace.name == "Highlighted")
        assert highlighted.x == approx((1 / 3, 0.5))
        # Formation energies per atom: Li2O on the hull, LiO 0.1 eV/atom above the Li2O-O2 tie line
        assert highlighted.y == approx((-11 / 3, -2.75 + 0.1))
        # NOTE LiO is reported with the peroxide reduced formula Li2O2
        assert highlighted.hovertext[0].startswith("Li<sub>2</sub>O (")
        assert highlighted.hovertext[1].startswith("Li<sub>2</sub>O<sub>2</sub> (")
        assert "Energy Above Hull: (+0.1 eV/atom)" in highlighted.hovertext[1]
        # LiO is no longer drawn in the "Above Hull" trace
        above_hull = next(trace for trace in fig.data if trace.name == "Above Hull")
        assert len(above_hull.x) == 0

    def test_plotly_highlight_order_independent(self):
        pd, stable, unstable = self._small_binary()
        fig = PDPlotter(pd, backend="plotly").get_plot(highlight_entries=[unstable, stable])
        highlighted = next(trace for trace in fig.data if trace.name == "Highlighted")
        assert len(highlighted.hovertext) == 2
        for x_coord, text in zip(highlighted.x, highlighted.hovertext, strict=True):
            expected = "Li<sub>2</sub>O<sub>2</sub> (" if x_coord == approx(0.5) else "Li<sub>2</sub>O ("
            assert text.startswith(expected)

    def test_plotly_binary_uncertainties(self):
        adjust = ConstantEnergyAdjustment(-0.5, uncertainty=0.1)
        entries = [
            ComputedEntry("Li", -1.9),
            ComputedEntry("O2", -9.8),
            ComputedEntry("Li2O", -14.3, energy_adjustments=[adjust]),
            ComputedEntry("Li2O2", -19.0, energy_adjustments=[adjust]),
        ]
        pd = PhaseDiagram(entries)
        fig = PDPlotter(pd, backend="plotly").get_plot(label_uncertainties=True)
        names = [trace.name for trace in fig.data]
        assert "Uncertainty (window)" in names
        stable_trace = next(trace for trace in fig.data if trace.name == "Stable")
        uncertainties = dict(zip(stable_trace.x, stable_trace.error_y["array"], strict=True))
        for entry in pd.stable_entries:
            if not entry.is_element:
                assert entry.correction_uncertainty_per_atom == approx(0.1 / entry.composition.num_atoms)
        assert any(unc > 0 for unc in uncertainties.values())
        assert all("Error: +/-" in text for text in stable_trace.hovertext)
        by_formula = {
            text.split(" (")[0]: unc
            for text, unc in zip(stable_trace.hovertext, stable_trace.error_y["array"], strict=True)
        }
        assert by_formula["Li<sub>2</sub>O"] == approx(round(0.1 / 3, 4))
        window = next(trace for trace in fig.data if trace.name == "Uncertainty (window)")
        assert window.fill == "toself"
        # Outline goes out along upper bound and back along lower bound (excluding last terminal)
        assert len(window.x) == 2 * len(stable_trace.x) - 1

    @pytest.mark.parametrize("ternary_style", ["2d", "3d"])
    def test_plotly_grand_potential_ternary(self, ternary_style):
        gppd = GrandPotentialPhaseDiagram(self.pd_quaternary.all_entries, {Element("C"): -1})
        assert len(gppd.elements) == 3
        fig = PDPlotter(gppd, backend="plotly", ternary_style=ternary_style).get_plot()
        assert isinstance(fig, go.Figure)
        stable_trace = next(trace for trace in fig.data if trace.name == "Stable")
        # Hover text uses the original (non-projected) compositions
        formulas = {htmlify(e.original_entry.composition.reduced_formula) for e in gppd.stable_entries}
        assert {text.split(" (")[0] for text in stable_trace.hovertext} == formulas
        assert "O<sub>2</sub>" in formulas
        if ternary_style == "2d":
            # Axis titles use the htmlified formula of the original (unprojected) terminal entries
            titles = [fig.layout.ternary[f"{ax}axis"].title.text for ax in "abc"]
            assert sorted(titles) == ["Fe", "Li", "O<sub>2</sub>"]

    def test_plotly_grand_potential_binary_uncertainties(self):
        gppd = GrandPotentialPhaseDiagram(self.entries, {Element("O"): -5})
        fig = PDPlotter(gppd, backend="plotly").get_plot(label_uncertainties=True)
        window = next(trace for trace in fig.data if trace.name == "Uncertainty (window)")
        stable_trace = next(trace for trace in fig.data if trace.name == "Stable")
        # For transformed PDs the lower bound includes the terminal compounds
        assert len(window.x) == 2 * len(stable_trace.x)

    def test_matplotlib_ordering_attributes_and_colormap(self):
        entries = [PDEntry(e.composition, e.energy) for e in self.entries]
        for entry in entries:
            entry.attribute = "new" if entry.reduced_formula == "Li5FeO4" else "existing"
        pd = PhaseDiagram(entries)
        plotter = PDPlotter(pd, backend="matplotlib")

        ax = plotter.get_plot(ordering=["Li", "Fe", "O2"], process_attributes=True)
        assert isinstance(ax, plt.Axes)
        texts = [t.get_text() for t in ax.texts]
        # The stable "new" entry is annotated in green; unstable entries in blue
        green_labels = [t.get_text() for t in ax.texts if same_color(t.get_color(), "g")]
        assert green_labels == ["Li$_{5}$FeO$_{4}$"]
        assert "Li$_{2}$O" in texts
        # "new" entries are drawn as stars, existing ones as circles
        markers = {line.get_marker() for line in ax.get_lines()}
        assert {"*", "o"} <= markers
        # With ordering ["Li", "Fe", "O2"], Li is at the top vertex of the triangle
        li_label = next(t for t in ax.texts if t.get_text() == "Li")
        assert li_label.xy == approx((0.5, np.sqrt(3) / 2))
        plt.close("all")

    @pytest.mark.parametrize(
        ("energy_colormap", "process_attributes"),
        [("default", False), ("default", True), ("viridis", False)],
    )
    def test_matplotlib_energy_colormap(self, energy_colormap, process_attributes):
        entries = [PDEntry(e.composition, e.energy) for e in self.entries]
        for entry in entries:
            entry.attribute = "new" if entry.reduced_formula == "Li5FeO4" else "existing"
        plotter = PDPlotter(PhaseDiagram(entries), backend="matplotlib")
        ax = plotter.get_plot(energy_colormap=energy_colormap, process_attributes=process_attributes)
        assert isinstance(ax, plt.Axes)
        # colorbar adds an additional axes to the figure
        assert len(ax.figure.axes) == 2
        assert "above hull" in ax.figure.axes[1].get_ylabel()
        plt.close("all")

    def test_matplotlib_energy_colormap_unstable_colors(self):
        # Only unstable entries with e_above_hull < show_unstable are drawn. Each square must be colored by
        # its own e_above_hull, not by that of the n-th unstable entry overall.
        plotter = PDPlotter(self.pd_ternary, backend="matplotlib", show_unstable=0.05)
        ax = plotter.get_plot(energy_colormap="viridis")
        unstable = plotter.pd_plot_data[2]
        assert any(self.pd_ternary.get_e_above_hull(entry) >= 0.05 for entry in unstable)  # some entries are skipped
        # Squares are drawn in the order of the shown unstable entries (polymorphs share coordinates)
        shown = [entry for entry in unstable if self.pd_ternary.get_e_above_hull(entry) < 0.05]
        squares = [line for line in ax.get_lines() if line.get_marker() == "s"]
        assert len(squares) == len(shown) > 0
        norm = Normalize(vmin=-0.06, vmax=0.06)
        for line, entry in zip(squares, shown, strict=True):
            assert_allclose(line.get_xydata()[0], unstable[entry])
            expected = plt.get_cmap("viridis")(norm(self.pd_ternary.get_e_above_hull(entry)))
            assert_allclose(line.get_markerfacecolor(), expected)
        plt.close("all")

    def test_write_image(self):
        plotter = PDPlotter(self.pd_binary, backend="matplotlib")
        stream = StringIO()
        plotter.write_image(stream, image_format="svg")
        assert stream.getvalue().lstrip().startswith("<?xml")
        assert "<svg" in stream.getvalue()
        plt.close("all")

        plotter = PDPlotter(self.pd_binary, backend="plotly")
        with patch.object(go.Figure, "write_image") as mock_write:
            plotter.write_image("pd.png", image_format="png")
        mock_write.assert_called_once_with("pd.png", format="png")

    def test_plot_element_profile_labels(self):
        plotter = PDPlotter(self.pd_ternary, backend="matplotlib")
        evolution = self.pd_ternary.get_element_profile(Element("O"), Composition("Li2O"))
        ax = plotter.plot_element_profile(Element("O"), Composition("Li2O"), show_label_index=[0, 1])
        assert isinstance(ax, plt.Axes)
        assert ax.get_ylabel() == "Uptake per atom"
        assert ax.get_xlim() == (0, 5)
        # Labelled steps are drawn in red, with one annotation each
        assert len(evolution) == 3
        red_lines = [line for line in ax.get_lines() if same_color(line.get_color(), "r")]
        assert len(red_lines) == 2
        assert len(ax.texts) == 2
        assert all(same_color(text.get_color(), "r") for text in ax.texts)
        plt.close("all")

    def test_chempot_range_map_plot(self):
        plotter = PDPlotter(self.pd_ternary, backend="matplotlib")
        for elements in ([Element("Li"), Element("O")], [Element("O"), Element("Li")]):
            ax = plotter.get_chempot_range_map_plot(elements)
            assert isinstance(ax, plt.Axes)
            labels = {t.get_text() for t in ax.texts}
            assert "Li$_{2}$O" in labels
            plt.close("all")
        with patch.object(plt.Axes, "show", create=True) as mock_show:
            assert plotter.plot_chempot_range_map([Element("Li"), Element("O")]) is None
        mock_show.assert_called_once()
        plt.close("all")
