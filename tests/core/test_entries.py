from __future__ import annotations

import json
import warnings
from collections import defaultdict

import numpy as np
import pytest
from monty.json import MontyDecoder
from pytest import approx

from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.entries import (
    CompositionEnergyAdjustment,
    ComputedEntry,
    ComputedStructureEntry,
    ConstantEnergyAdjustment,
    EnergyAdjustment,
    Entry,
    EntrySet,
    ManualEnergyAdjustment,
    TemperatureEnergyAdjustment,
    group_entries_by_composition,
    group_entries_by_structure,
)
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.util.testing import TEST_FILES_DIR, VASP_OUT_DIR

vasp_run = Vasprun(f"{VASP_OUT_DIR}/vasprun.xml.gz")


def test_energy_adjustment():
    ea = EnergyAdjustment(10)
    assert ea.name == "Manual adjustment"
    assert not ea.cls
    ea_dct = ea.as_dict()
    ea2 = EnergyAdjustment.from_dict(ea_dct)
    assert str(ea_dct) == str(ea2.as_dict())


# def test_energy_adjustment_repr():
#     comp_cls = MaterialsProject2020Compatibility()
#     cls_name = type(comp_cls).__name__
#     for cls, label in (
#         (None, "unknown"),
#         (comp_cls, cls_name),
#         ({"@class": cls_name}, cls_name),
#     ):
#         ea = EnergyAdjustment(10, cls=cls)
#         assert (
#             repr(ea) == "EnergyAdjustment(name='Manual adjustment', value=10.0, uncertainty=nan, description='', "
#             f"generated_by='{label}')"
#         )

#     # Make sure int uncertainty also works
#     assert "uncertainty=0.0" in repr(EnergyAdjustment(10, uncertainty=0))


def test_manual_energy_adjustment():
    ea = ManualEnergyAdjustment(10)
    assert ea.name == "Manual energy adjustment"
    assert ea.value == 10
    assert ea.explain == "Manual energy adjustment (10.000 eV)"
    ea_dct = ea.as_dict()
    ea2 = ManualEnergyAdjustment.from_dict(ea_dct)
    assert str(ea_dct) == str(ea2.as_dict())


def test_constant_energy_adjustment():
    ea = ConstantEnergyAdjustment(8)
    assert ea.name == "Constant energy adjustment"
    assert ea.value == 8
    assert ea.explain == "Constant energy adjustment (8.000 eV)"
    ea_dct = ea.as_dict()
    ea2 = ConstantEnergyAdjustment.from_dict(ea_dct)
    assert str(ea_dct) == str(ea2.as_dict())

    ea3 = ConstantEnergyAdjustment(8, uncertainty=0)
    assert "uncertainty=0.0" in repr(ea3)


def test_composition_energy_adjustment():
    ea = CompositionEnergyAdjustment(2, 2, uncertainty_per_atom=0, name="H")
    assert ea.name == "H"
    assert ea.value == 4
    assert ea.explain == "Composition-based energy adjustment (2.000 eV/atom x 2 atoms)"
    assert "uncertainty=0.0" in repr(ea)
    ea_dct = ea.as_dict()
    ea2 = CompositionEnergyAdjustment.from_dict(ea_dct)
    assert str(ea_dct) == str(ea2.as_dict())


def test_temp_energy_adjustment():
    ea = TemperatureEnergyAdjustment(-0.1, 298, 5, uncertainty_per_deg=0, name="entropy")
    assert ea.name == "entropy"
    assert ea.value == approx(-0.1 * 298 * 5)
    assert ea.n_atoms == 5
    assert ea.temp == 298
    assert ea.explain == "Temperature-based energy adjustment (-0.1000 eV/K/atom x 298 K x 5 atoms)"
    assert "uncertainty=0.0" in repr(ea)
    ea_dct = ea.as_dict()
    ea2 = TemperatureEnergyAdjustment.from_dict(ea_dct)
    assert str(ea_dct) == str(ea2.as_dict())


class TestComputedEntry:
    def setup_method(self):
        self.entry = ComputedEntry(
            vasp_run.final_structure.composition,
            vasp_run.final_energy,
            parameters=vasp_run.incar,
        )
        self.entry2 = ComputedEntry({"Fe": 2, "O": 3}, 2.3)
        self.entry3 = ComputedEntry("Fe2O3", 2.3)
        self.entry4 = ComputedEntry("Fe2O3", 2.3, entry_id=1)
        self.entry5 = ComputedEntry("Fe6O9", 6.9)
        ea = ConstantEnergyAdjustment(-5, name="Dummy adjustment")
        self.entry6 = ComputedEntry("Fe6O9", 6.9, correction=-10)
        self.entry7 = ComputedEntry("Fe6O9", 6.9, energy_adjustments=[ea])

    def test_energy(self):
        assert self.entry.energy == approx(-269.38319884)
        self.entry.correction = 1.0
        assert self.entry.energy == approx(-268.38319884)
        assert self.entry3.energy_per_atom == approx(2.3 / 5)

    def test_composition(self):
        assert self.entry.reduced_formula == "LiFe4(PO4)4"
        assert self.entry2.reduced_formula == "Fe2O3"
        assert self.entry5.reduced_formula == "Fe2O3"
        assert self.entry5.composition.get_reduced_formula_and_factor()[1] == 3

    def test_per_atom_props(self):
        entry = ComputedEntry("Fe6O9", 6.9)
        entry.energy_adjustments.append(CompositionEnergyAdjustment(-0.5, 9, uncertainty_per_atom=0.1, name="O"))
        assert entry.energy == approx(2.4)
        assert entry.energy_per_atom == approx(2.4 / 15)
        assert entry.uncorrected_energy == approx(6.9)
        assert entry.uncorrected_energy_per_atom == approx(6.9 / 15)
        assert entry.correction == approx(-4.5)
        assert entry.correction_per_atom == approx(-4.5 / 15)
        assert entry.correction_uncertainty == approx(0.9)
        assert entry.correction_uncertainty_per_atom == approx(0.9 / 15)

    def test_zero_uncertainty_adjustment_doesnt_warn(self):
        """Adjustments with uncertainty=0 (e.g. the MP GGA(+U)/r2SCAN mixing adjustment attached to entries
        from the Materials Project API) must not trigger "Using UFloat objects with std_dev==0" warnings.
        """
        entry = ComputedEntry("Fe6O9", 6.9, energy_adjustments=[ConstantEnergyAdjustment(-5, uncertainty=0)])
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Using UFloat objects")
            assert entry.correction == approx(-5)
            assert entry.energy == approx(1.9)
            assert np.isnan(entry.correction_uncertainty)  # zero uncertainty still means "unknown"

    def test_normalize(self):
        entry = ComputedEntry("Fe6O9", 6.9, correction=1)
        entry_formula = entry.normalize()
        assert entry_formula.formula == "Fe2 O3"
        assert entry_formula.uncorrected_energy == approx(6.9 / 3)
        assert entry_formula.correction == approx(1 / 3)
        assert entry_formula.energy * 3 == approx(6.9 + 1)
        assert entry_formula.energy_adjustments[0].value == approx(1 / 3)
        entry_atom = entry.normalize("atom")
        assert entry_atom.formula == "Fe0.4 O0.6"
        assert entry_atom.uncorrected_energy == approx(6.9 / 15)
        assert entry_atom.correction == approx(1 / 15)
        assert entry_atom.energy * 15 == approx(6.9 + 1)
        assert entry_atom.energy_adjustments[0].value == approx(1 / 15)

    def test_normalize_energy_adjustments(self):
        ene_adj_list = [
            ManualEnergyAdjustment(5),
            ConstantEnergyAdjustment(5),
            CompositionEnergyAdjustment(1, 5, uncertainty_per_atom=0, name="Na"),
            TemperatureEnergyAdjustment(0.005, 100, 10, uncertainty_per_deg=0),
        ]
        entry = ComputedEntry("Na5Cl5", 6.9, energy_adjustments=ene_adj_list)
        assert entry.correction == 20
        normed_entry = entry.normalize()
        assert normed_entry.correction == 4
        for ea in normed_entry.energy_adjustments:
            assert ea.value == 1

    def test_as_from_dict(self):
        dct = self.entry.as_dict()
        entry = ComputedEntry.from_dict(dct)
        assert self.entry == entry
        assert entry.energy == approx(-269.38319884)

    def test_as_from_dict_with_adjustment(self):
        """Legacy case where adjustment was provided manually."""
        dct = self.entry6.as_dict()
        entry = ComputedEntry.from_dict(dct)
        assert entry.uncorrected_energy == approx(6.9)
        assert entry.energy_adjustments[0].value == self.entry6.energy_adjustments[0].value

    def test_as_from_dict_with_adjustment_2(self):
        """Modern case where correction was provided manually."""
        dct = self.entry7.as_dict()
        entry = ComputedEntry.from_dict(dct)
        assert entry.uncorrected_energy == approx(6.9)
        assert entry.energy_adjustments[0].value == self.entry7.energy_adjustments[0].value

    def test_as_from_dict_with_adjustment_3(self):
        """
        Legacy case where the entry was serialized before the energy_adjustment
        attribute was part of ComputedEntry.
        """
        # same as entry6
        dct = {
            "@module": "pymatgen.entries.computed_entries",
            "@class": "ComputedEntry",
            "energy": 6.9,
            "composition": defaultdict(float, {"Fe": 6.0, "O": 9.0}),
            "parameters": {},
            "data": {},
            "entry_id": None,
            "correction": -10,
        }
        entry = ComputedEntry.from_dict(dct)
        assert entry.uncorrected_energy == approx(6.9)
        assert entry.correction == approx(-10)
        assert len(entry.energy_adjustments) == 1

    def test_conflicting_correction_adjustment(self):
        """Should raise a ValueError if a user tries to manually set both the correction
        and energy_adjustment, even if the values match.
        """
        ea = ConstantEnergyAdjustment(-10, name="Dummy adjustment")
        with pytest.raises(ValueError, match="Argument conflict!"):
            ComputedEntry("Fe6O9", 6.9, correction=-10, energy_adjustments=[ea])

    def test_entry_id(self):
        assert self.entry4.entry_id == 1
        assert self.entry2.entry_id is None

    def test_str(self):
        assert str(self.entry).startswith(
            "None ComputedEntry - Li1 Fe4 P4 O16 (LiFe4(PO4)4)\n"
            "Energy (Uncorrected)     = -269.3832 eV (-10.7753 eV/atom)\n"
            "Correction               = 0.0000    eV (0.0000   eV/atom)\n"
            "Energy (Final)           = -269.3832 eV (-10.7753 eV/atom)\n"
            "Energy Adjustments:"
        )

    def test_sulfide_energy(self):
        self.entry = ComputedEntry("BaS", -10.21249155)
        assert self.entry.energy == approx(-10.21249155)
        assert self.entry.energy_per_atom == approx(-10.21249155 / 2)
        self.entry.correction = 1.0
        assert self.entry.energy == approx(-9.21249155)

    def test_is_element(self):
        entry = ComputedEntry("Fe3", 2.3)
        assert entry.is_element

    def test_copy(self):
        for entry in (self.entry, self.entry2, self.entry3, self.entry4):
            copy = entry.copy()
            assert entry == copy
            assert str(entry) == str(copy)

    def test_from_dict_null_fields(self):
        ce_dict = self.entry.as_dict()
        for k in (
            "energy_adjustments",
            "parameters",
            "data",
        ):
            ce = ce_dict.copy()
            ce[k] = None
            new_ce = ComputedEntry.from_dict(ce)
            assert new_ce == self.entry
            assert getattr(new_ce, k, None) is not None


class TestComputedStructureEntry:
    def setup_method(self):
        self.entry = ComputedStructureEntry(vasp_run.final_structure, vasp_run.final_energy, parameters=vasp_run.incar)

    def test_energy(self):
        assert self.entry.energy == approx(-269.38319884)
        self.entry.correction = 1.0
        assert self.entry.energy == approx(-268.38319884)

    def test_composition(self):
        assert self.entry.reduced_formula == "LiFe4(PO4)4"

    def test_as_from_dict(self):
        dct = self.entry.as_dict()
        entry = ComputedStructureEntry.from_dict(dct)
        assert self.entry == entry
        assert entry.energy == approx(-269.38319884)

    def test_str(self):
        assert str(self.entry).startswith(
            "None ComputedStructureEntry - Li1 Fe4 P4 O16 (LiFe4(PO4)4)\n"
            "Energy (Uncorrected)     = -269.3832 eV (-10.7753 eV/atom)\n"
            "Correction               = 0.0000    eV (0.0000   eV/atom)\nEnergy (Final)"
        )

    def test_as_from_dict_structure_with_adjustment_3(self):
        """
        Legacy case where the structure entry was serialized before the energy_adjustment
        attribute was part of ComputedEntry.
        """
        # ComputedStructureEntry for Oxygen, mp-12957, as of April 2020
        # with an arbitrary 1 eV correction added
        dct = {
            "@module": "pymatgen.entries.computed_entries",
            "@class": "ComputedStructureEntry",
            "energy": -39.42116819,
            "composition": defaultdict(float, {"O": 8.0}),
            "parameters": {
                "run_type": "GGA",
                "is_hubbard": False,
                "pseudo_potential": {
                    "functional": "PBE",
                    "labels": ["O"],
                    "pot_type": "paw",
                },
                "hubbards": {},
                "potcar_symbols": ["PBE O"],
                "oxide_type": "None",
            },
            "data": {"oxide_type": "None"},
            "entry_id": "mp-12957",
            "correction": 1,
            "structure": {
                "@module": "pymatgen.core.structure",
                "@class": "Structure",
                "charge": None,
                "lattice": {
                    "matrix": [
                        [-1.7795583, 0.0, 3.86158265],
                        [4.17564656, -3.03266995, -0.01184798],
                        [4.17564656, 3.03266995, -0.01184798],
                    ],
                    "a": 4.251899376264673,
                    "b": 5.160741380296335,
                    "c": 5.160741380296335,
                    "alpha": 71.97975354157973,
                    "beta": 109.9211782454931,
                    "gamma": 109.9211782454931,
                    "volume": 97.67332322031668,
                },
                "sites": [
                    {
                        "species": [{"element": "O", "occu": 1}],
                        "abc": [0.8531272, 0.15466029, 0.15466029],
                        "xyz": [
                            -0.22657617390155504,
                            -1.750215367360042e-17,
                            3.2907563697176516,
                        ],
                        "label": "O",
                        "properties": {"magmom": 0.002},
                    },
                    {
                        "species": [{"element": "O", "occu": 1}],
                        "abc": [0.84038763, 0.71790132, 0.21754949],
                        "xyz": [
                            2.410593174641884,
                            -1.5174019592685084,
                            3.234143088794756,
                        ],
                        "label": "O",
                        "properties": {"magmom": -0.002},
                    },
                    {
                        "species": [{"element": "O", "occu": 1}],
                        "abc": [0.17255465, 0.21942628, 0.21942628],
                        "xyz": [
                            1.5254221229000986,
                            -2.121360826524921e-18,
                            0.6611345262629937,
                        ],
                        "label": "O",
                        "properties": {"magmom": 0.002},
                    },
                    {
                        "species": [{"element": "O", "occu": 1}],
                        "abc": [0.15961237, 0.78245051, 0.28209968],
                        "xyz": [
                            4.161145821004675,
                            -1.5173989265985586,
                            0.6037435893572642,
                        ],
                        "label": "O",
                        "properties": {"magmom": -0.002},
                    },
                    {
                        "species": [{"element": "O", "occu": 1}],
                        "abc": [0.84038763, 0.21754949, 0.71790132],
                        "xyz": [
                            2.410593174641884,
                            1.5174019592685082,
                            3.234143088794756,
                        ],
                        "label": "O",
                        "properties": {"magmom": -0.002},
                    },
                    {
                        "species": [{"element": "O", "occu": 1}],
                        "abc": [0.82744535, 0.78057372, 0.78057372],
                        "xyz": [
                            5.046312697099901,
                            -1.3574974398403584e-16,
                            3.176752163737006,
                        ],
                        "label": "O",
                        "properties": {"magmom": 0.002},
                    },
                    {
                        "species": [{"element": "O", "occu": 1}],
                        "abc": [0.15961237, 0.28209968, 0.78245051],
                        "xyz": [
                            4.161145821004675,
                            1.5173989265985584,
                            0.6037435893572642,
                        ],
                        "label": "O",
                        "properties": {"magmom": -0.002},
                    },
                    {
                        "species": [{"element": "O", "occu": 1}],
                        "abc": [0.1468728, 0.84533971, 0.84533971],
                        "xyz": [
                            6.798310993901555,
                            -1.7769364890338579e-16,
                            0.5471303202823484,
                        ],
                        "label": "O",
                        "properties": {"magmom": 0.002},
                    },
                ],
            },
        }
        entry = ComputedEntry.from_dict(dct)
        assert entry.uncorrected_energy == approx(-39.42116819)
        assert entry.energy == approx(-38.42116819)
        assert entry.correction == approx(1)
        assert len(entry.energy_adjustments) == 1

    def test_copy(self):
        copy = self.entry.copy()
        assert copy == self.entry
        assert copy is not self.entry
        assert copy.structure is not self.entry.structure
        assert str(copy) == str(self.entry)

    def test_eq(self):
        copy1 = self.entry.copy()
        copy2 = self.entry.copy()
        copy3 = self.entry.copy()

        # if no entry ids but same energy/comps --> equal
        assert copy1 == copy2

        # check different entry_ids --> not equal
        copy1.entry_id = "mp_1"
        copy2.entry_id = "mp_2"
        assert copy1 != copy2

        # same entry_id and energy --> equal
        copy3.entry_id = "mp_1"
        assert copy3 == copy1

        # check different energy adjustments (but same id) --> not equal
        copy3.energy_adjustments.append(ConstantEnergyAdjustment(0.1))
        assert copy3 != copy1


class TestEntry:
    """Tests for the generic Entry base class, exercised via the PDEntry subclass."""

    def test_properties(self):
        entry = PDEntry("Fe4O6", -60.0)
        assert entry.formula == "Fe4 O6"
        assert entry.reduced_formula == "Fe2O3"
        assert entry.elements == [Element.Fe, Element.O]
        assert entry.energy_per_atom == approx(-6.0)
        assert not entry.is_element
        assert PDEntry("O2", -9.0).is_element
        assert entry.composition == Composition("Fe4O6")

    def test_base_repr_and_abstract_energy(self):
        class MinimalEntry(Entry):
            @property
            def energy(self) -> float:
                return self._energy

        entry = MinimalEntry("Li2O", -14.0)
        assert repr(entry) == "MinimalEntry : Li2 O1 with energy = -14.0000"
        with pytest.raises(NotImplementedError):
            Entry.energy.fget(entry)
        with pytest.raises(TypeError, match="abstract"):
            Entry("Li2O", -14.0)

    def test_init_with_composition(self):
        comp = Composition("Li2O")
        entry = PDEntry(comp, -14.0)
        assert entry.composition is comp

    @pytest.mark.parametrize(
        ("mode", "formula", "energy"),
        [("formula_unit", "Fe2 O3", -30.0), ("atom", "Fe0.4 O0.6", -6.0)],
    )
    def test_normalize(self, mode, formula, energy):
        entry = PDEntry("Fe4O6", -60.0)
        normed = entry.normalize(mode)
        assert isinstance(normed, PDEntry)
        assert normed.formula == formula
        assert normed.energy == approx(energy)
        assert normed.energy_per_atom == approx(entry.energy_per_atom)
        # original entry untouched
        assert entry.energy == approx(-60.0)

    @pytest.mark.parametrize("entry_cls", [PDEntry, ComputedEntry])
    def test_normalize_invalid_mode(self, entry_cls):
        with pytest.raises(ValueError, match="bad_mode is not an allowed option for normalization"):
            entry_cls("Fe4O6", -60.0).normalize("bad_mode")

    def test_eq_and_hash(self):
        entry = PDEntry("Fe2O3", -30.0)
        assert entry == entry  # noqa: PLR0124
        assert entry == PDEntry("Fe2O3", -30.0)
        assert hash(entry) == hash(PDEntry("Fe2O3", -30.0))
        assert entry != PDEntry("Fe2O3", -30.1)
        assert entry != PDEntry("Fe4O6", -30.0)
        # comparison with a foreign type defers to NotImplemented -> unequal
        assert entry.__eq__("Fe2O3") is NotImplemented
        assert entry != "Fe2O3"
        assert len({entry, PDEntry("Fe2O3", -30.0), PDEntry("Li2O", -14.0)}) == 2


class TestComputedEntryExtra:
    def test_repr_with_adjustments_parameters_and_data(self):
        entry = ComputedEntry(
            "Fe2O3",
            -30.0,
            energy_adjustments=[ConstantEnergyAdjustment(-1.0, name="Dummy")],
            parameters={"run_type": "GGA"},
            data={"oxide_type": "oxide"},
            entry_id="mp-1",
        )
        text = repr(entry)
        assert text.startswith("mp-1 ComputedEntry - Fe2 O3")
        assert "  Dummy                  : -1.0000   eV (-0.2000  eV/atom)" in text
        assert "  run_type               = GGA" in text
        assert "  oxide_type             = oxide" in text
        assert "Energy (Final)           = -31.0000" in text

    def test_eq(self):
        entry = ComputedEntry("Fe2O3", -30.0, entry_id="mp-1")
        assert entry == entry  # noqa: PLR0124
        # objects lacking composition/energy/entry_id -> NotImplemented -> unequal
        assert entry.__eq__(1) is NotImplemented
        assert entry != 1
        # same id, different energy
        assert entry != ComputedEntry("Fe2O3", -31.0, entry_id="mp-1")
        # same energy, different composition, no ids
        assert ComputedEntry("Fe2O3", -30.0) != ComputedEntry("FeO", -30.0)

    def test_hash(self):
        with_id = ComputedEntry("Fe2O3", -30.0, entry_id="mp-1")
        assert hash(with_id) == hash("ComputedEntrymp-1")
        # hash depends only on entry_id when it is set
        assert hash(with_id) == hash(ComputedEntry("FeO", -5.0, entry_id="mp-1"))
        no_id = ComputedEntry("Fe2O3", -30.0)
        assert hash(no_id) == hash(f"ComputedEntry{no_id.composition.formula}{-30.0:.8f}")


@pytest.fixture
def fe_bcc() -> Structure:
    return Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])


class TestComputedStructureEntryExtra:
    @pytest.mark.parametrize("composition", ["Fe4", Composition("Fe")])
    def test_composition_argument(self, fe_bcc, composition):
        entry = ComputedStructureEntry(fe_bcc, -16.0, composition=composition)
        assert entry.composition == Composition(composition)
        assert entry.structure is fe_bcc

    def test_mismatching_composition(self, fe_bcc):
        with pytest.raises(ValueError, match="Mismatching composition provided"):
            ComputedStructureEntry(fe_bcc, -16.0, composition="FeO")

    def test_from_dict_legacy_correction(self, fe_bcc):
        dct = ComputedStructureEntry(
            fe_bcc, -16.0, parameters={"run_type": "GGA"}, data={"a": 1}, entry_id="mp-13"
        ).as_dict()
        dct["correction"] = -1.5
        del dct["energy_adjustments"]
        entry = ComputedStructureEntry.from_dict(dct)
        assert isinstance(entry, ComputedStructureEntry)
        assert entry.uncorrected_energy == approx(-16.0)
        assert entry.correction == approx(-1.5)
        assert entry.energy == approx(-17.5)
        assert len(entry.energy_adjustments) == 1
        assert isinstance(entry.energy_adjustments[0], ManualEnergyAdjustment)
        assert entry.parameters == {"run_type": "GGA"}
        assert entry.data == {"a": 1}
        assert entry.entry_id == "mp-13"
        assert entry.structure == fe_bcc

    @pytest.mark.parametrize("mode", ["formula_unit", "atom"])
    def test_normalize(self, fe_bcc, mode):
        entry = ComputedStructureEntry(fe_bcc, -16.0, correction=-2.0)
        with pytest.warns(UserWarning, match="inconsistent"):
            normed = entry.normalize(mode)
        assert isinstance(normed, ComputedStructureEntry)
        assert normed.uncorrected_energy == approx(-8.0)
        assert normed.correction == approx(-1.0)
        assert normed.energy == approx(-9.0)
        # structure is left unchanged
        assert normed.structure == fe_bcc
        assert len(normed.structure) == 2

    @pytest.mark.parametrize("mode", ["formula_unit", "atom"])
    def test_normalize_composition(self, fe_bcc, mode):
        entry = ComputedStructureEntry(fe_bcc, -16.0)
        with pytest.warns(UserWarning, match="inconsistent"):
            normed = entry.normalize(mode)
        assert normed.composition == Composition("Fe")
        assert normed.energy_per_atom == approx(entry.energy_per_atom)


@pytest.fixture(scope="module")
def tio2_entries() -> list[ComputedStructureEntry]:
    with open(f"{TEST_FILES_DIR}/entries/TiO2_entries.json", encoding="utf-8") as file:
        return json.load(file, cls=MontyDecoder)


def _group_ids(groups) -> list[list]:
    return sorted(sorted(entry.entry_id for entry in group) for group in groups)


TIO2_GROUPS = [
    [390, 34688],
    [430],
    [1439],
    [1840],
    [2657],
    [9173],
    [25262, 25263, 25334, 25434],
    [25433],
    [554278, 554439],
    [562577],
    [572822],
]


class TestGroupEntries:
    def test_group_entries_by_structure(self, tio2_entries):
        groups = group_entries_by_structure(tio2_entries)
        assert _group_ids(groups) == TIO2_GROUPS
        assert all(isinstance(entry, ComputedStructureEntry) for group in groups for entry in group)

    @pytest.mark.xfail(
        strict=True,
        raises=AttributeError,
        reason="ncpus path calls comparator.get_structure_hash, which no AbstractComparator subclass defines",
    )
    def test_group_entries_by_structure_parallel(self, tio2_entries):
        groups = group_entries_by_structure(tio2_entries, ncpus=2)
        assert _group_ids(groups) == TIO2_GROUPS

    def test_group_entries_by_structure_species_to_remove(self, tio2_entries):
        # Comparing only the Ti sublattice merges 9173 and 572822
        groups = group_entries_by_structure(tio2_entries, species_to_remove=["O"])
        expected = sorted([g for g in TIO2_GROUPS if g not in ([9173], [572822])] + [[9173, 572822]])
        assert _group_ids(groups) == expected
        # input structures must not be modified
        assert all(entry.structure.reduced_formula == "TiO2" for entry in tio2_entries)

    @pytest.mark.parametrize(
        ("sort_by_e_per_atom", "expected"),
        [(True, [["c", "a", "d"], ["b"]]), (False, [["a", "c", "d"], ["b"]])],
    )
    def test_group_entries_by_composition(self, sort_by_e_per_atom, expected):
        entries = [
            ComputedEntry("Fe2O3", -30.0, entry_id="a"),
            ComputedEntry("Li2O", -14.0, entry_id="b"),
            ComputedEntry("Fe4O6", -61.0, entry_id="c"),
            ComputedEntry("Fe2O3", -29.0, entry_id="d"),
        ]
        groups = group_entries_by_composition(entries, sort_by_e_per_atom=sort_by_e_per_atom)
        assert [group[0].reduced_formula for group in groups] == ["Fe2O3", "Li2O"]
        assert [[entry.entry_id for entry in group] for group in groups] == expected


class TestEntrySet:
    def setup_method(self):
        self.entries = [
            ComputedEntry("Li", -1.9, entry_id="Li"),
            ComputedEntry("O2", -9.8, entry_id="O"),
            ComputedEntry("Li2O", -14.3, entry_id="Li2O"),
            ComputedEntry("Li4O2", -28.0, entry_id="Li2O-hi"),
            ComputedEntry("Fe", -8.3, entry_id="Fe"),
            ComputedEntry("Fe2O3", -38.0, entry_id="Fe2O3"),
        ]
        self.entry_set = EntrySet(self.entries)

    def test_container(self):
        entry_set = self.entry_set
        assert len(entry_set) == 6
        assert self.entries[0] in entry_set
        assert {entry.entry_id for entry in entry_set} == {entry.entry_id for entry in self.entries}
        new = ComputedEntry("FeO", -16.0, entry_id="FeO")
        entry_set.add(new)
        assert len(entry_set) == 7
        assert new in entry_set
        entry_set.discard(new)
        assert len(entry_set) == 6
        assert new not in entry_set
        entry_set.discard(new)  # discarding a missing entry is a no-op
        assert len(entry_set) == 6

    def test_chemsys(self):
        assert self.entry_set.chemsys == {"Li", "O", "Fe"}

    def test_ground_states(self):
        assert {entry.entry_id for entry in self.entry_set.ground_states} == {"Li", "O", "Li2O", "Fe", "Fe2O3"}
        assert self.entry_set.is_ground_state(self.entries[2])
        assert not self.entry_set.is_ground_state(self.entries[3])
        self.entry_set.remove_non_ground_states()
        assert len(self.entry_set) == 5
        assert self.entries[3] not in self.entry_set

    def test_get_subset_in_chemsys(self):
        subset = self.entry_set.get_subset_in_chemsys(["Li", "O"])
        assert isinstance(subset, EntrySet)
        assert {entry.entry_id for entry in subset} == {"Li", "O", "Li2O", "Li2O-hi"}
        assert subset.chemsys == {"Li", "O"}
        with pytest.raises(ValueError, match=r"\['Li', 'Na'\] is not a subset of \['Fe', 'Li', 'O'\]"):
            self.entry_set.get_subset_in_chemsys(["Li", "Na"])

    def test_as_dict(self):
        dct = self.entry_set.as_dict()
        assert set(dct) == {"entries"}
        assert set(dct["entries"]) == set(self.entries)
        decoded = json.loads(self.entry_set.to_json(), cls=MontyDecoder)
        assert isinstance(decoded, EntrySet)
        assert {entry.entry_id for entry in decoded} == {entry.entry_id for entry in self.entries}

    @pytest.mark.parametrize(("latexify_names", "li2o_name"), [(False, "Li2O"), (True, "Li_{2}O")])
    def test_to_from_csv(self, latexify_names, li2o_name):
        self.entry_set.to_csv("entries.csv", latexify_names=latexify_names)
        with open("entries.csv", encoding="utf-8") as file:
            lines = file.read().splitlines()
        # element columns are sorted by electronegativity
        assert lines[0] == "Name,Li,Fe,O,Energy"
        assert len(lines) == 7
        assert li2o_name in {line.split(",")[0] for line in lines[1:]}

        loaded = EntrySet.from_csv("entries.csv")
        assert len(loaded) == len(self.entry_set)
        assert all(isinstance(entry, PDEntry) for entry in loaded)
        loaded_by_energy = {round(entry.energy, 6): entry for entry in loaded}
        for entry in self.entries:
            assert loaded_by_energy[round(entry.energy, 6)].composition == entry.composition
        assert li2o_name in {entry.name for entry in loaded}
