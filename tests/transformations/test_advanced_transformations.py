from __future__ import annotations

from shutil import which

import numpy as np
import orjson
import pytest
from monty.serialization import loadfn
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx

import pymatgen.transformations.advanced_transformations as advanced
from pymatgen.core import Lattice, Molecule, Species, Structure
from pymatgen.core.energy_models import IsingModel, SymmetryModel
from pymatgen.core.interface import GrainBoundaryGenerator
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.icet import ClusterSpace
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import (
    AddAdsorbateTransformation,
    ChargeBalanceTransformation,
    CubicSupercellTransformation,
    DisorderOrderedTransformation,
    DopingTransformation,
    EnumerateStructureTransformation,
    GrainBoundaryTransformation,
    MagOrderingTransformation,
    MagOrderParameterConstraint,
    MonteCarloRattleTransformation,
    MultipleSubstitutionTransformation,
    SlabTransformation,
    SQSTransformation,
    SubstituteSurfaceSiteTransformation,
    SubstitutionPredictorTransformation,
    SuperTransformation,
    find_codopant,
)
from pymatgen.transformations.standard_transformations import (
    AutoOxiStateDecorationTransformation,
    OrderDisorderedStructureTransformation,
    OxidationStateDecorationTransformation,
    SubstitutionTransformation,
)
from pymatgen.util.testing import TEST_FILES_DIR, VASP_IN_DIR, MatSciTest

try:
    import hiphive
except ImportError:
    hiphive = None


def get_table():
    """Loads a lightweight lambda table for use in unit tests to reduce
    initialization time, and make unit tests insensitive to changes in the
    default lambda table.
    """
    json_path = f"{TEST_FILES_DIR}/struct_predictor/test_lambda.json"
    with open(json_path, "rb") as file:
        return orjson.loads(file.read())


enum_cmd = which("enum.x") or which("multienum.x")
makestr_cmd = which("makestr.x") or which("makeStr.x") or which("makeStr.py")
mcsqs_cmd = which("mcsqs")
enumlib_present = enum_cmd and makestr_cmd


class TestSuperTransformation:
    def test_apply_transformation(self):
        trafo = SuperTransformation(
            [
                SubstitutionTransformation({"Li+": "Na+"}),
                SubstitutionTransformation({"Li+": "K+"}),
            ]
        )
        coords = [
            [0, 0, 0],
            [0.375, 0.375, 0.375],
            [0.5, 0.5, 0.5],
            [0.875, 0.875, 0.875],
            [0.125, 0.125, 0.125],
            [0.25, 0.25, 0.25],
            [0.625, 0.625, 0.625],
            [0.75, 0.75, 0.75],
        ]

        lattice = [
            [3.8401979337, 0.00, 0.00],
            [1.9200989668, 3.3257101909, 0.00],
            [0.00, -2.2171384943, 3.1355090603],
        ]
        struct = Structure(lattice, ["Li+", "Li+", "Li+", "Li+", "Li+", "Li+", "O2-", "O2-"], coords)
        struct_trafo = trafo.apply_transformation(struct, return_ranked_list=True)

        for s_and_t in struct_trafo:
            assert s_and_t["transformation"].apply_transformation(struct) == s_and_t["structure"]

    @pytest.mark.skipif(not enumlib_present, reason="enum_lib not present.")
    def test_apply_transformation_mult(self):
        # Test returning multiple structures from each transformation.
        disordered = Structure(
            np.eye(3) * 4.209,
            [{"Cs+": 0.5, "K+": 0.5}, "Cl-"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        disordered.make_supercell([2, 2, 1])

        tl = [
            EnumerateStructureTransformation(),
            OrderDisorderedStructureTransformation(),
        ]
        trafo = SuperTransformation(tl, nstructures_per_trans=10)
        assert len(trafo.apply_transformation(disordered, return_ranked_list=20)) == 8
        trafo = SuperTransformation(tl)
        assert len(trafo.apply_transformation(disordered, return_ranked_list=20)) == 2


class TestMultipleSubstitutionTransformation:
    def test_apply_transformation(self):
        sub_dict = {1: ["Na", "K"]}
        trafo = MultipleSubstitutionTransformation("Li+", 0.5, sub_dict, None)
        coords = [[0, 0, 0], [0.75, 0.75, 0.75], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
        lattice = [
            [3.8401979337, 0.00, 0.00],
            [1.9200989668, 3.3257101909, 0.00],
            [0.00, -2.2171384943, 3.1355090603],
        ]
        struct = Structure(lattice, ["Li+", "Li+", "O2-", "O2-"], coords)
        assert len(trafo.apply_transformation(struct, return_ranked_list=True)) == 2


class TestChargeBalanceTransformation:
    def test_apply_transformation(self):
        trafo = ChargeBalanceTransformation("Li+")
        coords = [
            [0, 0, 0],
            [0.375, 0.375, 0.375],
            [0.5, 0.5, 0.5],
            [0.875, 0.875, 0.875],
            [0.125, 0.125, 0.125],
            [0.25, 0.25, 0.25],
            [0.625, 0.625, 0.625],
            [0.75, 0.75, 0.75],
        ]

        lattice = [
            [3.8401979337, 0.00, 0.00],
            [1.9200989668, 3.3257101909, 0.00],
            [0.00, -2.2171384943, 3.1355090603],
        ]
        struct = Structure(lattice, ["Li+", "Li+", "Li+", "Li+", "Li+", "Li+", "O2-", "O2-"], coords)
        struct_trafo = trafo.apply_transformation(struct)

        assert struct_trafo.charge == approx(0, abs=1e-5)


@pytest.mark.skipif(not enumlib_present, reason="enum_lib not present.")
class TestEnumerateStructureTransformation:
    def test_apply_transformation(self):
        enum_trans = EnumerateStructureTransformation(refine_structure=True)
        enum_trans2 = EnumerateStructureTransformation(refine_structure=True, sort_criteria="nsites")
        struct = Structure.from_file(f"{VASP_IN_DIR}/POSCAR_LiFePO4")
        expected = [1, 3, 1]
        for idx, frac in enumerate([0.25, 0.5, 0.75]):
            trans = SubstitutionTransformation({"Fe": {"Fe": frac}})
            struct_trafo = trans.apply_transformation(struct)
            oxi_trans = OxidationStateDecorationTransformation({"Li": 1, "Fe": 2, "P": 5, "O": -2})
            struct_trafo = oxi_trans.apply_transformation(struct_trafo)
            all_structs = enum_trans.apply_transformation(struct_trafo, 100)
            assert len(all_structs) == expected[idx]
            assert isinstance(trans.apply_transformation(struct_trafo), Structure)
            for ss in all_structs:
                assert "energy" in ss
            all_structs = enum_trans2.apply_transformation(struct_trafo, 100)
            assert len(all_structs) == expected[idx]
            assert isinstance(trans.apply_transformation(struct_trafo), Structure)
            for ss in all_structs:
                assert "num_sites" in ss

        # make sure it works for non-oxidation state decorated structure
        trans = SubstitutionTransformation({"Fe": {"Fe": 0.5}})
        struct_trafo = trans.apply_transformation(struct)
        all_structs = enum_trans.apply_transformation(struct_trafo, 100)
        assert len(all_structs) == 3
        assert isinstance(trans.apply_transformation(struct_trafo), Structure)
        for struct_trafo in all_structs:
            assert "energy" not in struct_trafo

    @pytest.mark.xfail(reason="dgl don't support torch 2.4.1+, #4073")
    def test_m3gnet(self):
        pytest.importorskip("matgl")
        enum_trans = EnumerateStructureTransformation(refine_structure=True, sort_criteria="m3gnet_relax")
        struct = Structure.from_file(f"{VASP_IN_DIR}/POSCAR_LiFePO4")
        trans = SubstitutionTransformation({"Fe": {"Fe": 0.5, "Mn": 0.5}})
        struct_trafo = trans.apply_transformation(struct)
        all_structs = enum_trans.apply_transformation(struct_trafo, 100)
        assert len(all_structs) == 3
        assert isinstance(trans.apply_transformation(struct_trafo), Structure)
        for ss in all_structs:
            assert "energy" in ss

        # Check ordering of energy/atom
        assert (
            all_structs[0]["energy"] / all_structs[0]["num_sites"]
            <= all_structs[-1]["energy"] / all_structs[-1]["num_sites"]
        )

    @pytest.mark.xfail(reason="dgl don't support torch 2.4.1+, #4073")
    def test_callable_sort_criteria(self):
        matgl = pytest.importorskip("matgl")
        from matgl.ext.ase import Relaxer

        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        m3gnet_model = Relaxer(potential=pot)

        def sort_criteria(struct: Structure) -> tuple[Structure, float]:
            relax_results = m3gnet_model.relax(struct)
            energy = float(relax_results["trajectory"].energies[-1])
            return relax_results["final_structure"], energy

        enum_trans = EnumerateStructureTransformation(refine_structure=True, sort_criteria=sort_criteria)
        struct = Structure.from_file(f"{VASP_IN_DIR}/POSCAR_LiFePO4")
        trans = SubstitutionTransformation({"Fe": {"Fe": 0.5, "Mn": 0.5}})
        struct_trafo = trans.apply_transformation(struct)
        all_structs = enum_trans.apply_transformation(struct_trafo, 100)
        assert len(all_structs) == 3
        assert isinstance(trans.apply_transformation(struct_trafo), Structure)
        for ss in all_structs:
            assert "energy" in ss

        # Check ordering of energy/atom
        assert (
            all_structs[0]["energy"] / all_structs[0]["num_sites"]
            <= all_structs[-1]["energy"] / all_structs[-1]["num_sites"]
        )

    def test_max_disordered_sites(self):
        s_orig = Structure(
            Lattice.cubic(4),
            [{"Li": 0.2, "Na": 0.2, "K": 0.6}, {"O": 1}],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        est = EnumerateStructureTransformation(max_cell_size=None, max_disordered_sites=5)
        lst = est.apply_transformation(s_orig, return_ranked_list=100)
        assert len(lst) == 9
        for d in lst:
            assert len(d["structure"]) == 10

    def test_as_from_dict(self):
        trans = EnumerateStructureTransformation()
        dct = trans.as_dict()
        trans = EnumerateStructureTransformation.from_dict(dct)
        assert trans.symm_prec == approx(0.1)


class TestSubstitutionPredictorTransformation:
    def test_apply_transformation(self):
        trafo = SubstitutionPredictorTransformation(threshold=1e-3, alpha=-5, lambda_table=get_table())
        coords = [[0, 0, 0], [0.75, 0.75, 0.75], [0.5, 0.5, 0.5]]
        lattice = [
            [3.8401979337, 0.00, 0.00],
            [1.9200989668, 3.3257101909, 0.00],
            [0.00, -2.2171384943, 3.1355090603],
        ]
        struct = Structure(lattice, ["O2-", "Li1+", "Li1+"], coords)

        outputs = trafo.apply_transformation(struct, return_ranked_list=True)
        assert len(outputs) == 4, "incorrect number of structures"

    def test_as_dict(self):
        trafo = SubstitutionPredictorTransformation(threshold=2, alpha=-2, lambda_table=get_table())
        dct = trafo.as_dict()
        trafo = SubstitutionPredictorTransformation.from_dict(dct)
        assert trafo.threshold == 2, "incorrect threshold passed through dict"
        assert trafo._substitutor.p.alpha == -2, "incorrect alpha passed through dict"


@pytest.mark.skipif(not enumlib_present, reason="enum_lib not present.")
class TestMagOrderingTransformation(MatSciTest):
    def setup_method(self):
        lattice = Lattice.cubic(4.17)
        species = ["Ni", "O"]
        coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
        self.NiO = Structure.from_spacegroup(225, lattice, species, coords)

        lattice = Lattice([[2.085, 2.085, 0.0], [0.0, -2.085, -2.085], [-2.085, 2.085, -4.17]])
        species = ["Ni", "Ni", "O", "O"]
        coords = [[0.5, 0, 0.5], [0, 0, 0], [0.25, 0.5, 0.25], [0.75, 0.5, 0.75]]
        self.NiO_AFM_111 = Structure(lattice, species, coords)
        self.NiO_AFM_111.add_spin_by_site([-5, 5, 0, 0])

        lattice = Lattice([[2.085, 2.085, 0], [0, 0, -4.17], [-2.085, 2.085, 0]])
        species = ["Ni", "Ni", "O", "O"]
        coords = [[0.5, 0.5, 0.5], [0, 0, 0], [0, 0.5, 0], [0.5, 0, 0.5]]
        self.NiO_AFM_001 = Structure(lattice, species, coords)
        self.NiO_AFM_001.add_spin_by_site([-5, 5, 0, 0])

        self.Fe3O4 = Structure.from_file(f"{TEST_FILES_DIR}/cif/Fe3O4.cif")
        trans = AutoOxiStateDecorationTransformation()
        self.Fe3O4_oxi = trans.apply_transformation(self.Fe3O4)

        self.Li8Fe2NiCoO8 = Structure.from_file(f"{TEST_FILES_DIR}/cif/Li8Fe2NiCoO8.cif").remove_oxidation_states()

    def test_apply_transformation(self):
        trans = MagOrderingTransformation({"Fe": 5})
        struct = Structure.from_file(f"{VASP_IN_DIR}/POSCAR_LiFePO4")
        all_structs = trans.apply_transformation(struct, 10)
        assert len(all_structs) == 3
        spg_analyzer = SpacegroupAnalyzer(all_structs[0]["structure"], 0.1)
        assert spg_analyzer.get_space_group_number() == 31

        model = IsingModel(5, 5)
        trans = MagOrderingTransformation({"Fe": 5}, energy_model=model)
        alls2 = trans.apply_transformation(struct, 10)
        # Ising model with +J penalizes similar neighbor magmom.
        assert all_structs[0]["structure"] != alls2[0]["structure"]
        assert all_structs[0]["structure"] == alls2[2]["structure"]

        struct = self.get_structure("Li2O")
        # Li2O doesn't have magnetism of course, but this is to test the
        # enumeration.
        trans = MagOrderingTransformation({"Li+": 1}, max_cell_size=3)
        all_structs = trans.apply_transformation(struct, 100)
        # TODO: check this is correct, unclear what len(all_structs) should be
        # this assert just ensures it doesn't change unexpectedly
        assert len(all_structs) == 12

        trans = MagOrderingTransformation({"Ni": 5})
        all_structs = trans.apply_transformation(self.NiO.get_primitive_structure(), return_ranked_list=10)

        assert_allclose(self.NiO_AFM_111.lattice.parameters, all_structs[0]["structure"].lattice.parameters)
        assert_allclose(self.NiO_AFM_001.lattice.parameters, all_structs[1]["structure"].lattice.parameters)

    def test_ferrimagnetic(self):
        trans = MagOrderingTransformation({"Fe": 5}, order_parameter=0.75, max_cell_size=1)
        struct = Structure.from_file(f"{VASP_IN_DIR}/POSCAR_LiFePO4")
        spg_analyzer = SpacegroupAnalyzer(struct, 0.1)
        struct = spg_analyzer.get_refined_structure()
        all_structs = trans.apply_transformation(struct, 10)
        assert len(all_structs) == 1

    def test_as_from_dict(self):
        trans = MagOrderingTransformation({"Fe": 5}, order_parameter=0.75)
        dct = trans.as_dict()
        # Check JSON encodability
        _ = orjson.dumps(dct).decode()
        trans = MagOrderingTransformation.from_dict(dct)
        assert trans.mag_species_spin == {"Fe": 5}

        assert isinstance(trans.energy_model, SymmetryModel)

    def test_zero_spin_case(self):
        # ensure that zero spin case maintains sites and formula
        struct = self.get_structure("Li2O")
        trans = MagOrderingTransformation({"Li+": 0.0}, order_parameter=0.5)
        all_structs = trans.apply_transformation(struct)
        Li_site = all_structs.indices_from_symbol("Li")[0]
        # Ensure s does not have a spin property
        assert struct[Li_site].specie.spin is None
        # ensure sites are assigned a spin property in all_structs
        # assert "spin" in all_structs[Li_site].specie.properties
        assert all_structs.sites[Li_site].specie.spin == 0

    def test_advanced_usage(self):
        # test spin on just one oxidation state
        mag_types = {"Fe2+": 5}
        trans = MagOrderingTransformation(mag_types)
        all_structs = trans.apply_transformation(self.Fe3O4_oxi)
        assert isinstance(all_structs, Structure)
        assert str(all_structs[0].specie) == "Fe2+,spin=5"
        assert str(all_structs[2].specie) == "Fe3+"

        # test multiple order parameters
        # this should only order on Fe3+ site, but assign spin to both
        mag_types = {"Fe2+": 5, "Fe3+": 5}
        order_parameters = [
            MagOrderParameterConstraint(1, species_constraints="Fe2+"),
            MagOrderParameterConstraint(0.5, species_constraints="Fe3+"),
        ]
        trans = MagOrderingTransformation(mag_types, order_parameter=order_parameters)
        all_structs = trans.apply_transformation(self.Fe3O4_oxi)
        # using this 'sorted' syntax because exact order of sites in first
        # returned structure varies between machines: we just want to ensure
        # that the order parameter is accurate
        assert sorted(str(all_structs[idx].specie) for idx in range(2)) == sorted(["Fe2+,spin=5", "Fe2+,spin=5"])
        assert sorted(str(all_structs[idx].specie) for idx in range(2, 6)) == sorted(
            ["Fe3+,spin=5", "Fe3+,spin=5", "Fe3+,spin=-5", "Fe3+,spin=-5"]
        )
        assert str(all_structs[0].specie) == "Fe2+,spin=5"

        # this should give same results as previously
        # but with opposite sign on Fe2+ site
        mag_types = {"Fe2+": -5, "Fe3+": 5}
        order_parameters = [
            MagOrderParameterConstraint(1, species_constraints="Fe2+"),
            MagOrderParameterConstraint(0.5, species_constraints="Fe3+"),
        ]
        trans = MagOrderingTransformation(mag_types, order_parameter=order_parameters)
        all_structs = trans.apply_transformation(self.Fe3O4_oxi)
        assert sorted(str(all_structs[idx].specie) for idx in range(2)) == sorted(["Fe2+,spin=-5", "Fe2+,spin=-5"])
        assert sorted(str(all_structs[idx].specie) for idx in range(2, 6)) == sorted(
            ["Fe3+,spin=5", "Fe3+,spin=5", "Fe3+,spin=-5", "Fe3+,spin=-5"]
        )

        # while this should order on both sites
        mag_types = {"Fe2+": 5, "Fe3+": 5}
        order_parameters = [
            MagOrderParameterConstraint(0.5, species_constraints="Fe2+"),
            MagOrderParameterConstraint(0.25, species_constraints="Fe3+"),
        ]
        trans = MagOrderingTransformation(mag_types, order_parameter=order_parameters)
        all_structs = trans.apply_transformation(self.Fe3O4_oxi)
        assert sorted(str(all_structs[idx].specie) for idx in range(2)) == sorted(["Fe2+,spin=5", "Fe2+,spin=-5"])
        assert sorted(str(all_structs[idx].specie) for idx in range(2, 6)) == sorted(
            ["Fe3+,spin=5", "Fe3+,spin=-5", "Fe3+,spin=-5", "Fe3+,spin=-5"]
        )

        # add coordination numbers to our test case
        # don't really care what these are for the test case
        cns = [6, 6, 6, 6, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0]
        self.Fe3O4.add_site_property("cn", cns)

        # this should give FM ordering on cn=4 sites, and AFM ordering on cn=6 sites
        mag_types = {"Fe": 5}
        order_parameters = [
            MagOrderParameterConstraint(
                0.5,
                species_constraints="Fe",
                site_constraint_name="cn",
                site_constraints=6,
            ),
            MagOrderParameterConstraint(
                1.0,
                species_constraints="Fe",
                site_constraint_name="cn",
                site_constraints=4,
            ),
        ]
        trans = MagOrderingTransformation(mag_types, order_parameter=order_parameters)
        all_structs = trans.apply_transformation(self.Fe3O4)
        all_structs.sort(key=lambda x: x.properties["cn"], reverse=True)
        assert sorted(str(all_structs[idx].specie) for idx in range(4)) == sorted(
            ["Fe,spin=-5", "Fe,spin=-5", "Fe,spin=5", "Fe,spin=5"]
        )
        assert sorted(str(all_structs[idx].specie) for idx in range(4, 6)) == sorted(["Fe,spin=5", "Fe,spin=5"])

        # now ordering on both sites, equivalent to order_parameter = 0.5
        mag_types = {"Fe2+": 5, "Fe3+": 5}
        order_parameters = [
            MagOrderParameterConstraint(0.5, species_constraints="Fe2+"),
            MagOrderParameterConstraint(0.5, species_constraints="Fe3+"),
        ]
        trans = MagOrderingTransformation(mag_types, order_parameter=order_parameters)
        all_structs = trans.apply_transformation(self.Fe3O4_oxi, return_ranked_list=10)
        struct = all_structs[0]["structure"]
        assert sorted(str(struct[idx].specie) for idx in range(2)) == sorted(["Fe2+,spin=5", "Fe2+,spin=-5"])
        assert sorted(str(struct[idx].specie) for idx in range(2, 6)) == sorted(
            ["Fe3+,spin=5", "Fe3+,spin=-5", "Fe3+,spin=-5", "Fe3+,spin=5"]
        )
        assert len(all_structs) == 4

        # now mixed orderings where neither are equal or 1
        mag_types = {"Fe2+": 5, "Fe3+": 5}
        order_parameters = [
            MagOrderParameterConstraint(0.5, species_constraints="Fe2+"),
            MagOrderParameterConstraint(0.25, species_constraints="Fe3+"),
        ]
        trans = MagOrderingTransformation(mag_types, order_parameter=order_parameters)
        all_structs = trans.apply_transformation(self.Fe3O4_oxi, return_ranked_list=100)
        struct = all_structs[0]["structure"]
        assert sorted(str(struct[idx].specie) for idx in range(2)) == sorted(["Fe2+,spin=5", "Fe2+,spin=-5"])
        assert sorted(str(struct[idx].specie) for idx in range(2, 6)) == sorted(
            ["Fe3+,spin=5", "Fe3+,spin=-5", "Fe3+,spin=-5", "Fe3+,spin=-5"]
        )
        assert len(all_structs) == 2

        # now order on multiple species
        mag_types = {"Fe2+": 5, "Fe3+": 5}
        order_parameters = [MagOrderParameterConstraint(0.5, species_constraints=["Fe2+", "Fe3+"])]
        trans = MagOrderingTransformation(mag_types, order_parameter=order_parameters)
        all_structs = trans.apply_transformation(self.Fe3O4_oxi, return_ranked_list=10)
        struct = all_structs[0]["structure"]
        assert sorted(str(struct[idx].specie) for idx in range(2)) == sorted(["Fe2+,spin=5", "Fe2+,spin=-5"])
        assert sorted(str(struct[idx].specie) for idx in range(2, 6)) == sorted(
            ["Fe3+,spin=5", "Fe3+,spin=-5", "Fe3+,spin=-5", "Fe3+,spin=5"]
        )
        assert len(all_structs) == 6


@pytest.mark.skipif(not enumlib_present, reason="enum_lib not present.")
class TestDopingTransformation(MatSciTest):
    def test_apply_transformation(self):
        structure = MatSciTest.get_structure("LiFePO4")
        spga = SpacegroupAnalyzer(structure, 0.1)
        structure = spga.get_refined_structure()
        trafo = DopingTransformation("Ca2+", min_length=10)
        ss = trafo.apply_transformation(structure, 100)
        assert len(ss) == 1

        trafo = DopingTransformation("Al3+", min_length=15, ionic_radius_tol=0.1)
        ss = trafo.apply_transformation(structure, 100)
        assert len(ss) == 0

        # Aliovalent doping with vacancies
        for dopant, n_structures in [("Al3+", 2), ("N3-", 235), ("Cl-", 8)]:
            trafo = DopingTransformation(dopant, min_length=4, alio_tol=1, max_structures_per_enum=1000)
            ss = trafo.apply_transformation(structure, 1000)
            assert len(ss) == n_structures
            for d in ss:
                assert d["structure"].charge == approx(0)

        # Aliovalent doping with codopant
        for dopant, n_structures in [("Al3+", 3), ("N3-", 37), ("Cl-", 37)]:
            trafo = DopingTransformation(
                dopant,
                min_length=4,
                alio_tol=1,
                codopant=True,
                max_structures_per_enum=1000,
            )
            ss = trafo.apply_transformation(structure, 1000)
            assert len(ss) == n_structures
            for d in ss:
                assert d["structure"].charge == approx(0)

        # Make sure compensation is done with lowest oxi state
        structure = MatSciTest.get_structure("SrTiO3")
        trafo = DopingTransformation(
            "Nb5+",
            min_length=5,
            alio_tol=1,
            max_structures_per_enum=1000,
            allowed_doping_species=["Ti4+"],
        )
        ss = trafo.apply_transformation(structure, 1000)
        assert len(ss) == 3
        for d in ss:
            assert d["structure"].formula == "Sr7 Ti6 Nb2 O24"

    def test_as_from_dict(self):
        trans = DopingTransformation("Al3+", min_length=5, alio_tol=1, codopant=False, max_structures_per_enum=1)
        dct = trans.as_dict()
        # Check JSON encodability
        _ = orjson.dumps(dct).decode()
        trans = DopingTransformation.from_dict(dct)
        assert str(trans.dopant) == "Al3+"
        assert trans.max_structures_per_enum == 1

    def test_find_codopant(self):
        assert find_codopant(Species("Fe", 2), 1) == Species("Cu", 1)
        assert find_codopant(Species("Fe", 2), 3) == Species("In", 3)

    def test_find_codopant_no_candidates(self):
        # No species has oxidation state +99 with a known ionic radius
        with pytest.raises(ValueError, match="No species found with oxidation state 99"):
            find_codopant(Species("Fe", 2), 99)


class TestSlabTransformation(MatSciTest):
    def test_apply_transformation(self):
        struct = self.get_structure("LiFePO4")
        trans = SlabTransformation([0, 0, 1], 10, 10, shift=0.25)
        gen = SlabGenerator(struct, [0, 0, 1], 10, 10)
        slab_from_gen = gen.get_slab(0.25)
        slab_from_trans = trans.apply_transformation(struct)
        assert_allclose(slab_from_gen.lattice.matrix, slab_from_trans.lattice.matrix)
        assert_allclose(slab_from_gen.cart_coords, slab_from_trans.cart_coords)

        fcc = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        trans = SlabTransformation([1, 1, 1], 10, 10)
        slab_from_trans = trans.apply_transformation(fcc)
        gen = SlabGenerator(fcc, [1, 1, 1], 10, 10)
        slab_from_gen = gen.get_slab()
        assert_allclose(slab_from_gen.lattice.matrix, slab_from_trans.lattice.matrix)
        assert_allclose(slab_from_gen.cart_coords, slab_from_trans.cart_coords)


class TestGrainBoundaryTransformation(MatSciTest):
    def test_apply_transformation(self):
        Al_bulk = Structure.from_spacegroup("Fm-3m", Lattice.cubic(2.8575585), ["Al"], [[0, 0, 0]])
        gb_gen_params_s5 = {
            "rotation_axis": [1, 0, 0],
            "rotation_angle": 53.13010235415599,
            "expand_times": 3,
            "vacuum_thickness": 0.0,
            "normal": True,
            "plane": [0, -1, -3],
            "rm_ratio": 0.6,
        }
        gbg = GrainBoundaryGenerator(Al_bulk)
        gb_from_generator = gbg.gb_from_parameters(**gb_gen_params_s5)
        gbt_s5 = GrainBoundaryTransformation(**gb_gen_params_s5)
        gb_from_trans = gbt_s5.apply_transformation(Al_bulk)
        assert_allclose(gb_from_generator.lattice.matrix, gb_from_trans.lattice.matrix)
        assert_allclose(gb_from_generator.cart_coords, gb_from_trans.cart_coords)


class TestDisorderedOrderedTransformation(MatSciTest):
    def test_apply_transformation(self):
        # nonsensical example just for testing purposes
        struct = self.get_structure("BaNiO3")

        trans = DisorderOrderedTransformation()
        output = trans.apply_transformation(struct)

        assert not output.is_ordered
        assert output[-1].species.as_dict() == approx({"Ni": 0.5, "Ba": 0.5})


@pytest.mark.skipif(not mcsqs_cmd, reason="mcsqs not present.")
class TestSQSTransformation(MatSciTest):
    def test_apply_transformation(self):
        pzt_structs = loadfn(f"{TEST_FILES_DIR}/io/atat/mcsqs/pzt-structs.json")
        trans = SQSTransformation(scaling=[2, 1, 1], search_time=0.01, instances=1, wd=0)
        # nonsensical example just for testing purposes
        struct = self.get_structure("Pb2TiZrO6").copy()
        struct.replace_species({"Ti": {"Ti": 0.5, "Zr": 0.5}, "Zr": {"Ti": 0.5, "Zr": 0.5}})
        struct_out = trans.apply_transformation(struct)
        matches = [struct_out.matches(s) for s in pzt_structs]
        assert any(matches)

    def test_return_ranked_list(self):
        # list of structures
        pzt_structs_2 = loadfn(f"{TEST_FILES_DIR}/io/atat/mcsqs/pzt-structs-2.json")

        n_structs_expected = 1
        sqs_kwargs = {"scaling": 2, "search_time": 0.01, "instances": 8, "wd": 0}
        for all_structs in (True, False):
            if all_structs:
                # when we don't remove structures from the search, should get
                # return one structure for each instance run
                sqs_kwargs |= {"best_only": False, "remove_duplicate_structures": False}
                n_structs_expected = sqs_kwargs["instances"]

            trans = SQSTransformation(**sqs_kwargs)
            struct = self.get_structure("Pb2TiZrO6").copy()
            struct.replace_species({"Ti": {"Ti": 0.5, "Zr": 0.5}, "Zr": {"Ti": 0.5, "Zr": 0.5}})
            ranked_list_out = trans.apply_transformation(struct, return_ranked_list=True)
            matches = [ranked_list_out[0]["structure"].matches(struct) for struct in pzt_structs_2]
            assert any(matches)
            assert len(ranked_list_out) == n_structs_expected

    def test_spin(self):
        trans = SQSTransformation(scaling=[2, 1, 1], search_time=0.01, instances=1, wd=0)

        # nonsensical example just for testing purposes
        struct = self.get_structure("Pb2TiZrO6").copy()
        struct.replace_species({"Ti": {"Ti,spin=5": 0.5, "Ti,spin=-5": 0.5}})

        struct_out = trans.apply_transformation(struct)
        struct_out_specie_strings = [site.species_string for site in struct_out]
        assert "Ti0+,spin=-5" in struct_out_specie_strings
        assert "Ti0+,spin=5" in struct_out_specie_strings


@pytest.mark.skipif(ClusterSpace is None, reason="icet not installed.")
class TestSQSTransformationIcet(MatSciTest):
    stored_run: dict = loadfn(f"{TEST_FILES_DIR}/transformations/icet-sqs-fcc-Mg_75-Al_25-scaling_8.json.gz")
    scaling: int = 8

    def test_icet_import(self):
        from pymatgen.io import icet as icet_mod

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(icet_mod, "ClusterSpace", None)

            with pytest.raises(ImportError):
                icet_mod.IcetSQS(
                    structure=self.stored_run["disordered_structure"],
                    scaling=self.scaling,
                    instances=None,
                    cluster_cutoffs={2: 5.0},
                )

    def test_enumeration(self):
        sqs = SQSTransformation(
            scaling=self.scaling,
            sqs_method="icet-enumeration",
            instances=2,
            best_only=False,
        )
        sqs_structure = sqs.apply_transformation(self.stored_run["disordered_structure"], return_ranked_list=1)
        for key in ("structure", "objective_function"):
            assert sqs_structure[0][key] == self.stored_run[key]

    def test_monte_carlo(self):
        sqs = SQSTransformation(
            scaling=self.scaling,
            sqs_method="icet-monte_carlo",
            icet_sqs_kwargs={"n_steps": 5},
            instances=2,
        )
        sqs_structure = sqs.apply_transformation(self.stored_run["disordered_structure"], return_ranked_list=False)
        assert isinstance(sqs_structure, Structure)
        assert len(sqs_structure) == self.scaling * len(self.stored_run["disordered_structure"])

        sqs_output = sqs.apply_transformation(self.stored_run["disordered_structure"], return_ranked_list=1)

        assert isinstance(sqs_output, list)
        assert len(sqs_output) == 1
        assert isinstance(sqs_output[0], dict)
        expected_types = {"structure": Structure, "objective_function": float}
        for key, val in expected_types.items():
            assert isinstance(sqs_output[0][key], val)


class TestCubicSupercellTransformation(MatSciTest):
    def test_apply_transformation_cubic_supercell(self):
        structure = self.get_structure("TlBiSe2")
        min_atoms = 100
        max_atoms = 1000

        # Test the transformation without constraining trans_mat to be diagonal
        supercell_generator = CubicSupercellTransformation(min_atoms=min_atoms, max_atoms=max_atoms, min_length=13.0)
        superstructure = supercell_generator.apply_transformation(structure)

        n_atoms = len(superstructure)
        assert n_atoms >= min_atoms
        assert n_atoms <= max_atoms
        assert_allclose(
            superstructure.lattice.matrix[0],
            [1.49656087e01, -1.11448000e-03, 9.04924836e00],
        )
        assert_allclose(superstructure.lattice.matrix[1], [-0.95005506, 14.95766342, 10.01819773])
        assert_allclose(
            superstructure.lattice.matrix[2],
            [3.69130000e-02, 4.09320200e-02, 5.90830153e01],
        )
        assert len(superstructure) == 448
        assert_array_equal(
            supercell_generator.transformation_matrix,
            [[4, 0, 0], [1, 4, -4], [0, 0, 1]],
        )

        # Test the diagonal transformation
        structure2 = self.get_structure("Si")
        sga = SpacegroupAnalyzer(structure2)
        structure2 = sga.get_primitive_standard_structure()
        diagonal_supercell_generator = CubicSupercellTransformation(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=13.0,
            force_diagonal=True,
        )
        _ = diagonal_supercell_generator.apply_transformation(structure2)
        assert_array_equal(diagonal_supercell_generator.transformation_matrix, np.eye(3) * 4)

        # test force_90_degrees
        structure2 = self.get_structure("Si")
        sga = SpacegroupAnalyzer(structure2)
        structure2 = sga.get_primitive_standard_structure()
        diagonal_supercell_generator = CubicSupercellTransformation(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=13.0,
            force_90_degrees=True,
        )
        transformed_structure = diagonal_supercell_generator.apply_transformation(structure2)
        assert_allclose(list(transformed_structure.lattice.angles), [90.0, 90.0, 90.0])

        structure = self.get_structure("BaNiO3")
        min_atoms = 100
        max_atoms = 1000

        # Test the transformation without constraining trans_mat to be diagonal
        supercell_generator = CubicSupercellTransformation(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=10.0,
            force_90_degrees=True,
        )
        transformed_structure = supercell_generator.apply_transformation(structure)
        assert_allclose(list(transformed_structure.lattice.angles), [90.0, 90.0, 90.0])

    def test_apply_transformation_orthorhombic_supercell(self):
        structure = self.get_structure("Li3V2(PO4)3")
        min_atoms = 100
        max_atoms = 400

        supercell_generator_cubic = CubicSupercellTransformation(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=10.0,
            force_90_degrees=False,
            allow_orthorhombic=False,
            max_length=25,
        )

        transformed_cubic = supercell_generator_cubic.apply_transformation(structure)

        supercell_generator_orthorhombic = CubicSupercellTransformation(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=10.0,
            force_90_degrees=False,
            allow_orthorhombic=True,
            max_length=25,
        )

        transformed_orthorhombic = supercell_generator_orthorhombic.apply_transformation(structure)

        assert_array_equal(
            supercell_generator_orthorhombic.transformation_matrix,
            [[0, -2, 1], [-2, 0, 0], [0, 0, -2]],
        )

        # make sure that the orthorhombic supercell is different from the cubic cell
        assert not np.array_equal(
            supercell_generator_cubic.transformation_matrix,
            supercell_generator_orthorhombic.transformation_matrix,
        )
        assert not np.allclose(transformed_cubic.lattice.angles, transformed_orthorhombic.lattice.angles)
        assert not np.allclose(transformed_orthorhombic.lattice.abc, transformed_cubic.lattice.abc)

        structure = self.get_structure("Si")
        min_atoms = 100
        max_atoms = 400

        supercell_generator_cubic = CubicSupercellTransformation(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=10.0,
            force_90_degrees=True,
            allow_orthorhombic=False,
            max_length=25,
        )

        transformed_cubic = supercell_generator_cubic.apply_transformation(structure)

        supercell_generator_orthorhombic = CubicSupercellTransformation(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=10.0,
            force_90_degrees=True,
            allow_orthorhombic=True,
            max_length=25,
        )

        transformed_orthorhombic = supercell_generator_orthorhombic.apply_transformation(structure)

        assert_array_equal(
            supercell_generator_orthorhombic.transformation_matrix,
            [[3, 0, 0], [-2, 4, 0], [-2, 4, 6]],
        )

        # make sure that the orthorhombic supercell is different from the cubic cell
        assert not np.array_equal(
            supercell_generator_cubic.transformation_matrix,
            supercell_generator_orthorhombic.transformation_matrix,
        )
        assert not np.allclose(transformed_orthorhombic.lattice.abc, transformed_cubic.lattice.abc)
        # only angels are expected to be the same because of `force_90_degrees = True`
        assert_allclose(transformed_cubic.lattice.angles, transformed_orthorhombic.lattice.angles)


class TestAddAdsorbateTransformation(MatSciTest):
    def test_apply_transformation(self):
        co = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.23]])
        trans = AddAdsorbateTransformation(co)
        pt = Structure(Lattice.cubic(5), ["Pt"], [[0, 0, 0]])  # fictitious
        slab = SlabTransformation([0, 0, 1], 20, 10).apply_transformation(pt)
        out = trans.apply_transformation(slab)

        assert out.reduced_formula == "Pt4CO"


class TestSubstituteSurfaceSiteTransformation(MatSciTest):
    def test_apply_transformation(self):
        trans = SubstituteSurfaceSiteTransformation("Au")
        pt = Structure(Lattice.cubic(5), ["Pt"], [[0, 0, 0]])  # fictitious
        slab = SlabTransformation([0, 0, 1], 20, 10).apply_transformation(pt)
        out = trans.apply_transformation(slab)

        assert out.reduced_formula == "Pt3Au"


@pytest.mark.skipif(not hiphive, reason="hiphive not present")
class TestMonteCarloRattleTransformation(MatSciTest):
    def test_apply_transformation(self):
        struct = self.get_structure("Si")
        mcrt = MonteCarloRattleTransformation(0.01, 2, seed=1)
        s_trans = mcrt.apply_transformation(struct)

        assert not np.allclose(struct.cart_coords, s_trans.cart_coords, atol=0.01)
        assert_allclose(struct.cart_coords, s_trans.cart_coords, atol=1)

        # test using same seed gives same coords
        mcrt = MonteCarloRattleTransformation(0.01, 2, seed=1)
        s_trans2 = mcrt.apply_transformation(struct)
        assert_allclose(s_trans.cart_coords, s_trans2.cart_coords)


# -----------------------------------------------------------------------------
# Coverage improvements for externally-gated / edge-case branches in
# advanced_transformations.py. These exercise pure-Python paths (error
# handling, helper methods, and SQS post-processing) that do not require
# the external enumlib/mcsqs/hiphive binaries; see AGENTS.md.
# -----------------------------------------------------------------------------
DISORDERED = Structure(
    Lattice.cubic(4.0),
    [{"Fe": 0.5, "Ni": 0.5}],
    [[0.0, 0.0, 0.0]],
)


class _FakeSqs:
    """Minimal stand-in for mcsqs's Sqs object so the SQS post-processing
    helper can be exercised without the external ATAT binary."""

    def __init__(self, share: bool = False, objective: float = 4.0):
        self.bestsqs = Structure(Lattice.cubic(4.0), ["Fe", "Ni"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        s1 = Structure(Lattice.cubic(4.0), ["Fe", "Ni"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        s3 = Structure(Lattice.cubic(4.0), ["Fe", "Ni"], [[0.0, 0.0, 0.0], [0.25, 0.5, 0.5]])
        if share:
            # the SAME structure object appears twice in allsqs -> exercise the
            # cleanup-on-shared-objects path that _get_unique_best_sqs_structs must
            # tolerate
            self.allsqs = [
                {"structure": s1, "objective_function": 5.0},
                {"structure": s1, "objective_function": 5.0},
                {"structure": s3, "objective_function": objective},
            ]
        else:
            s1 = Structure(Lattice.cubic(4.0), ["Fe", "Ni"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
            s3 = Structure(Lattice.cubic(4.0), ["Fe", "Ni"], [[0.0, 0.0, 0.0], [0.25, 0.5, 0.5]])
            self.allsqs = [
                {"structure": s1, "objective_function": 5.0},
                {"structure": s3, "objective_function": objective},
            ]
        self.objective_function = objective


class _FakeSqsNonFloat:
    """A parallel Sqs object where one candidate carries a non-float objective
    function (e.g. an error string); used to exercise the sort fallback in
    ``_get_unique_best_sqs_structs``."""

    @staticmethod
    def _make():
        obj = _FakeSqsNonFloat()
        s1 = Structure(Lattice.cubic(4.0), ["Fe", "Ni"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        s2 = Structure(Lattice.cubic(4.0), ["Fe", "Ni"], [[0.0, 0.0, 0.0], [0.25, 0.5, 0.5]])
        obj.bestsqs = Structure(Lattice.cubic(4.0), ["Fe"], [[0.0, 0.0, 0.0]])
        obj.objective_function = 3.0
        obj.allsqs = [
            {"structure": s1, "objective_function": "err"},
            {"structure": s2, "objective_function": 3.0},
        ]
        return obj


class TestMagOrderParameterConstraintExtra:
    def test_site_constraint_name_required_when_site_constraints_given(self):
        with pytest.raises(ValueError, match="Specify the name of the site constraint"):
            MagOrderParameterConstraint(0.5, site_constraints=[6, 8])

    def test_site_constraints_required_when_name_given(self):
        with pytest.raises(ValueError, match="Please specify some site constraints"):
            MagOrderParameterConstraint(0.5, site_constraint_name="cn")

    @pytest.mark.parametrize("op", [1.5, 2.0, -0.1])
    def test_order_parameter_out_of_bounds(self, op):
        with pytest.raises(ValueError, match="Order parameter must lie between 0 and 1"):
            MagOrderParameterConstraint(op)

    def test_non_standard_order_parameter_warns(self):
        with pytest.warns(UserWarning, match="non-standard order parameter"):
            MagOrderParameterConstraint(0.3, species_constraints="Fe")

    def test_serialization_roundtrip(self):
        c = MagOrderParameterConstraint(
            0.5,
            species_constraints=["Fe", "Ni"],
            site_constraint_name="cn",
            site_constraints=[4, 6],
        )
        c2 = MagOrderParameterConstraint.from_dict(c.as_dict())
        assert c2.order_parameter == approx(0.5)
        assert c2.species_constraints == ["Fe", "Ni"]
        assert c2.site_constraint_name == "cn"
        assert c2.site_constraints == [4, 6]

    def test_satisfies_constraint(self):
        struct = Structure(Lattice.cubic(4.0), ["Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        struct.add_site_property("cn", [6, 8])
        constraint = MagOrderParameterConstraint(
            0.5,
            species_constraints="Fe",
            site_constraint_name="cn",
            site_constraints=[6],
        )
        assert constraint.satisfies_constraint(struct.sites[0]) is True
        assert constraint.satisfies_constraint(struct.sites[1]) is False
        assert constraint.satisfies_constraint(DISORDERED.sites[0]) is False
        assert MagOrderParameterConstraint(0.5, species_constraints="Ni").satisfies_constraint(struct.sites[0]) is False


class TestChargeBalanceTransformationExtra:
    def test_repr(self):
        assert "O2-" in repr(ChargeBalanceTransformation("O2-"))

    def test_addition_not_supported(self):
        # a positively charged structure with a negatively charged balance
        # specie would require *adding* it, which is not supported yet.
        struct = Structure(
            Lattice.cubic(4.0),
            ["Al3+", "Al3+", "O2-"],
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
        )
        assert struct.charge > 0
        with pytest.raises(ValueError, match="addition of specie not yet supported"):
            ChargeBalanceTransformation("O2-").apply_transformation(struct.copy())


class TestDisorderOrderedTransformationExtra:
    def test_disordered_input_raises(self):
        with pytest.raises(ValueError, match="disordered structures only"):
            DisorderOrderedTransformation().apply_transformation(DISORDERED)

    def test_no_partition_returns_none(self):
        struct = Structure(Lattice.cubic(4.0), ["Fe", "Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        assert DisorderOrderedTransformation().apply_transformation(struct) is None
        # returns None regardless of return_ranked_list
        assert DisorderOrderedTransformation().apply_transformation(struct, return_ranked_list=5) is None

    def test_return_ranked_list_truncates(self):
        struct = Structure(
            Lattice.cubic(4.0),
            ["Fe", "Ni", "O"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        )
        res = DisorderOrderedTransformation().apply_transformation(struct, return_ranked_list=2)
        assert len(res) == 2
        single = DisorderOrderedTransformation().apply_transformation(struct)
        assert isinstance(single, Structure)


class TestFindCodopantExtra:
    def test_happy_path(self):
        codop = find_codopant(Species("Ti", 4), 4, allowed_elements=["Ti", "Zr"])
        assert codop.oxi_state == 4

    def test_target_without_ionic_radius_raises(self):
        # N7+ has no tabulated ionic radius
        with pytest.raises(ValueError, match="has no ionic radius"):
            find_codopant(Species("N", 7), 2)

    def test_no_candidates_raises(self):
        # H has no known ionic radius at oxidation state +4
        with pytest.raises(ValueError, match="No species found with oxidation state"):
            find_codopant(Species("Fe", 2), 4, allowed_elements=["H"])


class TestCubicSupercellTransformationExtra:
    def test_orthorhombic_requires_max_length(self):
        st = CubicSupercellTransformation(allow_orthorhombic=True)
        with pytest.raises(AttributeError, match="max_length is required"):
            st.apply_transformation(Structure(Lattice.cubic(4.0), ["Fe"], [[0.0, 0.0, 0.0]]))

    def test_force_diagonal(self):
        st = CubicSupercellTransformation(force_diagonal=True, min_length=10.0)
        out = st.apply_transformation(Structure(Lattice.cubic(4.0), ["Fe"], [[0.0, 0.0, 0.0]]))
        # ceil(10 / 4.0) = 3, so the diagonal of the transformation matrix is 3x3x3
        assert np.all(np.diag(st.transformation_matrix) == np.ceil(10.0 / 4.0))
        assert int(st.transformation_matrix[0, 0] ** 3) == len(out)

    def test_check_constraints(self):
        st = CubicSupercellTransformation(min_length=10.0, max_atoms=10)
        vecs = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
        supercell = Structure(Lattice(vecs), ["Fe", "Ni"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        assert st.check_constraints(vecs, 1, supercell) is True
        # number of atoms beyond max_atoms fails the constraint
        assert st.check_constraints(vecs, 1000, supercell) is False

    def test_round_and_make_arr_singular_handles_zero_row_and_column(self):
        arr_zrow = np.array([[0.3, 0.3, 0.3], [1.0, 0.6, 0.6], [0.7, 0.7, 0.7]])
        out_row = advanced._round_and_make_arr_singular(arr_zrow)
        assert out_row.any(axis=1).all()
        arr_zcol = np.array([[1.0, 0.3, 0.3], [1.0, 0.3, 0.3], [1.0, 0.3, 0.3]])
        out_col = advanced._round_and_make_arr_singular(arr_zcol)
        assert out_col.any(axis=0).all()


class TestSQSTransformationHelpers:
    def test_cluster_estimate(self):
        clusters = SQSTransformation._sqs_cluster_estimate(DISORDERED, {2: 1, 3: 1})
        assert set(clusters) == {2, 3}
        assert all(isinstance(v, float) and v > 0 for v in clusters.values())

    def test_disordered_substructure(self):
        sub = SQSTransformation._get_disordered_substructure(DISORDERED)
        assert len(sub) == 1
        assert not sub.sites[0].is_ordered

    def test_max_neighbor_distance(self):
        assert SQSTransformation._get_max_neighbor_distance(DISORDERED, 1) > 0

    def test_apply_validation(self):
        sqs1 = SQSTransformation(scaling=2, sqs_method="mcsqs")
        with pytest.raises(ValueError, match="has no instances"):
            sqs1.apply_transformation(DISORDERED, return_ranked_list=5)
        sqs2 = SQSTransformation(scaling=2, sqs_method="mcsqs", instances=3)
        with pytest.raises(ValueError, match="cannot be greater than"):
            sqs2.apply_transformation(DISORDERED, return_ranked_list=10)

    def test_unique_best_sqs_structs(self):
        res = SQSTransformation._get_unique_best_sqs_structs(
            _FakeSqs(),
            best_only=True,
            return_ranked_list=100,
            remove_duplicate_structures=True,
            reduction_algo=False,
        )
        assert len(res) == 1
        assert approx(res[0]["objective_function"]) == 4.0

    def test_unique_best_sqs_structs_non_best_and_non_float(self):
        res = SQSTransformation._get_unique_best_sqs_structs(
            _FakeSqsNonFloat._make(),
            best_only=False,
            return_ranked_list=10,
            remove_duplicate_structures=False,
            reduction_algo=False,
        )
        assert len(res) == 2

    def test_unique_best_sqs_structs_with_reduction(self):
        res = SQSTransformation._get_unique_best_sqs_structs(
            _FakeSqs(),
            best_only=True,
            return_ranked_list=100,
            remove_duplicate_structures=True,
            reduction_algo="LLL",
        )
        assert res
        assert isinstance(res[0]["structure"], Structure)

    def test_unique_best_sqs_structs_shared_structure_repeats(self):
        """Regression test for a crash when the same Structure object is reused
        across several entries of ``allsqs`` with ``remove_duplicate_structures=False``.

        Previously the cleanup loop ran an unguarded ``del`` on the temporary
        ``objective_function`` attribute, which raised ``AttributeError`` on the
        second occurrence of the shared structure object.
        """
        res = SQSTransformation._get_unique_best_sqs_structs(
            _FakeSqs(share=True),
            best_only=False,
            return_ranked_list=100,
            remove_duplicate_structures=False,
            reduction_algo=False,
        )
        # 3 candidates, two of them the very same Structure object
        assert len(res) == 3
        # the temporary attribute must have been cleaned up
        for dct in res:
            assert not hasattr(dct["structure"], "objective_function")
