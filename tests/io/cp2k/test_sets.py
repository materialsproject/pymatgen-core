from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from pymatgen.core import Species
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.cp2k.inputs import Keyword
from pymatgen.io.cp2k.sets import (
    SETTINGS,
    CellOptSet,
    Cp2kValidationError,
    DftSet,
    GaussianTypeOrbitalBasisSet,
    GthPotential,
    HybridCellOptSet,
    HybridRelaxSet,
    HybridStaticSet,
    RelaxSet,
    StaticSet,
)
from pymatgen.io.vasp.inputs import Kpoints, KpointsSupportedModes
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

CP2K_TEST_DIR = f"{TEST_FILES_DIR}/io/cp2k"
# Local test-files dir (independent of the installed package's TEST_FILES_DIR).
LOCAL_CP2K_TEST_DIR = Path(__file__).parent.parent.parent.parent / "test-files" / "io" / "cp2k"

Si_structure = Structure(
    lattice=[[0, 2.734364, 2.734364], [2.734364, 0, 2.734364], [2.734364, 2.734364, 0]],
    species=["Si", "Si"],
    coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
)
molecule = Molecule(species=["Si"], coords=[[0, 0, 0]])

# Basis set / potential with objects
Si_gto_basis = """
Si SZV-MOLOPT-GTH SZV-MOLOPT-GTH-q4
1
2 0 1 6 1 1
    2.693604434572  0.015333179500 -0.005800105400
    1.359613855428 -0.283798205000 -0.059172026000
    0.513245176029 -0.228939692700  0.121487149900
    0.326563011394  0.728834000900  0.423382421100
    0.139986977410  0.446205299300  0.474592116300
    0.068212286977  0.122025292800  0.250129397700
"""
Si_gth_pot = """Si GTH-BLYP-q4 GTH-BLYP
2    2
0.44000000    1    -6.25958674
2
0.44465247    2     8.31460936    -2.33277947
                                    3.01160535
0.50279207    1     2.33241791"""

BASIS_AND_POTENTIAL: dict[str, dict[str, GaussianTypeOrbitalBasisSet | GthPotential]] = {
    "Si": {"basis": GaussianTypeOrbitalBasisSet.from_str(Si_gto_basis), "potential": GthPotential.from_str(Si_gth_pot)},
    "Fe": {
        "basis": GaussianTypeOrbitalBasisSet.from_str(Si_gto_basis.replace("Si", "Fe")),
        "potential": GthPotential.from_str(Si_gth_pot.replace("Si", "Fe")),
    },
}


class TestDftSet(MatSciTest):
    def test_dft_set(self) -> None:
        """Test various DFT set configurations."""
        SETTINGS["PMG_CP2K_DATA_DIR"] = CP2K_TEST_DIR

        # Basis sets / potentials searching
        basis_and_potential: dict[str, str | dict[str, str | None] | None] = {
            "basis_type": "SZV",
            "potential_type": "Pseudopotential",
            "functional": None,
        }
        dft_set = DftSet(Si_structure, basis_and_potential=basis_and_potential, xc_functionals="PBE")

        # Basis sets / potentials by name
        basis_and_potential = {"Si": {"basis": "SZV-GTH-q4", "potential": "GTH-PBE-q4"}}
        dft_set = DftSet(Si_structure, basis_and_potential=basis_and_potential, xc_functionals="PBE")

        # Basis sets / potentials by name with ADMM
        basis_and_potential = {
            "Si": {
                "basis": "SZV-GTH-q4",
                "potential": "GTH-PBE-q4",
                "aux_basis": "cFIT3",
            }
        }
        dft_set = DftSet(Si_structure, basis_and_potential=basis_and_potential, xc_functionals="PBE")
        basis_sets = dft_set["force_eval"]["subsys"]["Si_1"].get("basis_set")
        assert any("AUX_FIT" in b.values for b in basis_sets)
        assert any("cFIT3" in b.values for b in basis_sets)

        # Basis sets / potentials by hash value
        basis_and_potential = {
            "Si": {
                "basis": "30767c18f6e7e46c1b56c1d34ff6007d",
                "potential": "21e2f468a18404ff6119fe801da81e43",
            }
        }
        dft_set = DftSet(Si_structure, basis_and_potential=basis_and_potential, xc_functionals="PBE")

        set_kwargs = dict.fromkeys(("print_pdos", "print_dos", "print_v_hartree", "print_e_density"), False)
        dft_set = DftSet(
            Si_structure,
            basis_and_potential=BASIS_AND_POTENTIAL,
            xc_functionals="PBE",
            **set_kwargs,
        )
        assert dft_set.cutoff == approx(150)

        # Test that printing will activate sections
        assert not dft_set.check("motion")
        dft_set.activate_motion()
        assert dft_set.check("motion")
        assert not dft_set.check("force_eval/dft/print/pdos")
        dft_set.print_pdos()
        assert dft_set.check("force_eval/dft/print/pdos")
        assert not dft_set.check("force_eval/dft/print/v_hartree_cube")
        dft_set.print_v_hartree()
        assert dft_set.check("force_eval/dft/print/v_hartree_cube")

        # Test property activators
        assert not dft_set.check("force_eval/properties")
        dft_set.activate_nmr()
        dft_set.activate_epr()
        dft_set.activate_hyperfine()
        dft_set.activate_polar()
        dft_set.activate_tddfpt()
        assert dft_set.check("force_eval/properties/linres/localize")
        assert dft_set.check("force_eval/properties/linres/nmr/print/chi_tensor")
        assert dft_set.check("force_eval/properties/linres/epr/print/g_tensor")
        assert dft_set.check("force_eval/properties/tddfpt")
        assert dft_set.check("force_eval/dft/print/hyperfine_coupling_tensor")

        # For at least up to v2022.1, DOS doesn't work without kpoints
        assert not dft_set.check("force_eval/dft/print/dos")
        dft_set.print_dos()
        assert not dft_set.check("force_eval/dft/print/dos")

        assert not dft_set.check("force_eval/dft/xc/hf")
        dft_set.activate_hybrid()
        assert dft_set.check("force_eval/dft/xc/hf")
        assert dft_set.check("force_eval/dft/auxiliary_density_matrix_method")

        # Validator will trip for kpoints + hfx
        dft_set |= {"force_eval": {"dft": {"kpoints": {}}}}
        with pytest.raises(
            Cp2kValidationError,
            match=re.escape("CP2K v2022.1: Does not support hartree fock with kpoints"),
        ):
            dft_set.validate()

        # Test non-periodic settings
        dft_set = DftSet(molecule, basis_and_potential=basis_and_potential, xc_functionals="PBE")
        assert dft_set.check("force_eval/dft/poisson")
        assert dft_set["force_eval"]["dft"]["poisson"].get("periodic").values[0].upper() == "NONE"

    def test_kind_magnetization(self) -> None:
        """Test different ways of setting kind (i.e. element-specific) initial magnetization."""
        fe_structure = Structure(
            lattice=np.eye(3) * 3,
            species=["Fe", "Fe"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        )

        # Test 1: Setting via site properties
        dft_set = DftSet(
            fe_structure.copy().add_site_property("magmom", [2.0, -2.0]),
            basis_and_potential=BASIS_AND_POTENTIAL,
            xc_functionals="PBE",
        )
        fe1_kind = dft_set["force_eval"]["subsys"]["Fe_1"]
        fe2_kind = dft_set["force_eval"]["subsys"]["Fe_2"]
        assert {fe1_kind["magnetization"].values[0], fe2_kind["magnetization"].values[0]} == {2.0, -2.0}

        # Test 2: Setting via element defaults
        element_defaults = {"Fe": {"magnetization": 4.0}}
        dft_set = DftSet(
            fe_structure,
            basis_and_potential=BASIS_AND_POTENTIAL,
            xc_functionals="PBE",
            element_defaults=element_defaults,
        )
        fe_kind = dft_set["force_eval"]["subsys"]["Fe_1"]
        assert math.isclose(fe_kind["magnetization"].values[0], 4.0)

        # Test 3: Site properties take precedence over element defaults
        fe_structure.add_site_property("magmom", [2.0, -2.0])
        dft_set = DftSet(
            fe_structure,
            basis_and_potential=BASIS_AND_POTENTIAL,
            xc_functionals="PBE",
            element_defaults=element_defaults,
        )
        fe1_kind = dft_set["force_eval"]["subsys"]["Fe_1"]
        fe2_kind = dft_set["force_eval"]["subsys"]["Fe_2"]
        assert {fe1_kind["magnetization"].values[0], fe2_kind["magnetization"].values[0]} == {2.0, -2.0}

    def test_cell_parameters(self) -> None:
        """Test that cell parameters are properly set when provided."""
        # Test with cell parameters
        cell_params = {
            "SYMMETRY": "CUBIC",
            "MULTIPLE_UNIT_CELL": [2, 2, 2],
        }

        dft_set = DftSet(Si_structure, basis_and_potential=BASIS_AND_POTENTIAL, xc_functionals="PBE", cell=cell_params)

        # Check that cell parameters were properly set
        subsys = dft_set["force_eval"]["subsys"]
        assert subsys.check("cell")
        cell_section = subsys["cell"]
        assert cell_section["symmetry"].values[0] == "CUBIC"
        assert cell_section["multiple_unit_cell"].values == ([2, 2, 2],)

        # Test without cell parameters (default behavior)
        dft_set = DftSet(Si_structure, basis_and_potential=BASIS_AND_POTENTIAL, xc_functionals="PBE")

        # Check that only default cell parameters exist
        subsys = dft_set["force_eval"]["subsys"]
        assert subsys.check("cell")
        cell_section = subsys["cell"]
        assert not cell_section.check("symmetry")
        assert not cell_section.check("multiple_unit_cell")

    def test_no_yaml_unsafe_deprecation_warning(self) -> None:
        """Regression test for https://github.com/materialsproject/pymatgen-core/issues/4444.

        Loading CP2K element data files must not emit a PendingDeprecationWarning
        about ruamel.yaml's deprecated 'unsafe' loader type.
        """
        SETTINGS["PMG_CP2K_DATA_DIR"] = str(LOCAL_CP2K_TEST_DIR)
        basis_and_potential: dict[str, str | dict] = {
            "basis_type": "SZV",
            "potential_type": "Pseudopotential",
            "functional": None,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("error", PendingDeprecationWarning)
            # Constructing DftSet with a data dir triggers YAML loading of the
            # per-element files (e.g. test-files/io/cp2k/Si). The old code used
            # YAML(typ="unsafe") which raises a PendingDeprecationWarning from
            # ruamel.yaml. After the fix this must not raise.
            DftSet(Si_structure, basis_and_potential=basis_and_potential, xc_functionals="PBE")

    def test_yaml_data_loading_roundtrip(self) -> None:
        """CP2K element data files use MSONable-format YAML (no unsafe Python tags).

        Verify that the data can be loaded with the default (safe) ruamel.yaml
        loader and that the resulting dicts reconstruct valid objects.
        """
        from ruamel.yaml import YAML

        si_file = LOCAL_CP2K_TEST_DIR / "Si"
        yaml = YAML()
        with open(si_file, encoding="utf-8") as fh:
            data = yaml.load(fh)

        assert "basis_sets" in data
        assert "potentials" in data

        # Reconstruct a GaussianTypeOrbitalBasisSet from the YAML data
        basis_hash = next(iter(data["basis_sets"]))
        basis = GaussianTypeOrbitalBasisSet.from_dict(data["basis_sets"][basis_hash])
        assert isinstance(basis, GaussianTypeOrbitalBasisSet)
        assert basis.element is not None
        assert basis.name is not None

        # Reconstruct a GthPotential from the YAML data
        pot_hash = next(iter(data["potentials"]))
        potential = GthPotential.from_dict(data["potentials"][pot_hash])
        assert isinstance(potential, GthPotential)
        assert potential.element is not None
        assert potential.name is not None


class TestDftSetExtra(MatSciTest):
    """Coverage tests for the DftSet activation helpers, error paths, and sub-sets."""

    @pytest.fixture(autouse=True)
    def _set_data_dir(self) -> None:
        SETTINGS["PMG_CP2K_DATA_DIR"] = CP2K_TEST_DIR

    @property
    def si_structure(self) -> Structure:
        return Si_structure.copy()

    @property
    def bp(self) -> dict:
        return {"Si": {"basis": "SZV-GTH-q4", "potential": "GTH-PBE-q4"}}

    # ------ kpoints handling in __init__ ------

    def test_kpoints_monkhorst_kept_enables_dos(self) -> None:
        kp = Kpoints(style=KpointsSupportedModes.Monkhorst, kpts=[2, 2, 2])
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", kpoints=kp)
        assert s.check("force_eval/dft/kpoints")
        assert s.check("force_eval/dft/print/dos")

    def test_kpoints_gamma_is_dropped(self) -> None:
        kp = Kpoints(style=KpointsSupportedModes.Gamma, kpts=[1, 1, 1])
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", kpoints=kp)
        assert not s.check("force_eval/dft/kpoints")

    def test_kpoints_with_ot_downgrades_to_diagonalization(self) -> None:
        kp = Kpoints(style=KpointsSupportedModes.Monkhorst, kpts=[2, 2, 2])
        with pytest.warns(UserWarning, match=r"kpoints not supported with OT"):
            s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", kpoints=kp, ot=True)
        # Diagonalization branch is active
        assert s.check("force_eval/dft/scf/diagonalization")

    def test_smearing_activation(self) -> None:
        kp = Kpoints(style=KpointsSupportedModes.Monkhorst, kpts=[2, 2, 2])
        s = DftSet(
            self.si_structure,
            basis_and_potential=self.bp,
            xc_functionals="PBE",
            kpoints=kp,
            ot=False,
            smearing=True,
        )
        assert s.check("force_eval/dft/scf/smear")

    # ------ XC functional name expansion ------

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("LDA", ["PADE"]),
            ("LSDA", ["PADE"]),
            ("SCAN", ["MGGA_X_SCAN", "MGGA_C_SCAN"]),
            ("SCANL", ["MGGA_X_SCANL", "MGGA_C_SCANL"]),
            ("R2SCAN", ["MGGA_X_R2SCAN", "MGGA_C_R2SCAN"]),
            ("R2SCANL", ["MGGA_X_R2SCANL", "MGGA_C_R2SCANL"]),
            ("PBE", ["PBE"]),
        ],
    )
    def test_get_xc_functionals(self, name: str, expected: list[str]) -> None:
        assert DftSet.get_xc_functionals(name) == expected

    def test_get_xc_functionals_missing_raises(self) -> None:
        # No kwarg, no env var, no setting -> raise
        old = SETTINGS.get("PMG_DEFAULT_CP2K_FUNCTIONAL")
        SETTINGS["PMG_DEFAULT_CP2K_FUNCTIONAL"] = None
        try:
            import os

            os.environ.pop("DEFAULT_CP2K_FUNCTIONAL", None)
            with pytest.raises(ValueError, match="No XC functional provided"):
                DftSet.get_xc_functionals(None)
        finally:
            if old is None or old == "":
                del SETTINGS["PMG_DEFAULT_CP2K_FUNCTIONAL"]
            else:
                SETTINGS["PMG_DEFAULT_CP2K_FUNCTIONAL"] = old

    # ------ get_cutoff_from_basis ------

    def test_get_cutoff_from_basis_missing_exponent(self) -> None:
        basis = GaussianTypeOrbitalBasisSet(name="empty", exponents=[])
        with pytest.raises(ValueError, match="missing exponent"):
            DftSet.get_cutoff_from_basis([basis], rel_cutoff=50)

    # ------ print / activation helpers ------

    def test_print_ldos(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        s.print_ldos()
        assert s.check("force_eval/dft/print/pdos")
        # One LDOS per site ("LDOS 1", "LDOS 2", ...)
        sub = s["force_eval"]["dft"]["print"]["PDOS"]
        assert len(sub.subsections) == len(s.structure)

    def test_print_bandstructure_requires_kpoints(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        with pytest.raises(ValueError, match="Kpoints must be provided"):
            s.print_bandstructure()

    def test_print_bandstructure_success(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        s.kpoints = Kpoints(
            style=KpointsSupportedModes.Reciprocal,
            kpts=[[0.3, 0.3, 0.3], [0.6, 0.6, 0.6], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8]],
            labels=["a", "b", "c", "d"],
            coord_type="Reciprocal",
        )
        s.print_bandstructure()
        assert s.check("force_eval/dft/print/band_structure")

    def test_print_bandstructure_via_init_line_mode(self) -> None:
        kp = Kpoints(
            style=KpointsSupportedModes.Line_mode,
            kpts=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
            labels=["Gamma", "M"],
        )
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", kpoints=kp)
        s.print_bandstructure()
        assert s.check("force_eval/dft/print/band_structure")

    def test_print_hirshfeld_and_mulliken(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        s.print_hirshfeld()
        s.print_hirshfeld(on=False)
        s.print_mulliken()
        assert s.check("force_eval/dft/print/hirshfeld")
        assert s.check("force_eval/dft/print/mulliken")

    def test_set_charge(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        s.set_charge(2)
        assert s["force_eval"]["dft"]["CHARGE"].values[0] == 2

    # ------ activation helpers ------

    def test_activate_tddfpt_and_spinspin_and_epr_and_nmr(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        s.activate_tddfpt()
        assert s.check("force_eval/properties/tddfpt")
        s.activate_epr()
        assert s.check("force_eval/properties/linres/epr")
        s.activate_nmr()
        assert s.check("force_eval/properties/linres/nmr")
        s.activate_spinspin()
        assert s.check("force_eval/properties/linres/spinspin")

    def test_activate_polar(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        s.activate_polar()
        assert s.check("force_eval/properties/linres/polar")

    def test_activate_localize(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        s.activate_localize()
        assert s.check("force_eval/properties/linres/localize")
        assert s.check("force_eval/properties/linres/localize/print/loc_restart")

    def test_activate_vdw_potential_pair_and_nonlocal(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        with pytest.warns(UserWarning, match="Reference functional will not be checked"):
            s.activate_vdw_potential("PAIR_POTENTIAL", "DFTD3")
        assert s.check("force_eval/dft/xc/vdw_potential/pair_potential")
        s.activate_vdw_potential("NON_LOCAL", "DFTD3")
        assert s.check("force_eval/dft/xc/vdw_potential/non_local")

    def test_activate_fast_and_robust_and_strict_minimization(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        s.activate_fast_minimization(on=True)
        assert s.check("force_eval/dft/scf/ot")
        s.activate_robust_minimization()
        s.activate_very_strict_minimization()
        # Very strict uses GOLD linesearch
        ot = s["force_eval"]["dft"]["scf"]["OT"]
        assert ot.linesearch == "GOLD"

    def test_activate_vdw_potential_nonlocal_no_reference(self) -> None:
        s = DftSet(
            Structure(self.si_structure.lattice, ["Si"], [[0, 0, 0]]), basis_and_potential=self.bp, xc_functionals="PBE"
        )
        # NON_LOCAL does not require a reference functional
        s.activate_vdw_potential("NON_LOCAL", "DFTD3")
        assert s.check("force_eval/dft/xc/vdw_potential/non_local")

    # ------ activate_motion variants ------

    @pytest.mark.parametrize("run_type", ["GEO_OPT", "CELL_OPT"])
    def test_activate_motion_relaxation(self, run_type: str) -> None:
        s = DftSet(
            self.si_structure,
            basis_and_potential=self.bp,
            xc_functionals="PBE",
            run_type=run_type,
        )
        s.activate_motion(optimizer="BFGS")
        assert s.check("motion")
        assert s.check(f"motion/{run_type.lower()}")

    def test_activate_motion_md(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", run_type="MD")
        s.activate_motion(thermostat="CSVR")
        assert s.check("motion/md")
        assert "TYPE CSVR" in s.get_str()

    def test_project_name(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", project_name="Si_test")
        assert s["GLOBAL"]["PROJECT_NAME"] == Keyword("PROJECT_NAME", "Si_test")

    def test_input_structure_not_mutated(self) -> None:
        struct = self.si_structure
        struct.add_oxidation_state_by_element({"Si": 4})
        DftSet(struct, basis_and_potential=self.bp, xc_functionals="PBE")
        assert struct[0].specie == Species("Si", 4)
        assert struct.site_properties == {}

    def test_activate_motion_cg_optimizer(self) -> None:
        # CG optimizer inserts a LINE_SEARCH subsection under the run-type section
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", run_type="GEO_OPT")
        s.activate_motion(optimizer="CG")
        assert s.check("motion/geo_opt/cg")

    def test_activate_motion_band(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", run_type="BAND")
        s.activate_motion()
        assert s.check("motion/band")

    def test_activate_motion_with_fix_constraints(self) -> None:
        struct = self.si_structure.copy()
        struct.add_site_property("fix", [True, False])
        s = DftSet(struct, basis_and_potential=self.bp, xc_functionals="PBE", run_type="GEO_OPT")
        s.activate_motion()
        assert s.check("motion/constraint")

    # ------ hybrid activation variants ------

    @pytest.mark.parametrize("hf", ["PBE0", "HSE06", "RSH", "UNKNOWN_HF"])
    def test_activate_hybrid_variants(self, hf: str) -> None:
        kwargs = {"hybrid_functional": hf} if hf != "UNKNOWN_HF" else {}
        s = HybridStaticSet(
            structure=self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", activate_hybrid=kwargs
        )
        assert s.check("force_eval/dft/xc/hf")

    def test_activate_hybrid_non_admm(self) -> None:
        bp = {"Si": {"basis": "SZV-GTH-q4", "potential": "GTH-PBE-q4", "aux_basis": "cFIT3"}}
        s = DftSet(self.si_structure, basis_and_potential=bp, xc_functionals="PBE")
        s.activate_hybrid(admm=False)
        # aux basis should have been removed
        assert "aux_basis" not in s.basis_and_potential["Si"]

    def test_activate_hybrid_hse06_structure(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        s.activate_hybrid(hybrid_functional="HSE06")
        # HSE06 uses XWPBE + SHORTRANGE
        assert s.check("force_eval/dft/xc/xc_functional/xwpbe")

    def test_activate_hybrid_rsh_structure(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        s.activate_hybrid(hybrid_functional="RSH")
        assert s.check("force_eval/dft/xc/xc_functional/xwpbe")

    # ------ molecule (non-periodic) path ------

    def test_activate_hybrid_pbe0_molecule_coulomb(self) -> None:
        mol = Molecule(species=["Si"], coords=[[0, 0, 0]])
        s = DftSet(
            mol, basis_and_potential={"Si": {"basis": "SZV-GTH-q4", "potential": "GTH-PBE-q4"}}, xc_functionals="PBE"
        )
        s.activate_hybrid(hybrid_functional="PBE0")
        assert s.check("force_eval/dft/xc/hf")
        assert s.check("force_eval/dft/poisson")

    # ------ quick constructor sub-sets ------

    def test_quick_constructor_subsets(self) -> None:
        for cls in (StaticSet, RelaxSet, CellOptSet, HybridRelaxSet, HybridCellOptSet):
            obj = cls(structure=self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
            assert obj.check("force_eval")

    def test_relax_set_activates_motion(self) -> None:
        s = RelaxSet(structure=self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        assert s.check("motion")

    # ------ GAPW method adds radial/lebedev grids ------

    def test_gapw_adds_radial_and_lebedev_grids(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", qs_method="GAPW")
        kind = s["force_eval"]["subsys"]["Si_1"]
        assert kind["RADIAL_GRID"].values[0] == 200
        assert kind["LEBEDEV_GRID"].values[0] == 80

    # ------ site-properties in create_subsys ------

    def test_create_subsys_site_properties(self) -> None:
        struct = self.si_structure.copy()
        struct.add_site_property("ghost", [None, "true"])
        struct.add_site_property("oxi_state", [0, 0])
        s = DftSet(struct, basis_and_potential=self.bp, xc_functionals="PBE")
        kind = s["force_eval"]["subsys"]["Si_2"]
        assert kind["GHOST"] is not None

    # ------ modify_dft_print_iters ------

    def test_modify_dft_print_iters_bad_value(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        with pytest.raises(ValueError, match="add_list should be"):
            s.modify_dft_print_iters(0, add_last="bogus")

    def test_modify_dft_print_iters_for_md_run_type(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", run_type="MD")
        # MD run type with a DFT PRINT section present should insert EACH/ADD_LAST
        s.modify_dft_print_iters(10, add_last="numeric")
        assert s.check("force_eval/dft/print")

    # ------ vdw reference functional branch on molecule ------

    def test_create_subsys_molecule_cell_nonperiodic(self) -> None:
        mol = Molecule(species=["Si", "Si"], coords=[[0, 0, 0], [1.5, 0.0, 0.0]])
        s = DftSet(
            mol, basis_and_potential={"Si": {"basis": "SZV-GTH-q4", "potential": "GTH-PBE-q4"}}, xc_functionals="PBE"
        )
        # Cell is auto-generated and non-periodic
        cell = s["force_eval"]["subsys"]["CELL"]
        assert cell["PERIODIC"].values[0].upper() == "NONE"

    # ------ error paths in get_basis_and_potential via DATA ------

    def test_multiple_potential_filenames_raises(self) -> None:
        # Two elements each mapping to a different potential file -> raise
        bp = {
            "Si": {"basis": "SZV-GTH-q4", "potential": "GTH-PBE-q4", "potential_filename": "a.inp"},
            "Si2": {"basis": "SZV-GTH-q4", "potential": "GTH-PBE-q4", "potential_filename": "b.inp"},
        }
        struct = Structure(self.si_structure.lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        with pytest.raises(ValueError, match="more than one corresponding file"):
            DftSet.get_basis_and_potential(struct, bp)

    # ------ no basis matching fallback ----------

    def test_no_explicit_basis_match_falls_back_to_name(self) -> None:
        # Provide a basis name that does not match any in DATA but is non-empty
        # -> falls back to using the raw name (with a warning), in get_basis_and_potential
        with pytest.warns(UserWarning, match="Unable to validate basis"):
            data = DftSet.get_basis_and_potential(
                self.si_structure, {"Si": {"basis": "NON-EXISTENT-BASIS-NAME", "potential": "GTH-PBE-q4"}}
            )
        assert data["Si"]["basis"] == "NON-EXISTENT-BASIS-NAME"

    # ------ validate: all-electron requires GAPW ------

    def test_validate_all_electron_requires_gapw(self) -> None:
        from pymatgen.io.cp2k.inputs import Keyword

        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        # Patch the potential keyword on a Kind to "ALL" to trigger the validation
        s["force_eval"]["subsys"]["Si_1"]["POTENTIAL"] = Keyword("POTENTIAL", "ALL")
        with pytest.raises(Cp2kValidationError, match="All electron basis sets require GAPW"):
            s.validate()

    # ------ validate: hf + kpoints forbidden ------

    def test_validate_hf_kpoints_conflict(self) -> None:
        kp = Kpoints(style=KpointsSupportedModes.Line_mode, kpts=[[0, 0, 0], [0.5, 0.5, 0]], labels=["A", "B"])
        s = DftSet(self.si_structure, basis_and_potential=self.bp, kpoints=kp, xc_functionals="PBE")
        s.activate_hybrid(hybrid_functional="PBE0")
        with pytest.raises(Cp2kValidationError, match="Does not support hartree fock with kpoints"):
            s.validate()

    # ------ remaining branch coverage ------

    def test_get_cutoff_from_basis_all_electron_branch(self) -> None:
        # All-electron branches use a hard radius default (lines 606-610)
        # H -> radius 1.2
        h_basis = GaussianTypeOrbitalBasisSet.from_str("H DZVP-DZVP-DZVP-q1\n1\n1 0 0 1 0 0\n 1.0 0.5 0.1\n")
        h_basis.potential = "All Electron"
        assert DftSet.get_cutoff_from_basis([h_basis], rel_cutoff=50) > 0
        # non-H (O) -> radius 1.512
        o_basis = GaussianTypeOrbitalBasisSet.from_str(
            "O DZVP-DZVP-DZVP\n1\n2 1 1 3 0 0\n 2.0 0.01 0.1\n 1.0 0.001 0.05\n 0.5 0.0001 0.02\n"
        )
        o_basis.potential = "All Electron"
        assert DftSet.get_cutoff_from_basis([o_basis], rel_cutoff=50) > 0

    def test_unknown_hybrid_functional_warnings(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        with pytest.warns(UserWarning, match="Unknown hybrid functional"):
            s.activate_hybrid(hybrid_functional="NOT_A_REAL_HF", potential_type="COULOMB")
        assert s.check("force_eval/dft/xc/hf")

    def test_activate_spinspin_standalone_activates_localize(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        s.activate_spinspin()
        # spinspin requires localize, which gets activated
        assert s.check("force_eval/properties/linres/localize")
        assert s.check("force_eval/properties/linres/spinspin")

    def test_create_subsys_site_properties_basis_potential_aux(self) -> None:
        struct = self.si_structure.copy()
        struct.add_site_property("basis_set", ["SZV-GTH-q4", "SZV-GTH-q4"])
        struct.add_site_property("potential", ["GTH-PBE-q4", "GTH-PBE-q4"])
        struct.add_site_property("aux_basis", [None, None])
        struct.add_site_property("oxi_state", [4, 4])
        s = DftSet(struct, basis_and_potential=self.bp, xc_functionals="PBE")
        kind = s["force_eval"]["subsys"]["Si_1"]
        assert kind["BASIS_SET"] is not None
        assert kind["POTENTIAL"] is not None
        # broken-symmetry block present for a non-zero oxidation state
        assert kind.subsections is not None

    def test_no_explicit_basis_raises(self) -> None:
        # Element with no data file and no explicit basis -> ValueError
        struct = Structure(self.si_structure.lattice, ["O"], [[0, 0, 0]])
        with pytest.raises(ValueError, match="No explicit basis found for O"):
            DftSet.get_basis_and_potential(struct, {})

    def test_no_explicit_potential_raises(self) -> None:
        # Explicit basis but no potential and no matching potential -> raise
        struct = Structure(self.si_structure.lattice, ["O"], [[0, 0, 0]])
        with pytest.raises(ValueError, match="No explicit potential found"):
            DftSet.get_basis_and_potential(struct, {"O": {"basis": "SZV-GTH-q4"}})

    def test_aux_basis_fallback_warns(self) -> None:
        # aux_basis name that does not match a data entry -> fallback warning
        with pytest.warns(UserWarning, match="Unable to validate auxiliary basis"):
            DftSet(
                self.si_structure,
                basis_and_potential={
                    "Si": {"basis": "SZV-GTH-q4", "potential": "GTH-PBE-q4", "aux_basis": "NONEXIST-AUX"}
                },
                xc_functionals="PBE",
            )

    def test_aux_basis_by_object(self) -> None:
        aux = GaussianTypeOrbitalBasisSet.from_str(
            "Si SZV-MOLOPT-GTH SZV-MOLOPT-GTH-q4\n1\n2 0 1 6 1 1\n"
            " 2.693604434572  0.015333179500 -0.005800105400\n"
            " 1.359613855428 -0.283798205000 -0.059172026000\n"
            " 0.513245176029 -0.228939692700  0.121487149900\n"
            " 0.326563011394  0.728834000900  0.423382421100\n"
            " 0.139986977410  0.446205299300  0.474592116300\n"
            " 0.068212286977  0.122025292800  0.250129397700\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            DftSet(
                self.si_structure,
                basis_and_potential={"Si": {"basis": "SZV-GTH-q4", "potential": "GTH-PBE-q4", "aux_basis": aux}},
                xc_functionals="PBE",
            )

    def test_write_file_helpers(self, tmp_path) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE")
        # Grab a real basis object from the constructed set
        basis_obj = s.basis_and_potential["Si"]["basis"]
        potential_obj = s.basis_and_potential["Si"]["potential"]
        out = tmp_path / "basis.set"
        s.write_basis_set_file([basis_obj], str(out))
        assert out.exists()
        out2 = tmp_path / "potents.inp"
        s.write_potential_file([potential_obj], str(out2))
        assert out2.exists()

    def test_print_ldos_via_kwarg(self) -> None:
        s = DftSet(self.si_structure, basis_and_potential=self.bp, xc_functionals="PBE", print_ldos=True)
        assert s.check("force_eval/dft/print/pdos")

    def test_print_bandstructure_via_kwarg(self) -> None:
        kp = Kpoints(
            style=KpointsSupportedModes.Reciprocal,
            kpts=[[0.3, 0.3, 0.3], [0.6, 0.6, 0.6], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8]],
            labels=["a", "b", "c", "d"],
            coord_type="Reciprocal",
        )
        s = DftSet(
            self.si_structure,
            basis_and_potential=self.bp,
            xc_functionals="PBE",
            kpoints=kp,
            print_bandstructure=True,
            kpoints_line_density=10,
        )
        assert s.check("force_eval/dft/print/band_structure")

    def test_get_basis_and_potential_missing_basis_sets_in_data_raises(self, tmp_path) -> None:
        # Element file exists but has no "basis_sets" -> raise
        from ruamel.yaml import YAML

        bad_dir = tmp_path / "bad_data"
        bad_dir.mkdir()
        yaml = YAML()
        with open(bad_dir / "O", "w", encoding="utf-8") as fh:
            yaml.dump({"potentials": {}}, fh)
        struct = Structure(self.si_structure.lattice, ["O"], [[0, 0, 0]])
        with pytest.raises(ValueError, match="No standard basis sets available"):
            DftSet.get_basis_and_potential(struct, {}, cp2k_data_dir=str(bad_dir))

    def test_get_basis_and_potential_missing_potentials_in_data_raises(self, tmp_path) -> None:
        # Element file has basis_sets but no "potentials" -> raise
        from ruamel.yaml import YAML

        bad_dir = tmp_path / "bad_data_pot"
        bad_dir.mkdir()
        yaml = YAML()
        with open(bad_dir / "O", "w", encoding="utf-8") as fh:
            yaml.dump({"basis_sets": {"abc": {"name": "X"}}, "potentials": None}, fh)
        struct = Structure(self.si_structure.lattice, ["O"], [[0, 0, 0]])
        with pytest.raises(ValueError, match="No standard potentials available"):
            DftSet.get_basis_and_potential(struct, {}, cp2k_data_dir=str(bad_dir))
