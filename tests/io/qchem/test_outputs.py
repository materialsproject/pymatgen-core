from __future__ import annotations

import gzip
import shutil

import numpy as np
import pytest
from monty.serialization import dumpfn, loadfn
from numpy.testing import assert_allclose
from pytest import approx

from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.structure import Molecule
from pymatgen.io.qchem.outputs import (
    QCOutput,
    check_for_structure_changes,
    gradient_parser,
    hessian_parser,
    orbital_coeffs_parser,
)
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

try:
    from openbabel import openbabel
except ImportError:
    openbabel = None

TEST_DIR = f"{TEST_FILES_DIR}/io/qchem"
NEW_QCHEM_TEST_DIR = f"{TEST_DIR}/new_qchem_files"


__author__ = "Samuel Blau, Brandon Wood, Shyam Dwaraknath, Evan Spotte-Smith, Ryan Kingsbury"
__copyright__ = "Copyright 2018-2022, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Samuel Blau"
__email__ = "samblau1@gmail.com"

SINGLE_JOB_DICT = loadfn(f"{TEST_DIR}/single_job.json")
MULTI_JOB_DICT = loadfn(f"{TEST_DIR}/multi_job.json")

PROPERTIES = {
    "errors",
    "multiple_outputs",
    "completion",
    "unrestricted",
    "using_GEN_SCFMAN",
    "final_energy",
    "S2",
    "optimization",
    "energy_trajectory",
    "opt_constraint",
    "frequency_job",
    "charge",
    "multiplicity",
    "species",
    "initial_geometry",
    "initial_molecule",
    "SCF",
    "Mulliken",
    "optimized_geometry",
    "optimized_zmat",
    "molecule_from_optimized_geometry",
    "last_geometry",
    "molecule_from_last_geometry",
    "geometries",
    "gradients",
    "frequency_mode_vectors",
    "walltime",
    "cputime",
    "point_group",
    "frequencies",
    "IR_intens",
    "IR_active",
    "g_electrostatic",
    "g_cavitation",
    "g_dispersion",
    "g_repulsion",
    "total_contribution_pcm",
    "ZPE",
    "trans_enthalpy",
    "vib_enthalpy",
    "rot_enthalpy",
    "gas_constant",
    "trans_entropy",
    "vib_entropy",
    "rot_entropy",
    "total_entropy",
    "total_enthalpy",
    "warnings",
    "SCF_energy_in_the_final_basis_set",
    "Total_energy_in_the_final_basis_set",
    "solvent_method",
    "solvent_data",
    "using_dft_d3",
    "single_point_job",
    "force_job",
    "pcm_gradients",
    "CDS_gradients",
    "RESP",
    "trans_dip",
    "transition_state",
    "scan_job",
    "optimized_geometries",
    "molecules_from_optimized_geometries",
    "scan_energies",
    "scan_constraint_sets",
    "hf_scf_energy",
    "mp2_energy",
    "ccsd_correlation_energy",
    "ccsd_total_energy",
    "ccsd(t)_correlation_energy",
    "ccsd(t)_total_energy",
    "cdft_becke_excess_electrons",
    "cdft_becke_population",
    "cdft_becke_net_spin",
    "direct_coupling_Hif_Hartree",
    "direct_coupling_Sif_Hartree",
    "direct_coupling_Hii_Hartree",
    "direct_coupling_Sii_Hartree",
    "direct_coupling_Hff_Hartree",
    "direct_coupling_Sff_Hartree",
    "direct_coupling_eV",
    "almo_coupling_states",
    "almo_diabat_energies_Hartree",
    "almo_adiabat_energies_Hartree",
    "almo_hamiltonian",
    "almo_overlap_matrix",
    "almo_s2_matrix",
    "almo_diabat_basis_coeff",
    "almo_h_coupling_matrix",
    "almo_coupling_eV",
    "pod_coupling_eV",
    "fodft_had_eV",
    "fodft_hda_eV",
    "fodft_coupling_eV",
    "alpha_fock_matrix",
    "beta_fock_matrix",
    "alpha_eigenvalues",
    "beta_eigenvalues",
    "alpha_coeff_matrix",
    "beta_coeff_matrix",
    "final_soln_phase_e",
    "solute_internal_e",
    "total_solvation_free_e",
    "change_solute_internal_e",
    "reaction_field_free_e",
    "isosvp_dielectric",
    "dispersion_e",
    "exchange_e",
    "min_neg_field_e",
    "max_pos_field_e",
    "norm_of_stepsize",
    "version",
    "dipoles",
    "multipoles",
    "gap_info",
}

if openbabel is not None:
    PROPERTIES.add("structure_change")

SINGLE_JOB_OUT_NAMES = {
    "unable_to_determine_lambda_in_geom_opt.qcout",
    "thiophene_wfs_5_carboxyl.qcout",
    "hf.qcout",
    "hf_opt_failed.qcout",
    "no_reading.qcout",
    "exit_code_134.qcout",
    "negative_eigen.qcout",
    "insufficient_memory.qcout",
    "freq_seg_too_small.qcout",
    "crowd_gradient_number.qcout",
    "quinoxaline_anion.qcout",
    "tfsi_nbo.qcout",
    "crowd_nbo_charges.qcout",
    "h2o_aimd.qcout",
    "bsse.qcout",
    "time_nan_values.qcout",
    "pt_dft_180.0.qcout",
    "qchem_energies/hf-rimp2.qcout",
    "qchem_energies/hf_b3lyp.qcout",
    "qchem_energies/hf_ccsd(t).qcout",
    "qchem_energies/hf_cosmo.qcout",
    "qchem_energies/hf_hf.qcout",
    "qchem_energies/hf_lxygjos.qcout",
    "qchem_energies/hf_mosmp2.qcout",
    "qchem_energies/hf_mp2.qcout",
    "qchem_energies/hf_qcisd(t).qcout",
    "qchem_energies/hf_riccsd(t).qcout",
    "qchem_energies/hf_tpssh.qcout",
    "qchem_energies/hf_xyg3.qcout",
    "qchem_energies/hf_xygjos.qcout",
    "qchem_energies/hf_wb97xd_gen_scfman.qcout",
    "new_qchem_files/pt_n2_n_wb_180.0.qcout",
    "new_qchem_files/pt_n2_trip_wb_90.0.qcout",
    "new_qchem_files/pt_n2_gs_rimp2_pvqz_90.0.qcout",
    "new_qchem_files/VC_solv_eps10.2.qcout",
    "crazy_scf_values.qcout",
    "new_qchem_files/N2.qcout",
    "new_qchem_files/julian.qcout.gz",
    "new_qchem_files/Frequency_no_equal.qout",
    "new_qchem_files/gdm.qout",
    "new_qchem_files/DinfH.qout",
    "new_qchem_files/mpi_error.qout",
    "new_qchem_files/molecule_read_error.qout",
    "new_qchem_files/basis_not_supported.qout",
    "new_qchem_files/lebdevpts.qout",
    "new_qchem_files/Optimization_no_equal.qout",
    "new_qchem_files/2068.qout",
    "new_qchem_files/2620.qout",
    "new_qchem_files/1746.qout",
    "new_qchem_files/1570.qout",
    "new_qchem_files/1570_2.qout",
    "new_qchem_files/single_point.qout",
    "new_qchem_files/roothaan_diis_gdm.qout",
    "new_qchem_files/pes_scan_single_variable.qout",
    "new_qchem_files/pes_scan_double_variable.qout",
    "new_qchem_files/ts.out",
    "new_qchem_files/ccsd.qout",
    "new_qchem_files/ccsdt.qout",
    "new_qchem_files/almo.out",
    "new_qchem_files/cdft_simple.qout",
    "new_qchem_files/fodft.out",
    "new_qchem_files/fodft_2.out",
    "new_qchem_files/fodft_3.out",
    "new_qchem_files/pod1.out",
    "new_qchem_files/pod2_gs.out",
    "extra_scf_print.qcout",
    "new_qchem_files/cmirs_benzene_single.qcout",
    "new_qchem_files/cmirs_dielst10_single.qcout",
    "new_qchem_files/cmirs_water_single.qcout",
    "new_qchem_files/isosvp_water_single.qcout",
    "new_qchem_files/isosvp_dielst10_single.qcout",
    "new_qchem_files/custom_gdm_gdmqls_opt.qout",
    "new_qchem_files/unable.qout",
    "new_qchem_files/unexpected_ts.out",
    "new_qchem_files/svd_failed.qout",
    "new_qchem_files/v6_old_driver.out",
    "new_qchem_files/gap.qout",
    "new_qchem_files/3C.qout",
    "new_qchem_files/hyper.qout",
    "new_qchem_files/os_gap.qout",
}

MULTI_JOB_OUT_NAMES = {
    "not_enough_total_memory.qcout",
    "new_qchem_files/VC_solv_eps10.qcout",
    "new_qchem_files/MECLi_solv_eps10.qcout",
    "pcm_solvent_deprecated.qcout",
    "qchem43_batch_job.qcout",
    "ferrocenium_1pos.qcout",
    "CdBr2.qcout",
    "killed.qcout",
    "aux_mpi_time_mol.qcout",
    "new_qchem_files/VCLi_solv_eps10.qcout",
    "new_qchem_files/cdft_dc.qout",
    "new_qchem_files/cmirs_benzene.qcout",
    "new_qchem_files/cmirs_dielst10.qcout",
    "new_qchem_files/isosvp_water.qcout",
    "new_qchem_files/isosvp_dielst10.qcout",
}


# NBO-heavy outputs that take >20 s each to parse and already have dedicated tests below
SLOW_OUT_NAMES = {"new_qchem_files/3C.qout", "new_qchem_files/hyper.qout"}


def _assert_matches_reference(parsed, ref, path="data"):
    """Recursively compare parsed QCOutput data against the JSON reference.

    JSON round-tripping turns integer dict keys into strings and tuples into lists,
    so keys are compared via str() and sequences are compared element-wise.
    """
    if isinstance(ref, dict):
        assert isinstance(parsed, dict), path
        assert {str(key) for key in parsed} == {str(key) for key in ref}, path
        ref_by_str = {str(key): val for key, val in ref.items()}
        for key, val in parsed.items():
            _assert_matches_reference(val, ref_by_str[str(key)], f"{path}[{key!r}]")
    elif isinstance(ref, list | tuple | np.ndarray) and not isinstance(ref, str):
        assert len(parsed) == len(ref), path
        try:
            assert_allclose(np.asarray(parsed, dtype=float), np.asarray(ref, dtype=float), atol=1e-6, err_msg=path)
        except (ValueError, TypeError):
            for idx, (sub_parsed, sub_ref) in enumerate(zip(parsed, ref, strict=True)):
                _assert_matches_reference(sub_parsed, sub_ref, f"{path}[{idx}]")
    elif isinstance(ref, float):
        assert parsed == approx(ref, nan_ok=True), path
    elif isinstance(ref, Molecule):
        assert parsed == ref, path
    else:
        assert parsed == ref, path


@pytest.mark.parametrize("filename", sorted(SINGLE_JOB_OUT_NAMES - SLOW_OUT_NAMES))
def test_single_job_matches_reference(filename):
    data = QCOutput(f"{TEST_DIR}/{filename}").data
    ref = SINGLE_JOB_DICT[filename]
    for key, ref_val in ref.items():
        if key == "structure_change":  # requires OpenBabel
            continue
        _assert_matches_reference(data.get(key), ref_val, key)


@pytest.mark.parametrize("filename", sorted(MULTI_JOB_OUT_NAMES))
def test_multi_job_matches_reference(filename, tmp_path):
    local_file = tmp_path / filename.split("/")[-1]
    shutil.copy(f"{TEST_DIR}/{filename}", local_file)
    outputs = QCOutput.multiple_outputs_from_file(str(local_file), keep_sub_files=False)
    refs = MULTI_JOB_DICT[filename]
    assert len(outputs) == len(refs)
    for idx, (output, ref) in enumerate(zip(outputs, refs, strict=True)):
        assert not (tmp_path / f"{local_file.name}.{idx}").exists()
        for key, ref_val in ref.items():
            if key == "structure_change":
                continue
            _assert_matches_reference(output.data.get(key), ref_val, f"{filename}[{idx}].{key}")


class TestQCOutput(MatSciTest):
    @staticmethod
    def generate_single_job_dict():
        """Used to generate test dictionary for single jobs."""
        single_job_dict = {}
        for file in SINGLE_JOB_OUT_NAMES:
            single_job_dict[file] = QCOutput(f"{TEST_DIR}/{file}").data
        dumpfn(single_job_dict, "single_job.json")

    @staticmethod
    def generate_multi_job_dict():
        """Used to generate test dictionary for multiple jobs."""
        multi_job_dict = {}
        for file in MULTI_JOB_OUT_NAMES:
            outputs = QCOutput.multiple_outputs_from_file(f"{TEST_DIR}/{file}", keep_sub_files=False)
            multi_job_dict[file] = [sub_output.data for sub_output in outputs]
        dumpfn(multi_job_dict, "multi_job.json")

    def _check_property(self, key, single_outs, multi_outs):
        for filename, out_data in single_outs.items():
            try:
                assert out_data.get(key) == SINGLE_JOB_DICT[filename].get(key)
            except ValueError:
                try:
                    if isinstance(out_data.get(key), dict):
                        assert out_data.get(key) == approx(SINGLE_JOB_DICT[filename].get(key))
                    else:
                        assert_allclose(
                            out_data.get(key),
                            SINGLE_JOB_DICT[filename].get(key),
                            atol=1e-6,
                        )
                except AssertionError as exc:
                    raise RuntimeError(f"Issue with {filename=} Exiting...") from exc
            except AssertionError as exc:
                raise RuntimeError(f"Issue with {filename=} Exiting...") from exc

        for filename, outputs in multi_outs.items():
            for idx, sub_output in enumerate(outputs):
                try:
                    assert sub_output.data.get(key) == MULTI_JOB_DICT[filename][idx].get(key)
                except ValueError:
                    if isinstance(sub_output.data.get(key), dict):
                        assert sub_output.data.get(key) == approx(MULTI_JOB_DICT[filename][idx].get(key))
                    else:
                        assert_allclose(
                            sub_output.data.get(key),
                            MULTI_JOB_DICT[filename][idx].get(key),
                            atol=1e-6,
                        )

    # PR#3985: the following unit test is failing, and it seems that
    # the array dimension from out_data and SINGLE_JOB_DICT mismatch
    @pytest.mark.xfail(reason="TODO: need someone to fix this")
    @pytest.mark.skipif(openbabel is None, reason="OpenBabel not installed.")
    def test_all(self):
        single_outs = {file: QCOutput(f"{TEST_DIR}/{file}").data for file in SINGLE_JOB_OUT_NAMES}

        multi_outs = {
            file: QCOutput.multiple_outputs_from_file(f"{TEST_DIR}/{file}", keep_sub_files=False)
            for file in MULTI_JOB_OUT_NAMES
        }

        for key in PROPERTIES:
            self._check_property(key, single_outs, multi_outs)

    def test_multipole_parsing(self):
        sp = QCOutput(f"{NEW_QCHEM_TEST_DIR}/nbo.qout")

        mpoles = sp.data["multipoles"]
        assert len(mpoles["quadrupole"]) == 6
        assert len(mpoles["octopole"]) == 10
        assert len(mpoles["hexadecapole"]) == 15
        assert mpoles["quadrupole"]["XX"] == approx(-51.3957)
        assert mpoles["quadrupole"]["YZ"] == approx(3.5356)
        assert mpoles["octopole"]["XYY"] == approx(-15.0294)
        assert mpoles["octopole"]["XZZ"] == approx(-14.9756)
        assert mpoles["hexadecapole"]["YYYY"] == approx(-326.317)
        assert mpoles["hexadecapole"]["XYZZ"] == approx(58.0584)

        opt = QCOutput(f"{NEW_QCHEM_TEST_DIR}/ts.out")
        mpoles = opt.data["multipoles"]

        assert len(mpoles["quadrupole"]) == 5
        assert len(mpoles["octopole"]) == 5
        assert len(mpoles["hexadecapole"]) == 5

    @pytest.mark.skipif(openbabel is None, reason="OpenBabel not installed.")
    def test_structural_change(self):
        t1 = Molecule.from_file(f"{TEST_FILES_DIR}/analysis/structural_change/t1.xyz")
        t2 = Molecule.from_file(f"{TEST_FILES_DIR}/analysis/structural_change/t2.xyz")
        t3 = Molecule.from_file(f"{TEST_FILES_DIR}/analysis/structural_change/t3.xyz")

        thio_1 = Molecule.from_file(f"{TEST_FILES_DIR}/analysis/structural_change/thiophene1.xyz")
        thio_2 = Molecule.from_file(f"{TEST_FILES_DIR}/analysis/structural_change/thiophene2.xyz")

        frag_1 = Molecule.from_file(f"{NEW_QCHEM_TEST_DIR}/test_structure_change/frag_1.xyz")
        frag_2 = Molecule.from_file(f"{NEW_QCHEM_TEST_DIR}/test_structure_change/frag_2.xyz")

        assert check_for_structure_changes(t1, t1) == "no_change"
        assert check_for_structure_changes(t2, t3) == "no_change"
        assert check_for_structure_changes(t1, t2) == "fewer_bonds"
        assert check_for_structure_changes(t2, t1) == "more_bonds"

        assert check_for_structure_changes(thio_1, thio_2) == "unconnected_fragments"

        assert check_for_structure_changes(frag_1, frag_2) == "bond_change"

    def test_nbo_parsing(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/nbo.qout").data
        assert len(data["nbo_data"]["natural_populations"]) == 3
        assert len(data["nbo_data"]["hybridization_character"]) == 6
        assert len(data["nbo_data"]["perturbation_energy"]) == 2
        assert data["nbo_data"]["natural_populations"][0]["Density"][5] == approx(-0.08624)
        assert data["nbo_data"]["hybridization_character"][4]["atom 2 pol coeff"][35] == "-0.7059"
        next_to_last = list(data["nbo_data"]["perturbation_energy"][-1]["fock matrix element"])[-2]
        assert data["nbo_data"]["perturbation_energy"][-1]["fock matrix element"][next_to_last] == approx(0.071)
        assert data["nbo_data"]["perturbation_energy"][0]["acceptor type"][0] == "RY*"

    def test_nbo7_parsing(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/nbo7_1.qout").data
        assert data["nbo_data"]["perturbation_energy"][0]["perturbation energy"][9] == approx(15.73)
        assert len(data["nbo_data"]["perturbation_energy"][0]["donor bond index"]) == 84
        assert len(data["nbo_data"]["perturbation_energy"][1]["donor bond index"]) == 29

        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/nbo7_2.qout").data
        assert data["nbo_data"]["perturbation_energy"][0]["perturbation energy"][13] == approx(32.93)
        assert data["nbo_data"]["perturbation_energy"][0]["acceptor type"][13] == "LV"
        assert data["nbo_data"]["perturbation_energy"][0]["acceptor type"][12] == "RY"
        assert data["nbo_data"]["perturbation_energy"][0]["acceptor atom 1 symbol"][12] == "Mg"

        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/nbo7_3.qout").data
        assert data["nbo_data"]["perturbation_energy"][0]["perturbation energy"][13] == approx(34.54)
        assert data["nbo_data"]["perturbation_energy"][0]["acceptor type"][13] == "BD*"
        assert data["nbo_data"]["perturbation_energy"][0]["acceptor atom 1 symbol"][13] == "B"
        assert data["nbo_data"]["perturbation_energy"][0]["acceptor atom 2 symbol"][13] == "Mg"
        assert data["nbo_data"]["perturbation_energy"][0]["acceptor atom 2 number"][13] == 3

    def test_nbo5_vs_nbo7_hybridization_character(self):
        data5 = QCOutput(f"{NEW_QCHEM_TEST_DIR}/nbo5_1.qout").data
        data7 = QCOutput(f"{NEW_QCHEM_TEST_DIR}/nbo7_1.qout").data
        assert len(data5["nbo_data"]["hybridization_character"]) == len(data7["nbo_data"]["hybridization_character"])
        assert (
            data5["nbo_data"]["hybridization_character"][4]["atom 2 pol coeff"][9]
            == data7["nbo_data"]["hybridization_character"][4]["atom 2 pol coeff"][9]
        )
        assert (
            data5["nbo_data"]["hybridization_character"][0]["s"][0]
            == data7["nbo_data"]["hybridization_character"][0]["s"][0]
        )
        assert data5["nbo_data"]["hybridization_character"][1]["bond index"][7] == "149"
        assert data7["nbo_data"]["hybridization_character"][1]["bond index"][7] == "21"

    def test_nbo7_infinite_e2pert(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/nbo7_inf.qout").data
        assert data["nbo_data"]["perturbation_energy"][0]["perturbation energy"][0] == float("inf")

    def test_cdft_parsing(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/cdft_simple.qout").data
        assert data["cdft_becke_excess_electrons"][0][0] == approx(0.432641)
        assert len(data["cdft_becke_population"][0]) == 12
        assert data["cdft_becke_net_spin"][0][6] == approx(-0.000316)

    def test_cdft_dc_parsing(self):
        data = QCOutput.multiple_outputs_from_file(
            f"{NEW_QCHEM_TEST_DIR}/cdft_dc.qout",
            keep_sub_files=False,
        )[-1].data
        assert data["direct_coupling_eV"] == approx(0.0103038246)

    def test_almo_msdft2_parsing(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/almo.out").data
        assert data["almo_coupling_states"] == [[[1, 2], [0, 1]], [[0, 1], [1, 2]]]
        assert data["almo_hamiltonian"][0][0] == approx(-156.62929)
        assert data["almo_coupling_eV"] == approx(0.26895)

    def test_pod_parsing(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/pod2_gs.out").data
        assert data["pod_coupling_eV"] == approx(0.247818)

    def test_fodft_parsing(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/fodft.out").data
        assert data["fodft_coupling_eV"] == approx(0.268383)

    def test_isosvp_water(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/isosvp_water_single.qcout").data
        assert data["solvent_method"] == "ISOSVP"
        # ISOSVP parameters
        assert data["solvent_data"]["isosvp"]["isosvp_dielectric"] == approx(78.39)
        assert data["solvent_data"]["isosvp"]["final_soln_phase_e"] == approx(-40.4850599393)
        assert data["solvent_data"]["isosvp"]["solute_internal_e"] == approx(-40.4846329762)
        assert data["solvent_data"]["isosvp"]["change_solute_internal_e"] == approx(0.0000121967)
        assert data["solvent_data"]["isosvp"]["reaction_field_free_e"] == approx(-0.0004269631)
        assert data["solvent_data"]["isosvp"]["total_solvation_free_e"] == approx(-0.0004147664)

        # CMIRS parameters
        assert data["solvent_data"]["cmirs"]["CMIRS_enabled"] is False

    def test_isosvp_dielst10(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/isosvp_dielst10_single.qcout").data
        assert data["solvent_method"] == "ISOSVP"

        # ISOSVP parameters
        assert data["solvent_data"]["isosvp"]["isosvp_dielectric"] == 10
        assert data["solvent_data"]["isosvp"]["final_soln_phase_e"] == approx(-40.4850012952)
        assert data["solvent_data"]["isosvp"]["solute_internal_e"] == approx(-40.4846362547)
        assert data["solvent_data"]["isosvp"]["change_solute_internal_e"] == approx(0.0000089182)
        assert data["solvent_data"]["isosvp"]["reaction_field_free_e"] == approx(-0.0003650405)
        assert data["solvent_data"]["isosvp"]["total_solvation_free_e"] == approx(-0.0003561223)

        # CMIRS parameters
        assert data["solvent_data"]["cmirs"]["CMIRS_enabled"] is False

    def test_cmirs_benzene(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/cmirs_benzene_single.qcout").data
        assert data["solvent_method"] == "ISOSVP"
        assert data["solvent_data"]["isosvp"]["isosvp_dielectric"] == approx(2.28)
        assert data["solvent_data"]["cmirs"]["CMIRS_enabled"]
        assert data["solvent_data"]["cmirs"]["dispersion_e"] == approx(0.6955542829)
        assert data["solvent_data"]["cmirs"]["exchange_e"] == approx(0.2654553686)
        assert data["solvent_data"]["cmirs"]["min_neg_field_e"] == approx(0.0006019665)
        assert data["solvent_data"]["cmirs"]["max_pos_field_e"] == approx(0.0178177740)

    def test_cmirs_dielst10(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/cmirs_dielst10_single.qcout").data
        assert data["solvent_method"] == "ISOSVP"
        assert data["solvent_data"]["isosvp"]["isosvp_dielectric"] == 10
        assert data["solvent_data"]["cmirs"]["CMIRS_enabled"]
        assert data["solvent_data"]["cmirs"]["dispersion_e"] == approx(0.6955550107)
        assert data["solvent_data"]["cmirs"]["exchange_e"] == approx(0.2652679507)
        assert data["solvent_data"]["cmirs"]["min_neg_field_e"] == approx(0.0005235850)
        assert data["solvent_data"]["cmirs"]["max_pos_field_e"] == approx(0.0179866718)

    def test_cmirs_water(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/cmirs_water_single.qcout").data
        assert data["solvent_method"] == "ISOSVP"

        # ISOSVP parameters
        assert data["solvent_data"]["isosvp"]["isosvp_dielectric"] == approx(78.39)
        assert data["solvent_data"]["isosvp"]["final_soln_phase_e"] == approx(-40.4752415075)
        assert data["solvent_data"]["isosvp"]["solute_internal_e"] == approx(-40.4748535587)
        assert data["solvent_data"]["isosvp"]["change_solute_internal_e"] == approx(0.0000122982)
        assert data["solvent_data"]["isosvp"]["reaction_field_free_e"] == approx(-0.0003879488)
        assert data["solvent_data"]["isosvp"]["total_solvation_free_e"] == approx(0.0037602703)

        # CMIRS parameters
        assert data["solvent_data"]["cmirs"]["CMIRS_enabled"]
        assert data["solvent_data"]["cmirs"]["dispersion_e"] == approx(0.6722278965)
        assert data["solvent_data"]["cmirs"]["exchange_e"] == approx(0.2652032616)
        assert data["solvent_data"]["cmirs"]["min_neg_field_e"] == approx(0.0004967767)
        assert data["solvent_data"]["cmirs"]["max_pos_field_e"] == approx(0.0180445935)

    def test_nbo_hyperbonds(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/hyper.qout").data
        assert len(data["nbo_data"]["hyperbonds"][0]["hyperbond index"]) == 2
        assert data["nbo_data"]["hyperbonds"][0]["BD(A-B)"][1] == 106
        assert data["nbo_data"]["hyperbonds"][0]["bond atom 2 symbol"][0] == "C"
        assert data["nbo_data"]["hyperbonds"][0]["occ"][1] == approx(3.0802)

    def test_nbo_3_c(self):
        data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/3C.qout").data
        hybrid_char = data["nbo_data"]["hybridization_character"]
        assert len(hybrid_char) == 3
        hybrid_type = hybrid_char[2]["type"]
        assert hybrid_type[0] == "3C"
        assert hybrid_type[10] == "3Cn"
        assert hybrid_type[20] == "3C*"
        assert hybrid_char[2]["atom 3 pol coeff"][15] == "0.3643"
        assert hybrid_char[2]["atom 3 polarization"][8] == "56.72"
        assert hybrid_char[2]["atom 3 symbol"][3] == "B"
        perturb_ene = data["nbo_data"]["perturbation_energy"]
        assert perturb_ene[0]["donor atom 2 number"][2592] == 36
        assert perturb_ene[0]["donor atom 2 symbol"][2125] == "B12"
        assert perturb_ene[0]["donor atom 2 number"][2593] == "info_is_from_3C"
        assert perturb_ene[0]["acceptor type"][723] == "3C*"
        assert perturb_ene[0]["perturbation energy"][3209] == approx(3.94)

    def test_qchem_6_1_1(self):
        qc_out = QCOutput(f"{TEST_DIR}/6.1.1.wb97xv.out.gz")
        assert qc_out.data["final_energy"] == approx(-76.43205015)
        n_vals = sum(1 for val in qc_out.data.values() if val is not None)
        assert n_vals == 23

        qc_out_read_optimization = QCOutput(f"{TEST_DIR}/6.1.1.opt.out.gz")
        qc_out_read_optimization._read_optimization_data()
        assert qc_out_read_optimization.data["SCF_energy_in_the_final_basis_set"][-1] == approx(-76.36097614)
        assert qc_out_read_optimization.data["Total_energy_in_the_final_basis_set"][-1] == approx(-76.36097614)

        qc_out_read_frequency = QCOutput(f"{TEST_DIR}/6.1.1.freq.out.gz")
        qc_out_read_frequency._read_frequency_data()
        assert qc_out_read_frequency.data["SCF_energy_in_the_final_basis_set"] == approx(-76.36097614)
        assert qc_out_read_frequency.data["Total_energy_in_the_final_basis_set"] == approx(-76.36097614)


def test_gradient(tmp_path):
    with (
        gzip.open(f"{TEST_DIR}/131.0.gz", "rb") as f_in,
        open(tmp_path / "131.0", "wb") as f_out,
    ):
        shutil.copyfileobj(f_in, f_out)
    gradient = gradient_parser(tmp_path / "131.0")
    assert np.shape(gradient) == (14, 3)
    assert gradient.all()


def test_hessian(tmp_path):
    with (
        gzip.open(f"{TEST_DIR}/132.0.gz", "rb") as f_in,
        open(tmp_path / "132.0", "wb") as f_out,
    ):
        shutil.copyfileobj(f_in, f_out)
    hessian = hessian_parser(tmp_path / "132.0", n_atoms=14)
    assert np.shape(hessian) == (42, 42)
    assert hessian.all()

    hessian = hessian_parser(tmp_path / "132.0")
    assert np.shape(hessian) == (42 * 42,)
    assert hessian.all()


def test_prev_orbital_coeffs(tmp_path):
    with (
        gzip.open(f"{TEST_DIR}/53.0.gz", "rb") as f_in,
        open(tmp_path / "53.0", "wb") as f_out,
    ):
        shutil.copyfileobj(f_in, f_out)
    orbital_coeffs = orbital_coeffs_parser(tmp_path / "53.0")
    assert len(orbital_coeffs) == 360400
    assert orbital_coeffs.all()


def test_multiple_outputs_raises_on_single_init():
    with pytest.raises(ValueError, match="multiple calculation outputs found"):
        QCOutput(f"{NEW_QCHEM_TEST_DIR}/VCGenSCFMan.qcout")


def test_multiple_outputs_keep_sub_files(tmp_path):
    local_file = tmp_path / "VCGenSCFMan.qcout"
    shutil.copy(f"{NEW_QCHEM_TEST_DIR}/VCGenSCFMan.qcout", local_file)
    outputs = QCOutput.multiple_outputs_from_file(str(local_file))
    assert len(outputs) == 2
    assert (tmp_path / "VCGenSCFMan.qcout.0").is_file()
    assert (tmp_path / "VCGenSCFMan.qcout.1").is_file()

    first, second = (out.data for out in outputs)
    assert first["version"] == "4"
    assert first["optimization"] == [[]]
    assert first["multiplicity"] == 2
    assert first["unrestricted"] == [[]]
    assert first["warnings"]["spin_contamination"] == approx(6.751978901)
    assert first["energy_trajectory"] == []
    assert first["errors"] == ["unknown_error"]
    # The trailing chunk contains no Q-Chem banner
    assert second["errors"] == ["never_called_qchem"]
    assert second["charge"] is None
    assert second["multiplicity"] == 1


def test_homo_lumo_nan_values(tmp_path):
    local_file = tmp_path / "homo_lumo_nan_values.qcout"
    shutil.copy(f"{TEST_DIR}/homo_lumo_nan_values.qcout", local_file)
    opt, freq = (out.data for out in QCOutput.multiple_outputs_from_file(str(local_file), keep_sub_files=False))
    assert opt["errors"] == ["hessian_eigenvalue_error"]
    assert opt["final_energy"] == approx(-510.8010083182)
    assert_allclose(opt["energy_trajectory"], [-510.79683337, -510.79918813, -510.80081782, -510.80100832])
    assert opt["warnings"]["inaccurate_integrated_density"].shape == (4, 3)
    assert opt["warnings"]["inaccurate_integrated_density"][-1] == approx([85, 0, -100])
    assert_allclose(opt["warnings"]["spin_contamination"], [0.0362, 0.0405, 0.048, 0.0506])
    assert freq["frequency_job"] == [[]]
    assert freq["errors"] == ["SCF_failed_to_converge"]


def test_switch_to_cartesian():
    data = QCOutput(f"{NEW_QCHEM_TEST_DIR}/switch_to_cartesian.qout").data
    assert data["version"] == "6"
    assert data["new_optimizer"] == [[]]
    assert data["warnings"]["switch_to_cartesian"] is True
    assert data["errors"] == []
    assert data["completion"] == [[]]
    assert data["walltime"] == approx(46670.86)
    assert data["cputime"] == approx(919896.27)
    assert data["charge"] == 1
    assert data["multiplicity"] == 2
    assert len(data["species"]) == 46
    assert data["final_energy"] == approx(-1721.691878905881)
    assert len(data["energy_trajectory"]) == len(data["norm_of_stepsize"]) == 81
    assert data["norm_of_stepsize"][0] == approx(0.702538)
    assert data["energy_trajectory"][-1] == approx(-1721.691879, abs=1e-6)
    assert data["S2"][-1] == approx(0.755902, abs=1e-6)
    assert data["solvent_method"] == "PCM"
    assert data["solvent_data"]["PCM_dielectric"] == approx(3.0)
    assert len(data["solvent_data"]["g_electrostatic"]) == 81
    assert data["solvent_data"]["g_electrostatic"][0] == approx(-0.042089, abs=1e-6)
    assert np.shape(data["pcm_gradients"]) == np.shape(data["gradients"])
    assert_allclose(data["optimized_geometry"], data["last_geometry"])
    assert len(data["molecule_from_optimized_geometry"]) == 46
    assert len(data["RESP"]) == 81


def test_chelpg_single_point():
    data = QCOutput(f"{TEST_DIR}/chelpg_charges.qcout").data
    assert data["version"] == "unknown"
    assert data["single_point_job"] == [[]]
    assert data["point_group"] == "C1"
    assert len(data["species"]) == 31
    assert data["final_energy"] == approx(-895.7443105914)
    assert data["SCF_energy_in_the_final_basis_set"] == approx(-895.7443105914)
    assert data["walltime"] == approx(34.99)
    assert data["dipoles"]["total"] == approx(1.3169)
    assert data["dipoles"]["dipole"] == approx([-0.5894, -1.1773, -0.0288])
    assert data["multipoles"]["quadrupole"]["XX"] == approx(-96.9702)
    assert data["multipoles"]["octopole"]["XXX"] == approx(-149.6523)
    assert len(data["Mulliken"][0]) == 31
    assert data["errors"] == []


def test_hirshfeld_population():
    data = QCOutput(f"{TEST_DIR}/hirshfeld_population.qcout").data
    assert data["version"] == "4"
    assert data["point_group"] == "Cs"
    assert data["species"] == ["O", "H", "H"]
    assert data["unrestricted"] == [[]]
    assert data["final_energy"] == approx(-76.4086658117)
    assert data["dipoles"]["total"] == approx(2.0786)
    assert data["walltime"] == approx(1.33)


@pytest.mark.parametrize(
    ("filename", "is_opt"),
    [("scf_no_message_fail.qcout", False), ("scf_opt_no_message_fail.qcout", True)],
)
def test_scf_fail_no_message(filename, is_opt):
    data = QCOutput(f"{TEST_DIR}/{filename}").data
    assert data["errors"] == ["unknown_error"]
    assert data["completion"] is None
    assert data["mem_total"] == 1100
    assert data["charge"] == -1
    assert data["multiplicity"] == 2
    assert data["point_group"] == "Ci"
    assert data["warnings"]["inaccurate_integrated_density"][0] == approx([49, 48.000992, -2.038792])
    assert data["SCF"] == []
    assert data["final_energy"] is None
    assert (data["optimization"] == [[]]) is is_opt
    assert (data["single_point_job"] == [[]]) is not is_opt


def test_opt_fail_no_message():
    data = QCOutput(f"{TEST_DIR}/opt_fail_no_message.qcout").data
    assert data["errors"] == ["unknown_error"]
    assert data["charge"] == -1
    assert data["final_energy"] == approx(-741.6069065075)
    assert_allclose(
        data["energy_trajectory"], [-741.597154, -741.605806, -741.606859, -741.6069, -741.606907], atol=1e-6
    )
    assert len(data["geometries"]) == len(data["gradients"]) == 5
    assert len(data["SCF_energy_in_the_final_basis_set"]) == 6
    assert_allclose(data["S2"], [0.7593, 0.7612, 0.7618, 0.7619, 0.7619, 0.7619])
    assert_allclose(data["warnings"]["spin_contamination"], [0.0093, 0.0112, 0.0118, 0.0119, 0.0119, 0.0119])
    assert data["optimized_geometry"] is None


def test_gmd_scf():
    data = QCOutput(f"{TEST_DIR}/gmd_scf.qcout").data
    assert data["errors"] == ["unknown_error"]
    assert data["mem_total"] == 5000
    assert data["charge"] == 1
    assert data["multiplicity"] == 2
    assert data["species"][:4] == ["Mg", "N", "N", "Si"]
    assert data["final_energy"] == approx(-1944.956901148)
    assert data["S2"] == approx(0.7522)
    assert data["energy_trajectory"] == approx([-1944.956901148])
    assert data["warnings"]["inaccurate_integrated_density"][0] == approx([189, 190.007387, 0.533009])


def test_freq_ordinal_not_in_range():
    data = QCOutput(f"{TEST_DIR}/ordinal_not_in_range.qcout.gz").data
    assert data["multiple_outputs"] is None
    assert data["version"] == "4"
    assert data["frequency_job"] == [[]]
    assert data["mem_total"] == 1600
    assert data["cpscf_nseg"] == 16
    assert data["ZPE"] == approx(106.877)
    assert data["trans_enthalpy"] == approx(0.889)
    assert data["vib_enthalpy"] == approx(110.862)
    assert data["gas_constant"] == approx(0.592)
    assert data["vib_entropy"] == approx(25.103)
    assert data["total_enthalpy"] == approx(113.232)
    assert data["total_entropy"] == approx(96.054)
    n_modes = 3 * 21 - 6
    assert len(data["frequencies"]) == len(data["IR_intens"]) == len(data["IR_active"]) == n_modes
    assert data["frequencies"][0] == approx(24.1)
    assert data["frequencies"][-1] == approx(3283.55)
    assert data["IR_intens"][1] == approx(5.866)
    assert data["trans_dip"].shape == (n_modes, 3)
    assert data["frequency_mode_vectors"].shape == (n_modes, 21, 3)
    assert data["frequency_mode_vectors"][0, 0] == approx([-0.002, -0.129, -0.078])
    assert data["raman_intens"] is None
    assert "frequency_length_inconsistency" not in data["warnings"]


@pytest.mark.parametrize(
    ("filename", "n_ghosts", "real_formula"),
    [("ghost_atoms.qcout", 3, "H4 C1"), ("MgBF4_b_overalpped.qcout", 1, "Mg1 B1 F3")],  # codespell:ignore overalpped
)
def test_ghost_atoms(filename, n_ghosts, real_formula):
    data = QCOutput(f"{TEST_DIR}/{filename}").data
    # data["species"] keeps Q-Chem's labels; ghost atoms are printed as "GH" and listed first in these inputs
    assert data["species"][:n_ghosts] == ["GH"] * n_ghosts
    assert "GH" not in data["species"][n_ghosts:]

    # In the Molecule, ghost atoms are DummySpecies("X") sites in the same order and at the same positions
    mol = data["initial_molecule"]
    assert len(mol) == len(data["species"])
    assert [site.species_string for site in mol[:n_ghosts]] == ["X0+"] * n_ghosts
    assert all(isinstance(site.specie, DummySpecies) for site in mol[:n_ghosts])
    assert_allclose(mol.cart_coords, data["initial_geometry"])
    # Ghost atoms carry no electrons, so the charge and multiplicity of the real atoms are kept
    assert mol.composition.formula == f"X{n_ghosts} {real_formula}"
    assert (mol.charge, mol.spin_multiplicity) == (data["charge"], data["multiplicity"])

    # Q-Chem prints a Mulliken charge for every site, ghosts included, so per-atom data lines up with the sites
    assert len(data["Mulliken"][-1]) == len(mol)


def test_as_dict():
    qc_out = QCOutput(f"{TEST_DIR}/hf.qcout")
    dct = qc_out.as_dict()
    assert dct["filename"] == qc_out.filename
    assert dct["text"] == qc_out.text
    assert dct["data"]["final_energy"] == approx(qc_out.data["final_energy"])
    assert isinstance(dct["data"]["initial_geometry"], list)


QCHEM_BANNER = "Welcome to Q-Chem\n"
COMPLETION = "Thank you very much for using Q-Chem.  Have a nice day.\n"


def _write_and_parse(tmp_path, text: str) -> dict:
    out_file = tmp_path / "mol.qout"
    out_file.write_text(text, encoding="utf-8")
    return QCOutput(str(out_file)).data


@pytest.mark.parametrize(
    ("text", "expected_error"),
    [
        ("Coordinates do not transform within specified threshold\n", "failed_to_transform_coords"),
        ("The Q-Chem input file has failed to pass inspection\n", "input_file_error"),
        ("Error opening input stream\n", "failed_to_read_input"),
        ("FileMan error: End of file reached prematurely\n", "premature_end_FileMan_error"),
        ("need to increase the array of NLebdevPts\n", "NLebdevPts"),
        ("method not available\n", "method_not_available"),
        ("Could not find $molecule section in ParseQInput\n", "read_molecule_error"),
        ("No banner here\n", "never_called_qchem"),
        (
            QCHEM_BANNER + "***ERROR*** Hessian Appears to have all zero or negative eigenvalues\n",
            "hessian_eigenvalue_error",
        ),
        (QCHEM_BANNER + "FlexNet Licensing error\n", "licensing_error"),
        (QCHEM_BANNER + "Unable to validate license\n", "licensing_error"),
        (QCHEM_BANNER + "Could not open driver file in ReadDriverFromDisk\n", "driver_error"),
        (QCHEM_BANNER + "Basis not supported for the above atom\n", "basis_not_supported"),
        (QCHEM_BANNER + "Unable to find relaxed density\n", "failed_cpscf"),
        (QCHEM_BANNER + "Out of Iterations- IterZ\n", "failed_cpscf"),
        (QCHEM_BANNER + "RUN_NBO6 (rem variable) is not correct\n", "bad_old_nbo6_rem"),
        (QCHEM_BANNER + "NBO_EXTERNAL (rem variable) is not correct\n", "bad_new_nbo_external_rem"),
        (
            QCHEM_BANNER + "gen_scfman_exception:  GDM:: Zero or negative preconditioner scaling factor\n",
            "gdm_neg_precon_error",
        ),
        (QCHEM_BANNER + "too many atoms in ESPChgFit\n", "esp_chg_fit_error"),
        (QCHEM_BANNER + "Please use larger MEM_STATIC\n", "mem_static_too_small"),
        (QCHEM_BANNER + "Please increase MEM_STATIC\n", "mem_static_too_small"),
        (QCHEM_BANNER + "Please increase MEM_TOTAL\n", "mem_total_too_small"),
        (QCHEM_BANNER + "Computing fast CPCM-SWIG hessian \n", "probably_out_of_memory"),
        (QCHEM_BANNER + "Roots Converged\n", "probably_out_of_memory"),
        (QCHEM_BANNER + "1 failed line searches.  Resetting\n" * 11, "SCF_failed_to_converge"),
        (QCHEM_BANNER + "1 failed line searches.  Resetting\n" * 3, "unknown_error"),
    ],
)
def test_completion_errors(tmp_path, text, expected_error):
    data = _write_and_parse(tmp_path, text)
    assert data["completion"] is None
    assert data["errors"] == [expected_error]


@pytest.mark.parametrize(
    ("text", "warning"),
    [
        ("Intel MKL ERROR: Parameter 4 was incorrect\n", "mkl"),
        ("Inconsistent size for SCF MO coefficient file\n", "inconsistent_size"),
        (
            "***ERROR*** Angle 1 2 3 is near-linear\n But No atom available to define colinear bend\n",
            "colinear_bend",
        ),
        ("***ERROR*** Unable to Diagonalize B*B(t) in <MakeNIC>\n", "diagonalizing_BBt"),
        ("Starting finite difference calculation for IDERIV\n", "missing_analytical_derivates"),
    ],
)
def test_general_warnings(tmp_path, text, warning):
    data = _write_and_parse(tmp_path, QCHEM_BANNER + text)
    assert data["warnings"][warning] is True


@pytest.mark.parametrize("completed", [False, True])
def test_smd_unrecognized_solvent(tmp_path, completed):
    text = QCHEM_BANNER + "solvent_method = smd\n Solvent: foo\n Unrecognized solvent\n"
    if completed:
        text += COMPLETION
    data = _write_and_parse(tmp_path, text)
    assert data["solvent_method"] == "SMD"
    assert data["solvent_data"]["SMD_solvent"] == "foo"
    if completed:
        assert data["warnings"]["unrecognized_solvent"] is True
        assert "unrecognized_solvent" not in data["errors"]
    else:
        assert data["errors"] == ["unrecognized_solvent"]
    assert "questionable_SMD_parsing" not in data["warnings"]


def test_smd_two_solvents_and_questionable_parsing(tmp_path):
    data = _write_and_parse(tmp_path, QCHEM_BANNER + "solvent_method smd\n Solvent: water\n Solvent: ethanol\n")
    assert data["warnings"]["SMD_two_solvents"] == "water and ethanol"
    assert data["solvent_data"]["SMD_solvent"] == "water"

    data = _write_and_parse(tmp_path, QCHEM_BANNER + "solvent_method smd\n Solvent: water\n solvent for which\n")
    assert data["warnings"]["questionable_SMD_parsing"] is True
    assert "SMD_two_solvents" not in data["warnings"]


def test_dft_d3(tmp_path):
    d3_line = " -D3 energy without 3body term = {} hartrees\n"
    data = _write_and_parse(tmp_path, QCHEM_BANNER + "dft_d = d3\n" + d3_line.format("-0.0123"))
    assert data["using_dft_d3"] == [[]]
    assert data["dft_d3"] == approx(-0.0123)

    data = _write_and_parse(
        tmp_path, QCHEM_BANNER + "dft_d = d3\n" + d3_line.format("-0.0123") + d3_line.format("-0.0125")
    )
    assert_allclose(data["dft_d3"], [-0.0123, -0.0125])


def test_dft_d3_missing_energy(tmp_path):
    data = _write_and_parse(tmp_path, QCHEM_BANNER + "dft_d = d3\n")
    assert data["dft_d3"] is None


@pytest.mark.parametrize(
    ("trigger", "none_keys"),
    [
        (
            "Start with Direct-Coupling Calculation",
            [f"direct_coupling_{key}_Hartree" for key in ("Hif", "Sif", "Hii", "Sii", "Hff", "Sff")]
            + ["direct_coupling_eV"],
        ),
        ("POD based on the RSCF Fock matrix", ["pod_coupling_eV"]),
        ("FODFT(2n)@DA for ET", ["fodft_had_eV", "fodft_hda_eV", "fodft_coupling_eV"]),  # codespell:ignore hda
        (
            "CCMAN2: suite of methods based on coupled cluster",
            [
                "hf_scf_energy",
                "mp2_energy",
                "ccsd_correlation_energy",
                "ccsd_total_energy",
                "ccsd(t)_correlation_energy",
                "ccsd(t)_total_energy",
            ],
        ),
        (
            "ALMO(MSDFT2) method for electronic coupling",
            [
                "almo_coupling_states",
                "almo_diabat_energies_Hartree",
                "almo_adiabat_energies_Hartree",
                "almo_hamiltonian",
                "almo_overlap_matrix",
                "almo_s2_matrix",
                "almo_diabat_basis_coeff",
                "almo_h_coupling_matrix",
                "almo_coupling_eV",
            ],
        ),
        (
            "CDFT Becke Populations",
            ["cdft_becke_excess_electrons", "cdft_becke_population", "cdft_becke_net_spin"],
        ),
    ],
)
def test_truncated_coupling_sections(tmp_path, trigger, none_keys):
    """Jobs that die right after announcing a method should yield None rather than crash."""
    data = _write_and_parse(tmp_path, f"{QCHEM_BANNER}{trigger}\n")
    for key in none_keys:
        assert key in data, key
        assert data[key] is None, key


def test_initial_geometry_without_point_group(tmp_path):
    header = (
        "$molecule\n0 1\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n$end\n"
        "             Standard Nuclear Orientation (Angstroms)\n"
        "    I     Atom           X                Y                Z\n"
        " ----------------------------------------------------------------\n"
    )
    footer = (
        " ----------------------------------------------------------------\n Nuclear Repulsion Energy =  0.7 hartrees\n"
    )
    rows = "    1      H       0.0000000000     0.0000000000     0.0000000000\n"
    rows += "    2      H       0.0000000000     0.0000000000     0.7400000000\n"
    data = _write_and_parse(tmp_path, QCHEM_BANNER + header + rows + footer)
    assert data["species"] == ["H", "H"]
    assert data["point_group"] is None
    assert data["initial_geometry"][1] == approx([0, 0, 0.74])
    assert data["initial_molecule"].charge == 0
    assert data["initial_molecule"].spin_multiplicity == 1

    # Header printed but no parsable atoms table
    data = _write_and_parse(tmp_path, QCHEM_BANNER + " Nuclear Repulsion Energy =  0.7 hartrees\n")
    assert data["species"] is None
    assert data["initial_geometry"] is None
    assert data["initial_molecule"] is None
    assert data["point_group"] is None


def test_multiple_thresh_warnings(tmp_path):
    block = (
        "\nscfman.C::WARNING energy changes are now smaller than effective accuracy.\n"
        "scfman.C::    calculation will continue, but THRESH should be increased\n"
        "scfman.C::    or SCF_CONVERGENCE decreased. \n"
        "scfman.C::    effective_thresh = {}\n"
    )
    data = _write_and_parse(tmp_path, QCHEM_BANNER + block.format("1.2e-10"))
    assert data["warnings"]["thresh"] == approx(1.2e-10)
    data = _write_and_parse(tmp_path, QCHEM_BANNER + block.format("1.2e-10") + block.format("3.4e-11"))
    assert_allclose(data["warnings"]["thresh"], [1.2e-10, 3.4e-11])


def _opt_constraint_text(value: float, constraint: float) -> str:
    return (
        QCHEM_BANNER + "$opt\nCONSTRAINT\ntors 1 2 3 4 0.0\nENDCONSTRAINT\n$end\n"
        "Constraints and their Current Values\n Value Constraint\n"
        f" Dihedral: 1 2 3 4 {value} {constraint}\n"
    )


def test_opt_constraint(tmp_path):
    data = _write_and_parse(tmp_path, _opt_constraint_text(180.0, -180.0))
    assert data["opt_constraint"] == ["Dihedral", "1", "2", "3", "4", "180.0", "-180.0"]

    with pytest.raises(ValueError, match="should be the same"):
        _write_and_parse(tmp_path, _opt_constraint_text(90.0, 80.0))

    with pytest.raises(ValueError, match="can only differ by a sign"):
        _write_and_parse(tmp_path, _opt_constraint_text(90.0, -90.0))
