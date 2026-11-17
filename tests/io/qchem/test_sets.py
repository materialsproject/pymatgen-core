from __future__ import annotations

import copy
import json
import os
import warnings

import pytest
from monty.json import MontyDecoder, MontyEncoder

from pymatgen.core import Molecule
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.qchem.sets import (
    CMIRS_SETTINGS,
    ForceSet,
    ForceSetGenerator,
    FreqSet,
    FreqSetGenerator,
    OptSet,
    OptSetGenerator,
    PESScanSet,
    PESScanSetGenerator,
    QChemDictSet,
    QCInput,
    QCInputGenerator,
    QCInputSet,
    SinglePointSet,
    SinglePointSetGenerator,
    TransitionStateSet,
    TransitionStateSetGenerator,
)
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

__author__ = "Samuel Blau, Brandon Wood, Shyam Dwaraknath, Evan Spotte-Smith, Ryan Kingsbury"
__copyright__ = "Copyright 2018-2022, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Samuel Blau"
__email__ = "samblau1@gmail.com"


TEST_DIR = f"{TEST_FILES_DIR}/io/qchem/new_qchem_files"


class TestQChemDictSet(MatSciTest):
    def test_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_dict_set = QChemDictSet(
            molecule=test_molecule,
            job_type="opt",
            basis_set="6-31G*",
            scf_algorithm="diis",
        )
        ref_dict = {
            "job_type": "opt",
            "gen_scfman": "true",
            "basis": "6-31g*",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "geom_opt_max_cycles": "200",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_dict_set.rem == ref_dict
        assert test_dict_set.pcm == {}
        assert test_dict_set.solvent == {}
        assert test_dict_set.smx == {}
        assert test_dict_set.molecule == test_molecule

    def test_full_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule

        test_dict_set = QChemDictSet(
            molecule=test_molecule,
            job_type="opt",
            basis_set="6-31g*",
            scf_algorithm="diis",
            dft_rung=2,
            pcm_dielectric=10.0,
            max_scf_cycles=35,
        )
        ref_dict = {
            "job_type": "opt",
            "gen_scfman": "true",
            "basis": "6-31g*",
            "max_scf_cycles": "35",
            "method": "b97-d3",
            "dft_d": "d3_bj",
            "geom_opt_max_cycles": "200",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "pcm",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_dict_set.rem == ref_dict
        ref_dict = {
            "heavypoints": "194",
            "hpoints": "194",
            "radii": "uff",
            "theory": "cpcm",
            "vdwscale": "1.1",
        }
        assert test_dict_set.pcm == ref_dict
        assert test_dict_set.solvent == {"dielectric": "10.0"}
        assert test_dict_set.molecule == test_molecule

        test_dict_set = QChemDictSet(
            molecule=test_molecule,
            job_type="opt",
            basis_set="6-31g*",
            scf_algorithm="diis",
            dft_rung=2,
            smd_solvent="water",
            max_scf_cycles=35,
        )
        ref_dict = {
            "job_type": "opt",
            "gen_scfman": "true",
            "basis": "6-31g*",
            "max_scf_cycles": "35",
            "method": "b97-d3",
            "dft_d": "d3_bj",
            "geom_opt_max_cycles": "200",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "smd",
            "ideriv": "1",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_dict_set.rem == ref_dict
        assert test_dict_set.smx == {"solvent": "water"}

    def test_overwrite_input(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        overwrite_inputs = {
            "rem": {
                "method": "b3lyp",
                "basis": "6-31g*",
                "thresh": "10",
                "s2thresh": "12",
                "xc_grid": "000150000302",
            }
        }
        test_opt_set = OptSet(molecule=test_molecule, overwrite_inputs=overwrite_inputs)
        act_rem = {
            "job_type": "opt",
            "gen_scfman": "true",
            "basis": "6-31g*",
            "max_scf_cycles": "100",
            "method": "b3lyp",
            "scf_algorithm": "diis",
            "xc_grid": "000150000302",
            "geom_opt_max_cycles": "200",
            "thresh": "10",
            "s2thresh": "12",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert act_rem == test_opt_set.rem

    def test_double_solvation(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        raised_error = False
        dict_set = None
        try:
            dict_set = QChemDictSet(
                molecule=test_molecule,
                job_type="opt",
                basis_set="6-31g*",
                scf_algorithm="diis",
                dft_rung=1,
                pcm_dielectric=10.0,
                smd_solvent="water",
                max_scf_cycles=35,
            )
        except ValueError:
            raised_error = True

        assert raised_error
        assert dict_set is None

    def test_pcm_write(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        dict_set = QChemDictSet(
            molecule=test_molecule,
            job_type="opt",
            basis_set="6-31g*",
            scf_algorithm="diis",
            dft_rung=5,
            pcm_dielectric=10.0,
            max_scf_cycles=35,
        )
        dict_set.write(f"{self.tmp_path}/mol.qin")
        test_dict = QCInput.from_file(f"{self.tmp_path}/mol.qin").as_dict()
        rem = {
            "job_type": "opt",
            "basis": "6-31G*",
            "max_scf_cycles": "35",
            "method": "wb97m(2)",
            "geom_opt_max_cycles": "200",
            "gen_scfman": "true",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "pcm",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        pcm = {
            "heavypoints": "194",
            "hpoints": "194",
            "radii": "uff",
            "theory": "cpcm",
            "vdwscale": "1.1",
        }
        qc_input = QCInput(molecule=test_molecule, rem=rem, pcm=pcm, solvent={"dielectric": 10.0})
        for k, v in qc_input.as_dict().items():
            assert v == test_dict[k]

    def test_isosvp_write(self):
        """Also tests overwrite_inputs with a RHOISO value."""
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        dict_set = QChemDictSet(
            molecule=test_molecule,
            job_type="opt",
            basis_set="def2-SVPD",
            scf_algorithm="diis",
            dft_rung=4,
            isosvp_dielectric=62,
            max_scf_cycles=35,
            overwrite_inputs={"svp": {"RHOISO": 0.0009}},
        )
        dict_set.write(f"{self.tmp_path}/mol.qin")
        test_dict = QCInput.from_file(f"{self.tmp_path}/mol.qin").as_dict()
        rem = {
            "job_type": "opt",
            "basis": "def2-SVPD",
            "max_scf_cycles": "35",
            "method": "wb97mv",
            "geom_opt_max_cycles": "200",
            "gen_scfman": "false",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "isosvp",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }

        qc_input = QCInput(
            molecule=test_molecule,
            rem=rem,
            pcm_nonels=None,
            svp={
                "RHOISO": 0.0009,
                "DIELST": 62,
                "NPTLEB": 1202,
                "ITRNGR": 2,
                "IROTGR": 2,
            },
        )
        for k, v in qc_input.as_dict().items():
            assert v == test_dict[k]

    def test_smd_write(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        dict_set = QChemDictSet(
            molecule=test_molecule,
            job_type="opt",
            basis_set="6-31g*",
            scf_algorithm="diis",
            dft_rung=4,
            smd_solvent="water",
            max_scf_cycles=35,
        )
        dict_set.write(f"{self.tmp_path}/mol.qin")
        test_dict = QCInput.from_file(f"{self.tmp_path}/mol.qin").as_dict()
        rem = {
            "job_type": "opt",
            "basis": "6-31G*",
            "max_scf_cycles": "35",
            "method": "wb97mv",
            "geom_opt_max_cycles": "200",
            "gen_scfman": "true",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "smd",
            "ideriv": "1",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        qc_input = QCInput(molecule=test_molecule, rem=rem, smx={"solvent": "water"})
        for k, v in qc_input.as_dict().items():
            assert v == test_dict[k]

    def test_cmirs_write(self):
        """Also tests overwrite_inputs with a RHOISO value."""
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        dict_set = QChemDictSet(
            molecule=test_molecule,
            job_type="opt",
            basis_set="def2-SVPD",
            scf_algorithm="diis",
            dft_rung=4,
            cmirs_solvent="water",
            max_scf_cycles=35,
            overwrite_inputs={"svp": {"RHOISO": 0.0005}},
        )
        dict_set.write(f"{self.tmp_path}/mol.qin")
        test_dict = QCInput.from_file(f"{self.tmp_path}/mol.qin").as_dict()
        rem = {
            "job_type": "opt",
            "basis": "def2-SVPD",
            "max_scf_cycles": "35",
            "method": "wb97mv",
            "geom_opt_max_cycles": "200",
            "gen_scfman": "false",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "isosvp",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        pcm_nonels = {
            "A": -0.006496,
            "B": 0.050833,
            "C": -566.7,
            "D": -30.503,
            "gamma": 3.2,
            "solvrho": 0.05,
            "Delta": 7,
            "GauLag_N": 40,
        }

        qc_input = QCInput(
            molecule=test_molecule,
            rem=rem,
            pcm_nonels=pcm_nonels,
            svp={
                "RHOISO": 0.0005,
                "DIELST": 78.39,
                "NPTLEB": 1202,
                "ITRNGR": 2,
                "IROTGR": 2,
                "IPNRF": 1,
                "IDEFESR": 1,
            },
        )
        for k, v in qc_input.as_dict().items():
            assert v == test_dict[k]

    def test_custom_smd_write(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        dict_set = QChemDictSet(
            molecule=test_molecule,
            job_type="opt",
            basis_set="6-31g*",
            scf_algorithm="diis",
            dft_rung=4,
            smd_solvent="custom",
            custom_smd="90.00,1.415,0.00,0.735,20.2,0.00,0.00",
            max_scf_cycles=35,
        )
        dict_set.write(f"{self.tmp_path}/mol.qin")
        test_dict = QCInput.from_file(f"{self.tmp_path}/mol.qin").as_dict()
        rem = {
            "job_type": "opt",
            "basis": "6-31G*",
            "max_scf_cycles": "35",
            "method": "wb97mv",
            "geom_opt_max_cycles": "200",
            "gen_scfman": "true",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "smd",
            "ideriv": "1",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        qc_input = QCInput(molecule=test_molecule, rem=rem, smx={"solvent": "other"})
        for k, v in qc_input.as_dict().items():
            assert v == test_dict[k]
        with open("solvent_data", encoding="utf-8") as sd:
            lines = sd.readlines()
            assert lines[0] == "90.00,1.415,0.00,0.735,20.2,0.00,0.00"
        os.remove("solvent_data")

    def test_output_wavefunction(self):
        """Test function for outputting *.wfn files"""
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_dict_set = QChemDictSet(
            molecule=test_molecule,
            job_type="opt",
            basis_set="6-31G*",
            scf_algorithm="diis",
            output_wavefunction=True,
        )

        assert test_dict_set.rem["write_wfn"] == "wavefunction"

    def test_solvation_warnings(self):
        """Test warnings / errors resulting from nonsensical overwrite_inputs."""
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        with pytest.raises(RuntimeError, match="CMIRS is only parameterized"):
            QChemDictSet(
                molecule=test_molecule,
                job_type="opt",
                basis_set="def2-SVPD",
                scf_algorithm="diis",
                dft_rung=5,
                cmirs_solvent="water",
                max_scf_cycles=35,
                overwrite_inputs={"svp": {"RHOISO": 0.0007}},
            )
        with pytest.warns(UserWarning, match="Setting IDEFESR=0"):
            QChemDictSet(
                molecule=test_molecule,
                job_type="opt",
                basis_set="def2-SVPD",
                scf_algorithm="diis",
                dft_rung=5,
                cmirs_solvent="water",
                max_scf_cycles=35,
                overwrite_inputs={"svp": {"IDEFESR": 0}},
            )
        with pytest.warns(UserWarning, match="Setting IDEFESR=1"):
            QChemDictSet(
                molecule=test_molecule,
                job_type="opt",
                basis_set="def2-SVPD",
                scf_algorithm="diis",
                dft_rung=5,
                isosvp_dielectric=78,
                max_scf_cycles=35,
                overwrite_inputs={"svp": {"IDEFESR": 1}},
            )
        with pytest.warns(UserWarning, match="Setting DIELST"):
            QChemDictSet(
                molecule=test_molecule,
                job_type="opt",
                basis_set="def2-SVPD",
                scf_algorithm="diis",
                dft_rung=5,
                pcm_dielectric=78,
                max_scf_cycles=35,
                overwrite_inputs={"svp": {"DIELST": 67}},
            )
        with pytest.warns(UserWarning, match="The solvent section will be ignored"):
            QChemDictSet(
                molecule=test_molecule,
                job_type="opt",
                basis_set="def2-SVPD",
                scf_algorithm="diis",
                dft_rung=5,
                isosvp_dielectric=78,
                max_scf_cycles=35,
                overwrite_inputs={"solvent": {"dielectric": 67}},
            )


class TestSinglePointSet(MatSciTest):
    def test_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_sp_set = SinglePointSet(molecule=test_molecule)
        ref_dict = {
            "job_type": "sp",
            "gen_scfman": "true",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_sp_set.rem == ref_dict
        assert test_sp_set.pcm == {}
        assert test_sp_set.solvent == {}
        assert test_sp_set.molecule == test_molecule

    def test_scf_extra_print(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_sp_set = SinglePointSet(molecule=test_molecule, extra_scf_print=True)
        ref_dict = {
            "job_type": "sp",
            "gen_scfman": "true",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
            "scf_convergence": "8",
            "scf_final_print": "3",
        }
        assert test_sp_set.rem == ref_dict
        assert test_sp_set.pcm == {}
        assert test_sp_set.solvent == {}
        assert test_sp_set.molecule == test_molecule

    def test_pcm_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_sp_set = SinglePointSet(molecule=test_molecule, pcm_dielectric=10.0)
        ref_dict = {
            "job_type": "sp",
            "gen_scfman": "true",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "pcm",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_sp_set.rem == ref_dict
        ref_dict = {
            "heavypoints": "194",
            "hpoints": "194",
            "radii": "uff",
            "theory": "cpcm",
            "vdwscale": "1.1",
        }
        assert test_sp_set.pcm == ref_dict
        assert test_sp_set.solvent == {"dielectric": "10.0"}
        assert test_sp_set.molecule == test_molecule

    def test_isosvp_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_sp_set = SinglePointSet(molecule=test_molecule, isosvp_dielectric=10.0)
        ref_dict = {
            "job_type": "sp",
            "gen_scfman": "false",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "isosvp",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_sp_set.rem == ref_dict
        ref_dict = {
            "dielst": "10.0",
            "rhoiso": "0.001",
            "nptleb": "1202",
            "itrngr": "2",
            "irotgr": "2",
        }
        assert test_sp_set.svp == ref_dict
        assert test_sp_set.molecule == test_molecule

    def test_smd_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_sp_set = SinglePointSet(molecule=test_molecule, smd_solvent="water")
        ref_dict = {
            "job_type": "sp",
            "gen_scfman": "true",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "smd",
            "ideriv": "1",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_sp_set.rem == ref_dict
        assert test_sp_set.smx == {"solvent": "water"}
        assert test_sp_set.molecule == test_molecule

    def test_cmirs_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_sp_set = SinglePointSet(
            molecule=test_molecule,
            cmirs_solvent="benzene",
            overwrite_inputs={"svp": {"RHOISO": 0.0005}},
        )
        ref_dict = {
            "job_type": "sp",
            "gen_scfman": "false",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "isosvp",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_sp_set.rem == ref_dict
        ref_dict = {
            "dielst": "2.28",
            "rhoiso": "0.0005",
            "nptleb": "1202",
            "itrngr": "2",
            "irotgr": "2",
            "ipnrf": "1",
            "idefesr": "1",
        }
        assert test_sp_set.svp == ref_dict
        ref_dict = {
            "a": "-0.00572",
            "b": "0.01116",
            "c": None,
            "d": None,
            "gamma": None,
            "solvrho": "0.0421",
            "gaulag_n": "40",
            "delta": "7",
        }
        assert test_sp_set.pcm_nonels == ref_dict
        assert test_sp_set.molecule == test_molecule

    def test_plots_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_sp_set = SinglePointSet(molecule=test_molecule, smd_solvent="water", plot_cubes=True)
        ref_dict = {
            "job_type": "sp",
            "gen_scfman": "true",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "smd",
            "ideriv": "1",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
            "plots": "true",
            "make_cube_files": "true",
        }
        assert test_sp_set.rem == ref_dict
        assert test_sp_set.plots == {"grid_spacing": "0.05", "total_density": "0"}
        assert test_sp_set.smx == {"solvent": "water"}
        assert test_sp_set.molecule == test_molecule


class TestOptSet(MatSciTest):
    def test_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_opt_set = OptSet(molecule=test_molecule)
        ref_dict = {
            "job_type": "opt",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "geom_opt_max_cycles": "200",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_opt_set.rem == ref_dict
        assert test_opt_set.pcm == {}
        assert test_opt_set.solvent == {}
        assert test_opt_set.smx == {}
        assert test_opt_set.molecule == test_molecule

    def test_pcm_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_opt_set = OptSet(molecule=test_molecule, pcm_dielectric=10.0)
        ref_dict = {
            "job_type": "opt",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "geom_opt_max_cycles": "200",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "pcm",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_opt_set.rem == ref_dict
        ref_dict = {
            "heavypoints": "194",
            "hpoints": "194",
            "radii": "uff",
            "theory": "cpcm",
            "vdwscale": "1.1",
        }
        assert test_opt_set.pcm == ref_dict
        assert test_opt_set.solvent == {"dielectric": "10.0"}
        assert test_opt_set.molecule == test_molecule

    def test_smd_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_opt_set = OptSet(molecule=test_molecule, smd_solvent="water")
        ref_dict = {
            "job_type": "opt",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "geom_opt_max_cycles": "200",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "smd",
            "ideriv": "1",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_opt_set.rem == ref_dict
        assert test_opt_set.smx == {"solvent": "water"}
        assert test_opt_set.molecule == test_molecule

    def test_overwrite_opt_input(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        overwrite_inputs = {"opt": {"FIXED": ["1 XYZ", "2 XY"]}}
        test_opt_set = OptSet(molecule=test_molecule, overwrite_inputs=overwrite_inputs)
        act_opt = {"fixed": ["1 XYZ", "2 XY"]}
        assert act_opt == test_opt_set.opt

    def test_nbo_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_opt_set = OptSet(molecule=test_molecule, nbo_params={})
        ref_dict = {
            "job_type": "opt",
            "gen_scfman": "true",
            "geom_opt_max_cycles": "200",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
            "nbo": "true",
        }
        assert test_opt_set.rem == ref_dict
        assert test_opt_set.nbo == {}
        assert test_opt_set.molecule == test_molecule

    def test_v5_vs_v6(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        v5_opt_set = OptSet(molecule=test_molecule, qchem_version=5, basis_set="def2-tzvpd", geom_opt={})
        ref_dict = {
            "job_type": "opt",
            "gen_scfman": "true",
            "geom_opt_max_cycles": "200",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
            "geom_opt2": "3",
        }
        assert v5_opt_set.rem == ref_dict
        assert v5_opt_set.geom_opt == {"maxiter": "200"}
        assert v5_opt_set.molecule == test_molecule

        v6_opt_set = OptSet(molecule=test_molecule, qchem_version=6, basis_set="def2-tzvpd", geom_opt={})
        ref_dict = {
            "job_type": "opt",
            "gen_scfman": "true",
            "geom_opt_max_cycles": "200",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert v6_opt_set.rem == ref_dict
        ref_dict = {
            "maxiter": "200",
            "coordinates": "redundant",
            "max_displacement": "0.1",
            "optimization_restart": "false",
        }
        assert v6_opt_set.geom_opt == ref_dict
        assert v6_opt_set.molecule == test_molecule

        v6_opt_set_modified = OptSet(
            molecule=test_molecule,
            qchem_version=6,
            basis_set="def2-tzvpd",
            geom_opt={"coordinates": "delocalized", "initial_hessian": "read"},
        )
        ref_dict = {
            "maxiter": "200",
            "coordinates": "delocalized",
            "max_displacement": "0.1",
            "initial_hessian": "read",
            "optimization_restart": "false",
        }
        assert v6_opt_set_modified.geom_opt == ref_dict


class TestTransitionStateSet(MatSciTest):
    def test_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_ts_set = TransitionStateSet(molecule=test_molecule)
        ref_dict = {
            "job_type": "ts",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "geom_opt_max_cycles": "200",
            "resp_charges": "true",
            "sym_ignore": "true",
            "symmetry": "false",
        }
        assert test_ts_set.rem == ref_dict
        assert test_ts_set.pcm == {}
        assert test_ts_set.solvent == {}
        assert test_ts_set.smx == {}
        assert test_ts_set.molecule == test_molecule

    def test_pcm_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_ts_set = TransitionStateSet(molecule=test_molecule, pcm_dielectric=10.0)
        ref_dict = {
            "job_type": "ts",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "geom_opt_max_cycles": "200",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "pcm",
            "resp_charges": "true",
            "sym_ignore": "true",
            "symmetry": "false",
        }
        assert test_ts_set.rem == ref_dict
        ref_dict = {
            "heavypoints": "194",
            "hpoints": "194",
            "radii": "uff",
            "theory": "cpcm",
            "vdwscale": "1.1",
        }
        assert test_ts_set.pcm == ref_dict
        assert test_ts_set.solvent == {"dielectric": "10.0"}
        assert test_ts_set.molecule == test_molecule

    def test_smd_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_ts_set = TransitionStateSet(molecule=test_molecule, smd_solvent="water")
        ref_dict = {
            "job_type": "ts",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "geom_opt_max_cycles": "200",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "smd",
            "ideriv": "1",
            "resp_charges": "true",
            "sym_ignore": "true",
            "symmetry": "false",
        }
        assert test_ts_set.rem == ref_dict
        assert test_ts_set.smx == {"solvent": "water"}
        assert test_ts_set.molecule == test_molecule


class TestForceSet(MatSciTest):
    def test_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_forceset = ForceSet(molecule=test_molecule)
        ref_dict = {
            "job_type": "force",
            "gen_scfman": "true",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_forceset.rem == ref_dict
        assert test_forceset.pcm == {}
        assert test_forceset.solvent == {}
        assert test_forceset.molecule == test_molecule

    def test_pcm_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_forceset = ForceSet(molecule=test_molecule, pcm_dielectric=10.0)
        ref_dict = {
            "job_type": "force",
            "gen_scfman": "true",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "pcm",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_forceset.rem == ref_dict
        ref_dict = {
            "heavypoints": "194",
            "hpoints": "194",
            "radii": "uff",
            "theory": "cpcm",
            "vdwscale": "1.1",
        }
        assert test_forceset.pcm == ref_dict
        assert test_forceset.solvent == {"dielectric": "10.0"}
        assert test_forceset.molecule == test_molecule

    def test_smd_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_forceset = ForceSet(molecule=test_molecule, smd_solvent="water")
        ref_dict = {
            "job_type": "force",
            "gen_scfman": "true",
            "basis": "def2-tzvpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "smd",
            "ideriv": "1",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_forceset.rem == ref_dict
        assert test_forceset.smx, {"solvent": "water"}
        assert test_forceset.molecule == test_molecule


class TestPESScanSet(MatSciTest):
    def test_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pes_scan.qin").molecule

        test_pes_scan = PESScanSet(molecule=test_molecule, scan_variables={"stre": ["3 6 1.5 1.9 0.01"]})
        ref_dict = {
            "job_type": "pes_scan",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "geom_opt_max_cycles": "200",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "resp_charges": "true",
            "sym_ignore": "true",
            "symmetry": "false",
        }
        assert test_pes_scan.rem == ref_dict
        assert test_pes_scan.pcm == {}
        assert test_pes_scan.solvent == {}
        assert test_pes_scan.smx == {}
        assert test_pes_scan.scan == {"stre": ["3 6 1.5 1.9 0.01"]}
        assert test_pes_scan.molecule == test_molecule

    def test_pcm_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pes_scan.qin").molecule
        test_pes_scan = PESScanSet(
            molecule=test_molecule,
            pcm_dielectric=10.0,
            scan_variables={"stre": ["3 6 1.5 1.9 0.01"]},
        )
        ref_dict = {
            "job_type": "pes_scan",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "geom_opt_max_cycles": "200",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "pcm",
            "resp_charges": "true",
            "sym_ignore": "true",
            "symmetry": "false",
        }
        assert test_pes_scan.rem == ref_dict
        ref_dict = {
            "heavypoints": "194",
            "hpoints": "194",
            "radii": "uff",
            "theory": "cpcm",
            "vdwscale": "1.1",
        }
        assert test_pes_scan.pcm == ref_dict
        assert test_pes_scan.solvent == {"dielectric": "10.0"}
        assert test_pes_scan.scan == {"stre": ["3 6 1.5 1.9 0.01"]}
        assert test_pes_scan.molecule == test_molecule

    def test_smd_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pes_scan.qin").molecule
        test_pes_scan = PESScanSet(
            molecule=test_molecule,
            smd_solvent="water",
            scan_variables={"stre": ["3 6 1.5 1.9 0.01"]},
        )
        ref_dict = {
            "job_type": "pes_scan",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "geom_opt_max_cycles": "200",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "smd",
            "ideriv": "1",
            "resp_charges": "true",
            "sym_ignore": "true",
            "symmetry": "false",
        }
        assert test_pes_scan.rem == ref_dict
        assert test_pes_scan.smx == {"solvent": "water"}
        assert test_pes_scan.scan == {"stre": ["3 6 1.5 1.9 0.01"]}
        assert test_pes_scan.molecule == test_molecule


class TestFreqSet(MatSciTest):
    def test_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_freq_set = FreqSet(molecule=test_molecule)
        ref_dict = {
            "job_type": "freq",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_freq_set.rem == ref_dict
        assert test_freq_set.pcm == {}
        assert test_freq_set.solvent == {}
        assert test_freq_set.molecule == test_molecule

    def test_pcm_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_freq_set = FreqSet(molecule=test_molecule, pcm_dielectric=10.0)
        ref_dict = {
            "job_type": "freq",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "pcm",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_freq_set.rem == ref_dict
        ref_dict = {
            "heavypoints": "194",
            "hpoints": "194",
            "radii": "uff",
            "theory": "cpcm",
            "vdwscale": "1.1",
        }
        assert test_freq_set.pcm == ref_dict
        assert test_freq_set.solvent == {"dielectric": "10.0"}
        assert test_freq_set.molecule == test_molecule

    def test_smd_init(self):
        test_molecule = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
        test_freq_set = FreqSet(molecule=test_molecule, smd_solvent="water")
        ref_dict = {
            "job_type": "freq",
            "gen_scfman": "true",
            "basis": "def2-svpd",
            "max_scf_cycles": "100",
            "method": "wb97mv",
            "scf_algorithm": "diis",
            "xc_grid": "3",
            "thresh": "14",
            "s2thresh": "16",
            "solvent_method": "smd",
            "ideriv": "1",
            "symmetry": "false",
            "sym_ignore": "true",
            "resp_charges": "true",
        }
        assert test_freq_set.rem == ref_dict
        assert test_freq_set.smx == {"solvent": "water"}
        assert test_freq_set.molecule == test_molecule


def test_cmirs_does_not_mutate_module_settings():
    mol = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
    reference = copy.deepcopy(CMIRS_SETTINGS)

    SinglePointSet(mol, cmirs_solvent="water", overwrite_inputs={"svp": {"rhoiso": "0.0005"}})
    SinglePointSet(mol, cmirs_solvent="water", overwrite_inputs={"pcm_nonels": {"a": "-0.1"}})
    assert reference == CMIRS_SETTINGS

    # a later default (rhoiso = 0.001) calculation must get the 0.001 parameters
    sp_set = SinglePointSet(mol, cmirs_solvent="water")
    assert sp_set.pcm_nonels == {**CMIRS_SETTINGS["water"]["0.001"], "delta": "7", "gaulag_n": "40"}


def test_dict_set_does_not_mutate_user_inputs():
    mol = QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule
    geom_opt = {"coordinates": "delocalized"}
    opt_variables = {"CONSTRAINT": ["tors 2 3 4 5 25.0"]}
    scan_variables = {"stre": ["3 6 1.5 1.9 0.1"]}

    OptSet(
        mol,
        qchem_version=6,
        geom_opt=geom_opt,
        opt_variables=opt_variables,
        overwrite_inputs={"opt": {"FIXED": ["2 XY"]}},
    )
    PESScanSet(mol, scan_variables=scan_variables, overwrite_inputs={"scan": {"tors": ["1 2 3 4 -180 180 15"]}})
    assert geom_opt == {"coordinates": "delocalized"}
    assert opt_variables == {"CONSTRAINT": ["tors 2 3 4 5 25.0"]}
    assert scan_variables == {"stre": ["3 6 1.5 1.9 0.1"]}

    # reusing the same geom_opt dict with a different number of cycles must not raise
    opt_set = OptSet(mol, qchem_version=6, geom_opt=geom_opt, geom_opt_max_cycles=50)
    assert opt_set.geom_opt["maxiter"] == "50"


LEGACY_SETS_AND_GENERATORS = [
    (SinglePointSet, SinglePointSetGenerator),
    (OptSet, OptSetGenerator),
    (TransitionStateSet, TransitionStateSetGenerator),
    (ForceSet, ForceSetGenerator),
    (FreqSet, FreqSetGenerator),
    (PESScanSet, PESScanSetGenerator),
]

# settings accepted by all of the legacy input sets
COMMON_SETTINGS = {
    "default": {},
    "pcm_rung_2": {"dft_rung": 2, "pcm_dielectric": 78.39},
    "smd": {"smd_solvent": "water"},
    "isosvp": {"isosvp_dielectric": 62, "overwrite_inputs": {"svp": {"ipnrf": 1}}},
    "cmirs": {"cmirs_solvent": "water", "overwrite_inputs": {"svp": {"rhoiso": "0.0005"}}},
    "nbo7": {"nbo_params": {"version": 7, "plots": "PLOT"}},
    "qchem6": {"qchem_version": 6, "max_scf_cycles": 35, "basis_set": "6-31g*", "scf_algorithm": "gdm"},
    "overwrite": {
        "pcm_dielectric": 10.0,
        "plot_cubes": True,
        "output_wavefunction": True,
        "overwrite_inputs": {
            "rem": {"METHOD": "b3lyp", "symmetry": "true"},
            "pcm": {"theory": "ssvpe"},
            "van_der_waals": {"16": 3.14159},
            "plots": {"grid_spacing": "0.1"},
        },
    },
}


@pytest.fixture(scope="module")
def pcm_molecule() -> Molecule:
    return QCInput.from_file(f"{TEST_DIR}/pcm.qin").molecule


@pytest.fixture(scope="module")
def water() -> Molecule:
    return Molecule(["O", "H", "H"], [[0.0, 0.0, 0.121], [-0.783, 0.0, -0.485], [0.783, 0.0, -0.485]])


class TestQCInputGenerator:
    @pytest.mark.parametrize(("legacy_cls", "generator_cls"), LEGACY_SETS_AND_GENERATORS)
    @pytest.mark.parametrize("settings", COMMON_SETTINGS.values(), ids=COMMON_SETTINGS)
    def test_same_input_as_legacy_sets(self, legacy_cls, generator_cls, settings, pcm_molecule):
        settings = copy.deepcopy(settings)
        if legacy_cls is PESScanSet:
            settings["scan_variables"] = {"stre": ["3 6 1.5 1.9 0.1"]}

        legacy_set = legacy_cls(pcm_molecule, **copy.deepcopy(settings))
        input_set = generator_cls(**settings).get_input_set(pcm_molecule)

        assert isinstance(input_set, QCInputSet)
        assert type(input_set.qcinput) is QCInput
        assert str(input_set.qcinput) == str(legacy_set)
        assert input_set.qcinput.molecule == pcm_molecule

    @pytest.mark.parametrize(
        ("legacy_cls", "generator_cls", "job_type", "basis_set"),
        [
            (SinglePointSet, SinglePointSetGenerator, "sp", "def2-tzvpd"),
            (OptSet, OptSetGenerator, "opt", "def2-svpd"),
            (TransitionStateSet, TransitionStateSetGenerator, "ts", "def2-svpd"),
            (ForceSet, ForceSetGenerator, "force", "def2-tzvpd"),
            (FreqSet, FreqSetGenerator, "freq", "def2-svpd"),
            (PESScanSet, PESScanSetGenerator, "pes_scan", "def2-svpd"),
        ],
    )
    def test_defaults(self, legacy_cls, generator_cls, job_type, basis_set, pcm_molecule):
        generator = generator_cls()
        assert isinstance(generator, QCInputGenerator)
        assert isinstance(generator, InputGenerator)
        assert generator.job_type == job_type
        assert generator.basis_set == basis_set
        assert generator.scf_algorithm == "diis"
        assert generator.qchem_version == 5
        assert generator.dft_rung == 4

        scan = {"scan_variables": {"stre": ["3 6 1.5 1.9 0.1"]}} if legacy_cls is PESScanSet else {}
        legacy_set = legacy_cls(pcm_molecule, **scan)
        assert legacy_set.rem["job_type"] == job_type
        assert legacy_set.rem["basis"] == basis_set

    @pytest.mark.parametrize(
        "settings",
        [
            {"job_type": "opt", "qchem_version": 6, "geom_opt": {"coordinates": "delocalized"}},
            {"job_type": "opt", "geom_opt": {"maxiter": "50"}, "geom_opt_max_cycles": 50},
            {"job_type": "ts", "opt_variables": {"CONSTRAINT": ["tors 2 3 4 5 25.0"], "FIXED": ["2 XY"]}},
            {"job_type": "pes_scan", "scan_variables": {"stre": ["3 6 1.5 1.9 0.1"]}, "vdw_mode": "sequential"},
            {"job_type": "sp", "smd_solvent": "custom", "custom_smd": "90.00,1.415,0.00,0.735,20.2,0.00,0.00"},
            {
                "job_type": "sp",
                "qchem_version": 6,
                "smd_solvent": "custom",
                "custom_smd": "90.00,1.415,0.00,0.735,20.2,0.00,0.00",
            },
            {
                "job_type": "sp",
                "extra_scf_print": True,
                "cdft_constraints": [
                    [{"value": 1.0, "coefficients": [1.0], "first_atoms": [1], "last_atoms": [2], "types": [None]}]
                ],
            },
            {"job_type": "sp", "almo_coupling_states": [[(1, 2), (0, 1)], [(0, 1), (1, 2)]]},
        ],
    )
    def test_same_input_as_dict_set(self, settings, pcm_molecule):
        settings = {"basis_set": "6-31g*", "scf_algorithm": "diis", **settings}
        legacy_set = QChemDictSet(pcm_molecule, **copy.deepcopy(settings))
        input_set = QCInputGenerator(**settings).get_input_set(pcm_molecule)
        assert str(input_set.qcinput) == str(legacy_set)

    def test_get_input_set_is_repeatable(self, pcm_molecule, water):
        settings = {
            "pcm_dielectric": 10.0,
            "qchem_version": 6,
            "geom_opt": {"coordinates": "delocalized"},
            "opt_variables": {"CONSTRAINT": ["tors 2 3 4 5 25.0"]},
            "nbo_params": {"version": 7},
            "overwrite_inputs": {
                "rem": {"symmetry": "true"},
                "pcm": {"theory": "ssvpe"},
                "opt": {"FIXED": ["2 XY"]},
                "van_der_waals": {"16": 3.14159},
            },
        }
        reference = copy.deepcopy(settings)
        generator = OptSetGenerator(**settings)
        as_dict = generator.as_dict()

        first = str(generator.get_input_set(pcm_molecule).qcinput)
        second = str(generator.get_input_set(pcm_molecule).qcinput)
        assert first == second
        # neither the generator nor the user-supplied dicts are modified
        assert settings == reference
        assert generator.as_dict() == as_dict
        assert generator == OptSetGenerator(**reference)

        # one generator can be used for any number of molecules
        water_input = generator.get_input_set(water).qcinput
        assert water_input.molecule == water
        assert water_input.rem == generator.get_input_set(pcm_molecule).qcinput.rem

    def test_settings_can_be_changed_after_init(self, water):
        generator = SinglePointSetGenerator()
        assert "solvent_method" not in generator.get_input_set(water).qcinput.rem

        generator.overwrite_inputs = {"rem": {"solvent_method": "smd"}, "smx": {"solvent": "water"}}
        qcinput = generator.get_input_set(water).qcinput
        assert qcinput.rem["solvent_method"] == "smd"
        assert qcinput.smx == {"solvent": "water"}

        generator.overwrite_inputs = None
        assert "solvent_method" not in generator.get_input_set(water).qcinput.rem

    def test_cmirs_settings_not_mutated(self, water):
        reference = copy.deepcopy(CMIRS_SETTINGS)
        generator = SinglePointSetGenerator(
            cmirs_solvent="water", overwrite_inputs={"svp": {"rhoiso": "0.0005"}, "pcm_nonels": {"a": "-0.1"}}
        )
        generator.get_input_set(water)
        generator.get_input_set(water)
        assert reference == CMIRS_SETTINGS

    def test_invalid_settings(self):
        with pytest.raises(ValueError, match="dft_rung should be between 1 and 5"):
            SinglePointSetGenerator(dft_rung=6)
        with pytest.raises(ValueError, match="Only one of PCM, ISOSVP, SMD, and CMIRS"):
            SinglePointSetGenerator(pcm_dielectric=10.0, smd_solvent="water")
        with pytest.raises(ValueError, match="A user-defined SMD requires passing custom_smd"):
            SinglePointSetGenerator(smd_solvent="custom")
        with pytest.raises(RuntimeError, match="nbo params version should only be set to 7"):
            SinglePointSetGenerator(nbo_params={"version": 6})
        with pytest.raises(RuntimeError, match="Max # of optimization cycles must be the same"):
            OptSetGenerator(geom_opt={"maxiter": "100"})
        with pytest.raises(RuntimeError, match="Set nbo parameters directly with nbo_params input"):
            SinglePointSetGenerator(overwrite_inputs={"nbo": {"plots": "PLOT"}})
        with pytest.raises(RuntimeError, match="Set geom_opt params directly with geom_opt input"):
            OptSetGenerator(overwrite_inputs={"geom_opt": {"maxiter": "100"}})
        with pytest.raises(TypeError, match="missing 3 required positional arguments"):
            QCInputGenerator()

    def test_warnings_only_when_generating(self, water):
        overwrite_inputs = {"solvent": {"dielectric": "78.39"}}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            generator = SinglePointSetGenerator(smd_solvent="water", overwrite_inputs=overwrite_inputs)
        with pytest.warns(UserWarning, match="The solvent section will be ignored unless solvent_method=pcm"):
            generator.get_input_set(water)

        # also when no solvent method is set at all
        with pytest.warns(UserWarning, match="The solvent section will be ignored unless solvent_method=pcm"):
            SinglePointSetGenerator(overwrite_inputs=overwrite_inputs).get_input_set(water)
        with pytest.warns(UserWarning, match="Setting DIELST will have no effect"):
            SinglePointSetGenerator(overwrite_inputs={"svp": {"dielst": "10"}}).get_input_set(water)

    def test_pes_scan_needs_scan_variables(self, water):
        generator = PESScanSetGenerator()
        with pytest.raises(ValueError, match="Cannot run a pes_scan job without some variable to scan over"):
            generator.get_input_set(water)

        scan_variables = {"stre": ["1 2 0.95 1.35 0.1"]}
        assert PESScanSetGenerator(scan_variables=scan_variables).get_input_set(water).qcinput.scan == scan_variables
        generator.overwrite_inputs = {"scan": scan_variables}
        assert generator.get_input_set(water).qcinput.scan == scan_variables

    def test_msonable(self, water):
        generator = OptSetGenerator(
            qchem_version=6,
            pcm_dielectric=10.0,
            geom_opt={"coordinates": "delocalized"},
            overwrite_inputs={"rem": {"symmetry": "true"}},
        )
        for from_dict in (
            OptSetGenerator.from_dict(generator.as_dict()),
            json.loads(json.dumps(generator, cls=MontyEncoder), cls=MontyDecoder),
        ):
            assert type(from_dict) is OptSetGenerator
            assert from_dict == generator
            assert str(from_dict.get_input_set(water).qcinput) == str(generator.get_input_set(water).qcinput)


class TestQCInputSet:
    def test_is_input_set(self, water):
        input_set = SinglePointSetGenerator(pcm_dielectric=78.39).get_input_set(water)
        assert isinstance(input_set, InputSet)
        assert list(input_set) == ["mol.qin"]
        assert len(input_set) == 1
        assert input_set["mol.qin"] is input_set.qcinput
        assert input_set.optional_files == {}
        assert input_set.qcinput.solvent == {"dielectric": "78.39"}

        # the dict-like interface and the attributes are views of the same data
        input_set["extra.txt"] = "some text"
        assert input_set.optional_files == {"extra.txt": "some text"}
        new_input = QCInput(water, rem={"job_type": "sp", "basis": "6-31g*", "method": "b3lyp"})
        input_set["mol.qin"] = new_input
        assert input_set.qcinput is new_input

    def test_write_input_and_from_directory(self, water, tmp_path):
        input_set = OptSetGenerator(smd_solvent="water").get_input_set(water)
        input_set.write_input(tmp_path / "calc")
        assert os.listdir(tmp_path / "calc") == ["mol.qin"]

        from_file = QCInput.from_file(tmp_path / "calc" / "mol.qin")
        assert str(from_file) == str(input_set.qcinput)

        from_directory = QCInputSet.from_directory(tmp_path / "calc")
        assert isinstance(from_directory, QCInputSet)
        assert isinstance(from_directory.qcinput, QCInput)
        assert str(from_directory.qcinput) == str(input_set.qcinput)
        assert from_directory.optional_files == {}

        with pytest.raises(FileExistsError):
            input_set.write_input(tmp_path / "calc", overwrite=False)
        input_set.write_input(tmp_path / "calc", overwrite=True)

        with pytest.raises(FileNotFoundError):
            input_set.write_input(tmp_path / "does_not_exist", make_dir=False)
        with pytest.raises(FileNotFoundError):
            QCInputSet.from_directory(tmp_path / "does_not_exist")

    def test_from_directory_gzipped(self, water, tmp_path):
        import gzip

        input_set = SinglePointSetGenerator().get_input_set(water)
        with gzip.open(tmp_path / "mol.qin.gz", mode="wt", encoding="utf-8") as file:
            file.write(str(input_set.qcinput))
        assert str(QCInputSet.from_directory(tmp_path).qcinput) == str(input_set.qcinput)

    def test_custom_smd(self, water, tmp_path):
        custom_smd = "90.00,1.415,0.00,0.735,20.2,0.00,0.00"

        # Q-Chem 5 reads the solvent parameters from a separate file
        input_set = SinglePointSetGenerator(smd_solvent="custom", custom_smd=custom_smd).get_input_set(water)
        assert input_set.optional_files == {"solvent_data": custom_smd}
        assert input_set.qcinput.smx == {"solvent": "other"}

        input_set.write_input(tmp_path)
        assert sorted(os.listdir(tmp_path)) == ["mol.qin", "solvent_data"]
        with open(tmp_path / "solvent_data", encoding="utf-8") as file:
            assert file.read() == custom_smd

        # the legacy input sets write the same two files
        SinglePointSet(water, smd_solvent="custom", custom_smd=custom_smd).write(f"{tmp_path}/legacy.qin")
        with open(tmp_path / "legacy.qin", encoding="utf-8") as legacy, open(tmp_path / "mol.qin") as new:
            assert legacy.read() == new.read()

        assert QCInputSet.from_directory(tmp_path).optional_files == {}
        from_directory = QCInputSet.from_directory(tmp_path, optional_files={"solvent_data": None})
        assert from_directory.optional_files == {"solvent_data": custom_smd}
        assert str(from_directory.qcinput) == str(input_set.qcinput)

        # Q-Chem 6 reads the solvent parameters from the $smx section
        input_set = SinglePointSetGenerator(qchem_version=6, smd_solvent="custom", custom_smd=custom_smd).get_input_set(
            water
        )
        assert input_set.optional_files == {}
        assert input_set.qcinput.smx["epsilon"] == "90.00"

    def test_from_directory_optional_file_class(self, water, tmp_path):
        input_set = SinglePointSetGenerator().get_input_set(water)
        input_set["other.qin"] = OptSetGenerator().get_input_set(water).qcinput
        input_set.write_input(tmp_path)

        from_directory = QCInputSet.from_directory(tmp_path, optional_files={"other.qin": QCInput})
        assert from_directory.qcinput.rem["job_type"] == "sp"
        assert from_directory.optional_files["other.qin"].rem["job_type"] == "opt"

    def test_msonable(self, water):
        input_set = SinglePointSetGenerator(
            smd_solvent="custom", custom_smd="90.00,1.415,0.00,0.735,20.2,0.00,0.00"
        ).get_input_set(water)
        for from_dict in (
            QCInputSet.from_dict(input_set.as_dict()),
            json.loads(json.dumps(input_set, cls=MontyEncoder), cls=MontyDecoder),
        ):
            assert type(from_dict) is QCInputSet
            assert str(from_dict.qcinput) == str(input_set.qcinput)
            assert from_dict.optional_files == input_set.optional_files

    def test_copy(self, water):
        input_set = SinglePointSetGenerator().get_input_set(water)
        deep = copy.deepcopy(input_set)
        deep.qcinput.rem["basis"] = "sto-3g"
        assert input_set.qcinput.rem["basis"] == "def2-tzvpd"
        assert copy.copy(input_set).qcinput is input_set.qcinput
