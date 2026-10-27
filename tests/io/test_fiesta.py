from __future__ import annotations

import os

import pytest

from pymatgen.core.structure import Molecule
from pymatgen.io.fiesta import BasisSetReader, BSEOutput, FiestaInput, FiestaOutput, FiestaRun, Nwchem2Fiesta
from pymatgen.util.testing import TEST_FILES_DIR

TEST_DIR = f"{TEST_FILES_DIR}/io/fiesta"


class TestFiestaInput:
    def setup_method(self):
        coords = [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 1.089000],
            [1.026719, 0.000000, -0.363000],
            [-0.513360, -0.889165, -0.363000],
            [-0.513360, 0.889165, -0.363000],
        ]
        self.coords = coords
        mol = Molecule(["C", "H", "H", "H", "H"], coords)
        self.cell_in = FiestaInput(
            mol,
            correlation_grid={"dE_grid": "0.500", "n_grid": "14"},
            exc_dft_option={"rdVxcpsi": "1"},
            cohsex_options={
                "eigMethod": "C",
                "mix_cohsex": "0.500",
                "nc_cohsex": "0",
                "nit_cohsex": "0",
                "nv_cohsex": "0",
                "resMethod": "V",
                "scf_cohsex_wf": "0",
            },
            gw_options={"nc_corr": "10", "nit_gw": "3", "nv_corr": "10"},
            bse_tddft_options={
                "do_bse": "1",
                "do_tddft": "0",
                "nc_bse": "382",
                "nit_bse": "50",
                "npsi_bse": "1",
                "nv_bse": "21",
            },
        )

    def test_init(self):
        mol = Molecule(["C", "H", "H", "H", "H"], self.coords)
        cell_in = FiestaInput(mol)
        assert cell_in.molecule.spin_multiplicity == 1

    def test_str_and_from_str(self):
        result = (
            "# number of atoms and species\n   5    2\n# number of valence bands\n    5\n"
            "# number of points and spacing in eV for correlation grid\n    14    0.500\n"
            "# relire=1 ou recalculer=0 Exc DFT\n    1\n"
            "# number of COHSEX corrected occp and unoccp bands: C=COHSEX  H=HF\n    0    0   C\n"
            "# number of COHSEX iter, scf on wfns, mixing coeff; V=RI-V  I=RI-D\n    0   V       0       0.500\n"
            "# number of GW corrected occp and unoccp bands\n   10   10\n# number of GW iterations\n    3\n"
            "# dumping for BSE and TDDFT\n    1    0\n"
            "# number of occp. and virtual bands of BSE: nocore and up to 40 eVs\n    21   382\n"
            "# number of excitations needed and number of iterations\n    1   50\n"
            "# list of symbols in order\nC\nH\n"
            "# scaling factor\n    1.000\n# atoms x,y,z cartesian .. will be multiplied by scale\n 0.0 0.0 0.0 1\n"
            " 0.0 0.0 1.089 2\n 1.026719 0.0 -0.363 2\n -0.51336 -0.889165 -0.363 2\n -0.51336 0.889165 -0.363 2"
            "\n            "
        )
        assert str(self.cell_in) == result
        cell_in = FiestaInput.from_str(result)
        assert cell_in.GW_options["nc_corr"] == "10"
        assert cell_in.cohsex_options["eigMethod"] == "C"


class TestFiestaOutput:
    def setup_method(self):
        self.log_fiesta = FiestaOutput(f"{TEST_DIR}/log_fiesta")

    def test_props(self):
        out = self.log_fiesta
        assert out.data[0]["Gaps"]["Egap_QP_Linear"] == "10.4135"
        assert out.data[0]["HOMO"] == {
            "band": "HOMO",
            "eKS": "-7.3029",
            "eQP_Linear": "-9.5142",
            "eQP_SCF": "-8.9264",
            "eQP_old": "-7.7188",
            "eXX": "-15.9483",
            "sigma_c_Linear": "-0.4587",
            "sigma_c_SCF": "0.3900",
            "z": "0.87",
        }

    def test_all_bands(self):
        """Each GW iteration yields HOMO/LUMO band dicts plus a Gaps block."""
        out = self.log_fiesta
        assert len(out.data) > 1
        # last iteration should carry the full HOMO->LUMO ladder
        assert "HOMO" in out.data[-1]
        assert "LUMO" in out.data[-1]
        assert "LUMO+2" in out.data[-1]

        # per-band dict has the nine parsed energy fields
        homo = out.data[-1]["HOMO"]
        assert set(homo) == {
            "band",
            "eKS",
            "eXX",
            "eQP_old",
            "z",
            "sigma_c_Linear",
            "eQP_Linear",
            "sigma_c_SCF",
            "eQP_SCF",
        }
        # the gap block carries the three Egap fields
        assert set(out.data[-1]["Gaps"]) == {"Egap_KS", "Egap_QP_Linear", "Egap_QP_SCF"}


class TestFiestaInputExtras:
    """Extra coverage for the FiestaInput builder / serializer helpers."""

    @staticmethod
    def _mol():
        coords = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.089],
            [1.026719, 0.0, -0.363],
            [-0.51336, -0.889165, -0.363],
            [-0.51336, 0.889165, -0.363],
        ]
        return Molecule(["C", "H", "H", "H", "H"], coords)

    def test_set_gw_options(self):
        in_ = FiestaInput(self._mol())
        in_.set_gw_options(nv_band=5, nc_band=6, n_iteration=7, n_grid=8, dE_grid=0.1)
        assert in_.GW_options == {"nv_corr": 5, "nc_corr": 6, "nit_gw": 7}
        assert in_.correlation_grid == {"n_grid": 8, "dE_grid": 0.1}

    def test_set_bse_options(self):
        in_ = FiestaInput(self._mol())
        in_.set_bse_options(n_excitations=4, nit_bse=9)
        assert in_.bse_tddft_options["npsi_bse"] == 4
        assert in_.bse_tddft_options["nit_bse"] == 9

    def test_dump_bse_data_in_gw_run(self):
        in_ = FiestaInput(self._mol())
        in_.dump_bse_data_in_gw_run(BSE_dump=True)
        assert in_.bse_tddft_options["do_bse"] == 1
        assert in_.bse_tddft_options["do_tddft"] == 0

    def test_dump_tddft_data_in_gw_run(self):
        in_ = FiestaInput(self._mol())
        in_.dump_tddft_data_in_gw_run(tddft_dump=True)
        assert in_.bse_tddft_options["do_bse"] == "0"
        assert in_.bse_tddft_options["do_tddft"] == "1"
        in_.dump_tddft_data_in_gw_run(tddft_dump=False)
        assert in_.bse_tddft_options["do_tddft"] == "0"

    def test_make_full_bse_densities_folder(self, tmp_path):
        in_ = FiestaInput(self._mol())
        assert in_.make_full_bse_densities_folder(str(tmp_path)) == "makedirs FULL_BSE_Densities folder"
        assert os.path.isdir(str(tmp_path / "FULL_BSE_Densities"))

    def test_make_full_bse_densities_folder_already_exists(self, tmp_path):
        # the guard checks os.path.isfile, so a pre-existing FILE at that name
        # triggers the "already exists" branch.
        in_ = FiestaInput(self._mol())
        file = tmp_path / "FULL_BSE_Densities"
        file.write_text("")
        assert in_.make_full_bse_densities_folder(str(tmp_path)) == "FULL_BSE_Densities folder already exists"

    def test_infos_on_system_default(self):
        in_ = FiestaInput(self._mol())
        info = in_.infos_on_system
        assert "Number of atoms = 5" in info
        assert "Dumping data for BSE treatment" in info
        assert "Exchange and correlation energy read from Vxcpsi.mat" in info
        assert "COHSEX level" in info

    def test_infos_on_system_recompute_and_hf(self):
        in_ = FiestaInput(
            self._mol(),
            exc_dft_option={"rdVxcpsi": "0"},
            cohsex_options={
                "eigMethod": "HF",
                "nv_cohsex": "1",
                "nc_cohsex": "2",
                "nit_cohsex": "3",
                "resMethod": "V",
                "scf_cohsex_wf": "0",
                "mix_cohsex": "0.5",
            },
        )
        info = in_.infos_on_system
        assert "re-computed" in info
        assert "HF level" in info

    def test_infos_on_system_tddft_dump(self):
        in_ = FiestaInput(
            self._mol(), bse_tddft_options={"do_bse": "0", "do_tddft": "1", "nc_bse": "1", "nit_bse": "1"}
        )
        info = in_.infos_on_system
        assert "Dumping data for TD-DFT treatment" in info

    def test_molecule_property(self):
        in_ = FiestaInput(self._mol())
        assert isinstance(in_.molecule, Molecule)
        assert in_.molecule.composition.num_atoms == 5

    def test_write_file_and_from_file(self, tmp_path):
        in_ = FiestaInput(self._mol())
        fn = tmp_path / "cell.in"
        in_.write_file(str(fn))
        assert fn.stat().st_size > 0
        restored = FiestaInput.from_file(str(fn))
        assert restored.GW_options["nc_corr"] == "10"
        assert isinstance(restored.molecule, Molecule)

    def test_set_auxiliary_basis_set(self, tmp_path):
        in_ = FiestaInput(Molecule(["C", "H"], [[0, 0, 0], [0, 0, 1]]))
        aux = tmp_path / "aux"
        aux.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        # filenames carry the species + "2" prefix and the requested type
        for sp in ("C", "H"):
            (aux / f"{sp}2.ion_aug_cc_pvtz_RI_Weigend").write_text("dummy")
        in_.set_auxiliary_basis_set(str(out), str(aux), "aug_cc_pvtz")
        files = sorted(os.listdir(out))
        assert files == ["C2.ion", "H2.ion"]

    def test_as_dict_from_dict(self):
        in_ = FiestaInput(self._mol())
        d = in_.as_dict()
        assert "mol" in d
        assert d["correlation_grid"] == in_.correlation_grid
        # from_dict reads the MSONable geometry/symmetry/memory keys it defines
        d["geometry_options"] = in_.cohsex_options
        d["symmetry_options"] = in_.GW_options
        d["memory_options"] = in_.bse_tddft_options
        restored = FiestaInput.from_dict(d)
        assert restored.cohsex_options["eigMethod"] == "C"
        assert restored.GW_options["nc_corr"] == "10"

    def test_from_str_roundtrip(self):
        in_ = FiestaInput(self._mol())
        restored = FiestaInput.from_str(str(in_))
        assert str(restored) == str(in_)
        assert restored.molecule.composition.num_atoms == 5


class TestBasisSetReader:
    @staticmethod
    def _basis_text():
        return (
            "<preamble>\n"
            "basis set for C:\n"
            "aug-cc-pvdz\n"
            "</preamble>\n"
            "  11   4   #    lmax    n_nl_orbital\n"
            "  0   1   2   #    L    zeta     ng\n"
            "   0.8000   0.5000\n"
            "   2.4000   0.5000\n"
            "  1   1   2   #    L    zeta     ng\n"
            "   1.2000   0.7000\n"
            "   3.6000   0.3000\n"
        )

    def test_parse_file(self, tmp_path):
        fn = tmp_path / "C.basis"
        fn.write_text(self._basis_text())
        br = BasisSetReader(str(fn))
        # lmax / n_nlo / preamble plus the parsed "l_zeta_ng" blocks
        assert br.data["lmax"] == "11"
        assert br.data["n_nlo"] == "4"
        assert br.data["preamble"] == ["basis set for C:", "aug-cc-pvdz", "</preamble>"]
        found = tuple(k for k in br.data if k in {"0_1_2", "1_1_2"})
        assert found == ("0_1_2", "1_1_2")

    def test_set_n_nlmo_does_not_mutate(self, tmp_path):
        # regression: set_n_nlmo used to pop keys off the live dict, which
        # then broke infos_on_basis_set with a KeyError.
        fn = tmp_path / "C.basis"
        fn.write_text(self._basis_text())
        br = BasisSetReader(str(fn))
        assert br.data["n_nlmo"] == "4"  # 2*0+1 + 2*1+1
        # the live dict must still carry lmax / n_nlo after construction
        assert "lmax" in br.data
        assert "n_nlo" in br.data

    def test_infos_on_basis_set(self, tmp_path):
        fn = tmp_path / "C.basis"
        fn.write_text(self._basis_text())
        br = BasisSetReader(str(fn))
        info = br.infos_on_basis_set()
        assert "Maximum angular momentum = 11" in info
        assert "Number of atomics orbitals = 4" in info
        assert "Number of nlm orbitals = 4" in info


class TestBSEOutput:
    @staticmethod
    def _log():
        return (
            "FULL BSE eig.(eV), osc. strength and dipoles:\n"
            "exiton   1 :  3.6500( -0.1200 )  |  weight\n"
            "exiton   2 :  5.2000( -0.0800 )  |  weight\n"
            "FULL BSE main valence -> conduction transitions weight:\n"
            " total  time:    12.34  seconds\n"
            " program returned normally\n"
        )

    def test_exiton_parsing(self, tmp_path):
        fn = tmp_path / "log_bse"
        fn.write_text(self._log())
        bse = BSEOutput(str(fn))
        ex = bse.exiton
        assert ex["1"] == {"bse_eig": "3.6500", "osc_strength": "-0.1200"}
        assert ex["2"] == {"bse_eig": "5.2000", "osc_strength": "-0.0800"}
        assert ex["total_time"] == "12.34"
        assert ex["end_normally"] is True


class TestMSONable:
    def test_nwchem2fiesta_as_dict_from_dict(self):
        nw = Nwchem2Fiesta(folder="/data", filename="run", log_file="log")
        d = nw.as_dict()
        assert d["@class"] == "Nwchem2Fiesta"
        assert d["filename"] == "run"
        assert d["folder"] == "/data"
        restored = Nwchem2Fiesta.from_dict(d)
        assert restored.filename == "run"
        assert restored.folder == "/data"

    def test_fiesta_run_as_dict_from_dict(self):
        fr = FiestaRun(grid=(2, 2, 2), log_file="mylog")
        d = fr.as_dict()
        assert d["@class"] == "FiestaRun"
        assert d["grid"] == (2, 2, 2)
        restored = FiestaRun.from_dict(d)
        assert restored.grid == (2, 2, 2)
        assert restored.log_file == "mylog"

    def test_fiesta_run_bad_grid_raises(self):
        # a grid that is neither 2- nor 3-dimensional is rejected
        with pytest.raises(ValueError, match="Wrong grid size"):
            FiestaRun(grid=(5,)).run()
