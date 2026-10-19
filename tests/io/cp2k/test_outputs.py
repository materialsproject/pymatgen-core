from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest import approx

from pymatgen.core import Molecule, Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.io.cp2k.outputs import Cp2kOutput, parse_dos, parse_energy_file, parse_pdos
from pymatgen.util.testing import TEST_FILES_DIR

TEST_DIR = f"{TEST_FILES_DIR}/io/cp2k"


class TestCp2kOutput:
    def setup_method(self):
        self.out = Cp2kOutput(f"{TEST_DIR}/cp2k.out", auto_load=True)

    def test_files(self):
        """Can find files successfully."""
        self.out.parse_files()
        assert len(self.out.filenames["PDOS"]) == 1
        assert len(self.out.filenames["PDOS"]) == 1
        assert len(self.out.filenames["band_structure"]) == 1
        assert len(self.out.filenames["hyperfine_tensor"]) == 1
        assert len(self.out.filenames["chi_tensor"]) == 1
        assert len(self.out.filenames["g_tensor"]) == 1

    def test_run_info(self):
        """Can extract run info from out file."""
        assert self.out.spin_polarized
        assert self.out.completed
        assert self.out.num_warnings == [[2]]
        assert self.out.charge == 0
        assert self.out.cp2k_version == "2022.1"
        assert self.out.run_type.upper() == "ENERGY_FORCE"

    def energy_force(self):
        """Can get energy and forces."""
        assert self.out.final_energy == approx(-197.40000341992783)
        assert_allclose(
            self.out.data["forces"][0],
            [
                [-0.00000001, -0.00000001, -0.00000001],
                [0.00000002, 0.00000002, 0.00000002],
            ],
        )

    def test_band(self):
        """Can parse bandstructure files."""
        assert self.out.band_structure
        assert self.out.band_structure.get_band_gap().get("energy") == approx(0.27940141999999923)

    def test_dos(self):
        """Can parse dos files."""
        assert self.out.data["pdos"]["Si_1"]["s"]["efermi"] == approx(-6.7370756409404455)
        assert self.out.data["tdos"].energies[0] == approx(-6.781065751604123)

    def test_chi(self):
        self.out.parse_chi_tensor()
        assert len(self.out.data["chi_total"]) == 1
        assert self.out.data["PV1"][0] == approx(0.1587)
        assert self.out.data["PV2"][0] == approx(0.4582)
        assert self.out.data["PV3"][0] == approx(0.4582)
        assert self.out.data["ISO"][0] == approx(0.3584)
        assert self.out.data["ANISO"][0] == approx(0.1498)
        assert_allclose(
            self.out.data["chi_soft"][0],
            [
                [5.9508, -1.6579, -1.6579],
                [-1.6579, 5.9508, -1.6579],
                [-1.6579, -1.6579, 5.9508],
            ],
        )
        assert_allclose(self.out.data["chi_local"][0], np.zeros((3, 3)))
        assert_allclose(
            self.out.data["chi_total"][0],
            [
                [5.9508, -1.6579, -1.6579],
                [-1.6579, 5.9508, -1.6579],
                [-1.6579, -1.6579, 5.9508],
            ],
        )
        assert_allclose(
            self.out.data["chi_total_ppm_cgs"][0],
            [
                [0.3584, -0.0998, -0.0998],
                [-0.0998, 0.3584, -0.0998],
                [-0.0998, -0.0998, 0.3584],
            ],
        )

    def test_gtensor(self):
        self.out.parse_gtensor()
        assert len(self.out.data["gtensor_total"]) == 1
        assert_allclose(self.out.data["gmatrix_zke"][0], np.zeros((3, 3)))
        assert_allclose(self.out.data["gmatrix_so"][0], np.zeros((3, 3)))
        assert_allclose(self.out.data["gmatrix_soo"][0], np.zeros((3, 3)))
        assert_allclose(
            self.out.data["gmatrix_total"][0],
            [
                [2.0023193044, 0.0, 0.0],
                [0.0, 2.0023193044, 0.0],
                [0.0, 0.0, 2.0023193044],
            ],
        )
        assert_allclose(
            self.out.data["gtensor_total"][0],
            [
                [2.0023193044, 0.0, 0.0],
                [0.0, 2.0023193044, 0.0],
                [0.0, 0.0, 2.0023193044],
            ],
        )
        assert_allclose(
            self.out.data["delta_g"][0],
            [
                [0.7158445077, -0.6982592888, 0.0007786468],
                [0.4071365712, 0.4164835801, -0.8128845182],
                [0.5672798720, 0.5822159333, 0.5824243761],
            ],
        )

    def test_hyperfine(self):
        self.out.parse_hyperfine()
        dat = self.out.data["hyperfine_tensor"]
        ref = [
            [
                [0.0000000000, 0.0000001288, 0.0000001288],
                [0.0000001288, -0.0000000000, 0.0000001288],
                [0.0000001288, 0.0000001288, 0.0000000000],
            ],
            [
                [0.0000000000, -0.0000001288, -0.0000001288],
                [-0.0000001288, 0.0000000000, -0.0000001288],
                [-0.0000001288, -0.0000001288, 0.0000000000],
            ],
        ]
        assert_allclose(dat[0], ref)

    def test_properties(self):
        assert self.out.project_name is None  # "Project name" is parsed under the "Project_name" key
        assert self.out.data["global"]["Project_name"] == "Static"
        assert self.out.calculation_type == "GGA"
        assert self.out.multiplicity == 0
        assert not self.out.is_molecule
        assert not self.out.is_metal
        assert not self.out.is_hubbard
        assert self.out.complete_dos is None
        assert self.out.data["dft"]["functional"] == ["PBE"]
        assert self.out.data["dft"]["cutoffs"] == {"density": 1e-10, "gradient": 1e-10, "tau": 1e-10}
        assert self.out.data["scf"] == {"max_scf": "1", "eps_scf": "1"}
        info = self.out.data["atomic_kind_info"]["Si_1"]
        assert info["orbital_basis_set"] == "DZVP-MOLOPT-SR-GTH-q4"
        assert info["pseudo_potential"] == "GTH-PBE-q4"
        assert info["valence_electrons"] == 4
        assert info["core_electrons"] == 10
        assert info["element"] == "Si"

    def test_structure_energy_forces(self):
        assert self.out.final_structure == self.out.initial_structure
        assert self.out.initial_structure.formula == "Si2"
        assert self.out.initial_structure.lattice.volume == approx(40.88829, abs=1e-4)
        assert self.out.initial_structure.site_properties["ghost"] == [False, False]
        assert self.out.final_energy == approx(-7.254316323154326 * Ha_to_eV)
        assert_allclose(self.out.data["forces"][0], [[-1e-8, -1e-8, -1e-8], [2e-8, 2e-8, 2e-8]])
        # Stress tensor in this file is Infinity/NaN, which the regex does not match
        assert "stress_tensor" not in self.out.data
        assert len(self.out.ionic_steps) == 1
        step = self.out.ionic_steps[0]
        assert step["E"] == approx(self.out.final_energy)
        assert step["stress_tensor"] is None
        assert self.out.data["scf_converged"] == [True]
        assert self.out.data["geo_opt_not_converged"] == []

    def test_band_edges(self):
        assert self.out.vbm == approx(6.1938852)
        assert self.out.cbm == approx(6.47328662)
        assert self.out.efermi == approx(6.33358591)

    def test_timing(self):
        assert self.out.timing["CP2K"] == {
            "calls": {"max": 1},
            "asd": 1.0,
            "self_time": {"average": 0.023, "maximum": 0.023},
            "total_time": {"average": 4.006, "maximum": 4.006},
        }
        assert self.out.timing["dbcsr_zero"]["total_time"]["maximum"] == approx(0.082)

    def test_qs_params(self):
        self.out.parse_qs_params()
        qs = self.out.data["QS"]
        assert qs["Method:"] == "GPW"
        assert qs["Number_of_grid_levels:"] == 5
        assert qs["Multi_grid_cutoffs_[a.u.]"] == {1: 50.0, 2: 16.7, 3: 5.6, 4: 1.9, 5: 0.6}

    def test_misc_parsers(self):
        self.out.parse_total_numbers()
        assert self.out.data["atomic_kinds"] == [["1"]]
        assert self.out.data["atoms"] == [["2"]]
        assert self.out.data["shells"] == [["10"]]

        self.out.parse_overlap_condition()
        assert self.out.data["overlap_condition_number"] == []

        self.out.parse_plus_u_params()
        assert self.out.data["dft_plus_u_method"] == []

        self.out.parse_opt_steps()
        assert self.out.data["total_energy"] == []
        assert self.out.data["decrease_in_energy"] == []

    def test_scf_opt(self):
        self.out.parse_scf_opt()
        assert self.out.data["scf_time"] == [[0.4]]
        assert self.out.data["convergence"] == [[approx(0.8535836)]]
        assert self.out.data["electronic_steps"] == [[approx(-8.3426168857)]]

    def test_mulliken(self, caplog):
        # The spin-polarized Mulliken table (extra spin moment column) does not match the row pattern
        assert self.out.parse_mulliken() is None
        assert "not yet implemented" not in caplog.text

    def test_hirshfeld(self):
        self.out.parse_hirshfeld()
        hirshfeld = self.out.structures[0].site_properties["hirshfeld"]
        assert len(hirshfeld) == 2
        assert hirshfeld[0]["net_charge"] == "-0.000"
        assert hirshfeld[1]["spin_moment"] == "0.000"

    def test_hirshfeld_population(self):
        self.out.parse_hirshfeld()
        assert self.out.structures[0].site_properties["hirshfeld"][0]["population"] == ("2.000", "2.000")

    def test_mo_eigenvalues(self):
        self.out.parse_mo_eigenvalues()
        eigs = self.out.data["eigenvalues"]
        assert len(eigs) == 1
        assert len(eigs[0]["occupied"][Spin.up]) == 4
        assert len(eigs[0]["occupied"][Spin.down]) == 4
        assert len(eigs[0]["unoccupied"][Spin.up]) == 22
        assert len(eigs[0]["unoccupied"][Spin.down]) == 22
        assert eigs[0]["occupied"][Spin.up][0] == approx(-0.24758290 * Ha_to_eV)
        assert self.out.vbm == approx(0.18727300 * Ha_to_eV)
        assert self.out.cbm == approx(0.28117070 * Ha_to_eV)
        assert self.out.efermi == approx(5.095958)
        assert len(self.out.data["tdos"].energies) == 26

    def test_homo_lumo(self):
        self.out.parse_homo_lumo()
        assert self.out.data["band_gap"] == {Spin.up: [], Spin.down: []}
        assert self.out.band_gap is None
        assert self.out.is_metal

    def test_as_dict(self):
        dct = self.out.as_dict()
        assert dct["run_type"].upper() == "ENERGY_FORCE"
        assert dct["total_time"] == approx(4.006)
        assert dct["ran_successfully"]
        assert dct["cp2k_version"] == "2022.1"
        assert dct["input"]["calculation_type"] == "GGA"
        assert dct["output"]["energy"] == approx(self.out.final_energy)
        assert dct["output"]["energy_per_atom"] == approx(self.out.final_energy / 2)
        assert dct["output"]["bandgap"] == approx(self.out.cbm - self.out.vbm)
        assert dct["output"]["is_metal"] is False
        assert len(dct["output"]["ionic_steps"]) == 1
        assert dct["composition"] == {"Si": 2.0}

    @pytest.mark.parametrize(
        ("functional", "hfx", "expected"),
        [
            ([None], {}, "None"),
            (["B3LYP"], {}, "Hybrid"),
            (["PBE"], {"Interaction_Potential": "COULOMB", "FRACTION": 0.25}, "Hybrid"),
            (["TPSS"], {}, "METAGGA"),
            (["PBE"], {}, "GGA"),
            (["PADE"], {}, "LDA"),
            (["FOO"], {}, "Unknown"),
            (["PBE", "LYP"], {}, "Mixed: PBE, LYP"),
            (["PBE", "HYP"], {}, "Hybrid"),
        ],
    )
    def test_calculation_type(self, functional, hfx, expected):
        self.out.data["dft"]["functional"] = functional
        self.out.data["dft"]["hfx"] = hfx
        assert self.out.calculation_type == expected

    def test_calculation_type_u_vdw(self):
        self.out.data["atomic_kind_info"]["Si_1"]["DFT_PLUS_U"] = {"L": 2, "U_MINUS_J": 0.1}
        self.out.data["dft"]["vdw"] = "D3"
        assert self.out.is_hubbard
        assert self.out.calculation_type == "GGA+U+VDW"

    @pytest.mark.parametrize("method", ["parse_nmr_shift", "parse_tddfpt", "parse_raman"])
    def test_not_implemented(self, method):
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            getattr(self.out, method)()

    def test_gauss_smear(self):
        densities = [1.0, 2.0]
        assert Cp2kOutput._gauss_smear(densities, [0.0, 1.0], 10, None) is densities
        smeared = Cp2kOutput._gauss_smear(densities, [0.0, 1.0], 101, 0.1)
        assert smeared.shape == (101,)
        assert smeared[0] == approx(1 / (np.sqrt(np.pi) * 0.1), rel=1e-3)
        assert smeared[-1] == approx(2 / (np.sqrt(np.pi) * 0.1), rel=1e-3)

    def test_read_table_pattern_attribute_name(self):
        table = self.out.read_table_pattern(
            header_pattern=r"ATOMIC FORCES.+Z",
            row_pattern=r"\s+\d+\s+\d+\s+(?P<el>\w+)\s+(?P<x>-?\d+\.\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+",
            footer_pattern=r"SUM OF ATOMIC FORCES",
            attribute_name="force_x",
        )
        assert table == [{"el": "Si", "x": "-0.00000001"}, {"el": "Si", "x": "0.00000002"}]
        assert self.out.data["force_x"] is table


def _copy_run(dest: Path, replacements: dict[str, str] | None = None, *, with_input: bool = True) -> Path:
    """Copy cp2k.out (and optionally cp2k.inp) to dest, applying text replacements to cp2k.out."""
    text = Path(f"{TEST_DIR}/cp2k.out").read_text(encoding="utf-8")
    for old, new in (replacements or {}).items():
        assert old in text
        text = text.replace(old, new)
    (dest / "cp2k.out").write_text(text, encoding="utf-8")
    if with_input:
        for name in ("cp2k.inp", "include.inc"):
            shutil.copy(f"{TEST_DIR}/{name}", dest / name)
    return dest / "cp2k.out"


LATTICE_ROW = "0.0 2.734364 2.734364 2.734364 0.0 2.734364 2.734364 2.734364 0.0 40.888293"


class TestCp2kOutputModified:
    """Tests on modified copies of cp2k.out to exercise branches the reference run does not hit."""

    def test_not_completed(self, tmp_path):
        filename = _copy_run(tmp_path, {"PROGRAM ENDED AT": "PROGRAM DIED AT"})
        with pytest.raises(ValueError, match="did not finish running"):
            Cp2kOutput(filename, auto_load=True)

    def test_unconverged_warnings(self, tmp_path):
        filename = _copy_run(
            tmp_path,
            {
                "SCF run converged in": "SCF run NOT converged in",
                " The number of warnings": " MAXIMUM NUMBER OF OPTIMIZATION STEPS REACHED\n The number of warnings",
            },
        )
        out = Cp2kOutput(filename)
        with (
            pytest.warns(UserWarning, match="at least one unconverged SCF cycle"),
            pytest.warns(UserWarning, match="Geometry optimization did not converge"),
        ):
            out.convergence()
        assert out.data["scf_converged"] == [False]

    def test_missing_input(self, tmp_path):
        filename = _copy_run(tmp_path, with_input=False)
        with pytest.warns(UserWarning, match="Original input file not found"):
            out = Cp2kOutput(filename)
        assert out.input is None
        assert out.data["dft"]["functional"] == ["PBE"]
        assert out.calculation_type == "GGA"
        with pytest.warns(UserWarning, match="Reading cell params from summary"):
            lattice = out.parse_cell_params()
        assert_allclose(lattice, [[0, 2.734, 2.734], [2.734, 0, 2.734], [2.734, 2.734, 0]])

    def test_no_aux_files(self, tmp_path):
        out = Cp2kOutput(_copy_run(tmp_path), auto_load=True)
        assert out.band_structure is None
        assert out.parse_hyperfine() is None
        assert out.parse_gtensor() is None
        assert out.parse_chi_tensor() is None
        assert out.data["pdos"] == {}
        assert out.data["tdos"] is not None  # falls back to the MO eigenvalues in the output file
        assert out.vbm == approx(0.18727300 * Ha_to_eV)
        assert out.cbm == approx(0.28117070 * Ha_to_eV)

    def test_vdw(self, tmp_path):
        filename = _copy_run(
            tmp_path,
            {
                " FUNCTIONAL| PBE:": " vdW POTENTIAL| Pair potential: DFT-D3 (Version 3.1)\n"
                " vdW POTENTIAL| BJ damping\n FUNCTIONAL| PBE:"
            },
        )
        out = Cp2kOutput(filename)
        assert out.data["dft"]["vdw"] == "D3(BJ)"
        assert out.calculation_type == "GGA+VDW"

    def test_vdw_unknown(self, tmp_path):
        filename = _copy_run(tmp_path, {" FUNCTIONAL| PBE:": " vdW POTENTIAL| Some new potential\n FUNCTIONAL| PBE:"})
        assert Cp2kOutput(filename).data["dft"]["vdw"] == "Some new potential"

    def test_dft_plus_u(self, tmp_path):
        filename = _copy_run(
            tmp_path,
            {
                "     Orbital Basis Set ": "     DFT+U correction\n       Angular momentum:      2\n"
                "       U(eff) = (U - J) value in [a.u.]:      0.1\n\n     Orbital Basis Set "
            },
        )
        out = Cp2kOutput(filename)
        assert out.data["atomic_kind_info"]["Si_1"]["DFT_PLUS_U"] == {"L": 2, "U_MINUS_J": 0.1}
        assert out.is_hubbard
        assert out.data["dft"]["dft_plus_u"]
        assert out.calculation_type == "GGA+U"

    def test_molecule(self, tmp_path):
        filename = _copy_run(tmp_path, {"POISSON| Periodicity                    ": "POISSON| Periodicity  NONE  #"})
        out = Cp2kOutput(filename)
        assert out.is_molecule
        mol = out.parse_initial_structure()
        assert isinstance(mol, Molecule)
        assert mol.formula == "Si2"

    def test_molecule_trajectory(self, tmp_path):
        filename = _copy_run(tmp_path, {"POISSON| Periodicity                    ": "POISSON| Periodicity  NONE  #"})
        pos = "".join(f"2\n i = {i}, E = -7.25\nSi 1.367182 1.367182 1.367182\nSi 0.0 0.0 0.{i}\n" for i in range(2))
        (tmp_path / "Static-pos-1.xyz").write_text(pos, encoding="utf-8")
        out = Cp2kOutput(filename)
        out.parse_structures()
        assert len(out.structures) == 2
        assert all(isinstance(mol, Molecule) for mol in out.structures)
        assert_allclose(out.final_structure.cart_coords[1], [0, 0, 0.1])

    def test_ionic_steps_lazy(self, tmp_path):
        out = Cp2kOutput(_copy_run(tmp_path))
        steps = out.parse_ionic_steps()
        assert len(steps) == 1
        assert steps[0]["E"] == approx(-7.254316323154326 * Ha_to_eV)
        assert_allclose(steps[0]["forces"], [[-1e-8, -1e-8, -1e-8], [2e-8, 2e-8, 2e-8]])
        assert steps[0]["structure"] == out.initial_structure

    def test_abc_cell(self, tmp_path):
        filename = _copy_run(tmp_path)
        inp = tmp_path / "cp2k.inp"
        text = inp.read_text(encoding="utf-8")
        text = text.replace("\t\t\tA 0.0 2.734364 2.734364\n", "\t\t\tABC 5.0 6.0 7.0\n")
        text = text.replace("\t\t\tB 2.734364 0.0 2.734364\n", "").replace("\t\t\tC 2.734364 2.734364 0.0\n", "")
        inp.write_text(text, encoding="utf-8")
        assert Cp2kOutput(filename).parse_cell_params() == [[5.0, 0, 0], [0, 6.0, 0], [0, 0, 7.0]]

    def test_restricted_kohn_sham(self, tmp_path):
        hirshfeld_row = "Si     1       4.000    2.000   2.000           -0.000     -0.000"
        filename = _copy_run(
            tmp_path,
            {
                "Kohn-Sham calculation               UKS": "Kohn-Sham calculation               RKS",
                hirshfeld_row: "Si     1       4.000    4.000     -0.000",
                hirshfeld_row.replace("-", " "): "Si     1       4.000    4.000      0.000",
                " The number of warnings": "  HOMO - LUMO gap [eV] :    2.555087\n The number of warnings",
            },
        )
        out = Cp2kOutput(filename, auto_load=True)
        assert not out.spin_polarized
        assert out.band_gap == approx(2.555087)
        assert out.data["band_gap"] == {Spin.up: [2.555087], Spin.down: [2.555087]}

        out.parse_mo_eigenvalues()
        assert out.data["vbm"] == {Spin.up: approx(0.18727300 * Ha_to_eV), Spin.down: None}
        assert out.data["cbm"] == {Spin.up: approx(0.28117070 * Ha_to_eV), Spin.down: None}
        assert out.efermi == approx(5.095958)

        out.parse_hirshfeld()
        hirshfeld = out.structures[0].site_properties["hirshfeld"]
        assert hirshfeld == [
            {"population": "4.000", "net_charge": "-0.000"},
            {"population": "4.000", "net_charge": "0.000"},
        ]

    def test_homo_lumo_spin_polarized(self, tmp_path):
        filename = _copy_run(
            tmp_path,
            {
                " The number of warnings": "  HOMO - LUMO gap [eV] :    2.0\n  HOMO - LUMO gap [eV] :    3.0\n"
                " The number of warnings"
            },
        )
        out = Cp2kOutput(filename)
        out.parse_homo_lumo()
        assert sorted(out.data["band_gap"][Spin.up] + out.data["band_gap"][Spin.down]) == [2.0, 3.0]
        assert out.band_gap == approx(2.5)

    def test_stress_from_output(self, tmp_path):
        stress_rows = (
            " STRESS|      x                 Infinity           -Infinity           -Infinity\n"
            " STRESS|      y                -Infinity            Infinity           -Infinity\n"
            " STRESS|      z                -Infinity           -Infinity            Infinity\n"
        )
        finite_rows = (
            " STRESS|      x           1.00000000E+00     2.00000000E+00     3.00000000E+00\n"
            " STRESS|      y           4.00000000E+00     5.00000000E+00     6.00000000E+00\n"
            " STRESS|      z           7.00000000E+00     8.00000000E+00     9.00000000E+00\n"
        )
        out = Cp2kOutput(_copy_run(tmp_path, {stress_rows: finite_rows}))
        out.parse_stresses()
        assert out.data["stress_tensor"] == [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]

    def test_mo_eigenvalues_tdos_spin(self, tmp_path):
        filename = _copy_run(tmp_path)
        text = filename.read_text(encoding="utf-8")
        idx = text.index("unoccupied subspace spin            2")
        filename.write_text(text[:idx] + text[idx:].replace("0.28117070", "0.29000000", 1), encoding="utf-8")
        out = Cp2kOutput(filename)
        out.parse_mo_eigenvalues()
        assert out.data["tdos"].energies[4] == approx(0.28117070 * Ha_to_eV)

    def test_trajectory_files(self, tmp_path):
        filename = _copy_run(tmp_path)
        energies = [-7.25, -7.26]
        pos = "".join(
            f"2\n i = {i}, time = {i * 0.5:.3f}, E = {energy}\nSi 1.367182 1.367182 1.367182\nSi 0.0 0.0 0.{i}\n"
            for i, energy in enumerate(energies)
        )
        (tmp_path / "Static-pos-1.xyz").write_text(pos, encoding="utf-8")
        frc = "".join(
            f"2\n i = {i}, time = {i * 0.5:.3f}, E = {energy}\nSi 0.1 0.2 0.3\nSi -0.1 -0.2 -0.{i + 3}\n"
            for i, energy in enumerate(energies)
        )
        (tmp_path / "Static-frc-1.xyz").write_text(frc, encoding="utf-8")
        (tmp_path / "Static-1.cell").write_text(
            f"# Step Time Ax Ay Az Bx By Bz Cx Cy Cz Volume\n0 0.0 {LATTICE_ROW}\n1 0.5 {LATTICE_ROW}\n",
            encoding="utf-8",
        )
        (tmp_path / "Static-1.stress").write_text(
            "# Step Time xx xy xz yx yy yz zx zy zz\n0 0.0 1 2 3 4 5 6 7 8 9\n1 0.5 9 8 7 6 5 4 3 2 1\n",
            encoding="utf-8",
        )
        for name in ("Static-1.restart", "Static-1.restart.bak-1", "Static-RESTART.wfn", "Static-RESTART.wfn.bak-1"):
            (tmp_path / name).touch()

        out = Cp2kOutput(filename, auto_load=True)
        assert out.filenames["restart"] == [str(tmp_path / "Static-1.restart")]
        assert out.filenames["restart.bak"] == [str(tmp_path / "Static-1.restart.bak-1")]
        assert out.filenames["wfn"] == str(tmp_path / "Static-RESTART.wfn")
        assert out.filenames["wfn.bak"] == [str(tmp_path / "Static-RESTART.wfn.bak-1")]

        assert len(out.structures) == 2
        assert isinstance(out.final_structure, Structure)
        assert_allclose(out.final_structure.cart_coords[1], [0, 0, 0.1], atol=1e-10)
        assert out.data["total_energy"] == approx(np.multiply(energies, Ha_to_eV))
        assert out.final_energy == approx(-7.26 * Ha_to_eV)
        assert_allclose(out.data["forces"][1], [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.4]])
        assert out.data["stress_tensor"][1] == [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        assert len(out.ionic_steps) == 2
        assert out.ionic_steps[0]["stress_tensor"] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_single_frame_cell_and_stress(self, tmp_path):
        filename = _copy_run(tmp_path)
        (tmp_path / "Static-1.cell").write_text(f"# header\n0 0.0 {LATTICE_ROW}\n", encoding="utf-8")
        (tmp_path / "Static-1.stress").write_text("# header\n0 0.0 1 2 3 4 5 6 7 8 9\n", encoding="utf-8")
        out = Cp2kOutput(filename)
        out.parse_structures()
        assert out.structures == [out.initial_structure]
        out.parse_stresses()
        assert out.data["stress_tensor"] == [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]

    def test_explicit_lattice_file(self, tmp_path):
        filename = _copy_run(tmp_path)
        cell = tmp_path / "my_cell.txt"
        cell.write_text(f"# header\n0 0.0 {LATTICE_ROW}\n1 0.5 {LATTICE_ROW}\n", encoding="utf-8")
        pos = "".join(f"2\n i = {i}, E = -7.25\nSi 1.367182 1.367182 1.367182\nSi 0.0 0.0 0.0\n" for i in range(2))
        traj = tmp_path / "traj.xyz"
        traj.write_text(pos, encoding="utf-8")
        out = Cp2kOutput(filename)
        out.parse_structures(trajectory_file=str(traj), lattice_file=str(cell))
        assert len(out.structures) == 2

    @pytest.mark.parametrize("pattern", ["Static-pos-{}.xyz", "Static-{}.cell"])
    def test_ambiguous_files(self, tmp_path, pattern):
        filename = _copy_run(tmp_path)
        for i in (1, 2):
            (tmp_path / pattern.format(i)).write_text(f"# header\n0 0.0 {LATTICE_ROW}\n", encoding="utf-8")
        out = Cp2kOutput(filename)
        with pytest.raises(FileNotFoundError, match="More than one exist"):
            out.parse_structures()

    def test_dos_spin_and_ldos(self, tmp_path):
        filename = _copy_run(tmp_path)
        pdos_text = Path(f"{TEST_DIR}/Static-ALPHA_k1-1.pdos").read_text(encoding="utf-8")
        (tmp_path / "Static-ALPHA_k1-1.pdos").write_text(pdos_text, encoding="utf-8")
        (tmp_path / "Static-BETA_k1-1.pdos").write_text(pdos_text, encoding="utf-8")
        for i in (1, 2):
            ldos = pdos_text.replace("for atomic kind Si_1", f"for list {i} atoms")
            for spin in ("ALPHA", "BETA"):
                (tmp_path / f"Static-{spin}_list{i}-1.pdos").write_text(ldos, encoding="utf-8")

        out = Cp2kOutput(filename, auto_load=True)
        assert len(out.filenames["PDOS"]) == 2
        assert len(out.filenames["LDOS"]) == 4
        assert set(out.data["pdos"]["Si_1"]["s"]["densities"]) == {"1", "-1"}
        tdos = out.data["tdos"]
        assert set(tdos.densities) == {Spin.up, Spin.down}
        assert_allclose(tdos.densities[Spin.up], tdos.densities[Spin.down])
        assert set(out.data["ldos"]) == {"1", "2"}
        assert set(out.data["ldos"]["2"]["s"]["densities"]) == {"1", "-1"}
        cdos = out.complete_dos
        assert isinstance(cdos, CompleteDos)
        assert len(cdos.pdos) == 2


class TestParseFunctions:
    def test_parse_energy_file(self, tmp_path):
        ener = tmp_path / "Static-1.ener"
        ener.write_text(
            "# Step Nr. Time[fs] Kin.[a.u.] Temp[K] Pot.[a.u.] Cons Qty[a.u.] UsedTime[s]\n"
            "0 0.1 300.0 -7.25 -7.15 1.0\n"
            "1 0.2 310.0 -7.26 -7.06 2.0\n",
            encoding="utf-8",
        )
        dct = parse_energy_file(ener)
        assert set(dct) == {"step", "kinetic_energy", "temp", "potential_energy", "conserved_quantity", "used_time"}
        assert_allclose(dct["kinetic_energy"], np.array([0.1, 0.2]) * Ha_to_eV)
        assert_allclose(dct["potential_energy"], np.array([-7.25, -7.26]) * Ha_to_eV)
        assert_allclose(dct["temp"], [300.0, 310.0])

    def test_parse_dos(self):
        dos = parse_dos(f"{TEST_DIR}/CP2K-1.dos")
        assert dos.energies[0] == approx(-0.24919957 * Ha_to_eV)
        assert dos.efermi == approx(-0.24919957 * Ha_to_eV + 1e-6)
        assert set(dos.densities) == {Spin.up, Spin.down}
        assert_allclose(dos.densities[Spin.down], [0.0625, 0.0])

    def test_parse_pdos(self):
        pdos, tdos = parse_pdos(f"{TEST_DIR}/Static-ALPHA_k1-1.pdos", total=True)
        assert set(pdos["Si_1"]) == {
            Orbital.s,
            Orbital.py,
            Orbital.pz,
            Orbital.px,
            Orbital.dxy,
            Orbital.dyz,
            Orbital.dz2,
            Orbital.dxz,
            Orbital.dx2,
        }
        # two zero-density points are inserted in the gap
        assert len(tdos.energies) == 4
        assert tdos.efermi == approx(-0.247583 * Ha_to_eV + 1e-6)
        assert_allclose(tdos.densities[Spin.up], [1, 0, 0, 0.18690139 * 3 + 0.14643195 * 2 + 0.14643194])

    def test_parse_pdos_spin_and_labels(self, tmp_path):
        labels = ["p", "d", "f", "f-3", "f-2", "f-1", "f0", "f+1", "f+2", "f+3"]
        header = "#     MO Eigenvalue [a.u.]      Occupation" + "".join(f"{lbl:>18}" for lbl in labels)
        rows = ["1  -0.25  1.0" + " 0.1" * len(labels), "2   0.28  0.0" + " 0.2" * len(labels)]
        dos_file = tmp_path / "custom-BETA_k1-1.pdos"
        dos_file.write_text(
            "# Projected DOS for atomic kind Fe at iteration step i = 0, E(Fermi) = 0.0 a.u.\n"
            + header
            + "\n"
            + "\n".join(rows)
            + "\n",
            encoding="utf-8",
        )
        pdos = parse_pdos(str(dos_file))
        # "f" and "f-3" both map to f_3
        assert set(pdos["Fe"]) == {
            Orbital.px,
            Orbital.dxy,
            Orbital.f_3,
            Orbital.f_2,
            Orbital.f_1,
            Orbital.f0,
            Orbital.f1,
            Orbital.f2,
            Orbital.f3,
        }
        assert set(pdos["Fe"][Orbital.px].densities) == {Spin.down}

        pdos = parse_pdos(str(dos_file), spin_channel=1)
        assert set(pdos["Fe"][Orbital.f3].densities) == {Spin.up}
