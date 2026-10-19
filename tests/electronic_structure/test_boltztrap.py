from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import orjson
import pytest
from monty.serialization import loadfn
from numpy.testing import assert_allclose
from pytest import approx
from scipy.integrate import quad
from scipy.special import expit

from pymatgen.core import constants
from pymatgen.core.units import Energy
from pymatgen.electronic_structure.bandstructure import BandStructure, Kpoint
from pymatgen.electronic_structure.boltztrap import (
    BoltztrapAnalyzer,
    BoltztrapError,
    BoltztrapRunner,
    compare_sym_bands,
    eta_from_seebeck,
    read_cube_file,
    seebeck_eff_mass_from_carr,
    seebeck_eff_mass_from_seebeck_carr,
    seebeck_spb,
)
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import Dos
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.testing import TEST_FILES_DIR

try:
    import fdint
except ImportError:
    fdint = None


TEST_DIR = f"{TEST_FILES_DIR}/electronic_structure/boltztrap"


class TestBoltztrapAnalyzer:
    @classmethod
    def setup_class(cls):
        cls.bz = BoltztrapAnalyzer.from_files(f"{TEST_DIR}/transp/")
        cls.bz_bands = BoltztrapAnalyzer.from_files(f"{TEST_DIR}/bands/")
        cls.bz_up = BoltztrapAnalyzer.from_files(f"{TEST_DIR}/dos_up/", dos_spin=1)
        cls.bz_dw = BoltztrapAnalyzer.from_files(f"{TEST_DIR}/dos_dw/", dos_spin=-1)
        cls.bz_fermi = BoltztrapAnalyzer.from_files(f"{TEST_DIR}/fermi/")

    def test_properties(self):
        assert self.bz.gap == approx(1.6644932121620404, abs=1e-4)
        array = self.bz._cond[300][102]
        assert array[0][0] / 1e19 == approx(7.5756518, abs=1e-4)
        assert array[0][2] == approx(-11.14679)
        assert array[1][0] == approx(-88.203286)
        assert array[2][2] == approx(1.7133249e19)
        array = self.bz._seebeck[300][22]
        assert array[0][1] == approx(6.4546074e-22)
        assert array[1][1] == approx(-0.00032073711)
        assert array[1][2] == approx(-2.9868424e-24)
        assert array[2][2] == approx(-0.0003126543)
        array = self.bz._kappa[500][300]
        assert array[0][1] == approx(0.00014524309)
        assert array[1][1] == approx(328834400000000.0)
        assert array[1][2] == approx(3.7758069e-05)
        assert array[2][2] == approx(193943750000000.0)
        assert self.bz._hall[400][800][1][0][0] == approx(9.5623749e-28)
        assert self.bz._hall[400][68][1][2][2] == approx(3.2149951e-26)
        assert self.bz.doping["p"][3] == approx(1e18)
        assert self.bz.mu_doping["p"][300][2] == approx(0.1553770018406)
        assert self.bz.mu_doping["n"][300][-1] == approx(1.6486017632924719, abs=1e-4)
        assert self.bz._cond_doping["n"][800][3][1][1] / 1e16 == approx(1.5564085, abs=1e-4)
        assert self.bz._seebeck_doping["p"][600][2][0][1] / 1e-23 == approx(3.2860613, abs=1e-4)
        assert self.bz._carrier_conc[500][67] == approx(38.22832002)
        assert self.bz.vol == approx(612.97557323964838, abs=1e-4)
        assert self.bz.intrans["scissor"] == approx(0.0, abs=1e-1)
        assert self.bz._hall_doping["n"][700][-1][2][2][2] == approx(5.0136483e-26)
        assert self.bz.dos.efermi == approx(-0.0300005507057)
        assert self.bz.dos.energies[0] == approx(-2.4497049391830448, abs=1e-4)
        assert self.bz.dos.energies[345] == approx(-0.72708823447130944, abs=1e-4)
        assert self.bz.dos.energies[-1] == approx(3.7569398770153524, abs=1e-4)
        assert self.bz.dos.densities[Spin.up][400] == approx(118.70171)
        assert self.bz.dos.densities[Spin.up][200] == approx(179.58562)
        assert self.bz.dos.densities[Spin.up][300] == approx(289.43945)

        assert self.bz_bands._bz_bands.shape == approx((1316, 20))
        assert self.bz_bands._bz_kpoints.shape == approx((1316, 3))
        assert self.bz_up._dos_partial["0"]["pz"][2562] == approx(0.023862958)
        assert self.bz_dw._dos_partial["1"]["px"][3120] == approx(5.0192891)
        assert self.bz_fermi.fermi_surface_data.shape == approx((121, 121, 65))
        assert self.bz_fermi.fermi_surface_data[21][79][19] == approx(-1.8831911809439161, abs=1e-5)

    @pytest.mark.skipif(not fdint, reason="No FDINT")
    def test_get_seebeck_eff_mass(self):
        ref = [1.956090529381193, 2.0339311618566343, 1.1529383757896965]
        ref2 = [4258.4072823354145, 4597.0351887125289, 4238.1262696392705]
        sbk_mass_tens_mu = self.bz.get_seebeck_eff_mass(output="tensor", doping_levels=False, temp=300)[3]
        sbk_mass_tens_dop = self.bz.get_seebeck_eff_mass(output="tensor", doping_levels=True, temp=300)["n"][2]
        sbk_mass_avg_mu = self.bz.get_seebeck_eff_mass(output="average", doping_levels=False, temp=300)[3]
        sbk_mass_avg_dop = self.bz.get_seebeck_eff_mass(output="average", doping_levels=True, temp=300)["n"][2]

        assert sbk_mass_tens_mu == approx(ref2, abs=1e-1)
        assert sbk_mass_tens_dop == approx(ref, abs=1e-4)

        assert sbk_mass_avg_mu == approx(4361.4744008038842, abs=1e-1)
        assert sbk_mass_avg_dop == approx(1.661553842105382, abs=1e-4)

    @pytest.mark.skipif(not fdint, reason="No FDINT")
    def test_get_complexity_factor(self):
        ref = [2.7658776815227828, 2.9826088215568403, 0.28881335881640308]
        ref2 = [0.0112022048620205, 0.0036001049607186602, 0.0083028947173193028]
        sbk_mass_tens_mu = self.bz.get_complexity_factor(output="tensor", doping_levels=False, temp=300)[3]
        sbk_mass_tens_dop = self.bz.get_complexity_factor(output="tensor", doping_levels=True, temp=300)["n"][2]
        sbk_mass_avg_mu = self.bz.get_complexity_factor(output="average", doping_levels=False, temp=300)[3]
        sbk_mass_avg_dop = self.bz.get_complexity_factor(output="average", doping_levels=True, temp=300)["n"][2]

        assert sbk_mass_tens_mu == approx(ref2, abs=1e-4)
        assert sbk_mass_tens_dop == approx(ref, abs=1e-4)

        assert sbk_mass_avg_mu == approx(0.00628677029221, abs=1e-4)
        assert sbk_mass_avg_dop == approx(1.12322832119, abs=1e-4)

    def test_get_seebeck(self):
        ref = [-768.99078999999995, -724.43919999999991, -686.84682999999973]
        assert self.bz.get_seebeck()["n"][800][3] == approx(ref)
        assert self.bz.get_seebeck(output="average")["p"][800][3] == approx(697.608936667)
        assert self.bz.get_seebeck(output="average", doping_levels=False)[500][520] == approx(1266.7056)
        assert self.bz.get_seebeck(output="average", doping_levels=False)[300][65] == approx(-36.2459389333)

    def test_get_conductivity(self):
        ref = [5.9043185000000022, 17.855599000000002, 26.462935000000002]
        assert self.bz.get_conductivity()["p"][600][2] == approx(ref)
        assert self.bz.get_conductivity(output="average")["n"][700][1] == approx(1.58736609667)
        assert self.bz.get_conductivity(output="average", doping_levels=False)[300][457] == approx(2.87163566667)
        assert self.bz.get_conductivity(output="average", doping_levels=False, relaxation_time=1e-15)[200][
            63
        ] == approx(16573.0536667)

    def test_get_power_factor(self):
        ref = [6.2736602345523362, 17.900184232304138, 26.158282220458144]
        assert self.bz.get_power_factor()["p"][200][2] == approx(ref)
        assert self.bz.get_power_factor(output="average")["n"][600][4] == approx(411.230962976)
        assert self.bz.get_power_factor(output="average", doping_levels=False, relaxation_time=1e-15)[500][
            459
        ] == approx(6.59277148467)
        assert self.bz.get_power_factor(output="average", doping_levels=False)[800][61] == approx(2022.67064134)

    def test_get_thermal_conductivity(self):
        ref = [2.7719565628862623e-05, 0.00010048046886793946, 0.00015874549392499391]
        assert self.bz.get_thermal_conductivity()["p"][300][2] == approx(ref)
        assert self.bz.get_thermal_conductivity(output="average", relaxation_time=1e-15)["n"][500][0] == approx(
            1.74466575612e-07
        )
        assert self.bz.get_thermal_conductivity(output="average", doping_levels=False)[800][874] == approx(
            8.08066254813
        )
        assert self.bz.get_thermal_conductivity(output="average", doping_levels=False)[200][32] == approx(
            0.0738961845832
        )
        assert self.bz.get_thermal_conductivity(k_el=False, output="average", doping_levels=False)[200][32] == approx(
            0.19429052
        )

    def test_get_zt(self):
        ref = [0.097408810215, 0.29335112354, 0.614673998089]
        assert self.bz.get_zt()["n"][800][4] == approx(ref)
        assert self.bz.get_zt(output="average", k_l=0.5)["p"][700][2] == approx(0.0170001879916)
        assert self.bz.get_zt(output="average", doping_levels=False, relaxation_time=1e-15)[300][240] == approx(
            0.0041923533238348342
        )

        eig_vals = self.bz.get_zt(output="eigs", doping_levels=False)[700][65]
        ref_eig_vals = [0.082420053399668847, 0.29408035502671648, 0.40822061215079392]
        assert eig_vals == approx(ref_eig_vals, abs=1e-5)

    def test_get_average_eff_mass(self):
        ref = [0.76045816788363574, 0.96181142990667101, 2.9428428773308628]
        assert self.bz.get_average_eff_mass()["p"][300][2] == approx(ref)
        ref = [1.1295783824744523, 1.3898454041924351, 5.2459984671977935]
        ref2 = [6.6648842712692078, 31.492540105738343, 37.986369302138954]
        assert self.bz.get_average_eff_mass()["n"][600][1] == approx(ref)
        assert self.bz.get_average_eff_mass(doping_levels=False)[300][200] == approx(ref2)
        ref = [
            [9.61811430e-01, -8.25159596e-19, -4.70319444e-19],
            [-8.25159596e-19, 2.94284288e00, 3.00368916e-18],
            [-4.70319444e-19, 3.00368916e-18, 7.60458168e-01],
        ]
        ref2 = [
            [27.97604444269153, -2.39347589e-17, -1.36897140e-17],
            [-2.39347589e-17, 8.55969097e01, 8.74169648e-17],
            [-1.36897140e-17, 8.74169648e-17, 2.21151980e01],
        ]

        assert_allclose(self.bz.get_average_eff_mass(output="tensor")["p"][300][2], ref, atol=1e-4)
        assert_allclose(
            self.bz.get_average_eff_mass(output="tensor", doping_levels=False)[300][500],
            ref2,
            rtol=4,
        )
        assert_allclose(
            self.bz.get_average_eff_mass(output="average")["n"][300][2],
            1.53769093989,
            atol=1e-4,
        )

    def test_get_carrier_concentration(self):
        assert self.bz.get_carrier_concentration()[300][39] / 1e22 == approx(6.4805156617179151, abs=1e-4)
        assert self.bz.get_carrier_concentration()[300][693] / 1e15 == approx(-6.590800965604750, abs=1e-4)

    def test_get_hall_carrier_concentration(self):
        assert self.bz.get_hall_carrier_concentration()[600][120] / 1e21 == approx(6.773394626767555, abs=1e-4)
        assert self.bz.get_hall_carrier_concentration()[500][892] / 1e21 == approx(-9.136803845741777, abs=1e-4)

    def test_get_symm_bands(self):
        structure = loadfn(f"{TEST_DIR}/structure_mp-12103.json")
        sbs = loadfn(f"{TEST_DIR}/dft_bs_sym_line.json")
        kpoints = [kp.frac_coords for kp in sbs.kpoints]
        labels_dict = {k: sbs.labels_dict[k].frac_coords for k in sbs.labels_dict}
        for kpt_line, label_dict in zip(
            [None, sbs.kpoints, kpoints],
            [None, sbs.labels_dict, labels_dict],
            strict=True,
        ):
            sbs_bzt = self.bz_bands.get_symm_bands(structure, -5.25204548, kpt_line=kpt_line, labels_dict=label_dict)
            assert len(sbs_bzt.bands[Spin.up]) == approx(20)
            assert len(sbs_bzt.bands[Spin.up][1]) == approx(143)

    def test_check_acc_bzt_bands(self):
        structure = loadfn(f"{TEST_DIR}/structure_mp-12103.json")
        sbs = loadfn(f"{TEST_DIR}/dft_bs_sym_line.json")
        sbs_bzt = self.bz_bands.get_symm_bands(structure, -5.25204548)
        bcheck = BoltztrapAnalyzer.check_acc_bzt_bands(sbs_bzt, sbs)
        assert list(bcheck["nb_list"]) == list(range(6, 14))
        assert bcheck["avg_corr"] == approx(0.00049876, abs=1e-6)
        assert bcheck["avg_distance"] == approx(0.00189936, abs=1e-6)
        assert bcheck["acc_err"] == [False, False]
        assert bcheck["acc_thr"] == (0.03, 0.03)
        assert bcheck[6]["Dist"] == approx(0.00193709, abs=1e-6)
        assert bcheck[6]["Corr"] == approx(0.00050781, abs=1e-6)
        assert bcheck[6]["K-H"] == approx(0.0082036, abs=1e-6)

    def test_get_complete_dos(self):
        structure = loadfn(f"{TEST_DIR}/structure_mp-12103.json")
        cdos = self.bz_up.get_complete_dos(structure, self.bz_dw)
        spins = list(cdos.densities)
        assert Spin.down in spins
        assert Spin.up in spins
        assert cdos.get_spd_dos()[OrbitalType.p].densities[Spin.up][3134] == approx(43.839230100999991)
        assert cdos.get_spd_dos()[OrbitalType.s].densities[Spin.down][716] == approx(6.5383268000000001)

    def test_extreme(self):
        extreme = self.bz.get_extreme("seebeck")
        assert extreme["best"]["carrier_type"] == "n"
        assert extreme["p"]["value"] == approx(1255.365, abs=1e-2)
        assert extreme["n"]["isotropic"] is False
        assert extreme["n"]["temperature"] == 600

        extreme = self.bz.get_extreme("kappa", maximize=False, min_temp=400, min_doping=1e20)
        assert extreme["best"]["value"] == approx(0.105, abs=1e-2)
        assert extreme["n"]["value"] == approx(0.139, abs=1e-2)
        assert extreme["p"]["temperature"] == 400
        assert extreme["n"]["isotropic"] is False

    @pytest.mark.parametrize("target", ["power factor", "conductivity", "zt"])
    def test_extreme_targets(self, target):
        extreme = self.bz.get_extreme(target, max_temp=500, max_doping=1e20)
        for pn in ("p", "n"):
            assert extreme[pn]["temperature"] <= 500
            assert extreme[pn]["doping"] <= 1e20
        best_type = extreme["best"]["carrier_type"]
        assert extreme["best"]["value"] == max(extreme["p"]["value"], extreme["n"]["value"])
        assert extreme["best"]["value"] == extreme[best_type]["value"]

    def test_extreme_invalid_target(self):
        with pytest.raises(ValueError, match="Unrecognized target_prop='magnetism'"):
            self.bz.get_extreme("magnetism")

    def test_format_to_output_invalid(self):
        with pytest.raises(ValueError, match="Unknown output format: foo"):
            self.bz.get_seebeck(output="foo")
        with pytest.raises(ValueError, match="Unknown output format: foo"):
            self.bz.get_seebeck(output="foo", doping_levels=False)

    def test_get_seebeck_tensor_and_eig_alias(self):
        tensor = self.bz.get_seebeck(output="tensor")["n"][800][3]
        assert np.shape(tensor) == (3, 3)
        assert sorted(np.linalg.eigvalsh(tensor)) == approx(self.bz.get_seebeck(output="eig")["n"][800][3])

    def test_get_thermal_conductivity_k0_doping(self):
        kappa_0 = self.bz.get_thermal_conductivity(output="tensor", k_el=False)["p"][300][2]
        assert kappa_0 == approx(np.array(self.bz._kappa_doping["p"][300][2]) * 1e-14)
        # k_el = k_0 - T * sigma S^2 is smaller than k_0
        k_el = self.bz.get_thermal_conductivity(output="average", k_el=True)["p"][300][2]
        assert k_el < np.trace(kappa_0) / 3

    def test_get_mu_bounds(self):
        mu_min, mu_max = self.bz.get_mu_bounds(temp=300)
        assert mu_min == approx(min(self.bz.mu_doping["p"][300]))
        assert mu_max == approx(1.6486017632924719, abs=1e-4)
        assert mu_min < mu_max

    def test_parse_outputtrans(self):
        run_type, warning, efermi, gap, doping_levels = BoltztrapAnalyzer.parse_outputtrans(f"{TEST_DIR}/transp")
        assert run_type == "BOLTZ"
        assert warning is None
        assert gap == approx(self.bz.gap)
        assert efermi is not None
        assert len(doping_levels) == len(self.bz.doping["p"]) + len(self.bz.doping["n"])

    def test_parse_intrans(self):
        intrans = BoltztrapAnalyzer.parse_intrans(f"{TEST_DIR}/dos_up")
        assert intrans["scissor"] == approx(0.0)
        assert intrans["dos_type"] == "HISTO"

    def test_get_symm_bands_without_bands(self):
        structure = loadfn(f"{TEST_DIR}/structure_mp-12103.json")
        with pytest.raises(BoltztrapError, match="Bands are not in output of BoltzTraP"):
            self.bz.get_symm_bands(structure, 0.0)

    def test_check_acc_bzt_bands_warning(self):
        structure = loadfn(f"{TEST_DIR}/structure_mp-12103.json")
        sbs = loadfn(f"{TEST_DIR}/dft_bs_sym_line.json")
        sbs_bzt = self.bz_bands.get_symm_bands(structure, -5.25204548)
        with pytest.warns(UserWarning, match="some bands around gap are not accurate"):
            bcheck = BoltztrapAnalyzer.check_acc_bzt_bands(sbs_bzt, sbs, warn_thr=(0.0, 1.0))
        assert bcheck["acc_err"][0] is True

    def test_check_acc_bzt_bands_distance_threshold(self):
        structure = loadfn(f"{TEST_DIR}/structure_mp-12103.json")
        sbs = loadfn(f"{TEST_DIR}/dft_bs_sym_line.json")
        sbs_bzt = self.bz_bands.get_symm_bands(structure, -5.25204548)
        with pytest.warns(UserWarning, match="some bands around gap are not accurate"):
            bcheck = BoltztrapAnalyzer.check_acc_bzt_bands(sbs_bzt, sbs, warn_thr=(1.0, 0.0))
        assert bcheck["acc_err"] == [False, True]

    def test_compare_sym_bands(self):
        structure = loadfn(f"{TEST_DIR}/structure_mp-12103.json")
        sbs = loadfn(f"{TEST_DIR}/dft_bs_sym_line.json")
        sbs_bzt = self.bz_bands.get_symm_bands(structure, -5.25204548)
        bcheck = compare_sym_bands(sbs_bzt, sbs, 6)
        assert list(bcheck) == [6]
        assert bcheck[6]["Dist"] == approx(0.00193709, abs=1e-6)
        assert {br["name"] for br in sbs.branches} <= set(bcheck[6])
        # band index beyond the number of compared bands
        assert compare_sym_bands(sbs_bzt, sbs, [1000]) == "No nb given"

    def test_get_complete_dos_errors(self):
        structure = loadfn(f"{TEST_DIR}/structure_mp-12103.json")
        with pytest.raises(BoltztrapError, match="spin component are the same"):
            self.bz_up.get_complete_dos(structure, self.bz_up)
        shifted_dw = copy.copy(self.bz_dw)
        dos_dw = self.bz_dw.dos
        shifted_dw.dos = Dos(dos_dw.efermi, dos_dw.energies + 0.01, dos_dw.densities)
        with pytest.raises(BoltztrapError, match="energies of the two DOS are different"):
            self.bz_up.get_complete_dos(structure, shifted_dw)

    def test_get_complete_dos_single_spin(self):
        structure = loadfn(f"{TEST_DIR}/structure_mp-12103.json")
        cdos = self.bz_up.get_complete_dos(structure)
        assert list(cdos.densities) == [Spin.up]

    def test_as_from_dict(self):
        dct = self.bz.as_dict()
        bz = BoltztrapAnalyzer.from_dict(json.loads(json.dumps(dct)))
        assert bz.gap == approx(self.bz.gap)
        assert bz.vol == approx(self.bz.vol)
        assert bz.mu_steps == approx(self.bz.mu_steps)
        assert bz.doping == self.bz.doping
        assert bz.mu_doping["n"][300] == approx(self.bz.mu_doping["n"][300])
        assert bz.get_seebeck()["n"][800][3] == approx(self.bz.get_seebeck()["n"][800][3])
        assert bz.get_conductivity()["p"][600][2] == approx(self.bz.get_conductivity()["p"][600][2])
        assert bz.get_zt()["n"][800][4] == approx(self.bz.get_zt()["n"][800][4])
        assert bz.get_hall_carrier_concentration()[600][120] == approx(
            self.bz.get_hall_carrier_concentration()[600][120]
        )
        assert bz.dos.energies == approx(self.bz.dos.energies)

    def test_as_from_dict_carrier_conc(self):
        bz = BoltztrapAnalyzer.from_dict(json.loads(json.dumps(self.bz.as_dict())))
        assert bz.get_carrier_concentration()[300][39] == approx(self.bz.get_carrier_concentration()[300][39])

    def test_as_from_dict_intrans(self):
        bz = BoltztrapAnalyzer.from_dict(json.loads(json.dumps(self.bz.as_dict())))
        assert bz.intrans["scissor"] == approx(self.bz.intrans["scissor"])

    def test_as_from_dict_dos_run(self):
        bz_up = BoltztrapAnalyzer.from_dict(self.bz_up.as_dict())
        assert bz_up.dos.energies == approx(self.bz_up.dos.energies)
        assert bz_up._cond is None

    def test_seebeck_eff_mass_doping_levels(self, fdk_module):
        ref = [1.956090529381193, 2.0339311618566343, 1.1529383757896965]
        sbk_mass_tens = self.bz.get_seebeck_eff_mass(output="tensor", doping_levels=True, temp=300)
        assert sbk_mass_tens["n"][2] == approx(ref, abs=1e-4)
        sbk_mass_avg = self.bz.get_seebeck_eff_mass(output="average", doping_levels=True, temp=300)
        assert set(sbk_mass_avg) == {"n", "p"}
        assert len(sbk_mass_avg["p"]) == len(self.bz.doping["p"])
        assert sbk_mass_avg["n"][2] == approx(1.661553842105382, abs=1e-4)

    def test_complexity_factor_doping_levels(self, fdk_module):
        ref = [2.7658776815227828, 2.9826088215568403, 0.28881335881640308]
        cmplx_tens = self.bz.get_complexity_factor(output="tensor", doping_levels=True, temp=300)
        assert cmplx_tens["n"][2] == approx(ref, abs=1e-4)
        cmplx_avg = self.bz.get_complexity_factor(output="average", doping_levels=True, temp=300)
        assert cmplx_avg["n"][2] == approx(1.12322832119, abs=1e-4)


@pytest.fixture
def fdk_module(monkeypatch):
    """Provide fdint, falling back to a quadrature evaluation of the complete Fermi-Dirac integral."""
    if fdint is not None:
        return fdint

    def _fdk(k, phi):
        return quad(lambda x: x**k * expit(phi - x), 0, max(phi, 0) + 60)[0]

    module = types.SimpleNamespace(fdk=np.vectorize(_fdk, otypes=[float]))
    monkeypatch.setitem(sys.modules, "fdint", module)
    return module


class TestSPBHelpers:
    def test_seebeck_spb_nondegenerate_limit(self, fdk_module):
        # For eta -> -inf, F_k(eta) -> Gamma(k+1) exp(eta), so S = k_B/e * (2 + Lambda - eta).
        eta, lam = -15.0, 0.5
        expected = constants.k / constants.e * (2.0 + lam - eta) * 1e6
        assert seebeck_spb(eta, lam) == approx(expected, rel=1e-5)

    def test_eta_from_seebeck_roundtrip(self, fdk_module):
        seebeck = seebeck_spb(1.5, 0.5)
        assert eta_from_seebeck(seebeck, 0.5) == approx(1.5, abs=1e-6)
        # sign of the Seebeck coefficient is ignored
        assert eta_from_seebeck(-seebeck, 0.5) == approx(1.5, abs=1e-6)

    def test_seebeck_eff_mass_from_carr(self, fdk_module):
        # Invert the parabolic-band carrier density n = 2 (m kT / 2 pi hbar^2)^1.5 F_1/2(eta) * 2 / sqrt(pi)
        eta, temp, mass = 0.3, 300, 1.7
        hbar = constants.h / (2 * np.pi)
        n_si = 2 * (mass * constants.m_e * constants.k * temp / (2 * np.pi * hbar**2)) ** 1.5
        n_cm3 = n_si * float(fdk_module.fdk(0.5, eta)) * 2 / np.sqrt(np.pi) / 1e6
        assert seebeck_eff_mass_from_carr(eta, n_cm3, temp, 0.5) == approx(mass, rel=1e-6)
        assert seebeck_eff_mass_from_carr(eta, -n_cm3, temp, 0.5) == approx(mass, rel=1e-6)
        seebeck = seebeck_spb(eta, 0.5)
        assert seebeck_eff_mass_from_seebeck_carr(seebeck, n_cm3, temp, 0.5) == approx(mass, rel=1e-5)

    @pytest.mark.parametrize(
        ("func", "args"), [(seebeck_spb, (0.0, 0.5)), (seebeck_eff_mass_from_carr, (0.0, 1e20, 300, 0.5))]
    )
    def test_missing_fdint(self, monkeypatch, func, args):
        monkeypatch.setitem(sys.modules, "fdint", None)
        with pytest.raises(BoltztrapError, match="fdint module not found"):
            func(*args)


_CUBE_HEADER = """ BoltzTrap CUBE FILE
 OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z
    1   0.000000    0.000000    0.000000
    2    0.5    0.0    0.0
    2    0.0    0.5    0.0
    2    0.0    0.0    0.5
   26    0.000000    0.000000    0.000000    0.000000
"""


def _write_cube(path: Path, values_per_line: int) -> np.ndarray:
    """Write a 2x2x2 cube file with energies in Ry and return the expected energies in eV."""
    values = np.linspace(-0.4, 0.3, 8)
    rows = [values[i : i + values_per_line] for i in range(0, 8, values_per_line)]
    body = "".join(" ".join(f"{v:.6E}" for v in row) + "\n" for row in rows)
    path.write_text(_CUBE_HEADER + body, encoding="utf-8")
    return values.reshape(2, 2, 2) / _to_ry(1)


class TestReadCubeFile:
    def test_boltztrap_bz_cube(self):
        data = read_cube_file(f"{TEST_DIR}/fermi/boltztrap_BZ.cube")
        assert data.shape == (121, 121, 65)
        assert data[21][79][19] == approx(-1.8831911809439161, abs=1e-5)

    def test_boltztrap_bz_cube_synthetic(self, tmp_path):
        expected = _write_cube(tmp_path / "boltztrap_BZ.cube", values_per_line=4)
        assert read_cube_file(str(tmp_path / "boltztrap_BZ.cube")) == approx(expected)

    def test_fort30(self, tmp_path):
        # fort.30 has 3 values per line with a shorter last line
        expected = _write_cube(tmp_path / "fort.30", values_per_line=3)
        assert read_cube_file(str(tmp_path / "fort.30")) == approx(expected)

    def test_invalid_header(self, tmp_path):
        path = tmp_path / "boltztrap_BZ.cube"
        path.write_text("not a cube\n", encoding="utf-8")
        with pytest.raises(ValueError, match="CUBE file format not recognized"):
            read_cube_file(str(path))

    def test_unknown_filename(self, tmp_path):
        _write_cube(tmp_path / "energies.cube", values_per_line=4)
        with pytest.raises(RuntimeError, match="Cannot read cube file"):
            read_cube_file(str(tmp_path / "energies.cube"))


class TestBoltztrapAnalyzerFromFiles:
    @staticmethod
    def _copy_fermi_inputs(dst: Path) -> str:
        for fname in ("boltztrap.outputtrans", "boltztrap.struct", "boltztrap.intrans"):
            shutil.copy(f"{TEST_DIR}/fermi/{fname}", dst / fname)
        return str(dst)

    def test_fermi_fort30(self, tmp_path):
        path_dir = self._copy_fermi_inputs(tmp_path)
        expected = _write_cube(tmp_path / "fort.30", values_per_line=3)
        bz = BoltztrapAnalyzer.from_files(path_dir)
        assert bz.fermi_surface_data == approx(expected)

    def test_fermi_missing_data(self, tmp_path):
        with pytest.raises(BoltztrapError, match="No data file found for fermi surface"):
            BoltztrapAnalyzer.from_files(self._copy_fermi_inputs(tmp_path))

    def test_unknown_run_type(self, tmp_path):
        path_dir = self._copy_fermi_inputs(tmp_path)
        outputtrans = tmp_path / "boltztrap.outputtrans"
        outputtrans.write_text(
            outputtrans.read_text(encoding="utf-8").replace("Calc type: FERMI", "Calc type: FOO"), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="run_type='FOO' not recognized"):
            BoltztrapAnalyzer.from_files(path_dir)


@pytest.fixture(scope="module")
def cu2o_bs():
    with open(f"{TEST_FILES_DIR}/electronic_structure/bandstructure/Cu2O_361_bandstructure.json", "rb") as file:
        return BandStructure.from_dict(orjson.loads(file.read()))


@pytest.fixture(scope="module")
def spin_bs(cu2o_bs):
    """Spin-polarized variant of the Cu2O band structure (spin down shifted up by 0.1 eV)."""
    bands = np.array(cu2o_bs.bands[Spin.up])
    return BandStructure(
        [kpt.frac_coords for kpt in cu2o_bs.kpoints],
        {Spin.up: bands, Spin.down: bands + 0.1},
        cu2o_bs.lattice_rec,
        cu2o_bs.efermi,
        structure=cu2o_bs.structure,
    )


@pytest.fixture
def make_runner(monkeypatch):
    """BoltztrapRunner.__init__ is gated on the x_trans binary, but input writing is pure Python."""
    init = BoltztrapRunner.__init__
    monkeypatch.setattr(BoltztrapRunner, "__init__", getattr(init, "__wrapped__", init))
    return BoltztrapRunner


def _to_ry(energy_ev: float) -> float:
    return float(Energy(energy_ev, "eV").to("Ry"))


class TestBoltztrapRunner:
    def test_init_defaults(self, make_runner, cu2o_bs):
        runner = make_runner(cu2o_bs, 10)
        assert runner.bs is cu2o_bs
        assert runner.nelec == 10
        assert len(runner.doping) == 25
        assert runner.doping[:4] == approx([1e16, 2.5e16, 5e16, 7.5e16])
        assert runner.doping[-1] == approx(1e22)
        assert runner.cb_cut == approx(0.1)
        assert runner.energy_span_around_fermi == approx(1.5)

        runner = make_runner(cu2o_bs, 10, doping=[1e19, 1e20], cb_cut=20)
        assert runner.doping == [1e19, 1e20]
        assert runner.cb_cut == approx(0.2)

    @pytest.mark.parametrize(("bs_name", "run_type"), [("cu2o_bs", "BANDS"), ("spin_bs", "DOS")])
    def test_auto_set_energy_range(self, make_runner, request, bs_name, run_type):
        bs = request.getfixturevalue(bs_name)
        runner = make_runner(bs, 10, run_type=run_type, spin=1)
        const = _to_ry(2)
        assert float(runner._ll) < -const
        assert float(runner._hl) > const
        expected_span = max(abs(float(runner._ll)), abs(float(runner._hl))) / _to_ry(1) * 1.01
        assert float(runner.energy_span_around_fermi) == approx(expected_span)

    @pytest.mark.parametrize("cb_cut", [10, 0])
    def test_write_energy(self, make_runner, cu2o_bs, tmp_path, cb_cut):
        runner = make_runner(cu2o_bs, 10, cb_cut=cb_cut)
        runner.write_energy(tmp_path / "boltztrap.energy")
        lines = (tmp_path / "boltztrap.energy").read_text(encoding="utf-8").splitlines()
        n_bands = math.floor(cu2o_bs.nb_bands * (1 - cb_cut / 100))
        n_kpts = len(cu2o_bs.kpoints)
        assert lines[0] == "test"
        assert int(lines[1]) == n_kpts
        assert len(lines) == 2 + n_kpts * (n_bands + 1)
        kpt_tokens = lines[2].split()
        assert [float(t) for t in kpt_tokens[:3]] == approx(cu2o_bs.kpoints[0].frac_coords, abs=1e-8)
        assert int(kpt_tokens[3]) == n_bands
        energies = [float(line) for line in lines[3 : 3 + n_bands]]
        assert energies == sorted(energies)
        expected = sorted(_to_ry(cu2o_bs.bands[Spin.up][j][0] - cu2o_bs.efermi) for j in range(n_bands))
        assert energies == approx(expected, abs=1e-7)

    @pytest.mark.parametrize(("cond_band", "sign"), [(False, 1), (True, -1)])
    def test_write_energy_fermi(self, make_runner, cu2o_bs, tmp_path, cond_band, sign):
        runner = make_runner(cu2o_bs, 10, run_type="FERMI", band_nb=20, spin=1, cond_band=cond_band)
        runner.write_energy(tmp_path / "boltztrap.energyso")
        lines = (tmp_path / "boltztrap.energyso").read_text(encoding="utf-8").splitlines()
        assert int(lines[1]) == len(cu2o_bs.kpoints)
        assert len(lines) == 2 + 2 * len(cu2o_bs.kpoints)
        for kpt_idx in (0, 5):
            value = float(lines[3 + 2 * kpt_idx])
            expected = sign * _to_ry(cu2o_bs.bands[Spin.up][20][kpt_idx] - cu2o_bs.efermi)
            assert value == approx(expected, abs=1e-7)

    def test_write_energy_fermi_kpoint_line(self, make_runner, cu2o_bs, tmp_path):
        runner = make_runner(cu2o_bs, 10, run_type="FERMI", band_nb=20, spin=1)
        runner.write_energy(tmp_path / "boltztrap.energyso")
        kpt_line = (tmp_path / "boltztrap.energyso").read_text(encoding="utf-8").splitlines()[4]
        tokens = kpt_line.split()
        assert len(tokens) == 4
        assert [float(t) for t in tokens[:3]] == approx(cu2o_bs.kpoints[1].frac_coords, abs=1e-8)
        assert tokens[3] == "1"

    def test_write_energy_dos_spin_polarized(self, make_runner, spin_bs, tmp_path):
        runner = make_runner(spin_bs, 10, run_type="DOS", spin=-1)
        runner.write_energy(tmp_path / "boltztrap.energyso")
        lines = (tmp_path / "boltztrap.energyso").read_text(encoding="utf-8").splitlines()
        n_eigs = math.floor(spin_bs.nb_bands * 0.9) + 2  # padded with the low/high energy limits
        assert int(lines[2].split()[3]) == n_eigs
        energies = [float(line) for line in lines[3 : 3 + n_eigs]]
        assert energies[0] == approx(float(runner._ll), abs=1e-7)
        assert energies[-1] == approx(float(runner._hl), abs=1e-7)
        assert energies[1] == approx(_to_ry(spin_bs.bands[Spin.down][0][0] - spin_bs.efermi), abs=1e-7)

    @pytest.mark.parametrize("symprec", [1e-3, None])
    def test_write_struct(self, make_runner, cu2o_bs, tmp_path, symprec):
        runner = make_runner(cu2o_bs, 10, symprec=symprec)
        runner.write_struct(tmp_path / "boltztrap.struct")
        lines = (tmp_path / "boltztrap.struct").read_text(encoding="utf-8").splitlines()
        n_ops = int(lines[4])
        assert len(lines) == 5 + 3 * n_ops
        if symprec is None:
            assert lines[0] == f"{cu2o_bs.structure.formula} symmetries disabled"
            assert n_ops == 1
            assert [[float(t) for t in line.split()] for line in lines[5:8]] == np.eye(3).tolist()
        else:
            assert lines[0] == f"{cu2o_bs.structure.formula} Pn-3m"
            assert n_ops == 48
        # lattice vectors are written in bohr; parse_struct converts back to Angstrom
        assert BoltztrapAnalyzer.parse_struct(str(tmp_path)) == approx(cu2o_bs.structure.volume, rel=1e-5)

    @pytest.mark.parametrize(
        ("bs_name", "soc", "energy_file"),
        [
            ("cu2o_bs", False, "boltztrap.energy"),
            ("cu2o_bs", True, "boltztrap.energyso"),
            ("spin_bs", False, "boltztrap.energyso"),
        ],
    )
    def test_write_def(self, make_runner, request, tmp_path, bs_name, soc, energy_file):
        runner = make_runner(request.getfixturevalue(bs_name), 10, soc=soc)
        runner.write_def(tmp_path / "BoltzTraP.def")
        lines = (tmp_path / "BoltzTraP.def").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 12
        assert lines[3].startswith(f"10,'{energy_file}',")
        assert lines[0].startswith("5, 'boltztrap.intrans'")

    @pytest.mark.parametrize(("scissor", "set_gap"), [(0.5, "1"), (0.0, "0")])
    def test_write_intrans_boltz(self, make_runner, cu2o_bs, tmp_path, scissor, set_gap):
        doping = [1e18, 1e19]
        runner = make_runner(cu2o_bs, 10, scissor=scissor, dos_type="TETRA", doping=doping, tmax=800, tgrid=100)
        runner.write_intrans(tmp_path / "boltztrap.intrans")
        lines = (tmp_path / "boltztrap.intrans").read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("GENE")
        assert lines[1].split()[2] == set_gap
        assert lines[5].split()[0] == "BOLTZ"
        assert lines[7].split()[:2] == ["800", "100"]
        assert lines[9] == "TETRA"
        assert int(lines[11]) == 2 * len(doping)
        assert [float(line) for line in lines[12:]] == approx([1e18, 1e19, -1e18, -1e19])

        intrans = BoltztrapAnalyzer.parse_intrans(str(tmp_path))
        assert intrans["scissor"] == approx(scissor)
        assert intrans["dos_type"] == "TETRA"

    def test_write_intrans_boltz_nelec(self, make_runner, cu2o_bs, tmp_path):
        runner = make_runner(cu2o_bs, 10)
        runner.write_intrans(tmp_path / "boltztrap.intrans")
        lines = (tmp_path / "boltztrap.intrans").read_text(encoding="utf-8").splitlines()
        assert lines[2].split("#")[0].split()[-1] == "10.0"

    @pytest.mark.parametrize("run_type", ["BOLTZ", "FERMI", "BANDS"])
    def test_write_intrans_plain_numbers(self, make_runner, cu2o_bs, tmp_path, run_type):
        runner = make_runner(cu2o_bs, 10, run_type=run_type, band_nb=6, spin=1, scissor=0.5)
        runner.write_intrans(tmp_path / "boltztrap.intrans")
        lines = (tmp_path / "boltztrap.intrans").read_text(encoding="utf-8").splitlines()
        # Fortran list-directed input: only numbers before the comment
        for line in lines[1:3]:
            assert "Ry" not in line.split("#")[0]
        assert float(lines[2].split()[1]) == approx(_to_ry(0.005))

    def test_write_intrans_fermi(self, make_runner, cu2o_bs, tmp_path):
        runner = make_runner(cu2o_bs, 10, run_type="FERMI", band_nb=6, spin=1)
        runner.write_intrans(tmp_path / "boltztrap.intrans")
        lines = (tmp_path / "boltztrap.intrans").read_text(encoding="utf-8").splitlines()
        assert lines[1].split()[:4] == ["1", "0", "0", "0.0"]
        assert lines[2].split("#")[0].split()[-1] == "10.0"
        assert lines[5].split()[0] == "FERMI"
        assert lines[6].endswith("actual band selected: 7 spin: 1")
        # same content as the intrans from the reference FERMI run
        ref_lines = Path(f"{TEST_DIR}/fermi/boltztrap.intrans").read_text(encoding="utf-8").splitlines()
        assert lines[6] == ref_lines[6]

    def test_write_intrans_bands_default_kpath(self, make_runner, cu2o_bs, tmp_path):
        runner = make_runner(cu2o_bs, 10, run_type="BANDS")
        runner.write_intrans(tmp_path / "boltztrap.intrans")
        lines = (tmp_path / "boltztrap.intrans").read_text(encoding="utf-8").splitlines()
        n_kpts = len(HighSymmKpath(cu2o_bs.structure).get_kpoints(coords_are_cartesian=False)[0])
        assert lines[5].split()[0] == "BANDS"
        assert lines[6] == f"P {n_kpts}"
        assert len(lines) == 7 + n_kpts
        assert len(runner.kpt_line) == n_kpts
        assert [float(t) for t in lines[7].split()] == approx(runner.kpt_line[0])

    def test_write_intrans_bands_kpoints(self, make_runner, cu2o_bs, tmp_path):
        lattice = cu2o_bs.structure.lattice.reciprocal_lattice
        kpt_line = [Kpoint([0, 0, 0], lattice), Kpoint([0.5, 0, 0], lattice), Kpoint([0.5, 0.5, 0], lattice)]
        runner = make_runner(cu2o_bs, 10, run_type="BANDS", kpt_line=kpt_line)
        runner.write_intrans(tmp_path / "boltztrap.intrans")
        lines = (tmp_path / "boltztrap.intrans").read_text(encoding="utf-8").splitlines()
        assert lines[6] == "P 3"
        assert [[float(t) for t in line.split()] for line in lines[7:]] == [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]]
        assert all(isinstance(kpt, np.ndarray) for kpt in runner.kpt_line)

    @pytest.mark.parametrize(
        ("bs_name", "soc", "energy_file"),
        [
            ("cu2o_bs", False, "boltztrap.energy"),
            ("cu2o_bs", True, "boltztrap.energyso"),
            ("spin_bs", False, "boltztrap.energyso"),
        ],
    )
    def test_write_input(self, make_runner, request, tmp_path, bs_name, soc, energy_file):
        runner = make_runner(request.getfixturevalue(bs_name), 10, soc=soc)
        runner.write_input(str(tmp_path))
        expected = {energy_file, "boltztrap.struct", "boltztrap.intrans", "BoltzTraP.def"}
        assert {path.name for path in tmp_path.iterdir()} == expected

    def test_write_input_dos_projections(self, make_runner, cu2o_bs, tmp_path):
        runner = make_runner(cu2o_bs, 10, run_type="DOS", spin=1)
        runner.write_input(str(tmp_path))
        files = {path.name for path in tmp_path.iterdir()}
        assert {"boltztrap.energy", "boltztrap.struct", "boltztrap.intrans", "BoltzTraP.def"} <= files
        proj_files = {name for name in files if name.startswith("boltztrap.proj_")}
        n_orbs = np.shape(cu2o_bs.projections[Spin.up])[2]
        assert len(proj_files) == len(cu2o_bs.structure) * n_orbs
        assert {"boltztrap.proj_0_s", "boltztrap.proj_5_dx2_y2"} <= proj_files
        # DOS run uses the full energy range: span set automatically
        assert runner.energy_span_around_fermi > 1.5
        # write_proj overwrites BoltzTraP.def with one entry per projection file
        def_lines = (tmp_path / "BoltzTraP.def").read_text(encoding="utf-8").splitlines()
        assert len(def_lines) == 12 + len(proj_files)
        assert def_lines[12].startswith("1000,")
        assert def_lines[-1].startswith(f"{1000 + len(proj_files) - 1},")

    def test_write_proj(self, make_runner, cu2o_bs, tmp_path):
        runner = make_runner(cu2o_bs, 10, run_type="DOS", spin=1)
        runner.write_proj(f"{tmp_path}/boltztrap.proj", f"{tmp_path}/BoltzTraP.def")
        n_bands = math.floor(cu2o_bs.nb_bands * 0.9)
        lines = (tmp_path / "boltztrap.proj_1_px").read_text(encoding="utf-8").splitlines()
        assert lines[0] == cu2o_bs.structure.formula
        assert int(lines[1]) == len(cu2o_bs.kpoints)
        assert len(lines) == 2 + len(cu2o_bs.kpoints) * (n_bands + 1)
        assert int(lines[2].split()[3]) == n_bands
        orb_idx = Orbital.px.value
        projs = [float(line) for line in lines[3 : 3 + n_bands]]
        expected = [cu2o_bs.projections[Spin.up][j][0][orb_idx][1] for j in range(n_bands)]
        assert projs == approx(expected, abs=1e-8)

    def test_write_proj_def_format(self, make_runner, cu2o_bs, tmp_path):
        runner = make_runner(cu2o_bs, 10, run_type="DOS", spin=1)
        runner.write_proj(f"{tmp_path}/boltztrap.proj", f"{tmp_path}/BoltzTraP.def")
        content = (tmp_path / "BoltzTraP.def").read_text(encoding="utf-8")
        # format of the reference DOS run in test-files/electronic_structure/boltztrap/dos_up/BoltzTraP.def
        assert "1000,'boltztrap.proj_0_s' 'old', 'formatted',0" in content

    def test_run_invalid_args(self, make_runner, cu2o_bs, spin_bs, tmp_path):
        runner = make_runner(cu2o_bs, 10)
        with pytest.raises(ValueError, match="Convergence mode requires write_input to be true"):
            runner.run(path_dir=str(tmp_path), convergence=True, write_input=False)

        runner = make_runner(spin_bs, 10, run_type="FERMI", band_nb=3)
        with pytest.raises(BoltztrapError, match="Spin parameter must be specified"):
            runner.run(path_dir=str(tmp_path))

    def test_run_writes_input_before_calling_binary(self, make_runner, cu2o_bs, tmp_path, monkeypatch):
        class PopenCalledError(Exception):
            pass

        def fake_popen(*args, **kwargs):
            raise PopenCalledError(args)

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        runner = make_runner(cu2o_bs, 10)
        with pytest.raises(PopenCalledError):
            runner.run(path_dir=str(tmp_path), convergence=False)
        assert (tmp_path / "boltztrap" / "boltztrap.intrans").is_file()

    def test_as_dict(self, make_runner, cu2o_bs):
        runner = make_runner(cu2o_bs, 10, run_type="BOLTZ", scissor=0.2, doping=[1e19])
        dct = runner.as_dict()
        assert orjson.loads(orjson.dumps(dct)) == dct
        assert dct["@class"] == "BoltztrapRunner"
        assert dct["nelec"] == 10
        assert dct["scissor"] == approx(0.2)
        assert dct["doping"] == [1e19]
        assert dct["symprec"] == approx(1e-3)
        assert BandStructure.from_dict(dct["bs"]).nb_bands == cu2o_bs.nb_bands
