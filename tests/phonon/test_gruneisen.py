from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colors
from pytest import approx

from pymatgen.core import Structure
from pymatgen.core import constants as const
from pymatgen.io.phonopy import get_gruneisen_ph_bs_symm_line, get_gruneisenparameter
from pymatgen.phonon.gruneisen import GruneisenParameter
from pymatgen.phonon.plotter import GruneisenPhononBandStructureSymmLine, GruneisenPhononBSPlotter, GruneisenPlotter
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

try:
    import phonopy
    from phonopy.phonon.dos import TotalDos
except ImportError as exc:
    print(exc)
    phonopy = TotalDos = None

TEST_DIR = f"{TEST_FILES_DIR}/phonon/gruneisen"


class TestGruneisenPhononBandStructureSymmLine(MatSciTest):
    def setup_method(self) -> None:
        self.bs_symm_line = get_gruneisen_ph_bs_symm_line(
            gruneisen_path=f"{TEST_DIR}/gruneisen_eq_plus_minus_InP.yaml",
            structure_path=f"{TEST_DIR}/eq/POSCAR_InP",
            fit=True,
        )

    def test_plot(self):
        plotter = GruneisenPhononBSPlotter(bs=self.bs_symm_line)
        ax = plotter.get_plot_gs()
        assert isinstance(ax, plt.Axes)

    def test_ph_plot_w_gruneisen(self):
        plotter = GruneisenPhononBSPlotter(bs=self.bs_symm_line)
        ax = plotter.get_plot_gs(plot_ph_bs_with_gruneisen=True, units="THz", cmap=["red", "royalblue"])
        assert ax.get_ylabel() == "Frequencies (THz)"
        assert ax.get_xlabel() == "$\\mathrm{Wave\\ Vector}$"
        assert ax.get_figure()._localaxes[-1].get_ylabel() == "$\\gamma \\ \\mathrm{(logarithmized)}$"
        assert len(ax._children) == plotter.n_bands + 1  # check for number of bands
        # check for x and y data is really the band-structure data
        for inx, band in enumerate(plotter._bs.bands):
            xy_data = {
                "x": [point[0] for point in ax._children[inx].get_offsets().data],
                "y": [point[1] for point in ax._children[inx].get_offsets().data],
            }
            assert band == approx(xy_data["y"])
            assert plotter._bs.distance == approx(xy_data["x"])

        # check if color bar max value matches maximum gruneisen parameter value
        data = plotter.bs_plot_data()

        # get reference min and max Grüneisen parameter values
        max_gruneisen = np.array(data["gruneisen"]).max()
        min_gruneisen = np.array(data["gruneisen"]).min()

        norm = colors.SymLogNorm(
            vmin=min_gruneisen,
            vmax=max_gruneisen,
            linthresh=1e-2,
            linscale=1,
        )

        assert max(norm.inverse(ax.get_figure()._localaxes[-1].get_yticks())) == approx(max_gruneisen)
        assert isinstance(ax, plt.Axes)

    def test_as_dict_from_dict(self):
        new_dict = self.bs_symm_line.as_dict()
        self.new_bs_symm_line = GruneisenPhononBandStructureSymmLine.from_dict(new_dict)
        plotter = GruneisenPhononBSPlotter(bs=self.new_bs_symm_line)
        ax = plotter.get_plot_gs()
        assert isinstance(ax, plt.Axes)


class TestGruneisenParameterDebyeTempShim:
    def test_debye_temp_phonopy_run_debye_frequency_uses_new_signature(self, monkeypatch: pytest.MonkeyPatch):
        class FakeTotalDos:
            def __init__(self) -> None:
                self.freq_max_fit: float | None = None

            def run_debye_frequency(self, freq_max_fit=None) -> float:
                self.freq_max_fit = freq_max_fit
                return 2.5

        fake_tdos = FakeTotalDos()
        monkeypatch.setattr(GruneisenParameter, "tdos", property(lambda _self: fake_tdos))
        gruneisen = GruneisenParameter(
            qpoints=np.zeros((1, 3)),
            gruneisen=np.zeros((1, 1)),
            frequencies=np.zeros((1, 1)),
            structure=Structure(np.eye(3), ["H"], [[0, 0, 0]]),
        )

        debye_temp = gruneisen.debye_temp_phonopy(freq_max_fit=1.23)

        assert fake_tdos.freq_max_fit == approx(1.23)
        assert debye_temp == approx(
            const.value("Planck constant") * 2.5 * const.tera / const.value("Boltzmann constant")
        )

    def test_debye_temp_phonopy_run_debye_frequency_internal_failure(self, monkeypatch: pytest.MonkeyPatch):
        class FakeTotalDos:
            def run_debye_frequency(self, freq_max_fit=None) -> float:
                raise AttributeError("'TempMesh' object has no attribute 'primitive'")

        monkeypatch.setattr(GruneisenParameter, "tdos", property(lambda _self: FakeTotalDos()))
        gruneisen = GruneisenParameter(
            qpoints=np.zeros((1, 3)),
            gruneisen=np.zeros((1, 1)),
            frequencies=np.zeros((1, 1)),
            structure=Structure(np.eye(3), ["H"], [[0, 0, 0]]),
        )

        with pytest.raises(RuntimeError, match="cannot compute Debye frequency"):
            gruneisen.debye_temp_phonopy()


class TestAverageGruneisen:
    def test_weights_and_acoustic_modes_index_the_right_axis(self):
        # frequencies/gruneisen have shape (n_bands, n_qpoints) and multiplicities one entry
        # per q-point. average_gruneisen used to index the weights by band and take the
        # first three q-points as the "acoustic modes".
        rng = np.random.default_rng(0)
        n_bands, n_qpts = 6, 4
        freqs = np.sort(rng.uniform(0.5, 8, size=(n_bands, n_qpts)), axis=0)
        freqs[0, 0] = -0.1  # an imaginary mode, must be skipped
        gruneisen = rng.uniform(0.5, 2.5, size=(n_bands, n_qpts))
        multiplicities = np.array([1, 6, 12, 8])
        grun = GruneisenParameter(
            qpoints=rng.random((n_qpts, 3)),
            gruneisen=gruneisen,
            frequencies=freqs,
            multiplicities=multiplicities,
        )

        temp = 300
        x = freqs * const.tera / (const.value("Boltzmann constant in Hz/K") * temp)
        cv = np.where(freqs > 0, x**2 * np.exp(x) / np.expm1(x) ** 2, 0)
        weights = np.broadcast_to(multiplicities, freqs.shape)

        def expected(mask, squared):
            gamma = gruneisen**2 if squared else gruneisen
            avg = (weights * cv * gamma)[mask].sum() / (weights * cv)[mask].sum()
            return np.sqrt(avg) if squared else avg

        positive = freqs >= 0
        acoustic = positive & (np.arange(n_bands) < 3)[:, None]
        assert grun.average_gruneisen(t=temp) == approx(expected(positive, squared=True))
        assert grun.average_gruneisen(t=temp, squared=False) == approx(expected(positive, squared=False))
        assert grun.average_gruneisen(t=temp, limit_frequencies="acoustic") == approx(expected(acoustic, squared=True))


@pytest.mark.skipif(TotalDos is None, reason="Phonopy not present")
class TestGruneisenParameter(MatSciTest):
    # Parsing the mesh yaml files is slow, so load once per class. Tests must not mutate these objects.
    @classmethod
    def setup_class(cls) -> None:
        cls.gruneisen_obj = get_gruneisenparameter(
            f"{TEST_DIR}/gruneisen_mesh_InP.yaml",
            structure_path=f"{TEST_DIR}/eq/POSCAR_InP",
        )
        cls.gruneisen_obj_small = get_gruneisenparameter(
            f"{TEST_DIR}/gruneisen_mesh_only_one_q_InP.yaml",
            structure_path=f"{TEST_DIR}/eq/POSCAR_InP",
        )
        cls.gruneisen_obj_Si = get_gruneisenparameter(
            f"{TEST_DIR}/gruneisen_mesh_Si.yaml",
            structure_path=f"{TEST_DIR}/eq/POSCAR_Si",
        )

    def test_plot(self):
        plotter = GruneisenPlotter(self.gruneisen_obj)
        ax = plotter.get_plot(units="mev")
        assert isinstance(ax, plt.Axes)

    def test_as_from_dict(self):
        new_dict = self.gruneisen_obj.as_dict()
        self.gruneisen_obj2 = GruneisenParameter.from_dict(new_dict)

    def test_frequencies(self):
        assert self.gruneisen_obj_small.frequencies == approx(
            [0.12642146, 0.12642146, 0.25272004, 8.85202452, 8.85202452, 9.66016595]
        )

    def test_multi(self):
        assert self.gruneisen_obj_small.multiplicities[0] == 1
        assert self.gruneisen_obj.multiplicities[0] == 2

    def test_gruneisen(self):
        assert self.gruneisen_obj_small.gruneisen[0] == approx(-0.6176464482)
        assert self.gruneisen_obj_small.gruneisen[5] == approx(1.7574050911)

    def test_tdos(self):
        tot_dos = self.gruneisen_obj.tdos
        assert isinstance(tot_dos, phonopy.phonon.dos.TotalDos)

    def test_phdos(self):
        assert self.gruneisen_obj.phdos.cv(298.15) == approx(45.17772584681599)

    def test_average_gruneisen(self):
        assert self.gruneisen_obj.average_gruneisen() == approx(1.1327827283219214)
        assert self.gruneisen_obj.average_gruneisen(squared=False) == approx(0.6547326857785946)
        assert self.gruneisen_obj.average_gruneisen(limit_frequencies="debye") == approx(0.8511832975568538)
        assert self.gruneisen_obj.average_gruneisen(limit_frequencies="acoustic") == approx(0.8511832975568538)
        assert self.gruneisen_obj_Si.average_gruneisen() == approx(1.1184641206671346)

    def test_thermal_conductivity_slack(self):
        assert self.gruneisen_obj.thermal_conductivity_slack() == approx(82.68257209206129)
        assert self.gruneisen_obj.thermal_conductivity_slack(t=300) == approx(94.31452714528649)
        assert self.gruneisen_obj_Si.thermal_conductivity_slack(t=300) == approx(125.4193305607479)

    def test_debye_temp_phonopy(self):
        # This is the correct conversion when starting from THz in the debye_freq
        try:
            debye_temp = self.gruneisen_obj_small.debye_temp_phonopy()
        except RuntimeError as exc:
            pytest.skip(str(exc))

        assert debye_temp == approx(473.31932718764284)

    def test_acoustic_debye_temp(self):
        assert self.gruneisen_obj_small.acoustic_debye_temp == approx(317.54811309631845)
        assert self.gruneisen_obj.acoustic_debye_temp == approx(342.2046198151735)
        assert self.gruneisen_obj_Si.acoustic_debye_temp == approx(526.0725636300882)
