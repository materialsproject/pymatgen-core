from __future__ import annotations

import copy
import gzip
import itertools
import math
import os
from importlib.util import find_spec
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import orjson
import pytest
from matplotlib import rc
from matplotlib.backend_bases import KeyEvent, ResizeEvent
from numpy.testing import assert_allclose
from pytest import approx

from pymatgen.core.structure import Structure
from pymatgen.electronic_structure import plotter as plotter_module
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapAnalyzer, BoltztrapError
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.electronic_structure.plotter import (
    BoltztrapPlotter,
    BSDOSPlotter,
    BSPlotter,
    BSPlotterProjected,
    CohpPlotter,
    DosPlotter,
    fold_point,
    plot_brillouin_zone,
    plot_ellipsoid,
)
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.testing import TEST_FILES_DIR, VASP_IN_DIR, VASP_OUT_DIR, MatSciTest

BAND_TEST_DIR = f"{TEST_FILES_DIR}/electronic_structure/bandstructure"

rc("text", usetex=False)  # Disabling latex is needed for this test to work.


class TestDosPlotter(MatSciTest):
    def setup_method(self):
        with open(f"{BAND_TEST_DIR}/../dos/complete_dos.json", "rb") as file:
            self.dos = CompleteDos.from_dict(orjson.loads(file.read()))
            self.plotter = DosPlotter(sigma=0.2, stack=True)

    def test_add_dos_dict(self):
        dct = self.plotter.get_dos_dict()
        assert len(dct) == 0
        self.plotter.add_dos_dict(self.dos.get_element_dos(), key_sort_func=lambda x: x.X)
        dct = self.plotter.get_dos_dict()
        assert len(dct) == 4

    def test_get_dos_dict(self):
        self.plotter.add_dos_dict(self.dos.get_element_dos(), key_sort_func=lambda x: x.X)
        dct = self.plotter.get_dos_dict()
        assert list(dct) == ["Li", "Fe", "P", "O"]

    # Minimal baseline testing for get_plot. not a true test. Just checks that
    # it can actually execute.
    def test_get_plot(self):
        self.plotter.add_dos_dict(self.dos.get_element_dos(), key_sort_func=lambda x: x.X)
        ax = self.plotter.get_plot()
        assert len(ax.lines) == 2
        out_path = f"{self.tmp_path}/dosplot.png"
        self.plotter.save_plot(out_path)
        assert os.path.isfile(out_path)

    def test_get_plot_limits(self):
        # Tests limit determination and if inverted_axes case
        # reproduces the same energy and DOS axis limits
        self.plotter.add_dos_dict(self.dos.get_element_dos(), key_sort_func=lambda x: x.X)
        # Contains energy and DOS limits and expected results
        with open(f"{BAND_TEST_DIR}/../plotter/complete_dos_limits.json", "rb") as file:
            limits_results = orjson.loads(file.read())

        for item in limits_results:
            ax = self.plotter.get_plot(xlim=item["energy_limit"], ylim=item["DOS_limit"])
            param_dict = self.get_plot_attributes(ax)
            ax_invert = self.plotter.get_plot(invert_axes=True, xlim=item["DOS_limit"], ylim=item["energy_limit"])
            param_dict_invert = self.get_plot_attributes(ax_invert)
            assert item["energy_result"] == approx(param_dict["xaxis_limits"])
            assert item["energy_result"] == approx(param_dict_invert["yaxis_limits"])
            assert item["DOS_result"] == approx(param_dict["yaxis_limits"])
            assert item["DOS_result"] == approx(param_dict_invert["xaxis_limits"])

    @staticmethod
    def get_plot_attributes(ax: plt.Axes):
        return {
            "xaxis_limits": list(ax.get_xlim()),
            "yaxis_limits": list(ax.get_ylim()),
        }


class TestBSPlotter(MatSciTest):
    def setup_method(self):
        with open(f"{BAND_TEST_DIR}/CaO_2605_bandstructure.json", encoding="utf-8") as file:
            dct = orjson.loads(file.read())
            self.bs = BandStructureSymmLine.from_dict(dct)
            self.plotter = BSPlotter(self.bs)

        assert len(self.plotter._bs) == 1, "wrong number of band objects"

        with open(f"{BAND_TEST_DIR}/N2_12103_bandstructure.json", encoding="utf-8") as file:
            dct = orjson.loads(file.read())
            self.sbs_sc = BandStructureSymmLine.from_dict(dct)

        with open(f"{BAND_TEST_DIR}/C_48_bandstructure.json", encoding="utf-8") as file:
            dct = orjson.loads(file.read())
            self.sbs_met = BandStructureSymmLine.from_dict(dct)

        self.plotter_multi = BSPlotter([self.sbs_sc, self.sbs_met])
        assert len(self.plotter_multi._bs) == 2, "wrong number of band objects"
        assert self.plotter_multi._nb_bands == [96, 96], "wrong number of bands"

    def test_add_bs(self):
        self.plotter_multi.add_bs(self.sbs_sc)
        assert len(self.plotter_multi._bs) == 3, "wrong number of band objects"
        assert self.plotter_multi._nb_bands == [96, 96, 96], "wrong number of bands"

    def test_get_branch_steps(self):
        steps_idx = BSPlotter._get_branch_steps(self.sbs_sc.branches)
        assert steps_idx == [0, 121, 132, 143], "wrong list of steps idx"

    def test_rescale_distances(self):
        rescaled_distances = self.plotter_multi._rescale_distances(self.sbs_sc, self.sbs_met)
        assert len(rescaled_distances) == len(self.sbs_met.distance), "wrong length of distances list"
        assert rescaled_distances[-1] == approx(6.5191398067252875), "wrong last distance value"
        assert rescaled_distances[148] == self.sbs_sc.distance[19], "wrong distance at high symm k-point"

    def test_interpolate_bands(self):
        data = self.plotter.bs_plot_data()
        dct = data["distances"]
        en = data["energy"]["1"]
        int_distances, int_energies = self.plotter._interpolate_bands(dct, en)

        assert len(int_distances) == 10, "wrong length of distances list"
        assert len(int_distances[0]) == 100, "wrong length of distances in a branch"
        assert len(int_energies) == 10, "wrong length of distances list"
        assert int_energies[0].shape == (16, 100), "wrong length of distances list"

    def test_bs_plot_data(self):
        assert len(self.plotter.bs_plot_data()["distances"]) == 10, "wrong number of sequences of branches"
        assert len(self.plotter.bs_plot_data()["distances"][0]) == 16, (
            "wrong number of distances in the first sequence of branches"
        )
        assert sum(len(dist) for dist in self.plotter.bs_plot_data()["distances"]) == 160, "wrong number of distances"

        length = len(self.plotter.bs_plot_data(split_branches=False)["distances"][0])
        assert length == 144, "wrong number of distances in the first sequence of branches"

        length = len(self.plotter.bs_plot_data(split_branches=False)["distances"])
        assert length == 2, "wrong number of distances in the first sequence of branches"

        assert self.plotter.bs_plot_data()["ticks"]["label"][5] == "K", "wrong tick label"
        assert len(self.plotter.bs_plot_data()["ticks"]["label"]) == 19, "wrong number of tick labels"

    def test_get_ticks(self):
        assert self.plotter.get_ticks()["label"][5] == "K", "wrong tick label"
        assert self.plotter.get_ticks()["distance"][5] == approx(2.406607625322699), "wrong tick distance"

    # Minimal baseline testing for get_plot. not a true test. Just checks that
    # it can actually execute.
    def test_get_plot(self):
        # zero_to_efermi = True, ylim = None, smooth = False,
        # vbm_cbm_marker = False, smooth_tol = None

        ax = self.plotter.get_plot()
        assert_allclose(ax.get_ylim(), (-4.0, 7.6348)), "wrong ylim"
        ax = self.plotter.get_plot(smooth=True)
        ax = self.plotter.get_plot(vbm_cbm_marker=True)
        self.plotter.save_plot(f"{self.tmp_path}/bsplot.png")
        assert os.path.isfile(f"{self.tmp_path}/bsplot.png")
        plt.close("all")

        # test plotter with 2 bandstructures
        ax = self.plotter_multi.get_plot()
        assert len(ax.get_lines()) == 874, "wrong number of lines"
        assert_allclose(ax.get_ylim(), (-10.0, 10.0)), "wrong ylim"
        ax = self.plotter_multi.get_plot(zero_to_efermi=False)
        assert_allclose(ax.get_ylim(), (-15.2379, 12.67141266)), "wrong ylim"
        ax = self.plotter_multi.get_plot(smooth=True)
        self.plotter_multi.save_plot(f"{self.tmp_path}/bsplot.png")
        assert os.path.isfile(f"{self.tmp_path}/bsplot.png")
        plt.close("all")


class TestBSPlotterProjected:
    def setup_method(self):
        with open(f"{BAND_TEST_DIR}/Cu2O_361_bandstructure.json", "rb") as file:
            self.bs_Cu2O = BandStructureSymmLine.from_dict(orjson.loads(file.read()))
        self.plotter_Cu2O = BSPlotterProjected(self.bs_Cu2O)

        with open(f"{TEST_FILES_DIR}/electronic_structure/boltztrap2/PbTe_bandstructure.json", "rb") as file:
            self.bs_PbTe = BandStructureSymmLine.from_dict(orjson.loads(file.read()))

    def test_orbitals_sum_orbitals_without_sum_morbs(self):
        assert self.plotter_Cu2O._Orbitals_SumOrbitals({"Cu": ["dxy"]}, None) == ({"Cu": ["dxy"]}, None)

    def test_methods(self):
        # Minimal baseline testing for get_plot. not a true test. Just checks that
        # it can actually execute.
        self.plotter_Cu2O.get_elt_projected_plots()
        self.plotter_Cu2O.get_elt_projected_plots_color()
        axes = self.plotter_Cu2O.get_projected_plots_dots({"Cu": ["d", "s"], "O": ["p"]})
        assert len(axes) == 4, f"{len(axes)=}"
        assert len(axes[0].get_lines()) == 4903, f"{len(axes[0].get_lines())=}"
        assert len(axes[-1].get_lines()) == 0, f"{len(axes[-1].get_lines())=}"

        # Test if a plot with a single column works
        axes = self.plotter_Cu2O.get_projected_plots_dots({"Cu": ["d", "s"]})
        assert len(axes) == 2, f"{len(axes)=}"
        assert len(axes[0].get_lines()) == 4903, f"{len(axes[0].get_lines())=}"
        assert len(axes[-1].get_lines()) == 4903, f"{len(axes[-1].get_lines())=}"

        # Test if a plot with a single row works
        axes = self.plotter_Cu2O.get_projected_plots_dots({"Cu": ["d"], "O": ["p"]})
        assert len(axes) == 2, f"{len(axes)=}"
        assert len(axes[0].get_lines()) == 4903, f"{len(axes[0].get_lines())=}"
        assert len(axes[-1].get_lines()) == 4903, f"{len(axes[-1].get_lines())=}"

        axs = self.plotter_Cu2O.get_projected_plots_dots_patom_pmorb(
            {"Cu": ["dxy", "s", "px"], "O": ["px", "py", "pz"]},
            {"Cu": [3, 5], "O": [1]},
        )
        assert isinstance(axs, list)
        assert isinstance(axs[0], plt.Axes)
        assert len(axs) == 9
        assert len(axs[0].get_lines()) == 4903
        assert axs[0].get_ylim() == approx((-4.0, 4.5047))

        # Test if num_column other than 2 works correctly
        axs_col = self.plotter_Cu2O.get_projected_plots_dots_patom_pmorb(
            {"Cu": ["dxy", "s", "px"], "O": ["px", "py", "pz"]}, {"Cu": [3, 5], "O": [1]}, num_column=3
        )
        axs_spec = axs_col[2].get_subplotspec()
        assert axs_spec.rowspan.start == 0
        assert axs_spec.colspan.start == 2

        with pytest.raises(
            ValueError,
            match="Can't plot projections on a band structure without projections data",
        ):
            BSPlotterProjected(self.bs_PbTe)


class TestBSDOSPlotter:
    # Minimal baseline testing for get_plot. not a true test. Just checks that
    # it can actually execute.
    def test_methods(self):
        vasp_run = Vasprun(f"{VASP_OUT_DIR}/vasprun_Si_bands.xml.gz")
        plotter = BSDOSPlotter()
        band_struct = vasp_run.get_band_structure(kpoints_filename=f"{VASP_IN_DIR}/KPOINTS_Si_bands")
        ax = plotter.get_plot(band_struct)
        assert isinstance(ax, plt.Axes)
        plt.close()
        bs_ax, dos_ax = plotter.get_plot(band_struct, vasp_run.complete_dos)
        assert isinstance(bs_ax, plt.Axes)
        assert isinstance(dos_ax, plt.Axes)
        plt.close("all")

        with open(f"{TEST_FILES_DIR}/electronic_structure/plotter/SrBa2Sn2O7.json", "rb") as file:
            band_struct_dict = orjson.loads(file.read())
        # Generate random sparse projections: for each (i, j, k) cell, populate two
        # randomly-chosen orbital slots out of the first 7. Vectorised equivalent of
        # the original 4-deep Python loop.
        rng = np.random.default_rng()
        n_i, n_j, n_k, n_orb = 90, 70, 9, 12
        projections = np.zeros((n_i, n_j, n_k, n_orb))
        flat = projections.reshape(-1, n_orb)
        n_cells = flat.shape[0]
        cell_idx = np.arange(n_cells)
        flat[cell_idx, rng.integers(0, 7, size=n_cells)] = rng.random(n_cells)
        flat[cell_idx, rng.integers(0, 7, size=n_cells)] = rng.random(n_cells)
        band_struct_dict["projections"]["1"] = projections.tolist()
        band_struct = BandStructureSymmLine.from_dict(band_struct_dict)
        ax = plotter.get_plot(band_struct)
        assert isinstance(ax, plt.Axes)


class TestPlotBZ:
    def setup_method(self):
        self.rec_latt = Structure.from_file(f"{TEST_FILES_DIR}/io/cssr/Si.cssr").lattice.reciprocal_lattice
        self.kpath = [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.25, 0.75], [0.375, 0.375, 0.75]]]
        self.labels = {
            "\\Gamma": [0.0, 0.0, 0.0],
            "K": [0.375, 0.375, 0.75],
            "L": [0.5, 0.5, 0.5],
            "U": [0.625, 0.25, 0.625],
            "W": [0.5, 0.25, 0.75],
            "X": [0.5, 0.0, 0.5],
        }
        self.hessian = [
            [17.64757034, 3.90159625, -4.77845607],
            [3.90159625, 14.88874142, 6.75776076],
            [-4.77845607, 6.75776076, 12.12987493],
        ]
        self.center = [0.41, 0.0, 0.41]
        self.points = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]

    def test_bz_plot(self):
        _, ax = plot_ellipsoid(self.hessian, self.center, lattice=self.rec_latt)
        plot_brillouin_zone(
            self.rec_latt,
            lines=self.kpath,
            labels=self.labels,
            kpoints=self.points,
            ax=ax,
            show=False,
        )

    def test_fold_point(self):
        assert_allclose(
            fold_point([0.0, -0.5, 0.5], lattice=self.rec_latt),
            self.rec_latt.get_cartesian_coords([0.0, 0.5, 0.5]),
            atol=1e-8,
        )
        assert_allclose(
            fold_point([0.1, -0.6, 0.2], lattice=self.rec_latt),
            self.rec_latt.get_cartesian_coords([0.1, 0.4, 0.2]),
            atol=1e-8,
        )


BOLTZTRAP_DIR = f"{TEST_FILES_DIR}/electronic_structure/boltztrap"

# Every BoltztrapPlotter method except plot_dos and plot_eff_mass_temp calls
# ax.set_[xy]ticks(ax.get_[xy]ticks(), fontsize=25), which matplotlib >= 3.5 rejects.
BZ_METHODS_WITH_TICK_FONTSIZE = (
    "plot_carriers",
    "plot_hall_carriers",
    "plot_seebeck_mu",
    "plot_conductivity_mu",
    "plot_power_factor_mu",
    "plot_zt_mu",
    "plot_seebeck_temp",
    "plot_conductivity_temp",
    "plot_power_factor_temp",
    "plot_zt_temp",
    "plot_seebeck_dop",
    "plot_conductivity_dop",
    "plot_power_factor_dop",
    "plot_zt_dop",
    "plot_eff_mass_dop",
)


@pytest.fixture(scope="module")
def bz_analyzer():
    # BoltztrapAnalyzer.from_files only parses existing output files; the x_trans binary is not needed.
    return BoltztrapAnalyzer.from_files(f"{BOLTZTRAP_DIR}/transp/")


def _all_lines(ax: plt.Axes) -> list:
    return [line for axis in ax.figure.axes for line in axis.get_lines()]


class TestBoltztrapPlotter:
    def setup_method(self):
        plt.close("all")

    def teardown_method(self):
        plt.close("all")

    @pytest.fixture(autouse=True)
    def _plotter(self, bz_analyzer):
        self.bz = bz_analyzer
        self.plotter = BoltztrapPlotter(bz_analyzer)

    @pytest.mark.parametrize("method", BZ_METHODS_WITH_TICK_FONTSIZE)
    def test_plot_methods_run(self, method):
        assert isinstance(getattr(self.plotter, method)(), plt.Axes)

    def test_plot_dos(self):
        ax = self.plotter.plot_dos(sigma=0.05)
        lines = ax.get_lines()
        # total DOS + E_f marker + zero-DOS line
        assert len(lines) == 3
        assert_allclose(lines[0].get_xdata(), self.bz.dos.energies - self.bz.dos.efermi)
        assert_allclose(lines[0].get_ydata(), self.bz.dos.get_smeared_densities(0.05)[Spin.up])
        assert lines[0].get_data()[0][0] == approx(-2.4197044934588674)

    @pytest.mark.parametrize(
        ("method", "getter", "kwargs", "legend"),
        [
            ("plot_seebeck_mu", "get_seebeck", {}, ["S$_1$", "S$_2$", "S$_3$"]),
            (
                "plot_conductivity_mu",
                "get_conductivity",
                {"relaxation_time": 1e-14},
                ["$\\Sigma_1$", "$\\Sigma_2$", "$\\Sigma_3$"],
            ),
            ("plot_power_factor_mu", "get_power_factor", {"relaxation_time": 1e-14}, ["PF$_1$", "PF$_2$", "PF$_3$"]),
            ("plot_zt_mu", "get_zt", {"relaxation_time": 1e-14}, ["ZT$_1$", "ZT$_2$", "ZT$_3$"]),
        ],
    )
    def test_plot_mu(self, method, getter, kwargs, legend):
        bz, temp = self.bz, 600
        ax = getattr(self.plotter, method)(temp=temp, output="eig", **kwargs)
        expected = np.array(getattr(bz, getter)(output="eig", doping_levels=False, **kwargs)[temp])
        lines = ax.get_lines()
        # 3 eigenvalues + 2 band-gap limits + 4 doping-level markers
        assert len(lines) == 9
        for idx in range(3):
            assert_allclose(lines[idx].get_xdata(), bz.mu_steps)
            assert_allclose(lines[idx].get_ydata(), expected[:, idx])
        assert [text.get_text() for text in ax.get_legend().get_texts()] == legend
        # set_xticks(get_xticks()) may widen the view to the outermost ticks, so only check containment
        x_lo, x_hi = ax.get_xlim()
        assert x_lo <= -0.5
        assert x_hi >= bz.gap + 0.5
        vlines_x = [line.get_xdata()[0] for line in lines[3:]]
        assert vlines_x == approx(
            [
                0,
                bz.gap,
                bz.mu_doping["n"][temp][0],
                bz.mu_doping["n"][temp][-1],
                bz.mu_doping["p"][temp][0],
                bz.mu_doping["p"][temp][-1],
            ]
        )
        assert len(ax.texts) == 4

        # average output and custom xlim
        ax = getattr(self.plotter, method)(temp=temp, output="average", xlim=(-1, 1), **kwargs)
        expected = getattr(bz, getter)(output="average", doping_levels=False, **kwargs)[temp]
        assert len(ax.get_lines()) == 7
        assert_allclose(ax.get_lines()[0].get_ydata(), expected)
        assert ax.get_legend() is None
        x_lo, x_hi = ax.get_xlim()
        assert x_lo <= -1
        assert x_hi >= 1

    def test_plot_seebeck_mu_first_point(self):
        ax = self.plotter.plot_seebeck_mu()
        assert ax.get_lines()[0].get_data()[0][0] == approx(-2.0702422655947665)
        assert ax.get_lines()[0].get_data()[1][0] == approx(-433.11096000000003)

    def test_plot_conductivity_mu_scale(self):
        ax = self.plotter.plot_conductivity_mu(relaxation_time=1e-13)
        assert ax.get_yscale() == "log"
        # ylim is set to (1e13, 1e20) * tau; set_yticks may then widen it to the outer ticks
        y_lo, y_hi = ax.get_ylim()
        assert y_lo <= 1e13 * 1e-13
        assert y_hi >= 1e20 * 1e-13

    def test_plot_carriers(self):
        bz = self.bz
        ax = self.plotter.plot_carriers(temp=300)
        lines = ax.get_lines()
        assert len(lines) == 7
        expected = [abs(c / (bz.vol * 1e-24)) for c in bz._carrier_conc[300]]
        assert_allclose(lines[0].get_xdata(), bz.mu_steps)
        assert_allclose(lines[0].get_ydata(), expected)
        assert lines[0].get_data()[1][0] == approx(6.525490122298364e22)
        assert ax.get_yscale() == "log"
        assert ax.get_ylabel() == "carrier concentration (cm-3)"

    def test_plot_hall_carriers(self):
        bz = self.bz
        ax = self.plotter.plot_hall_carriers(temp=300)
        lines = ax.get_lines()
        assert len(lines) == 7
        expected = [abs(c) for c in bz.get_hall_carrier_concentration()[300]]
        assert_allclose(lines[0].get_ydata(), expected)
        assert lines[0].get_data()[1][0] == approx(9.538187273102463e17)
        assert ax.get_ylabel() == "Hall carrier concentration (cm-3)"

    @pytest.mark.parametrize(
        ("method", "getter", "kwargs"),
        [
            ("plot_seebeck_dop", "get_seebeck", {}),
            ("plot_conductivity_dop", "get_conductivity", {"relaxation_time": 1e-14}),
            ("plot_power_factor_dop", "get_power_factor", {"relaxation_time": 1e-14}),
            ("plot_zt_dop", "get_zt", {"relaxation_time": 1e-14}),
            ("plot_eff_mass_dop", "get_average_eff_mass", {}),
        ],
    )
    @pytest.mark.parametrize("output", ["average", "eigs"])
    def test_plot_dop(self, method, getter, kwargs, output):
        bz = self.bz
        expected = getattr(bz, getter)(output=output, **kwargs)
        temps = sorted(expected["n"]) if output == "average" else [300]
        plot_kwargs = {"output": output} if output == "average" else {"output": output, "temps": temps}
        ax = getattr(self.plotter, method)(**plot_kwargs, **kwargs)
        lines = _all_lines(ax)
        n_comp = 1 if output == "average" else 3
        assert len(lines) == 2 * len(temps) * n_comp
        for dop_type in "np":
            for temp in temps:
                values = np.array(expected[dop_type][temp]).reshape(len(bz.doping[dop_type]), n_comp)
                for comp in range(n_comp):
                    assert any(
                        np.allclose(line.get_xdata(), bz.doping[dop_type])
                        and np.allclose(line.get_ydata(), values[:, comp])
                        for line in lines
                    )

    @pytest.mark.parametrize(
        ("method", "getter", "kwargs"),
        [
            ("plot_seebeck_temp", "get_seebeck", {}),
            ("plot_conductivity_temp", "get_conductivity", {"relaxation_time": 1e-14}),
            ("plot_power_factor_temp", "get_power_factor", {"relaxation_time": 1e-14}),
            ("plot_zt_temp", "get_zt", {"relaxation_time": 1e-14}),
            ("plot_eff_mass_temp", "get_average_eff_mass", {}),
        ],
    )
    @pytest.mark.parametrize("output", ["average", "eigs"])
    def test_plot_temp(self, method, getter, kwargs, output):
        bz = self.bz
        expected = getattr(bz, getter)(output=output, **kwargs)
        temps = sorted(expected["n"])
        dopings = bz.doping["n"] if output == "average" else [1e18]
        plot_kwargs = {"output": output} if output == "average" else {"output": output, "doping": dopings}
        ax = getattr(self.plotter, method)(**plot_kwargs, **kwargs)
        lines = _all_lines(ax)
        n_comp = 1 if output == "average" else 3
        assert len(lines) == 2 * len(dopings) * n_comp
        for dop_type in "np":
            for dop in dopings:
                dop_idx = bz.doping[dop_type].index(dop)
                values = np.array([expected[dop_type][temp][dop_idx] for temp in temps]).reshape(len(temps), n_comp)
                for comp in range(n_comp):
                    assert any(
                        np.allclose(line.get_xdata(), temps) and np.allclose(line.get_ydata(), values[:, comp])
                        for line in lines
                    )

    def test_plot_eff_mass_temp_panels(self):
        # plot_eff_mass_temp does not use the broken set_xticks call
        bz = self.bz
        ax = self.plotter.plot_eff_mass_temp()
        panels = {axis.get_title(): axis for axis in ax.figure.axes if axis.get_title()}
        assert set(panels) == {"n-type", "p-type"}
        eff_mass = bz.get_average_eff_mass(output="average")
        temps = sorted(eff_mass["n"])
        for dop_type, axis in panels.items():
            lines = axis.get_lines()
            assert len(lines) == len(bz.doping["n"])
            dop_key = dop_type[0]
            assert_allclose(lines[0].get_xdata(), temps)
            assert_allclose(lines[0].get_ydata(), [eff_mass[dop_key][temp][0] for temp in temps])
        assert panels["n-type"].get_ylabel() == "Effective mass (m$_e$)"
        assert panels["p-type"].get_ylabel() == ""

    def test_plot_eff_mass_temp_returns_populated_axes(self):
        ax = self.plotter.plot_eff_mass_temp()
        assert len(ax.get_lines()) > 0

    @pytest.mark.parametrize(
        "method",
        [
            "plot_seebeck_temp",
            "plot_conductivity_temp",
            "plot_power_factor_temp",
            "plot_zt_temp",
            "plot_seebeck_dop",
            "plot_conductivity_dop",
            "plot_power_factor_dop",
            "plot_zt_dop",
            "plot_eff_mass_dop",
        ],
    )
    def test_plot_temp_dop_panels(self, method):
        ax = getattr(self.plotter, method)()
        panels = {axis.get_title(): axis for axis in ax.figure.axes if axis.get_title()}
        assert set(panels) == {"n-type", "p-type"}
        assert len(panels["n-type"].get_lines()) == len(panels["p-type"].get_lines()) > 0

    def test_plot_seebeck_eff_mass_mu(self):
        pytest.importorskip("fdint")
        ax = self.plotter.plot_seebeck_eff_mass_mu()
        assert len(ax.get_lines()) == 2, "wrong number of lines"
        assert ax.get_lines()[0].get_data()[0][0] == approx(-2.0702422655947665), "wrong 0 data in line 0"
        assert ax.get_lines()[0].get_data()[1][0] == approx(6412.881888198197), "wrong 1 data in line 0"

    def test_plot_complexity_factor_mu(self):
        pytest.importorskip("fdint")
        ax = self.plotter.plot_complexity_factor_mu()
        assert len(ax.get_lines()) == 2, "wrong number of lines"
        assert ax.get_lines()[0].get_data()[0][0] == approx(-2.0702422655947665), "wrong 0 data in line 0"
        assert ax.get_lines()[0].get_data()[1][0] == approx(0.004708835456903449), "wrong 1 data in line 0"


class TestCohpPlotter(MatSciTest):
    def setup_method(self):
        path = f"{TEST_FILES_DIR}/electronic_structure/cohp/complete_cohp_lobster.json"
        with open(path, "rb") as file:
            self.cohp = CompleteCohp.from_dict(orjson.loads(file.read()))
        path = f"{TEST_FILES_DIR}/electronic_structure/cohp/complete_coop_lobster.json"
        with open(path, "rb") as file:
            self.coop = CompleteCohp.from_dict(orjson.loads(file.read()))
        self.cohp_plot = CohpPlotter(zero_at_efermi=False)
        self.coop_plot = CohpPlotter(are_coops=True)

    def test_attributes(self):
        assert not self.cohp_plot.are_coops
        assert self.coop_plot.are_coops
        assert not self.cohp_plot.zero_at_efermi
        assert self.coop_plot.zero_at_efermi
        self.cohp_plot.add_cohp_dict(self.cohp.all_cohps)
        cohp_energies = self.cohp_plot._cohps["1"]["energies"]
        assert len(cohp_energies) == 301
        assert cohp_energies[0] == approx(-0.27768)
        assert cohp_energies[-1] == approx(14.77248)
        self.coop_plot.add_cohp_dict(self.coop.all_cohps)
        coop_energies = self.coop_plot._cohps["10"]["energies"]
        assert len(coop_energies) == 241
        assert coop_energies[0] == approx(-6.02510)
        assert coop_energies[-1] == approx(6.02510)

    def test_add_cohp_dict(self):
        # Sorts the populations by z-coordinates of the sites
        def sortkeys(sites):
            return sites[0].z, sites[1].z

        sorted_keys = ["3", "4", "7", "8", "9", "10", "11", "6", "5", "2", "1"]

        d_coop = self.coop_plot.get_cohp_dict()
        assert len(d_coop) == 0
        bonds = self.coop.bonds
        self.coop_plot.add_cohp_dict(self.coop.all_cohps, key_sort_func=lambda x: sortkeys(bonds[x]["sites"]))
        d_coop = self.coop_plot.get_cohp_dict()
        assert len(d_coop) == 11
        assert list(self.coop_plot._cohps) == sorted_keys

    def test_get_cohp_dict(self):
        self.cohp_plot.add_cohp_dict(self.cohp.all_cohps)
        d_cohp = self.cohp_plot.get_cohp_dict()
        for bond in ["1", "2"]:
            assert bond in d_cohp

    def test_get_plot(self):
        self.cohp_plot.add_cohp_dict(self.cohp.all_cohps)
        ax_cohp = self.cohp_plot.get_plot()

        assert ax_cohp.get_xlabel() == "-COHP"
        assert ax_cohp.get_ylabel() == "$E$ (eV)"
        legend_labels = ax_cohp.get_legend_handles_labels()[1]
        assert len(self.cohp_plot._cohps) == len(legend_labels)
        assert ax_cohp.lines[0].get_linestyle() == "-"
        assert ax_cohp.lines[1].get_linestyle() == "--"
        for label in legend_labels:
            assert label in self.cohp_plot._cohps
        lines_index = legend_labels.index("1")
        line_styles = {Spin.up: "-", Spin.down: "--"}
        cohp_fe_fe = self.cohp.all_cohps["1"]
        for s, spin in enumerate([Spin.up, Spin.down]):
            lines = ax_cohp.lines[2 * lines_index + s]
            assert_allclose(lines.get_xdata(), -cohp_fe_fe.cohp[spin])
            assert_allclose(lines.get_ydata(), self.cohp.energies)
            assert lines.get_linestyle() == line_styles[spin]
        plt.close()

        ax_cohp = self.cohp_plot.get_plot(invert_axes=False, plot_negative=False)

        assert ax_cohp.get_xlabel() == "$E$ (eV)"
        assert ax_cohp.get_ylabel() == "COHP"
        for s, spin in enumerate([Spin.up, Spin.down]):
            lines = ax_cohp.lines[2 * lines_index + s]
            assert_allclose(lines.get_xdata(), self.cohp.energies)
            assert_allclose(lines.get_ydata(), cohp_fe_fe.cohp[spin])
        plt.close()

        ax_cohp = self.cohp_plot.get_plot(integrated=True)

        assert ax_cohp.get_xlabel() == "-ICOHP (eV)"
        for s, spin in enumerate([Spin.up, Spin.down]):
            lines = ax_cohp.lines[2 * lines_index + s]
            assert_allclose(lines.get_xdata(), -cohp_fe_fe.icohp[spin])

        coop_dict = {"Bi5-Bi6": self.coop.all_cohps["10"]}
        self.coop_plot.add_cohp_dict(coop_dict)
        ax_coop = self.coop_plot.get_plot()
        assert ax_coop.get_xlabel() == "COOP"
        assert ax_coop.get_ylabel() == "$E - E_f$ (eV)"
        lines_coop = ax_coop.get_lines()[0]
        assert_allclose(lines_coop.get_ydata(), self.coop.energies - self.coop.efermi)
        coop_bi_bi = self.coop.all_cohps["10"].cohp[Spin.up]
        assert_allclose(lines_coop.get_xdata(), coop_bi_bi)

        # cleanup
        plt.close("all")

    def test_save_plot(self):
        self.cohp_plot.add_cohp_dict(self.cohp.all_cohps)
        ax = self.cohp_plot.get_plot()
        assert isinstance(ax, plt.Axes)
        self.cohp_plot.save_plot(f"{self.tmp_path}/cohpplot.png")
        assert os.path.isfile(f"{self.tmp_path}/cohpplot.png")
        plt.close("all")


def _load_bs(path: str) -> BandStructureSymmLine:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as file:
        return BandStructureSymmLine.from_dict(orjson.loads(file.read()))


def _truncate_bands(bs: BandStructureSymmLine, band_indices: list[int], structure=None) -> BandStructureSymmLine:
    """Keep only a few bands (and their projections) so projected plots stay fast."""
    return BandStructureSymmLine(
        kpoints=[kpt.frac_coords for kpt in bs.kpoints],
        eigenvals={spin: bs.bands[spin][band_indices] for spin in bs.bands},
        lattice=bs.lattice_rec,
        efermi=bs.efermi,
        labels_dict={kpt.label: kpt.frac_coords for kpt in bs.kpoints if kpt.label},
        structure=structure or bs.structure,
        projections={spin: bs.projections[spin][band_indices] for spin in bs.projections},
    )


@pytest.fixture(scope="module")
def bs_cu2o():
    """Non-spin-polarized insulator, 2 elements (O sites 1-2, Cu sites 3-6), 4 bands around the gap."""
    return _truncate_bands(_load_bs(f"{BAND_TEST_DIR}/Cu2O_361_bandstructure.json"), [38, 39, 40, 41])


@pytest.fixture(scope="module")
def bs_n2():
    """Spin-polarized insulator with projections (N sites 1-2), 4 bands around the gap."""
    return _truncate_bands(_load_bs(f"{BAND_TEST_DIR}/N2_12103_bandstructure.json"), [4, 5, 6, 7])


@pytest.fixture(scope="module")
def bs_c_metal():
    """Metallic C with projections, only the 2 bands crossing E_f."""
    return _truncate_bands(_load_bs(f"{BAND_TEST_DIR}/C_48_bandstructure.json"), [7, 8])


@pytest.fixture(scope="module")
def bs_spin_metal(bs_cu2o):
    """Synthetic spin-polarized metal with equal-length branches, built from the Cu2O bands.

    Spin down is spin up rigidly shifted by +0.05 eV; E_f is placed inside the top valence band.
    """
    up = bs_cu2o.bands[Spin.up]
    return BandStructureSymmLine(
        kpoints=[kpt.frac_coords for kpt in bs_cu2o.kpoints],
        eigenvals={Spin.up: up, Spin.down: up + 0.05},
        lattice=bs_cu2o.lattice_rec,
        efermi=float(np.mean(up[1])),
        labels_dict={kpt.label: kpt.frac_coords for kpt in bs_cu2o.kpoints if kpt.label},
        structure=bs_cu2o.structure,
        projections=dict.fromkeys((Spin.up, Spin.down), bs_cu2o.projections[Spin.up]),
    )


@pytest.fixture(scope="module")
def bs_cao():
    return _load_bs(f"{BAND_TEST_DIR}/CaO_2605_bandstructure.json")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestDosPlotterOptions:
    @pytest.fixture(autouse=True)
    def _dos(self):
        with open(f"{BAND_TEST_DIR}/../dos/complete_dos.json", "rb") as file:
            self.dos = CompleteDos.from_dict(orjson.loads(file.read()))

    def test_beta_dashed_unstacked(self):
        plotter = DosPlotter(stack=False)
        plotter.add_dos("total", self.dos)
        ax = plotter.get_plot(beta_dashed=True)
        # spin up, spin down, E_f marker, zero-DOS line
        lines = ax.get_lines()
        assert len(lines) == 4
        assert lines[0].get_linestyle() == "-"
        assert lines[1].get_linestyle() == "--"
        assert_allclose(lines[0].get_ydata(), self.dos.densities[Spin.up])
        assert_allclose(lines[1].get_ydata(), -self.dos.densities[Spin.down])
        assert_allclose(lines[0].get_xdata(), self.dos.energies - self.dos.efermi)
        assert ax.get_ylabel() == "Density of states (states/eV)"
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ["total"]

    def test_norm_vol_labels(self):
        dos = Dos(self.dos.efermi, self.dos.energies, self.dos.densities, norm_vol=1.0)
        plotter = DosPlotter(zero_at_efermi=False)
        plotter.add_dos("normed", dos)
        ax = plotter.get_plot()
        assert ax.get_ylabel() == "Density of states (states/eV/Å³)"
        assert_allclose(ax.get_lines()[0].get_xdata(), dos.energies)
        ax = plotter.get_plot(invert_axes=True)
        assert ax.get_xlabel() == "Density of states (states/eV/Å³)"
        assert ax.get_ylabel() == "Energies (eV)"

    def test_show_and_save(self, monkeypatch, tmp_path):
        shown = []
        monkeypatch.setattr(plt, "show", lambda: shown.append(True))
        plotter = DosPlotter()
        plotter.add_dos_dict(self.dos.get_element_dos())
        plotter.show(xlim=(-2, 2))
        assert shown == [True]
        assert plt.gca().get_xlim() == approx((-2, 2))
        plotter.save_plot(f"{tmp_path}/dos.png", invert_axes=True)
        assert os.path.isfile(f"{tmp_path}/dos.png")


class TestBSPlotterExtra:
    def test_rejects_non_symm_line(self, bs_cao):
        uniform = BandStructure(
            [kpt.frac_coords for kpt in bs_cao.kpoints], bs_cao.bands, bs_cao.lattice_rec, bs_cao.efermi
        )
        with pytest.raises(TypeError, match="only works with BandStructureSymmLine"):
            BSPlotter(uniform)

    def test_rejects_different_kpath(self, bs_cao, bs_n2):
        plotter = BSPlotter(bs_cao)
        with pytest.raises(ValueError, match="which have the same kpath"):
            plotter.add_bs(bs_n2)
        with pytest.raises(ValueError, match="which have the same kpath"):
            BSPlotter([bs_cao, bs_n2])

    def test_interpolate_bands_short_branches(self):
        distances = [np.linspace(0, 1, 5), np.array([1.0, 1.5]), np.array([2.0])]
        energies = [np.vstack([dist**2, dist]) for dist in distances]
        with pytest.warns(UserWarning, match="number of points") as record:
            int_dist, int_en = BSPlotter._interpolate_bands(distances, energies, smooth_np=7)
        assert any("Skipping single point branch" in str(warning.message) for warning in record)
        # single-point branch is dropped
        assert len(int_dist) == len(int_en) == 2
        assert int_en[0].shape == int_en[1].shape == (2, 7)
        # a cubic spline through x**2 reproduces it exactly; the 2-point branch falls back to linear
        assert_allclose(int_en[0][0], int_dist[0] ** 2, atol=1e-10)
        assert_allclose(int_en[1][1], int_dist[1], atol=1e-10)

    def test_get_plot_ylim_and_layout_callbacks(self, bs_cao):
        plotter = BSPlotter(bs_cao)
        ax = plotter.get_plot(ylim=(-2, 3), bs_labels=["CaO"])
        assert ax.get_ylim() == approx((-2, 3))
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ["CaO up"]
        canvas = ax.figure.canvas
        # the registered fix_layout callback must handle both event types without error
        canvas.callbacks.process("resize_event", ResizeEvent("resize_event", canvas))
        canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key="t"))

    def test_show(self, bs_cao, monkeypatch):
        shown = []
        monkeypatch.setattr(plt, "show", lambda: shown.append(True))
        BSPlotter(bs_cao).show(ylim=(-1, 1))
        assert shown == [True]
        assert plt.gca().get_ylim() == approx((-1, 1))

    def test_get_ticks_old(self, bs_cao):
        plotter = BSPlotter(bs_cao)
        ticks_old = plotter.get_ticks_old()
        ticks = plotter.get_ticks()
        # The legacy k-point-based implementation agrees with the branch-based one
        assert ticks_old["label"] == ticks["label"]
        assert ticks_old["distance"] == approx(ticks["distance"])
        assert len(ticks_old["label"]) == 19
        assert ticks_old["label"][0] == "$\\Gamma$"
        # the K|U discontinuity is joined with a mid bar
        assert ticks_old["label"][-2] == "K$\\mid$U"
        assert ticks_old["distance"][-1] == approx(8.334526261560772)

    def test_plot_compare(self, bs_cao):
        plotter = BSPlotter(bs_cao)
        with pytest.warns(UserWarning, match="Deprecated method"):
            ax = plotter.plot_compare(BSPlotter(bs_cao))
        assert len(ax.get_legend().get_texts()) == 4

    def test_plot_brillouin(self, bs_cao, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)
        fig = BSPlotter(bs_cao).plot_brillouin()
        ax = fig.axes[0]
        labels = {kpt.label for kpt in bs_cao.kpoints if kpt.label}
        assert {text.get_text() for text in ax.texts} == {
            f"${lbl}$" if lbl.startswith("\\") or "_" in lbl else lbl for lbl in labels
        }
        # 3 lattice vectors + WS cell edges + one path segment per branch
        assert len(ax.get_lines()) > 3 + len(bs_cao.branches)


class TestBSPlotterProjectedExtra:
    def test_list_input_warns(self, bs_cu2o):
        with pytest.warns(UserWarning, match="Only the first in the list"):
            plotter = BSPlotterProjected([bs_cu2o, bs_cu2o])
        assert plotter._bs is bs_cu2o
        assert plotter._nb_bands == 4

    @pytest.mark.parametrize("bs_name", ["bs_cu2o", "bs_n2"])
    def test_projections_by_branches_spin_up(self, bs_name, request):
        bs = request.getfixturevalue(bs_name)
        element = bs.structure.elements[-1].symbol
        plotter = BSPlotterProjected(bs)
        proj_br = plotter._get_projections_by_branches({element: ["s", "p"]})
        ref = bs.get_projections_on_elements_and_orbitals({element: ["s", "p"]})
        assert len(proj_br) == len(bs.branches)
        for br_idx, branch in enumerate(bs.branches):
            n_pts = branch["end_index"] - branch["start_index"] + 1
            for band_idx in range(bs.nb_bands):
                assert len(proj_br[br_idx]["1"][band_idx]) == n_pts
                for j in (0, n_pts - 1):
                    k_idx = branch["start_index"] + j
                    assert proj_br[br_idx]["1"][band_idx][j][element] == approx(ref[Spin.up][band_idx][k_idx][element])

    def test_projections_by_branches_spin_down(self, bs_n2):
        plotter = BSPlotterProjected(bs_n2)
        proj_br = plotter._get_projections_by_branches({"N": ["s", "p"]})
        ref = bs_n2.get_projections_on_elements_and_orbitals({"N": ["s", "p"]})
        branch = bs_n2.branches[1]
        n_pts = branch["end_index"] - branch["start_index"] + 1
        assert len(proj_br[1]["-1"][0]) == n_pts
        assert proj_br[1]["-1"][0][0]["N"] == approx(ref[Spin.down][0][branch["start_index"]]["N"])

    def test_projected_plots_dots_spin_polarized(self, bs_spin_metal):
        bs = bs_spin_metal
        plotter = BSPlotterProjected(bs)
        axes = plotter.get_projected_plots_dots({"Cu": ["d"]}, marker_size=5)
        assert len(axes) == 1
        ax = axes[0]
        n_branches = len(bs.branches)
        lines = ax.get_lines()
        up_lines = [line for line in lines if line.get_linestyle() == "-" and line.get_color() == "b"]
        down_lines = [line for line in lines if line.get_linestyle() == "--" and line.get_color() == "r"]
        # one band line per (branch, band) per spin
        assert len(down_lines) == n_branches * bs.nb_bands
        assert len(up_lines) == n_branches * bs.nb_bands
        assert_allclose(down_lines[0].get_ydata(), up_lines[0].get_ydata() + 0.05)
        # one dot per k-point per band per spin
        dots = [line for line in lines if line.get_marker() == "o"]
        assert len(dots) == 2 * len(bs.kpoints) * bs.nb_bands
        assert ax.get_title() == "${\\mathrm{Cu}}_{\\mathrm{d}}$"

    def test_projected_plots_dots_unequal_branches(self, bs_n2):
        axes = BSPlotterProjected(bs_n2).get_projected_plots_dots({"N": ["p"]})
        assert axes[0].get_xlim() == approx((0, bs_n2.distance[-1]))

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"zero_to_efermi": True}, (-10, 10)),
            ({"zero_to_efermi": False}, "efermi"),
            ({"ylim": (-1, 2)}, (-1, 2)),
        ],
    )
    def test_projected_plots_dots_metal_ylim(self, bs_spin_metal, kwargs, expected):
        plotter = BSPlotterProjected(bs_spin_metal)
        axes = plotter.get_projected_plots_dots({"Cu": ["s", "p"], "O": ["p"]}, **kwargs)
        assert len(axes) == 4
        if expected == "efermi":
            expected = (bs_spin_metal.efermi - 10, bs_spin_metal.efermi + 10)
        for ax in axes[:3]:
            assert ax.get_ylim() == approx(expected)
        # O has a single orbital so its second row is hidden
        assert not axes[3].get_visible()

    def test_projected_plots_dots_vbm_cbm_marker(self, bs_cu2o):
        plotter = BSPlotterProjected(bs_cu2o)
        (ax,) = plotter.get_projected_plots_dots({"Cu": ["d"]}, vbm_cbm_marker=True)
        data = plotter.bs_plot_data()
        assert len(ax.collections) == len(data["vbm"]) + len(data["cbm"])
        assert ax.get_ylim() == approx((data["vbm"][0][1] - 4, data["cbm"][0][1] + 4))

    @pytest.mark.parametrize(
        ("bs_name", "kwargs", "expected"),
        [
            ("bs_c_metal", {"zero_to_efermi": True}, (-10, 10)),
            ("bs_c_metal", {"zero_to_efermi": False}, "efermi"),
            ("bs_cu2o", {"ylim": (-3, 3)}, (-3, 3)),
            ("bs_n2", {"vbm_cbm_marker": True}, "gap"),
        ],
    )
    def test_elt_projected_plots(self, bs_name, kwargs, expected, request):
        bs = request.getfixturevalue(bs_name)
        plotter = BSPlotterProjected(bs)
        axs = plotter.get_elt_projected_plots(**kwargs)
        assert axs.shape == (2, 2)
        n_elems = len(bs.structure.elements)
        if expected == "efermi":
            expected = (bs.efermi - 10, bs.efermi + 10)
        elif expected == "gap":
            data = plotter.bs_plot_data()
            expected = (data["vbm"][0][1] - 4, data["cbm"][0][1] + 4)
        for idx, ax in enumerate(axs.flat[:n_elems]):
            assert ax.get_title() == str(bs.structure.elements[idx])
            assert ax.get_ylim() == approx(expected)
        if kwargs.get("vbm_cbm_marker"):
            data = plotter.bs_plot_data()
            assert len(axs.flat[0].collections) == len(data["vbm"]) + len(data["cbm"])
        if bs.is_spin_polarized:
            dashed = [line for line in axs.flat[0].get_lines() if line.get_linestyle() == "--"]
            assert len(dashed) == len(bs.branches) * bs.nb_bands

    def test_elt_projected_plots_no_structure(self, bs_cu2o, monkeypatch):
        plotter = BSPlotterProjected(bs_cu2o)
        monkeypatch.setattr(plotter._bs, "structure", None)
        with pytest.raises(RuntimeError, match="cannot be None"):
            plotter.get_elt_projected_plots()
        with pytest.raises(RuntimeError, match="cannot be None"):
            plotter.get_elt_projected_plots_color()

    def test_elt_projected_plots_color_binary(self, bs_cu2o):
        plotter = BSPlotterProjected(bs_cu2o)
        ax = plotter.get_elt_projected_plots_color()
        # band segments, excluding the vertical high-symmetry tick lines
        segments = [line for line in ax.get_lines() if line.get_xdata()[0] != line.get_xdata()[-1]]
        assert len(segments) == (len(bs_cu2o.kpoints) - len(bs_cu2o.branches)) * bs_cu2o.nb_bands
        for line in segments:
            red, green, blue = line.get_color()[:3]
            # binary compound: 1st element -> red, 2nd -> blue, no green; fractions sum to 1
            assert green == approx(0)
            assert red + blue == approx(1) or red == blue == approx(0)
        data = plotter.bs_plot_data()
        assert ax.get_ylim() == approx((data["vbm"][0][1] - 4, data["cbm"][0][1] + 2))

    def test_elt_projected_plots_color_spin_polarized(self, bs_spin_metal):
        bs = bs_spin_metal
        ax = BSPlotterProjected(bs).get_elt_projected_plots_color(
            zero_to_efermi=False, elt_ordered=bs.structure.elements[::-1]
        )
        dashed = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
        # exclude the vertical high-symmetry tick lines
        segments = [line for line in ax.get_lines() if line.get_xdata()[0] != line.get_xdata()[-1]]
        n_segments = (len(bs.kpoints) - len(bs.branches)) * bs.nb_bands
        assert len(dashed) == n_segments
        assert len(segments) == 2 * n_segments
        assert ax.get_xlim() == approx((0, bs.distance[-1]))

    def test_elt_projected_plots_color_too_many_elements(self):
        with open(f"{TEST_FILES_DIR}/electronic_structure/plotter/SrBa2Sn2O7.json", "rb") as file:
            bs = BandStructureSymmLine.from_dict(orjson.loads(file.read()))
        with pytest.raises(ValueError, match="Can only plot binary and ternary compounds, got 4 elements"):
            BSPlotterProjected(bs).get_elt_projected_plots_color()

    def test_elt_projected_plots_color_metal_ylim(self, bs_spin_metal):
        ax = BSPlotterProjected(bs_spin_metal).get_elt_projected_plots_color(zero_to_efermi=True)
        assert ax.get_ylim() == approx((-10, 10))

    @pytest.mark.parametrize(
        ("dictio", "sum_morbs", "exc", "match"),
        [
            (["Cu"], None, TypeError, "should be dict type"),
            ({}, None, KeyError, "'dictio' is empty"),
            ({"Xx": ["s"]}, None, KeyError, "invalid element was put into 'dictio'"),
            ({"Cu": "s"}, None, TypeError, "should be list type"),
            ({"Cu": []}, None, ValueError, "is empty"),
            ({"Cu": [1]}, None, TypeError, "They should be string"),
            ({"Cu": ["g"]}, None, ValueError, "invalid name of orbital"),
            ({"Cu": ["p", "px"]}, None, ValueError, "contains orbitals repeated"),
            ({"Cu": ["s", "s"]}, None, ValueError, "two similar orbitals"),
            ({"Cu": ["s"]}, ["s"], TypeError, "should be dict or 'None' type"),
            ({"Cu": ["s"]}, {}, KeyError, "'sum_morbs' is empty"),
            ({"Cu": ["s"]}, {"Xx": ["s"]}, KeyError, "invalid element was put into 'sum_morbs'"),
            ({"Cu": ["s"]}, {"Cu": "s"}, TypeError, "should be list type"),
            ({"Cu": ["s"]}, {"Cu": [1]}, TypeError, "They should be string"),
            ({"Cu": ["s"]}, {"Cu": ["g"]}, ValueError, "invalid name of orbital in 'sum_morbs"),
            ({"Cu": ["d", "s"]}, {"Cu": ["d", "dxy"]}, ValueError, "contains orbitals repeated"),
            ({"Cu": ["d", "s"]}, {"Cu": ["dxy", "dxy"]}, ValueError, "two similar orbitals"),
            ({"Cu": ["d"]}, {"O": ["px", "py"]}, ValueError, "not mentioned in 'dictio'"),
            ({"Cu": ["dxy"]}, {"Cu": ["dxy"]}, ValueError, "over one individual orbital"),
            ({"Cu": ["d"]}, {"Cu": []}, ValueError, "empty list is an invalid value"),
            ({"Cu": ["d"]}, {"Cu": ["dxy", "px"]}, ValueError, "invalid orbital 'px'"),
            ({"Cu": ["d"]}, {"Cu": ["s"]}, ValueError, "invalid orbital 's'"),
            ({"Cu": ["d", "s"]}, {"Cu": []}, ValueError, "empty list is an invalid value"),
            ({"Cu": ["d", "s"]}, {"Cu": ["s"]}, ValueError, "only 's' orbital"),
            ({"Cu": ["d", "s"]}, {"Cu": ["dxy"]}, ValueError, "never sum projection over one orbital"),
            ({"Cu": ["dxy", "s"]}, {"Cu": ["p"]}, ValueError, "conflict with those of dictio"),
        ],
    )
    def test_orbitals_sum_orbitals_errors(self, dictio, sum_morbs, exc, match):
        with pytest.raises(exc, match=match):
            BSPlotterProjected._Orbitals_SumOrbitals(dictio, sum_morbs)

    @pytest.mark.parametrize(
        ("dictio", "sum_morbs", "expected_dictio", "expected_sum_morbs"),
        [
            # a single subshell in both dictio and sum_morbs expands both to the individual orbitals
            (
                {"Cu": ["d"]},
                {"Cu": ["d"]},
                {"Cu": ["dxy", "dyz", "dxz", "dx2_y2", "dz2"]},
                {"Cu": ["dxy", "dyz", "dxz", "dx2_y2", "dz2"]},
            ),
            (
                {"Cu": ["d", "s"]},
                {"Cu": ["d"]},
                {"Cu": ["s", "dxy", "dyz", "dxz", "dx2_y2", "dz2"]},
                {"Cu": ["dxy", "dyz", "dxz", "dx2_y2", "dz2"]},
            ),
            (
                {"Cu": ["p", "s"], "O": ["s"]},
                {"Cu": ["p", "s"]},
                {"Cu": ["s", "px", "py", "pz"], "O": ["s"]},
                {"Cu": ["s", "px", "py", "pz"]},
            ),
            ({"Cu": ["s", "dxy"]}, None, {"Cu": ["s", "dxy"]}, None),
        ],
    )
    def test_orbitals_sum_orbitals_expansion(self, dictio, sum_morbs, expected_dictio, expected_sum_morbs):
        dictio_out, sum_morbs_out = BSPlotterProjected._Orbitals_SumOrbitals(
            copy.deepcopy(dictio), copy.deepcopy(sum_morbs)
        )
        assert dictio_out == expected_dictio
        assert sum_morbs_out == expected_sum_morbs

    @pytest.mark.parametrize(
        ("dictio", "dictpa", "sum_atoms", "exc", "match"),
        [
            ({"Cu": ["s"]}, ["Cu"], None, TypeError, "'dictpa' was bound"),
            ({"Cu": ["s"]}, {}, None, KeyError, "'dictpa' is empty"),
            ({"Cu": ["s"]}, {"Xx": [3]}, None, KeyError, "invalid element was put into 'dictpa'"),
            ({"Cu": ["s"]}, {"Cu": 3}, None, TypeError, "should be list type"),
            ({"Cu": ["s"]}, {"Cu": []}, None, ValueError, "is empty"),
            ({"Cu": ["s"]}, {"Cu": ["three"]}, None, ValueError, "wrong site numbers"),
            ({"Cu": ["s"]}, {"Cu": [1]}, None, ValueError, "wrong site numbers"),
            ({"Cu": ["s"]}, {"Cu": [3.0]}, None, TypeError, "wrong site numbers"),
            ({"Cu": ["s"]}, {"Cu": [3, 3]}, None, ValueError, "two similar site numbers"),
            ({"Cu": ["s"]}, {"Cu": [3], "O": [1]}, None, KeyError, "are not the same"),
            ({"Cu": ["s"]}, {"O": [1]}, None, KeyError, "is not in both dictpa and dictio"),
            ({"Cu": ["s"]}, {"Cu": [3, 4]}, ["Cu"], TypeError, "'sum_atoms' was bound"),
            ({"Cu": ["s"]}, {"Cu": [3, 4]}, {}, KeyError, "'sum_atoms' is empty"),
            ({"Cu": ["s"]}, {"Cu": [3, 4]}, {"Xx": [3]}, KeyError, "invalid element was put into 'sum_atoms'"),
            ({"Cu": ["s"]}, {"Cu": [3, 4]}, {"Cu": 3}, TypeError, "should be list type"),
            ({"Cu": ["s"]}, {"Cu": [3, 4]}, {"Cu": []}, ValueError, "is empty"),
            ({"Cu": ["s"]}, {"Cu": [3, 4]}, {"Cu": ["three"]}, ValueError, "wrong site numbers"),
            ({"Cu": ["s"]}, {"Cu": [3, 4]}, {"Cu": [1]}, ValueError, "wrong site numbers"),
            ({"Cu": ["s"]}, {"Cu": [3, 4]}, {"Cu": [3, 5]}, ValueError, "not mentioned in dicpta"),
            ({"Cu": ["s"]}, {"Cu": [3, 4]}, {"Cu": [3.0]}, TypeError, "wrong site numbers"),
            ({"Cu": ["s"]}, {"Cu": [3, 4]}, {"Cu": [3, 3]}, ValueError, "two similar site numbers"),
            ({"Cu": ["s"]}, {"Cu": [3, 4]}, {"Cu": [3]}, ValueError, "only one atom"),
        ],
    )
    def test_number_of_subfigures_errors(self, bs_cu2o, dictio, dictpa, sum_atoms, exc, match):
        plotter = BSPlotterProjected(bs_cu2o)
        with pytest.raises(exc, match=match):
            plotter._number_of_subfigures(dictio, dictpa, sum_atoms, None)

    def test_number_of_subfigures_sum_atoms_not_in_dictpa(self, bs_cu2o):
        plotter = BSPlotterProjected(bs_cu2o)
        with pytest.raises(ValueError, match="You cannot sum projection over atoms 'O'"):
            plotter._number_of_subfigures({"Cu": ["s"]}, {"Cu": [3, 4]}, {"O": [1, 2]}, None)

    @pytest.mark.parametrize(
        ("dictio", "dictpa", "sum_atoms", "sum_morbs", "n_figs"),
        [
            ({"Cu": ["s", "dxy"], "O": ["px"]}, {"Cu": [3, 5], "O": [1]}, None, None, 5),
            ({"Cu": ["s", "dxy"], "O": ["px"]}, {"Cu": [3, 5], "O": [1]}, {"Cu": [3, 5]}, None, 3),
            ({"Cu": ["s", "dxy"], "O": ["px"]}, {"Cu": [3, 5], "O": [1]}, None, {"Cu": ["s", "dxy"]}, 3),
            ({"Cu": ["s", "dxy"], "O": ["px"]}, {"Cu": [3, 5], "O": [1]}, {"Cu": [3, 5]}, {"Cu": ["s", "dxy"]}, 2),
            (
                {"Cu": ["s"], "O": ["px", "py"]},
                {"Cu": [3, 5], "O": [1, 2]},
                {"Cu": [3, 5]},
                {"O": ["px", "py"]},
                3,
            ),
            ({"Cu": ["s"]}, {"Cu": ["all"]}, {"Cu": ["all"]}, None, 1),
        ],
    )
    def test_number_of_subfigures(self, bs_cu2o, dictio, dictpa, sum_atoms, sum_morbs, n_figs):
        plotter = BSPlotterProjected(bs_cu2o)
        dictpa_out, sum_atoms_out, n_out = plotter._number_of_subfigures(
            dictio, copy.deepcopy(dictpa), copy.deepcopy(sum_atoms), sum_morbs
        )
        assert n_out == n_figs
        if dictpa.get("Cu") == ["all"]:
            assert dictpa_out["Cu"] == [3, 4, 5, 6]
            assert sum_atoms_out["Cu"] == [3, 4, 5, 6]

    @pytest.mark.parametrize(
        ("dictio", "dictpa", "sum_atoms", "sum_morbs", "expected_dictio", "expected_dictpa"),
        [
            ({"Cu": ["s"]}, {"Cu": [3, 4, 5, 6]}, {"Cu": [3, 4, 5, 6]}, None, {"Cu": ["s"]}, {"Cu": ["all"]}),
            ({"Cu": ["s"]}, {"Cu": [3, 4, 5, 6]}, {"Cu": [3, 4, 6]}, None, {"Cu": ["s"]}, {"Cu": ["5", "3-4,6"]}),
            (
                {"Cu": ["s", "px", "py", "pz"], "O": ["s"]},
                {"Cu": [3], "O": [1]},
                None,
                {"Cu": ["px", "py", "pz"]},
                {"Cu": ["s", "p"], "O": ["s"]},
                {"Cu": ["3"], "O": ["1"]},
            ),
            (
                {"Cu": ["s", "px", "py", "pz"], "O": ["s"]},
                {"Cu": [3, 4], "O": [1]},
                {"Cu": [3, 4]},
                {"Cu": ["px", "py", "pz"]},
                {"Cu": ["s", "p"], "O": ["s"]},
                {"Cu": ["3-4"], "O": ["1"]},
            ),
            (
                {"Cu": ["s"], "O": ["s"]},
                {"Cu": [3, 4, 5, 6], "O": [1]},
                {"Cu": [3, 4, 5, 6]},
                {"Cu": ["s"]},
                {"Cu": ["s"], "O": ["s"]},
                {"Cu": ["all"], "O": ["1"]},
            ),
        ],
    )
    def test_summarize_keys_for_plot(
        self, bs_cu2o, dictio, dictpa, sum_atoms, sum_morbs, expected_dictio, expected_dictpa
    ):
        plotter = BSPlotterProjected(bs_cu2o)
        dictio_d, dictpa_d = plotter._summarize_keys_for_plot(dictio, dictpa, sum_atoms, sum_morbs)
        assert dictio_d == expected_dictio
        assert dictpa_d == expected_dictpa

    def test_summarize_keys_partial_subshell_label(self, bs_cu2o):
        plotter = BSPlotterProjected(bs_cu2o)
        dictio_d, _ = plotter._summarize_keys_for_plot(
            {"Cu": ["s", "dxy", "dyz"]}, {"Cu": [3]}, None, {"Cu": ["dxy", "dyz"]}
        )
        assert dictio_d == {"Cu": ["s", "dxy-yz"]}

    @pytest.mark.parametrize(
        ("selected_branches", "exc", "match"),
        [
            (1, TypeError, "should be 'list' type"),
            ([], ValueError, "is empty"),
            (["1"], TypeError, "should be 'int' type"),
            ([0], ValueError, "incorrect index of symmetry lines: 0"),
            ([7], ValueError, "incorrect index of symmetry lines: 7"),
        ],
    )
    def test_patom_pmorb_selected_branches_errors(self, bs_cu2o, selected_branches, exc, match):
        plotter = BSPlotterProjected(bs_cu2o)
        with pytest.raises(exc, match=match):
            plotter._get_projections_by_branches_patom_pmorb({"Cu": ["s"]}, {"Cu": [3]}, None, None, selected_branches)

    @staticmethod
    def _site_orb(bs, spin, band, k_idx, orb, site):
        setos = {"s": 0, "py": 1, "pz": 2, "px": 3, "dxy": 4, "dyz": 5, "dz2": 6, "dxz": 7, "dx2_y2": 8}
        return bs.projections[spin][band][k_idx][setos[orb]][site - 1]

    @pytest.mark.parametrize("bs_name", ["bs_cu2o", "bs_spin_metal"])
    def test_patom_pmorb_projection_values_mixed(self, bs_name, request):
        """Cu summed over atoms only, O summed over orbitals only."""
        bs = request.getfixturevalue(bs_name)
        plotter = BSPlotterProjected(bs)
        dictio, sum_morbs = plotter._Orbitals_SumOrbitals({"Cu": ["s", "px"], "O": ["px", "py"]}, {"O": ["px", "py"]})
        dictpa, sum_atoms, n_figs = plotter._number_of_subfigures(
            dictio, {"Cu": [3, 5], "O": [1]}, {"Cu": [3, 5]}, sum_morbs
        )
        assert n_figs == 3
        proj_br_d, dictio_d, dictpa_d, _ = plotter._get_projections_by_branches_patom_pmorb(
            dictio, dictpa, sum_atoms, sum_morbs, selected_branches=None
        )
        assert dictpa_d == {"Cu": ["3,5"], "O": ["1"]}
        assert dictio_d["Cu"] == ["s", "px"]
        assert len(dictio_d["O"]) == 1
        branch, band, j = bs.branches[2], 2, 5
        k_idx = branch["start_index"] + j
        for spin in bs.bands:
            edict = proj_br_d[2][str(spin)][band][j]
            for orb in ("s", "px"):
                assert edict["Cu3,5"][orb] == approx(
                    sum(self._site_orb(bs, spin, band, k_idx, orb, site) for site in (3, 5))
                )
            assert edict["O1"][dictio_d["O"][0]] == approx(
                sum(self._site_orb(bs, spin, band, k_idx, orb, 1) for orb in ("px", "py"))
            )

    @pytest.mark.parametrize("bs_name", ["bs_cu2o", "bs_n2", "bs_spin_metal"])
    @pytest.mark.parametrize("mode", ["none", "atoms", "morbs", "both"])
    def test_patom_pmorb_projection_values(self, bs_name, mode, request):
        bs = request.getfixturevalue(bs_name)
        plotter = BSPlotterProjected(bs)
        if bs_name == "bs_n2":
            el, sites, other = "N", [1, 2], None
        else:
            el, sites, other = "Cu", [3, 5], ("O", 1)
        dictio = {el: ["s", "px", "py"]}
        dictpa = {el: list(sites)}
        if other:
            dictio[other[0]] = ["s"]
            dictpa[other[0]] = [other[1]]
        sum_atoms = {el: list(sites)} if mode in ("atoms", "both") else None
        sum_morbs = {el: ["px", "py"]} if mode in ("morbs", "both") else None
        dictio, sum_morbs = plotter._Orbitals_SumOrbitals(dictio, sum_morbs)
        dictpa, sum_atoms, _ = plotter._number_of_subfigures(dictio, dictpa, sum_atoms, sum_morbs)
        proj_br_d, dictio_d, dictpa_d, branches = plotter._get_projections_by_branches_patom_pmorb(
            dictio, dictpa, sum_atoms, sum_morbs, selected_branches=[1, 3]
        )
        assert list(branches) == [0, 2]
        assert len(proj_br_d) == 2

        spins = [Spin.up, Spin.down] if bs.is_spin_polarized else [Spin.up]
        band, br_pos = 1, 1
        branch = bs.branches[branches[br_pos]]
        j = 3
        k_idx = branch["start_index"] + j

        # summed Cu 3,5 is a subset of the Cu sites (labelled "3,5"); N 1,2 covers all N sites ("all")
        summed_key = "all" if bs_name == "bs_n2" else "3,5"
        atom_keys = [summed_key] if sum_atoms else [str(site) for site in sites]
        assert dictpa_d[el] == atom_keys
        if sum_morbs:
            assert len(dictio_d[el]) == 2
            assert dictio_d[el][0] == "s"
        else:
            assert dictio_d[el] == ["s", "px", "py"]

        for spin in spins:
            edict = proj_br_d[br_pos][str(spin)][band][j]
            assert len(proj_br_d[br_pos][str(spin)][band]) == branch["end_index"] - branch["start_index"] + 1

            def val(orb, site, spin=spin):
                return self._site_orb(bs, spin, band, k_idx, orb, site)

            for key in atom_keys:
                site_list = sites if sum_atoms else [int(key)]
                assert edict[f"{el}{key}"]["s"] == approx(sum(val("s", site) for site in site_list))
                if sum_morbs:
                    morb_label = dictio_d[el][-1]
                    assert edict[f"{el}{key}"][morb_label] == approx(
                        sum(val(orb, site) for site in site_list for orb in ("px", "py"))
                    )
                else:
                    assert edict[f"{el}{key}"]["px"] == approx(sum(val("px", site) for site in site_list))
            if other:
                assert edict[f"{other[0]}{other[1]}"]["s"] == approx(val("s", other[1]))

    def test_patom_pmorb_plot_selected_branches(self, bs_cu2o):
        plotter = BSPlotterProjected(bs_cu2o)
        axes = plotter.get_projected_plots_dots_patom_pmorb(
            {"Cu": ["s", "dxy"], "O": ["s"]},
            {"Cu": [3, 5], "O": [1]},
            sum_atoms={"Cu": [3, 5]},
            selected_branches=[1, 3],
            vbm_cbm_marker=True,
        )
        # Cu(3,5 summed) x {s, dxy} + O1 x {s}
        assert [ax.get_title() for ax in axes] == ["Cu 3,5 s", "Cu 3,5 dxy", "O 1 s"]
        ax = axes[0]
        # Gamma-X and M-Gamma are discontinuous, so the middle tick is merged
        assert [lbl.get_text() for lbl in ax.get_xticklabels()] == ["$\\Gamma$", "X$\\mid$M", "$\\Gamma$"]
        # Both selected branches are shifted to start right after each other
        assert min(min(line.get_xdata()) for line in ax.get_lines()) == approx(0)
        data = plotter.bs_plot_data()
        assert ax.get_ylim() == approx((data["vbm"][0][1] - 4, data["cbm"][0][1] + 4))
        assert len(ax.collections) == len(data["vbm"]) + len(data["cbm"])

    def test_patom_pmorb_plot_single_figure_spin_polarized(self, bs_n2):
        plotter = BSPlotterProjected(bs_n2)
        axes = plotter.get_projected_plots_dots_patom_pmorb({"N": ["s"]}, {"N": [1]}, ylim=(-2, 2))
        assert len(axes) == 1
        ax = axes[0]
        assert ax.get_title() == "N 1 s"
        assert ax.get_ylim() == approx((-2, 2))
        n_lines = len(bs_n2.branches) * bs_n2.nb_bands
        assert len([line for line in ax.get_lines() if line.get_linestyle() == "--"]) == n_lines
        cyan_dots = [line for line in ax.get_lines() if line.get_color() == "c"]
        assert len(cyan_dots) == len(bs_n2.kpoints) * bs_n2.nb_bands

    def test_patom_pmorb_plot_num_column(self, bs_cu2o):
        plotter = BSPlotterProjected(bs_cu2o)
        axes = plotter.get_projected_plots_dots_patom_pmorb(
            {"Cu": ["s", "dxy"]}, {"Cu": [3, 4]}, num_column=3, selected_branches=[1]
        )
        assert len(axes) == 4
        # 4 figures in 3 columns -> 2 rows
        assert axes[3].get_subplotspec().rowspan.start == 1
        assert axes[3].get_subplotspec().colspan.start == 0
        with pytest.raises(ValueError, match="invalid 'num_column'"):
            plotter.get_projected_plots_dots_patom_pmorb({"Cu": ["s"]}, {"Cu": [3]}, num_column="2")

    def test_patom_pmorb_plot_metal(self, bs_c_metal):
        plotter = BSPlotterProjected(bs_c_metal)
        axes = plotter.get_projected_plots_dots_patom_pmorb({"C": ["s"]}, {"C": [1]}, selected_branches=[1])
        assert axes[0].get_ylim() == approx((-10, 10))

    def test_patom_pmorb_plot_metal_absolute_energies(self, bs_c_metal):
        plotter = BSPlotterProjected(bs_c_metal)
        axes = plotter.get_projected_plots_dots_patom_pmorb(
            {"C": ["s"]}, {"C": [1]}, zero_to_efermi=False, selected_branches=[1]
        )
        assert axes[0].get_ylim() == approx((bs_c_metal.efermi - 10, bs_c_metal.efermi + 10))


class TestBSDOSPlotterExtra:
    @pytest.fixture(scope="class")
    def si_run(self):
        vasp_run = Vasprun(f"{VASP_OUT_DIR}/vasprun_Si_bands.xml.gz")
        band_struct = vasp_run.get_band_structure(kpoints_filename=f"{VASP_IN_DIR}/KPOINTS_Si_bands")
        return band_struct, vasp_run.complete_dos

    def test_get_colordata_binary(self, bs_cu2o):
        elements = [el.symbol for el in bs_cu2o.structure.elements]
        assert elements == ["O", "Cu"]
        colordata = BSDOSPlotter._get_colordata(bs_cu2o, elements, "elements")
        rgb = colordata[Spin.up]
        assert rgb.shape == (bs_cu2o.nb_bands, len(bs_cu2o.kpoints), 3)
        projections = bs_cu2o.get_projection_on_elements()[Spin.up]
        band, k_idx = 1, 10
        p_o, p_cu = projections[band][k_idx]["O"], projections[band][k_idx]["Cu"]
        norm = p_o**2 + p_cu**2
        # 2nd element -> red, 1st element -> blue, green unused
        assert rgb[band, k_idx] == approx([math.sqrt(p_cu**2 / norm), 0, math.sqrt(p_o**2 / norm)])
        assert_allclose(rgb[..., 1], 0)
        assert_allclose(rgb[..., 0] ** 2 + rgb[..., 2] ** 2, 1)

    def test_get_colordata_no_projection(self, bs_n2):
        colordata = BSDOSPlotter._get_colordata(bs_n2, ["N"], None)
        assert_allclose(colordata[Spin.up], 0)
        assert_allclose(colordata[Spin.down][..., 2], 1)
        assert_allclose(colordata[Spin.down][..., :2], 0)

    def test_binary_rb_legend(self, bs_cu2o):
        bs_ax = BSDOSPlotter(fixed_cb_energy=True, cb_energy_range=3).get_plot(bs_cu2o)
        assert bs_ax.get_ylim() == approx((-4, 3))
        insets = [ax for ax in bs_ax.figure.axes if ax is not bs_ax]
        assert len(insets) == 1
        assert {text.get_text() for text in insets[0].texts} == {"O", "Cu"}
        # one LineCollection per (band, branch)
        assert len(bs_ax.collections) == bs_cu2o.nb_bands * len(bs_cu2o.branches) + 1  # + Fermi level hlines

    def test_ternary_rgb_legend(self, bs_cu2o):
        structure = bs_cu2o.structure.copy()
        structure[2] = "Ag"
        bs = _truncate_bands(bs_cu2o, [0, 1, 2, 3], structure=structure)
        bs_ax = BSDOSPlotter().get_plot(bs)
        insets = [ax for ax in bs_ax.figure.axes if ax is not bs_ax]
        assert len(insets) == 1
        assert {text.get_text() for text in insets[0].texts} == {"O", "Cu", "Ag"}

    def test_elements_legend_without_rgb(self, bs_cu2o):
        bs_ax = BSDOSPlotter(rgb_legend=False).get_plot(bs_cu2o)
        assert [text.get_text() for text in bs_ax.get_legend().get_texts()] == ["O", "Cu"]

    def test_no_projection_legend(self, bs_n2):
        bs_ax = BSDOSPlotter(bs_projection=None).get_plot(bs_n2)
        assert [text.get_text() for text in bs_ax.get_legend().get_texts()] == ["spin up", "spin down"]
        # spin up and spin down band lines
        assert len(bs_ax.collections) == 2 * bs_n2.nb_bands * len(bs_n2.branches) + 1

    @pytest.mark.parametrize(
        ("dos_projection", "expected_labels"),
        [("orbitals", ["total", "s", "p", "d"]), (None, ["total"]), ("elements", ["total", "Si"])],
    )
    def test_dos_projection(self, si_run, dos_projection, expected_labels):
        band_struct, dos = si_run
        plotter = BSDOSPlotter(dos_projection=dos_projection)
        with pytest.warns(UserWarning, match="Cannot get element projected data"):
            bs_ax, dos_ax = plotter.get_plot(band_struct, dos)
        labels = [text.get_text() for text in dos_ax.get_legend().get_texts()]
        assert labels[: len(expected_labels)] == expected_labels
        assert dos_ax.get_ylim() == bs_ax.get_ylim()
        assert dos_ax.get_xlabel() == "DOS"
        assert_allclose(dos_ax.get_lines()[0].get_xdata(), dos.densities[Spin.up])
        assert_allclose(dos_ax.get_lines()[0].get_ydata(), dos.energies - dos.efermi)
        # element projection fell back to no projection, so the BS legend shows spins
        assert [text.get_text() for text in bs_ax.get_legend().get_texts()] == ["spin up", "spin down"]

    def test_no_legends(self, si_run):
        band_struct, dos = si_run
        plotter = BSDOSPlotter(bs_legend=None, dos_legend=None, bs_projection=None)
        bs_ax, dos_ax = plotter.get_plot(band_struct, dos)
        assert bs_ax.get_legend() is None
        assert dos_ax.get_legend() is None


class FakeMlab:
    """Minimal stand-in for mayavi.mlab that records calls."""

    def __init__(self):
        self.figures = []
        self.calls = {"plot3d": 0, "points3d": 0, "text3d": 0, "show": 0}
        self.contours = []

    def figure(self, **kwargs):
        fig = SimpleNamespace(scene=SimpleNamespace(isometric_view=lambda: None), kwargs=kwargs)
        self.figures.append(fig)
        return fig

    def plot3d(self, *args, **kwargs):
        self.calls["plot3d"] += 1

    def points3d(self, *args, **kwargs):
        self.calls["points3d"] += 1

    def text3d(self, *args, **kwargs):
        self.calls["text3d"] += 1

    def show(self):
        self.calls["show"] += 1

    def contour3d(self, data, contours, **kwargs):
        polydata = SimpleNamespace(points=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]))
        self.contours.append({"data": data, "contours": contours, "polydata": polydata, **kwargs})
        mapper = SimpleNamespace(input=polydata)
        return SimpleNamespace(actor=SimpleNamespace(actors=[SimpleNamespace(mapper=mapper)]))


class TestPlotFermiSurface:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.mlab = FakeMlab()
        monkeypatch.setattr(plotter_module, "mlab", self.mlab)
        self.structure = Structure.from_file(f"{TEST_FILES_DIR}/io/cssr/Si.cssr")
        grid = np.indices((4, 5, 6)).sum(axis=0).astype(float)
        self.data = grid / grid.max()  # energies in [0, 1]
        self.plot = plotter_module.plot_fermi_surface.__wrapped__

    def _n_bz_edges(self):
        bz = self.structure.lattice.reciprocal_lattice.get_wigner_seitz_cell()
        n_edges = 0
        for iface, face in enumerate(bz):
            for line in itertools.combinations(face, 2):
                for jface in range(iface + 1, len(bz)):
                    if any(np.all(line[0] == x) for x in bz[jface]) and any(np.all(line[1] == x) for x in bz[jface]):
                        n_edges += 1
        return n_edges

    def test_requires_mayavi(self):
        if find_spec("mayavi") is not None:
            pytest.skip("mayavi is installed")
        with pytest.raises(RuntimeError, match="MayAvi mlab not installed"):
            plotter_module.plot_fermi_surface(self.data, self.structure, cbm=False)

    def test_energy_level_out_of_range(self):
        with pytest.raises(BoltztrapError, match="not in the range of possible energies"):
            self.plot(self.data, self.structure, cbm=False, energy_levels=[2.0])

    def test_default_level_valence_band(self):
        fig, mlab = self.plot(self.data, self.structure, cbm=False, interactive=False)
        assert mlab is self.mlab
        assert fig is self.mlab.figures[-1]
        assert len(self.mlab.contours) == 1
        assert self.mlab.contours[0]["contours"] == approx([1.0 - 0.01])
        assert_allclose(self.mlab.contours[0]["data"], self.data)
        assert self.mlab.calls["show"] == 0

    def test_default_level_conduction_band(self):
        self.plot(self.data, self.structure, cbm=True, interactive=True)
        # for a conduction band the data are negated
        assert self.mlab.contours[0]["contours"] == approx([-1.0 + 0.01])
        assert_allclose(self.mlab.contours[0]["data"], -self.data)
        assert self.mlab.calls["show"] == 1

    def test_multiple_figures(self):
        levels = [0.3, 0.6]
        self.plot(
            self.data,
            self.structure,
            cbm=False,
            energy_levels=levels,
            kpoints_dict={"\\Gamma": [0, 0, 0], "X": [0.5, 0, 0.5]},
            colors=[(1, 0, 0), (0, 1, 0)],
            transparency_factor=[0.5, 0.8],
            interactive=False,
        )
        assert len(self.mlab.figures) == 2
        assert [contour["contours"] for contour in self.mlab.contours] == [[0.3], [0.6]]
        assert [contour["color"] for contour in self.mlab.contours] == [(1, 0, 0), (0, 1, 0)]
        assert [contour["opacity"] for contour in self.mlab.contours] == [0.5, 0.8]
        assert self.mlab.calls["points3d"] == self.mlab.calls["text3d"] == 4
        assert self.mlab.calls["plot3d"] == 2 * self._n_bz_edges()

        # surface points are mapped to Cartesian reciprocal space and centered (scaled by 2)
        cell = self.structure.lattice.reciprocal_lattice.matrix
        raw = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        cart = raw @ (cell / np.array(self.data.shape)[:, None])
        expected = (cart - cart.mean(axis=0)) * 2
        assert_allclose(self.mlab.contours[0]["polydata"].points, expected)

    def test_single_figure(self):
        self.plot(
            self.data,
            self.structure,
            cbm=False,
            energy_levels=[0.3, 0.6],
            multiple_figure=False,
            kpoints_dict={"X": [0.5, 0, 0.5]},
            interactive=False,
        )
        assert len(self.mlab.figures) == 1
        assert len(self.mlab.contours) == 2
        assert all(contour["figure"] is self.mlab.figures[0] for contour in self.mlab.contours)
        assert self.mlab.calls["plot3d"] == self._n_bz_edges()
        assert self.mlab.calls["points3d"] == 1


class TestBrillouinZoneHelpers:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.structure = Structure.from_file(f"{TEST_FILES_DIR}/io/cssr/Si.cssr")
        self.rec_latt = self.structure.lattice.reciprocal_lattice

    def test_lattice_required(self):
        with pytest.raises(ValueError, match="requires the lattice"):
            plotter_module.plot_path([[0, 0, 0], [0.5, 0, 0]])
        with pytest.raises(ValueError, match="requires the lattice"):
            plotter_module.plot_labels({"X": [0.5, 0, 0]})
        with pytest.raises(ValueError, match="require the lattice"):
            plotter_module.plot_points([[0, 0, 0]])
        with pytest.raises(ValueError, match="require the lattice"):
            plotter_module.plot_points([[0, 0, 0]], coords_are_cartesian=True, fold=True)
        with pytest.raises(ValueError, match="require the lattice"):
            plot_ellipsoid(np.eye(3), [0, 0, 0])

    @pytest.mark.parametrize(
        "frac", [[0.5, 0.5, 0.5], [0.45, 0.45, -0.45], [0.9, 0.1, 0.4], [-0.6, 0.7, 0.2], [0.0, 0.0, 0.0]]
    )
    @pytest.mark.parametrize("cartesian", [False, True])
    def test_fold_point_in_first_bz(self, frac, cartesian):
        point = self.rec_latt.get_cartesian_coords(frac) if cartesian else frac
        folded = fold_point(point, self.rec_latt, coords_are_cartesian=cartesian)
        # folded point is equivalent by a reciprocal lattice vector
        diff_frac = self.rec_latt.get_fractional_coords(folded) - np.array(frac)
        assert_allclose(diff_frac, np.round(diff_frac), atol=1e-8)
        # and is not closer to any other lattice point than to Gamma
        for shift in itertools.product((-1, 0, 1), repeat=3):
            g_vec = self.rec_latt.get_cartesian_coords(shift)
            assert np.linalg.norm(folded) <= np.linalg.norm(folded - g_vec) + 1e-8

    def test_plot_points_fold(self):
        fig, ax = plotter_module.plot_points([[0.9, 0.1, 0.4]], lattice=self.rec_latt, fold=True, color="r")
        assert len(ax.collections) == 1
        offsets = np.array(ax.collections[0]._offsets3d).ravel()
        assert_allclose(offsets, fold_point([0.9, 0.1, 0.4], self.rec_latt), atol=1e-8)
        assert fig is ax.figure

    def test_plot_ellipsoid_arrows(self):
        hessian = np.diag([4.0, 1.0, 0.25])
        fig, ax = plot_ellipsoid(hessian, [0, 0, 0], coords_are_cartesian=True, arrows=True)
        # wireframe + 3 principal-axis arrows
        assert len(ax.collections) == 4
        assert fig is ax.figure

    def test_plot_brillouin_zone_from_kpath(self):
        kpath = HighSymmKpath(self.structure)
        fig = plotter_module.plot_brillouin_zone_from_kpath(kpath, show=False)
        ax = fig.axes[0]
        labels = {f"${label}$" if label.startswith("\\") or "_" in label else label for label in kpath.kpath["kpoints"]}
        assert {text.get_text() for text in ax.texts} == labels
        n_segments = sum(len(path) - 1 for path in kpath.kpath["path"])
        # 3 lattice vectors + WS-cell edges + path segments
        assert len(ax.get_lines()) >= 3 + n_segments
