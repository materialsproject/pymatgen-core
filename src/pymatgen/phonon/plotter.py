"""This module implements plotter for DOS and band structure."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from monty.json import jsanitize

from pymatgen.core import constants as const
from pymatgen.electronic_structure.plotter import BSDOSPlotter, _autoscale_visible, plot_brillouin_zone
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.gruneisen import GruneisenPhononBandStructureSymmLine
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

    from pymatgen.core import Structure
    from pymatgen.phonon.dos import PhononDos
    from pymatgen.phonon.gruneisen import GruneisenParameter
    from pymatgen.util.typing import PathLike

logger = logging.getLogger(__name__)

_COLORS = ("blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive")


class FreqUnits(NamedTuple):
    """Conversion factor from THz to the required units and the label."""

    factor: float
    label: str


def freq_units(units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"]) -> FreqUnits:
    """
    Args:
        units: str, accepted values: thz, ev, mev, ha, cm-1, cm^-1.

    Returns:
        Conversion factor from THz to the required units and the label in the form of a namedtuple.
    """
    dct = {
        "thz": FreqUnits(1, "THz"),
        "ev": FreqUnits(const.value("hertz-electron volt relationship") * const.tera, "eV"),
        "mev": FreqUnits(
            const.value("hertz-electron volt relationship") * const.tera / const.milli,
            "meV",
        ),
        "ha": FreqUnits(const.value("hertz-hartree relationship") * const.tera, "Ha"),
        "cm-1": FreqUnits(
            const.value("hertz-inverse meter relationship") * const.tera * const.centi,
            "cm^{-1}",
        ),
        "cm^-1": FreqUnits(
            const.value("hertz-inverse meter relationship") * const.tera * const.centi,
            "cm^{-1}",
        ),
    }
    try:
        return dct[units.lower().strip()]
    except KeyError:
        raise KeyError(f"Value for units `{units}` unknown\nPossible values are:\n {list(dct)}")


class PhononDosPlotter:
    """Plot phonon DOSs. The interface is very flexible to accommodate the many
    different ways in which people want to view DOS.
    Typical usage is:
        # Initializes plotter with some optional args. Defaults are usually fine
        plotter = PhononDosPlotter().

        # Add DOS with a label
        plotter.add_dos("Total DOS", dos)

        # Alternatively, you can add a dict of DOSes. This is the typical form
        # returned by CompletePhononDos.get_element_dos().
    """

    def __init__(self, stack: bool = False, sigma: float | None = None) -> None:
        """
        Args:
            stack: Whether to plot the DOS as a stacked area graph
            sigma: A float specifying a standard deviation for Gaussian smearing
                the DOS for nicer looking plots. Defaults to None for no smearing.
        """
        # a likely user mistake is to try to pass a DOS as the first argument (similar to PhononBSPlotter) but
        # without the isinstance check, this would not raise an error and just silently cause a blank plot
        if not isinstance(stack, bool):
            raise TypeError(
                "The first argument stack expects a boolean. If you are trying to add a DOS, use the add_dos() method."
            )
        self.stack = stack
        self.sigma = sigma
        self._doses: dict[str, dict[str, np.ndarray]] = {}

    def add_dos(self, label: str, dos: PhononDos, **kwargs: Any) -> None:
        """Add a dos for plotting.

        Args:
            label (str): label for the DOS. Must be unique.
            dos (PhononDos): DOS object
            **kwargs: kwargs supported by matplotlib.pyplot.plot
        """
        densities = dos.get_smeared_densities(self.sigma) if self.sigma else dos.densities
        self._doses[label] = {
            "frequencies": dos.frequencies,
            "densities": densities,
            **kwargs,
        }

    def add_dos_dict(self, dos_dict: dict, key_sort_func=None) -> None:
        """Add a dictionary of doses, with an optional sorting function for the
        keys.

        Args:
            dos_dict: dict of {label: Dos}
            key_sort_func: function used to sort the dos_dict keys.
        """
        keys = sorted(dos_dict, key=key_sort_func) if key_sort_func else list(dos_dict)
        for label in keys:
            self.add_dos(label, dos_dict[label])

    def get_dos_dict(self) -> dict:
        """Get the added doses as a json-serializable dict. Note that if you
        have specified smearing for the DOS plot, the densities returned will
        be the smeared densities, not the original densities.

        Returns:
            dict: DOS data. Generally of the form {label: {'frequencies':.., 'densities': ...}}
        """
        return jsanitize(self._doses)

    def get_plot(
        self,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        invert_axes: bool = False,
        units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz",
        legend: dict | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Get a matplotlib plot showing the DOS.

        Args:
            xlim: Specifies the x-axis limits. Set to None for automatic determination.
            ylim: Specifies the y-axis limits.
            invert_axes (bool): Whether to invert the x and y axes. Enables chemist style DOS plotting.
                Defaults to False.
            units (thz | ev | mev | ha | cm-1 | cm^-1): units for the frequencies. Defaults to "thz".
            legend: dict with legend options. For example, {"loc": "upper right"}
                will place the legend in the upper right corner. Defaults to {"fontsize": 30}.
            ax (Axes): An existing axes object onto which the plot will be added.
                If None, a new figure will be created.
        """
        legend = {"fontsize": 30} | (legend or {})
        unit = freq_units(units)

        n_colors = min(9, max(3, len(self._doses)))

        ax = pretty_plot(*(8, 12) if invert_axes else (12, 8), ax=ax)

        # Stacked plots show the cumulative densities
        all_densities = [dos["densities"] for dos in self._doses.values()]
        if self.stack:
            all_densities = list(np.cumsum(all_densities, axis=0))

        # plot in reverse order so the first DOS ends up on top
        curves = []
        for idx, (key, densities) in enumerate(zip(reversed(self._doses), reversed(all_densities), strict=True)):
            dos = self._doses[key]
            frequencies = dos["frequencies"] * unit.factor
            color = dos.get("color", _COLORS[idx % n_colors])
            linewidth = dos.get("linewidth", 3)
            kwargs = {k: val for k, val in dos.items() if k not in {"frequencies", "densities", "color", "linewidth"}}
            xs, ys = (densities, frequencies) if invert_axes else (frequencies, densities)
            curves.append((xs, ys))
            if self.stack:
                ax.fill(xs, ys, color=color, label=str(key), **kwargs)  # type: ignore[arg-type]
            else:
                ax.plot(xs, ys, color=color, label=str(key), linewidth=linewidth, **kwargs)  # type:ignore[arg-type]

        _autoscale_visible(ax, curves, invert_axes, xlim, ylim)

        freq_label = rf"$\mathrm{{Frequencies\ ({unit.label})}}$"
        dos_label = r"$\mathrm{Density\ of\ states}$"
        if invert_axes:
            ax.axhline(0, linewidth=2, color="black", linestyle="--")
            ax.set_xlabel(dos_label, fontsize=legend["fontsize"])
            ax.set_ylabel(freq_label, fontsize=legend["fontsize"])
        else:
            ax.axvline(0, linewidth=2, color="black", linestyle="--")
            ax.set_xlabel(freq_label, fontsize=legend["fontsize"])
            ax.set_ylabel(dos_label, fontsize=legend["fontsize"])

        # only show legend if there are labels
        if any(str(key) for key in self._doses):
            ax.legend(**legend)

        return ax

    def save_plot(
        self,
        filename: PathLike,
        img_format: str = "eps",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        invert_axes: bool = False,
        units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz",
    ) -> None:
        """Save matplotlib plot to a file.

        Args:
            filename: Filename to write to.
            img_format: Image format to use. Defaults to EPS.
            xlim: Specifies the x-axis limits. Set to None for automatic
                determination.
            ylim: Specifies the y-axis limits.
            invert_axes: Whether to invert the x and y axes. Enables chemist style DOS plotting.
                Defaults to False.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1
        """
        self.get_plot(xlim, ylim, invert_axes=invert_axes, units=units)
        plt.savefig(filename, format=img_format)
        plt.close()

    def show(
        self,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        invert_axes: bool = False,
        units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz",
    ) -> None:
        """Show the plot using matplotlib.

        Args:
            xlim: Specifies the x-axis limits. Set to None for automatic
                determination.
            ylim: Specifies the y-axis limits.
            invert_axes: Whether to invert the x and y axes. Enables chemist style DOS plotting.
                Defaults to False.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
        """
        self.get_plot(xlim, ylim, invert_axes=invert_axes, units=units)
        plt.show()


class PhononBSPlotter:
    """Plot or get data to facilitate the plot of band structure objects."""

    def __init__(self, bs: PhononBandStructureSymmLine, label: str | None = None) -> None:
        """
        Args:
            bs: A PhononBandStructureSymmLine object.
            label: A label for the plot. Defaults to None for no label. Esp. useful with
                the plot_compare method to distinguish the band structures.
        """
        if not isinstance(bs, PhononBandStructureSymmLine):
            raise TypeError(
                "PhononBSPlotter only works with PhononBandStructureSymmLine objects. "
                "A PhononBandStructure object (on a uniform grid for instance and "
                "not along symmetry lines won't work)"
            )
        self._bs = bs
        self._label = label

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return self._bs.nb_bands

    def _make_ticks(self, ax: Axes) -> Axes:
        """Utility private method to add ticks to a band structure."""
        ticks = self.get_ticks()
        if ticks["distance"]:
            ax.set_xticks(ticks["distance"])
            ax.set_xticklabels(ticks["label"])

        # plot vertical lines at each of the ticks
        for distance, label in zip(ticks["distance"], ticks["label"], strict=True):
            if label is not None:
                ax.axvline(distance, color="black")
        return ax

    def _branch_slices(self) -> list[slice]:
        """Q-point index range of each branch."""
        return [slice(branch["start_index"], branch["end_index"] + 1) for branch in self._bs.branches]

    def bs_plot_data(self) -> dict[str, Any]:
        """Get the data nicely formatted for a plot.

        Returns:
            A dict of the following format:
            ticks: A dict with the 'distances' at which there is a qpoint (the
            x axis) and the labels (None if no label)
            frequencies: A list (one element for each branch) of frequencies for
            each qpoint: [branch][qpoint][mode]. The data is
            stored by branch to facilitate the plotting
            lattice: The reciprocal lattice.
        """
        slices = self._branch_slices()
        return {
            "ticks": self.get_ticks(),
            "distances": [self._bs.distance[slc] for slc in slices],
            "frequency": [[list(band) for band in self._bs.bands[:, slc]] for slc in slices],
            "lattice": self._bs.lattice_rec.as_dict(),
        }

    def get_plot(
        self,
        ylim: float | None = None,
        units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz",
        **kwargs,
    ) -> Axes:
        """Get a matplotlib object for the bandstructure plot.

        Args:
            ylim: Specify the y-axis (frequency) limits; by default None let
                the code choose.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
                Defaults to "thz".
            **kwargs: passed to ax.plot function.
        """
        u = freq_units(units)

        ax = pretty_plot(12, 8)

        data = self.bs_plot_data()
        kwargs.setdefault("color", "blue")
        for dists, freqs in zip(data["distances"], data["frequency"], strict=True):
            for band in freqs:
                ax.plot(dists, [freq * u.factor for freq in band], **kwargs)

        self._make_ticks(ax)

        # plot y=0 line
        ax.axhline(0, linewidth=1, color="black")

        # Main X and Y Labels
        ax.set_xlabel(r"$\mathrm{Wave\ Vector}$", fontsize=30)
        ylabel = rf"$\mathrm{{Frequencies\ ({u.label})}}$"
        ax.set_ylabel(ylabel, fontsize=30)

        # X range (K)
        # last distance point
        x_max = data["distances"][-1][-1]
        ax.set_xlim(0, x_max)

        if ylim is not None:
            ax.set_ylim(ylim)

        return ax

    def _get_weights(self, indices: list[list[int]]) -> np.ndarray:
        """Weight of each group of sites in every eigendisplacement, normalized over the groups.

        Returns:
            np.ndarray: shape (n_bands, n_qpoints, n_groups).
        """
        # amplitude of the eigendisplacement of each site, shape (n_bands, n_qpoints, n_sites)
        amplitudes = np.linalg.norm(self._bs.eigendisplacements, axis=-1)
        projectors = np.zeros((len(indices), amplitudes.shape[-1]))
        for group_idx, comb in enumerate(indices):
            projectors[group_idx, comb] = 1
        weights = amplitudes @ projectors.T
        return weights / weights.sum(axis=-1, keepdims=True)

    @staticmethod
    def _make_color(colors: np.ndarray) -> np.ndarray:
        """Convert group weights (last axis, 2 to 4 groups) to rgb colors."""
        colors = np.asarray(colors)
        # if there are two groups, use red and blue
        if colors.shape[-1] == 2:
            return np.stack([colors[..., 0], np.zeros(colors.shape[:-1]), colors[..., 1]], axis=-1)
        if colors.shape[-1] == 3:
            return colors
        # if there are four groups, use cyan, magenta, yellow and black
        if colors.shape[-1] == 4:
            black = 1 - colors[..., 3]
            return np.stack([(1 - colors[..., idx]) * black for idx in range(3)], axis=-1)
        raise ValueError(f"Expected 2, 3 or 4 colors, got {colors.shape[-1]}")

    def get_proj_plot(
        self,
        site_comb: str | list[list[int]] = "element",
        ylim: tuple[float | None, float | None] | None = None,
        units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz",
        rgb_labels: tuple[str | None] | None = None,
    ) -> Axes:
        """Get a matplotlib object for the bandstructure plot projected along atomic
        sites.

        Args:
            site_comb: a list of list, for example, [[0],[1],[2,3,4]];
                the numbers in each sublist represents the indices of atoms;
                the atoms in a same sublist will be plotted in a same color;
                if not specified, unique elements are automatically grouped.
            ylim: Specify the y-axis (frequency) limits; by default None let
                the code choose.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
                Defaults to "thz".
            rgb_labels: a list of rgb colors for the labels; if not specified,
                the colors will be automatically generated.
        """
        if self._bs.structure is None:
            raise ValueError("Structure is required for get_proj_plot")
        elements = [elem.symbol for elem in self._bs.structure.elements]
        if site_comb == "element":
            if len(elements) not in {2, 3, 4}:
                raise ValueError("the compound must have 2, 3 or 4 unique elements")
            indices: list[list[int]] = [[] for _ in range(len(elements))]
            for idx, elem in enumerate(self._bs.structure.species):
                for j, unique_species in enumerate(self._bs.structure.elements):
                    if elem == unique_species:
                        indices[j].append(idx)
        else:
            if not isinstance(site_comb, list):
                raise TypeError("Site_comb should be a list.")
            if len(site_comb) not in {2, 3, 4}:
                raise ValueError("the length of site_comb must be 2, 3 or 4")
            all_sites = self._bs.structure.sites
            all_indices = {*range(len(all_sites))}
            for comb in site_comb:
                for idx in comb:
                    if not 0 <= idx < len(all_sites):
                        raise RuntimeError("one or more indices in site_comb does not exist")
                    all_indices.remove(idx)
            if len(all_indices) != 0:
                raise ValueError(f"not all {len(all_sites)} indices are included in site_comb")
            indices = site_comb  # type: ignore[assignment]
        if rgb_labels is not None and len(rgb_labels) != len(indices):
            raise ValueError("wrong number of rgb_labels")

        u = freq_units(units)
        _fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        self._make_ticks(ax)

        k_dist = np.concatenate(self.bs_plot_data()["distances"])
        weights = self._get_weights(indices)
        # each segment between 2 consecutive q-points takes the mean color of its end points
        seg_colors = self._make_color((weights[:, :-1] + weights[:, 1:]) / 2)
        # segments ordered by q-point, then band: shape (n_qpoints - 1, n_bands, 2 end points, (x, y))
        segments = np.zeros((len(k_dist) - 1, self.n_bands, 2, 2))
        segments[:, :, 0, 0] = k_dist[:-1, None]
        segments[:, :, 1, 0] = k_dist[1:, None]
        freqs = self._bs.bands.T * u.factor
        segments[:, :, 0, 1] = freqs[:-1]
        segments[:, :, 1, 1] = freqs[1:]
        ax.add_collection(
            LineCollection(
                segments.reshape(-1, 2, 2),  # type:ignore[arg-type]
                colors=seg_colors.transpose(1, 0, 2).reshape(-1, 3),
                linestyles="-",
                linewidths=2.5,
            )
        )
        if ylim is None:
            y_max: float = np.max(self._bs.bands) * u.factor
            y_min: float = np.min(self._bs.bands) * u.factor
            y_margin = (y_max - y_min) * 0.05
            ylim = (y_min - y_margin, y_max + y_margin)
        ax.set_ylim(ylim)  # type:ignore[arg-type]
        xlim = (min(k_dist), max(k_dist))
        ax.set_xlim(xlim)
        ax.set_xlabel(r"$\mathrm{Wave\ Vector}$", fontsize=28)
        ylabel = rf"$\mathrm{{Frequencies\ ({u.label})}}$"
        ax.set_ylabel(ylabel, fontsize=28)
        ax.tick_params(labelsize=28)
        # make color legend
        labels: list[str]
        if rgb_labels is not None:
            labels = rgb_labels  # type: ignore[assignment]
        elif site_comb == "element":
            labels = [elem.symbol for elem in self._bs.structure.elements]
        else:
            labels = [f"{idx}" for idx in range(len(site_comb))]
        if len(indices) == 2:
            BSDOSPlotter._rb_line(ax, labels[0], labels[1], "best")
        elif len(indices) == 3:
            BSDOSPlotter._rgb_triangle(ax, labels[0], labels[1], labels[2], "best")
        else:
            # for 4 combinations, build a color square?
            pass
        return ax

    def show(
        self,
        ylim: float | None = None,
        units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz",
    ) -> None:
        """Show the plot using matplotlib.

        Args:
            ylim (float): Specifies the y-axis limits.
            units ("thz" | "ev" | "mev" | "ha" | "cm-1" | "cm^-1"): units for the frequencies.
        """
        self.get_plot(ylim, units=units)
        plt.show()

    def save_plot(
        self,
        filename: PathLike,
        ylim: float | None = None,
        units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz",
    ) -> None:
        """Save matplotlib plot to a file.

        Args:
            filename (str | Path): Filename to write to.
            ylim (float): Specifies the y-axis limits.
            units ("thz" | "ev" | "mev" | "ha" | "cm-1" | "cm^-1"): units for the frequencies.
        """
        self.get_plot(ylim=ylim, units=units)
        plt.savefig(filename)
        plt.close()

    def show_proj(
        self,
        site_comb: str | list[list[int]] = "element",
        ylim: tuple[float | None, float | None] | None = None,
        units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz",
        rgb_labels: tuple[str] | None = None,
    ) -> None:
        """Show the projected plot using matplotlib.

        Args:
            site_comb: A list of list of indices of sites to combine. For example,
                [[0, 1], [2, 3]] will combine the projections of sites 0 and 1,
                and sites 2 and 3. Defaults to "element", which will combine
                sites by element.
            ylim: Specify the y-axis (frequency) limits; by default None let
                the code choose.
            units: units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
                Defaults to "thz".
            rgb_labels: A list of labels for the rgb triangle. Defaults to None,
                which will use the element symbols.
        """
        self.get_proj_plot(site_comb=site_comb, ylim=ylim, units=units, rgb_labels=rgb_labels)
        plt.show()

    def get_ticks(self) -> dict[str, list]:
        """Get all ticks and labels for a band structure plot.

        Returns:
            A dict with 'distance': a list of distance at which ticks should
            be set and 'label': a list of label for each of those ticks.
        """
        tick_distance = []
        tick_labels: list[str] = []
        prev_label = self._bs.qpoints[0].label
        prev_branch = self._bs.branches[0]["name"]
        for idx, point in enumerate(self._bs.qpoints):
            if point.label is not None:
                tick_distance.append(self._bs.distance[idx])
                this_branch = None
                for branch in self._bs.branches:
                    if branch["start_index"] <= idx <= branch["end_index"]:
                        this_branch = branch["name"]
                        break
                if point.label != prev_label and prev_branch != this_branch:
                    label1 = point.label
                    if label1.startswith("\\") or label1.find("_") != -1:
                        label1 = f"${label1}$"
                    label0 = prev_label or ""
                    if label0.startswith("\\") or label0.find("_") != -1:
                        label0 = f"${label0}$"
                    tick_labels.pop()
                    tick_distance.pop()
                    tick_labels.append(f"{label0}|{label1}")
                elif point.label.startswith("\\") or point.label.find("_") != -1:
                    tick_labels.append(f"${point.label}$")
                else:
                    tick_labels.append(point.label)
                prev_label = point.label
                prev_branch = this_branch
        # map atomate2 all-upper-case labels like GAMMA/DELTA to pretty symbols
        tick_labels = [label.replace("GAMMA", "Γ").replace("DELTA", "Δ").replace("SIGMA", "Σ") for label in tick_labels]
        return {"distance": tick_distance, "label": tick_labels}

    def plot_compare(
        self,
        other_plotter: PhononBSPlotter | dict[str, PhononBSPlotter],
        units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz",
        self_label: str = "self",
        colors: Sequence[str] | None = None,
        legend_kwargs: dict | None = None,
        on_incompatible: Literal["raise", "warn", "ignore"] = "raise",
        other_kwargs: dict | None = None,
        **kwargs,
    ) -> Axes | None:
        """Plot two band structure for comparison. self in blue, others in red, green, ...
        The band structures need to be defined on the same symmetry lines!
        The distance between symmetry lines is determined by the band structure used to
        initialize PhononBSPlotter (self).

        Args:
            other_plotter (PhononBSPlotter | dict[str, PhononBSPlotter]): Other PhononBSPlotter object(s) defined along
                the same symmetry lines
            units (str): units for the frequencies. Accepted values thz, ev, mev, ha, cm-1, cm^-1.
                Defaults to 'thz'.
            self_label (str): label for the self band structure. Defaults to to the label passed to PhononBSPlotter.init
                or, if None, 'self'.
            colors (list[str]): list of colors for the other band structures. Defaults to None for automatic colors.
            legend_kwargs: dict[str, Any]: kwargs passed to ax.legend().
            on_incompatible ('raise' | 'warn' | 'ignore'): What to do if the band structures
                are not compatible. Defaults to 'raise'.
            other_kwargs: dict[str, Any]: kwargs passed to other_plotter ax.plot().
            **kwargs: passed to ax.plot().

        Returns:
            plt.Axes: with two band structures.
        """
        unit = freq_units(units)
        legend_kwargs = {"fontsize": 20} | (legend_kwargs or {})
        other_kwargs = other_kwargs or {}
        if isinstance(other_plotter, PhononBSPlotter):
            other_plotter = {other_plotter._label or "other": other_plotter}
        if colors and len(colors) != len(other_plotter) + 1:
            raise ValueError("Wrong number of colors")

        self_data = self.bs_plot_data()
        n_points = [len(dists) for dists in self_data["distances"]]

        line_width = kwargs.setdefault("linewidth", 1)
        ax = self.get_plot(units=units, color=colors[0] if colors else _COLORS[0], **kwargs)

        colors_other = []

        for idx, plotter in enumerate(other_plotter.values()):
            other_data = plotter.bs_plot_data()

            if n_points != [len(dists) for dists in other_data["distances"]]:
                if on_incompatible == "raise":
                    raise ValueError("The two band structures are not compatible.")
                if on_incompatible == "warn":
                    logger.warning("The two band structures are not compatible.")
                return None

            # Only set the color of the other band structures here: setting it in kwargs (used by
            # self.get_plot above) would color every band the same.
            _kwargs = {"color": colors[idx + 1] if colors else _COLORS[1 + idx % len(_COLORS)]} | kwargs
            colors_other.append(_kwargs["color"])

            for band_idx in range(plotter.n_bands):
                for dists, freqs in zip(self_data["distances"], other_data["frequency"], strict=True):
                    ax.plot(dists, [freq * unit.factor for freq in freqs[band_idx]], **(_kwargs | other_kwargs))

        # add legend showing which color corresponds to which band structure
        color_self = ax.lines[0].get_color()
        ax.plot([], [], label=self._label or self_label, linewidth=2 * line_width, color=color_self)
        linestyle = other_kwargs.get("linestyle", "-")
        for color_other, label_other in zip(colors_other, other_plotter, strict=True):
            ax.plot([], [], label=label_other, linewidth=2 * line_width, color=color_other, linestyle=linestyle)
        if other_plotter:
            ax.legend(**legend_kwargs)

        return ax

    def plot_brillouin(self) -> None:
        """Plot the Brillouin zone."""
        q_pts = self._bs.qpoints
        # get labels and lines
        labels = {q_pt.label: q_pt.frac_coords for q_pt in q_pts if q_pt.label}

        lines = [
            [
                q_pts[branch["start_index"]].frac_coords,
                q_pts[branch["end_index"]].frac_coords,
            ]
            for branch in self._bs.branches
        ]

        plot_brillouin_zone(self._bs.lattice_rec, lines=lines, labels=labels)


class ThermoPlotter:
    """Plotter for thermodynamic properties obtained from phonon DOS.
    If the structure corresponding to the DOS, it will be used to extract the formula unit and provide
    the plots in units of mol instead of mole-cell.
    """

    def __init__(self, dos: PhononDos, structure: Structure = None) -> None:
        """
        Args:
            dos: A PhononDos object.
            structure: A Structure object corresponding to the structure used for the calculation.
        """
        self.dos = dos
        self.structure = structure

    def _plot_thermo(
        self,
        func: Callable[[float, Structure | None], float],
        temperatures: ArrayLike,
        factor: float = 1,
        ax: Axes = None,
        ylabel: str | None = None,
        label: str | None = None,
        ylim: float | None = None,
        **kwargs,
    ) -> Figure:
        """Plots a thermodynamic property for a generic function from a PhononDos instance.

        Args:
            func (Callable[[float, Structure | None], float]): Takes a temperature and structure (in that order)
                and returns a thermodynamic property (e.g., heat capacity, entropy, etc.).
            temperatures (list[float]): temperatures (in K) at which to evaluate func.
            factor: a multiplicative factor applied to the thermodynamic property calculated. Used to change
                the units. Defaults to 1.
            ax: matplotlib Axes or None if a new figure should be created.
            ylabel: label for the y axis
            label: label of the plot
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
        ax, fig = get_ax_fig(ax)

        values: list = [func(temp, self.structure) * factor for temp in temperatures]  # type: ignore[arg-type]

        ax.plot(temperatures, values, label=label, **kwargs)

        if ylim:
            ax.set_ylim(ylim)

        ax.set_xlim((np.min(temperatures), np.max(temperatures)))
        _ylim = ax.get_ylim()
        if _ylim[0] < 0 < _ylim[1]:
            ax.plot(ax.get_xlim(), [0, 0], "k-", linewidth=1)

        ax.set_xlabel(r"$T$ (K)")
        if ylabel:
            ax.set_ylabel(ylabel)

        return fig

    @add_fig_kwargs
    def plot_cv(self, tmin: float, tmax: float, ntemp: int, ylim: float | None = None, **kwargs) -> Figure:
        """Plots the constant volume specific heat C_v in a temperature range.

        Args:
            tmin: minimum temperature
            tmax: maximum temperature
            ntemp: number of steps
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
        temperatures = np.linspace(tmin, tmax, ntemp)

        ylabel = "$C_v$ (J/K/mol)" if self.structure else "$C_v$ (J/K/mol-c)"

        return self._plot_thermo(self.dos.cv, temperatures, ylabel=ylabel, ylim=ylim, **kwargs)

    @add_fig_kwargs
    def plot_entropy(self, tmin: float, tmax: float, ntemp: int, ylim: float | None = None, **kwargs) -> Figure:
        """Plots the vibrational entrpy in a temperature range.

        Args:
            tmin: minimum temperature
            tmax: maximum temperature
            ntemp: number of steps
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
        temperatures = np.linspace(tmin, tmax, ntemp)

        ylabel = "$S$ (J/K/mol)" if self.structure else "$S$ (J/K/mol-c)"

        return self._plot_thermo(self.dos.entropy, temperatures, ylabel=ylabel, ylim=ylim, **kwargs)

    @add_fig_kwargs
    def plot_internal_energy(self, tmin: float, tmax: float, ntemp: int, ylim: float | None = None, **kwargs) -> Figure:
        """Plots the vibrational internal energy in a temperature range.

        Args:
            tmin: minimum temperature
            tmax: maximum temperature
            ntemp: number of steps
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
        temperatures = np.linspace(tmin, tmax, ntemp)

        ylabel = "$\\Delta E$ (kJ/mol)" if self.structure else "$\\Delta E$ (kJ/mol-c)"

        return self._plot_thermo(
            self.dos.internal_energy,
            temperatures,
            ylabel=ylabel,
            ylim=ylim,
            factor=1e-3,
            **kwargs,
        )

    @add_fig_kwargs
    def plot_helmholtz_free_energy(
        self, tmin: float, tmax: float, ntemp: int, ylim: float | None = None, **kwargs
    ) -> Figure:
        """Plots the vibrational contribution to the Helmoltz free energy in a temperature range.

        Args:
            tmin: minimum temperature
            tmax: maximum temperature
            ntemp: number of steps
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
        temperatures = np.linspace(tmin, tmax, ntemp)

        ylabel = "$\\Delta F$ (kJ/mol)" if self.structure else "$\\Delta F$ (kJ/mol-c)"

        return self._plot_thermo(
            self.dos.helmholtz_free_energy,
            temperatures,
            ylabel=ylabel,
            ylim=ylim,
            factor=1e-3,
            **kwargs,
        )

    @add_fig_kwargs
    def plot_thermodynamic_properties(
        self, tmin: float, tmax: float, ntemp: int, ylim: float | None = None, **kwargs
    ) -> Figure:
        """Plots all the thermodynamic properties in a temperature range.

        Args:
            tmin: minimum temperature
            tmax: maximum temperature
            ntemp: number of steps
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
        temperatures = np.linspace(tmin, tmax, ntemp)

        mol = "" if self.structure else "-c"

        fig = self._plot_thermo(
            self.dos.cv,
            temperatures,
            ylabel="Thermodynamic properties",
            ylim=ylim,
            label=rf"$C_v$ (J/K/mol{mol})",
            **kwargs,
        )
        self._plot_thermo(
            self.dos.entropy,
            temperatures,
            ylim=ylim,
            ax=fig.axes[0],
            label=rf"$S$ (J/K/mol{mol})",
            **kwargs,
        )
        self._plot_thermo(
            self.dos.internal_energy,
            temperatures,
            ylim=ylim,
            ax=fig.axes[0],
            factor=1e-3,
            label=rf"$\Delta E$ (kJ/mol{mol})",
            **kwargs,
        )
        self._plot_thermo(
            self.dos.helmholtz_free_energy,
            temperatures,
            ylim=ylim,
            ax=fig.axes[0],
            factor=1e-3,
            label=rf"$\Delta F$ (kJ/mol{mol})",
            **kwargs,
        )

        fig.axes[0].legend(loc="best")

        return fig


class GruneisenPlotter:
    """Plot GruneisenParameter."""

    def __init__(self, gruneisen: GruneisenParameter) -> None:
        """Plot information from GruneisenParameter.

        Args:
            gruneisen (GruneisenParameter): containing the data to plot.
        """
        self._gruneisen = gruneisen

    def get_plot(
        self,
        marker: str = "o",
        markersize: float | None = None,
        units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz",
    ) -> Axes:
        """Will produce a plot.

        Args:
            marker: marker for the depiction
            markersize: size of the marker
            units: unit for the plots, accepted units: thz, ev, mev, ha, cm-1, cm^-1.

        Returns:
            plt.Axes: matplotlib axes object
        """
        freq_unit = freq_units(units)

        xs = self._gruneisen.frequencies.flatten() * freq_unit.factor
        ys = self._gruneisen.gruneisen.flatten()

        ax = pretty_plot(12, 8)

        ax.set_xlabel(rf"$\mathrm{{Frequency\ ({freq_unit.label})}}$")
        ax.set_ylabel(r"$\mathrm{Grüneisen\ parameter}$")

        n_points = len(ys) - 1
        for idx, (xi, yi) in enumerate(zip(xs, ys, strict=True)):
            color = (1.0 / n_points * idx, 0, 1.0 / n_points * (n_points - idx))

            ax.plot(xi, yi, marker, color=color, markersize=markersize)

        plt.tight_layout()
        return ax

    def show(self, units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz") -> None:
        """Will show the plot.

        Args:
            units: units for the plot, accepted units: thz, ev, mev, ha, cm-1, cm^-1.
        """
        self.get_plot(units=units)
        plt.show()

    def save_plot(
        self,
        filename: PathLike,
        img_format: str = "pdf",
        units: Literal["thz", "ev", "mev", "ha", "cm-1", "cm^-1"] = "thz",
    ) -> None:
        """Will save the plot to a file.

        Args:
            filename: name of the filename
            img_format: format of the saved plot
            units: accepted units: thz, ev, mev, ha, cm-1, cm^-1.
        """
        self.get_plot(units=units)
        plt.savefig(filename, format=img_format)
        plt.close()


class GruneisenPhononBSPlotter(PhononBSPlotter):
    """Plot or get data to facilitate the plot of band structure objects."""

    def __init__(self, bs: GruneisenPhononBandStructureSymmLine) -> None:
        """
        Args:
            bs: A GruneisenPhononBandStructureSymmLine object.
        """
        if not isinstance(bs, GruneisenPhononBandStructureSymmLine):
            raise TypeError(
                "GruneisenPhononBSPlotter only works with GruneisenPhononBandStructureSymmLine objects. "
                "A GruneisenPhononBandStructure object (on a uniform grid for instance and "
                "not along symmetry lines won't work)"
            )
        super().__init__(bs)

    def bs_plot_data(self) -> dict[str, Any]:
        """Get the data nicely formatted for a plot.

        Returns:
            A dict of the following format:
            ticks: A dict with the 'distances' at which there is a qpoint (the
            x axis) and the labels (None if no label)
            frequencies: A list (one element for each branch) of frequencies for
            each qpoint: [branch][qpoint][mode]. The data is
            stored by branch to facilitate the plotting
            gruneisen: GruneisenPhononBandStructureSymmLine
            lattice: The reciprocal lattice.
        """
        data = super().bs_plot_data()
        gruneisen = np.asarray(self._bs.gruneisen)
        data["gruneisen"] = [[list(band) for band in gruneisen[:, slc]] for slc in self._branch_slices()]
        # keep the documented key order
        return {key: data[key] for key in ("ticks", "distances", "frequency", "gruneisen", "lattice")}

    def get_plot_gs(
        self,
        ylim: float | None = None,
        plot_ph_bs_with_gruneisen: bool = False,
        **kwargs,
    ) -> Axes:
        """Get a matplotlib object for the Gruneisen bandstructure plot.

        Args:
            ylim: Specify the y-axis (gruneisen) limits; by default None let
                the code choose.
            plot_ph_bs_with_gruneisen (bool): Plot phonon band-structure with bands coloured
                as per Gruneisen parameter values on a logarithmic scale
            **kwargs: additional keywords passed to ax.plot().
        """
        # units and cmap must not be passed on to ax.plot
        u = freq_units(kwargs.pop("units", "THz"))
        ax = pretty_plot(12, 8)

        # Create a colormap (default is red to blue)
        cmap = LinearSegmentedColormap.from_list("cmap", kwargs.pop("cmap", ["red", "blue"]))

        kwargs.setdefault("linewidth", 2)
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 2)

        data = self.bs_plot_data()

        # LogNormalize colormap based on the min and max Grüneisen parameter values
        all_gruneisen = np.concatenate([np.ravel(branch) for branch in data["gruneisen"]])
        norm = colors.SymLogNorm(
            vmin=all_gruneisen.min(),
            vmax=all_gruneisen.max(),
            linthresh=1e-2,
            linscale=1,
        )

        sc = None
        for dists, freqs, gruneisen in zip(data["distances"], data["frequency"], data["gruneisen"], strict=True):
            for band_idx in range(self.n_bands):
                if plot_ph_bs_with_gruneisen:
                    ys = [freq * u.factor for freq in freqs[band_idx]]
                    sc = ax.scatter(dists, ys, c=gruneisen[band_idx], cmap=cmap, norm=norm, marker="o", s=1)
                else:
                    ax.plot(dists, gruneisen[band_idx], "b-", **kwargs)

        self._make_ticks(ax)

        # plot y=0 line
        ax.axhline(0, linewidth=1, color="black")

        # Main X and Y Labels
        if plot_ph_bs_with_gruneisen:
            ax.set_xlabel(r"$\mathrm{Wave\ Vector}$", fontsize=30)
            unit_label = f"${u.label}$" if "^" in u.label else u.label
            ax.set_ylabel(f"Frequencies ({unit_label})", fontsize=30)

            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(r"$\gamma \ \mathrm{(logarithmized)}$", fontsize=30)
        else:
            ax.set_xlabel(r"$\mathrm{Wave\ Vector}$", fontsize=30)
            ax.set_ylabel(r"$\mathrm{Grüneisen\ Parameter}$", fontsize=30)

        # X range (K)
        # last distance point
        x_max = data["distances"][-1][-1]
        ax.set_xlim(0, x_max)

        if ylim is not None:
            ax.set_ylim(ylim)

        plt.tight_layout()

        return ax

    def show_gs(
        self,
        ylim: float | None = None,
        plot_ph_bs_with_gruneisen: bool = False,
        **kwargs,
    ) -> None:
        """Show the plot using matplotlib.

        Args:
            ylim: Specifies the y-axis limits.
            plot_ph_bs_with_gruneisen: Plot phonon band-structure with bands coloured
                as per Gruneisen parameter values on a logarithmic scale
            **kwargs: kwargs passed to get_plot_gs
        """
        self.get_plot_gs(ylim=ylim, plot_ph_bs_with_gruneisen=plot_ph_bs_with_gruneisen, **kwargs)
        plt.show()

    def save_plot_gs(
        self,
        filename: PathLike,
        img_format: str = "eps",
        ylim: float | None = None,
        plot_ph_bs_with_gruneisen: bool = False,
        **kwargs,
    ) -> None:
        """Save matplotlib plot to a file.

        Args:
            filename: Filename to write to.
            img_format: Image format to use. Defaults to EPS.
            ylim: Specifies the y-axis limits.
            plot_ph_bs_with_gruneisen: Plot phonon band-structure with bands coloured
                as per Gruneisen parameter values on a logarithmic scale
            **kwargs: kwargs passed to get_plot_gs
        """
        self.get_plot_gs(ylim=ylim, plot_ph_bs_with_gruneisen=plot_ph_bs_with_gruneisen, **kwargs)
        plt.savefig(filename, format=img_format)
        plt.close()

    def plot_compare_gs(self, other_plotter: GruneisenPhononBSPlotter) -> Axes:
        """Plot two band structure for comparison. One is in red the other in blue.
        The two band structures need to be defined on the same symmetry lines!
        and the distance between symmetry lines is
        the one of the band structure used to build the PhononBSPlotter.

        Args:
            other_plotter (GruneisenPhononBSPlotter): another phonon DOS plotter defined along
                the same symmetry lines.

        Raises:
            ValueError: if the two plotters are incompatible (due to different data lengths)

        Returns:
            plt.Axes: with both band structures
        """
        data_orig = self.bs_plot_data()
        data = other_plotter.bs_plot_data()

        len_orig = len(data_orig["distances"])
        len_other = len(data["distances"])
        if len_orig != len_other:
            raise ValueError(
                f"The two plotters are incompatible, plotting data have different lengths ({len_orig} vs {len_other})."
            )

        ax = self.get_plot_gs()
        band_linewidth = 1
        for band_idx in range(other_plotter.n_bands):
            for dists, gruneisen in zip(data_orig["distances"], data["gruneisen"], strict=True):
                ax.plot(
                    dists,
                    [gruneisen[band_idx][j] for j in range(len(dists))],
                    "r-",
                    linewidth=band_linewidth,
                )

        return ax
