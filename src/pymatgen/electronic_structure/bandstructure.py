"""This module provides classes to define things related to band structures."""

from __future__ import annotations

import functools
import itertools
import math
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, overload

import numpy as np
from monty.json import MSONable

from pymatgen.core import Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, Self

    from numpy.typing import ArrayLike, NDArray

    from pymatgen.util.typing import SpeciesLike

__author__ = "Geoffroy Hautier, Shyue Ping Ong, Michael Kocher"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Geoffroy Hautier"
__email__ = "geoffroy@uclouvain.be"
__status__ = "Development"
__date__ = "March 14, 2012"


class Kpoint(MSONable):
    """A kpoint defined with a lattice and frac or Cartesian coordinates,
    similar to the Site object in pymatgen.core.structure.
    """

    def __init__(
        self,
        coords: ArrayLike,
        lattice: Lattice,
        to_unit_cell: bool = False,
        coords_are_cartesian: bool = False,
        label: str | None = None,
    ) -> None:
        """Initialize a Kpoint.

        Args:
            coords (NDArray): Coordinate of the Kpoint.
            lattice (Lattice): The reciprocal lattice of the kpoint.
            to_unit_cell (bool): Translate fractional coordinate to the basic unit
                cell, i.e., all fractional coordinates satisfy 0 <= a < 1.
                Defaults to False.
            coords_are_cartesian (bool): Whether the coordinates given are
                in Cartesian (True) or fractional coordinates (by default fractional).
            label (str): The label of the Kpoint if any (None by default).
        """
        self._lattice = lattice
        # Copy so that to_unit_cell (and later caller-side mutation) never touches the input array
        self._frac_coords = (
            lattice.get_fractional_coords(coords) if coords_are_cartesian else np.array(coords, dtype=float)
        )
        self._label = label

        if to_unit_cell:
            for idx, fc in enumerate(self._frac_coords):
                self._frac_coords[idx] -= math.floor(fc)

        self._cart_coords = lattice.get_cartesian_coords(self._frac_coords)

    def __str__(self) -> str:
        """String with fractional, Cartesian coordinates and label."""
        return f"{self.frac_coords} {self.cart_coords} {self.label}"

    def __eq__(self, other: object) -> bool:
        """Whether two Kpoints are equal."""
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            np.allclose(self.frac_coords, other.frac_coords)
            and self.lattice == other.lattice
            and self.label == other.label
        )

    @property
    def lattice(self) -> Lattice:
        """The lattice associated with the kpoint, as a Lattice object."""
        return self._lattice

    @property
    def label(self) -> str | None:
        """The label associated with the kpoint."""
        return self._label

    @label.setter
    def label(self, label: str | None) -> None:
        """Set the label of the kpoint."""
        self._label = label

    @property
    def frac_coords(self) -> NDArray:
        """The fractional coordinates of the kpoint as a NumPy array."""
        return np.copy(self._frac_coords)

    @property
    def cart_coords(self) -> NDArray:
        """The Cartesian coordinates of the kpoint as a NumPy array."""
        return np.copy(self._cart_coords)

    @property
    def a(self) -> float:
        """Fractional a coordinate of the kpoint."""
        return self._frac_coords[0]

    @property
    def b(self) -> float:
        """Fractional b coordinate of the kpoint."""
        return self._frac_coords[1]

    @property
    def c(self) -> float:
        """Fractional c coordinate of the kpoint."""
        return self._frac_coords[2]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable dict representation of the kpoint."""
        return {
            "lattice": self.lattice.as_dict(),
            "fcoords": self.frac_coords.tolist(),
            "ccoords": self.cart_coords.tolist(),
            "label": self.label,
            "@module": type(self).__module__,
            "@class": type(self).__name__,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Create from a dict.

        Args:
            dct (dict): A dict with all data for a kpoint object.

        Returns:
            Kpoint
        """
        lattice = Lattice.from_dict(dct["lattice"])
        return cls(
            coords=dct["fcoords"],
            lattice=lattice,
            coords_are_cartesian=False,
            label=dct["label"],
        )


class BandStructure:
    """Generic band structure data, defined by a list of Kpoints
        and corresponding energies for each of them.

    Attributes:
        kpoints (list[Kpoint]): Kpoints in the band structure.
        lattice_rec (Lattice): The reciprocal lattice of the band structure.
        efermi (float): The Fermi level.
        is_spin_polarized (bool): Whether the band structure is spin-polarized.
        bands (dict[Spin, NDArray]): The energy eigenvalues. Note that the use of an
            array is necessary for computational and memory efficiency due to the large
            amount of numerical data. The indices of the array are (band_index, kpoint_index).
        nb_bands (int): The number of bands in the band structure.
        structure (Structure): The structure.
        projections (dict[Spin, NDArray]): The projections. Note that the use of an
            array is necessary for computational and memory efficiency due to the large
            amount of numerical data. The indices of the array are (band_index, kpoint_index,
            orbital_index, ion_index).
    """

    def __init__(
        self,
        kpoints: ArrayLike,
        eigenvals: Mapping[Spin, ArrayLike],
        lattice: Lattice,
        efermi: float,
        labels_dict: Mapping[str, Kpoint] | None = None,
        coords_are_cartesian: bool = False,
        structure: Structure | None = None,
        projections: Mapping[Spin, NDArray] | None = None,
    ) -> None:
        """Initialize a BandStructure.

        Args:
            kpoints (NDArray): Kpoint as NumPy array, in frac_coords of the
                given lattice by default.
            eigenvals (dict): Energies for spin up and spin down as
                {Spin.up:[][], Spin.down:[][]}, the first index of the array
                [][] refers to the band and the second to the index of the kpoint.
                The kpoints are ordered according to the kpoints array.
                If the band structure is not spin polarized, we
                only store one data set under Spin.up.
            lattice (Lattice): The reciprocal lattice. Pymatgen uses the physics
                convention of reciprocal lattice vectors with a 2*pi coefficient.
            efermi (float): The Fermi level.
            labels_dict (dict[str, Kpoint]): Dict mapping label to Kpoint.
            coords_are_cartesian (bool): Whether coordinates are cartesian.
            structure (Structure): The crystal structure
                associated with the band structure. This is needed if we
                provide projections to the band structure.
            projections (dict[Spin, NDArray]): Orbital projections. The
                indices of the array are (band_index, kpoint_index, orbital_index,
                ion_index). If the band structure is not spin polarized, we only
                store one data set under Spin.up.
        """
        self.efermi = efermi
        self.lattice_rec = lattice
        self.kpoints = []
        self.labels_dict = {}
        self.structure = structure
        self.projections = projections or {}
        self.projections = {k: np.array(v) for k, v in self.projections.items()}

        if labels_dict is None:
            labels_dict = {}

        if self.projections and self.structure is None:
            raise RuntimeError("if projections are provided a structure object is also required")

        # Match every kpoint against every label at once: is_labeled[kpt_idx, label_idx]
        kpts = np.asarray(kpoints, dtype=float)
        label_names = list(labels_dict)
        if label_names and len(kpts) > 0:
            label_coords = np.array([labels_dict[c] for c in label_names], dtype=float)
            is_labeled = np.linalg.norm(kpts[:, None, :] - label_coords[None, :, :], axis=-1) < 0.0001
        else:
            is_labeled = np.zeros((len(kpts), 0), dtype=bool)

        for kpt, matches in zip(kpts, is_labeled, strict=True):
            # If several labels match, the last one is used for the kpoint
            label = None
            for c in itertools.compress(label_names, matches):
                label = c
                self.labels_dict[label] = Kpoint(kpt, lattice, label=label, coords_are_cartesian=coords_are_cartesian)
            self.kpoints.append(Kpoint(kpt, lattice, label=label, coords_are_cartesian=coords_are_cartesian))
        self.bands = {spin: np.array(v) for spin, v in eigenvals.items()}
        self.nb_bands = len(self.bands[Spin.up])
        self.is_spin_polarized = len(self.bands) == 2

    def get_projection_on_elements(self) -> dict[Spin, list[list[dict[str, float]]]]:
        """Get projections on elements.

        Returns:
            dict[Spin, NDArray]: Dict in {Spin.up:[][{Element: [values]}],
                Spin.down: [][{Element: [values]}]} format.
                If there is no projections in the band structure, return {}.
        """
        if self.structure is None:
            raise ValueError("structure is None.")
        site_indices: dict[str, list[int]] = defaultdict(list)
        for idx, site in enumerate(self.structure):
            site_indices[str(site.specie)].append(idx)

        n_bands, n_kpts = self.nb_bands, len(self.kpoints)
        result: dict[Spin, list[list[dict[str, float]]]] = {}
        for spin, val in self.projections.items():
            result[spin] = [[defaultdict(float) for _ in range(n_kpts)] for _ in range(n_bands)]
            # Sum over orbitals, then over the sites of each species
            orb_summed = val[:n_bands, :n_kpts].sum(axis=2)
            for el, sites in site_indices.items():
                proj = orb_summed[:, :, sites].sum(axis=2)
                for band_res, band_proj in zip(result[spin], proj, strict=True):
                    for kpt_res, value in zip(band_res, band_proj, strict=True):
                        kpt_res[el] = value
        return result

    def get_projections_on_elements_and_orbitals(self, el_orb_spec: dict[str, list[str]]):
        """Get projections on elements and specific orbitals.

        Args:
            el_orb_spec (dict[str, list[str]]): Elements and orbitals to project onto.
                Format is {Element: [orbitals]}, e.g. {"Cu": ["d", "s"]}.

        Returns:
            dict[str, list[str]: Projections on elements in the
                {Spin.up: [][{Element: {orb: values}}],
                Spin.down: [][{Element: {orb: values}}]} format.
                If there is no projections in the band structure, return {}.
        """
        if self.structure is None:
            raise ValueError("Structure is required for this method")
        result: dict[Spin, list] = {}
        species_orb_spec = {get_el_sp(el): orbs for el, orbs in el_orb_spec.items()}
        # Indices of the sites of each requested species in the structure
        site_indices: dict[SpeciesLike, list[int]] = defaultdict(list)
        for idx, site in enumerate(self.structure):
            if site.specie in species_orb_spec:
                site_indices[site.specie].append(idx)

        for spin, v in self.projections.items():
            n_bands, n_kpts = self.nb_bands, len(self.kpoints)
            result[spin] = [
                [{str(e): defaultdict(float) for e in species_orb_spec} for _ in range(n_kpts)] for _ in range(n_bands)
            ]
            # Indices of the orbitals sharing each orbital-type letter (s, p, d, f)
            orb_indices: dict[str, list[int]] = defaultdict(list)
            for orb_i in range(v.shape[2]):
                orb_indices[Orbital(orb_i).name[0]].append(orb_i)

            for sp, sites in site_indices.items():
                key = str(sp)
                for orb, orbs in orb_indices.items():
                    if orb not in species_orb_spec[sp]:
                        continue
                    proj = v[:n_bands, :n_kpts][:, :, orbs][:, :, :, sites].sum(axis=(2, 3))
                    for band_res, band_proj in zip(result[spin], proj, strict=True):
                        for kpt_res, val in zip(band_res, band_proj, strict=True):
                            kpt_res[key][orb] = val
        return result

    def is_metal(self, efermi_tol: float = 1e-4) -> bool:
        """Check if the band structure indicates a metal,
        by looking at if the fermi level crosses a band.

        Returns:
            bool: True if is metal.
        """
        for vals in self.bands.values():
            diff = vals[: self.nb_bands] - self.efermi
            if np.any(np.any(diff < -efermi_tol, axis=1) & np.any(diff > efermi_tol, axis=1)):
                return True
        return False

    def _get_band_edge(self, valence: bool) -> dict[str, Any]:
        """Get data about the VBM (valence=True) or CBM (valence=False).

        See get_vbm and get_cbm for the format of the returned dict.
        """
        if self.is_metal():
            return {
                "band_index": [],
                "kpoint_index": [],
                "kpoint": [],
                "energy": None,
                "projections": {},
            }

        # Highest eigenvalue below (VBM) or lowest at/above (CBM) the Fermi level.
        # Ties are resolved in favour of the first spin channel, then the lowest (band, kpoint) index.
        energy = -float("inf") if valence else float("inf")
        kpt_idx = kpoint = None
        for value in self.bands.values():
            if value.size == 0:
                continue
            if valence:
                candidates = np.where(value < self.efermi, value, -np.inf)
                flat_idx = np.argmax(candidates)
            else:
                candidates = np.where(value >= self.efermi, value, np.inf)
                flat_idx = np.argmin(candidates)
            extremum = candidates.flat[flat_idx]
            if (extremum > energy) if valence else (extremum < energy):
                energy = float(extremum)
                kpt_idx = np.unravel_index(flat_idx, value.shape)[1]
                kpoint = self.kpoints[kpt_idx]

        list_ind_kpts = []
        if kpoint is not None and kpoint.label is not None:
            list_ind_kpts = [idx for idx, kpt in enumerate(self.kpoints) if kpt.label == kpoint.label]
        else:
            list_ind_kpts.append(kpt_idx)  # type:ignore[arg-type]

        # Get all other bands sharing the band edge
        list_ind_band = defaultdict(list)
        for spin, bands in self.bands.items():
            for idx in range(self.nb_bands):
                if math.isclose(bands[idx][kpt_idx], energy, abs_tol=1e-3, rel_tol=0):
                    list_ind_band[spin].append(idx)
        proj = {}
        for spin, value in self.projections.items():
            if len(list_ind_band[spin]) == 0:
                continue
            proj[spin] = value[list_ind_band[spin][0]][list_ind_kpts[0]]

        return {
            "band_index": list_ind_band,
            "kpoint_index": list_ind_kpts,
            "kpoint": kpoint,
            "energy": energy,
            "projections": proj,
        }

    def get_vbm(self) -> dict[str, Any]:
        """Get data about the valence band maximum (VBM).

        Returns:
            dict with keys "band_index", "kpoint_index", "kpoint", "energy":
                - "band_index" (dict): A dict with spin keys pointing to a list of the
                indices of the band containing the VBM (please note that you
                can have several bands sharing the VBM) {Spin.up:[],
                Spin.down:[]}.
                - "kpoint_index": The list of indices in self.kpoints for the
                kpoint VBM. Please note that there can be several
                kpoint_indices relating to the same kpoint (e.g., Gamma can
                occur at different spots in the band structure line plot).
                - "kpoint" (Kpoint): The kpoint.
                - "energy" (float): The energy of the VBM.
                - "projections": The projections along sites and orbitals of the
                VBM if any projection data is available (else it is an empty
                dictionary). The format is similar to the projections field in
                BandStructure: {spin:{'Orbital': [proj]}} where the array
                [proj] is ordered according to the sites in structure.
        """
        return self._get_band_edge(valence=True)

    def get_cbm(self) -> dict[str, Any]:
        """Get data about the conduction band minimum (CBM).

        Returns:
            dict with keys "band_index", "kpoint_index", "kpoint", "energy":
                - "band_index" (dict): A dict with spin keys pointing to a list of the
                indices of the band containing the CBM (please note that you
                can have several bands sharing the CBM) {Spin.up:[], Spin.down:[]}.
                - "kpoint_index": The list of indices in self.kpoints for the
                kpoint CBM. Please note that there can be several
                kpoint_indices relating to the same kpoint (e.g., Gamma can
                occur at different spots in the band structure line plot).
                - "kpoint" (Kpoint): The kpoint.
                - "energy" (float): The energy of the CBM.
                - "projections": The projections along sites and orbitals of the
                CBM if any projection data is available (else it is an empty
                dictionary). The format is similar to the projections field in
                BandStructure: {spin:{'Orbital': [proj]}} where the array
                [proj] is ordered according to the sites in structure.
        """
        return self._get_band_edge(valence=False)

    def get_band_gap(self) -> dict[str, Any]:
        r"""Get band gap.

        Returns:
            dict with keys "energy", "direct", "transition":
                "energy" (float): Band gap energy.
                "direct" (bool): Whether the gap is direct.
                "transition" (str): Kpoint labels of the transition (e.g., "\\Gamma-X").
        """
        if self.is_metal():
            return {"energy": 0.0, "direct": False, "transition": None}

        cbm = self.get_cbm()
        vbm = self.get_vbm()
        result = {
            "direct": False,
            "transition": None,
            "energy": cbm["energy"] - vbm["energy"],
        }

        if (cbm["kpoint"].label is not None and cbm["kpoint"].label == vbm["kpoint"].label) or np.linalg.norm(
            cbm["kpoint"].cart_coords - vbm["kpoint"].cart_coords
        ) < 0.01:
            result["direct"] = True

        result["transition"] = "-".join(
            [
                (str(c.label) if c.label is not None else f"({','.join(f'{c.frac_coords[i]:.3f}' for i in range(3))})")
                for c in [vbm["kpoint"], cbm["kpoint"]]
            ]
        )

        return result

    def get_direct_band_gap_dict(self) -> dict[Spin, dict[str, Any]]:
        """Get information about the direct band gap.

        Spin channels without both fully occupied and fully unoccupied bands
        (e.g. a fully spin-polarized system whose minority channel has no
        occupied bands) cannot define a direct gap and are omitted from the result.

        Returns:
            dict[Spin, dict[str, Any]]: The band gaps indexed by spin
                along with their band indices and kpoint index.

        Raises:
            ValueError: If the band structure is metallic, or if no spin
                channel has both occupied and unoccupied bands.
        """
        if self.is_metal():
            raise ValueError("get_direct_band_gap_dict should only be used with non-metals")

        direct_gap_dict = {}
        for spin, v in self.bands.items():
            above = v[np.all(v > self.efermi, axis=1)]
            below = v[np.all(v < self.efermi, axis=1)]
            if len(above) == 0 or len(below) == 0:
                continue
            min_above = np.min(above, axis=0)
            max_below = np.max(below, axis=0)
            diff = min_above - max_below
            kpoint_index = np.argmin(diff)
            band_indices = [
                np.argmax(below[:, kpoint_index]),
                np.argmin(above[:, kpoint_index]) + len(below),
            ]
            direct_gap_dict[spin] = {
                "value": diff[kpoint_index],
                "kpoint_index": kpoint_index,
                "band_indices": band_indices,
            }
        if not direct_gap_dict:
            raise ValueError("No spin channel has both occupied and unoccupied bands; direct gap is undefined")
        return direct_gap_dict

    def get_direct_band_gap(self) -> float:
        """Get the direct band gap.

        Returns:
            float: The direct band gap value.
        """
        if self.is_metal():
            return 0.0

        dg = self.get_direct_band_gap_dict()
        return min(v["value"] for v in dg.values())

    def get_sym_eq_kpoints(
        self,
        kpoint: NDArray,
        cartesian: bool = False,
        tol: float = 1e-2,
    ) -> NDArray | None:
        """Get unique symmetrically equivalent Kpoints.

        Args:
            kpoint (1x3 array): Coordinate of the Kpoint.
            cartesian (bool): Whether kpoint is in Cartesian or fractional coordinates.
            tol (float): Tolerance below which coordinates are considered equal.

        Returns:
            (1x3 NDArray) | None: None if structure is not available.
        """
        if not self.structure:
            return None

        sg = SpacegroupAnalyzer(self.structure)
        symm_ops = sg.get_point_group_operations(cartesian=cartesian)
        points = np.dot(kpoint, [m.rotation_matrix for m in symm_ops])
        rm_list = []
        # Identify and remove duplicates from equivalent k-points
        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                if np.allclose(pbc_diff(points[i], points[j]), [0, 0, 0], atol=tol):
                    rm_list.append(i)
                    break
        return np.delete(points, rm_list, axis=0)

    def get_kpoint_degeneracy(
        self,
        kpoint: NDArray,
        cartesian: bool = False,
        tol: float = 1e-2,
    ) -> int | None:
        """Get degeneracy of a given kpoint based on structure symmetry.

        Args:
            kpoint (1x3 NDArray): Coordinate of the k-point.
            cartesian (bool): Whether kpoint is in Cartesian or fractional coordinates.
            tol (float): Tolerance below which coordinates are considered equal.

        Returns:
            int | None: Degeneracy, or None if structure is not available.
        """
        all_kpts = self.get_sym_eq_kpoints(kpoint, cartesian, tol=tol)
        return len(all_kpts) if all_kpts is not None else None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable dict representation of BandStructure."""
        dct: dict[str, Any] = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "lattice_rec": self.lattice_rec.as_dict(),
            "efermi": self.efermi,
            "kpoints": [],
        }
        # kpoints are not kpoint objects dicts but are frac coords (this makes
        # the dict smaller and avoids the repetition of the lattice).
        for k in self.kpoints:
            dct["kpoints"].append(k.as_dict()["fcoords"])

        dct["bands"] = {str(int(spin)): self.bands[spin].tolist() for spin in self.bands}
        dct["is_metal"] = self.is_metal()
        vbm = self.get_vbm()
        dct["vbm"] = {
            "energy": vbm["energy"],
            "kpoint_index": vbm["kpoint_index"],
            "band_index": {str(int(spin)): vbm["band_index"][spin] for spin in vbm["band_index"]},
            "projections": {str(spin): v.tolist() for spin, v in vbm["projections"].items()},
        }
        cbm = self.get_cbm()
        dct["cbm"] = {
            "energy": cbm["energy"],
            "kpoint_index": cbm["kpoint_index"],
            "band_index": {str(int(spin)): cbm["band_index"][spin] for spin in cbm["band_index"]},
            "projections": {str(spin): v.tolist() for spin, v in cbm["projections"].items()},
        }
        dct["band_gap"] = self.get_band_gap()
        dct["labels_dict"] = {}
        dct["is_spin_polarized"] = self.is_spin_polarized

        # MongoDB does not accept keys starting with "$", add a space to fix this.
        for c, label in self.labels_dict.items():
            mongo_key = f" {c}" if c.startswith("$") else c
            dct["labels_dict"][mongo_key] = label.as_dict()["fcoords"]
        dct["projections"] = {}
        if len(self.projections) != 0 and self.structure is not None:
            dct["structure"] = self.structure.as_dict()
            dct["projections"] = {str(int(spin)): np.array(v).tolist() for spin, v in self.projections.items()}
        return dct

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """Create from a dict.

        Args:
            dct: A dict with all data for a BandStructure.

        Returns:
            BandStructure
        """
        # Strip the label to recover initial string
        # (see trick used in as_dict to handle "$"" chars)
        labels_dict = {k.strip(): v for k, v in dct["labels_dict"].items()}

        if isinstance(next(iter(dct["bands"].values())), dict):
            eigenvals = {Spin(int(k)): np.array(dct["bands"][k]["data"]) for k in dct["bands"]}
        else:
            eigenvals = {Spin(int(k)): dct["bands"][k] for k in dct["bands"]}

        structure = None
        if "structure" in dct:
            structure = Structure.from_dict(dct["structure"])

        projections = {}
        try:
            if dct.get("projections"):
                if isinstance(dct["projections"]["1"][0][0], dict):
                    raise ValueError("Old band structure dict format detected!")
                projections = {Spin(int(spin)): np.array(v) for spin, v in dct["projections"].items()}

            return cls(
                dct["kpoints"],
                eigenvals,
                Lattice(dct["lattice_rec"]["matrix"]),
                dct["efermi"],
                labels_dict,
                structure=structure,
                projections=projections,
            )

        except Exception:
            warnings.warn(
                "Trying from_dict failed. Now we are trying the old "
                "format. Please convert your BS dicts to the new "
                "format. The old format will be retired in pymatgen "
                "5.0.",
                stacklevel=2,
            )
            return cls.from_old_dict(dct)

    @classmethod
    def from_old_dict(cls, dct: dict[str, Any]) -> Self:
        """Reconstruct a BandStructure from a legacy dict representation.

        Args:
            dct (dict): A dict with all data for a BandStructure object.

        Returns:
            BandStructure
        """
        # Strip the label to recover initial string
        # (see trick used in as_dict to handle "$" chars)
        labels_dict = {k.strip(): v for k, v in dct["labels_dict"].items()}
        projections: dict = {}
        structure = None
        warned_legacy_dx2 = False
        if dct.get("projections"):
            structure = Structure.from_dict(dct["structure"])
            projections = {}
            for spin in dct["projections"]:
                dd = []
                for ii in range(len(dct["projections"][spin])):
                    ddd = []
                    for jj in range(len(dct["projections"][spin][ii])):
                        dddd = []
                        for kk in range(len(dct["projections"][spin][ii][jj])):
                            orb = Orbital(kk).name
                            projection = dct["projections"][spin][ii][jj]
                            if orb == "dx2_y2" and "dx2_y2" not in projection and "dx2" in projection:
                                if not warned_legacy_dx2:
                                    warnings.warn(
                                        "The 'dx2' orbital name in serialized band structures is deprecated; "
                                        "use 'dx2_y2' instead. It will be removed on 2027-08-17.",
                                        DeprecationWarning,
                                        stacklevel=2,
                                    )
                                    warned_legacy_dx2 = True
                                orb = "dx2"
                            ddddd = [projection[orb][ll] for ll in range(len(projection[orb]))]

                            dddd.append(np.array(ddddd))
                        ddd.append(np.array(dddd))
                    dd.append(np.array(ddd))
                projections[Spin(int(spin))] = np.array(dd)

        return cls(
            dct["kpoints"],
            {Spin(int(k)): dct["bands"][k] for k in dct["bands"]},
            Lattice(dct["lattice_rec"]["matrix"]),
            dct["efermi"],
            labels_dict,
            structure=structure,
            projections=projections,
        )


class BandStructureSymmLine(BandStructure, MSONable):
    r"""Store band structures along selected (symmetry) lines in the Brillouin zone.
    We call the different symmetry lines (ex: \\Gamma to Z) "branches".
    """

    def __init__(
        self,
        kpoints: ArrayLike,
        eigenvals: Mapping[Spin, ArrayLike],
        lattice: Lattice,
        efermi: float,
        labels_dict: Mapping[str, Kpoint],
        coords_are_cartesian: bool = False,
        structure: Structure | None = None,
        projections: Mapping[Spin, NDArray] | None = None,
    ) -> None:
        """Initialize a BandStructureSymmLine.

        Args:
            kpoints (NDArray): Array of kpoint, in frac_coords of the
                given lattice by default
            eigenvals (dict[Spin, list]): Energies for spin up and spin down
                {Spin.up:[][],Spin.down:[][]}, the first index of the array
                [][] refers to the band and the second to the index of the
                kpoint. The kpoints are ordered according to the order of the
                kpoints array. If the band structure is not spin polarized, we
                only store one data set under Spin.up.
            lattice (Lattice): The reciprocal lattice. Pymatgen uses the physics
                convention of reciprocal lattice vectors with a 2*pi coefficient.
            efermi (float): The Fermi level.
            labels_dict (dict[str, Kpoint]): Dict mapping label to Kpoint.
            coords_are_cartesian (bool): Whether coordinates are cartesian.
            structure (Structure): The crystal structure associated with the
                band structure. This is needed if we provide projections to
                the band structure.
            projections (dict[Spin, NDArray]): Orbital projections as {spin: array}.
                The indices of the array are [band_index, kpoint_index, orbital_index,
                ion_index].If the band structure is not spin polarized, we only
                store one data set under Spin.up.
        """
        super().__init__(
            kpoints,
            eigenvals,
            lattice,
            efermi,
            labels_dict,
            coords_are_cartesian,
            structure,
            projections,
        )
        self.distance = []
        self.branches = []
        one_group: list = []
        branches_tmp = []
        # Get labels and distance for each kpoint
        previous_kpoint = self.kpoints[0]
        previous_distance = 0.0

        previous_label = self.kpoints[0].label
        for i, kpt in enumerate(self.kpoints):
            label = kpt.label
            if label is not None and previous_label is not None:
                self.distance.append(previous_distance)
            else:
                self.distance.append(
                    np.linalg.norm(kpt.cart_coords - previous_kpoint.cart_coords)  # type: ignore[arg-type]
                    + previous_distance
                )
            previous_kpoint = kpt
            previous_distance = self.distance[i]
            if label and previous_label:
                if len(one_group) != 0:
                    branches_tmp.append(one_group)
                one_group = []
            previous_label = label
            one_group.append(i)

        if len(one_group) != 0:
            branches_tmp.append(one_group)
        for branch in branches_tmp:
            self.branches.append(
                {
                    "start_index": branch[0],
                    "end_index": branch[-1],
                    "name": f"{self.kpoints[branch[0]].label}-{self.kpoints[branch[-1]].label}",
                }
            )

        self.is_spin_polarized = False
        if len(self.bands) == 2:
            self.is_spin_polarized = True

    def get_equivalent_kpoints(self, index: int) -> list[int]:
        """Get kpoint indices equivalent (having the same coords) to the given one.

        Args:
            index (int): The kpoint index

        Returns:
            list[int]: Equivalent indices.

        TODO: now it uses the label, we might want to use coordinates
        instead in case there was a mislabel.
        """
        # If the kpoint has no label it can't have a repetition
        # along the BandStructureSymmLine object
        if self.kpoints[index].label is None:
            return [index]

        list_index_kpoints = []
        for idx, kpt in enumerate(self.kpoints):
            if kpt.label == self.kpoints[index].label:
                list_index_kpoints.append(idx)

        return list_index_kpoints

    def get_branch(self, index: int) -> list[dict[str, Any]]:
        """Get what branch(es) is the kpoint. It takes into account the
            fact that one kpoint (e.g., Gamma) can be in several branches.

        Args:
            index (int): The kpoint index.

        Returns:
            A list of dicts [{"name", "start_index", "end_index", "index"}]
                indicating all branches in which the k_point is.
        """
        return [
            {
                "name": branch["name"],
                "start_index": branch["start_index"],
                "end_index": branch["end_index"],
                "index": idx,
            }
            for idx in self.get_equivalent_kpoints(index)
            for branch in self.branches
            if branch["start_index"] <= idx <= branch["end_index"]
        ]

    def apply_scissor(self, new_band_gap: float) -> Self:
        """Apply a scissor operator (shift of the CBM) to fit the given band gap.
        If it's a metal, we look for the band crossing the Fermi level
        and shift this one up. This will not work all the time for metals!

        Args:
            new_band_gap (float): The band gap the scissor band structure need to have.

        Returns:
            BandStructureSymmLine: With the applied scissor shift.
        """
        if self.is_metal():
            # Move then the highest index band crossing the Fermi level find this band...
            max_index = -1000
            # spin_index = None
            for idx in range(self.nb_bands):
                below = above = False
                for j in range(len(self.kpoints)):
                    if self.bands[Spin.up][idx][j] < self.efermi:
                        below = True
                    if self.bands[Spin.up][idx][j] > self.efermi:
                        above = True
                if above and below and idx > max_index:
                    max_index = idx
                    # spin_index = Spin.up
                if self.is_spin_polarized:
                    below = above = False
                    for j in range(len(self.kpoints)):
                        if self.bands[Spin.down][idx][j] < self.efermi:
                            below = True
                        if self.bands[Spin.down][idx][j] > self.efermi:
                            above = True
                    if above and below and idx > max_index:
                        max_index = idx
                        # spin_index = Spin.down
            old_dict = self.as_dict()
            shift = new_band_gap
            for spin in old_dict["bands"]:
                for k in range(len(old_dict["bands"][spin])):
                    for v in range(len(old_dict["bands"][spin][k])):
                        if k >= max_index:
                            old_dict["bands"][spin][k][v] += shift

        else:
            shift = new_band_gap - self.get_band_gap()["energy"]
            old_dict = self.as_dict()
            for spin in old_dict["bands"]:
                for k in range(len(old_dict["bands"][spin])):
                    for v in range(len(old_dict["bands"][spin][k])):
                        if old_dict["bands"][spin][k][v] >= old_dict["cbm"]["energy"]:
                            old_dict["bands"][spin][k][v] += shift
            old_dict["efermi"] += shift

        return self.from_dict(old_dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable dict representation of BandStructureSymmLine."""
        dct = super().as_dict()
        dct["branches"] = self.branches
        return dct


class LobsterBandStructureSymmLine(BandStructureSymmLine):
    """LOBSTER subclass of BandStructure with customized functions."""

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable dict representation of BandStructureSymmLine."""
        dct: dict[str, Any] = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "lattice_rec": self.lattice_rec.as_dict(),
            "efermi": self.efermi,
            "kpoints": [],
        }
        # kpoints are not kpoint objects dicts but are frac coords (this makes
        # the dict smaller and avoids the repetition of the lattice
        for k in self.kpoints:
            dct["kpoints"].append(k.as_dict()["fcoords"])
        dct["branches"] = self.branches
        dct["bands"] = {str(int(spin)): self.bands[spin].tolist() for spin in self.bands}
        dct["is_metal"] = self.is_metal()
        vbm = self.get_vbm()
        dct["vbm"] = {
            "energy": vbm["energy"],
            "kpoint_index": [int(x) for x in vbm["kpoint_index"]],
            "band_index": {str(int(spin)): vbm["band_index"][spin] for spin in vbm["band_index"]},
            "projections": {str(spin): v for spin, v in vbm["projections"].items()},
        }
        cbm = self.get_cbm()
        dct["cbm"] = {
            "energy": cbm["energy"],
            "kpoint_index": [int(x) for x in cbm["kpoint_index"]],
            "band_index": {str(int(spin)): cbm["band_index"][spin] for spin in cbm["band_index"]},
            "projections": {str(spin): v for spin, v in cbm["projections"].items()},
        }
        dct["band_gap"] = self.get_band_gap()
        dct["labels_dict"] = {}
        dct["is_spin_polarized"] = self.is_spin_polarized

        # MongoDB does not accept keys starting with "$", add a space to fix this.
        for c, label in self.labels_dict.items():
            mongo_key = f" {c}" if c.startswith("$") else c
            dct["labels_dict"][mongo_key] = label.as_dict()["fcoords"]
        if len(self.projections) != 0 and self.structure is not None:
            dct["structure"] = self.structure.as_dict()
            dct["projections"] = {str(int(spin)): np.array(v).tolist() for spin, v in self.projections.items()}
        return dct

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """Reconstruct LobsterBandStructureSymmLine from its MSONable dict representation.

        Args:
            dct (dict): All data for a LobsterBandStructureSymmLine object.

        Returns:
            A LobsterBandStructureSymmLine object.
        """
        try:
            # Strip the label to recover initial string
            # (see trick used in as_dict to handle "$" chars)
            labels_dict = {k.strip(): v for k, v in dct["labels_dict"].items()}
            projections = {}
            structure = None
            if dct.get("projections"):
                if isinstance(dct["projections"]["1"][0][0], dict):
                    raise ValueError("Old band structure dict format detected!")
                structure = Structure.from_dict(dct["structure"])
                projections = {Spin(int(spin)): np.array(v) for spin, v in dct["projections"].items()}

            return cls(
                dct["kpoints"],
                {Spin(int(k)): dct["bands"][k] for k in dct["bands"]},
                Lattice(dct["lattice_rec"]["matrix"]),
                dct["efermi"],
                labels_dict,
                structure=structure,
                projections=projections,
            )
        except Exception:
            warnings.warn(
                "Trying from_dict failed. Now we are trying the old "
                "format. Please convert your BS dicts to the new "
                "format. The old format will be retired in pymatgen "
                "5.0.",
                stacklevel=2,
            )
            return cls.from_old_dict(dct)

    @classmethod
    def from_old_dict(cls, dct: dict[str, Any]) -> Self:
        """Reconstruct a LobsterBandStructureSymmLine from a legacy dict representation.

        Args:
            dct (dict): All data for a LobsterBandStructureSymmLine object.

        Returns:
            A LobsterBandStructureSymmLine object
        """
        # Strip the label to recover initial string
        # (see trick used in as_dict to handle "$" chars)
        labels_dict = {k.strip(): v for k, v in dct["labels_dict"].items()}
        projections: dict = {}
        structure = None
        if "projections" in dct and len(dct["projections"]) != 0:
            structure = Structure.from_dict(dct["structure"])
            projections = {}
            for spin in dct["projections"]:
                dd = []
                for i in range(len(dct["projections"][spin])):
                    ddd = [dct["projections"][spin][i][j] for j in range(len(dct["projections"][spin][i]))]
                    dd.append(np.array(ddd))
                projections[Spin(int(spin))] = np.array(dd)

        return cls(
            dct["kpoints"],
            {Spin(int(k)): dct["bands"][k] for k in dct["bands"]},
            Lattice(dct["lattice_rec"]["matrix"]),
            dct["efermi"],
            labels_dict,
            structure=structure,
            projections=projections,
        )

    def get_projection_on_elements(self) -> dict[Spin, list]:
        """Get projections on elements. It sums over all available orbitals
        for each element.

        Returns:
            dict[Spin, list]: dict in the {Spin.up:[][{Element:values}],
                Spin.down:[][{Element:values}]} format.
                If there is no projections in the band structure, return {}.
        """
        result: dict[Spin, list] = {}
        for spin, v in self.projections.items():
            result[spin] = [[defaultdict(float) for _ in range(len(self.kpoints))] for _ in range(self.nb_bands)]
            for i, j in itertools.product(range(self.nb_bands), range(len(self.kpoints))):
                for key, item in v[i][j].items():
                    specie = _lobster_site_label_to_element(key)
                    for item2 in item.values():
                        result[spin][i][j][specie] += item2
        return result

    def get_projections_on_elements_and_orbitals(
        self,
        el_orb_spec: dict[SpeciesLike, list],
    ) -> dict[Spin, list]:
        """Get projections on elements and specific orbitals.

        Args:
            el_orb_spec (dict): Elements and Orbitals for which we want
                to project on. It is given as {Element: [orbitals]},
                e.g. {"Si": ["3s", "3p"]} or {"Si": ["3s", "3p_x", "3p_y", "3p_z']}
                depending on input files.

        Returns:
            A dictionary of projections on elements in the
            {Spin.up:[][{Element:{orb:values}}],
            Spin.down:[][{Element:{orb:values}}]} format
            if there is no projections in the band structure returns an empty
            dict.
        """
        result: dict[Spin, list] = {}
        el_orb_spec = {get_el_sp(el): orbs for el, orbs in el_orb_spec.items()}
        for spin, v in self.projections.items():
            result[spin] = [
                [{str(e): defaultdict(float) for e in el_orb_spec} for _ in range(len(self.kpoints))]
                for _ in range(self.nb_bands)
            ]

            for i, j in itertools.product(range(self.nb_bands), range(len(self.kpoints))):
                for key, item in v[i][j].items():
                    specie = _lobster_site_label_to_element(key)
                    orbs = el_orb_spec.get(get_el_sp(specie))
                    if orbs is None:
                        continue
                    for key2, item2 in item.items():
                        if key2 in orbs:
                            result[spin][i][j][specie][key2] += item2
        return result


@functools.cache
def _lobster_site_label_to_element(label: str) -> str:
    """Get the element symbol of a LOBSTER site label, e.g. "Si" for "Si1"."""
    return str(Element(re.split(r"[0-9]+", label)[0]))


@overload
def get_reconstructed_band_structure(
    list_bs: list[BandStructure],
    efermi: float | None = None,
) -> BandStructure:
    pass


@overload
def get_reconstructed_band_structure(
    list_bs: list[BandStructureSymmLine],
    efermi: float | None = None,
) -> BandStructureSymmLine:
    pass


def get_reconstructed_band_structure(
    list_bs: list[BandStructure] | list[BandStructureSymmLine],
    efermi: float | None = None,
) -> BandStructure | BandStructureSymmLine:
    """Merge multiple BandStructure(SymmLine) objects to a single one.

    This is typically useful when you split non self-consistent band
    structure runs to several independent jobs and want to merge the results.

    Args:
        list_bs (list): BandStructure or BandStructureSymmLine objects.
        efermi (float): The Fermi level of the reconstructed band structure.
            If None, an average of all the Fermi levels in each
            object in the list_bs is used.

    Returns:
        A BandStructure or BandStructureSymmLine object (depending on
        the type of the objects in list_bs).
    """
    if efermi is None:
        efermi = sum(b.efermi for b in list_bs) / len(list_bs)

    rec_lattice = list_bs[0].lattice_rec
    nb_bands = min(list_bs[i].nb_bands for i in range(len(list_bs)))

    kpoints = np.concatenate([[kpt.frac_coords for kpt in bs.kpoints] for bs in list_bs])
    dicts = [bs.labels_dict for bs in list_bs]
    labels_dict = {key: val.frac_coords for dct in dicts for key, val in dct.items()}

    eigenvals = {Spin.up: np.concatenate([bs.bands[Spin.up][:nb_bands] for bs in list_bs], axis=1)}
    if list_bs[0].is_spin_polarized:
        eigenvals[Spin.down] = np.concatenate([bs.bands[Spin.down][:nb_bands] for bs in list_bs], axis=1)

    projections = {}
    if len(list_bs[0].projections) != 0:
        projs = [bs.projections[Spin.up][:nb_bands] for bs in list_bs]
        projections[Spin.up] = np.concatenate(projs, axis=1)

        if list_bs[0].is_spin_polarized:
            projs = [bs.projections[Spin.down][:nb_bands] for bs in list_bs]
            projections[Spin.down] = np.concatenate(projs, axis=1)

    if isinstance(list_bs[0], BandStructureSymmLine):
        return BandStructureSymmLine(
            kpoints,
            eigenvals,
            rec_lattice,
            efermi,
            labels_dict,  # type:ignore[arg-type]
            structure=list_bs[0].structure,
            projections=projections,
        )
    return BandStructure(
        kpoints,
        eigenvals,
        rec_lattice,
        efermi,
        labels_dict,  # type:ignore[arg-type]
        structure=list_bs[0].structure,
        projections=projections,
    )
