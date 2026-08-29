"""Entries are containers for calculated information used in many analyses.

This module implements the base Entry class and ComputedEntry-related classes.
ComputedEntries contain calculated information, typically from VASP or other
electronic structure codes, and can be used as inputs for analyses such as
phase diagrams. Other Entry subclasses (e.g. ComputedEntry, PDEntry) inherit
from Entry. Utility functions for operations on entries (such as grouping
entries by structure) are also provided.
"""

from __future__ import annotations

import abc
import collections
import collections.abc
import csv
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import numpy as np
import orjson
from monty.json import MontyDecoder, MontyEncoder, MSONable

from pymatgen.core.structure_matcher import SpeciesComparator, StructureMatcher

from .composition import Composition
from .periodic_table import Element

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal, Self

    from pymatgen.analysis.phase_diagram import PDEntry
    from pymatgen.core import DummySpecies, Species, Structure
    from pymatgen.util.typing import SpeciesLike

logger = logging.getLogger(__name__)


__author__ = "Ryan Kingsbury, Matt McDermott, Shyue Ping Ong, Anubhav Jain"
__copyright__ = "Copyright 2011-2020, The Materials Project"
__version__ = "1.1"
__date__ = "April 2020"

with open(os.path.join(os.path.dirname(__file__), "data/g_els.json"), "rb") as file:
    G_ELEMS = orjson.loads(file.read())
with open(os.path.join(os.path.dirname(__file__), "data/nist_gas_gf.json"), "rb") as file:
    G_GASES = orjson.loads(file.read())


__author__ = "Shyue Ping Ong, Anubhav Jain, Ayush Gupta"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "1.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyuep@gmail.com"
__status__ = "Production"
__date__ = "Mar 03, 2020"


class Entry(MSONable, ABC):
    """A lightweight object containing the energy associated with a specific chemical composition.

    This base class is not intended to be instantiated directly. Subclasses of
    Entry must define a .energy property.
    """

    def __init__(self, composition: Composition | str | dict[str, float], energy: float) -> None:
        """Initialize an Entry.

        Args:
            composition (Composition): Composition of the entry. For
                flexibility, this can take the form of all the typical input taken by a
                Composition, including a {symbol: amt} dict, a string formula, and others.
            energy (float): Energy of the entry.
        """
        if isinstance(composition, Composition):
            self._composition = composition
        else:
            self._composition = Composition(composition)
        # self._composition = Composition(composition)
        self._energy = energy

    @property
    def is_element(self) -> bool:
        """Whether composition of entry is an element."""
        # NOTE _composition rather than composition as GrandPDEntry edge case exists if we
        # have a compound where chempots are given for all bar one element type
        return self._composition.is_element

    @property
    def composition(self) -> Composition:
        """The composition of the entry."""
        return self._composition

    @property
    def formula(self) -> str:
        """The formula of the entry."""
        return self._composition.formula

    @property
    def reduced_formula(self) -> str:
        """The reduced formula of the entry."""
        return self._composition.reduced_formula

    @property
    @abstractmethod
    def energy(self) -> float:
        """The energy of the entry."""
        raise NotImplementedError

    @property
    def elements(self) -> list[Element | Species | DummySpecies]:
        """The set of elements in the entry."""
        return self._composition.elements

    @property
    def energy_per_atom(self) -> float:
        """The energy per atom of the entry."""
        return self.energy / self.composition.num_atoms

    def __repr__(self):
        return f"{type(self).__name__} : {self.composition} with energy = {self.energy:.4f}"

    def normalize(self, mode: Literal["formula_unit", "atom"] = "formula_unit") -> Entry:
        """Normalize the entry's composition and energy.

        Args:
            mode ("formula_unit" | "atom"): "formula_unit" (the default) normalizes to composition.reduced_formula.
                "atom" normalizes such that the composition amounts sum to 1.
        """
        factor = self._normalization_factor(mode)
        new_composition = self._composition / factor
        new_energy = self._energy / factor

        new_entry_dict = self.as_dict()
        new_entry_dict["composition"] = new_composition.as_dict()
        new_entry_dict["energy"] = new_energy

        return cast("Entry", self.from_dict(new_entry_dict))

    def _normalization_factor(self, mode: Literal["formula_unit", "atom"] = "formula_unit") -> float:
        # NOTE here we use composition rather than _composition in order to ensure
        # that we have the expected behavior downstream in cases where composition
        # is overwritten (GrandPotPDEntry, TransformedPDEntry)
        if mode == "atom":
            factor = self.composition.num_atoms
        elif mode == "formula_unit":
            factor = self.composition.get_reduced_composition_and_factor()[1]
        else:
            raise ValueError(f"{mode} is not an allowed option for normalization")

        return factor

    def as_dict(self) -> dict:
        """MSONable dict."""
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "energy": self._energy,
            "composition": self._composition.as_dict(),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        # NOTE: Scaled duplicates i.e. physically equivalent materials
        # are not equal unless normalized separately.
        if self is other:
            return True

        # Equality is defined based on composition and energy
        # If structures are involved, it is assumed that a {composition, energy} is
        # vanishingly unlikely to be the same if the structures are different

        if not np.allclose(self.energy, other.energy):
            return False

        return self.composition == other.composition

    def __hash__(self) -> int:
        # NOTE truncate _energy to 8 dp to ensure same robustness
        # as np.allclose
        return hash(f"{type(self).__name__}{self._composition.formula}{self._energy:.8f}")


class EnergyAdjustment(MSONable):
    """Lightweight class to contain information about an energy adjustment or
    energy correction.
    """

    def __init__(
        self,
        value: float,
        uncertainty: float = np.nan,
        name: str = "Manual adjustment",
        cls: dict | None = None,
        description: str = "",
    ) -> None:
        """Initialize an EnergyAdjustment.

        Args:
            value (float): value of the energy adjustment in eV
            uncertainty (float): uncertainty of the energy adjustment in eV. Default: np.nan
            name (str): human-readable name of the energy adjustment.
                (Default: Manual adjustment)
            cls (dict): Serialized Compatibility class used to generate the energy adjustment. Defaults to {}.
            description (str): human-readable explanation of the energy adjustment.
        """
        self.name = name
        self.cls = cls or {}
        self.description = description
        self._value = value
        self._uncertainty = float(uncertainty)

    @property
    def value(self) -> float:
        """The value of the energy correction in eV."""
        return self._value

    @property
    def uncertainty(self) -> float:
        """The uncertainty in the value of the energy adjustment in eV."""
        return self._uncertainty

    @abc.abstractmethod
    def normalize(self, factor: float) -> None:
        """Scale the value of the current energy adjustment by factor in-place.

        This method is utilized in ComputedEntry.normalize() to scale the energies to a formula unit basis
        (e.g. E_Fe6O9 = 3 x E_Fe2O3).
        """

    @property
    @abc.abstractmethod
    def explain(self) -> str:
        """Return an explanation of how the energy adjustment is calculated."""

    def __repr__(self) -> str:
        name, value, uncertainty, description = (
            self.name,
            float(self.value),
            self.uncertainty,
            self.description,
        )
        # self.cls might not be a dict if monty decoding is enabled in the new MPRester
        # which hydrates all dicts with @class and @module keys into classes in which case
        # we expect a Compatibility subclass
        generated_by = self.cls.get("@class", "unknown") if isinstance(self.cls, dict) else type(self.cls).__name__
        return f"{type(self).__name__}({name=}, {value=:.3}, {uncertainty=:.3}, {description=}, {generated_by=})"


class ConstantEnergyAdjustment(EnergyAdjustment):
    """A constant energy adjustment applied to a ComputedEntry. Useful in energy referencing
    schemes such as the Aqueous energy referencing scheme.
    """

    def __init__(
        self,
        value: float,
        uncertainty: float = np.nan,
        name: str = "Constant energy adjustment",
        cls: dict | None = None,
        description: str = "Constant energy adjustment",
    ) -> None:
        """Initialize a ConstantEnergyAdjustment.

        Args:
            value (float): the energy adjustment in eV
            uncertainty (float): uncertainty of the energy adjustment in eV. (Default: np.nan)
            name (str): human-readable name of the energy adjustment.
                (Default: Constant energy adjustment)
            cls (dict): Serialized Compatibility class used to generate the energy
                adjustment. (Default: None)
            description (str): human-readable explanation of the energy adjustment.
        """
        super().__init__(value, uncertainty, name=name, cls=cls, description=description)
        self._value = value
        self._uncertainty = float(uncertainty)

    @property
    def explain(self) -> str:
        """An explanation of how the energy adjustment is calculated."""
        return f"{self.description} ({self.value:.3f} eV)"

    def normalize(self, factor: float) -> None:
        """Normalize energy adjustment (in place), dividing value/uncertainty by a
        factor.

        Args:
            factor (float): factor to divide by.
        """
        self._value /= factor
        self._uncertainty /= factor


class ManualEnergyAdjustment(ConstantEnergyAdjustment):
    """A manual energy adjustment applied to a ComputedEntry."""

    def __init__(self, value: float) -> None:
        """Initialize a ManualEnergyAdjustment.

        Args:
            value (float): the energy adjustment in eV.
        """
        name = "Manual energy adjustment"
        description = "Manual energy adjustment"
        super().__init__(value, name=name, cls=None, description=description)


class CompositionEnergyAdjustment(EnergyAdjustment):
    """An energy adjustment applied to a ComputedEntry based on the atomic composition.
    Used in various DFT energy correction schemes.
    """

    def __init__(
        self,
        adj_per_atom: float,
        n_atoms: float,
        uncertainty_per_atom: float = np.nan,
        name: str = "",
        cls: dict | None = None,
        description: str = "Composition-based energy adjustment",
    ) -> None:
        """Initialize a CompositionEnergyAdjustment.

        Args:
            adj_per_atom (float): energy adjustment to apply per atom, in eV/atom
            n_atoms (float): number of atoms.
            uncertainty_per_atom (float): uncertainty in energy adjustment to apply per atom, in eV/atom.
                (Default: np.nan)
            name (str): human-readable name of the energy adjustment.
                (Default: "")
            cls (dict): Serialized Compatibility class used to generate the energy
                adjustment. (Default: None)
            description (str): human-readable explanation of the energy adjustment.
        """
        self._adj_per_atom = adj_per_atom
        self.uncertainty_per_atom = float(uncertainty_per_atom)
        self.n_atoms = n_atoms
        self.cls = cls or {}
        self.name = name
        self.description = description

    @property
    def value(self) -> float:
        """The value of the energy adjustment in eV."""
        return self._adj_per_atom * self.n_atoms

    @property
    def uncertainty(self) -> float:
        """The value of the energy adjustment in eV."""
        return self.uncertainty_per_atom * self.n_atoms

    @property
    def explain(self) -> str:
        """An explanation of how the energy adjustment is calculated."""
        return f"{self.description} ({self._adj_per_atom:.3f} eV/atom x {self.n_atoms} atoms)"

    def normalize(self, factor: float) -> None:
        """Normalize energy adjustment (in place), dividing value/uncertainty by a
        factor.

        Args:
            factor: factor to divide by.
        """
        self.n_atoms /= factor


class TemperatureEnergyAdjustment(EnergyAdjustment):
    """An energy adjustment applied to a ComputedEntry based on the temperature.
    Used, for example, to add entropy to DFT energies.
    """

    def __init__(
        self,
        adj_per_deg: float,
        temp: float,
        n_atoms: float,
        uncertainty_per_deg: float = np.nan,
        name: str = "",
        cls: dict | None = None,
        description: str = "Temperature-based energy adjustment",
    ) -> None:
        """Initialize a TemperatureEnergyAdjustment.

        Args:
            adj_per_deg (float): energy adjustment to apply per degree K, in eV/atom
            temp (float): temperature in Kelvin
            n_atoms (float): number of atoms
            uncertainty_per_deg (float): uncertainty in energy adjustment to apply per degree K,
                in eV/atom. (Default: np.nan)
            name (str): human-readable name of the energy adjustment.
                (Default: "")
            cls (dict): Serialized Compatibility class used to generate the energy
                adjustment. (Default: None)
            description (str): human-readable explanation of the energy adjustment.
        """
        self._adj_per_deg = adj_per_deg
        self.uncertainty_per_deg = float(uncertainty_per_deg)
        self.temp = temp
        self.n_atoms = n_atoms
        self.name = name
        self.cls = cls or {}
        self.description = description

    @property
    def value(self) -> float:
        """The value of the energy correction in eV."""
        return self._adj_per_deg * self.temp * self.n_atoms

    @property
    def uncertainty(self) -> float:
        """The value of the energy adjustment in eV."""
        return self.uncertainty_per_deg * self.temp * self.n_atoms

    @property
    def explain(self) -> str:
        """An explanation of how the energy adjustment is calculated."""
        return f"{self.description} ({self._adj_per_deg:.4f} eV/K/atom x {self.temp} K x {self.n_atoms} atoms)"

    def normalize(self, factor: float) -> None:
        """Normalize energy adjustment (in place), dividing value/uncertainty by a
        factor.

        Args:
            factor: factor to divide by.
        """
        self.n_atoms /= factor


class ComputedEntry(Entry):
    """Lightweight Entry object for computed data. Contains facilities
    for applying corrections to the energy attribute and for storing
    calculation parameters.
    """

    def __init__(
        self,
        composition: Composition | str | dict[str, float],
        energy: float,
        correction: float = 0.0,
        energy_adjustments: list | None = None,
        parameters: dict | None = None,
        data: dict | None = None,
        entry_id: str | None = None,
    ) -> None:
        """Initialize a ComputedEntry.

        Args:
            composition (Composition): Composition of the entry. For
                flexibility, this can take the form of all the typical input
                taken by a Composition, including a {symbol: amt} dict,
                a string formula, and others.
            energy (float): Energy of the entry. Usually the final calculated
                energy from VASP or other electronic structure codes.
            correction (float): Manually set an energy correction, will ignore
                energy_adjustments if specified.
            energy_adjustments: An optional list of EnergyAdjustment to
                be applied to the energy. This is used to modify the energy for
                certain analyses. Defaults to None.
            parameters: An optional dict of parameters associated with
                the entry. Defaults to None.
            data: An optional dict of any additional data associated
                with the entry. Defaults to None.
            entry_id: An optional id to uniquely identify the entry.
        """
        super().__init__(composition, energy)
        self.energy_adjustments = energy_adjustments or []

        if not math.isclose(correction, 0.0):
            if energy_adjustments:
                raise ValueError(
                    f"Argument conflict! Setting correction = {correction:.3f} conflicts "
                    "with setting energy_adjustments. Specify one or the "
                    "other."
                )

            self.correction = correction

        self.parameters = parameters or {}
        self.data = data or {}
        self.entry_id = entry_id
        self.name = self.reduced_formula

    @property
    def uncorrected_energy(self) -> float:
        """The *uncorrected* energy of the entry."""
        return self._energy

    @property
    def energy(self) -> float:
        """The *corrected* energy of the entry."""
        return self.uncorrected_energy + self.correction

    @property
    def uncorrected_energy_per_atom(self) -> float:
        """The *uncorrected* energy of the entry, normalized by atoms in eV/atom."""
        return self.uncorrected_energy / self.composition.num_atoms

    @property
    def correction(self) -> float:
        """The total energy correction / adjustment applied to the entry in eV."""
        return math.fsum(ea.value for ea in self.energy_adjustments)

    @correction.setter
    def correction(self, x: float) -> None:
        corr = ManualEnergyAdjustment(x)
        self.energy_adjustments = [corr]

    @property
    def correction_per_atom(self) -> float:
        """The total energy correction / adjustment applied to the entry in eV/atom."""
        return self.correction / self.composition.num_atoms

    @property
    def correction_uncertainty(self) -> float:
        """The uncertainty of the energy adjustments applied to the entry in eV."""
        # Quadrature sum of the (independent) adjustment uncertainties, skipping unknown (nan) ones and those of
        # zero-valued adjustments:
        quad_sum = math.hypot(
            *(ea.uncertainty for ea in self.energy_adjustments if ea.value and not np.isnan(ea.uncertainty))
        )
        # A total of zero means no uncertainty information supplied, so return nan rather than implying exact certainty:
        return quad_sum or np.nan

    @property
    def correction_uncertainty_per_atom(self) -> float:
        """The uncertainty of the energy adjustments applied to the entry in eV/atom."""
        return self.correction_uncertainty / self.composition.num_atoms

    def normalize(self, mode: Literal["formula_unit", "atom"] = "formula_unit") -> ComputedEntry:
        """Normalize the entry's composition and energy.

        Args:
            mode ("formula_unit" | "atom"): "formula_unit" (the default) normalizes to composition.reduced_formula.
                "atom" normalizes such that the composition amounts sum to 1.
        """
        factor = self._normalization_factor(mode)
        new_composition = self._composition / factor
        new_energy = self._energy / factor

        new_entry_dict = self.as_dict()
        new_entry_dict["composition"] = new_composition.as_dict()
        new_entry_dict["energy"] = new_energy

        # TODO: make sure EnergyAdjustments are _also_ immutable to avoid this hacking
        new_energy_adjustments = MontyDecoder().process_decoded(new_entry_dict["energy_adjustments"])
        for ea in new_energy_adjustments:
            ea.normalize(factor)
        new_entry_dict["energy_adjustments"] = [ea.as_dict() for ea in new_energy_adjustments]

        return self.from_dict(new_entry_dict)

    def __repr__(self) -> str:
        n_atoms = self.composition.num_atoms
        output = [
            f"{self.entry_id} {type(self).__name__:<10} - {self.formula:<12} ({self.reduced_formula})",
            f"{'Energy (Uncorrected)':<24} = {self._energy:<9.4f} eV ({self._energy / n_atoms:<8.4f} eV/atom)",
            f"{'Correction':<24} = {self.correction:<9.4f} eV ({self.correction / n_atoms:<8.4f} eV/atom)",
            f"{'Energy (Final)':<24} = {self.energy:<9.4f} eV ({self.energy_per_atom:<8.4f} eV/atom)",
            "Energy Adjustments:",
        ]
        if len(self.energy_adjustments) == 0:
            output.append("  None")
        else:
            output.extend(
                f"  {e.name:<23}: {e.value:<9.4f} eV ({e.value / n_atoms:<8.4f} eV/atom)"
                for e in self.energy_adjustments
            )
        output.append("Parameters:")
        for k, v in self.parameters.items():
            output.append(f"  {k:<22} = {v}")
        output.append("Data:")
        for k, v in self.data.items():
            output.append(f"  {k:<22} = {v}")
        return "\n".join(output)

    def __eq__(self, other: object) -> bool:
        # NOTE: Scaled duplicates i.e. physically equivalent materials
        # are not equal unless normalized separately.
        if self is other:
            return True

        needed_attrs = ("composition", "energy", "entry_id")
        if not all(hasattr(other, attr) for attr in needed_attrs):
            return NotImplemented

        other = cast("ComputedEntry", other)

        # Equality is defined based on composition and energy.
        # If structures are involved, it is assumed that a {composition, energy} is
        # vanishingly unlikely to be the same if the structures are different.
        # If entry_ids are different, assume that the entries are different.
        # However, if entry_id is same, they may have different corrections (e.g., due
        # to mixing scheme used) and thus should be compared on corrected energy.

        if getattr(self, "entry_id", False) and getattr(other, "entry_id", False) and self.entry_id != other.entry_id:
            return False

        if not math.isclose(self.energy, other.energy):
            return False

        # assumes that data, parameters are equivalent
        return self.composition == other.composition

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Reconstruct ComputedEntry from its MSONable dict representation.

        Args:
           dct (dict): Dict representation.

        Returns:
            ComputedEntry
        """
        # Must handle cases where some kwargs exist in `dct` but are None
        # include extra logic to ensure these get properly treated
        energy_adj = [MontyDecoder().process_decoded(e) for e in (dct.get("energy_adjustments", []) or [])]

        # this is the preferred / modern way of instantiating ComputedEntry
        # we don't pass correction explicitly because it will be calculated
        # on the fly from energy_adjustments
        correction = 0
        if not math.isclose(dct["correction"], 0) and len(energy_adj) == 0:
            # this block is for legacy ComputedEntry that were
            # serialized before we had the energy_adjustments attribute.
            correction = dct["correction"]

        return cls(
            dct["composition"],
            dct["energy"],
            correction=correction,
            energy_adjustments=energy_adj,
            parameters={k: MontyDecoder().process_decoded(v) for k, v in (dct.get("parameters", {}) or {}).items()},
            data={k: MontyDecoder().process_decoded(v) for k, v in (dct.get("data", {}) or {}).items()},
            entry_id=dct.get("entry_id"),
        )

    def as_dict(self) -> dict:
        """MSONable dict."""
        return_dict = super().as_dict()
        return_dict |= {
            "entry_id": self.entry_id,
            "correction": self.correction,
            "energy_adjustments": json.loads(json.dumps(self.energy_adjustments, cls=MontyEncoder)),
            "parameters": json.loads(json.dumps(self.parameters, cls=MontyEncoder)),
            "data": json.loads(json.dumps(self.data, cls=MontyEncoder)),
        }
        return return_dict

    def __hash__(self) -> int:
        # NOTE It is assumed that the user will ensure entry_id is a
        # unique identifier for ComputedEntry type classes.
        if self.entry_id is not None:
            return hash(f"{type(self).__name__}{self.entry_id}")

        return super().__hash__()

    def copy(self) -> ComputedEntry:
        """Get a copy of the ComputedEntry."""
        return ComputedEntry(
            composition=self.composition,
            energy=self.uncorrected_energy,
            energy_adjustments=self.energy_adjustments,
            parameters=self.parameters,
            data=self.data,
            entry_id=self.entry_id,
        )


class ComputedStructureEntry(ComputedEntry):
    """A heavier version of ComputedEntry which contains a structure as well. The
    structure is needed for some analyses.
    """

    def __init__(
        self,
        structure: Structure,
        energy: float,
        correction: float = 0.0,
        composition: Composition | str | dict[str, float] | None = None,
        energy_adjustments: list | None = None,
        parameters: dict | None = None,
        data: dict | None = None,
        entry_id: str | None = None,
    ) -> None:
        """Initialize a ComputedStructureEntry.

        Args:
            structure (Structure): The actual structure of an entry.
            energy (float): Energy of the entry. Usually the final calculated
                energy from VASP or other electronic structure codes.
            correction (float, optional): A correction to the energy. This is mutually exclusive with
                energy_adjustments, i.e. pass either or neither but not both. Defaults to 0.
            composition (Composition): Composition of the entry. For
                flexibility, this can take the form of all the typical input
                taken by a Composition, including a {symbol: amt} dict,
                a string formula, and others.
            energy_adjustments: An optional list of EnergyAdjustment to
                be applied to the energy. This is used to modify the energy for
                certain analyses. Defaults to None.
            parameters: An optional dict of parameters associated with
                the entry. Defaults to None.
            data: An optional dict of any additional data associated
                with the entry. Defaults to None.
            entry_id: An optional id to uniquely identify the entry.
        """
        if composition:
            if isinstance(composition, Composition):
                pass
            else:
                composition = Composition(composition)
            # composition = Composition(composition)
            if (
                composition.get_integer_formula_and_factor()[0]
                != structure.composition.get_integer_formula_and_factor()[0]
            ):
                raise ValueError("Mismatching composition provided.")
        else:
            composition = structure.composition

        super().__init__(
            composition,
            energy,
            correction=correction,
            energy_adjustments=energy_adjustments,
            parameters=parameters,
            data=data,
            entry_id=entry_id,
        )
        self._structure = structure

    @property
    def structure(self) -> Structure:
        """The structure of the entry."""
        return self._structure

    def as_dict(self) -> dict:
        """MSONable dict."""
        dct = super().as_dict()
        dct["structure"] = self.structure.as_dict()
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Reconstruct ComputedStructureEntry from its MSONable dict representation.

        Args:
            dct (dict): Dict representation.

        Returns:
            ComputedStructureEntry
        """
        # the first block here is for legacy ComputedEntry that were
        # serialized before we had the energy_adjustments attribute.
        if not math.isclose(dct["correction"], 0) and not dct.get("energy_adjustments"):
            struct = MontyDecoder().process_decoded(dct["structure"])
            return cls(
                struct,
                dct["energy"],
                correction=dct["correction"],
                parameters={k: MontyDecoder().process_decoded(v) for k, v in dct.get("parameters", {}).items()},
                data={k: MontyDecoder().process_decoded(v) for k, v in dct.get("data", {}).items()},
                entry_id=dct.get("entry_id"),
            )
        # this is the preferred / modern way of instantiating ComputedEntry
        # we don't pass correction explicitly because it will be calculated
        # on the fly from energy_adjustments
        return cls(
            MontyDecoder().process_decoded(dct["structure"]),
            dct["energy"],
            composition=dct.get("composition"),
            correction=0,
            energy_adjustments=[MontyDecoder().process_decoded(e) for e in dct.get("energy_adjustments", {})],
            parameters={k: MontyDecoder().process_decoded(v) for k, v in dct.get("parameters", {}).items()},
            data={k: MontyDecoder().process_decoded(v) for k, v in dct.get("data", {}).items()},
            entry_id=dct.get("entry_id"),
        )

    def normalize(
        self,
        mode: Literal["formula_unit", "atom"] = "formula_unit",
    ) -> ComputedStructureEntry:
        """Normalize the entry's composition and energy. The structure remains unchanged.

        Args:
            mode ("formula_unit" | "atom"): "formula_unit" (the default) normalizes to composition.reduced_formula.
                "atom" normalizes such that the composition amounts sum to 1.
        """
        # TODO: this should raise TypeError since normalization does not make sense
        # raise TypeError("You cannot normalize a structure.")
        warnings.warn(
            f"Normalization of a `{type(self).__name__}` makes "
            "`self.composition` and `self.structure.composition` inconsistent"
            " - please use self.composition for all further calculations.",
            stacklevel=2,
        )
        # TODO: find a better solution for creating copies instead of as/from dict
        factor = self._normalization_factor(mode)
        dct = super().normalize(mode).as_dict()
        dct["structure"] = self.structure.as_dict()
        entry = self.from_dict(dct)
        entry._composition /= factor
        return entry

    def copy(self) -> ComputedStructureEntry:
        """Get a copy of the ComputedStructureEntry."""
        return ComputedStructureEntry(
            structure=self.structure.copy(),
            energy=self.uncorrected_energy,
            composition=self.composition,
            energy_adjustments=self.energy_adjustments,
            parameters=self.parameters,
            data=self.data,
            entry_id=self.entry_id,
        )


def _get_host(structure, species_to_remove):
    if species_to_remove:
        struct = structure.copy()
        struct.remove_species(species_to_remove)
        return struct
    return structure


def _perform_grouping(args):
    (
        entries_json,
        hosts_json,
        ltol,
        stol,
        angle_tol,
        primitive_cell,
        scale,
        comparator,
        groups,
    ) = args

    entries = json.loads(entries_json, cls=MontyDecoder)
    hosts = json.loads(hosts_json, cls=MontyDecoder)
    unmatched = list(zip(entries, hosts, strict=True))
    while len(unmatched) > 0:
        ref_host = unmatched[0][1]
        logger.info(f"Reference tid = {unmatched[0][0].entry_id}, formula = {ref_host.formula}")
        ref_formula = ref_host.reduced_formula
        logger.info(f"Reference host = {ref_formula}")
        matches = [unmatched[0]]
        for idx in range(1, len(unmatched)):
            test_host = unmatched[idx][1]
            logger.info(f"Testing tid = {unmatched[idx][0].entry_id}, formula = {test_host.formula}")
            test_formula = test_host.reduced_formula
            logger.info(f"Test host = {test_formula}")
            matcher = StructureMatcher(
                ltol=ltol,
                stol=stol,
                angle_tol=angle_tol,
                primitive_cell=primitive_cell,
                scale=scale,
                comparator=comparator,
            )
            if matcher.fit(ref_host, test_host):
                logger.info("Fit found")
                matches.append(unmatched[idx])
        groups.append(json.dumps([m[0] for m in matches], cls=MontyEncoder))
        unmatched = list(filter(lambda x: x not in matches, unmatched))
        logger.info(f"{len(unmatched)} unmatched remaining")


def group_entries_by_structure(
    entries,
    species_to_remove=None,
    ltol=0.2,
    stol=0.4,
    angle_tol=5,
    primitive_cell=True,
    scale=True,
    comparator=None,
    ncpus=None,
):
    """Given a sequence of ComputedStructureEntries, use structure fitter to group
    them by structural similarity.

    Args:
        entries: Sequence of ComputedStructureEntries.
        species_to_remove: Sometimes you want to compare a host framework
            (e.g., in Li-ion battery analysis). This allows you to specify
            species to remove before structural comparison.
        ltol (float): Fractional length tolerance. Default is 0.2.
        stol (float): Site tolerance in Angstrom. Default is 0.4 Angstrom.
        angle_tol (float): Angle tolerance in degrees. Default is 5 degrees.
        primitive_cell (bool): If true: input structures will be reduced to
            primitive cells prior to matching. Defaults to True.
        scale: Input structures are scaled to equivalent volume if true;
            For exact matching, set to False.
        comparator: A comparator object implementing an equals method that
            declares equivalency of sites. Default is SpeciesComparator,
            which implies rigid species mapping.
        ncpus: Number of cpus to use. Use of multiple cpus can greatly improve
            fitting speed. Default of None means serial processing.

    Returns:
        Sequence of sequence of entries by structural similarity. e.g,
        [[ entry1, entry2], [entry3, entry4, entry5]]
    """
    if comparator is None:
        comparator = SpeciesComparator()
    start = datetime.now(tz=UTC)
    logger.info(f"Started at {start}")
    entries_host = [(entry, _get_host(entry.structure, species_to_remove)) for entry in entries]
    if ncpus:
        symm_entries = defaultdict(list)
        for entry, host in entries_host:
            symm_entries[comparator.get_structure_hash(host)].append((entry, host))

        logger.info(f"Using {ncpus} cpus")
        manager = mp.Manager()
        groups = manager.list()
        with mp.Pool(ncpus) as pool:
            # Parallel processing only supports Python primitives and not objects.
            pool.map(
                _perform_grouping,
                [
                    (
                        json.dumps([e[0] for e in eh], cls=MontyEncoder),
                        json.dumps([e[1] for e in eh], cls=MontyEncoder),
                        ltol,
                        stol,
                        angle_tol,
                        primitive_cell,
                        scale,
                        comparator,
                        groups,
                    )
                    for eh in symm_entries.values()
                ],
            )
    else:
        groups = []
        hosts = [host for entry, host in entries_host]
        _perform_grouping(
            (
                json.dumps(entries, cls=MontyEncoder),
                json.dumps(hosts, cls=MontyEncoder),
                ltol,
                stol,
                angle_tol,
                primitive_cell,
                scale,
                comparator,
                groups,
            )
        )
    entry_groups = [json.loads(g, cls=MontyDecoder) for g in groups]
    logger.info(f"Finished at {datetime.now(tz=UTC)}")
    logger.info(f"Took {datetime.now(tz=UTC) - start}")
    return entry_groups


def group_entries_by_composition(entries, sort_by_e_per_atom=True):
    """Given a sequence of Entry-like objects, group them by composition and
        optionally sort by energy above hull.

    Args:
        entries (list): Sequence of Entry-like objects.
        sort_by_e_per_atom (bool): Whether to sort the grouped entries by
            energy per atom (lowest energy first). Default True.

    Returns:
        Sequence of sequence of entries by composition. e.g,
        [[ entry1, entry2], [entry3, entry4, entry5]]
    """
    entry_groups = []
    entries = sorted(entries, key=lambda e: e.reduced_formula)
    for _, g in itertools.groupby(entries, key=lambda e: e.reduced_formula):
        group = list(g)
        if sort_by_e_per_atom:
            group = sorted(group, key=lambda e: e.energy_per_atom)

        entry_groups.append(group)

    return entry_groups


class EntrySet(collections.abc.MutableSet, MSONable):
    """A convenient container for manipulating entries. Allows for generating
    subsets, dumping into files, etc.
    """

    def __init__(self, entries: Iterable[PDEntry | ComputedEntry | ComputedStructureEntry]):
        """Initialize an EntrySet.

        Args:
            entries: All the entries.
        """
        self.entries = set(entries)

    def __contains__(self, item):
        return item in self.entries

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def add(self, element):
        """Add an entry.

        Args:
            element: Entry
        """
        self.entries.add(element)

    def discard(self, element):
        """Discard an entry.

        Args:
            element: Entry
        """
        self.entries.discard(element)

    @property
    def chemsys(self) -> set:
        """Set of element symbols representing the chemical system, e.g. {"Li", "Fe", "P", "O"}."""
        chemsys = set()
        for e in self.entries:
            chemsys.update([el.symbol for el in e.composition])
        return chemsys

    @property
    def ground_states(self) -> set:
        """A set containing only the entries that are ground states, i.e., the lowest energy
        per atom entry at each composition.
        """
        entries = sorted(self.entries, key=lambda e: e.reduced_formula)
        return {
            min(g, key=lambda e: e.energy_per_atom)
            for _, g in itertools.groupby(entries, key=lambda e: e.reduced_formula)
        }

    def remove_non_ground_states(self):
        """Removes all non-ground state entries, i.e., only keep the lowest energy
        per atom entry at each composition.
        """
        self.entries = self.ground_states

    def is_ground_state(self, entry) -> bool:
        """Boolean indicating whether a given Entry is a ground state."""
        return entry in self.ground_states

    def get_subset_in_chemsys(self, chemsys: list[str]):
        """Get an EntrySet containing only the set of entries belonging to
        a particular chemical system (in this definition, it includes all sub
        systems). For example, if the entries are from the
        Li-Fe-P-O system, and chemsys=["Li", "O"], only the Li, O,
        and Li-O entries are returned.

        Args:
            chemsys: Chemical system specified as list of elements. e.g.
                ["Li", "O"]

        Returns:
            EntrySet
        """
        chem_sys = set(chemsys)
        if not chem_sys.issubset(self.chemsys):
            raise ValueError(
                f"{sorted(chem_sys)} is not a subset of {sorted(self.chemsys)}, extra: {chem_sys - self.chemsys}"
            )
        subset = set()
        for e in self.entries:
            elements = [sp.symbol for sp in e.composition]
            if chem_sys.issuperset(elements):
                subset.add(e)
        return EntrySet(subset)

    def as_dict(self) -> dict[Literal["entries"], list[Entry]]:
        """Get MSONable dict."""
        return {"entries": list(self.entries)}

    def to_csv(self, filename: str, latexify_names: bool = False) -> None:
        """Exports PDEntries to a csv.

        Args:
            filename: Filename to write to.
            entries: PDEntries to export.
            latexify_names: Format entry names to be LaTex compatible,
                e.g. Li_{2}O
        """
        els: set[SpeciesLike] = set()
        for entry in self.entries:
            els.update(entry.elements)
        elements = sorted(els, key=lambda a: a.X)
        with open(filename, mode="w", encoding="utf-8") as file:
            writer = csv.writer(
                file,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writerow(["Name"] + [el.symbol for el in elements] + ["Energy"])
            for entry in self.entries:
                row: list[str] = [(entry.name if not latexify_names else re.sub(r"([0-9]+)", r"_{\1}", entry.name))]
                row.extend([str(entry.composition[el]) for el in elements])
                row.append(str(entry.energy))
                writer.writerow(row)

    @classmethod
    def from_csv(cls, filename: str) -> Self:
        """Imports PDEntries from a csv.

        Args:
            filename: Filename to import from.

        Returns:
            List of Elements, List of PDEntries
        """
        from pymatgen.analysis.phase_diagram import PDEntry

        with open(filename, encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            entries = []
            header_read = False
            elements: list[str] = []
            for row in reader:
                if not header_read:
                    elements = row[1 : (len(row) - 1)]
                    header_read = True
                else:
                    name = row[0]
                    energy = float(row[-1])
                    comp = {}
                    for ind in range(1, len(row) - 1):
                        if float(row[ind]) > 0:
                            comp[Element(elements[ind - 1])] = float(row[ind])
                    entries.append(PDEntry(Composition(comp), energy, name))
        return cls(entries)
