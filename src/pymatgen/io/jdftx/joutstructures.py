"""Module for JOutStructures class.

This module contains the JOutStructures class for storing a series of
JOutStructure.
"""

from __future__ import annotations

import pprint
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from pymatgen.core.structure import Structure
    from pymatgen.io.jdftx.jelstep import JElSteps


from pymatgen.io.jdftx._output_utils import correct_geom_opt_type
from pymatgen.io.jdftx.joutstructure import JOutStructure, _line_type_map

__author__ = "Ben Rich"

# TODO: Break this up into data also stored in `properties` and `site_properties`
_joss_atrs_from_last_slice = (
    "etype",
    "eopt_type",
    "emin_flag",
    "ecomponents",
    "elecmindata",
    "stress",
    "strain",
    "forces",
    "nstep",
    "e",
    "grad_k",
    "alpha",
    "linmin",
    "nelectrons",
    "abs_magneticmoment",
    "tot_magneticmoment",
    "mu",
    "elec_nstep",
    "elec_e",
    "elec_grad_k",
    "elec_alpha",
    "elec_linmin",
    "charges",
    "magnetic_moments",
    "selective_dynamics",
    "structure",
    "t_s",
    "pe",
    "ke",
    "t_k",
    "p_bar",
    "tmd_fs",
    "thermostat_velocity",
)


@dataclass
class JOutStructures:
    """
    Class for storing a series of JOutStructure objects.

    A class for storing a series of JOutStructure objects.

    Attributes:
        out_slice_start_flag (str): The string that marks the beginning of the portion of an out file slice
            that contains data for a JOutStructures object.
        opt_type (str | None): The type of optimization performed on the structures in the JOutStructures object.
        geom_converged (bool): Whether the geometry of the last structure in the list has converged.
        geom_converged_reason (str | None): The reason the geometry of the last structure in the list has converged.
        elec_converged (bool): Whether the electronic density of the last structure in the list has converged.
        elec_converged_reason (str | None): The reason the electronic density of the last structure in the list has
            converged.
        slices (list[JOutStructure]): A list of JOutStructure objects.
        eopt_type (str | None): The type of electronic optimization performed on the last structure in the list.
        etype (str | None): String representation of total energy-type of system. Commonly "G"
        (grand-canonical potential) for GC calculations, "F" for canonical (fixed electron count) calculations with
        electronic smearing, and "Etot" for those without.
        emin_flag (str | None): The flag for the electronic minimization.
        ecomponents (dict | None): The energy components of the last structure in the list, in eV.
        elecmindata (JElSteps): The electronic minimization data.
        stress (np.ndarray | None): The stress tensor.
        strain (np.ndarray | None): The strain tensor.
        thermostat_velocity (np.ndarray | None): The Nose-Hoover chain thermostat velocities (one per chain element,
            chainLengthT in ionic-dynamics), in JDFTx units (inverse Hartree atomic units of time) so they can be
            written back unchanged as the thermostat-velocity tag of a restart input. None if not printed.
        nstep (int | None): The step number of the last geometric step (counted from 0).
        e (float | None): The total energy in eV.
        grad_k (float | None): The final norm of the preconditioned gradient for geometric optimization of the most
            recent JDFTx call (evaluated as dot(g, Kg), where g is the gradient and Kg is the preconditioned gradient).
            (written as "|grad|_K" in JDFTx output).
        alpha (float | None): The step size of the final geometric step in the most recent JDFTx call.
        linmin (float | None): The final normalized projection of the geometric step direction onto the gradient for
            the most recent JDFTx call.
        abs_magneticmoment (float | None): The absolute magnetic moment of the most recent JDFTx call.
        tot_magneticmoment (float | None): The total magnetic moment of the most recent JDFTx call.
        mu (float | None): The Fermi energy of the most recent JDFTx call.
        elec_e (float) | None: The final energy of the most recent electronic optimization step.
        elec_nstep (int): The step number of the last electronic optimization step of the last geometric step
            (counted from 0).
        elec_grad_k (float | None): The final norm of the preconditioned gradient for electronic optimization of the
            most recent JDFTx call (evaluated as dot(g, Kg), where g is the gradient and Kg is the preconditioned
            gradient). (written as "|grad|_K" in JDFTx output).
        elec_alpha (float) | None: The step size of the final electronic step in the most recent JDFTx call.
        elec_linmin (float | None): The final normalized projection of the electronic step direction onto the gradient
            for the most recent JDFTx call.
        charges (np.ndarray[float] | None): The most recent Lowdin-charges.
        magnetic_moments (np.ndarray[float] | None): The most recent Lowdin-magnetic moments.
        selective_dynamics (list[list[int]] | None): The selective dynamics flags for the most recent JDFTx call
            (three 0/1 values per atom).
        structure (Structure | None): Cleaned pymatgen Structure object of final JOutStructure
        initial_structure (Structure | None): Cleaned pymatgen Structure object of initial JOutStructure
    """

    out_slice_start_flag = "-------- Electronic minimization -----------"
    opt_type: str | None = None
    geom_converged: bool = False
    geom_converged_reason: str | None = None
    elec_converged: bool = False
    elec_converged_reason: str | None = None
    t_s: float | None = None
    slices: list[JOutStructure] = field(default_factory=list, init=True)
    eopt_type: str | None = None
    etype: str | None = None
    emin_flag: str | None = None
    ecomponents: list[str] | None = None
    elecmindata: JElSteps | None = None
    stress: NDArray[np.float64] | None = None
    strain: NDArray[np.float64] | None = None
    forces: NDArray[np.float64] | None = None
    nstep: int | None = None
    e: float | None = None
    grad_k: float | None = None
    alpha: float | None = None
    linmin: float | None = None
    nelectrons: float | None = None
    abs_magneticmoment: float | None = None
    tot_magneticmoment: float | None = None
    mu: float | None = None
    elec_nstep: int | None = None
    elec_e: float | None = None
    elec_grad_k: float | None = None
    elec_alpha: float | None = None
    elec_linmin: float | None = None
    charges: NDArray[np.float64] | None = None
    magnetic_moments: NDArray[np.float64] | None = None
    selective_dynamics: list[int] | None = None
    structure: Structure | None = None
    initial_structure: Structure | None = None

    @classmethod
    def _from_out_slice(
        cls,
        out_slice: list[str],
        opt_type: str = "IonicMinimize",
        eopt_type: str = "ElecMinimize",
        init_struc: Structure | None = None,
        is_md: bool = False,
        has_igp: bool = False,
        expected_etype: str | None = None,
        skim_levels: list[str] | None = None,
        skip_props: list[str] | None = None,
    ) -> JOutStructures:
        """
        Return JOutStructures object.

        Create a JOutStructures object from a slice of an out file's text
        corresponding to a single JDFTx call.

        Args:
            out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx).
            opt_type (str): The type of geometric optimization ("IonicMinimize", "LatticeMinimize" or
                "IonicDynamics").
            eopt_type (str): The type of electronic optimizer ("ElecMinimize" for electronic-minimize, "SCF" for
                electronic-scf).
            init_struc (Structure | None): The initial structure if available, otherwise None.
            is_md (bool): True if the calculation is a molecular dynamics (MD) simulation, False otherwise.
            has_igp (bool): Whether a ionic gaussian potential(s) are present.
            expected_etype (str | None): The expected type of energy from the electronic minimization data.
            skim_levels (list[str] | None): The levels to skim ("geom" keeps only the last geometric step, "elec"
                only the last electronic step of each).
            skip_props (list[str] | None): The properties to skip parsing. Options are "struct", "forces", and
                "charges".

        Returns:
            JOutStructures: The created JOutStructures object.
        """
        # Is this needed? I don't think any users are personally initializing this class.
        if opt_type not in ["IonicMinimize", "LatticeMinimize", "IonicDynamics"]:
            opt_type = correct_geom_opt_type(opt_type)
        start_idx = _get_joutstructures_start_idx(out_slice)
        slices = _get_joutstructure_list(
            out_slice[start_idx:],
            opt_type,
            eopt_type=eopt_type,
            init_structure=init_struc,
            is_md=is_md,
            has_igp=has_igp,
            expected_etype=expected_etype,
            skim_levels=skim_levels,
            skip_props=skip_props,
        )
        return cls(slices=slices)

    def __post_init__(self):
        # "self.slices[-1]" needs at least one slice: _get_joutstructure_list gives an empty JOutStructure for a slice
        # without geometric steps, and opt_type has a default value. (No slices at all, when every step fails to
        # parse, raises an IndexError here.)
        self.opt_type = self.slices[-1].opt_type
        if self.opt_type is None and len(self) > 1:
            raise Warning("iter type interpreted as single-point calculation, but multiple structures found")
        # TODO: Change this to fetch from `properties` or `site_properties` instead of a slices attributes.
        for var in _joss_atrs_from_last_slice:
            val = None
            for i in range(1, len(self.slices) + 1):
                if hasattr(self.slices[-i], var):
                    val = getattr(self.slices[-i], var)
                    if val is not None:
                        break
            setattr(self, var, val)
        self.initial_structure = self._get_initial_structure()
        self.t_s = self._get_t_s()
        self._check_convergence()

    def get_frame_property_list(self, prop: str) -> list[Any] | None:
        """Return list of named property.

        Get a property (attribute) from every JOutStructure in the list. If the property is not
        defined for a JOutStructure, its entry is None.

        Args:
            prop (str): The name of the property to get.

        Returns:
            list[Any]: The value of the property for each JOutStructure (None where it is not defined).
        """
        if len(self) == 0:
            return []
        return [getattr(s, prop, None) for s in self.slices]

    def _get_initial_structure(self) -> Structure | None:
        """Return initial structure.

        Returns:
            Structure | None: The initial structure.
        """
        if len(self):
            return self.slices[0].structure
        return None

    def _get_t_s(self) -> float | None:
        """Return time of calculation.

        Return time of calculation. Backup function to reference elecmindata for t_s for single point calculations.

        Returns:
            float: The total time in seconds for the calculation.
        """
        if self.t_s is not None:
            return self.t_s
        if len(self) and ((self.opt_type in ["single point", None]) and (isinstance(self.elecmindata.t_s, float))):
            return self.elecmindata.t_s
        return None

    def _check_convergence(self) -> None:
        """Set convergence flags.

        Check if the geometry and electronic density of last structure in the
        list has converged.
        """
        jst = self.slices[-1]
        if jst.elecmindata.converged:
            self.elec_converged = True
            self.elec_converged_reason = jst.elecmindata.converged_reason
        if jst.geom_converged:
            self.geom_converged = True
            self.geom_converged_reason = jst.geom_converged_reason

    def as_dict(self) -> dict:
        """
        Convert the JOutStructures object to a dictionary.

        Returns:
            dict: A dictionary representation of the JOutStructures object.
        """
        dct = {}
        for fld in self.__dataclass_fields__:
            if fld == "slices":
                dct[fld] = [slc.as_dict() for slc in self.slices]
                continue
            value = getattr(self, fld)
            if hasattr(value, "as_dict"):
                dct[fld] = value.as_dict()
            else:
                dct[fld] = value
        return dct

    def __getitem__(self, key: int | str) -> JOutStructure | Any:
        """Return item.

        Args:
            key (int | str): The key of the item.

        Returns:
            JOutStructure | Any: The value of the item.
        """
        val = None
        if type(key) is int:
            val = self._getitem_int(key)
        if type(key) is str:
            val = self._getitem_str(key)
        return val

    def _getitem_int(self, key: int) -> JOutStructure:
        """Return a JOutStructure object.

        Args:
            key (int): The index of the JOutStructure object.

        Returns:
            JOutStructure: The JOutStructure object at the key index.
        """
        return self.slices[key]

    def _getitem_str(self, key: str) -> Any:
        """Return attribute value.

        Args:
            key (str): The name of the attribute.

        Returns:
            Any: The value of the attribute.
        """
        return getattr(self, key)

    def __len__(self) -> int:
        """Return length of JOutStructures object.

        Returns:
            int: The number of geometric optimization steps in the JOutStructures object.
        """
        return len(self.slices)

    def __str__(self) -> str:
        """Return string representation.

        Returns:
            str: A string representation of the JOutStructures object.
        """
        return pprint.pformat(self)


_elec_min_start_flag: str = "-------- Electronic minimization -----------"
_geom_step_line_flags: tuple[str, ...] = ("IonicMinimize: Iter:", "LatticeMinimize: Iter:", "IonicDynamics: Step:")


def _is_geom_step_line(line_text: str) -> bool:
    return line_text.startswith(_geom_step_line_flags)


def _get_joutstructure_step_bounds(
    out_slice: list[str],
    out_slice_start_flag: str = _elec_min_start_flag,
) -> list[list[int]]:
    """Return list of boundary indices for each structure in out_slice.

    Args:
        out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx).
        out_slice_start_flag (str): The string that marks the beginning of the portion of an out file slice
            that contains data for a JOutStructures object.

    Returns:
        list[list[int]]: A list of lists of integers where each sublist contains the start and end
        of an individual optimization step (or SCF cycle if no optimization).
    """
    bounds_list = []
    bounds = None
    has_geom_step_line = False
    for i, line in enumerate(out_slice):
        if out_slice_start_flag in line:
            # A step that logged its geometry step line is ended by the start of the next one. Otherwise the open
            # bounds only hold an electronic minimization at a line-search trial position, which is discarded.
            # (The Lowdin block can't end a step, since JDFTx skips it when no pseudopotential has atomic orbitals.)
            if bounds is not None and has_geom_step_line:
                bounds.append(i)
                bounds_list.append(bounds)
            bounds = [i]
            has_geom_step_line = False
        elif bounds is not None and _is_geom_step_line(line):
            has_geom_step_line = True
    # The last step (or the step a prematurely interrupted jdftx call was on) runs to the end of the slice
    if bounds is not None:
        bounds.append(len(out_slice) - 1)
        bounds_list.append(bounds)
    return bounds_list


def _get_joutstructures_start_idx(
    out_slice: list[str],
    out_slice_start_flag: str = _elec_min_start_flag,
) -> int | None:
    """Return index of first line of first structure.

    Args:
        out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx).
        out_slice_start_flag (str): The string that marks the beginning of the portion of an out file slice
            that contains data for a JOutStructures object.

    Returns:
        int | None: The index of the first line of the first structure in the out_slice.
    """
    for i, line in enumerate(out_slice):
        if out_slice_start_flag in line:
            return i
    return None


def _get_joutstructure_list(
    out_slice: list[str],
    opt_type: str | None,
    eopt_type: str = "ElecMinimize",
    init_structure: Structure | None = None,
    is_md: bool = False,
    has_igp: bool = False,
    expected_etype: str | None = None,
    skim_levels: list[str] | None = None,
    skip_props: list[str] | None = None,
) -> list[JOutStructure]:
    """Return list of JOutStructure objects.

    Get list of JOutStructure objects by splitting out_slice into slices and constructing
    a JOutStructure object for each slice. Used in initialization.

    Args:
        out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx).
        opt_type (str | None): Flag string indicating optimization type.
        eopt_type (str): Flag string indicating the electronic optimizer type ("ElecMinimize" or "SCF").
        init_structure (Structure | None): The initial structure if available, otherwise None.
        is_md (bool): True if the calculation is a molecular dynamics (MD) simulation, False otherwise.

    Returns:
        list[JOutStructure]: The list of JOutStructure objects.
    """
    out_bounds = _get_joutstructure_step_bounds(out_slice)
    joutstructure_list: list[JOutStructure] = []
    if not len(out_bounds):
        # This case should only be called if no optimization steps are found to avoid errors down the line.
        # If this is changed to always be the first structure, logic down the line on what is considered
        # a "single point" calculation will be broken.
        joutstructure_list.append(
            JOutStructure._from_text_slice(
                [],
                init_structure=init_structure,
                opt_type=opt_type,
                eopt_type=eopt_type,
                is_md=is_md,
                has_igp=has_igp,
                expected_etype=expected_etype,
                skim_levels=skim_levels,
                skip_props=skip_props,
            )
        )
    if skim_levels is not None and "geom" in skim_levels:
        # Only require a new geometry if the slice prints one (vibrations runs bypass ionic minimization and don't)
        req_new_geom = any(_line_type_map["posns"] in line for line in out_slice)
        for bounds in out_bounds[::-1]:
            joutstructure = _parse_joutstructure_bounds(
                init_structure,
                out_slice,
                bounds,
                opt_type,
                is_md,
                has_igp,
                expected_etype,
                skim_levels,
                raise_on_error=False,
                skip_props=skip_props,
                req_new_geom=req_new_geom,
                eopt_type=eopt_type,
            )
            if joutstructure is not None:
                joutstructure_list.append(joutstructure)
                break
        return joutstructure_list
    for i, bounds in enumerate(out_bounds):
        if i > 0 and joutstructure_list:
            init_structure = joutstructure_list[-1]
        try:
            joutstructure = _parse_joutstructure_bounds(
                init_structure,
                out_slice,
                bounds,
                opt_type,
                is_md,
                has_igp,
                expected_etype,
                skim_levels,
                raise_on_error=True,
                skip_props=skip_props,
                eopt_type=eopt_type,
            )
        except (ValueError, IndexError, TypeError, KeyError, AttributeError) as err:
            # A step that fails to parse is skipped; only the last one (often cut short in a running calculation)
            # is skipped silently
            if i < len(out_bounds) - 1:
                warnings.warn(
                    f"Skipping geometry step {i} of {len(out_bounds)} (counting from 0), which could not be parsed "
                    f"({err!r}); the next step starts from the step before it",
                    stacklevel=2,
                )
            continue
        if joutstructure is not None:
            joutstructure_list.append(joutstructure)
    return joutstructure_list


def _parse_joutstructure_bounds(
    init_structure: Structure | None,
    out_slice: list[str],
    bounds: list[int],
    opt_type: str | None,
    is_md: bool,
    has_igp: bool,
    expected_etype: str | None,
    skim_levels: list[str] | None,
    raise_on_error: bool,
    skip_props: list[str] | None = None,
    # If we're skimming geometry level frequency, we need to set req_new_geom to True as otherwise
    # the JOutStructure will fall back to its
    # initial structure if no posn lines are found in the slice, which is bad as initial structure
    # is the input structure in this case,
    # not the structure from the previous geometry optimization step.
    req_new_geom: bool = False,
    eopt_type: str = "ElecMinimize",
) -> JOutStructure | None:
    joutstructure = None
    try:
        joutstructure = JOutStructure._from_text_slice(
            out_slice[bounds[0] : bounds[1]],
            init_structure=init_structure,
            opt_type=opt_type,
            eopt_type=eopt_type,
            is_md=is_md,
            has_igp=has_igp,
            expected_etype=expected_etype,
            skim_levels=skim_levels,
            skip_props=skip_props,
            req_new_geom=req_new_geom,
        )
    except (ValueError, IndexError, TypeError, KeyError, AttributeError):
        if raise_on_error:
            raise
    return joutstructure
