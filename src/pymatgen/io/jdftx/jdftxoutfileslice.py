"""JDFTx Outfile Slice Class.

This module defines the JDFTxOutfileSlice class, which is used to read and
process a JDFTx out file.
"""

from __future__ import annotations

import inspect
import math
import pprint
import re
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from pymatgen.io.jdftx.jelstep import JElSteps
from scipy.constants import Planck as h_j
from scipy.constants import e as e_charge
from scipy.constants import speed_of_light

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.core.trajectory import Trajectory
from pymatgen.core.units import Ha_to_eV, ang_to_bohr, bohr_to_ang
from pymatgen.io.jdftx._output_utils import (
    _dataclass_from_dict,
    _get_eigstats_varsdict,
    _json_dict,
    _none_to_nan,
    _OutputMSONable,
    _vibrational_modes_as_dict,
    find_all_key,
    find_first_range_key,
    find_key,
    find_key_first,
    get_colon_val,
    key_exists,
)
from pymatgen.io.jdftx.generic_tags import JDFTXOptionWarning, element_from_species_name
from pymatgen.io.jdftx.inputs import (
    JDFTX_SPECIES_SITE_PROPERTY,
    JDFTXInfile,
    clean_infile_of_nans,
    get_species_elements,
    get_species_names,
)
from pymatgen.io.jdftx.joutstructures import JOutStructures, _get_joutstructures_start_idx

__author__ = "Ben Rich"

_jofs_atr_from_jstrucs = (
    "structure",
    "eopt_type",
    "elecmindata",
    "stress",
    "strain",
    "forces",
    "nstep",
    "e",
    "grad_k",
    "alpha",
    "linmin",
    "abs_magneticmoment",
    "tot_magneticmoment",
    "elec_nstep",
    "elec_e",
    "elec_grad_k",
    "elec_alpha",
    "elec_linmin",
    "t_s",
    "pe",
    "ke",
    "t_k",
    "p_bar",
    "tmd_fs",
    "thermostat_velocity",
)


@dataclass
class JDFTXOutfileSlice(_OutputMSONable):
    """A class to read and process a slice of a JDFTx out file.

    A class to read and process a slice of a JDFTx out file, where a "slice" is a segment of an out file corresponding
    to a single call of JDFTx.

    Methods:
        from_out_slice(text: list[str]): Read slice of out file into a JDFTXOutfileSlice instance.

    Attributes:
        prefix (str | None): Prefix of dump files for JDFTx calculation.
        jstrucs (JOutStructures | None): JOutStructures instance containing intermediate structures. Holds a "slices"
            attribute, which is a list of JOutStructure instances. (A JOutStructure instance functions as a Structure
            object, along with a JElSteps instance (stored as elecmindata) and other JDFTx-calculation-specific data.)
        xc_func (str | None): Exchange-correlation functional used in the calculation.
        lattice_initial (np.ndarray | None): Initial lattice matrix in Angstroms.
        lattice_final (np.ndarray | None): Final lattice matrix in Angstroms.
        lattice (np.ndarray | None): Current lattice matrix in Angstroms.
        a (float | None): Lattice parameter a in Angstroms.
        b (float | None): Lattice parameter b in Angstroms.
        c (float | None): Lattice parameter c in Angstroms.
        fftgrid (list[int] | None): Shape of FFT grid used in calculation (3 integers).
        geom_opt (bool | None): True if geometric (lattice or ionic) optimization was performed.
        geom_opt_type (str | None): Type of geometric optimization performed (lattice or ionic, where lattice implies
            ionic as well unless geometry was given in direct coordinates).
        efermi (float | None): Fermi energy in eV (may be None if eigstats are not dumped).
        egap (float | None): Band gap in eV (None if eigstats are not dumped, or if JDFTx reports it unavailable,
            e.g. when there are no empty bands).
        emin (float | None): Minimum energy in eV (None if eigstats are not dumped).
        emax (float | None): Maximum energy in eV (None if eigstats are not dumped).
        homo (float | None): Energy of last band-state before Fermi level (acronym for Highest Occupied Molecular
            Orbital, even though these are not molecular orbitals and this state may not be entirely occupied)
            (None if eigstats are not dumped, or if JDFTx reports it unavailable).
        lumo (float | None): Energy of first band-state after Fermi level (acronym for Lowest Unoccupied Molecular
            Orbital, even though these are not molecular orbitals, and this state may not be entirely unoccupied)
            (None if eigstats are not dumped, or if JDFTx reports it unavailable, e.g. when there are no empty bands).
        homo_filling (float | None): Filling of "homo" band-state as calculated within this class object from the homo
            energy, Fermi level, electronic broadening type and electronic broadening parameter. (None if eigstats are
            not dumped, or if the homo energy or Fermi level needed with broadening is unavailable).
        lumo_filling (float | None): Filling of "lumo" band-state as calculated within this class object from the homo
            energy, Fermi level, electronic broadening type and electronic broadening parameter. (None if eigstats are
            not dumped, or if the lumo energy or Fermi level needed with broadening is unavailable).
        is_metal (bool | None): True if fillings of homo and lumo band-states are off-set by 1 and 0 by at least an
            arbitrary tolerance of 0.01 (ie 1 - 0.015 and 0.012 for homo/lumo fillings would be metallic, while 1-0.001
            and 0 would not be). (None unless eigstats were dumped and both fillings are available).
        etype (str | None): String representation of total energy-type of system. Commonly "G" (grand-canonical
            potential) for GC calculations, and "F" for canonical (fixed electron count) calculations.
        broadening_type (str): Type of broadening for electronic filling about Fermi-level requested. Either "Fermi",
            "Cold", "MP1", or "Gauss".
        broadening (float): Magnitude of broadening for electronic filling.
        kgrid (list[int]): Shape of k-point grid used in calculation. (equivalent to k-point folding).
        truncation_type (str): Type of coulomb truncation used to prevent interaction between periodic images along
            certain directions. "periodic" means no coulomb truncation was used.
        truncation_radius (float | None): If spherical truncation_type, this is the radius of the coulomb truncation
            sphere.
        pwcut (float): The plane-wave cutoff energy in Hartrees used in the most recent JDFTx call.
        rhocut (float): The density cutoff energy in Hartrees used in the most recent JDFTx call.
        pp_type (str): The pseudopotential library used in the most recent JDFTx call. Currently only "GBRV" and "SG15"
            are supported by this output parser.
        total_electrons (float): The total number of electrons in the most recent JDFTx call (redundant to nelectrons).
        semicore_electrons (int): The number of semicore electrons in the most recent JDFTx call.
        valence_electrons (float): The number of valence electrons in the most recent JDFTx call.
        total_electrons_uncharged (int): The total number of electrons in the most recent JDFTx call, uncorrected for
            charge. (ie total_electrons + charge).
        semicore_electrons_uncharged (int): The number of semicore electrons in the most recent JDFTx call, uncorrected
            for charge. (ie semicore_electrons + charge).
        valence_electrons_uncharged (int): The number of valence electrons in the most recent JDFTx call, uncorrected
            for charge. (ie valence_electrons + charge).
        nbands (int): The number of bands used in the most recent JDFTx call.
        atom_elements (list[str]): The list of each ion's element symbol in the most recent JDFTx call.
        atom_elements_int (list[int]): The list of ion's atomic numbers in the most recent JDFTx call.
        atom_types (list[str]): Non-repeating list of the ions' species names in the most recent JDFTx call. These
            are element symbols, unless a species is named after its pseudopotential file (e.g. Fe1 for Fe1.upf).
        spintype (str): The spin type used in the most recent JDFTx call. Options are "none", "collinear".
        nspin (int): The number of spins used in the most recent JDFTx call.
        nat (int): The number of atoms in the most recent JDFTx call.
        atom_coords_initial (list[list[float]]): The initial atomic coordinates of the most recent JDFTx call.
        atom_coords_final (list[list[float]]): The final atomic coordinates of the most recent JDFTx call.
        atom_coords (list[list[float]]): The atomic coordinates of the most recent JDFTx call.
        has_solvation (bool): True if the most recent JDFTx call included a solvation calculation.
        fluid (str): The fluid used in the most recent JDFTx call.
        is_gc (bool): True if the most recent slice is a grand canonical calculation.
        is_bgw (bool): True if data must be usable for a BerkeleyGW calculation (user-set).
        has_eigstats (bool): True if eigstats were dumped in the most recent JDFTx call.
        has_parsable_pseudo (bool): True if the most recent JDFTx call used a pseudopotential that can be parsed by this
            output parser. Options are currently "GBRV" and "SG15".

    Properties:
        t_s (float | None): The total time in seconds for the calculation.
        converged (bool | None): True if calculation converged.
        trajectory (Trajectory): pymatgen Trajectory object containing intermediate Structure's of outfile slice
            calculation.
        electronic_output (dict): Dictionary with all relevant electronic information dumped from an eigstats log.
        structure (Structure): Calculation result as pymatgen Structure.
        initial_structure (Structure): Initial structure of the calculation, as read from inputs portion of out file.
        eopt_type (str | None): eopt_type from most recent JOutStructure.
        elecmindata (JElSteps): elecmindata from most recent JOutStructure.
        stress (np.ndarray | None): Stress tensor from most recent JOutStructure in units eV/Ang^3.
        strain (np.ndarray | None): Strain tensor from most recent JOutStructure (unitless).
        thermostat_velocity (np.ndarray | None): Nose-Hoover chain thermostat velocities from most recent JOutStructure
            (one per chain element, chainLengthT in ionic-dynamics), in JDFTx units (inverse Hartree atomic units of
            time) so they can be written back unchanged as the thermostat-velocity tag of a restart input. None if not
            printed.
        nstep (int | None): (geometric) nstep from most recent JOutStructure.
        e (float | None): Energy of system "etype" from most recent JOutStructure.
        grad_k (float): The final norm of the preconditioned gradient for geometric optimization of the most recent
            JDFTx call (evaluated as dot(g, Kg), where g is the gradient and Kg is the preconditioned gradient).
            (written as "|grad|_K" in JDFTx output).
        alpha (float): The step size of the final geometric step in the most recent JDFTx call.
        linmin (float): The final normalized projection of the geometric step direction onto the gradient for the most
            recent JDFTx call.
        abs_magneticmoment (float | None): The absolute magnetic moment of the most recent JDFTx call.
        tot_magneticmoment (float | None): The total magnetic moment of the most recent JDFTx call.
        mu (float): The Fermi energy of the most recent JDFTx call.
        elec_e (float): The final energy of the most recent electronic optimization step.
        elec_nstep (int): The number of electronic optimization steps in the most recent JDFTx call.
        elec_grad_k (float): The final norm of the preconditioned gradient for electronic optimization of the most
            recent JDFTx call (evaluated as dot(g, Kg), where g is the gradient and Kg is the preconditioned gradient).
            (written as "|grad|_K" in JDFTx output).
        elec_alpha (float): The step size of the final electronic step in the most recent JDFTx call.
        elec_linmin (float): The final normalized projection of the electronic step direction onto the gradient for the
            most recent JDFTx call.

    Magic Methods:
        __getattr__(name: str) -> Any: Overwrite of default __getattr__ method to allow for reference of un-defined
            attributes to the "jstrucs" class field. This referring behavior is ideally never used as (currently) all
            referrable attributes are defined properties, but is included to prevent errors in the case of future
            changes.
        __str__() -> str: Return a string representation of the class instance using pprint module.
        __repr__() -> str: Create string representation of the class instance. Overwritten from default behavior for
            dataclass so that properties are included in the string, and verbose attributes with redundant information
            are trimmed.
    """

    prefix: str | None = None

    jstrucs: JOutStructures | None = None
    constant_lattice: bool | None = None
    is_md: bool = False

    xc_func: str | None = None

    lattice_initial: np.ndarray | None = None
    lattice_final: np.ndarray | None = None
    lattice: np.ndarray | None = None
    a: float | None = None
    b: float | None = None
    c: float | None = None

    fftgrid: list[int] | None = None
    geom_opt: bool | None = None
    geom_opt_type: str | None = None

    # grouping fields related to electronic parameters.
    # Used by the get_electronic_output() method
    _electronic_output: ClassVar[list[str]] = [
        "efermi",
        "egap",
        "optical_egap",
        "emin",
        "emax",
        "homo",
        "lumo",
        "homo_filling",
        "lumo_filling",
        "is_metal",
    ]
    efermi: float | None = None
    egap: float | None = None
    optical_egap: float | None = None
    emin: float | None = None
    emax: float | None = None
    homo: float | None = None
    lumo: float | None = None
    homo_filling: float | None = None
    lumo_filling: float | None = None
    is_metal: bool | None = None
    etype: str | None = None

    broadening_type: str | None = None
    broadening: float | None = None
    kgrid: list | None = None
    truncation_type: str | None = None
    truncation_radius: float | None = None
    pwcut: float | None = None
    rhocut: float | None = None

    pp_type: str | None = None
    semicore_electrons: int | None = None
    valence_electrons: float | None = None
    total_electrons_uncharged: int | None = None
    semicore_electrons_uncharged: int | None = None
    valence_electrons_uncharged: int | None = None
    nbands: int | None = None

    atom_elements: list | None = None
    atom_elements_int: list | None = None
    atom_types: list | None = None
    spintype: str | None = None
    nspin: int | None = None
    nat: int | None = None
    atom_coords_initial: np.ndarray | None = None
    atom_coords_final: np.ndarray | None = None
    atom_coords: np.ndarray | None = None

    has_solvation: bool = False
    fluid: str | None = None
    is_gc: bool | None = None
    is_bgw: bool = False
    has_eigstats: bool = False
    parsable_pseudos: ClassVar[list[str]] = ["GBRV", "SG15"]
    has_parsable_pseudo: bool = False

    _total_electrons_backup: float | None = None
    total_electrons: float | None = None
    _mu_backup: float | None = None

    t_s: float | None = None
    converged: bool | None = None
    structure: Structure | None = None
    initial_structure: Structure | None = None
    _trajectory: Trajectory | None = None
    electronic_output: dict | None = None
    eopt_type: str | None = None
    elecmindata: JElSteps | None = None
    stress: np.ndarray | None = None
    strain: np.ndarray | None = None
    forces: np.ndarray | None = None
    nstep: int | None = None
    e: float | None = None
    grad_k: float | None = None
    alpha: float | None = None
    linmin: float | None = None
    abs_magneticmoment: float | None = None
    tot_magneticmoment: float | None = None
    mu: float | None = None
    elec_nstep: int | None = None
    elec_e: float | None = None
    elec_grad_k: float | None = None
    elec_alpha: float | None = None
    elec_linmin: float | None = None
    vibrational_modes: list[dict[str, Any]] | None = None
    vibrational_energy_components: dict[str, float] | None = None
    # Set while parsing; fields so that as_dict keeps them
    infile: JDFTXInfile = field(default_factory=JDFTXInfile)
    geom_opt_label: str = "IonicMinimize"
    ecomponents: dict | None = None
    pe: float | None = None
    ke: float | None = None
    t_k: float | None = None
    p_bar: float | None = None
    tmd_fs: float | None = None
    thermostat_velocity: np.ndarray | None = None

    def _get_mu(self) -> float | None:
        """Sets mu from most recent JOutStructure. (Equivalent to efermi)"""
        _mu = None
        if self.jstrucs is not None:
            _mu = self.jstrucs.mu
        if _mu is None:
            _mu = self._mu_backup
        return _mu

    ###########################################################################
    # Creation methods
    ###########################################################################

    # TODO: There are a littany of points in a JDFTx out file slice where an unexpected termination
    # (due to a partial calculation) could lead to a fatal error in the parser. As a fix for this, this
    # method contains a try-except block enabled by `none_on_error`. In the long term though
    # all the subobjects should be able to handle partial initialization while returning as much
    # information as possible.
    @classmethod
    def _from_out_slice(
        cls,
        text: list[str],
        skim_levels: list[str],
        skip_props: list[str],
        is_bgw: bool = False,
        none_on_error: bool = True,
    ) -> JDFTXOutfileSlice | None:
        """
        Read slice of out file into a JDFTXOutfileSlice instance.

        Args:
            text (list[str]): File to read.
            is_bgw (bool): True if data must be usable for a BerkeleyGW calculation.
            none_on_error (bool): If True, return None if an error occurs. If False, raise the error.

        Returns:
            JDFTXOutfileSlice | None: An instance of JDFTXOutfileSlice or None if an error occurs and
            none_on_error is True.
        """
        instance = cls()
        instance.is_bgw = is_bgw
        try:
            instance._from_out_slice_init_all(text, skim_levels, skip_props)
        except (ValueError, IndexError, TypeError, KeyError, AttributeError):
            if none_on_error:
                return None
            raise
        return instance

    def _from_out_slice_init_all(self, text: list[str], skim_levels: list[str], skip_props: list[str]) -> None:
        self._set_internal_infile(text)
        self._set_geomopt_vars(text)
        self._set_jstrucs(text, skim_levels=skim_levels, skip_props=skip_props)
        self._set_backup_vars(text)
        self.prefix = self._get_prefix(text)
        spintype, nspin = self._get_spinvars(text)
        self.xc_func = self._get_xc_func(text)
        self.spintype = spintype
        self.nspin = nspin
        broadening_type, broadening = self._get_broadeningvars(text)
        self.broadening_type = broadening_type
        self.broadening = broadening
        self.kgrid = self._get_kgrid(text)
        truncation_type, truncation_radius = self._get_truncationvars(text)
        self.truncation_type = truncation_type
        self.truncation_radius = truncation_radius
        self.pwcut = self._get_pw_cutoff(text)
        self.rhocut = self._get_rho_cutoff(text)
        self.fftgrid = self._get_fftgrid(text)
        self._set_eigvars(text)
        self._set_orb_fillings()
        self.is_metal = self.determine_is_metal()
        self._set_fluid(text)
        self._set_nbands(text)
        self._set_atom_vars(text)
        self._set_total_electrons()
        self._set_pseudo_vars(text)
        self._set_lattice_vars(text)
        self.has_solvation = self._check_solvation()

        # @ Cooper added @#
        self.is_gc = key_exists("target-mu", text)
        self._set_ecomponents(text)

        # Previously were properties, but are now set as attributes
        self._from_out_slice_init_all_post_init()
        if "vibrations" in self.infile:
            self._read_vibrational_data(text)

    def _set_internal_infile(self, text: list[str]) -> None:
        """Set the internal infile for the JDFTXOutfileSlice.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        start_line_idx = find_key("Input parsed successfully", text)
        if start_line_idx is None:
            raise ValueError("JDFTx input parsing failed on most recent call.")
        start_line_idx += 2
        end_line_idx = None
        for i in range(start_line_idx, len(text)):
            # Sometimes the dumped in file will have a blank line in the middle, so we actually need to
            # find two consecutive blank lines to determine the end of the infile. (hopefully)
            if (not len(text[i].strip())) and (not len(text[i + 1].strip())):
                end_line_idx = i
                break
        if end_line_idx is None:
            raise ValueError("Calculation did not begin for this out file slice.")
        with warnings.catch_warnings():  # Warnings on the choice of options are for whoever sets them, not readers
            warnings.simplefilter("ignore", JDFTXOptionWarning)
            self.infile = JDFTXInfile.from_str(
                "\n".join(text[start_line_idx:end_line_idx]), validate_value_boundaries=False, skip_invalid_tags=True
            )
        self.constant_lattice = True
        if "lattice-minimize" in self.infile:
            latsteps = self.infile["lattice-minimize"]["nIterations"]
            self.constant_lattice = not (int(latsteps) > 0)
        if "ionic-dynamics" in self.infile:
            self.is_md = int(self.infile["ionic-dynamics"]["nSteps"]) > 0

    def _set_t_s(self) -> None:
        """Return the total time in seconds for the calculation.

        Returns:
            float: The total time in seconds for the calculation.
        """
        _t_s = None
        if self.jstrucs:
            _t_s = self.jstrucs.t_s
        self.t_s = _t_s

    def _set_converged(self) -> None:
        """Return True if calculation converged.

        Returns:
            bool: True if the electronic and geometric optimization have converged (or only the former if a single-point
            calculation).
        """
        if self.jstrucs is None:
            return
        converged = self.jstrucs.elec_converged
        if self.geom_opt:
            converged = converged and self.jstrucs.geom_converged
        self.converged = converged

    def _set_trajectory(self) -> None:
        """Return pymatgen trajectory object.

        Returns:
            Trajectory: pymatgen Trajectory object containing intermediate Structure's of outfile slice calculation.
        """
        if self.jstrucs is not None:
            structures = [slc.structure for slc in self.jstrucs]
            constant_lattice = self.constant_lattice if self.constant_lattice is not None else False
            frame_properties = [slc.properties for slc in self.jstrucs]
            self._trajectory = Trajectory.from_structures(
                structures=structures,
                constant_lattice=constant_lattice,
                frame_properties=frame_properties,
            )

    @property
    def trajectory(self) -> Trajectory | None:
        """Return pymatgen trajectory object.

        Returns:
            Trajectory: pymatgen Trajectory object containing intermediate Structure's of outfile slice calculation.
        """
        if self._trajectory is None:
            self._set_trajectory()
        return self._trajectory

    def _set_electronic_output(self) -> None:
        """Return a dictionary with all relevant electronic information.

        Returns:
            dict: Dictionary with values corresponding to these keys in _electronic_output field.
        """
        dct = {}
        for key in self._electronic_output:
            if key in self.__dataclass_fields__:
                value = getattr(self, key)
                dct[key] = value
        self.electronic_output = dct

    def _from_out_slice_init_all_post_init(self) -> None:
        """Post init for running at end of "_from_out_slice_init_all" method.

        Sets class variables previously defined as properties.
        """
        self._set_t_s()
        self._set_converged()
        self._set_electronic_output()

    def _get_xc_func(self, text: list[str]) -> str | None:
        """Get the exchange-correlation functional used in the calculation.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            str: Exchange-correlation functional used.
        """
        line = find_key("elec-ex-corr", text)
        if line is None:
            return None
        return text[line].strip().split()[-1].strip()

    def _get_prefix(self, text: list[str]) -> str | None:
        """
        Get output prefix from the out file.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            str: Prefix of dump files for JDFTx calculation.
        """
        line = find_key("dump-name", text)
        if line is None:
            return None
        dumpname = text[line].split()[1]
        return dumpname.split(".")[0] if "." in dumpname else dumpname

    def _get_spinvars(self, text: list[str]) -> tuple[str, int]:
        """
        Set spintype and nspin from out file text for instance.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            tuple:
            spintype (str): Type of spin in calculation.
            nspin (int): Number of spin types in calculation.
        """
        line = find_key("spintype ", text)
        if line is None:
            return "no-spin", 1
        spintype = text[line].split()[1]
        # vector-spin and spin-orbit have one set of (spinor) states, like no-spin
        nspin = 2 if spintype == "z-spin" else 1
        return spintype, nspin

    def _get_broadeningvars(self, text: list[str]) -> tuple[str | None, float]:
        """Get broadening type and value from out file text.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            tuple[str, float]: Broadening type and parameter for electronic smearing.
        """
        line = find_key("elec-smearing ", text)
        if line is not None:
            broadening_type = text[line].split()[1]
            broadening = float(text[line].split()[2])
        else:
            broadening_type = None
            broadening = 0
        return broadening_type, broadening

    def _get_truncationvars(self, text: list[str]) -> tuple[str | None, Any | None]:
        """Get truncation type and value from out file text.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            tuple[str, float] | tuple[None, None]: Truncation type and radius of truncation
            (if truncation_type is spherical).
        """
        maptypes = {
            "Periodic": "periodic",
            "Slab": "slab",
            "Cylindrical": "wire",
            "Wire": "wire",
            "Spherical": "spherical",
            "Isolated": "box",
        }
        line = find_key("coulomb-interaction", text)
        if line is None:
            return None, None
        truncation_type = None
        truncation_radius = None
        if line is not None:
            truncation_type = text[line].split()[1]
            if truncation_type not in maptypes:
                raise ValueError("Problem with this truncation!")
            truncation_type = maptypes[truncation_type]
            direc = None
            if len(text[line].split()) == 3:
                direc = text[line].split()[2]
            if self.is_bgw:
                if truncation_type == "slab" and direc != "001":
                    raise ValueError("BGW slab Coulomb truncation must be along z!")
                if truncation_type == "wire" and direc != "001":
                    raise ValueError("BGW wire Coulomb truncation must be periodic in z!")
            if truncation_type == "spherical":
                line = find_key("Initialized spherical truncation of radius", text)
                if line is None:
                    raise ValueError("No spherical truncation found in out file.")
                truncation_radius = float(text[line].split()[5]) / ang_to_bohr
        else:
            raise ValueError("No truncation type found in out file.")
        return truncation_type, truncation_radius

    def _get_pw_cutoff(self, text: list[str]) -> float | None:
        """Get the electron cutoff from the out file text.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            float | None: Plane wave cutoff used in calculation.
        """
        line = find_key("elec-cutoff ", text)
        if line is None:
            return None
        return float(text[line].split()[1]) * Ha_to_eV

    def _get_rho_cutoff(self, text: list[str]) -> float | None:
        """Get the electron cutoff from the out file text.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            float: Electron density cutoff used in calculation.
        """
        line = find_key("elec-cutoff ", text)
        if line is None:
            return None
        lsplit = text[line].split()
        if len(lsplit) == 3:
            rhocut = float(lsplit[2]) * Ha_to_eV
        else:
            if self.pwcut is None:
                self.pwcut = self._get_pw_cutoff(text)
            rhocut = None if self.pwcut is None else float(self.pwcut * 4)
        return rhocut

    def _get_fftgrid(self, text: list[str]) -> list[int] | None:
        """Get the FFT grid from the out file text.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            list[int]: FFT grid used in calculation.
        """
        line = find_key_first("Chosen fftbox size", text)
        if line is None:
            return None
        return [int(x) for x in text[line].split()[6:9]]

    def _get_kgrid(self, text: list[str]) -> list[int] | None:
        """Get the kpoint grid from the out file text.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            list[int]: Kpoint grid used in calculation.
        """
        line = find_key("kpoint-folding ", text)
        if line is None:
            return None
        return [int(x) for x in text[line].split()[1:4]]

    def _get_eigstats_varsdict(self, text: list[str]) -> dict[str, float | None]:
        """Get the eigenvalue statistics from the out file text.

        Args:
            text (list[str]): Output of read_file for out file.
            prefix (str): Prefix for the eigStats section in the out file.

        Returns:
            dict[str, float | None]: Dictionary of eigenvalue statistics.
        """
        has_eigstats, varsdict = _get_eigstats_varsdict(text)
        self.has_eigstats = has_eigstats
        return varsdict

    def _set_eigvars(self, text: list[str]) -> None:
        """Set the eigenvalue statistics variables.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        eigstats = self._get_eigstats_varsdict(text)
        for key, val in eigstats.items():
            setattr(self, key, val)
        if self.efermi is None:
            if self.mu is None:
                self.mu = self._get_mu()
            self.efermi = self.mu

    def _get_pp_type(self, text: list[str]) -> str | None:
        """Get the pseudopotential type used in calculation.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            str: Pseudopotential library used. Returns None if not GBRV or SG15
            (pseudopotentials parsable by this parser).
        """
        skey = "Reading pseudopotential file"
        line = find_key(skey, text)
        if line is None:
            raise ValueError("Unable to find pseudopotential dump lines")
        ppfile_example = text[line].split(skey)[1].split(":")[0].strip("'").strip()
        pptype = None
        readable = self.parsable_pseudos
        for _pptype in readable:
            if _pptype in ppfile_example:
                if pptype is not None:
                    if ppfile_example.index(pptype) < ppfile_example.index(_pptype):
                        pptype = _pptype
                    else:
                        pass
                else:
                    pptype = _pptype
        if pptype is not None:
            self.has_parsable_pseudo = True
        return pptype

    def _set_pseudo_vars(self, text: list[str]) -> None:
        """Set the pseudopotential variables.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        self.pp_type = self._get_pp_type(text)
        if self.has_parsable_pseudo and self.pp_type in ["GBRV", "SG15"]:
            self._set_pseudo_vars_t1(text)
        # Otherwise variables requiring parsing pseudopotential output will be kept as None
        self._check_species_elements(text)

    def _check_species_elements(self, text: list[str]) -> None:
        """Warn if a pseudopotential file is for another element than the one taken from its species' name.

        JDFTx doesn't print a species' element, so pymatgen takes it from the species name (see
        element_from_species_name). UPF files (e.g. SG15) state theirs, which JDFTx prints as
        "'<element>' pseudopotential". JDFTx reads the pseudopotential files in the order of its species, the order of
        atom_types.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        section_starts = find_all_key("Reading pseudopotential file", text)
        if self.atom_types is None or len(section_starts) != len(self.atom_types):
            return
        mismatches = []
        for start, species in zip(section_starts, self.atom_types, strict=True):
            for line in text[start + 1 : start + 3]:
                if (match := re.match(r"\s*'(\w+)' pseudopotential", line)) is not None:
                    element = element_from_species_name(species)
                    if element is not None and match.group(1).lower() != element.lower():
                        mismatches.append(f"{species} (read as {element}, file for {match.group(1)})")
                    break
        if mismatches:
            warnings.warn(
                "pymatgen takes a species' element from its name, but these species' pseudopotential files are for "
                f"other elements: {', '.join(mismatches)}",
                stacklevel=2,
            )

    def _set_pseudo_vars_t1(self, text: list[str]) -> None:
        """Set the pseudopotential variables for SG15 and GBRV pseudopotentials.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        all_val_lines = find_all_key("valence electrons", text)
        atom_total_elec = []
        bounds_list = get_pseudo_read_section_bounds(text)
        for bounds in bounds_list:
            startline = bounds[0]
            endline = bounds[1]
            val_lines = [x for x in all_val_lines if x < endline and x > startline]
            val_line = val_lines[0]
            val_elec = int(text[val_line].split("valence electrons")[0].strip().split()[-1])
            atom_total_elec.append(val_elec)
        total_elec_dict = {}
        if self.atom_types is not None:
            for i, atom in enumerate(self.atom_types):
                total_elec_dict[atom] = atom_total_elec[i]
        else:
            raise ValueError("Pseudopotential data cannot be allocated without atom types.")
        if self.atom_elements is None:
            raise ValueError("Atom elements not set yet.")
        # Each species has its own pseudopotential (two species of one element can differ, e.g. in semicore states)
        atom_species = [ion["species-id"] for ion in self.infile["ion"]] if "ion" in self.infile else self.atom_elements
        element_total_electrons = np.array([total_elec_dict[x] for x in atom_species])
        pmg_elements = [Element(x) for x in self.atom_elements]
        element_valence_electrons = np.array(
            [
                np.sum(np.array([int(v[2:]) for v in el.electronic_structure.split(".") if "]" not in v]))
                for el in pmg_elements
            ]
        )
        element_semicore_electrons = element_total_electrons - element_valence_electrons
        self.total_electrons_uncharged = sum(element_total_electrons)
        self.valence_electrons_uncharged = sum(element_valence_electrons)
        self.semicore_electrons_uncharged = sum(element_semicore_electrons)
        self.semicore_electrons = self.semicore_electrons_uncharged
        if (self.total_electrons is not None) and (self.semicore_electrons is not None):
            self.valence_electrons = self.total_electrons - self.semicore_electrons  # accounts for if system is charged
        else:
            raise ValueError("Total electrons and semicore electrons must be set.")

    def _set_geomopt_vars(self, text: list[str]) -> None:
        """Set the geom_opt and geom_opt_type class variables.

        Set vars geom_opt and geom_opt_type for initializing self.jstrucs.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        found = False
        if "ionic-dynamics" in self.infile and int(self.infile["ionic-dynamics"]["nSteps"]) > 0:
            self.geom_opt = True
            self.geom_opt_type = "ionic"
            self.geom_opt_label = "IonicDynamics"
            found = True
        if (not found and "lattice-minimize" in self.infile) and int(
            self.infile["lattice-minimize"]["nIterations"] > 0
        ):
            self.geom_opt = True
            self.geom_opt_label = "LatticeMinimize"
            self.geom_opt_type = "lattice"
            found = True
        if (not found and "ionic-minimize" in self.infile) and int(self.infile["ionic-minimize"]["nIterations"] > 0):
            self.geom_opt = True
            self.geom_opt_label = "IonicMinimize"
            self.geom_opt_type = "ionic"
            found = True
        if not found:
            self.geom_opt = False
            self.geom_opt_type = "single point"
            self.geom_opt_label = "IonicMinimize"

    def _get_initial_structure(self, text: list[str]) -> Structure | None:
        """Get the initial structure from the out file text.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            Structure | None: Initial structure of the calculation.
        """
        output_start_idx = _get_joutstructures_start_idx(text)
        if output_start_idx is None:
            init_struc = _get_init_structure(text)
        else:
            init_struc = _get_init_structure(text[:output_start_idx])
        if init_struc is None:
            raise ValueError("Provided out file slice's inputs preamble does not contain input structure data.")
        return init_struc

    def _set_jstrucs(self, text: list[str], skim_levels: list[str], skip_props: list[str]) -> None:
        """Set the jstrucs class variable.

        Set the JStructures object to jstrucs from the out file text and all class attributes initialized from jstrucs.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        self.initial_structure = self.infile.to_pmg_structure(self.infile)
        # In the case where no ion optimization updates are printed, JOutStructures
        if self.geom_opt_type is None:
            raise ValueError("geom_opt_type not set yet.")
        # The energy JDFTx minimizes, as relevantFreeEnergyName (electronic/Energies.cpp) names it from the input:
        # only elec-smearing makes the fillings vary, and target-mu requires it
        if "fix-electron-density" in self.infile or "fix-electron-potential" in self.infile:
            expected_etype = "Eband"
        elif "elec-smearing" not in self.infile:
            expected_etype = "Etot"
        else:
            expected_etype = "G" if "target-mu" in self.infile else "F"
        self.jstrucs = JOutStructures._from_out_slice(
            text,
            opt_type=self.geom_opt_label,
            # JDFTx optimizes the electrons by SCF instead of the default variational minimizer if electronic-scf is set
            eopt_type="SCF" if "electronic-scf" in self.infile else "ElecMinimize",
            init_struc=self.initial_structure,
            is_md=self.is_md,
            has_igp="ionic-gaussian-potential" in self.infile,
            expected_etype=expected_etype,
            skim_levels=skim_levels,
            skip_props=skip_props,
        )
        if self.jstrucs is not None:
            if self.etype is None:
                self.etype = self.jstrucs[-1].etype
            self.mu = self._get_mu()
            for var in _jofs_atr_from_jstrucs:
                setattr(self, var, getattr(self.jstrucs, var))

    def _set_backup_vars(self, text: list[str]) -> None:
        """Set backups for important variables.

        Set backup versions of critical variables if missing from constructed jstrucs (that can be easily fetched
        through type-casting strings).

        Args:
            text (list[str]): Output of read_file for out file.
        """
        if self.total_electrons is None:
            lines = find_all_key("nElectrons", text)
            _val = None
            for line in lines[::-1]:
                _val = get_colon_val(text[line], "nElectrons:")
                if _val is not None:
                    break
            if _val is not None and np.isnan(_val):
                nval = None
            elif _val is not None:
                nval = float(_val)
            else:
                nval = None
            self._total_electrons_backup = nval

        if self.mu is None:
            lines = find_all_key("mu", text)
            _val = None
            for line in lines[::-1]:
                _val = get_colon_val(text[line], "mu:")
                if _val is not None:
                    break
            if _val is not None and np.isnan(_val):
                mval = None
            elif _val is not None:
                mval = float(_val) * Ha_to_eV
            else:
                mval = None
            self._mu_backup = mval

    def _set_orb_fillings_nobroad(self, nspin: float) -> None:
        """Set the orbital fillings without broadening.

        Args:
            nspin (float): Number of spins in calculation.
        """
        self.homo_filling = 2 / nspin
        self.lumo_filling = 0

    def _set_orb_fillings_broad(
        self,
        nspin: float,
        ehomo: float | None,
        elumo: float | None,
        efermi: float | None,
        broadening_type: str,
        broadening: float,
    ) -> None:
        """Set the orbital fillings with broadening.

        A filling is set to None if an energy it needs is None (JDFTx reports e.g. the LUMO as unavailable when there
        are no empty bands).

        Args:
            nspin (float): Number of spins in calculation.
            ehomo (float | None): Energy of the highest occupied molecular orbital.
            elumo (float | None): Energy of the lowest unoccupied molecular orbital.
            efermi (float | None): Fermi energy.
            broadening_type (str): Type of broadening.
            broadening (float): Broadening parameter.
        """
        self.homo_filling, self.lumo_filling = (
            None
            if eig is None or efermi is None
            else (2 / nspin) * self._calculate_filling(broadening_type, broadening, eig, efermi)
            for eig in (ehomo, elumo)
        )

    def _set_orb_fillings(self) -> None:
        """Set the orbital fillings.

        Calculate and set homo and lumo fillings.
        """
        if self.has_eigstats:
            if self.nspin is not None:
                if self.broadening_type is not None:
                    if self.broadening is not None:
                        # homo, lumo and efermi may be None (unavailable), leaving the fillings that need them None
                        self._set_orb_fillings_broad(
                            self._get_filling_nspin(self.nspin),
                            self.homo,
                            self.lumo,
                            self.efermi,
                            self.broadening_type,
                            self.broadening,
                        )
                    else:
                        raise ValueError("Cannot set orbital fillings with broadening with self.broadening as None")
                else:
                    self._set_orb_fillings_nobroad(self._get_filling_nspin(self.nspin))
            else:
                raise ValueError("Cannot set homo/lumo filling with self.nspin as None")

    def _get_filling_nspin(self, nspin: int) -> int:
        """Return the nspin for which 2 / nspin is the number of electrons a band holds (one for a spinor band)."""
        return 2 if self.spintype in ("vector-spin", "spin-orbit") else nspin

    def _set_fluid(self, text: list[str]) -> None:
        """Set the fluid class variable.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        line = find_first_range_key("fluid ", text)
        val = text[line[0]].split()[1]
        self.fluid = None if val.lower() == "none" else val

    def _set_total_electrons(self) -> None:
        """Sets total_electrons from most recent JOutStructure."""
        tot_elec = None
        if self.jstrucs is not None:
            _tot_elec = self.jstrucs.nelectrons
            if _tot_elec is not None:
                tot_elec = _tot_elec
        if (tot_elec is None) and (self._total_electrons_backup is not None):
            tot_elec = self._total_electrons_backup
        self.total_electrons = tot_elec

    def _set_nbands(self, text: list[str]) -> None:
        """Set the Nbands class variable.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        # JDFTx always prints the number of bands it uses in the "nBands:" line. The echoed "elec-n-bands <n>
        # [<multiplier>]" command is only a fallback, since <n> = 0 means a (multiplied) default number of bands.
        lines = find_all_key("nBands:", text)
        if len(lines):
            self.nbands = int(text[lines[0]].split("nBands:")[1].split()[0])
            return
        lines = find_all_key("elec-n-bands", text)
        if len(lines):
            nbands = int(text[lines[0]].split()[1])
            if nbands > 0:
                self.nbands = nbands
                return
        warnings.warn("Could not determine the number of bands from the out file.", stacklevel=2)

    def _set_atom_vars(self, text: list[str]) -> None:
        """Set the atom variables.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        startline = find_key("Input parsed successfully", text)
        if startline is None:
            raise ValueError("Unable to find start line for atom lines")
        endline = find_key("---------- Initializing the Grid ----------", text)
        if endline is None:
            raise ValueError("Unable to find end line for atom lines")
        lines = find_first_range_key("ion ", text, startline=startline, endline=endline)
        species_names = [text[x].split()[1] for x in lines]
        atom_elements, _ = get_species_elements(species_names)
        self.nat = len(atom_elements)
        atom_coords = [text[x].split()[2:5] for x in lines]
        self.atom_coords_initial = np.array(atom_coords, dtype=float)
        atom_types = []
        for x in species_names:
            if x not in atom_types:
                atom_types.append(x)
        self.atom_elements = atom_elements
        self.atom_elements_int = [Element(x).Z for x in self.atom_elements]
        self.atom_types = atom_types
        if isinstance(self.structure, Structure):
            self.atom_coords = self.structure.cart_coords
            self.atom_coords_final = self.structure.cart_coords

    def _set_lattice_vars(self, text: list[str]) -> None:
        """Set the lattice variables.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        if self.jstrucs is not None:
            self.lattice_initial = self.jstrucs[0].lattice.matrix
            self.lattice_final = self.jstrucs[-1].lattice.matrix
            self.lattice = self.jstrucs[-1].lattice.matrix.copy()
            self.a, self.b, self.c = np.sum(self.jstrucs[-1].lattice.matrix ** 2, axis=1) ** 0.5
        else:
            raise ValueError("No structures found in out file.")

    def _set_ecomponents(self, text: list[str]) -> None:
        """
        Set the energy components dictionary from the out file text.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        if self.jstrucs is not None:
            ecomp = self.jstrucs[-1].ecomponents
            if not isinstance(ecomp, dict):
                ecomp = {}
            if self.etype not in ecomp:
                if self.etype is None:
                    raise ValueError("etype not set yet.")
                ecomp[self.etype] = self.jstrucs[-1].e
            self.ecomponents = ecomp
        else:
            raise ValueError("No structures found in out file.")

    def _calculate_filling(self, broadening_type: str, broadening: float, eig: float, efermi: float) -> float:
        """
        Calculate the filling for a given eigenvalue.

        Use the broadening type, broadening value, eigenvalue, and fermi energy to calculate the filling at the
        eigenvalue.

        Args:
            broadening_type (str): Type of broadening to use.
            broadening (float): Broadening parameter.
            eig (float): Eigenvalue.
            efermi (float): Fermi energy.

        Returns:
            float: Filling at the eigenvalue.
        """
        # most broadening implementations do not have the denominator factor
        # of 2, but JDFTx does currently.
        x = (eig - efermi) / (2.0 * broadening)
        if broadening_type == "Fermi":
            filling = 0.5 * (1 - np.tanh(x))
        elif broadening_type == "Gauss":
            filling = 0.5 * (1 - math.erf(x))
        elif broadening_type == "MP1":
            filling = 0.5 * (1 - math.erf(x)) - x * np.exp(-1 * x**2) / (2 * np.pi**0.5)
        elif broadening_type == "Cold":
            filling = 0.5 * (1 - math.erf(x + 0.5**0.5)) + np.exp(-1 * (x + 0.5**0.5) ** 2) / (2 * np.pi) ** 0.5
        else:
            raise NotImplementedError("Have not added other broadening types")
        return filling

    def _determine_is_metal(self, tol_partial: float, nspin: int, homo_filling: float, lumo_filling: float) -> bool:
        """Return boolean for whether system is metallic.

        Return boolean for whether system is metallic. True if difference in filling in homo and lumo states
        exceed 0 and 1 by a given tol_partial parameter.

        Returns:
            bool: True if system is metallic
        """
        return not (homo_filling / (2 / nspin) > (1 - tol_partial) and lumo_filling / (2 / nspin) < tol_partial)

    def determine_is_metal(self) -> bool | None:
        """Determine if the system is a metal based on the fillings of homo and lumo.

        Returns:
            bool | None: True if system is metallic, None if eigstats were not dumped or a filling is unavailable.
        """
        tol_partial = 0.01
        if self.has_eigstats:
            if self.nspin is not None:
                # A filling is None if an energy it needs was unavailable, so metallicity can't be determined
                if self.homo_filling is None or self.lumo_filling is None:
                    return None
                return self._determine_is_metal(
                    tol_partial, self._get_filling_nspin(self.nspin), self.homo_filling, self.lumo_filling
                )
            raise ValueError("Cannot determine if system is metal - self.nspin undefined")
        return None

    def _read_vibrational_data(self, text: list[str]) -> None:
        """Read vibrational data from the output file.

        Args:
            text (list[str]): Output of read_file for out file.
        """
        # TODO: Once t`he total number of computed configurations can be determined, a lot of the
        # out file can be trimmed by setting the start line as the instance of "Completed n of n configurations"
        mode_start_lines = (
            find_all_key("Imaginary mode ", text) + find_all_key("Zero mode ", text) + find_all_key("Real mode ", text)
        )
        mode_dicts = []
        for start_line in mode_start_lines:
            end_line = start_line + 1
            while end_line < len(text) and text[end_line].strip():
                end_line += 1
            mode_dicts.append(self._parse_vibrational_mode_lines(text[start_line:end_line]))
        self.vibrational_modes = mode_dicts
        vib_nrg_components_start_line = find_all_key("Vibrational free energy components", text)[-1]
        end_line = vib_nrg_components_start_line + 1
        while end_line < len(text) and text[end_line].strip():
            end_line += 1
        vib_nrg_lines = text[vib_nrg_components_start_line:end_line]
        self.vibrational_energy_components = self._parse_vibrational_energy_components_lines(vib_nrg_lines)

    def _parse_vibrational_energy_components_lines(self, text: list[str]) -> dict:
        """Parse vibrational energy components from the output file.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            dict: Dictionary containing vibrational energy components.
        """
        vib_nrg_components = {}
        if "T" in text[0]:
            vib_nrg_components["T"] = float(text[0].split("=")[1].split("K")[0].strip())
        for line in text[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                vib_nrg_components[key.strip()] = float(value.strip()) * Ha_to_eV
        return vib_nrg_components

    def _parse_vibrational_mode_lines(self, text: list[str]) -> dict:
        """Parse vibrational mode lines from the output file.

        Args:
            text (list[str]): Output of read_file for out file.

        Returns:
            dict: Dictionary containing vibrational mode information.
        """
        vib_mode_data: dict[str, float | int | complex | str | None] = {}
        disp_start_idx = None
        for i, line in enumerate(text):
            if "Displacements:" in line:
                disp_start_idx = i + 1
                break
            key = line.split(":")[0].strip()
            value = line.split(":")[1].strip()
            entry_val: float | int | complex | str | None = None
            if key in ["Frequency", "IR intensity"]:
                value = value.split()[0].strip()
                entry_val = float(value.rstrip("i")) * 1j if value.endswith("i") else float(value)
                if key == "Frequency":
                    entry_val *= Ha_to_eV
            elif key == "Degeneracy":
                entry_val = int(value.split()[-1].strip())
            elif "mode" in key:
                vib_mode_data["Type"] = key.split()[0].strip()  # "Imaginary", "Zero", or "Real"
                vib_mode_data["Type index"] = int(key.split()[2].strip())  # ie the third real mode
            if entry_val is not None:
                vib_mode_data[key] = entry_val
        if disp_start_idx is not None:
            _displacements = []
            for line in text[disp_start_idx:]:
                if not line.strip():
                    break
                _displacements.append(np.array([float(x) for x in line.split()[2:5]]))
            vib_mode_data["Displacements"] = np.array(_displacements) * bohr_to_ang  # Length of displacement
            # specified in inputs
        vib_mode_data["cm^-1"] = vib_mode_data["Frequency"] / (100 * speed_of_light * (h_j / e_charge))
        return vib_mode_data

    def to_jdftxinfile(self) -> JDFTXInfile:
        """
        Convert the JDFTXOutfile object to a JDFTXInfile object with the most recent structure.
        If the input structure is desired, simply fetch JDFTXOutfile.infile

        Returns:
            JDFTXInfile: A JDFTXInfile object representing the input parameters of the JDFTXOutfile.
        """
        # Use internal infile as a reference for calculation parameters
        with warnings.catch_warnings():  # Warnings on the choice of options were for whoever set them
            warnings.simplefilter("ignore", JDFTXOptionWarning)
            # Copy without the values JDFTx echoes as nan when unset (e.g. ionic-dynamics P0), which it can't read
            base_infile = clean_infile_of_nans(self.infile)
        if self.structure is None:
            base_infile.strip_structure_tags()
            return base_infile
        # Optimization restrictions and velocities are carried over the structure site properties. Ions are written in
        # the input's coordinate system, so they match the positions JDFTx printed
        in_cart_coords = base_infile.get("coords-type", "Lattice").lower() == "cartesian"
        add_infile = JDFTXInfile.from_structure(self.structure, write_cart_coords=in_cart_coords)
        printed_ions = self._get_printed_ions(in_cart_coords)
        if printed_ions is not None:
            add_infile["ion"] = printed_ions
        if self.initial_structure is not None and np.array_equal(
            self.structure.lattice.matrix, self.initial_structure.lattice.matrix
        ):
            # Keep the input's own lattice and latt-scale tags for an unchanged lattice, so the lattice stays exact
            del add_infile["lattice"]
            for tag in ("ion", "coords-type"):
                if tag in base_infile:
                    del base_infile[tag]
        else:
            base_infile.strip_structure_tags()
        infile = base_infile + add_infile
        # Set most recent thermostat velocity if MD with Nose-Hoover (none is printed for some runs)
        if self.is_md and self.infile["ionic-dynamics"]["statMethod"] == "NoseHoover":
            tv = self.jstrucs.thermostat_velocity
            if tv is not None:
                infile["thermostat-velocity"] = {"v1": tv[0], "v2": tv[1], "v3": tv[2]}
        return infile

    def _get_printed_ions(self, in_cart_coords: bool) -> list[dict] | None:
        """Get the final ion tags exactly as JDFTx printed them, if they still describe the final structure.

        Converting positions to pymatgen units and back can change them in the last digit, so a restart input uses the
        values JDFTx printed (as if they were copied into a new input file) unless the structure has been changed.

        Args:
            in_cart_coords (bool): Whether the ion tags are written in Cartesian coordinates.

        Returns:
            list[dict] | None: Ion tags as printed by JDFTx, or None if they can't be used.
        """
        jstruc = self.jstrucs.slices[-1] if self.jstrucs is not None and self.jstrucs.slices else None
        lines = getattr(jstruc, "_printed_ion_lines", None)
        if self.structure is None or not lines:
            return None
        if jstruc._printed_coords_type != ("cartesian" if in_cart_coords else "lattice"):
            return None
        try:
            ions = JDFTXInfile.from_str("\n".join(lines), dont_require_structure=True, validate_value_boundaries=False)
        except ValueError as err:
            warnings.warn(
                f"Could not read the printed ion positions, using the parsed structure instead: {err}", stacklevel=2
            )
            return None
        ion_tags = ions["ion"]
        positions = np.array([[ion[f"x{k}"] for k in range(3)] for ion in ion_tags])
        expected = self.structure.cart_coords / bohr_to_ang if in_cart_coords else self.structure.frac_coords
        species_names = [name.lower() for name in get_species_names(self.structure)]  # JDFTx reads them ignoring case
        if (
            [ion["species-id"].lower() for ion in ion_tags] != species_names
            or positions.shape != expected.shape
            or not np.allclose(positions, expected, rtol=0, atol=1e-8)
        ):
            return None  # The structure has been changed since it was parsed
        return ion_tags

    def _check_solvation(self) -> bool:
        """Check for implicit solvation.

        Returns:
            bool: True if calculation used implicit solvation.
        """
        return self.fluid is not None

    def write(self) -> None:
        """Return an error.

        Raises:
            NotImplementedError: There is no need to write a JDFTx out file.
        """
        raise NotImplementedError("There is no need to write a JDFTx out file")

    def as_dict(self) -> dict:
        """Convert dataclass to dictionary representation.

        Returns:
            dict: JDFTXOutfileSlice in dictionary format.
        """
        dct: dict[str, Any] = {"@module": self.__class__.__module__, "@class": self.__class__.__name__}
        for fld in self.__dataclass_fields__:
            if fld == "_trajectory":  # A cache the trajectory property fills on first read
                continue
            value = (
                _vibrational_modes_as_dict(self.vibrational_modes) if fld == "vibrational_modes" else getattr(self, fld)
            )
            if hasattr(value, "as_dict"):
                dct[fld] = value.as_dict()
            else:
                dct[fld] = value
        return _json_dict(dct)

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> JDFTXOutfileSlice:
        """Return a JDFTXOutfileSlice from the dictionary as_dict gives."""
        instance = _dataclass_from_dict(cls, {key: value for key, value in dct.items() if key != "infile"})
        # The echoed input has JDFTx's placeholders for unset values, which fail JDFTXInfile's bounds checks, and
        # warnings on the choice of options were for whoever set them (as when parsing the out file)
        if dct.get("infile") is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", JDFTXOptionWarning)
                instance.infile = JDFTXInfile.from_dict(_none_to_nan(dct["infile"]), validate_value_boundaries=False)
        if instance.vibrational_modes is not None:  # Imaginary frequencies were stored as [real, imag]
            instance.vibrational_modes = [
                {key: complex(*v) if isinstance(v, list) else v for key, v in mode.items()}
                for mode in instance.vibrational_modes
            ]
        return instance

    # TODO: Re-do this now that there are no properties
    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: String representation of the JDFTXOutfileSlice.
        """
        out_str = f"{self.__class__.__name__}("
        for cls in inspect.getmro(self.__class__):
            for key, value in cls.__dict__.items():
                if not key.startswith("_") and (not callable(value) or isinstance(value, property)):
                    suff = ", \n"
                    val = repr(getattr(self, key))
                    if key == "jstrucs":
                        val = "(... JOutStructures object ...)"
                    out_str += f"{key}={val}{suff}"
        out_str += ")"
        return out_str

    def __str__(self) -> str:
        """Return string representation.

        Returns:
            str: String representation of the JDFTXOutfileSlice.
        """
        return pprint.pformat(self)


eigstats_keymap = {
    "eMin": "emin",
    "HOMO": "homo",
    "mu": "efermi",
    "LUMO": "lumo",
    "eMax": "emax",
    "HOMO-LUMO gap": "egap",
    "Optical gap": "optical_egap",
}


def get_pseudo_read_section_bounds(text: list[str]) -> list[list[int]]:
    """Get the boundary line numbers for the pseudopotential read section.

    Args:
        text (list[str]): Output of read_file for out file.

    Returns:
        list[list[int]]: List of line numbers for the pseudopotential read sections.
    """
    start_lines = find_all_key("Reading pseudopotential file", text)
    section_bounds = []
    for start_line in start_lines:
        bounds = [start_line]
        for i in range(start_line, len(text)):
            if not len(text[i].strip()):
                bounds.append(i)
                break
        section_bounds.append(bounds)
    return section_bounds


def _get_init_structure(pre_out_slice: list[str]) -> Structure | None:
    """
    Return initial structure.

    Return the initial structure from the pre_out_slice, corresponding to all data cut from JOutStructure list
    initialization. This is needed to ensure structural data that is not being updated (and therefore not being
    logged in the out file) is still available.

    Args:
        pre_out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx) that
        contains the initial structure information.

    Returns:
        Structure | None: The initial structure if available, otherwise None.
    """
    try:
        lat_mat = _get_initial_lattice(pre_out_slice)
        coords = _get_initial_coords(pre_out_slice)
        elements, species_names = get_species_elements(_get_initial_species(pre_out_slice))
        structure = Structure(lattice=lat_mat, species=elements, coords=coords, coords_are_cartesian=True)
        if species_names is not None:  # Species not named after their element (e.g. Fe1)
            structure.add_site_property(JDFTX_SPECIES_SITE_PROPERTY, species_names)
        return structure
    except AttributeError:
        return None


def _get_initial_lattice(pre_out_slice: list[str]) -> np.ndarray:
    """Return initial lattice.

    Return the initial lattice from the pre_out_slice, corresponding to all data cut from JOutStructure list
    initialization. This is needed to ensure lattice data that is not being updated (and therefore not being
    logged in the out file) is still available.

    Args:
        pre_out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx) that
            contains the initial lattice information.

    Returns:
        np.ndarray: The initial lattice matrix.
    """
    lat_lines = find_first_range_key("lattice  ", pre_out_slice)
    if len(lat_lines):
        lat_line = lat_lines[0]
        lat_mat = np.zeros([3, 3])
        for i in range(3):
            line_text = pre_out_slice[lat_line + i + 1].strip().split()
            for j in range(3):
                lat_mat[i, j] = float(line_text[j])
        return lat_mat.T * bohr_to_ang
    raise AttributeError("Lattice not found in pre_out_slice")


def _get_initial_coords(pre_out_slice: list[str]) -> np.ndarray:
    """Return initial coordinates.

    Return the initial coordinates from the pre_out_slice, corresponding to all data cut from JOutStructure list
    initialization. This is needed to ensure coordinate data that is not being updated (and therefore not being
    logged in the out file) is still available.

    Args:
        pre_out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx) that
            contains the initial coordinates information.

    Returns:
        np.ndarray: The initial coordinates.
    """
    lines = _get_ion_lines(pre_out_slice)
    coords = np.zeros([len(lines), 3])
    for i, line in enumerate(lines):
        line_text = pre_out_slice[line].strip().split()[2:]
        for j in range(3):
            coords[i, j] = float(line_text[j])
    coords_type_lines = find_first_range_key("coords-type", pre_out_slice)
    if len(coords_type_lines):
        coords_type = pre_out_slice[coords_type_lines[0]].strip().split()[1]
        if coords_type.lower() != "cartesian":
            coords = np.dot(coords, _get_initial_lattice(pre_out_slice))
        else:
            coords *= bohr_to_ang
    return coords


def _get_initial_species(pre_out_slice: list[str]) -> list[str]:
    """Return initial species.

    Return the initial species from the pre_out_slice, corresponding to all data cut from JOutStructure list
    initialization. This is needed to ensure species data that is not being updated (and therefore not being
    logged in the out file) is still available.

    Args:
        pre_out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx) that
            contains the initial species information.

    Returns:
        list[str]: The initial species.
    """
    lines = _get_ion_lines(pre_out_slice)

    return [pre_out_slice[line].strip().split()[1] for line in lines]


def _get_ion_lines(pre_out_slice: list[str]) -> list[int]:
    """Return ion lines.

    Return the ion lines from the pre_out_slice, ensuring that all the ion lines are consecutive.

    Args:
        pre_out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx) that
            contains the ion lines information.

    Returns:
        list[int]: The ion lines.
    """
    _lines = find_first_range_key("ion ", pre_out_slice)
    if not len(_lines):
        raise AttributeError("Ion lines not found in pre_out_slice")
    gaps = [_lines[i + 1] - _lines[i] for i in range(len(_lines) - 1)]
    if not all(g == 1 for g in gaps):
        # TODO: Write the fix for this case
        raise AttributeError("Ion lines not consecutive in pre_out_slice")
    return _lines
