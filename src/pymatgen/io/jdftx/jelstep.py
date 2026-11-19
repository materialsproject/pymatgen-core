"""Module for parsing SCF steps from JDFTx.

This module contains the JElStep class for parsing a single SCF step from a JDFTx out file, and the JElSteps class
for the series of SCF steps of one electronic optimization.
"""

from __future__ import annotations

import pprint
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


from pymatgen.core.units import Ha_to_eV
from pymatgen.io.jdftx._output_utils import get_colon_val

__author__ = "Ben Rich"

# Keyword JDFTx logs after "<opt_type>: " on each step of an electronic optimizer: electronic-minimize logs
# "ElecMinimize: Iter: <n> ..." and electronic-scf logs "SCF: Cycle: <n> ..." (core/Pulay.h)
_eopt_step_keys = {"ElecMinimize": "Iter", "SCF": "Cycle"}


def _get_iter_flag(opt_type: str) -> str:
    """Return the flag that starts each step line of an electronic optimizer (e.g. "SCF: Cycle:").

    Args:
        opt_type (str): The type of electronic optimizer ("ElecMinimize" or "SCF").

    Returns:
        str: The flag that starts each step line.
    """
    return f"{opt_type}: {_eopt_step_keys.get(opt_type, 'Iter')}:"


@dataclass
class JElStep:
    """Electronic minimization data for a single SCF step.

    Class object for storing logged electronic minimization data for a single
    SCF step.

    Attributes:
        opt_type (str | None): The type of electronic optimizer ("ElecMinimize" for electronic-minimize, "SCF" for
            electronic-scf).
        etype (str | None): The type of energy component (G, F, Etot or Eband).
        nstep (int | None): The SCF step number.
        e (float | None): The total electronic energy in eV.
        grad_k (float | None): The norm of the preconditioned gradient of the energy (written as "|grad|_K" in JDFTx
            output). None for SCF steps.
        alpha (float | None): The step length. None for SCF steps.
        linmin (float | None): Normalized line minimization direction / energy
            gradient projection (-1 for perfectly opposite, 1 for perfectly aligned). None for SCF steps.
        t_s (float | None): Time elapsed from beginning of JDFTx calculation.
        mu (float | None): The chemical potential in eV.
        nelectrons (float | None): The number of electrons.
        abs_magneticmoment (float | None): The absolute magnetic moment.
        tot_magneticmoment (float | None): The total magnetic moment.
        subspacerotationadjust (float | None): The subspace rotation adjustment factor.
    """

    opt_type: str | None = None
    etype: str | None = None
    nstep: int | None = None
    e: float | None = None
    grad_k: float | np.float64 | None = None
    alpha: float | np.float64 | None = None
    linmin: float | np.float64 | None = None
    t_s: float | np.float64 | None = None
    mu: float | np.float64 | None = None
    nelectrons: float | np.float64 | None = None
    abs_magneticmoment: float | np.float64 | None = None
    tot_magneticmoment: float | np.float64 | None = None
    subspacerotationadjust: float | np.float64 | None = None
    converged: bool = False
    converged_reason: str | None = None

    @classmethod
    def _from_lines_collect(cls, lines_collect: list[str], opt_type: str, etype: str) -> JElStep:
        """Return JElStep object.

        Create a JElStep object from a list of lines of text from a JDFTx out
        file corresponding to a single SCF step.

        Args:
        lines_collect (list[str]): A list of lines of text from a JDFTx out file corresponding to a single SCF step.
        opt_type (str): The type of electronic minimization step.
        etype (str): The type of energy component.

        Returns:
            JElStep: The created JElStep object.
        """
        instance = cls()
        instance.opt_type = opt_type
        instance.etype = etype
        _iter_flag = _get_iter_flag(opt_type)
        for i, line_text in enumerate(lines_collect):
            if instance._is_iter_line(i, line_text, _iter_flag):
                instance._read_iter_line(line_text)
            elif instance._is_fillings_line(i, line_text):
                instance._read_fillings_line(line_text)
            elif instance._is_subspaceadjust_line(i, line_text):
                instance._read_subspaceadjust_line(line_text)
        return instance

    def _is_iter_line(self, i: int, line_text: str, _iter_flag: str) -> bool:
        """Return True if opt iter line.

        Return True if the line_text is the start of a log message for a
        JDFTx optimization step.

        Args:
            i (int): The index of the line in the text slice.
            line_text (str): A line of text from a JDFTx out file.
            _iter_flag (str): The flag that indicates the start of a log message for a JDFTx optimization step.

        Returns:
            bool: True if the line_text is the start of a log message for a JDFTx optimization step.
        """
        return _iter_flag in line_text

    def _read_iter_line(self, line_text: str) -> None:
        """Set attributes nstep, e, grad_k, alpha, linmin, t_s.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Args:
            line_text (str): A line of text from a JDFTx out file containing the electronic minimization data.
        """
        ########
        # Only read what follows "<opt_type>: ", since that prefix can itself look like a value label ("SCF: " ends
        # in "F: ")
        step_key = _eopt_step_keys.get(str(self.opt_type), "Iter")
        line_text = line_text.split(f"{self.opt_type}: ", maxsplit=1)[-1]
        # Quickly fail so JOutStructure can retry with the energy label the lines do have
        if f"{self.etype}: " not in line_text:
            raise RuntimeError(f"Could not find {self.etype} in line_text")
        nstep_float = get_colon_val(line_text, f"{step_key}: ")
        if isinstance(nstep_float, float):
            self.nstep = int(nstep_float)
        elif nstep_float is None:
            raise ValueError("Could not find nstep in line_text")
        self.e = get_colon_val(line_text, f"{self.etype}: ") * Ha_to_eV
        self.grad_k = get_colon_val(line_text, "|grad|_K: ")
        self.alpha = get_colon_val(line_text, "alpha: ")
        self.linmin = get_colon_val(line_text, "linmin: ")
        self.t_s = get_colon_val(line_text, "t[s]: ")

    def _is_fillings_line(self, i: int, line_text: str) -> bool:
        """Return True if fillings line.

        Return True if the line_text is a FillingsUpdate log message (chemical potential, number of electrons and
        magnetic moments of an SCF step).

        Args:
            i (int): The index of the line in the text slice.
            line_text (str): A line of text from a JDFTx out file.

        Returns:
            bool: True if the line_text is a FillingsUpdate log message.
        """
        return "FillingsUpdate" in line_text

    def _read_fillings_line(self, fillings_line: str) -> None:
        """Set class variables mu, nelectrons, magneticmoment.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Args:
            fillings_line (str): A line of text from a JDFTx out file containing the electronic
            minimization data.
        """
        if "FillingsUpdate:" in fillings_line:
            self._set_mu(fillings_line)
            self._set_nelectrons(fillings_line)
            if "magneticMoment" in fillings_line:
                self._set_magdata(fillings_line)
        else:
            raise ValueError("FillingsUpdate string not found")

    def _is_subspaceadjust_line(self, i: int, line_text: str) -> bool:
        """Return True if the line_text is a SubspaceRotationAdjust log message.

        Args:
            i (int): The index of the line in the text slice.
            line_text (str): A line of text from a JDFTx out file.

        Returns:
            bool: True if the line_text is a SubspaceRotationAdjust log message.
        """
        return "SubspaceRotationAdjust" in line_text

    def _read_subspaceadjust_line(self, line_text: str) -> None:
        """Set attribute subspacerotationadjust.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Args:
            line_text (str): A line of text from a JDFTx out file containing the electronic
            minimization data.
        """
        self.subspacerotationadjust = get_colon_val(line_text, "SubspaceRotationAdjust: set factor to")

    def _set_magdata(self, fillings_line: str) -> None:
        """Set attributes abs_magneticmoment, tot_magneticmoment.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Args:
            fillings_line (str): A line of text from a JDFTx out file containing the electronic
            minimization data.
        """
        _fillings_line = fillings_line.split("magneticMoment: [ ")[1].split(" ]", maxsplit=1)[0].strip()
        self.abs_magneticmoment = get_colon_val(_fillings_line, "Abs: ")
        self.tot_magneticmoment = get_colon_val(_fillings_line, "Tot: ")

    def _set_mu(self, fillings_line: str) -> None:
        """Set mu class variable.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Args:
            fillings_line (str): A line of text from a JDFTx out file containing the electronic
            minimization data.
        """
        self.mu = get_colon_val(fillings_line, "mu: ") * Ha_to_eV

    def _set_nelectrons(self, fillings_line: str) -> None:
        """Set nelectrons class variable.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Args:
            fillings_line(str): A line of text from a JDFTx out file containing the electronic minimization data
        """
        self.nelectrons = get_colon_val(fillings_line, "nElectrons: ")

    def as_dict(self) -> dict:
        """Return dictionary representation of JElStep object.

        Returns:
            dict: Dictionary representation of JElStep object.
        """
        dct = {}
        for fld in self.__dataclass_fields__:
            value = getattr(self, fld)
            if hasattr(value, "as_dict"):
                dct[fld] = value.as_dict()
            else:
                dct[fld] = value
        return dct

    def __str__(self) -> str:
        """
        Return string representation of JElStep object.

        Returns:
            str: String representation of JElStep object.
        """
        return pprint.pformat(self)


_jelsteps_atrs_from_last_slice = [
    "e",
    "grad_k",
    "alpha",
    "linmin",
    "t_s",
    "mu",
    "nelectrons",
    "abs_magneticmoment",
    "tot_magneticmoment",
    "subspacerotationadjust",
]


@dataclass
class JElSteps:
    """Class object for series of SCF steps.

    Class object for collecting and storing a series of SCF steps done between
    geometric optimization steps.

    Attributes:
        opt_type (str | None): The type of electronic minimization step.
        etype (str | None): The type of energy component.
        iter_flag (str | None): The flag that indicates the start of a log message for a JDFTx optimization step.
        converged (bool): True if the SCF steps converged.
        converged_reason (str | None): The reason for convergence.
        slices (list[JElStep]): A list of JElStep objects.
        e (float | None): The total electronic energy in eV.
        grad_k (float | None): The norm of the preconditioned gradient of the energy (written as "|grad|_K" in JDFTx
            output).
        alpha (float | None): The step length.
        linmin (float | None): Normalized line minimization direction / energy
            gradient projection (-1 for perfectly opposite, 1 for perfectly aligned).
        t_s (float | None): Time elapsed from beginning of JDFTx calculation.
        mu (float | None): The chemical potential in eV.
        nelectrons (float | None): The number of electrons.
        abs_magneticmoment (float | None): The absolute magnetic moment.
        tot_magneticmoment (float | None): The total magnetic moment.
        subspacerotationadjust (float | None): The subspace rotation adjustment factor.
        nstep (int | None): The SCF step number.
    """

    opt_type: str | None = None
    etype: str | None = None
    iter_flag: str | None = None
    converged: bool | None = field(default=None, init=True)
    converged_reason: str | None = field(default=None, init=True)
    slices: list[JElStep] = field(default_factory=list, init=True)
    e: float | None = field(default=None, init=False)
    grad_k: float | None = field(default=None, init=False)
    alpha: float | None = field(default=None, init=False)
    linmin: float | None = field(default=None, init=False)
    t_s: float | None = field(default=None, init=False)
    mu: float | None = field(default=None, init=False)
    nelectrons: float | None = field(default=None, init=False)
    abs_magneticmoment: float | None = field(default=None, init=False)
    tot_magneticmoment: float | None = field(default=None, init=False)
    subspacerotationadjust: float | None = field(default=None, init=False)
    nstep: int | None = field(default=None, init=False)

    def _get_nstep(self) -> int | None:
        """Return the nstep attribute of the last JElStep object in the slices.

        The nstep attribute signifies the SCF step number.

        Returns:
            int: The nstep attribute of the last JElStep object in the slices, or the index of the last JElStep
            object if nstep is None.

        Raises:
            AttributeError: If there are no JElStep objects in the slices.
        """
        if len(self.slices):
            if self.slices[-1].nstep is not None:
                nstep = self.slices[-1].nstep
            else:
                warnings.warn(
                    "No nstep attribute in JElStep object. Returning number of JElStep objects.", stacklevel=2
                )
                nstep = len(self.slices) - 1
            return nstep
        raise AttributeError("No JElStep objects in JElSteps object slices class variable.")

    @classmethod
    def _from_text_slice(
        cls,
        text_slice: list[str],
        opt_type: str = "ElecMinimize",
        etype: str = "F",
        skim_levels: list[str] | None = None,
    ) -> JElSteps:
        """Return JElSteps object.

        Create a JElSteps object from a slice of an out file's text
        corresponding to a series of SCF steps.

        Args:
            text_slice (list[str]): A slice of text from a JDFTx out file corresponding to a series of SCF steps.
            opt_type (str): The type of electronic minimization step.
            etype (str): The type of energy component.
            skim_levels (list[str] | None): If it includes "elec", only the last SCF step that parses is kept.
        """
        line_collections, lines_collect = _gather_JElSteps_line_collections(opt_type, text_slice)
        slices = []
        if skim_levels is not None and "elec" in skim_levels:
            eslice = None
            for _lines_collect in line_collections[::-1]:
                try:
                    eslice = JElStep._from_lines_collect(_lines_collect, opt_type, etype)
                except (ValueError, IndexError, TypeError, KeyError, AttributeError):
                    pass  # Do not pass on RunTime error - needed for JOutStructure to retry with another energy label
                if eslice is not None:
                    slices.append(eslice)
                    break
        else:
            slices = [
                JElStep._from_lines_collect(_lines_collect, opt_type, etype) for _lines_collect in line_collections
            ]
        converged = None
        converged_reason = None
        if len(lines_collect):
            converged, converged_reason = _parse_ending_lines(lines_collect, opt_type)
        instance = cls(slices=slices, converged=converged, converged_reason=converged_reason)
        instance.opt_type = opt_type
        instance.etype = etype
        return instance

    @classmethod
    def _from_nothing(cls, opt_type: str = "ElecMinimize", etype: str = "F") -> JElSteps:
        """Return JElSteps object.

        Create an empty JElSteps object.

        Args:
            opt_type (str): The type of electronic minimization step.
            etype (str): The type of energy component.
        """
        return cls(slices=[], converged=False, converged_reason=None, opt_type=opt_type, etype=etype)

    def __post_init__(self) -> None:
        """Post initialization method."""
        if len(self.slices):
            self.nstep = self._get_nstep()
            for var in _jelsteps_atrs_from_last_slice:
                val = None
                for i in range(1, len(self.slices) + 1):
                    val = getattr(self.slices[-i], var)
                    if val is not None:
                        break
                setattr(self, var, val)
        else:
            self.nstep = 0
            for var in _jelsteps_atrs_from_last_slice:
                setattr(self, var, None)

    def as_dict(self) -> dict[str, Any]:
        """Return dictionary representation of JElSteps object.

        Returns:
            dict: Dictionary representation of JElSteps object.
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

    def __getitem__(self, key: int | str) -> JElStep | Any:
        """Return item.

        Return the value of an item.

        Parameters
        ----------
        key: int | str
            The key of the item

        Returns
        -------
        val
            The value of the item
        """
        val = None
        if type(key) is int:
            val = self._getitem_int(key)
        if type(key) is str:
            val = self._getitem_str(key)
        return val

    def _getitem_int(self, key: int) -> JElStep:
        """Return JElStep object.

        Return the JElStep object at the key index.

        Parameters
        ----------
        key: int
            The index of the JElStep object

        Returns
        -------
        JElStep: JElStep
            The JElStep object at the key index
        """
        return self.slices[key]

    def _getitem_str(self, key: str) -> Any:
        """Return attribute value.

        Return the value of an attribute.

        Parameters
        ----------
        key: str
            The name of the attribute

        Returns
        -------
        value
            The value of the attribute
        """
        return getattr(self, key)

    def __len__(self) -> int:
        """Return length of JElSteps object.

        Returns the number of SCF steps in the JElSteps object.

        Returns
        -------
        length: int
            The number of SCF steps in the JElSteps object
        """
        return len(self.slices)

    def __str__(self) -> str:
        """Return string representation of JElSteps object.

        Returns
        -------
        str: str
            String representation of JElSteps object
        """
        return pprint.pformat(self)


def _gather_JElSteps_line_collections(opt_type: str, text_slice: list[str]) -> tuple[list[list[str]], list[str]]:
    """Gather line collections for JElSteps initialization.

    Gathers list of line lists where each line list initializes a JElStep object,
    and the remaining lines that do not initialize a JElStep object are used
    for initialization unique to the JElSteps object.

    Args:
        opt_type (str): The type of electronic minimization step.
        text_slice (list[str]): A slice of text from a JDFTx out file corresponding to a series of SCF steps.

    Returns:
        tuple: A tuple containing:
            line_collections (list[list[str]]): A list of lists of lines of text from a JDFTx out file
            corresponding to a single SCF step.
            lines_collect (list[str]): The lines of text after the last SCF step (the ending lines, which say whether
            the series converged).
    """
    lines_collect = []
    line_collections = []
    _iter_flag = _get_iter_flag(opt_type)
    for line_text in text_slice:
        if len(line_text.strip()):
            lines_collect.append(line_text)
            if _iter_flag in line_text:
                line_collections.append(lines_collect)
                lines_collect = []
        else:
            break
    return line_collections, lines_collect


def _parse_ending_lines(ending_lines: list[str], opt_type: str) -> tuple[bool | None, str | None]:
    """Parse ending lines.

    Parses the ending lines of text from a JDFTx out file corresponding to
    a series of SCF steps.

    Args:
        ending_lines (list[str]): The ending lines of text from a JDFTx out file corresponding to a
        series of SCF steps.
    """
    converged = False
    converged_reason = None
    for i, line in enumerate(ending_lines):
        if _is_converged_line(i, line, opt_type):
            converged, converged_reason = _read_converged_line(line)
        elif _is_stopping_line(i, line, opt_type):
            converged, converged_reason = _read_stopping_line(line, opt_type)
    return converged, converged_reason


def _is_converged_line(i: int, line_text: str, opt_type: str) -> bool:
    """Return True if converged line.

    Return True if the line_text is the start of a log message about
    convergence for a JDFTx optimization step.

    Args:
        i (int): The index of the line in the text slice.
        line_text (str): A line of text from a JDFTx out file.

    Returns:
        bool: True if the line_text is the start of a log message about
        convergence for a JDFTx optimization step.
    """
    return f"{opt_type}: Converged" in line_text


def _is_stopping_line(i: int, line_text: str, opt_type: str) -> bool:
    """Return True if stopping line.

    Return True if the line_text is the start of a log message about
    stopping for a JDFTx optimization step.

    Args:
        i (int): The index of the line in the text slice.
        line_text (str): A line of text from a JDFTx out file.

    Returns:
        bool: True if the line_text is the start of a log message about
        stopping for a JDFTx optimization step.
    """
    return f"{opt_type}:" in line_text and "Stopping" in line_text


def _read_converged_line(line_text: str) -> tuple[bool, str | None]:
    """Return converged and converged_reason.

    Args:
        line_text (str): A line of text from a JDFTx out file containing a message about
        convergence for a JDFTx optimization step.
    """
    converged = True
    converged_reason = line_text.split("(")[1].split(")", maxsplit=1)[0].strip()
    return converged, converged_reason


def _read_stopping_line(line_text: str, opt_type: str) -> tuple[bool, str | None]:
    """Return converged and converged_reason.

    Args:
        line_text (str): A line of text from a JDFTx out file containing a message about
        stopping for a JDFTx optimization step.
        opt_type (str): The type of electronic optimizer ("ElecMinimize" or "SCF").
    """
    converged = False
    converged_reason = line_text.split(f"{opt_type}:")[1].split("Stopping", maxsplit=1)[0].rstrip("(").strip()
    if "roundoff" in converged_reason:
        converged = True
    return converged, converged_reason
