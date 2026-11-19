"""Classes for reading/manipulating/writing JDFTx input files.

Classes for reading/manipulating/writing JDFTx input files.

Notes on units - within a JDFTXInfile object, all values are stored in JDFTx-consistent units, and positions,
velocities and constraint vectors are expressed in the coordinate system set by the coords-type tag (lattice
coordinates by default), exactly as they would appear in a JDFTx input file.
Initializing a JDFTXInfile object from a file or a string will assume JDFTx-consistent units (perform no conversion).
Modifying tags directly will also assume JDFTx-consistent units (perform no conversion).
Initializing a JDFTXInfile object from a pymatgen Structure object (from_structure) or a JDFTXStructure object
(from_jdftxstructure) will convert pymatgen-consistent units (Cartesian angstrom, and angstrom/fs for velocities) to
JDFTx-consistent units. This includes the lattice and ion coordinates/velocities/constraint vectors.
Converting a JDFTXInfile object to a pymatgen Structure or JDFTXStructure (structure, to_pmg_structure,
to_jdftxstructure) performs the reverse conversion.

Note: JDFTXInfile will be moved back to its own module once a more broad inputs
class is written.
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from monty.json import MSONable

from pymatgen.core import Lattice, Structure
from pymatgen.core import constants as const
from pymatgen.core.periodic_table import Element
from pymatgen.core.units import Ha_to_eV, bohr_to_ang
from pymatgen.io.jdftx.generic_tags import (
    AbstractTag,
    BoolTagContainer,
    DumpTagContainer,
    MultiformatTag,
    StrListTag,
    TagContainer,
    element_from_species_name,
)
from pymatgen.io.jdftx.jdftxinfile_default_inputs import default_inputs, get_default_fluid_params
from pymatgen.io.jdftx.jdftxinfile_master_format import (
    __PHONON_TAGS__,
    __TAG_LIST__,
    __WANNIER_TAGS__,
    MASTER_TAG_LIST,
    get_tag_object,
    get_tag_object_on_val,
)
from pymatgen.util.io_utils import clean_lines

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Self

    from numpy.typing import ArrayLike

    from pymatgen.util.typing import PathLike

__author__ = "Jacob Clary, Ben Rich"

logger = logging.getLogger(__name__)

# JDFTx's unit of time (the Hartree atomic unit of time, hbar / E_h) in fs, used to convert velocities to angstrom/fs
JDFTX_TIME_UNIT_TO_FS = const.hbar / (Ha_to_eV * const.e) * 1e15

# Site property holding each site's JDFTx species name, for structures whose species aren't all named after their
# element (e.g. Fe1 and Fe2, from pseudopotential files Fe1.upf and Fe2.upf, to tell apart two Fe sublattices)
JDFTX_SPECIES_SITE_PROPERTY = "jdftx_species"


def get_species_elements(species_names: Sequence[str]) -> tuple[list[str], list[str] | None]:
    """Get the elements of JDFTx species names, and the names to keep as the jdftx_species site property.

    Args:
        species_names (Sequence[str]): The species name of each ion.

    Returns:
        tuple[list[str], list[str] | None]: The element symbol of each ion (see element_from_species_name), and the
            species names if any of them isn't its element symbol (ignoring case, as JDFTx does), otherwise None.

    Raises:
        ValueError: If the element of a species name can't be found.
    """
    elements = []
    for name in species_names:
        element = element_from_species_name(name)
        if element is None:
            raise ValueError(
                f"Can't tell the element of species '{name}': pymatgen takes it from the element symbol the species "
                "name starts with"
            )
        elements.append(element)
    if all(name.lower() == element.lower() for name, element in zip(species_names, elements, strict=True)):
        return elements, None
    return elements, list(species_names)


def get_species_names(structure: Structure) -> list[str]:
    """Get the JDFTx species name of each site of a structure.

    Args:
        structure (Structure): The structure.

    Returns:
        list[str]: Each site's name from the jdftx_species site property, or its element symbol if it has none.
    """
    names = structure.site_properties.get(JDFTX_SPECIES_SITE_PROPERTY, [None] * len(structure))
    return [
        name if isinstance(name, str) else str(getattr(site.specie, "symbol", site.specie))
        for name, site in zip(names, structure, strict=True)
    ]


# TODO: Add check for whether all ions have or lack velocities.
# TODO: Incorporate something to collapse repeated dump tags of the same frequency
# into a single value.
# TODO: Add a method to strip all tags that contain their default values for simpler
# files written (especially when a `JDFTXInfile` is created from `JDFTXOutfileSlice`)
class JDFTXInfile(dict, MSONable):
    """Class for reading/writing JDFTx input files.

    JDFTXInfile object for reading and writing JDFTx input files.
    Essentially a dictionary with some helper functions.
    """

    path_parent: str | None = None  # Only gets a value if JDFTXInfile is initialized with from_file

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Create a JDFTXInfile object.

        Args:
            params (dict): Input parameters as a dictionary.
        """
        super().__init__()
        if params is not None:
            self.update(params)

    def __str__(self) -> str:
        """Return str representation of JDFTXInfile.

        Returns:
            str: String representation of JDFTXInfile.
        """
        return "".join([line + "\n" for line in self.get_text_list()])

    def __add__(self, other: JDFTXInfile) -> JDFTXInfile:
        """Add existing JDFTXInfile object to method caller JDFTXInfile object.

        Add all the values of another JDFTXInfile object to this object. Facilitate the use of "standard" JDFTXInfiles.
        Repeatable tags are appended together. Non-repeatable tags are replaced with their value from the other object.

        Args:
            other (JDFTXInfile): JDFTXInfile object to add to the method caller object.

        Returns:
            JDFTXInfile: The combined JDFTXInfile object.
        """
        # Deepcopy needed here, or else in `jif1 = jif2 + jif3`, `jif1` will become a reference to `jif2`
        params: dict[str, Any] = deepcopy(dict(self.items()))
        for key, val in other.items():
            if key in self:
                if val is params[key]:
                    # Unlinking the two objects fully cannot be done by deepcopy for some reason
                    continue
                tag_object = get_tag_object(key)
                if tag_object.can_repeat:
                    if isinstance(val, list):
                        for subval in list(val):
                            # Minimum effort to avoid duplicates
                            if subval not in params[key]:
                                params[key].append(subval)
                    else:
                        params[key].append(val)
                else:
                    params[key] = val
            else:
                params[key] = val
        instance = type(self)(params)
        instance.validate_tags(try_auto_type_fix=True, error_on_failed_fix=True)
        return instance

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an item in the JDFTXInfile.

        Set an item in the JDFTXInfile. This magic method is set explicitly to immediately validate when a user sets a
        tag's value, and to perform any conversion necessary.

        Args:
            key (str): Key to set.
            value (Any): Value to set.
        """
        if not isinstance(key, str):
            raise TypeError(f"{key} is not a string!")
        try:
            tag_object = get_tag_object(key)
        except KeyError:
            raise KeyError(f"The {key} tag is not in MASTER_TAG_LIST")
        if key in ("thermostat-velocity", "barostat-velocity") and isinstance(value, dict) and "v" not in value:
            value = {"v": [value[f"v{i + 1}"] for i in range(len(value))]}  # Older form: {"v1": ..., "v2": ..., ...}
        _remaining_values = None
        if tag_object.can_repeat and isinstance(value, list):
            if len(value) > 1:
                _remaining_values = value[1:]
            value = value[0]
        tag_object = get_tag_object_on_val(key, value)

        if tag_object.can_repeat and key in self:
            del self[key]
        if tag_object.can_repeat and not isinstance(value, list):
            value = [value]
        params: dict[str, Any] = {}
        if self._is_numeric(value):
            value = str(value)
        if not tag_object.can_repeat:
            value = [value]  # Shortcut to avoid writing a separate block for non-repeatable tags
        for v in value:
            processed_value = tag_object.read(key, v) if isinstance(v, str) else v
            params = self._store_value(params, tag_object, key, processed_value)
            self.update(params)
            self.validate_tags(try_auto_type_fix=True, error_on_failed_fix=True)
        if _remaining_values is not None:
            for v in _remaining_values:
                self.append_tag(key, v)

    @classmethod
    def from_dict(cls, d: dict[str, Any], validate_value_boundaries=True) -> JDFTXInfile:
        """Create JDFTXInfile from a dictionary.

        Args:
            d (dict): Dictionary to create JDFTXInfile from.
            validate_value_boundaries (bool, optional): Whether to warn about out of bounds values. Defaults to True.

        Returns:
            JDFTXInfile: The created JDFTXInfile object.
        """
        instance = cls()
        for k, v in d.items():
            if k not in ("@module", "@class"):
                instance[k] = v
        if validate_value_boundaries:
            instance.validate_boundaries(warn_only=True)
        return instance

    # TODO: This can be cleaned up by generalizing kwargs that are passed to from_str
    @classmethod
    def from_file(
        cls,
        filename: PathLike,
        dont_require_structure: bool = False,
        sort_tags: bool = True,
        assign_path_parent: bool = True,
        validate_value_boundaries: bool = True,
    ) -> JDFTXInfile:
        """Read a JDFTXInfile object from a file.

        Args:
            filename (PathLike): Filename to read from.
            dont_require_structure (bool, optional): Whether to accept an input without any of the structure tags
                (lattice, ion, ion-species) instead of raising an error. Defaults to False.
            sort_tags (bool, optional): Whether to sort the tags. Defaults to True.
            assign_path_parent (bool, optional): Whether to assign the parent directory of the input file for include
                tags. Defaults to True.
            validate_value_boundaries (bool, optional): Whether to warn about out of bounds values. Defaults to True.

        Returns:
            JDFTXInfile: The created JDFTXInfile object.
        """
        path_parent = None
        if assign_path_parent:
            path_parent = Path(filename).parents[0]
        with open(filename, encoding="utf-8") as file:
            return cls.from_str(
                file.read(),
                dont_require_structure=dont_require_structure,
                sort_tags=sort_tags,
                path_parent=path_parent,
                validate_value_boundaries=validate_value_boundaries,
            )

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
        selective_dynamics: ArrayLike | None = None,
        write_cart_coords: bool = False,
        movescale: list | None = None,
        velocities: list | None = None,
        constraint_types: list | None = None,
        constraint_vectors: list | None = None,
        hyperplane_group_names: list | None = None,
        ignore_site_properties: bool = False,
    ) -> JDFTXInfile:
        """Create a JDFTXInfile object from a pymatgen Structure.

        Each ion is written with its element symbol, or with its species name from the "jdftx_species" site property
        when it has one (set by pymatgen's JDFTx readers for species not named after their element, e.g. Fe1 and Fe2
        from pseudopotential files Fe1.upf and Fe2.upf). A species name that isn't of the site's element (e.g. after
        replace_species) is ignored with a warning. Site labels are not used as species names, as labels from other
        sources (e.g. "Fe1" and "O2" from CIF files) usually aren't. Note that get_primitive_structure only keeps sites
        of different species apart with use_site_props=True, and symmetry refinement drops site properties.

        Args:
            structure (Structure): Structure to convert.
            selective_dynamics (ArrayLike, optional): Selective dynamics flags for each site (see
                JDFTXStructure.selective_dynamics). By default None. If None, will check structure.site_properties for
                "selective_dynamics".
            write_cart_coords (bool, optional): Whether to write Cartesian coordinates instead of fractional.
                Defaults to False.
            movescale (list, optional): JDFTx's moveScale for each site (see JDFTXStructure.movescale). By default
                None. If None, will check structure.site_properties for "movescale".
            velocities (list, optional): Velocities for each site if available, in Cartesian angstrom/fs (the
                same units as structure.site_properties["velocities"]), regardless of write_cart_coords. Converted
                to JDFTx units on writing. Shape Nx3, by default None. If None, will check
                structure.site_properties for "velocities".
            constraint_types (list, optional): Constraint type ("Linear", "Planar", "HyperPlane", or None) for
                each site if available. By default None. If None, will check structure.site_properties for
                "constraint_types".
            constraint_vectors (list, optional): Constraint vectors for each site if available, in Cartesian
                coordinates (see _constraint_vector_to_cartesian), regardless of write_cart_coords. A list of vectors
                for a "HyperPlane" site, a single vector otherwise. By default None. If None, will check
                structure.site_properties for "constraint_vectors".
            hyperplane_group_names (list, optional): Group names for each site if available (a list with one
                name per hyperplane for a "HyperPlane" site, None otherwise). By default None. If None, will check
                structure.site_properties for "group_names".
            ignore_site_properties (bool, optional): Whether to ignore site properties in the structure.
                Defaults to False. If False, the per-site arguments above are taken from site_properties when they
                are provided as None. If True, they are left as provided, and the duplicated structure used for
                JDFTXStructure initialization has its site_properties cleared, except for the species names
                ("jdftx_species").

        Returns:
            JDFTXInfile: The created JDFTXInfile object.
        """
        _structure = structure.copy()
        if ignore_site_properties:
            for site_property in list(_structure.site_properties):
                if site_property != JDFTX_SPECIES_SITE_PROPERTY:  # Kept, as they decide each ion's species
                    _structure.remove_site_property(site_property)
        jdftxstructure = JDFTXStructure(
            _structure,
            selective_dynamics=selective_dynamics,
            movescale=movescale,
            velocities=velocities,
            constraint_types=constraint_types,
            constraint_vectors=constraint_vectors,
            hyperplane_group_names=hyperplane_group_names,
            write_cart_coords=write_cart_coords,
        )
        return cls.from_jdftxstructure(jdftxstructure)

    @classmethod
    def from_jdftxstructure(
        cls,
        jdftxstructure: JDFTXStructure,
    ) -> JDFTXInfile:
        """Create a JDFTXInfile object from a JDFTXStructure object.

        Args:
            jdftxstructure (JDFTXStructure): JDFTXStructure object to convert.

        Returns:
            JDFTXInfile: The created JDFTXInfile object.
        """
        jstr = jdftxstructure.get_str()
        return cls.from_str(jstr)

    def read_line(
        self,
        line: str,
        validate_value_boundaries: bool = True,
        autofix: bool = True,
        overwrite_nonrepeatable: bool = True,
    ) -> None:
        """Read a single line and update the JDFTXInfile object.

        Convenience method for reading a single line and updating the JDFTXInfile object.

        Args:
            line (str): Line to read.
            validate_value_boundaries (bool, optional): Whether to warn about out of bounds values. Defaults to True.
            autofix (bool, optional): Whether to try to fix the types of the stored values. Defaults to True.
            overwrite_nonrepeatable (bool, optional): Whether a tag that can't repeat replaces the value this object
                already has for it. If False, such a line raises an error. Defaults to True.
        """
        line = line.strip()
        tag_object, tag, value = self._preprocess_line(line)
        if not tag_object.can_repeat and overwrite_nonrepeatable and tag in self:
            del self[tag]
        processed_value = tag_object.read(tag, value)
        _params = self.as_dict(skip_module_keys=True)
        _params = self._store_value(_params, tag_object, tag, processed_value)
        self.update(_params)
        self.validate_tags(try_auto_type_fix=autofix, error_on_failed_fix=True)
        if validate_value_boundaries:
            self.validate_boundaries(warn_only=True)

    @classmethod
    def from_str(
        cls,
        string: str,
        dont_require_structure: bool = False,
        sort_tags: bool = True,
        path_parent: Path | None = None,
        validate_value_boundaries: bool = True,
        skip_invalid_tags: bool = False,
    ) -> JDFTXInfile:
        """Read a JDFTXInfile object from a string.

        Args:
            string (str): String to read from.
            dont_require_structure (bool, optional): Whether to accept an input without any of the structure tags
                (lattice, ion, ion-species) instead of raising an error. Defaults to False.
            sort_tags (bool, optional): Whether to sort the tags. Defaults to True.
            path_parent (Path, optional): Path to the parent directory of the input file for include tags.
                                          Defaults to None.
            validate_value_boundaries (bool, optional): Whether to warn about out of bounds values. Defaults to True.
            skip_invalid_tags (bool, optional): Whether to skip lines that can't be read (with a warning) instead of
                raising an error. Defaults to False.

        Returns:
            JDFTXInfile: The created JDFTXInfile object.
        """
        lines = cls._splice_includes(cls._get_lines(string), path_parent)

        params: dict[str, Any] = {}
        # process all tag value lines using specified tag formats in MASTER_TAG_LIST
        for line in lines:
            try:
                tag_object, tag, value = cls._preprocess_line(line)
                processed_value = tag_object.read(tag, value)
                params = cls._store_value(
                    params, tag_object, tag, processed_value
                )  # this will change with tag categories
            except ValueError as e:
                if skip_invalid_tags:
                    warnings.warn(f"Skipping invalid tag line: '{line}'. Error: {e}", stacklevel=2)
                    continue
                raise

        if (
            not dont_require_structure
            and "lattice" not in params
            and "ion" not in params
            and "ion-species" not in params
        ):
            raise ValueError("This input file is missing required structure tags")
        if sort_tags:
            params = {tag: params[tag] for tag in __TAG_LIST__ if tag in params}
        instance = cls(params)
        if validate_value_boundaries:
            instance.validate_boundaries(warn_only=True)
        return instance

    @classmethod
    def get_list_representation(cls, jdftxinfile: JDFTXInfile) -> JDFTXInfile:
        """Convert JDFTXInfile object properties into list representation.

        Args:
            jdftxinfile (JDFTXInfile): JDFTXInfile object to convert.
        """
        reformatted_params = deepcopy(jdftxinfile.as_dict(skip_module_keys=True))
        # rest of code assumes lists are lists and not np.arrays
        reformatted_params = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in reformatted_params.items()}
        for tag, value in reformatted_params.items():
            tag_object = get_tag_object(tag)
            if all(
                [
                    tag_object.allow_list_representation,
                    tag_object.is_tag_container,
                    cls._needs_conversion("dict-to-list", value),
                ]
            ):
                reformatted_params.update({tag: tag_object.get_list_representation(tag, value)})
        return cls(reformatted_params)

    @classmethod
    def get_dict_representation(cls, jdftxinfile: JDFTXInfile) -> JDFTXInfile:
        """Convert JDFTXInfile object properties into dict representation.

        Args:
            jdftxinfile (JDFTXInfile): JDFTXInfile object to convert.
        """
        reformatted_params = deepcopy(jdftxinfile.as_dict(skip_module_keys=True))
        # Just to make sure only passing lists and no more numpy arrays
        reformatted_params = {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in reformatted_params.items()
        }  # rest of code assumes lists are lists and not np.arrays
        for tag, value in reformatted_params.items():
            tag_object = get_tag_object(tag)
            if all(
                [
                    tag_object.allow_list_representation,
                    tag_object.is_tag_container,
                    cls._needs_conversion("list-to-dict", value),
                ]
            ):
                reformatted_params.update({tag: tag_object.get_dict_representation(tag, value)})
        return cls(reformatted_params)

    def as_dict(self, sort_tags: bool = True, skip_module_keys: bool = False) -> dict:
        """Return JDFTXInfile as MSONable dict.

        Args:
            sort_tags (bool, optional): Whether to sort the tags. Defaults to True.
            skip_module_keys (bool, optional): Whether to skip the module keys. Defaults to False.

        Returns:
            dict: JDFTXInfile as MSONable dict.
        """
        params = dict(self)
        if sort_tags:
            params = {tag: params[tag] for tag in __TAG_LIST__ if tag in params}
        if not skip_module_keys:
            params["@module"] = type(self).__module__
            params["@class"] = type(self).__name__
        return params

    def copy(self) -> JDFTXInfile:
        """Return a copy of the JDFTXInfile object.

        Returns:
            JDFTXInfile: Copy of the JDFTXInfile object.
        """
        # Deep copy so the copy's nested tag values (e.g. repeated tag entries) aren't shared with this object
        return self.from_dict(deepcopy(self.as_dict(skip_module_keys=True)), validate_value_boundaries=False)

    def get_text_list(self) -> list[str]:
        """Get a list of strings representation of the JDFTXInfile.

        Returns:
            list[str]: List of strings representation of the JDFTXInfile.
        """
        self_as_dict = self.get_dict_representation(self)

        text: list[str] = []
        for tag_group in MASTER_TAG_LIST:
            added_tag_in_group = False
            for tag in MASTER_TAG_LIST[tag_group]:
                if tag not in self:
                    continue
                added_tag_in_group = True
                tag_object: AbstractTag = MASTER_TAG_LIST[tag_group][tag]
                value = self_as_dict[tag]
                for entry in value if tag_object.can_repeat and isinstance(value, list) else [value]:
                    entry_object = tag_object
                    if isinstance(tag_object, MultiformatTag):  # Entries of a repeated tag can use different formats
                        i, _ = tag_object._determine_format_option(tag, entry)
                        entry_object = tag_object.format_options[i]
                    text.append(entry_object.write(tag, entry))
            if added_tag_in_group:
                text.append("")
        return text

    # TODO: Detect for and warn for tags that can be used together but likely shouldn't be,
    # ie (ion-width being 0 while fluid is not None)
    def write_file(self, filename: PathLike, strip_nan: bool = False) -> None:
        """Write JDFTXInfile to an in file.

        Args:
            filename (PathLike): Filename to write to.
            strip_nan (bool, optional): Whether to strip all subtags with nan values before writing (see
                                         clean_infile_of_nans). Defaults to False.
        """
        write_infile = self
        if strip_nan:
            write_infile = clean_infile_of_nans(self)
        with open(filename, encoding="utf-8", mode="w") as file:
            file.write(str(write_infile))

    @classmethod
    def to_jdftxstructure(
        cls, jdftxinfile: JDFTXInfile, sort_structure: bool = False, fill_site_properties: bool = True
    ) -> JDFTXStructure:
        """Convert JDFTXInfile to JDFTXStructure object.

        Converts JDFTx lattice, latt-scale, ion tags into JDFTXStructure, with Pymatgen structure as attribute.

        Args:
            jdftxinfile (JDFTXInfile): JDFTXInfile object to convert.
            sort_structure (bool, optional): Whether to sort the sites into the order JDFTx prints them in (see
                JDFTXStructure.sort_structure, also for when that order can differ from JDFTx's). Useful if species
                are not grouped properly together. Defaults to False.
            fill_site_properties (bool, optional): Whether to fill in the structure's site properties from the ion
                tags (see JDFTXStructure.from_jdftxinfile). Defaults to True.

        Returns:
            JDFTXStructure: The created JDFTXStructure object.
        """
        # use dict representation so it's easy to get the right column for moveScale,
        # rather than checking for velocities
        jdftxinfile_dict = cls.get_dict_representation(jdftxinfile)
        return JDFTXStructure.from_jdftxinfile(
            jdftxinfile_dict, sort_structure=sort_structure, fill_site_properties=fill_site_properties
        )

    @classmethod
    def to_pmg_structure(
        cls, jdftxinfile: JDFTXInfile, sort_structure: bool = False, fill_site_properties: bool = True
    ) -> Structure | None:
        """Convert JDFTXInfile to pymatgen Structure object.

        Converts JDFTx lattice, latt-scale, ion tags into pymatgen Structure.

        Args:
            jdftxinfile (JDFTXInfile): JDFTXInfile object to convert.
            sort_structure (bool, optional): Whether to sort the sites into the order JDFTx prints them in (see
                JDFTXStructure.sort_structure, also for when that order can differ from JDFTx's). Useful if species
                are not grouped properly together. Defaults to False.
            fill_site_properties (bool, optional): Whether to fill in the structure's site properties from the ion
                tags (see JDFTXStructure.from_jdftxinfile). Defaults to True.

        Returns:
            Structure: The created pymatgen Structure object.
        """
        # use dict representation so it's easy to get the right column for
        # moveScale, rather than checking for velocities
        logger.info(
            "INPUT DICT REP: %s",
            jdftxinfile.get_dict_representation(jdftxinfile),
        )
        jdftxstructure = JDFTXStructure.from_jdftxinfile(
            jdftxinfile.get_dict_representation(jdftxinfile),
            sort_structure=sort_structure,
            fill_site_properties=fill_site_properties,
        )
        return jdftxstructure.structure

    def strip_structure_tags(self) -> None:
        """Strip all structural tags from the JDFTXInfile.

        Strip all structural tags from the JDFTXInfile. This is useful for preparing a JDFTXInfile object
        from a pre-existing calculation for a new structure.
        """
        strucural_tags = ["lattice", "ion", "latt-scale", "coords-type"]
        for tag in strucural_tags:
            if tag in self:
                del self[tag]

    def append_tag(self, tag: str, value: Any) -> None:
        """Append a value to a tag.

        Append a value to a tag. Use this method instead of directly appending the list contained in the tag, such that
        the value is properly processed.

        Args:
            tag (str): Tag to append to.
            value (Any): Value to append.
        """
        tag_object = get_tag_object(tag)
        if isinstance(tag_object, MultiformatTag):
            if isinstance(value, str):
                i = tag_object.get_format_index_for_str_value(tag, value)
            else:
                i, _ = tag_object._determine_format_option(tag, value)
            tag_object = tag_object.format_options[i]
        if not tag_object.can_repeat:
            raise ValueError(f"The tag '{tag}' cannot be repeated and thus cannot be appended")
        params: dict[str, Any] = self.as_dict(skip_module_keys=True)
        processed_value = tag_object.read(tag, value) if isinstance(value, str) else value
        params = self._store_value(params, tag_object, tag, processed_value)
        self.update(params)

    # This method is called by setitem, but setitem is circumvented by update,
    # so this method's parameters is still necessary
    def validate_tags(
        self,
        try_auto_type_fix: bool = False,
        error_on_failed_fix: bool = True,
        return_list_rep: bool = False,
    ) -> None:
        """Validate the tags in the JDFTXInfile.

        Validate the tags in the JDFTXInfile. If try_auto_type_fix is True, will attempt to fix the tags. If
        error_on_failed_fix is True, will raise an error if the tags cannot be fixed. If return_list_rep is True, the
        fixed tags are stored in list representation.

        Args:
            try_auto_type_fix (bool, optional): Whether to attempt to fix the tags. Defaults to False.
            error_on_failed_fix (bool, optional): Whether to raise an error if the tags cannot be fixed.
                                                  Defaults to True.
            return_list_rep (bool, optional): Whether to store the fixed tags that allow it in list representation
                (only with try_auto_type_fix; nothing is returned). Defaults to False.
        """
        for tag in self:
            tag_object = get_tag_object(tag)
            _checked_tag, is_tag_valid, value = tag_object.validate_value_type(
                tag, self[tag], try_auto_type_fix=try_auto_type_fix
            )
            should_warn = not is_tag_valid
            if return_list_rep and tag_object.allow_list_representation:
                value = tag_object.get_list_representation(tag, value)
            if error_on_failed_fix and should_warn and try_auto_type_fix:
                raise ValueError(f"The {tag} tag with value:\n{self[tag]}\ncould not be fixed!")
            if try_auto_type_fix and is_tag_valid:
                self.update({tag: value})
            if should_warn:
                warnmsg = f"The {tag} tag with value:\n{self[tag]}\nhas incorrect typing!"
                if any(isinstance(tag_object, tc) for tc in [TagContainer, DumpTagContainer, BoolTagContainer]):
                    warnmsg += "(Check earlier warnings for more details)\n"
                warnings.warn(warnmsg, stacklevel=2)

    def validate_boundaries(self, warn_only: bool = False) -> None:
        """Validate the boundaries of the JDFTXInfile.

        Check every numeric value that has bounds (see AbstractNumericTag) against them.

        Args:
            warn_only (bool, optional): Whether to warn about out of bounds values instead of raising an error, as
                done when reading (so that stored inputs stay loadable). Defaults to False.

        Raises:
            ValueError: If any value is out of bounds (unless warn_only), listing all of them.
        """
        error_strs: list[str] = []
        for tag in self:
            tag_object = get_tag_object(tag)
            is_valid, error_str = tag_object.validate_value_bounds(tag, self[tag])
            if not is_valid:
                error_strs.append(error_str)
        if len(error_strs) > 0:
            err_cat = "\n".join(error_strs)
            msg = (
                f"The following boundary errors were found in the JDFTXInfile:\n{err_cat}\n"
                "Out of bounds values are kept, but left out when the input is written. JDFTx out files print such "
                "values for unset parameters (e.g. 0.0 for values strictly > 0.0). Set validate_value_boundaries to "
                "False when reading to skip this check."
            )
            if not warn_only:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

    def validate_initial_magnetic_moments(self) -> None:
        """Check the initial-magnetic-moments tag for consistency with the spintype and ion tags.

        This is not run automatically when reading an input file, since a partial input (e.g. one read with
        dont_require_structure=True and merged with defaults later) may legitimately lack the spintype or ion tags.

        Raises:
            ValueError: If spintype is missing, a species has a number of values that does not fit the spintype,
                or (when ion tags are present) a species is absent or has a mismatched ion count.
        """
        tag = "initial-magnetic-moments"
        if tag not in self:
            return
        if "spintype" not in self:
            raise ValueError(f"spintype tag must be specified to use the {tag} tag!")
        if self["spintype"] in ["no-spin", "spin-orbit"]:
            raise ValueError(f"JDFTx doesn't allow the {tag} tag in an unpolarized calculation ({self['spintype']})!")
        # z-spin has one value (M) per ion; vector-spin has three (M, theta, phi)
        mult = 3 if self["spintype"] == "vector-spin" else 1
        # JDFTx's species names are case-insensitive
        ion_species = [ion["species-id"].lower() for ion in self["ion"]] if "ion" in self else None
        for group in self[tag]["species-values"]:
            sp, n_values = group["species-id"], len(group["values"])
            if n_values % mult != 0:
                error_msg = f"Species {sp} in {tag} tag must have a multiple of {mult} values, got {n_values}!"
                if mult == 3:
                    error_msg += " Hint: for non-collinear spin, each ion needs 3 values: M, theta, and phi."
                raise ValueError(error_msg)
            if ion_species is None:
                continue
            n_ions = ion_species.count(sp.lower())
            if n_ions == 0:
                raise ValueError(f"Species {sp} in {tag} tag is not present in the ion tags!")
            if n_values != n_ions * mult:
                raise ValueError(
                    f"Species {sp} in {tag} tag has enough values for {n_values // mult} ions, "
                    f"but there are {n_ions} ions of this species in the ion tags!"
                )

    # TODO: Add examples in the docstring on how to use this function
    def is_comparable_to(
        self,
        other: JDFTXInfile,
        exclude_tags: list[str] | None = None,
        exclude_tag_categories: list[str] | None = None,
        ensure_include_tags: list[str] | None = None,
    ) -> bool:
        """Check if two JDFTXInfile objects are comparable.

        Args:
            other (JDFTXInfile): Other JDFTXInfile object to compare to.
            exclude_tags (list[str], optional): Tags to exclude from comparison. Defaults to None.
                Set to contain tags you are already accounting for being different to exclude them.
            exclude_tag_categories (list[str], optional): Tag categories to exclude from comparison. Defaults to None.
                If left as None, will exclude all tags in the "export", "restart", and "structure" categories.
                Add all tags of specified tag categories to be excluded in the comparison.
            ensure_include_tags (list[str], optional): Tags to ensure are included in the comparison. Defaults to None
                (no tags). Helpful for tags in categories you want to exclude but want to ensure are included in the
                comparison (e.g. 'ion-species', which is in the "structure" category).

        Returns:
            bool: Whether the two JDFTXInfile objects are comparable.
        """
        if exclude_tag_categories is None:
            exclude_tag_categories = ["export", "restart", "structure"]
        if ensure_include_tags is None:
            ensure_include_tags = []
        differing_tags = self.get_filtered_differing_tags(
            other,
            exclude_tags=exclude_tags,
            exclude_tag_categories=exclude_tag_categories,
            ensure_include_tags=ensure_include_tags,
        )
        return len(differing_tags) == 0

    def get_filtered_differing_tags(
        self,
        other: JDFTXInfile,
        exclude_tags: list[str] | None = None,
        exclude_tag_categories: list[str] | None = None,
        ensure_include_tags: list[str] | None = None,
    ) -> list[str]:
        """Return which tags differ between self and other JDFTXInfile.

        Returns a list of tags that differ between the two JDFTXInfile objects, excluding any tags specified in
        exclude_tags or exclude_tag_categories. This is useful for comparing two JDFTXInfile objects while ignoring
        certain tags that are expected to differ.

        Args:
            other (JDFTXInfile): Other JDFTXInfile object to compare to.
            exclude_tags (list[str], optional): Tags to exclude from comparison. Defaults to None.
                Set to contain tags you are already accounting for being different to exclude them.
            exclude_tag_categories (list[str], optional): Tag categories to exclude from comparison. Defaults to None.
                Add all tags of specified tag categories to be excluded in the comparison.
            ensure_include_tags (list[str], optional): Tags to ensure are included in the comparison. Defaults to None.
                Helpful for tags in categories you want to exclude but want to ensure are included in the comparison.

        Returns:
            list[str]: Tags that differ between the two JDFTXInfile objects.
        """
        # A set, so a tag excluded more than once (named and in a category) is still re-included in one step
        excluded = set(exclude_tags or [])
        for category in exclude_tag_categories or []:
            excluded.update(MASTER_TAG_LIST[category])
        excluded.difference_update(ensure_include_tags or [])
        return [tag for tag in self.get_differing_tags(other) if tag not in excluded]

    # get_differing_tags for a full list of tags that differ between two JDFTXInfile objects
    def get_differing_tags(self, other: JDFTXInfile) -> list[str]:
        """Return which tags differ between self and other JDFTXInfile.

        Args:
            other (JDFTXInfile): Other JDFTXInfile object to compare to.

        Returns:
            list[str]: Tags that differ between the two JDFTXInfile objects.
        """
        list1 = self.get_differing_tags_from(other)
        list2 = other.get_differing_tags_from(self)
        return list(set(list1 + list2))

    # get_differing_tags_from for only tags contained in self that differ from other JDFTXInfile
    def get_differing_tags_from(self, other: JDFTXInfile) -> list[str]:
        """Return which self-contained tags differ from other JDFTXInfile.

        Args:
            other (JDFTXInfile): Other JDFTXInfile object to compare to.

        Returns:
            list[str]: Tags from this JDFTXInfile that differ from the other JDFTXInfile.
        """
        # Missing tags and subtags compare as JDFTx's defaults, so an input matches JDFTx's echo of it
        differing_tags = []
        for tag in self:
            val2 = other[tag] if tag in other else _get_default_value(tag, other)
            if val2 is None or not _is_equal_tag_value(
                tag, _fill_default_values(tag, self[tag], self), _fill_default_values(tag, val2, other)
            ):
                differing_tags.append(tag)
        return differing_tags

    @property
    def structure(self) -> Structure | None:
        """Return a pymatgen Structure object.

        Returns:
            Structure: Pymatgen structure object.
        """
        return self.to_pmg_structure(self)

    @classmethod
    def _from_dict(cls, dct: dict[str, Any]) -> JDFTXInfile:
        """Parse a dictionary to create a JDFTXInfile object.

        Args:
            dct (dict): Dictionary to parse.

        Returns:
            JDFTXInfile: The created JDFTXInfile object.
        """
        temp = cls({k: v for k, v in dct.items() if k not in ("@module", "@class")})
        temp = cls.get_dict_representation(cls.get_list_representation(temp))
        return cls.get_list_representation(temp)

    @staticmethod
    def _get_lines(string: str) -> list[str]:
        """Return the lines of an input file's text, without comments and with continued lines joined.

        Args:
            string (str): Text of the input file.

        Returns:
            list[str]: The lines of the input file.
        """
        # Join lines ending in "\" (line continuation) with the line after them
        return re.sub(r"[ \t]*\\\n", " ", "\n".join(clean_lines(string.splitlines()))).splitlines()

    @classmethod
    def _splice_includes(
        cls, lines: list[str], path_parent: Path | None, include_stack: tuple[Path, ...] = ()
    ) -> list[str]:
        """Replace each include line with the lines of the file it names.

        JDFTx reads an included file's lines in place of the include line (so repeatable tags such as ion are added to
        those of the including file), and does the same for includes within included files. An include line for a
        file that cannot be found is kept, with a warning.

        Args:
            lines (list[str]): Lines of the input file.
            path_parent (Path, optional): Directory to look for include files in if they are not found relative to the
                current working directory.
            include_stack (tuple[Path, ...]): Resolved paths of the include files being read, to detect cyclic
                includes.

        Returns:
            list[str]: Lines of the input file with the include files' lines spliced in.
        """
        spliced_lines = []
        for line in lines:
            line_list = line.split(maxsplit=1)
            if len(line_list) != 2 or line_list[0] != "include":
                spliced_lines.append(line)
                continue
            filename = line_list[1].strip()
            path = Path(filename)
            if not path.exists() and path_parent is not None:
                path = path_parent / filename
            if not path.exists():
                warnings.warn(f"The include file {filename} ({path}) does not exist!", stacklevel=2)
                spliced_lines.append(line)
                continue
            resolved_path = path.resolve()
            if resolved_path in include_stack:
                raise ValueError(f"The include file {filename} ({path}) (in)directly includes itself!")
            included_lines = cls._get_lines(path.read_text(encoding="utf-8"))
            spliced_lines += cls._splice_includes(included_lines, path_parent, (*include_stack, resolved_path))
        return spliced_lines

    @staticmethod
    def _preprocess_line(line: str) -> tuple[AbstractTag, str, str]:
        """Preprocess a line from a JDFTXInfile, splitting it into a tag object, tag name, and value.

        Args:
            line (str): Line from the input file.

        Returns:
            tuple[AbstractTag, str, str]: Tag object, tag name, and value.
        """
        line_list = line.strip().split(maxsplit=1)
        tag: str = line_list[0].strip()
        if tag in __PHONON_TAGS__:
            raise ValueError("Phonon functionality has not been added!")
        if tag in __WANNIER_TAGS__:
            raise ValueError("Wannier functionality has not been added!")
        if tag not in __TAG_LIST__:
            err_str = f"The {tag} tag in {line_list} is not in MASTER_TAG_LIST and is not a comment, "
            err_str += "something is wrong with this input data!"
            raise ValueError(err_str)
        tag_object = get_tag_object(tag)
        value: str = ""
        if len(line_list) == 2:
            value = line_list[1].strip()
        elif len(line_list) == 1:
            value = ""  # exception for tags where only tagname is used, e.g. dump-only tag
        if isinstance(tag_object, MultiformatTag):
            i = tag_object.get_format_index_for_str_value(tag, value)
            tag_object = tag_object.format_options[i]
        return tag_object, tag, value

    @staticmethod
    def _store_value(
        params: dict[str, list | list[dict[str, dict]] | Any],
        tag_object: AbstractTag,
        tag: str,
        value: Any,
    ) -> dict:
        """Store the value in the params dictionary.

        Args:
            params (dict): Dictionary to store the value in.
            tag_object (AbstractTag): Tag object for holding tag/value pair. This tag object should be the tag object
                which the "tag" string maps to in MASTER_TAG_LIST (for a MultiformatTag, the format option the value
                was read with).
            tag (str): Tag name.
            value (Any): Value to store.

        Returns:
            dict: Updated dictionary with the stored value.
        """
        if isinstance(tag_object, StrListTag) and isinstance(value, list):  # one entry per option read
            if value:
                params.setdefault(tag, []).extend(value)
        elif tag_object.can_repeat:  # store tags that can repeat in a list
            if isinstance(tag_object, TagContainer) and tag_object.bare_is_noop and value == {}:
                return params  # a bare line changes nothing
            params.setdefault(tag, []).append(value)
        elif tag in params:
            raise ValueError(f"The '{tag}' tag appears multiple times in this input when it should not!")
        else:
            params[tag] = value
        return params

    @staticmethod
    def _needs_conversion(conversion: str, value: dict | list[dict] | list | list[list]) -> bool:
        """Determine if a value needs to be converted.

        This method is only ever called by cls.get_list/dict_representation.

        Args:
            conversion (str): Conversion type. ('dict-to-list' (value : dict | list[dict]) or
                                'list-to-dict' (value : list | list[list]))
            value (dict | list[dict] | list | list[list]): Value to check.

        Returns:
            bool: Whether the value needs to be converted.
        """
        # Check if value is not iterable
        try:
            iter(value)
        except TypeError:
            # This is triggered when JDFTXInfile is attempting to convert a non-tagcontainer to list/dict representation
            # The return boolean is meaningless in this case, so just returning False to avoid the value hitting the
            # "for x in value" loop below.
            return False
        if conversion == "list-to-dict":
            flag = False
        elif conversion == "dict-to-list":
            flag = True
        else:
            raise ValueError(f"Conversion type {conversion} is not 'list-to-dict' or 'dict-to-list'")
        if isinstance(value, dict) or all(isinstance(x, dict) for x in value):
            return flag
        return not flag

    def _is_numeric(self, value: Any) -> bool:
        """Check if a value is numeric.

        Args:
            value (Any): Value to check.

        Returns:
            bool: Whether the value is numeric.
        """
        # data-types that might accidentally be identified as numeric
        if type(value) is bool:
            return False
        try:
            float(value)
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False
        return is_numeric


ref_infile = JDFTXInfile.from_dict(default_inputs, validate_value_boundaries=False)


def _get_default_value(tag: str, infile: JDFTXInfile) -> Any:
    """Return JDFTx's default value of a tag missing from infile, or None if JDFTx has none.

    Most defaults are fixed (`ref_infile`); these depend on other tags (JDFTx's commands/minimize.cpp,
    pcm_params.cpp and elec_fillings.cpp). Missing fluid component subtags are filled in by `_fill_default_values`.

    Args:
        tag (str): Tag name.
        infile (JDFTXInfile): The input file the tag is missing from.

    Returns:
        Any: The default value of the tag.
    """
    fluid_type = infile.get("fluid", {}).get("type", "None")
    is_pcm = fluid_type in ("LinearPCM", "NonlinearPCM")
    if tag == "fluid-solvent":
        return [{}]  # Water
    if tag == "ion-width" and is_pcm:
        return "Ecut"
    if tag == "fluid-minimize":
        default = dict(ref_infile[tag])
        if fluid_type in ("LinearPCM", "SaLSA"):
            default |= {"nIterations": 400, "knormThreshold": 1e-12 if fluid_type == "SaLSA" else 1e-11}
        elif fluid_type == "NonlinearPCM":
            default["energyDiffThreshold"] = 1e-10
        if is_pcm and not infile.get("pcm-nonlinear-debug", {}).get("linearScreening", False):
            default["nAlphaAdjustMax"] = 6
        return default
    if tag == "lcao-params" and "elec-smearing" in infile:
        return dict(ref_infile[tag]) | {"smearingWidth": infile["elec-smearing"]["smearingWidth"]}
    return ref_infile.get(tag)


def _fill_default_values(tag: str, value: Any, infile: JDFTXInfile) -> Any:
    """Return a tag's value with the subtags it leaves out set to JDFTx's defaults for infile.

    Args:
        tag (str): Tag name.
        value (Any): Value of the tag in infile (or its default, if infile lacks it).
        infile (JDFTXInfile): The input file the value belongs to.

    Returns:
        Any: The filled-in value.
    """
    if tag in ("fluid-solvent", "fluid-cation", "fluid-anion"):
        temperature = infile.get("fluid", {}).get("Temperature", ref_infile["fluid"]["Temperature"])
        return [  # "bulk" is JDFTx's name for the default concentration
            get_default_fluid_params(entry.get("name", "H2O"), temperature)
            | {k: v for k, v in entry.items() if v != "bulk"}
            for entry in (value if isinstance(value, list) else [value])
        ]
    default = _get_default_value(tag, infile)
    return default | value if isinstance(default, dict) and isinstance(value, dict) else value


def _is_equal_tag_value(tag: str, val1: Any, val2: Any) -> bool:
    """Check if two values of a tag are equal.

    Entries of a repeated multiformat tag can each use a different format (e.g. ion tags with different constraint
    types), so each entry is compared using the format it matches.

    Args:
        tag (str): Tag name.
        val1 (Any): First value of the tag.
        val2 (Any): Second value of the tag.

    Returns:
        bool: Whether the two values are equal.
    """
    tag_object = get_tag_object(tag)
    if not (
        isinstance(tag_object, MultiformatTag)
        and tag_object.can_repeat
        and isinstance(val1, list)
        and isinstance(val2, list)
    ):
        return get_tag_object_on_val(tag, val1).is_equal_to(val1, get_tag_object_on_val(tag, val2), val2)
    if len(val1) != len(val2):
        return False

    def entries_equal(v1: Any, v2: Any) -> bool:
        return get_tag_object_on_val(tag, v1).is_equal_to([v1], get_tag_object_on_val(tag, v2), [v2])

    if tag_object.format_options[0].ordered:
        return all(entries_equal(v1, v2) for v1, v2 in zip(val1, val2, strict=True))
    return all(any(entries_equal(v1, v2) for v2 in val2) for v1 in val1)


def _allnone(val) -> bool:
    """Check if all elements in a list are None.

    Args:
        val (list | None): List to check.

    Returns:
        bool: Whether all elements in the list are None.
    """
    if val is None:
        return True
    if isinstance(val, list):
        return all(_allnone(x) for x in val)
    return False


def _is_homogeneous(*vals: list | None) -> bool:
    """Return whether per-site lists each form a regular array, as they must to be site properties.

    pymatgen's AseAtomsAdaptor turns every site property into an array, so a ragged one (e.g. constraint vectors on
    only some ions, or HyperPlane ions with different numbers of planes) breaks conversion to ASE Atoms. Ragged values
    are kept as attributes instead (JDFTXStructure's, or JOutStructure's private ones). Constraint types, vectors and
    group names are checked together, as they are only usable together.

    Args:
        *vals (list | None): Per-site values (None is skipped).

    Returns:
        bool: Whether np.array works on each.
    """
    try:
        for val in vals:
            if val is not None:
                np.array(val)
    except ValueError:
        return False
    return True


def selective_dynamics_site_prop_to_jdftx_interpretable(
    selective_dynamics: Sequence[Any] | ArrayLike,
) -> ArrayLike:
    """Convert selective dynamics site property to JDFTX interpretable format.

    Args:
        selective_dynamics (list[list[bool]]): Selective dynamics site property.

    Returns:
        ArrayLike: JDFTX interpretable format of selective dynamics.
    """
    # A site can move (1) only if it can move along every direction
    return np.array([int(np.all(sd)) for sd in selective_dynamics])


def clean_infile_of_nans(infile: JDFTXInfile) -> JDFTXInfile:
    """Return a copy of a JDFTXInfile with all nan values removed.

    JDFTx out files dump unset values as nan (e.g. "P0 nan" in ionic-dynamics), which JDFTx can't read back. A nan in
    a non-optional subtag removes the entry that contains it instead, since the subtag can't be dropped on its own.

    Args:
        infile (JDFTXInfile): JDFTXInfile to clean.

    Returns:
        JDFTXInfile: Copy of infile without nan values.
    """
    infile_cleaned = infile.copy()
    _apply_popmaps(infile_cleaned, _get_nanmaps(infile_cleaned))
    return infile_cleaned


def _get_nanmaps(val: Any) -> list[list[str | int]]:
    """Get the path to each nan inside a value.

    Each path is the sequence of dict keys / list indices leading to a nan, e.g. {"a": [1.0, nan]} gives [["a", 1]]
    and a bare nan gives [[]]. For a JDFTXInfile, each path starts at its tag name (e.g. ["fluid-solvent", 0,
    "epsBulk"]).

    Args:
        val (Any): Value to search (number, or nested lists/dicts).

    Returns:
        list[list[str | int]]: Path to each nan in val.
    """
    if type(val) in [float, int]:
        return [[]] if np.isnan(val) else []
    # Any other value (str, bool, None, ...) holds no nans to map
    items = val.items() if isinstance(val, dict) else enumerate(val) if isinstance(val, list) else []
    return [[key, *nanmap] for key, v in items for nanmap in _get_nanmaps(v)]


def _apply_popmaps(edit_infile: JDFTXInfile, popmaps: list[list[str | int]]) -> None:
    """Remove the value at the end of each path from a JDFTXInfile, in place.

    If a path runs through a non-optional subtag of a TagContainer, the entry containing that subtag is removed
    instead (or the whole tag, if the first subtag on the path is non-optional). Each entry is removed once even if
    several paths lead to it, later list entries are removed first so list indices in the other paths stay valid,
    and a list left empty by a removal is removed as well.

    Args:
        edit_infile (JDFTXInfile): JDFTXInfile to edit in place.
        popmaps (list[list[str | int]]): Paths to remove, each starting at a tag name (see _get_nanmaps).

    Raises:
        TypeError: If a path does not start with a tag name string.
    """
    removal_paths = {tuple(_get_removal_path(edit_infile, popmap)) for popmap in popmaps}
    # Sort so later list indices (and deeper paths) come first; the bool keeps ints and strs from being compared
    for removal_path in sorted(
        removal_paths, key=lambda path: [(isinstance(step, int), step) for step in path], reverse=True
    ):
        _remove_at_path(edit_infile, list(removal_path))


def _get_removal_path(edit_infile: JDFTXInfile, popmap: list[str | int]) -> list[str | int]:
    """Get the path to the entry to remove for a path to a nan.

    The path is cut short at the first non-optional subtag of a TagContainer, so that the entry containing that
    subtag is removed instead (or the whole tag, if the first subtag on the path is non-optional).

    Args:
        edit_infile (JDFTXInfile): JDFTXInfile the path points into.
        popmap (list[str | int]): Path to a nan, starting at a tag name (see _get_nanmaps).

    Returns:
        list[str | int]: Path to the entry to remove, starting at the tag name.

    Raises:
        TypeError: If the path does not start with a tag name string.
    """
    tag = popmap[0]
    if not isinstance(tag, str):
        raise TypeError("First element of popmap must be a tag string here")
    value = edit_infile[tag]
    if get_tag_object(tag).can_repeat and isinstance(value, list) and len(popmap) > 1 and isinstance(popmap[1], int):
        value = value[popmap[1]]  # Entries of a repeated multiformat tag can each use a different format
    tag_object = get_tag_object_on_val(tag, value)
    if not isinstance(tag_object, TagContainer):
        return list(popmap)
    subtag: AbstractTag = tag_object
    removal_path: list[str | int] = [tag]
    for step in popmap[1:]:
        if isinstance(step, str):
            subtag = subtag.subtags[step]
            if not subtag.optional:
                break
        removal_path.append(step)
    return removal_path


def _remove_at_path(edit_infile: JDFTXInfile, removal_path: list[str | int]) -> None:
    """Remove the entry at a path from a JDFTXInfile, in place.

    Dict entries (and tags) are removed by key and list entries by index. If removing a list entry leaves the list
    empty, the list itself is removed too. Paths that no longer exist are ignored.

    Args:
        edit_infile (JDFTXInfile): JDFTXInfile to edit in place.
        removal_path (list[str | int]): Path to the entry to remove, starting at the tag name.
    """
    container: Any = edit_infile
    try:
        for step in removal_path[:-1]:
            container = container[step]
    except (KeyError, IndexError):
        return
    last_step = removal_path[-1]
    if isinstance(container, list):
        if isinstance(last_step, int) and last_step < len(container):
            del container[last_step]
        if len(container) == 0:
            _remove_at_path(edit_infile, removal_path[:-1])
    else:
        container.pop(last_step, None)


@dataclass
class JDFTXStructure(MSONable):
    """Object for representing the data in the structure tags of a JDFTx input file (lattice, ion, coords-type).

    Species not named after their element (e.g. Fe1 and Fe2, from pseudopotential files Fe1.upf and Fe2.upf) are kept
    as the structure's "jdftx_species" site property, and written back from it (see JDFTXInfile.from_structure). The
    element of each species is the element symbol its name starts with (see element_from_species_name).

    movescale and selective_dynamics are kept in sync with the structure's site properties of the same names (on a copy
    of the given structure): each is read from its site property if not given, and written to it. A movescale given as
    an argument replaces the selective_dynamics site property.

    Attributes:
        structure (Structure): Associated Structure.
        selective_dynamics (list[list[bool]]): Selective dynamics flags for each site (one per direction, as in
            structure.site_properties["selective_dynamics"]; one per site is also accepted). A site is fixed (moveScale
            0) if either these or movescale mark it fixed, with a warning where the other marks it free; flags that fix
            only some directions of a site raise an error, as JDFTx's moveScale applies to every direction. Once
            set, they match movescale ([True] * 3 for a non-zero moveScale, [False] * 3 for 0).
        sort_structure (bool): Whether to sort the sites into the order JDFTx prints them in (grouped by species, in
            the order the species first appear), along with the per-site data below. Useful if species are not
            grouped properly together. Defaults to False. The order may not match JDFTx's if the input file names
            pseudopotential files with ion-species commands (instead of a $ID wildcard): JDFTx then orders the
            species as those commands do, wherever they are in the file, which is not known here.
        write_cart_coords (bool): Whether to write cartesian coordinates. Defaults to False.
        velocities (list[np.ndarray | list[float] | None]): Velocities for each site if available, in Cartesian
            angstrom/fs. Kept separate from structure.site_properties["velocities"] to allow inhomogeneous shaping.
        constraint_types (list[str | None]): Constraint type ("Linear", "Planar", "HyperPlane", or None) for each site
            if available. Read from structure.site_properties["constraint_types"] if not given.
        constraint_vectors (list[np.ndarray | list[np.ndarray] | None]): Constraint vectors for each site if
            available, in Cartesian coordinates (see _constraint_vector_to_cartesian). Read from
            structure.site_properties["constraint_vectors"] if not given. i'th entry must be a list of vectors if the
            i'th constraint_type is "HyperPlane". Otherwise, a single vector is expected.
        hyperplane_group_names (list[list[str | None] | None]): Group names for each site if available. Read from
            structure.site_properties["group_names"] if not given. i'th entry must be a list of group names (one per
            hyperplane, None for an unlabeled hyperplane) if the i'th constraint_type is "HyperPlane". Otherwise, None
            is expected.
        movescale (list[float]): JDFTx's moveScale for each site: 0 holds the ion fixed, and a non-zero value also
            scales the ion's steps in ionic and lattice minimizations (MD only uses whether it is 0). Must not be
            negative (JDFTx rejects it). Set to 0 for sites that selective_dynamics mark fixed, and to 1 for free sites
            it isn't given for.
    """

    structure: Structure | None = None
    selective_dynamics: ArrayLike | list | None = None
    sort_structure: bool = False
    write_cart_coords: bool = False
    velocities: list | None = None
    constraint_types: list | None = None
    constraint_vectors: list | None = None
    hyperplane_group_names: list | None = None
    movescale: list | None = None

    def __post_init__(self) -> None:
        """Post init function for JDFTXStructure.

        Checks that self.structure is ordered, sorts the sites if sort_structure is set, fills the per-site data that
        isn't given from the structure's site properties, and syncs movescale and selective_dynamics with each other
        and with them.
        """
        # `Structure.is_ordered` checks for partial occupancies, which JDFTx cannot handle.
        # This does not check for ordering of species.
        if not self.structure.is_ordered:
            raise ValueError("Disordered structure with partial occupancies cannot be converted into JDFTXStructure!")
        n_sites = len(cast("Structure", self.structure))
        for x in ("movescale", "selective_dynamics"):
            values = getattr(self, x)
            if values is not None and len(values) != n_sites:
                raise ValueError(f"{x} has {len(values)} values, but the structure has {n_sites} sites")
        optional_properties = [
            "movescale",
            "selective_dynamics",
            "velocities",
            "constraint_types",
            "constraint_vectors",
            "hyperplane_group_names",
        ]
        if self.sort_structure:
            # JDFTx's order: it adds a species at its first ion line and prints the ions species by species, so sites
            # are grouped by species name (which JDFTx reads ignoring case) in the order the species first appear,
            # keeping the order within each species. The per-site data given for the sites is sorted the same way.
            structure = cast("Structure", self.structure)
            names = [name.lower() for name in get_species_names(structure)]
            rank = {name: r for r, name in enumerate(dict.fromkeys(names))}  # Species in order of first appearance
            order = sorted(range(len(names)), key=lambda i: rank[names[i]])
            self.structure = type(structure).from_sites(
                [structure[i] for i in order], charge=structure._charge, properties=structure.properties
            )
            for x in optional_properties:
                if getattr(self, x) is not None:
                    setattr(self, x, [getattr(self, x)[i] for i in order])
        # For the optional properties, if they are all None, set to None to avoid writing them out, but check for
        # data in structure.site_properties to populate them if available.
        movescale_given = not _allnone(self.movescale)
        for x in optional_properties:
            if _allnone(getattr(self, x)):
                setattr(self, x, None)
                if not (x == "selective_dynamics" and movescale_given):  # Replaced, as an intentional change
                    self._sync_site_property(x)
        if self.movescale is not None or self.selective_dynamics is not None:
            self._sync_movescale_and_selective_dynamics()

    def _sync_movescale_and_selective_dynamics(self) -> None:
        """Combine movescale and selective_dynamics (a site is fixed if either marks it fixed, with a warning where the
        other marks it free), set selective_dynamics to match, and set both as the structure's site properties.
        """
        structure = cast("Structure", self.structure)
        movescale = list(self.movescale) if self.movescale is not None else [None] * len(structure)
        if self.selective_dynamics is None:
            self.selective_dynamics = [None] * len(structure)
        selective_dynamics: list[list[bool] | None] = [
            sd if sd is None else [bool(x) for x in sd] if np.ndim(sd) else [bool(sd)] * 3
            for sd in cast("list", self.selective_dynamics)
        ]
        # TODO: write a site fixed along only some directions with a Linear or Planar constraint instead
        partial = [i for i, sd in enumerate(selective_dynamics) if sd is not None and any(sd) and not all(sd)]
        if partial:
            raise NotImplementedError(
                f"Selective dynamics that fix only some directions of a site are not supported yet (sites {partial}), "
                "as JDFTx's moveScale applies to every direction. Give selective_dynamics or movescale that fix or "
                "free each of these sites entirely (a 'Linear' or 'Planar' constraint, see constraint_types, can limit "
                "the directions a free site moves in)"
            )
        mismatched = [
            i
            for i, (scale, sd) in enumerate(zip(movescale, selective_dynamics, strict=True))
            if scale is not None and sd is not None and bool(scale) != all(sd)
        ]
        if mismatched:
            warnings.warn(
                f"movescale and selective_dynamics disagree for sites {mismatched} (one marks them fixed, the other "
                "free): they are held fixed",
                stacklevel=2,
            )
        # A site is fixed if either marks it fixed; a site without either (e.g. one appended to a structure read from
        # JDFTx) can move
        filled_movescale = [
            0.0 if sd is not None and not all(sd) else 1.0 if scale is None else float(scale)
            for scale, sd in zip(movescale, selective_dynamics, strict=True)
        ]
        if negative := [i for i, scale in enumerate(filled_movescale) if scale < 0]:
            raise ValueError(f"moveScale cannot be negative (sites {negative}), as JDFTx rejects it")
        filled_selective_dynamics = [[bool(scale)] * 3 for scale in filled_movescale]
        self.movescale, self.selective_dynamics = filled_movescale, filled_selective_dynamics
        self.structure = structure.copy()  # Leave the given structure as it was
        self.structure.add_site_property("movescale", filled_movescale)
        self.structure.add_site_property("selective_dynamics", filled_selective_dynamics)

    def _sync_site_property(self, x: str) -> None:
        """Update a single JDFTXStructure attribute from the structure's site properties."""
        site_property = "group_names" if x == "hyperplane_group_names" else x
        if self.structure is not None and site_property in self.structure.site_properties:
            setattr(self, x, self.structure.site_properties[site_property])

    def __repr__(self) -> str:
        """Return representation of JDFTXStructure file.

        Returns:
            str: Representation of JDFTXStructure file.
        """
        return f"JDFTXStructure({self.get_str()})"

    def __str__(self) -> str:
        """Return string representation of JDFTXStructure file.

        Returns:
            str: String representation of JDFTXStructure file.
        """
        return self.get_str()

    @property
    def natoms(self) -> int:
        """Return number of atoms.

        Returns:
            int: Number of sites.
        """
        return len(self.structure.species)

    @classmethod
    def from_str(cls, data: str) -> JDFTXStructure:
        """Read JDFTXStructure from string.

        Args:
            data (str): String to read from.

        Returns:
            JDFTXStructure: The created JDFTXStructure object.
        """
        return cls.from_jdftxinfile(JDFTXInfile.from_str(data))

    @classmethod
    def from_file(cls, filename: str) -> JDFTXStructure:
        """Read JDFTXStructure from file.

        Args:
            filename (str): Filename to read from.

        Returns:
            JDFTXStructure: The created JDFTXStructure object.
        """
        return cls.from_jdftxinfile(JDFTXInfile.from_file(filename))

    @classmethod
    def from_jdftxinfile(
        cls, jdftxinfile: JDFTXInfile, sort_structure: bool = False, fill_site_properties: bool = True
    ) -> JDFTXStructure:
        """Get JDFTXStructure from JDFTXInfile.

        Args:
            jdftxinfile (JDFTXInfile): JDFTXInfile object.
            sort_structure (bool, optional): Whether to sort the sites into the order JDFTx prints them in (see
                JDFTXStructure.sort_structure, also for when that order can differ from JDFTx's). Useful if species
                are not grouped properly together.
            fill_site_properties (bool, optional): Whether to fill in site properties for
                `JDFTXStructure.structure: Structure` from the JDFTXInfile tags. Species names that aren't their
                element symbol (e.g. Fe1) are always kept, as the jdftx_species site property, and so are the
                movescale and selective_dynamics site properties (see JDFTXStructure). Velocities and constraints that
                don't form a regular array (e.g. constraints on only some ions) are not set as site properties, so the
                structure still converts to ASE Atoms; they are kept as the JDFTXStructure's attributes.

        Returns:
            JDFTXStructure: The created JDFTXStructure object.
        """
        lattice = _infile_to_pmg_lattice(jdftxinfile)
        elements, species_names = get_species_elements([x["species-id"] for x in jdftxinfile["ion"]])
        coords = np.array([[x["x0"], x["x1"], x["x2"]] for x in jdftxinfile["ion"]])
        movescale = [x["moveScale"] for x in jdftxinfile["ion"]]
        # Gather optional tags
        velocities = [
            np.array([float(v) for v in [x["v"]["vx0"], x["v"]["vx1"], x["v"]["vx2"]]]) if "v" in x else None
            for x in jdftxinfile["ion"]
        ]
        coords_are_cartesian = False  # is default for JDFTx
        if "coords-type" in jdftxinfile:
            coords_are_cartesian = jdftxinfile["coords-type"].lower() == "cartesian"
        # Velocities are stored in Cartesian angstrom/fs, converted to Cartesian coordinates like the positions are
        if coords_are_cartesian:
            coords *= bohr_to_ang
            velocities = [v * bohr_to_ang if v is not None else None for v in velocities]
        else:
            velocities = [lattice.get_cartesian_coords(v) if v is not None else None for v in velocities]
        velocities = [v / JDFTX_TIME_UNIT_TO_FS if v is not None else None for v in velocities]
        constraint_types: list[str | None] = []
        constraint_vectors: list = []
        hyperplane_groups: list[list[str | None] | None] = []
        for ion in jdftxinfile["ion"]:
            ctype: str | None = None
            vectors: list | None = None
            groups: list[str | None] | None = None
            if "HyperPlane" in ion:  # One or more labeled hyperplanes
                ctype, groups = "HyperPlane", [hp["group"] for hp in ion["HyperPlane"]]
                vectors = [[hp["d0"], hp["d1"], hp["d2"]] for hp in ion["HyperPlane"]]
            elif "constraint type" in ion:  # Linear, Planar, or a single unlabeled hyperplane
                ctype = ion["constraint type"]
                groups = [None] if ctype == "HyperPlane" else None
                vectors = [[ion["d0"], ion["d1"], ion["d2"]]]
            if ctype is not None and vectors is not None:
                if not coords_are_cartesian:
                    vectors = [_constraint_vector_to_cartesian(ctype, v, lattice).tolist() for v in vectors]
                if ctype != "HyperPlane":
                    vectors = vectors[0]
            constraint_types.append(ctype)
            constraint_vectors.append(vectors)
            hyperplane_groups.append(groups)

        struct: Structure = Structure(
            lattice,
            elements,
            coords,
            to_unit_cell=False,
            validate_proximity=False,
            coords_are_cartesian=coords_are_cartesian,
        )
        if species_names is not None:  # Kept even without fill_site_properties, as they decide each ion's species
            struct.add_site_property(JDFTX_SPECIES_SITE_PROPERTY, species_names)
        if fill_site_properties:
            site_properties: list[list] = [velocities, constraint_types, constraint_vectors, hyperplane_groups]
            constraints_homogeneous = _is_homogeneous(constraint_types, constraint_vectors, hyperplane_groups)
            for prop, val, homogeneous in zip(
                ["velocities", "constraint_types", "constraint_vectors", "group_names"],
                site_properties,
                [_is_homogeneous(velocities)] + [constraints_homogeneous] * 3,
                strict=True,
            ):
                if (val is not None) and (not all(x is None for x in val)) and homogeneous:
                    struct.add_site_property(prop, val)
        return cls(
            structure=struct,
            movescale=movescale,  # Added to the structure's site properties along with selective_dynamics
            sort_structure=sort_structure,
            velocities=velocities,
            constraint_types=constraint_types,
            constraint_vectors=constraint_vectors,
            hyperplane_group_names=hyperplane_groups,
        )

    def get_str(self, in_cart_coords: bool | None = None) -> str:
        """Return a string to be written as JDFTXInfile tags.

        Allows extra options as compared to calling str(JDFTXStructure) directly.

        Args:
            in_cart_coords (bool, optional): Whether coordinates are output in Cartesian (True) or lattice (False)
                coordinates. Defaults to None, which uses write_cart_coords.

        Returns:
            str: Representation of JDFTXInfile structure tags.
        """
        jdftx_tag_dict: dict[str, Any] = {}

        if in_cart_coords is None:
            in_cart_coords = self.write_cart_coords

        lattice = self.structure.lattice.matrix.T / bohr_to_ang  # column-vector format, in bohr

        jdftx_tag_dict["lattice"] = {f"R{i}{j}": lattice[i][j] for i in range(3) for j in range(3)}
        jdftx_tag_dict["ion"] = []
        jdftx_tag_dict["coords-type"] = "Cartesian" if in_cart_coords else "Lattice"
        valid_labels = {el.symbol for el in Element}
        if (self.constraint_types is not None) and (self.constraint_vectors is None):
            raise ValueError("Constraint types are present but constraint vectors are not!")
        if (self.constraint_vectors is not None) and (self.constraint_types is None):
            raise ValueError("Constraint vectors are present but constraint types are not!")
        if self.structure is None:
            return ""
        species_names = self.structure.site_properties.get(JDFTX_SPECIES_SITE_PROPERTY)
        mismatched_sites = []  # Sites whose species name isn't of their element
        for i, site in enumerate(self.structure):
            label = site.label
            coords = site.coords / bohr_to_ang if in_cart_coords else site.frac_coords
            movescale = self.movescale[i] if self.movescale is not None else 1.0
            name = species_names[i] if species_names is not None else None
            if isinstance(name, str) and element_from_species_name(name) == getattr(site.specie, "symbol", None):
                label = name  # The site's JDFTx species name (e.g. Fe1)
            else:
                if name is not None:
                    mismatched_sites.append(i)
                if label not in valid_labels:
                    # Fall back to first the species string (ie "Ag" for label "Ag0"), and then the element name
                    # (e.g. "Fe" for species "Fe2+")
                    fallbacks: tuple[str, ...] = (site.species_string, getattr(site.specie, "name", ""))
                    label = next((x for x in fallbacks if x in valid_labels), label)
                    if label not in valid_labels:
                        raise ValueError(f"Could not correct site label {label} for site (index {i})")
            ion_dict_rep = {
                "species-id": label,
                "x0": coords[0],
                "x1": coords[1],
                "x2": coords[2],
                "moveScale": movescale,
            }
            # Gather velocities if present
            if (self.velocities is not None) and (self.velocities[i] is not None):
                # angstrom/fs to JDFTx units
                velocity = np.asarray(self.velocities[i], dtype=float) * JDFTX_TIME_UNIT_TO_FS
                if in_cart_coords:
                    velocity = velocity / bohr_to_ang
                else:
                    velocity = self.structure.lattice.get_fractional_coords(velocity)
                ion_dict_rep["v"] = {"vx0": velocity[0], "vx1": velocity[1], "vx2": velocity[2]}
            # Gather constraints if present
            if self.constraint_types is not None and self.constraint_types[i] is not None:
                ion_dict_rep.update(self._get_constraint_tags(i, movescale, in_cart_coords))
            jdftx_tag_dict["ion"].append(ion_dict_rep)
        if mismatched_sites:
            warnings.warn(
                f"The {JDFTX_SPECIES_SITE_PROPERTY} species names of sites {mismatched_sites} aren't of their elements "
                "(e.g. after replace_species), so these sites are written with their element symbols instead",
                stacklevel=2,
            )
        return str(JDFTXInfile.from_dict(jdftx_tag_dict))

    def _get_constraint_tags(self, i: int, move_scale: float, in_cart_coords: bool) -> dict[str, Any]:
        """Get the ion tag entries for the constraint on a site.

        Args:
            i (int): Site index.
            move_scale (float): moveScale written for the site.
            in_cart_coords (bool): Whether the ion tags are written in Cartesian coordinates.

        Returns:
            dict[str, Any]: Constraint entries to add to the site's ion tag.

        Raises:
            ValueError: If the constraint can't be written as a valid JDFTx constraint.
        """
        ctype = cast("list", self.constraint_types)[i]
        if ctype not in ("Linear", "Planar", "HyperPlane"):
            raise ValueError(f"Unknown constraint type {ctype} for site (index {i})")
        if cast("list", self.constraint_vectors)[i] is None:
            raise ValueError(f"Site (index {i}) has a {ctype} constraint but has no constraint vector")
        if not move_scale:  # JDFTx rejects constraints on fixed ions
            raise ValueError(f"Site (index {i}) has a {ctype} constraint but a moveScale of 0")
        vectors = np.reshape(cast("list", self.constraint_vectors)[i], (-1, 3)).astype(float)
        if ctype != "HyperPlane" and (len(vectors) != 1 or not vectors.any()):
            raise ValueError(f"{ctype} constraint vector for site (index {i}) must be a single non-zero vector")
        if not in_cart_coords:
            vectors = np.array([_constraint_vector_from_cartesian(ctype, v, self.structure.lattice) for v in vectors])
        groups = None if self.hyperplane_group_names is None else self.hyperplane_group_names[i]
        if groups is None:
            groups = [None] * len(vectors)
        if len(vectors) == 1 and groups[0] is None:  # Linear, Planar, or a single unlabeled hyperplane
            return {"constraint type": ctype, "d0": vectors[0][0], "d1": vectors[0][1], "d2": vectors[0][2]}
        if None in groups:
            raise ValueError(f"Each hyperplane on site (index {i}) needs a group name when there is more than one")
        return {
            "HyperPlane": [
                {"d0": v[0], "d1": v[1], "d2": v[2], "group": group} for v, group in zip(vectors, groups, strict=True)
            ]
        }

    def write_file(self, filename: PathLike, **kwargs) -> None:
        """Write JDFTXStructure to a file.

        The supported kwargs are the same as those for the JDFTXStructure.get_str method and are passed through
        directly.

        Args:
            filename (PathLike): Filename to write to.
            **kwargs: Kwargs to pass to JDFTXStructure.get_str.
        """
        with open(filename, encoding="utf-8", mode="w") as file:
            file.write(self.get_str(**kwargs))

    def as_dict(self) -> dict:
        """MSONable dict.

        Returns:
            dict: MSONable dictionary representation of the JDFTXStructure.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "structure": self.structure.as_dict(),
            "selective_dynamics": np.array(self.selective_dynamics).tolist(),
            "velocities": self.velocities,
            "constraint_types": self.constraint_types,
            "constraint_vectors": self.constraint_vectors,
            "hyperplane_group_names": self.hyperplane_group_names,
            "movescale": self.movescale,
        }

    @classmethod
    def from_dict(cls, params: dict) -> Self:
        """Get JDFTXStructure from dict.

        Args:
            params (dict): Serialized JDFTXStructure.

        Returns:
            JDFTXStructure: The created JDFTXStructure object.
        """
        return cls(
            Structure.from_dict(params["structure"]),
            selective_dynamics=params["selective_dynamics"],
            velocities=params.get("velocities"),
            constraint_types=params.get("constraint_types"),
            constraint_vectors=params.get("constraint_vectors"),
            hyperplane_group_names=params.get("hyperplane_group_names"),
            movescale=params.get("movescale"),  # Older dicts only have selective_dynamics (one 0/1 value per site)
        )


def _infile_to_pmg_lattice(jdftxinfile: JDFTXInfile) -> Lattice:
    """Convert JDFTX lattice tag to pymatgen Lattice object.

    Handles both the matrix format (R00 ... R22) and the named-lattice format (e.g. "lattice Cubic 10.0").

    Args:
        jdftxinfile (JDFTXInfile): JDFTXInfile containing a lattice tag.

    Returns:
        Lattice: Pymatgen Lattice object.
    """
    jl = jdftxinfile["lattice"]
    if "R00" not in jl:
        return _infile_special_format_to_pmg_lattice(jdftxinfile)
    return _infile_matrix_format_to_pmg_lattice(jdftxinfile)


def _infile_special_format_to_pmg_lattice(jdftxinfile: JDFTXInfile) -> Lattice:
    """Convert JDFTX named-lattice tag to pymatgen Lattice object.

    Handles the named-lattice format (e.g. "lattice Cubic 10.0"), converting lengths from bohr to angstrom and
    applying the latt-scale tag if present.

    Args:
        jdftxinfile (JDFTXInfile): JDFTXInfile containing a named-lattice lattice tag.

    Returns:
        Lattice: Pymatgen Lattice object.

    Raises:
        NotImplementedError: If the lattice uses a modification.
        ValueError: If the lattice type is not recognized.
    """
    jl = jdftxinfile["lattice"]
    if "modification" in jl:
        raise NotImplementedError("Special case lattices with modification not implemented yet")
    # Lengths (bohr) and angles (degrees) a, b, c, alpha, beta, gamma of each named lattice
    parameters = {
        "Cubic": lambda: (jl["a"], jl["a"], jl["a"], 90.0, 90.0, 90.0),
        "Tetragonal": lambda: (jl["a"], jl["a"], jl["c"], 90.0, 90.0, 90.0),
        "Orthorhombic": lambda: (jl["a"], jl["b"], jl["c"], 90.0, 90.0, 90.0),
        "Hexagonal": lambda: (jl["a"], jl["a"], jl["c"], 90.0, 90.0, 120.0),
        "Rhombohedral": lambda: (jl["a"], jl["a"], jl["a"], jl["alpha"], jl["alpha"], jl["alpha"]),
        "Monoclinic": lambda: (jl["a"], jl["b"], jl["c"], 90.0, jl["beta"], 90.0),
        "Triclinic": lambda: (jl["a"], jl["b"], jl["c"], jl["alpha"], jl["beta"], jl["gamma"]),
    }
    lattice_type = next((name for name in parameters if name in jl), None)
    if lattice_type is None:
        raise ValueError(f"Unable to convert JDFTX lattice tag {jl} to pymatgen Lattice object")
    r = _jdftx_named_lattice_vectors(*(float(x) for x in parameters[lattice_type]()))
    if "latt-scale" in jdftxinfile:
        # JDFTx scales lattice vector k (column k) by s_k
        r = r * np.array([jdftxinfile["latt-scale"][x] for x in ["s0", "s1", "s2"]], dtype=float)
    # Row vectors in Angstrom, C-ordered like any Lattice made from lists (else Cartesian coordinates computed from it
    # can differ in the last digit from those of the same structure loaded from JSON)
    return Lattice(np.ascontiguousarray(r.T) * bohr_to_ang)


def _jdftx_named_lattice_vectors(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Get the lattice vectors of a named lattice as JDFTx builds them (GridInfo::setLatticeVectors).

    JDFTx puts a along x and b in the xy-plane, which pymatgen's Lattice.from_parameters does not (it puts c along z).
    The orientation decides where ions given in Cartesian coordinates sit in the cell, so JDFTx's construction (and
    arithmetic) is followed exactly.

    Args:
        a (float): Length of the first lattice vector (bohr).
        b (float): Length of the second lattice vector (bohr).
        c (float): Length of the third lattice vector (bohr).
        alpha (float): Angle between b and c (degrees).
        beta (float): Angle between a and c (degrees).
        gamma (float): Angle between a and b (degrees).

    Returns:
        np.ndarray: Lattice vectors as columns (bohr).
    """
    cos_alpha = math.cos(alpha * math.pi / 180.0)
    cos_beta = math.cos(beta * math.pi / 180.0)
    sin_gamma, cos_gamma = math.sin(gamma * math.pi / 180.0), math.cos(gamma * math.pi / 180.0)
    r = np.zeros((3, 3))
    r[:, 0] = [a, 0.0, 0.0]
    r[:, 1] = [b * cos_gamma, b * sin_gamma, 0.0]
    r[:, 2] = [c * cos_beta, c * ((cos_alpha - cos_beta * cos_gamma) / sin_gamma), 0.0]
    r[2, 2] = math.sqrt(c * c - r[0, 2] * r[0, 2] - r[1, 2] * r[1, 2])
    return r


def _infile_matrix_format_to_pmg_lattice(jdftxinfile: JDFTXInfile) -> Lattice:
    """Convert JDFTX matrix-format lattice tag to pymatgen Lattice object.

    Handles the matrix format (R00 ... R22, lattice vectors as columns), converting lengths from bohr to angstrom and
    applying the latt-scale tag if present.

    Args:
        jdftxinfile (JDFTXInfile): JDFTXInfile containing a matrix-format lattice tag.

    Returns:
        Lattice: Pymatgen Lattice object.
    """
    jl = jdftxinfile["lattice"]
    _lattice = np.array([[jl[f"R{i}{j}"] for j in range(3)] for i in range(3)], dtype=float)
    if "latt-scale" in jdftxinfile:
        latt_scale = np.array([jdftxinfile["latt-scale"][x] for x in ["s0", "s1", "s2"]])
        _lattice *= latt_scale
    return Lattice(np.ascontiguousarray(_lattice.T) * bohr_to_ang)  # Row vectors in Angstrom, C-ordered (see above)


def _constraint_vector_to_cartesian(constraint_type: str, vector: ArrayLike, lattice: Lattice) -> np.ndarray:
    """Convert an ion constraint vector from JDFTx lattice coordinates to Cartesian coordinates.

    As in JDFTx, a Linear constraint direction is covariant (like a direction index) and a Planar or HyperPlane
    normal is contravariant (like a plane index). The result is in the bohr-based units JDFTx uses internally, so it
    matches what JDFTx reads from an input file with Cartesian coords-type.

    Args:
        constraint_type (str): "Linear", "Planar", or "HyperPlane".
        vector (ArrayLike): Constraint vector in lattice coordinates.
        lattice (Lattice): Lattice of the structure.

    Returns:
        np.ndarray: Constraint vector in Cartesian coordinates.
    """
    r = lattice.matrix.T / bohr_to_ang  # JDFTx lattice vectors (columns), in bohr
    vector = np.asarray(vector, dtype=float)
    return r @ vector if constraint_type == "Linear" else np.linalg.solve(r.T, vector)


def _constraint_vector_from_cartesian(constraint_type: str, vector: ArrayLike, lattice: Lattice) -> np.ndarray:
    """Convert an ion constraint vector from Cartesian coordinates to JDFTx lattice coordinates.

    Inverse of _constraint_vector_to_cartesian.

    Args:
        constraint_type (str): "Linear", "Planar", or "HyperPlane".
        vector (ArrayLike): Constraint vector in Cartesian coordinates.
        lattice (Lattice): Lattice of the structure.

    Returns:
        np.ndarray: Constraint vector in lattice coordinates.
    """
    r = lattice.matrix.T / bohr_to_ang  # JDFTx lattice vectors (columns), in bohr
    vector = np.asarray(vector, dtype=float)
    return np.linalg.solve(r, vector) if constraint_type == "Linear" else r.T @ vector
