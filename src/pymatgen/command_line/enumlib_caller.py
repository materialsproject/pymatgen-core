"""This module implements an interface to enumlib, Gus Hart's excellent Fortran
code for enumerating derivative structures.

This module depends on a compiled enumlib with the executables enum.x and
makestr.x available in the path. Please download the library at
https://github.com/msg-byu/enumlib and follow the instructions in the README to
compile these two executables accordingly.

Alternatively, ``enum.x`` and ``makestr.x`` can be provided by Enumlib.jl
(https://github.com/glwhart/Enumlib.jl), a from-scratch Julia reimplementation of
enumlib by one of its authors. It is a drop-in replacement: the
``struct_enum.in`` / ``struct_enum.out`` file contract is unchanged. Note that
Enumlib.jl's ``makestr.x`` strictly validates that the site coordinates in
``struct_enum.out`` are consistent with the space group it rederives, which the
legacy Fortran tools did not; this module therefore writes symmetry-idealized
coordinates to ``struct_enum.in``. This module's own test suite passes against
both engines.

If you use this module, please cite:

    - Gus L. W. Hart and Rodney W. Forcade, "Algorithm for generating derivative
    structures," Phys. Rev. B 77 224115 (26 June 2008)

    - Gus L. W. Hart and Rodney W. Forcade, "Generating derivative structures from
    multilattices: Application to hcp alloys," Phys. Rev. B 80 014120 (July 2009)

    - Gus L. W. Hart, Lance J. Nelson, and Rodney W. Forcade, "Generating
    derivative structures at a fixed concentration," Comp. Mat. Sci. 59
    101-107 (March 2012)

    - Wiley S. Morgan, Gus L. W. Hart, Rodney W. Forcade, "Generating derivative
    superstructures for systems with high configurational freedom," Comp. Mat.
    Sci. 136 144-149 (May 2017)
"""

from __future__ import annotations

import fractions
import functools
import itertools
import logging
import math
import re
import subprocess
import warnings
from glob import glob
from shutil import which
from typing import TYPE_CHECKING

import numpy as np
from monty.dev import requires
from monty.tempfile import ScratchDir

from pymatgen.core import DummySpecies, PeriodicSite, Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

if TYPE_CHECKING:
    from typing import ClassVar

    from pymatgen.core.structure import IStructure

logger = logging.getLogger(__name__)

# Favor the use of the newer "enum.x" by Gus Hart over "multienum.x"
ENUM_CMD = which("enum.x") or which("multienum.x")
# Prefer makestr.x at present
MAKESTR_CMD = which("makestr.x") or which("makeStr.x") or which("makeStr.py")


@functools.cache
def _is_enumlib_jl(enum_cmd: str | None) -> bool:
    """Detect whether the resolved enum.x is the Julia Enumlib.jl engine.

    Enumlib.jl responds to ``enum.x --version`` by printing a line containing
    the token ``Enumlib.jl`` (e.g. ``enum.x (Enumlib.jl) 0.3.2``). The legacy
    Fortran enum.x does not support ``--version``. The probe is cached so the
    subprocess is spawned at most once per process per command path, and it
    never raises: any failure is treated as "not Enumlib.jl".

    Args:
        enum_cmd (str | None): Resolved path to the enum.x executable.

    Returns:
        bool: True iff the executable identifies itself as Enumlib.jl.
    """
    if not enum_cmd:
        return False
    try:
        proc = subprocess.run(
            [enum_cmd, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        )
        return "Enumlib.jl" in (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        # Never let the version probe crash enumeration: any failure
        # (FileNotFoundError, subprocess.TimeoutExpired, OSError, ...) simply
        # means we cannot confirm the Julia engine.
        return False


def _warn_if_not_enumlib_jl(enum_cmd: str | None) -> None:
    """Advise switching to Enumlib.jl when the legacy Fortran enum.x is used.

    Purely advisory: this never raises or changes enumeration behavior. The
    underlying probe is cached, so the subprocess runs at most once per command
    path; how often the warning itself is shown is up to the active warning
    filters (Python's default filter shows it once per call site).

    Args:
        enum_cmd (str | None): Resolved path to the enum.x executable.
    """
    if enum_cmd and not _is_enumlib_jl(enum_cmd):
        warnings.warn(
            "You appear to be using the legacy Fortran 'enum.x'. Consider installing "
            "Enumlib.jl (https://github.com/glwhart/Enumlib.jl), a faster, "
            "better-maintained drop-in replacement for enumlib.",
            UserWarning,
            stacklevel=2,
        )


@requires(
    ENUM_CMD and MAKESTR_CMD,  # type: ignore[arg-type]
    "EnumlibAdaptor requires the executables 'enum.x' or 'multienum.x' "
    "and 'makestr.x' or 'makeStr.py' to be in the path. Please download the "
    "library at https://github.com/msg-byu/enumlib and follow the instructions "
    "in the README to compile these two executables accordingly.",
)
class EnumlibAdaptor:
    """An adaptor for enumlib.

    Attributes:
        structures (list[Structure]): all enumerated structures.
    """

    amount_tol: ClassVar[float] = 1e-5

    def __init__(
        self,
        structure: Structure | IStructure,
        min_cell_size: int = 1,
        max_cell_size: int = 1,
        symm_prec: float = 0.1,
        enum_precision_parameter: float = 0.001,
        refine_structure: bool = False,
        check_ordered_symmetry: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Initialize the adapter with a structure and some parameters.

        Args:
            structure (Structure): An input structure.
            min_cell_size (int): The minimum cell size wanted. Defaults to 1.
            max_cell_size (int): The maximum cell size wanted. Defaults to 1.
            symm_prec (float): Symmetry precision. Defaults to 0.1.
            enum_precision_parameter (float): Finite precision parameter for
                enumlib. Default of 0.001 is usually ok, but you might need to
                tweak it for certain cells.
            refine_structure (bool): If you are starting from a structure that
                has been relaxed via some electronic structure code,
                it is usually much better to start with symmetry determination
                and then obtain a refined structure. The refined structure have
                cell parameters and atomic positions shifted to the expected
                symmetry positions, which makes it much less sensitive precision
                issues in enumlib. If you are already starting from an
                experimental cif, refinement should have already been done and
                it is not necessary. Defaults to False.
            check_ordered_symmetry (bool): Whether to check the symmetry of
                the ordered sites. If the symmetry of the ordered sites is
                lower, the lowest symmetry ordered sites is included in the
                enumeration. This is important if the ordered sites break
                symmetry in a way that is important getting possible
                structures. But sometimes including ordered sites
                slows down enumeration to the point that it cannot be
                completed. Switch to False in those cases. Defaults to True.
            timeout (float): If specified, will kill enumlib after specified
                time in minutes. This can be useful for gracefully handling
                enumerations in a high-throughput context, for some enumerations
                which will not terminate in a realistic length of time.
        """
        if refine_structure:
            finder = SpacegroupAnalyzer(structure, symm_prec)
            self.structure = finder.get_refined_structure()
        else:
            self.structure = structure  # type: ignore[assignment]

        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        self.symm_prec = symm_prec
        self.enum_precision_parameter = enum_precision_parameter
        self.check_ordered_symmetry = check_ordered_symmetry
        self.timeout = timeout

        # Recommend the faster, better-maintained Enumlib.jl when the legacy
        # Fortran enum.x is detected. Advisory only; the version probe behind
        # this is cached, so repeated instantiation costs no extra subprocesses.
        _warn_if_not_enumlib_jl(ENUM_CMD)

    def run(self) -> None:
        """Run the enumeration."""
        # Work in a temporary directory
        with ScratchDir(".") as tmp_dir:
            logger.debug(f"Temp dir : {tmp_dir}")
            # Generate input files
            self._gen_input_file()

            # Perform the actual enumeration
            num_structs = self._run_multienum()

            # Read in the enumeration output as structures.
            if num_structs > 0:
                self.structures = self._get_structures(num_structs)
            else:
                raise EnumError("Unable to enumerate structure.")

    def _idealize_coords(self, structure: Structure | IStructure) -> Structure:
        """Project fractional coordinates onto ideal symmetry positions.

        Coordinates of sites in the same crystallographic orbit (determined at
        ``self.symm_prec``) are group-averaged over all their space-group
        images, which makes them exactly invariant under the detected
        symmetry operations. The input file generated for enumlib declares
        sites as symmetry-equivalent orbits, so the coordinates written must
        actually obey that symmetry: relaxed structures typically break it by
        ~1e-4 Angstrom, which Enumlib.jl's makestr.x rejects during its
        internal symmetry consistency check.

        Args:
            structure (Structure | IStructure): Input structure.

        Returns:
            Structure: Copy of the structure with idealized fractional coords.
        """
        spga = SpacegroupAnalyzer(structure, self.symm_prec)
        ops = spga.get_symmetry_operations()
        sym_struct = spga.get_symmetrized_structure()
        frac_coords = np.array(structure.frac_coords)
        idealized = Structure.from_sites(structure.sites)
        for orbit in sym_struct.equivalent_indices:
            orbit_coords = frac_coords[orbit]
            images = np.concatenate([op.operate_multi(orbit_coords) for op in ops])
            # Assign every image to the nearest orbit site under the
            # minimum-image convention, then average the images per site.
            delta = images[:, None, :] - orbit_coords[None, :, :]
            delta -= np.round(delta)
            nearest = np.linalg.norm(delta, axis=2).argmin(axis=1)
            for k, site_idx in enumerate(orbit):
                idealized[site_idx].frac_coords = orbit_coords[k] + delta[nearest == k, k].mean(axis=0)
        return idealized

    def _gen_input_file(self) -> None:
        """Generate the necessary struct_enum.in file for enumlib. See enumlib
        documentation for details.
        """
        coord_format = "{:.6f} {:.6f} {:.6f}"
        # Use symmetry finder to get the symmetrically distinct sites
        structure = self._idealize_coords(self.structure)
        fitter = SpacegroupAnalyzer(structure, self.symm_prec)
        symmetrized_structure = fitter.get_symmetrized_structure()
        logger.debug(
            f"Spacegroup {fitter.get_space_group_symbol()} ({fitter.get_space_group_number()}) "
            f"with {len(symmetrized_structure.equivalent_sites)} distinct sites"
        )

        # Enumlib doesn"t work when the number of species get too large. To
        # simplify matters, we generate the input file only with disordered sites
        # and exclude the ordered sites from the enumeration. The fact that
        # different disordered sites with the exact same species may belong to
        # different equivalent sites is dealt with by having determined the
        # spacegroup earlier and labelling the species differently.

        # `index_species` and `index_amounts` store mappings between the indices
        # used in the enum input file, and the actual species and amounts.
        index_species = []
        index_amounts = []

        # Store the ordered sites, which are not enumerated.
        ordered_sites = []
        disordered_sites = []
        coord_str: list[str] = []
        for sites in symmetrized_structure.equivalent_sites:
            if sites[0].is_ordered:
                ordered_sites.append(sites)
            else:
                _sp_label: list[int] = []
                species = dict(sites[0].species.items())
                if sum(species.values()) < 1 - EnumlibAdaptor.amount_tol:
                    # Let us first make add a dummy element for every single
                    # site whose total occupancies don't sum to 1.
                    species[DummySpecies("X")] = 1 - sum(species.values())
                for sp, amt in species.items():
                    if sp not in index_species:
                        index_species.append(sp)
                        _sp_label.append(len(index_species) - 1)
                        index_amounts.append(amt * len(sites))
                    else:
                        ind = index_species.index(sp)
                        _sp_label.append(ind)
                        index_amounts[ind] += amt * len(sites)
                sp_label: str = "/".join(f"{i}" for i in sorted(_sp_label))
                coord_str.extend(f"{coord_format.format(*site.coords)} {sp_label}" for site in sites)
                disordered_sites.append(sites)

        def get_sg_number(ss) -> int:
            finder = SpacegroupAnalyzer(Structure.from_sites(ss), self.symm_prec)
            return finder.get_space_group_number()

        target_sg_num = get_sg_number(list(symmetrized_structure))
        curr_sites = list(itertools.chain.from_iterable(disordered_sites))
        sg_num = get_sg_number(curr_sites)
        ordered_sites = sorted(ordered_sites, key=len)
        logger.debug(f"Disordered sites has sg # {sg_num}")
        self.ordered_sites = []

        # Progressively add ordered sites to our disordered sites
        # until we match the symmetry of our input structure
        if self.check_ordered_symmetry:
            while sg_num != target_sg_num and len(ordered_sites) > 0:
                sites = ordered_sites.pop(0)
                temp_sites = list(curr_sites) + sites
                new_sg_num = get_sg_number(temp_sites)
                if sg_num != new_sg_num:
                    logger.debug(f"Adding {sites[0].specie} in enum. New sg # {new_sg_num}")
                    index_species.append(sites[0].specie)
                    index_amounts.append(len(sites))
                    coord_str.extend(f"{coord_format.format(*site.coords)} {len(index_species) - 1}" for site in sites)
                    disordered_sites.append(sites)
                    curr_sites = temp_sites
                    sg_num = new_sg_num
                else:
                    self.ordered_sites.extend(sites)

        for sites in ordered_sites:
            self.ordered_sites.extend(sites)

        self.index_species = index_species

        lattice = self.structure.lattice

        output = [self.structure.formula, "bulk"]
        output.extend(coord_format.format(*vec) for vec in lattice.matrix)
        output.extend((f"{len(index_species)}", f"{len(coord_str)}"))
        output.extend(coord_str)

        output.extend(
            (
                f"{self.min_cell_size} {self.max_cell_size}",
                str(self.enum_precision_parameter),
                "full",
            )
        )

        n_disordered = sum(len(s) for s in disordered_sites)
        base = int(
            n_disordered
            * math.lcm(
                *(
                    fraction.limit_denominator(n_disordered * self.max_cell_size).denominator
                    for fraction in map(fractions.Fraction, index_amounts)
                )
            )
        )

        # This multiplicative factor of 10 is to prevent having too small bases
        # which can lead to rounding issues in the next step.
        # An old bug was that a base was set to 8, with a conc of 0.4:0.6. That
        # resulted in a range that overlaps and a conc of 0.5 satisfying this
        # enumeration. See Cu7Te5.cif test file.
        base *= 10

        # base = n_disordered # 10 ** math.ceil(math.log10(n_disordered))
        # To get a reasonable number of structures, we fix concentrations to the
        # range expected in the original structure.
        total_amounts = sum(index_amounts)
        for amt in index_amounts:
            conc = amt / total_amounts

            if abs(conc * base - round(conc * base)) < 1e-5:
                output.append(f"{round(conc * base)} {round(conc * base)} {base}")
            else:
                min_conc = math.floor(conc * base)
                output.append(f"{min_conc - 1} {min_conc + 1} {base}")
        output.append("")

        logger.debug("Generated input file:\n" + "\n".join(output))
        with open("struct_enum.in", mode="w", encoding="utf-8") as file:
            file.write("\n".join(output))

    def _run_multienum(self) -> int:
        """Run enumlib to get multiple structure.

        Returns:
            int: number of structures.
        """
        if ENUM_CMD is None:
            raise RuntimeError("enumlib is not available")

        with subprocess.Popen([ENUM_CMD], stdout=subprocess.PIPE, stdin=subprocess.PIPE, close_fds=True) as process:
            timeout = self.timeout * 60 if self.timeout is not None else None

            try:
                output = process.communicate(timeout=timeout)[0].decode("utf-8")
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise TimeoutError(f"Enumeration took more than timeout {self.timeout} minutes") from exc

        count = 0
        start_count = False
        for line in output.strip().split("\n"):
            if line.strip().endswith("RunTot"):
                start_count = True
            elif start_count and re.match(r"\d+\s+.*", line.strip()):
                count = int(line.split()[-1])
        logger.debug(f"Enumeration resulted in {count} structures")
        return count

    def _get_structures(self, num_structs: int) -> list[Structure]:
        if MAKESTR_CMD is None:
            raise RuntimeError("makestr.x is not available")

        structs: list[Structure] = []

        if ".py" in MAKESTR_CMD:
            options: tuple[str, ...] = ("-input", "struct_enum.out", str(1), str(num_structs))
        else:
            options = ("struct_enum.out", str(0), str(num_structs - 1))

        with subprocess.Popen(
            [MAKESTR_CMD, *options],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        ) as rs:
            _stdout, stderr = rs.communicate()

        if stderr:
            logger.warning(stderr.decode())

        # Sites retrieved from enumlib will lack site properties
        # to ensure consistency, we keep track of what site properties
        # are missing and set them to None
        # TODO: improve this by mapping ordered structure to original
        # disordered structure, and retrieving correct site properties
        disordered_site_properties: dict = {}

        if len(self.ordered_sites) > 0:
            original_latt = self.ordered_sites[0].lattice
            # Need to strip sites of site_properties, which would otherwise
            # result in an index error. Hence Structure is reconstructed in
            # the next step.
            site_properties: dict = {}
            for site in self.ordered_sites:
                for k, v in site.properties.items():
                    disordered_site_properties[k] = None
                    if k in site_properties:
                        site_properties[k].append(v)
                    else:
                        site_properties[k] = [v]
            ordered_structure = Structure(
                original_latt,
                [site.species for site in self.ordered_sites],
                [site.frac_coords for site in self.ordered_sites],
                site_properties=site_properties,
            )
            inv_org_latt = np.linalg.inv(original_latt.matrix)
        else:
            ordered_structure = None  # type: ignore[assignment]
            inv_org_latt = None  # type: ignore[assignment]

        for file in glob("vasp.*"):
            with open(file, encoding="utf-8") as _file:
                data = _file.read()
                data = re.sub(r"scale factor", "1", data)
                data = re.sub(r"(\d+)-(\d+)", r"\1 -\2", data)
                poscar = Poscar.from_str(data, self.index_species)  # type: ignore[arg-type]
                sub_structure = poscar.structure
                # Enumeration may have resulted in a super lattice. We need to
                # find the mapping from the new lattice to the old lattice, and
                # perform supercell construction if necessary.
                new_latt = sub_structure.lattice

                sites = []

                if len(self.ordered_sites) > 0:
                    transformation = np.dot(new_latt.matrix, inv_org_latt)  # type:ignore[arg-type]
                    transformation = [[round(cell) for cell in row] for row in transformation]
                    logger.debug(f"Supercell matrix: {transformation}")
                    struct = ordered_structure * transformation
                    sites.extend([site.to_unit_cell() for site in struct])
                    super_latt = sites[-1].lattice
                else:
                    super_latt = new_latt

                for site in sub_structure:
                    if site.specie.symbol != "X":  # We exclude vacancies.
                        sites.append(
                            PeriodicSite(
                                site.species,
                                site.frac_coords,
                                super_latt,
                                to_unit_cell=True,
                                properties=disordered_site_properties,
                            )
                        )
                    else:
                        logger.debug("Skipping sites that include species X.")
                structs.append(Structure.from_sites(sorted(sites)))  # type:ignore[arg-type]

        if len(structs) != num_structs:
            raise EnumError(
                f"{MAKESTR_CMD} produced {len(structs)} of the {num_structs} structures reported by "
                f"enumeration. Error output: {stderr.decode().strip() if stderr else '(none)'}"
            )

        logger.debug(f"Read in a total of {num_structs} structures.")
        return structs


class EnumError(BaseException):
    """Error subclass for enumeration errors."""
