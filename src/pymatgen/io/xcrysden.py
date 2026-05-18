"""Read and write XCrySDen XSF files.

This module provides a lightweight interface for XCrySDen structure files.

Reference: http://www.xcrysden.org/doc/XSF.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
from monty.io import zopen
from monty.json import MSONable

from pymatgen.core import Element, Lattice, Structure

if TYPE_CHECKING:
    from typing import IO, Self

    from numpy.typing import NDArray

    from pymatgen.core.trajectory import Trajectory
    from pymatgen.util.typing import PathLike


@dataclass
class XSFBand(MSONable):
    """Static BXSF band-grid data.

    The ``lattice`` field stores BXSF grid spanning vectors in reciprocal
    space, not a pymatgen ``Lattice`` object.

    Args:
        fermi_energy: Fermi energy parsed from the BXSF ``BEGIN_INFO`` section.
        data: Band energies with shape ``(n_bands, nx, ny, nz)``.
        lattice: Reciprocal-space grid spanning vectors.
        origin: Reciprocal-space grid origin.
        comment: Optional comment associated with the band grid.
        labels: Labels for parsed band sections, typically ``"grid/<band_label>"``.
            If the source file omits labels, the parser should assign ``"UNK1"``,
            ``"UNK2"``, and so on.
    """

    fermi_energy: float
    data: NDArray
    lattice: NDArray
    origin: NDArray
    comment: str = ""
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.labels and len(self.labels) != self.data.shape[0]:
            raise ValueError("XSFBand labels must be empty or match the number of bands")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the band energy array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Dimensionality of each band grid."""
        return self.data.ndim - 1

    def as_dict(self) -> dict:
        """Return the MSONable dict representation."""
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "fermi_energy": self.fermi_energy,
            "data": self.data.tolist(),
            "lattice": self.lattice.tolist(),
            "origin": self.origin.tolist(),
            "comment": self.comment,
            "labels": self.labels,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create an XSFBand from an MSONable dict."""
        return cls(
            fermi_energy=d["fermi_energy"],
            data=np.asarray(d["data"]),
            lattice=np.asarray(d["lattice"]),
            origin=np.asarray(d["origin"]),
            comment=d.get("comment", ""),
            labels=d.get("labels", []),
        )


@dataclass
class XSFGrid(MSONable):
    """Static XSF DATAGRID data.

    The ``lattice`` field stores XSF grid spanning vectors, not a pymatgen
    ``Lattice`` object. These vectors may describe 2D or 3D grids and are not
    necessarily equivalent to the associated structure lattice.

    Args:
        data: Scalar grid values. The first axis enumerates datagrids within
            the block and is aligned with ``labels``.
        lattice: XSF grid spanning vectors. For 2D grids, this may be a
            three-vector array where the third vector is derived as the cross
            product of the first two vectors and was not present in the source
            XSF record.
        origin: XSF grid origin.
        comment: Optional comment associated with the grid.
        labels: Labels for parsed datagrids, typically ``"grid/<grid_label>"``.
            If the source file omits labels, the parser should assign ``"UNK1"``,
            ``"UNK2"``, and so on.
    """

    data: NDArray
    lattice: NDArray
    origin: NDArray
    comment: str = ""
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.labels and len(self.labels) != self.data.shape[0]:
            raise ValueError("XSFGrid labels must be empty or match the number of grids")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the scalar grid array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Dimensionality of each scalar grid."""
        return self.data.ndim - 1

    def as_dict(self) -> dict:
        """Return the MSONable dict representation."""
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "data": self.data.tolist(),
            "lattice": self.lattice.tolist(),
            "origin": self.origin.tolist(),
            "comment": self.comment,
            "labels": self.labels,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create an XSFGrid from an MSONable dict."""
        return cls(
            data=np.asarray(d["data"]),
            lattice=np.asarray(d["lattice"]),
            origin=np.asarray(d["origin"]),
            comment=d.get("comment", ""),
            labels=d.get("labels", []),
        )


@dataclass(eq=False, slots=True)
class XSF:
    """XCrySDen XSF structure adapter.

    The class stores one static structure, optional scalar DATAGRID blocks, and
    optional static BXSF band-grid data.

    Args:
        structure: Optional structure to write in XSF format.

    Attributes:
        structure: Parsed or assigned structure.
        forces: Optional force array aligned with ``structure`` sites.
        kind: Parsed structure family keyword, such as ``"crystal"``.
        ndim: Structural dimensionality from the XSF family keyword.
        conventional_lattice: Optional conventional lattice matrix from ``CONVVEC``.
        grids: Parsed DATAGRID blocks stored by ``block_name``.
        bands: Parsed BXSF band grids stored by ``block_name``.
        info: Metadata parsed from BXSF ``BEGIN_INFO`` sections.
        comment: Comments preserved from legal inter-section comment lines.
    """

    structure: Structure | None = None
    forces: np.ndarray | None = None
    kind: str | None = None
    ndim: int | None = None
    conventional_lattice: np.ndarray | None = None
    grids: dict[str, XSFGrid] = field(default_factory=dict)
    bands: dict[str, XSFBand] = field(default_factory=dict)
    info: dict[str, str] = field(default_factory=dict)
    comment: str = ""

    @property
    def lattice(self) -> Lattice | None:
        """Return the structure lattice, if a structure is present."""
        if self.structure is None:
            return None
        return self.structure.lattice

    def to_str(self, atom_symbol: bool = True) -> str:
        """Return the structure in XSF format.

        Args:
            atom_symbol: Whether to write atomic symbols instead of atomic numbers.

        Returns:
            XSF representation of the structure.

        Notes:
            Site property ``"vect"`` is written as the optional three-vector
            trailing the Cartesian coordinates. In XCrySDen this field is
            commonly used for forces.
        """
        if self.structure is None:
            raise ValueError("Cannot write XSF without a structure")

        lines: list[str] = []

        lines.extend(("CRYSTAL", "# Primitive lattice vectors in Angstrom", "PRIMVEC"))
        cell = self.structure.lattice.matrix
        lines.extend(f" {cell[i][0]:.14f} {cell[i][1]:.14f} {cell[i][2]:.14f}" for i in range(3))

        cart_coords = self.structure.cart_coords
        lines.extend(
            (
                "# Cartesian coordinates in Angstrom.",
                "PRIMCOORD",
                f" {len(cart_coords)} 1",
            )
        )

        for site, coord in zip(self.structure, cart_coords, strict=True):
            sp = site.specie.symbol if atom_symbol else f"{site.specie.Z}"
            x, y, z = coord
            lines.append(f"{sp} {x:20.14f} {y:20.14f} {z:20.14f}")
            if "vect" in site.properties:
                vx, vy, vz = site.properties["vect"]
                lines[-1] += f" {vx:20.14f} {vy:20.14f} {vz:20.14f}"

        return "\n".join(lines)

    def write_file(self, filename: PathLike, atom_symbol: bool = True) -> None:
        """Write the structure to an XSF file.

        Args:
            filename: Destination filename.
            atom_symbol: Whether to write atomic symbols instead of atomic numbers.
        """
        with zopen(filename, mode="wt", encoding="utf-8") as file:
            file.write(self.to_str(atom_symbol=atom_symbol))

    @classmethod
    def from_file(cls, filename: PathLike) -> Self:
        """Read an XSF-family file.

        Args:
            filename: Source filename.

        Returns:
            Parsed XSF adapter.
        """
        with zopen(filename, mode="rt", encoding="utf-8") as file:
            return cls.parse_file(file)

    @classmethod
    def from_str(cls, input_string: str) -> Self:
        """Read an XSF-family string.

        Args:
            input_string: XSF-family text.

        Returns:
            Parsed XSF adapter.
        """
        return cls.parse_file(StringIO(input_string))

    @classmethod
    def parse_file(cls, file: IO) -> Self:
        """Parse an XSF-family text stream.

        Args:
            file: Text stream with a ``readline`` method.

        Returns:
            XSF object containing parsed structures and metadata.

        Raises:
            ValueError: If the input does not contain supported XSF structure
                data.
            NotImplementedError: If a recognized XSF-family section is planned
                but not implemented yet.
        """

        xsf = cls()
        pending_line: str | None = None
        current_lattice: np.ndarray | None = None

        while True:
            raw = pending_line or file.readline()
            pending_line = None

            if raw == "":
                break

            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                comment = line[1:].strip()
                xsf.comment = comment if not xsf.comment else f"{xsf.comment}\n{comment}"
                continue

            tokens = line.split()
            keyword = tokens[0].upper()

            if keyword == "ANIMSTEPS":
                raise ValueError("Use AnimatedXSF to parse AXSF files")

            if keyword in {"MOLECULE", "POLYMER", "SLAB", "CRYSTAL"}:
                xsf.kind = keyword.lower()
                xsf.ndim = {"MOLECULE": 0, "POLYMER": 1, "SLAB": 2, "CRYSTAL": 3}[keyword]
                continue

            if keyword == "PRIMVEC":
                vectors = np.loadtxt([file.readline() for _ in range(3)])
                current_lattice = vectors
                continue

            if keyword == "CONVVEC":
                xsf.conventional_lattice = np.loadtxt([file.readline() for _ in range(3)])
                continue

            if keyword == "PRIMCOORD":
                if current_lattice is None:
                    raise ValueError("PRIMCOORD encountered before PRIMVEC")

                header = file.readline().split()
                if len(header) != 2:
                    raise ValueError("PRIMCOORD header must contain atom count and the required value 1")
                n_sites = int(header[0])
                if int(header[1]) != 1:
                    raise ValueError("PRIMCOORD header second value must be 1")

                species: list[str] = []
                coords: list[list[float]] = []
                forces: list[list[float]] = []

                for _ in range(n_sites):
                    atom_tokens = file.readline().split()
                    if len(atom_tokens) not in {4, 7}:
                        raise ValueError("PRIMCOORD atom rows must contain 4 fields or 7 fields with forces")
                    species.append(
                        atom_tokens[0] if atom_tokens[0].isalpha() else Element.from_Z(int(atom_tokens[0])).symbol
                    )
                    coords.append([float(value) for value in atom_tokens[1:4]])
                    if len(atom_tokens) == 7:
                        forces.append([float(value) for value in atom_tokens[4:7]])

                if forces and len(forces) != n_sites:
                    raise ValueError("Forces must be provided for every site or no sites")

                structure = Structure(current_lattice, species, coords, coords_are_cartesian=True)
                force_array = np.array(forces) if forces else None
                if force_array is not None:
                    structure.add_site_property("vect", force_array)

                if xsf.structure is not None:
                    raise ValueError("XSF only supports a single structure; use AnimatedXSF for multiple frames")

                xsf.structure = structure
                xsf.forces = force_array
                continue

            if keyword in {"ATOMS", "CONVCOORD"}:
                # TODO: normalize ATOMS/CONVCOORD into Structure objects.
                raise NotImplementedError(f"{keyword} parsing is not implemented yet")

            if keyword.startswith("BEGIN_BLOCK_DATAGRID_"):
                # TODO: parse DATAGRID blocks into XSFGrid objects keyed by block_name.
                # Keep XSFGrid separate from VolumetricData; conversion is only valid
                # when a structure-backed 3D grid maps cleanly onto the structure lattice.
                # Preserve the raw grid shape; XSF does not imply any periodic padding.
                raise NotImplementedError("DATAGRID parsing is not implemented yet")

            if keyword == "BEGIN_INFO":
                # TODO: parse BXSF key/value metadata into xsf.info.
                raise NotImplementedError("BXSF INFO parsing is not implemented yet")

            if keyword.startswith("BEGIN_BLOCK_BANDGRID_"):
                # TODO: parse BXSF band grids into named XSFBand objects keyed by block_name.
                # Preserve the raw grid shape; XSF does not imply any periodic padding.
                raise NotImplementedError("BANDGRID parsing is not implemented yet")

            raise ValueError(f"Unsupported or misplaced XSF keyword: {line}")

        if xsf.structure is None and not xsf.grids and not xsf.bands:
            raise ValueError("Invalid XSF data")

        return xsf


@dataclass(eq=False, slots=True)
class AnimatedXSF:
    """XCrySDen animated XSF trajectory adapter.

    Args:
        structures: Optional list of parsed trajectory frames.
    """

    structures: list[Structure] = field(default_factory=list)
    forces: list[np.ndarray | None] = field(default_factory=list)
    steps: list[int | None] = field(default_factory=list)
    comment: str = ""

    @classmethod
    def from_file(cls, filename: PathLike) -> Self:
        """Read an animated XSF file.

        Args:
            filename: Source filename.

        Returns:
            Parsed animated XSF adapter.
        """
        with zopen(filename, mode="rt", encoding="utf-8") as file:
            return cls.parse_file(file)

    @classmethod
    def from_str(cls, input_string: str) -> Self:
        """Read an animated XSF string.

        Args:
            input_string: AXSF text.

        Returns:
            Parsed animated XSF adapter.
        """
        return cls.parse_file(StringIO(input_string))

    @classmethod
    def parse_file(cls, file: IO) -> Self:
        # TODO: parse AXSF with trajectory-like frame storage.
        raise NotImplementedError("AXSF parsing is not implemented yet")

    def as_trajectory(self) -> Trajectory:
        # TODO: build a Trajectory from parsed periodic AXSF frames. Convert Cartesian XSF coordinates to fractional
        # coordinates first, and propagate frame-aligned forces through site_properties when available.
        raise NotImplementedError("AXSF trajectory conversion is not implemented yet")
