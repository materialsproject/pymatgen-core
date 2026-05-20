"""Read and write XCrySDen XSF files.

This module provides a lightweight interface for XCrySDen structure files.

Reference: http://www.xcrysden.org/doc/XSF.html
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BufferedIOBase, BytesIO
from typing import TYPE_CHECKING, Literal

import numpy as np
from monty.io import zopen
from monty.json import MSONable

from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.optimization.fast_parser import parse_n_doubles

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

    from pymatgen.core.trajectory import Trajectory
    from pymatgen.util.typing import PathLike

XSF_KEYWORDS = {
    b"ANIMSTEPS",
    b"MOLECULE",
    b"POLYMER",
    b"SLAB",
    b"CRYSTAL",
    b"PRIMVEC",
    b"CONVVEC",
    b"PRIMCOORD",
    b"ATOMS",
    b"CONVCOORD",
    b"BEGIN_BLOCK_DATAGRID_",
    b"END_BLOCK_DATAGRID",
    b"BEGIN_DATAGRID_",
    b"END_DATAGRID",
    b"BEGIN_INFO",
    b"END_INFO",
    b"BEGIN_BLOCK_BANDGRID_",
    b"END_BLOCK_BANDGRID",
    b"BEGIN_BANDGRID_",
    b"END_BANDGRID",
}


@dataclass
class XSFBand(MSONable):
    """Static BXSF band-grid data.

    The ``lattice`` field stores BXSF grid spanning vectors in reciprocal
    space, not a pymatgen ``Lattice`` object.

    Args:
        data: Band energies with shape ``(n_bands, nx, ny, nz)``.
        lattice: Reciprocal-space grid spanning vectors.
        origin: Reciprocal-space grid origin.
        comment: Optional comment associated with the band grid.
        labels: Labels for parsed band sections, typically ``"grid/<band_label>"``.
            If the source file omits labels, the parser should assign
            ``"UNKBAND0"``, ``"UNKBAND1"``, and so on.
    """

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
            If the source file omits labels, the parser should assign
            ``"UNKGRID0"``, ``"UNKGRID1"``, and so on.
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
        fermi_energy: Fermi energy parsed from the BXSF ``BEGIN_INFO`` section.
        comment: Comments preserved from legal inter-section comment lines.
    """

    structure: Structure | Molecule | None = None
    forces: np.ndarray | None = None
    kind: str | None = None
    ndim: int | None = None
    conventional_lattice: np.ndarray | None = None
    grids: dict[str, XSFGrid] = field(default_factory=dict)
    bands: dict[str, XSFBand] = field(default_factory=dict)
    comment: str = ""
    fermi_energy: float | None = None

    @property
    def lattice(self) -> Lattice | None:
        """Return the structure lattice, if a structure is present."""
        if isinstance(self.structure, Structure):
            return self.structure.lattice
        return None

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
        n_sites = len(self.structure)

        def append_comment(text: str) -> None:
            for comment_line in text.splitlines():
                if not comment_line:
                    continue
                lines.append(comment_line if comment_line.startswith("#") else f"# {comment_line}")

        def format_row(values: np.ndarray | list[float]) -> str:
            arr = np.asarray(values, dtype=float)
            return " ".join(f"{value:.14f}" for value in arr)

        def append_flat_values(values: np.ndarray, order: Literal["C", "F"] = "F") -> None:
            flat = np.asarray(values, dtype=float).ravel(order=order)
            lines.extend(
                " ".join(f"{value:.14f}" for value in flat[start : start + 6]) for start in range(0, flat.size, 6)
            )

        def grid_label(label: str, default: str) -> str:
            if label.startswith("grid/"):
                label = label.split("/", maxsplit=1)[1]
            return label or default

        def write_datagrid_block(block_name: str, grid: XSFGrid) -> None:
            data = np.asarray(grid.data, dtype=float)
            if data.ndim not in {3, 4}:
                raise ValueError("XSFGrid data must have shape (n, nx, ny) or (n, nx, ny, nz)")

            ndim = data.ndim - 1
            if ndim not in {2, 3}:
                raise ValueError("XSFGrid data must represent 2D or 3D grids")

            lattice = np.asarray(grid.lattice, dtype=float)
            if lattice.shape not in {(2, 3), (3, 3)}:
                raise ValueError("XSFGrid lattice must have shape (2, 3) or (3, 3)")

            origin = np.asarray(grid.origin, dtype=float)
            if origin.shape != (3,):
                raise ValueError("XSFGrid origin must have shape (3,)")

            block_labels = grid.labels or [f"UNKGRID{i}" for i in range(data.shape[0])]
            if len(block_labels) != data.shape[0]:
                raise ValueError("XSFGrid labels must be empty or match the number of grids")

            lines.append(f"BEGIN_BLOCK_DATAGRID_{ndim}D")
            lines.append(block_name)
            if grid.comment:
                append_comment(grid.comment)

            expected_shape = data.shape[1:]
            for index, label in enumerate(block_labels):
                lines.append(f"BEGIN_DATAGRID_{ndim}D_{grid_label(label, f'UNKGRID{index}')}")
                lines.append(" ".join(str(int(value)) for value in expected_shape))
                lines.append(format_row(origin))
                lines.extend(format_row(vector) for vector in lattice)
                append_flat_values(data[index], order="F")
                lines.append(f"END_DATAGRID_{ndim}D")

            lines.append(f"END_BLOCK_DATAGRID_{ndim}D")

        def write_bandgrid_block(block_name: str, band: XSFBand) -> None:
            data = np.asarray(band.data, dtype=float)
            if data.ndim != 4:
                raise ValueError("XSFBand data must have shape (n_bands, nx, ny, nz)")

            lattice = np.asarray(band.lattice, dtype=float)
            if lattice.shape != (3, 3):
                raise ValueError("XSFBand lattice must have shape (3, 3)")

            origin = np.asarray(band.origin, dtype=float)
            if origin.shape != (3,):
                raise ValueError("XSFBand origin must have shape (3,)")

            band_labels = band.labels or [f"UNKBAND{i}" for i in range(data.shape[0])]
            if len(band_labels) != data.shape[0]:
                raise ValueError("XSFBand labels must be empty or match the number of bands")

            lines.append("BEGIN_BLOCK_BANDGRID_3D")
            lines.append(block_name)
            if band.comment:
                append_comment(band.comment)
            lines.append(f"BEGIN_BANDGRID_3D_{block_name}")
            lines.append(str(data.shape[0]))
            lines.append(" ".join(str(int(value)) for value in data.shape[1:]))
            lines.append(format_row(origin))
            lines.extend(format_row(vector) for vector in lattice)
            for index, label in enumerate(band_labels):
                lines.append(f"BAND: {label}")
                append_flat_values(data[index], order="C")
            lines.append("END_BANDGRID_3D")
            lines.append("END_BLOCK_BANDGRID_3D")

        if self.comment:
            append_comment(self.comment)

        cart_coords = self.structure.cart_coords
        if isinstance(self.structure, Molecule):
            lines.extend(("MOLECULE", "ATOMS"))
        else:
            kind = (self.kind or "crystal").upper()
            if kind not in {"POLYMER", "SLAB", "CRYSTAL"}:
                raise ValueError(f"Unsupported XSF structure kind for periodic output: {self.kind}")
            lines.extend((kind, "# Primitive lattice vectors in Angstrom", "PRIMVEC"))
            cell = self.structure.lattice.matrix
            lines.extend(f" {cell[i][0]:.14f} {cell[i][1]:.14f} {cell[i][2]:.14f}" for i in range(3))
            lines.extend(
                (
                    "# Cartesian coordinates in Angstrom.",
                    "PRIMCOORD",
                    f" {len(cart_coords)} 1",
                )
            )

        forces = self.forces
        if forces is None and any("vect" in site.properties for site in self.structure):
            if not all("vect" in site.properties for site in self.structure):
                raise ValueError("site property 'vect' must be present on every site or none")
            forces = np.asarray([site.properties["vect"] for site in self.structure], dtype=float)

        if forces is not None:
            forces = np.asarray(forces, dtype=float)
            if forces.shape != (n_sites, 3):
                raise ValueError("Forces must have shape (n_sites, 3)")

        for index, (site, coord) in enumerate(zip(self.structure, cart_coords, strict=True)):
            sp = site.specie.symbol if atom_symbol else f"{site.specie.Z}"
            x, y, z = coord
            lines.append(f"{sp} {x:20.14f} {y:20.14f} {z:20.14f}")
            if forces is not None:
                vx, vy, vz = forces[index]
                lines[-1] += f" {vx:20.14f} {vy:20.14f} {vz:20.14f}"

        if self.conventional_lattice is not None:
            conventional_lattice = np.asarray(self.conventional_lattice, dtype=float)
            if conventional_lattice.shape != (3, 3):
                raise ValueError("conventional_lattice must have shape (3, 3)")
            lines.extend(("# Conventional lattice vectors in Angstrom", "CONVVEC"))
            lines.extend(format_row(vector) for vector in conventional_lattice)

        for block_name, grid in self.grids.items():
            write_datagrid_block(block_name, grid)

        if self.fermi_energy is not None or self.bands:
            if self.bands and self.fermi_energy is None:
                raise ValueError("Cannot write BANDGRID blocks without a Fermi energy")
            if self.fermi_energy is not None:
                lines.extend(("BEGIN_INFO", f"  Fermi Energy: {self.fermi_energy:.14f}", "END_INFO"))

        for block_name, band in self.bands.items():
            write_bandgrid_block(block_name, band)

        return "\n".join(lines)

    def structure_properties(self) -> dict[str, object]:
        """Return XSF extras as flat structure properties."""
        properties: dict[str, object] = {}

        def strip_label_prefix(label: str) -> str:
            return label.split("/", maxsplit=1)[-1] if "/" in label else label

        for block_name, grid in self.grids.items():
            labels = grid.labels or [f"UNKGRID{i}" for i in range(grid.data.shape[0])]
            for index, label in enumerate(labels):
                key = f"grids/{block_name}/{strip_label_prefix(label)}"
                properties[key] = XSFGrid(
                    data=np.asarray(grid.data[index : index + 1]),
                    lattice=np.asarray(grid.lattice),
                    origin=np.asarray(grid.origin),
                    comment=grid.comment,
                    labels=[label],
                )

        for block_name, band in self.bands.items():
            labels = band.labels or [f"UNKBAND{i}" for i in range(band.data.shape[0])]
            for index, label in enumerate(labels):
                key = f"bands/{block_name}/{strip_label_prefix(label)}"
                properties[key] = XSFBand(
                    data=np.asarray(band.data[index : index + 1]),
                    lattice=np.asarray(band.lattice),
                    origin=np.asarray(band.origin),
                    comment=band.comment,
                    labels=[label],
                )

        if self.fermi_energy is not None:
            properties["bands/fermi_energy"] = self.fermi_energy

        return properties

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
        """Read an XSF-family file path.

        Args:
            filename: Source filename.

        Returns:
            Parsed XSF adapter.
        """

        with zopen(filename, mode="rb") as file:
            return cls.parse_file(file)

    @classmethod
    def from_str(cls, input_string: str) -> Self:
        """Read an XSF-family string.

        Args:
            input_string: XSF-family text.

        Returns:
            Parsed XSF adapter.
        """
        return cls.parse_file(BytesIO(input_string.encode("utf-8")))

    @classmethod
    def parse_file(cls, file) -> Self:
        """Parse an XSF-family binary stream or a text wrapper exposing ``.buffer``.

        Args:
            file: Seekable binary stream with a ``readline`` method that returns ``bytes``,
                or a text stream exposing ``.buffer``.
        """

        if hasattr(file, "buffer"):
            file = file.buffer
        elif not isinstance(file, (BytesIO, BufferedIOBase)):
            raise TypeError("XSF.parse_file requires a binary stream opened in bytes mode")
        xsf = cls()
        comments = []
        block_name: str | None = None
        block_dim: int | None = None
        iframe = None
        current_lattice = None
        block_labels: list[str] = []
        block_data: list[np.ndarray] = []
        block_origin: np.ndarray | None = None
        block_lattice: np.ndarray | None = None

        while True:
            raw = file.readline()
            if raw == b"":  # EOF
                break

            line = raw.strip()
            if not line:  # empty or whitespace-only line
                continue

            if line.startswith(b"#"):  # comment line
                comments.append(line)
                continue

            tokens = line.split()
            keyword = tokens[0].upper()

            if keyword == b"ANIMSTEPS":
                raise ValueError("ANIMSTEPS keyword is not allowed in static XSF files; use AnimatedXSF for AXSF files")

            if keyword in {b"MOLECULE", b"POLYMER", b"SLAB", b"CRYSTAL"}:
                xsf.kind = keyword.decode("utf-8").lower()
                xsf.ndim = {b"MOLECULE": 0, b"POLYMER": 1, b"SLAB": 2, b"CRYSTAL": 3}[keyword]
                continue

            if keyword == b"PRIMVEC":
                index = int(tokens[1]) if len(tokens) > 1 else None
                if iframe is not None and index is not None and iframe != index:
                    break  # new frame starts, stop parsing for static XSF
                if iframe is None and index is not None:
                    iframe = index
                current_lattice = np.loadtxt(file, max_rows=3)
                continue

            if keyword == b"CONVVEC":
                index = int(tokens[1]) if len(tokens) > 1 else None
                if iframe is not None and index is not None and iframe != index:
                    break  # new frame starts, stop parsing for static XSF
                if iframe is None and index is not None:
                    iframe = index
                xsf.conventional_lattice = np.loadtxt(file, max_rows=3)
                continue

            if keyword == b"PRIMCOORD":
                index = int(tokens[1]) if len(tokens) > 1 else None
                if iframe is not None and index is not None and iframe != index:
                    break  # new frame starts, stop parsing for static XSF
                if iframe is None and index is not None:
                    iframe = index
                if current_lattice is None:
                    raise ValueError("PRIMCOORD encountered before PRIMVEC")
                if xsf.kind not in {"crystal", "slab", "polymer"}:
                    raise ValueError("PRIMCOORD is only valid in periodic sections")
                if xsf.structure is not None:
                    raise ValueError("XSF only supports a single structure; use AnimatedXSF for multiple frames")

                n_sites, code = map(int, file.readline().split())
                if code != 1:
                    raise ValueError("PRIMCOORD header second value must be 1")

                now = file.tell()
                query = file.readline().split()
                if len(query) not in {4, 7}:
                    raise ValueError("PRIMCOORD atom rows must contain 4 fields or 7 fields with forces")
                file.seek(now)
                names = ["species", "x", "y", "z", "fx", "fy", "fz"][: len(query)]
                formats = ["U8" if query[0].isalpha() else "i8", "f8", "f8", "f8", "f8", "f8", "f8"][: len(query)]

                _data = np.atleast_1d(np.loadtxt(file, max_rows=n_sites, dtype={"names": names, "formats": formats}))

                species = _data["species"].tolist()
                coords = np.column_stack((_data["x"], _data["y"], _data["z"]))

                xsf.structure = Structure(current_lattice, species, coords, coords_are_cartesian=True)

                if len(query) == 7:
                    xsf.forces = np.column_stack((_data["fx"], _data["fy"], _data["fz"]))

                continue

            if keyword == b"ATOMS":
                index = int(tokens[1]) if len(tokens) > 1 else None
                if iframe is not None and index is not None and iframe != index:
                    break  # new frame starts, stop parsing for static XSF
                if iframe is None and index is not None:
                    iframe = index
                if xsf.kind != "molecule":
                    raise ValueError("ATOMS is only valid in MOLECULE sections")
                atom_species: list[str] = []
                atom_coords: list[list[float]] = []
                atom_forces: list[list[float]] | None = []

                while True:
                    now = file.tell()
                    line = file.readline().strip()
                    if not line:
                        break
                    keyword = line.split()[0].upper()
                    if keyword in XSF_KEYWORDS or any(keyword.startswith(k) for k in XSF_KEYWORDS if k.endswith(b"_")):
                        file.seek(now)
                        break
                    symbol, x, y, z, *force = line.split()
                    atom_species.append(symbol.decode("utf-8"))
                    atom_coords.append([float(x), float(y), float(z)])
                    atom_forces.append([float(f) for f in force])
                atom_coords = np.asarray(atom_coords)
                if not atom_forces or len(atom_forces[0]) == 0:
                    atom_forces = None
                elif any(len(force) != 3 for force in atom_forces):
                    raise ValueError("Each ATOMS row must have 3 coordinate fields followed by optional force fields")
                else:
                    atom_forces = np.asarray(atom_forces)

                xsf.structure = Molecule(atom_species, atom_coords)
                xsf.forces = atom_forces
                continue

            if keyword == b"CONVCOORD":
                raise NotImplementedError("CONVCOORD section is not allowed in XSF files")

            if keyword.startswith(b"BEGIN_BLOCK_DATAGRID_"):
                if block_name is not None:
                    raise ValueError("Nested BEGIN_BLOCK_DATAGRID is not allowed")
                block_name = file.readline().strip().decode("utf-8")
                block_dim = int(line.removeprefix(b"BEGIN_BLOCK_DATAGRID_").rstrip(b"D"))
                if block_dim not in {2, 3}:
                    raise ValueError(f"Unsupported DATAGRID dimensionality: {block_dim}D")
                if block_name in xsf.grids:
                    raise ValueError(f"Duplicate DATAGRID block name: {block_name}")

                block_labels = []
                block_data = []
                block_origin = None
                block_lattice = None
                continue

            if keyword.startswith(b"END_BLOCK_DATAGRID"):
                if block_name is None:
                    raise ValueError("END_BLOCK_DATAGRID encountered without a matching BEGIN_BLOCK_DATAGRID")
                if block_origin is None or block_lattice is None:
                    raise ValueError("DATAGRID block is missing required origin or lattice information")
                xsf.grids[block_name] = XSFGrid(
                    data=np.asarray(block_data),
                    lattice=block_lattice,
                    origin=block_origin,
                    comment="",
                    labels=block_labels,
                )
                block_name = None
                block_dim = None
                block_labels = []
                block_data = []
                block_origin = None
                block_lattice = None
                continue

            if keyword.startswith(b"BEGIN_DATAGRID_"):
                if block_name is None:
                    raise ValueError("BEGIN_DATAGRID encountered without a matching BEGIN_BLOCK_DATAGRID")
                if block_dim is None:
                    raise ValueError("BEGIN_DATAGRID encountered before a DATAGRID block header")
                header = line.removeprefix(b"BEGIN_DATAGRID_")
                grid_dim, grid_name = header.split(b"_", maxsplit=1)
                grid_dim = int(grid_dim.rstrip(b"D"))
                if grid_dim != block_dim:
                    raise ValueError(
                        f"Declared DATAGRID dimension {grid_dim}D does not match block keyword {block_dim}D"
                    )
                if grid_name == b"":
                    grid_name = f"UNKGRID{len(block_labels)}".encode()

                block_shape = np.loadtxt(file, max_rows=1, dtype=int)
                if block_origin is None:
                    block_origin = np.loadtxt(file, max_rows=1)
                elif not np.allclose(block_origin, np.loadtxt(file, max_rows=1)):
                    raise ValueError("Inconsistent DATAGRID origin within the same block")
                if block_lattice is None:
                    block_lattice = np.loadtxt(file, max_rows=block_dim)
                elif not np.allclose(block_lattice, np.loadtxt(file, max_rows=block_dim)):
                    raise ValueError("Inconsistent DATAGRID lattice within the same block")

                data = np.empty(block_shape, dtype=np.float64, order="F")
                parse_num = parse_n_doubles(file, data.ravel(order="F"), nelem=np.prod(block_shape))
                if parse_num != np.prod(block_shape):
                    raise ValueError(f"Expected {np.prod(block_shape)} grid values but parsed {parse_num}")

                block_labels.append(f"grid/{grid_name.decode('utf-8')}")
                block_data.append(data)
                continue

            if keyword.startswith(b"END_DATAGRID"):
                if block_origin is None or block_lattice is None:
                    raise ValueError("DATAGRID block is missing required origin or lattice information")
                continue

            if keyword.startswith(b"BEGIN_INFO"):
                if xsf.fermi_energy is not None:
                    raise ValueError("Multiple BEGIN_INFO sections are not supported")
                fermi_line = file.readline().strip()
                match = re.search(rb"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)", fermi_line)
                if match is None:
                    raise ValueError(f"Could not parse Fermi energy from line: {fermi_line!r}")
                xsf.fermi_energy = float(match.group(1))
                continue

            if keyword.startswith(b"END_INFO"):
                if xsf.fermi_energy is None:
                    raise ValueError("END_INFO encountered without a preceding BEGIN_INFO")
                continue

            if keyword.startswith(b"BEGIN_BLOCK_BANDGRID_"):
                if block_name is not None:
                    raise ValueError("Nested BEGIN_BLOCK_BANDGRID is not allowed")
                block_dim = int(line.removeprefix(b"BEGIN_BLOCK_BANDGRID_").rstrip(b"D"))
                block_name = file.readline().strip().decode("utf-8")
                if block_dim != 3:
                    raise ValueError(f"Unsupported BANDGRID dimensionality: {block_dim}D")
                if block_name in xsf.bands:
                    raise ValueError(f"Duplicate BANDGRID block name: {block_name}")
                if len(xsf.bands) >= 1:
                    raise ValueError("Multiple BANDGRID blocks are not supported")

                block_labels = []
                block_data = []
                block_origin = None
                block_lattice = None
                continue

            if keyword.startswith(b"END_BLOCK_BANDGRID"):
                if block_name is None:
                    raise ValueError("END_BLOCK_BANDGRID encountered without a matching BEGIN_BLOCK_BANDGRID")
                if block_origin is None or block_lattice is None:
                    raise ValueError("BANDGRID block is missing required origin or lattice information")
                if xsf.fermi_energy is None:
                    raise ValueError("BANDGRID block is missing required Fermi energy from BEGIN_INFO section")
                xsf.bands[block_name] = XSFBand(
                    data=np.asarray(block_data),
                    lattice=block_lattice,
                    origin=block_origin,
                    comment="",
                    labels=block_labels,
                )
                block_name = None
                block_dim = None
                block_labels = []
                block_data = []
                block_origin = None
                block_lattice = None
                continue

            if keyword.startswith(b"BEGIN_BANDGRID_"):
                if block_name is None:
                    raise ValueError("BEGIN_BANDGRID encountered without a matching BEGIN_BLOCK_BANDGRID")
                line = line.removeprefix(b"BEGIN_BANDGRID_")
                grid_dim, grid_name = line.split(b"_", maxsplit=1)
                grid_dim = int(grid_dim.rstrip(b"D"))
                if grid_dim != 3:
                    raise ValueError(
                        f"Declared BANDGRID dimension {grid_dim}D does not match expected 3D for band grids"
                    )
                if grid_name == b"":
                    grid_name = f"UNKBAND{len(block_labels)}".encode()

                n_bands = int(file.readline().strip())
                block_shape = np.loadtxt(file, max_rows=1, dtype=int)
                if block_origin is None:
                    block_origin = np.loadtxt(file, max_rows=1)
                elif not np.allclose(block_origin, np.loadtxt(file, max_rows=1)):
                    raise ValueError("Inconsistent BANDGRID origin within the same block")

                if block_lattice is None:
                    block_lattice = np.loadtxt(file, max_rows=3)
                elif not np.allclose(block_lattice, np.loadtxt(file, max_rows=3)):
                    raise ValueError("Inconsistent BANDGRID lattice within the same block")

                for i in range(n_bands):
                    line = file.readline().strip()
                    if not line.startswith(b"BAND:"):
                        if not line or line.startswith((b"END_BANDGRID", b"END_BLOCK_BANDGRID")):
                            raise ValueError(f"Expected {n_bands} bands but parsed {len(block_labels)}")
                        raise ValueError(f"Expected BAND header for band {i} but got: {line}")
                    band_name = line.lstrip(b"BAND:").strip()
                    if band_name == b"":
                        band_name = f"UNK{i}".encode()
                    data = np.empty(block_shape, dtype=np.float64)
                    parse_num = parse_n_doubles(file, data.ravel(order="C"), nelem=np.prod(block_shape))
                    if parse_num != np.prod(block_shape):
                        raise ValueError(f"Expected {np.prod(block_shape)} band energy values but parsed {parse_num}")

                    block_labels.append(band_name.decode("utf-8"))
                    block_data.append(data)

                if len(block_labels) != n_bands:
                    raise ValueError(f"Expected {n_bands} bands but parsed {len(block_labels)}")

                continue

            if keyword.startswith(b"END_BANDGRID"):
                if block_origin is None or block_lattice is None:
                    raise ValueError("BANDGRID block is missing required origin or lattice information")
                if xsf.fermi_energy is None:
                    raise ValueError("BANDGRID block is missing required Fermi energy from BEGIN_INFO section")
                continue

            raise ValueError(f"Unsupported or misplaced XSF keyword: {line}")

        xsf.comment = b"\n".join(comments).decode("utf-8")

        if xsf.structure is None and not xsf.grids and not xsf.bands:
            raise ValueError("No data parsed from XSF file")

        return xsf


@dataclass(eq=False, slots=True)
class AnimatedXSF:
    """XCrySDen animated XSF trajectory adapter.

    Args:
        data: Optional list of parsed XSF frames.
    """

    data: list[XSF] = field(default_factory=list)

    @classmethod
    def from_file(cls, filename: PathLike) -> Self:
        """Read an animated XSF file.

        Args:
            filename: Source filename.

        Returns:
            Parsed animated XSF adapter.
        """
        with zopen(filename, mode="rb") as file:
            return cls.parse_file(file)

    @classmethod
    def from_str(cls, input_string: str) -> Self:
        """Read an animated XSF string.

        Args:
            input_string: AXSF text.

        Returns:
            Parsed animated XSF adapter.
        """
        return cls.parse_file(BytesIO(input_string.encode("utf-8")))

    @classmethod
    def parse_file(cls, file) -> Self:
        """Parse an animated XSF binary stream or file stream.

        Args:
            file: Seekable binary stream with a ``readline`` method that returns ``bytes``.

        Returns:
            AnimatedXSF object containing parsed frames and metadata.

        """
        if hasattr(file, "buffer"):
            file = file.buffer

        raise NotImplementedError("Parsing of AnimatedXSF files is not implemented yet")

    def __len__(self) -> int:
        """Number of frames in the trajectory."""
        return len(self.data)

    def __getitem__(self, index: int | slice):
        """Get a specific frame from the trajectory."""
        if isinstance(index, int):
            return self.data[index]
        return self.__class__(self.data[index])

    def as_trajectory(self) -> Trajectory:
        """Convert periodic AXSF frames to a pymatgen trajectory."""
        raise NotImplementedError("Conversion from AnimatedXSF to Trajectory is not implemented yet")
