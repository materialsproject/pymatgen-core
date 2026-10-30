from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

from pymatgen.electronic_structure.core import Spin
from pymatgen.io.lobster.future.constants import LOBSTER_ORBITALS

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def natural_sort(list_to_sort: str) -> list[Any]:
    """Sort a list of strings in human order.

    This function sorts strings in a way that humans would expect,
    taking into account numerical values within the strings.

    Args:
        list_to_sort (str): List of strings to sort.

    Returns:
        list[Any]: Sorted list of strings in human order.

    Example:
        >>> natural_sort(["file10", "file2", "file1"])
        ['file1', 'file2', 'file10']
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", list_to_sort)]


def parse_orbital_from_text(text: str) -> str | None:
    """Parse the orbital from a text string of an ICOXXLIST file.

    This function extracts the orbital information from a given text string.
    It checks for valid orbital patterns and returns the matched orbital.

    Args:
        text (str): Text string to parse the orbital from.

    Returns:
        str | None: Parsed orbital string if a valid orbital is found,
        otherwise None.

    Example:
        >>> parse_orbital_from_text("1s_2p_x")
        '2p_x'
    """
    parts = text.split("_")

    if len(parts) == 1:
        return None

    for orbital in LOBSTER_ORBITALS:
        if match := re.search(rf"\d+{re.escape(orbital)}", "_".join(parts[-2:])):
            return match.group(0)

    return parts[-1] if re.match(r"\d+[a-z]+", parts[-1]) else None


def convert_spin_keys(obj: Any) -> Any:
    """Convert an object to a JSON-compatible format recursively.

    This function ensures that the input object is converted into a format
    that can be serialized into JSON. It handles lists, tuples, dictionaries,
    and enums.

    Args:
        obj (Any): Input object to convert.

    Returns:
        Any: JSON-compatible representation of the input object.

    Example:
        >>> make_json_compatible({"key": Enum("Example", "value")})
        {'key': 'value'}
    """
    if isinstance(obj, (list, tuple)):
        return [convert_spin_keys(item) for item in obj]

    if isinstance(obj, dict):
        new_dict = {}

        for k, v in obj.items():
            new_key = f"@Spin({k})" if isinstance(k, Spin) else k
            new_dict[new_key] = convert_spin_keys(v)

        return new_dict

    return obj


def restore_spin_keys(obj: Any) -> Any:
    """Restore Spin enum keys from JSON-serialized format.

    This function recursively processes an object and converts string keys
    "1" and "-1" back to Spin.up and Spin.down enum values. It handles
    nested lists, tuples, and dictionaries.

    Args:
        obj (Any): Input object with potential Spin keys as strings.

    Returns:
        Any: Object with Spin enum keys restored.

    Example:
        >>> restore_spin_keys({"1": [1, 2], "-1": [3, 4]})
        {<Spin.up: 1>: [1, 2], <Spin.down: -1>: [3, 4]}
    """
    if isinstance(obj, (list, tuple)):
        return type(obj)(restore_spin_keys(item) for item in obj)

    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if k == "@Spin(1)":
                new_key = Spin.up
            elif k == "@Spin(-1)":
                new_key = Spin.down
            else:
                new_key = k

            new_dict[new_key] = restore_spin_keys(v)

        return new_dict

    return obj


def _wavefunction_on_periodic_grid(
    grid: tuple[int, int, int],
    lattice_matrix: ArrayLike,
    points: ArrayLike,
    reals: ArrayLike,
    imaginaries: ArrayLike,
) -> tuple[NDArray, NDArray, NDArray]:
    """Map a LOBSTER real-space wave function onto the periodic grid used by VolumetricData.

    LOBSTER writes the wave function on a (Nx+1, Ny+1, Nz+1) grid that includes the periodic images of the
    first grid point along each axis. These are dropped and the remaining points are checked against the
    expected grid positions.

    Args:
        grid (tuple[int, int, int]): Grid on which the wave function was calculated, (Nx+1, Ny+1, Nz+1).
        lattice_matrix (ArrayLike): 3x3 lattice matrix of the structure.
        points (ArrayLike): Cartesian coordinates of all grid points, in file order.
        reals (ArrayLike): Real parts of the wave function, in file order.
        imaginaries (ArrayLike): Imaginary parts of the wave function, in file order.

    Returns:
        tuple[NDArray, NDArray, NDArray]: Real part, imaginary part and density, each of shape (Nx, Ny, Nz).

    Raises:
        ValueError: If the wave function does not cover the full unit cell.
    """
    shape = tuple(int(n) for n in grid)
    n_x, n_y, n_z = (n - 1 for n in shape)
    frac_axes = [np.arange(n + 1) / float(n) for n in (n_x, n_y, n_z)]
    frac_coords = np.stack(np.meshgrid(*frac_axes, indexing="ij"), axis=-1)[:n_x, :n_y, :n_z]
    expected = frac_coords.reshape(-1, 3) @ np.asarray(lattice_matrix)
    actual = np.asarray(points, dtype=np.float64).reshape(*shape, 3)[:n_x, :n_y, :n_z].reshape(-1, 3)

    # A point is only rejected if none of its three coordinates matches the expected position
    if np.any(np.all(~np.isclose(actual, expected, rtol=1e-3), axis=1)):
        raise ValueError(
            "The provided wavefunction from Lobster does not contain all relevant"
            " points. "
            "Please use a line similar to: printLCAORealSpaceWavefunction kpoint 1 "
            "coordinates 0.0 0.0 0.0 coordinates 1.0 1.0 1.0 box bandlist 1 "
        )

    real = np.asarray(reals, dtype=np.float64).reshape(shape)[:n_x, :n_y, :n_z].copy()
    imaginary = np.asarray(imaginaries, dtype=np.float64).reshape(shape)[:n_x, :n_y, :n_z].copy()
    return real, imaginary, real**2 + imaginary**2
