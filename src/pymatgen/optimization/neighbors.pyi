import numpy as np
from numpy.typing import NDArray

def find_points_in_spheres(
    all_coords: NDArray[np.float64],
    center_coords: NDArray[np.float64],
    r: float,
    pbc: NDArray[np.int64],
    lattice: NDArray[np.float64],
    tol: float = ...,
    min_r: float = ...,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]: ...
