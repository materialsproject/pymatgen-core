"""Classes for parsing and manipulating VASP optical properties calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import numpy as np
import scipy.special
from monty.json import MSONable

from pymatgen.core import constants as _const
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun, Waveder

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import ArrayLike, NDArray

    from pymatgen.util.typing import PathLike

__author__ = "Jimmy-Xuan Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen"
__email__ = "jmmshn@gmail.com"

au2ang: float = _const.physical_constants["atomic unit of length"][0] / 1e-10
ryd2ev: float = _const.physical_constants["Rydberg constant times hc in eV"][0]
edeps: float = 4 * np.pi * 2 * ryd2ev * au2ang  # from constant.inc in VASP

KB: float = _const.physical_constants["Boltzmann constant in eV/K"][0]


@dataclass
class DielectricFunctionCalculator(MSONable):
    """Post-process VASP optical properties calculations.

    This objects helps load the different parameters from the vasprun.xml file but allows users to override
    them as needed.

    The standard vasprun.xml from an ``LOPTICS=.True.`` calculation already contains
    the complex frequency dependent dielectric functions.  However you have no way to decompose
    the different contributions.  Since the ``WAVEDER`` file is also written during an optical calculation,
    you can reconstruct the dielectric functions purely in Python and have full control over contribution
    from different bands and k-points.

    VASP's linear optics follow these steps:
        - Calculate the imaginary part
        - Perform symmetry operations (this is not implemented here)
        - Calculate the real part

    Currently, this Calculator only works for ``ISYM=0`` calculations since we cannot guarantee that our
    externally defined symmetry operations are the same as VASP's. This can be fixed by printing the
    symmetry operators into the vasprun.xml file. If this happens in future versions of VASP,
    we can dramatically speed up the calculations here by considering only the irreducible kpoints.
    """

    cder_real: NDArray
    cder_imag: NDArray
    eigs: NDArray
    kweights: NDArray
    nedos: int
    deltae: float
    ismear: int
    sigma: float
    efermi: float
    cshift: float
    ispin: int
    volume: float

    @classmethod
    def from_vasp_objects(cls, vrun: Vasprun, waveder: Waveder) -> Self:
        """Construct a DielectricFunction from Vasprun, Kpoint, and Waveder.

        Args:
            vrun: Vasprun object
            kpoint: Kpoint object
            waveder: Waveder object
        """
        bands = vrun.eigenvalues
        if bands is None:
            raise RuntimeError("eigenvalues cannot be None.")

        sspins = [Spin.up, Spin.down]
        eigs = np.stack([bands[spin] for spin in sspins[: vrun.parameters["ISPIN"]]], axis=2)[..., 0]
        eigs = np.swapaxes(eigs, 0, 1)
        kweights = vrun.actual_kpoints_weights
        nedos = vrun.parameters["NEDOS"]
        deltae = vrun.dielectric[0][1]
        ismear = vrun.parameters["ISMEAR"]
        sigma = vrun.parameters["SIGMA"]
        cshift = vrun.parameters["CSHIFT"]
        efermi = vrun.efermi
        ispin = vrun.parameters["ISPIN"]
        volume = vrun.final_structure.volume
        if vrun.parameters["ISYM"] != 0:
            raise NotImplementedError("ISYM != 0 is not implemented yet")

        if efermi is None:
            raise ValueError("efermi cannot be None.")

        return cls(
            cder_real=waveder.cder_real,
            cder_imag=waveder.cder_imag,
            eigs=eigs,
            kweights=kweights,  # type:ignore[arg-type]
            nedos=nedos,
            deltae=deltae,
            ismear=ismear,
            sigma=sigma,
            efermi=efermi,
            cshift=cshift,
            ispin=ispin,
            volume=volume,
        )

    @classmethod
    def from_directory(cls, directory: PathLike) -> Self:
        """Construct a DielectricFunction from a directory containing
        vasprun.xml and WAVEDER files.
        """

        def _try_reading(dtypes):
            """Return None if failed."""
            for dtype in dtypes:
                try:
                    return Waveder.from_binary(f"{directory}/WAVEDER", data_type=dtype)
                except ValueError as exc:
                    if "reshape" in str(exc):
                        continue
                    raise
            return None

        vrun = Vasprun(f"{directory}/vasprun.xml")
        if "gamma" in vrun.generator["subversion"].lower():
            waveder = _try_reading(["float64", "float32"])  # large one first should give value error
        else:
            waveder = _try_reading(["complex128", "complex64"])
        return cls.from_vasp_objects(vrun, waveder)

    @property
    def cder(self):
        """Complex CDER from WAVEDER."""
        return self.cder_real + self.cder_imag * 1.0j

    def get_epsilon(
        self,
        idir: int,
        jdir: int,
        efermi: float | None = None,
        nedos: int | None = None,
        deltae: float | None = None,
        ismear: int | None = None,
        sigma: float | None = None,
        cshift: float | None = None,
        mask: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Compute the frequency dependent dielectric function.

        Args:
            idir: First direction of the dielectric tensor
            jdir: Second direction of the dielectric tensor
            efermi: Fermi energy
            nedos: Number of points in the DOS
            deltae: Energy step in the DOS
            ismear: Smearing method (only has 0:gaussian, >0:Methfessel-Paxton)
            sigma: Smearing width
            cshift: Complex shift used for Kramer-Kronig transformation
            mask: Mask for the bands/kpoint/spin index to include in the calculation
        """

        @overload
        def _use_default(param: int | None, default: int) -> int:
            pass

        @overload
        def _use_default(param: float | None, default: float) -> float:
            pass

        def _use_default(param: float | None, default: float) -> float:
            return param if param is not None else default

        _efermi = _use_default(efermi, self.efermi)
        _nedos = _use_default(nedos, self.nedos)
        _deltae = _use_default(deltae, self.deltae)
        _ismear = _use_default(ismear, self.ismear)
        _sigma = _use_default(sigma, self.sigma)
        _cshift = _use_default(cshift, self.cshift)

        egrid, eps_imag = epsilon_imag(
            cder=self.cder,
            eigs=self.eigs,
            kweights=self.kweights,
            efermi=_efermi,
            nedos=_nedos,
            deltae=_deltae,
            ismear=_ismear,
            sigma=_sigma,
            idir=idir,
            jdir=jdir,
            mask=mask,
        )
        # scaling constant: edeps * np.pi / structure.volume
        eps_in = eps_imag * edeps * np.pi / self.volume
        eps = kramers_kronig(eps_in, nedos=_nedos, deltae=_deltae, cshift=_cshift)
        if idir == jdir:
            eps += 1.0 + 0.0j
        return egrid, eps

    def plot_weighted_transition_data(
        self,
        idir: int,
        jdir: int,
        mask: NDArray | None = None,
        min_val: float = 0.0,
    ):
        """Data for plotting the weight matrix elements as a scatter plot.

        Since the computation of the final spectrum (especially the smearing part)
        is still fairly expensive.  This function can be used to check the values
        of some portion of the spectrum (defined by the mask).
        In a sense, we are looking at the imaginary part of the dielectric function
        before the smearing is applied.

        Args:
            idir: First direction of the dielectric tensor.
            jdir: Second direction of the dielectric tensor.
            mask: Mask to apply to the CDER for the bands/kpoint/spin
                index to include in the calculation
            min_val: Minimum value below this value the matrix element will not be shown.
        """
        cderm = self.cder * mask if mask is not None else self.cder

        # Limit the first two indices based on the mask
        try:
            min_band0, min_band1, weight, decel, A = _get_transitions(
                cderm, self.eigs, self.kweights, self.efermi, self.sigma, self.ismear, idir, jdir
            )
        except ValueError as exc:
            if "zero-size array" in str(exc):
                raise ValueError("No matrix elements found. Check the mask.")
            raise

        matrix_el = np.abs(A) * weight  # can have negative weight due to fermi function
        x_val = []
        y_val = []
        text = []
        for ib, jb, ik, ispin in zip(*np.nonzero(matrix_el > min_val), strict=True):
            x_val.append(decel[ib, jb, ik, ispin])
            y_val.append(matrix_el[ib, jb, ik, ispin])
            text.append(f"s:{ispin}, k:{ik}, {min_band0 + ib} -> {min_band1 + jb} ({decel[ib, jb, ik, ispin]:.2f})")
        return x_val, y_val, text


def delta_methfessel_paxton(x: NDArray, n: int) -> NDArray:
    """
    D_n (x) = exp -x^2 * sum_i=0^n A_i H_2i(x)
    where H is a Hermite polynomial and
    A_i = (-1)^i / ( i! 4^i sqrt(pi) ).
    """
    ii = np.arange(0, n + 1)
    A = (-1) ** ii / (scipy.special.factorial(ii) * 4**ii * np.sqrt(np.pi))
    H = scipy.special.eval_hermite(ii * 2, np.tile(x, (len(ii), 1)).T)
    return np.exp(-(x * x)) * np.dot(A, H.T)


def step_methfessel_paxton(x: NDArray, n: int) -> NDArray:
    """
    S_n (x) = (1 + erf x)/2 - exp -x^2 * sum_i=1^n A_i H_{2i-1}(x)
    where H is a Hermite polynomial and
    A_i = (-1)^i / ( i! 4^i sqrt(pi) ).
    """
    ii = np.arange(1, n + 1)
    A = (-1) ** ii / (scipy.special.factorial(ii) * 4**ii * np.sqrt(np.pi))
    H = scipy.special.eval_hermite(ii * 2 - 1, np.tile(x, (len(ii), 1)).T)
    return (1.0 + scipy.special.erf(x)) / 2.0 - np.exp(-(x * x)) * np.dot(A, H.T)


def delta_func(x: NDArray, ismear: int) -> NDArray:
    """Replication of VASP's delta function."""
    if ismear < -1:
        raise ValueError("Delta function not implemented for ismear < -1")
    if ismear == -1:
        return step_func(x, -1) * (1 - step_func(x, -1))
    if ismear == 0:
        return np.exp(-(x * x)) / np.sqrt(np.pi)
    return delta_methfessel_paxton(x, ismear)


def step_func(x: NDArray, ismear: int) -> NDArray:
    """Replication of VASP's step function."""
    if ismear < -1:
        raise ValueError("Delta function not implemented for ismear < -1")
    if ismear == -1:
        return 1 / (1.0 + np.exp(-x))
    if ismear == 0:
        return 0.5 + 0.5 * scipy.special.erf(x)
    return step_methfessel_paxton(x, ismear)


def get_delta(x0: float, sigma: float, nx: int, dx: float, ismear: int = 3) -> NDArray:
    """Get the smeared delta function to be added to form the spectrum.

    This replaces the `SLOT` function from VASP. Uses finite differences instead of
    evaluating the delta function since the step function is more likely to have analytic form.

    Args:
        x0: The center of the dielectric function.
        sigma: The width of the smearing
        nx: The number of grid points in the output grid.
        dx: The gridspacing of the output grid.
        ismear: The smearing parameter used by the ``step_func``.

    Returns:
        np.ndarray: Array of size `nx` with delta function on the desired outputgrid.
    """
    xgrid = np.linspace(0, nx * dx, nx, endpoint=False)
    xgrid -= x0
    x_scaled = (xgrid + (dx / 2)) / sigma
    sfun = step_func(x_scaled, ismear)
    dfun = np.zeros_like(xgrid)
    dfun[1:] = (sfun[1:] - sfun[:-1]) / dx
    return dfun


def get_step(x0: float, sigma: float, nx: int, dx: float, ismear: int) -> NDArray:
    """Get the smeared step function to be added to form the spectrum.

    This replaces the `SLOT` function from VASP.

    Args:
        x0: The center of the dielectric function.
        sigma: The width of the smearing
        nx: The number of grid points in the output grid.
        dx: The gridspacing of the output grid.
        ismear: The smearing parameter used by the ``step_func``.

    Returns:
        np.ndarray: Array of size `nx` with step function on the desired outputgrid.
    """
    xgrid = np.linspace(0, nx * dx, nx, endpoint=False)
    xgrid -= x0
    x_scaled = (xgrid + (dx / 2)) / sigma
    return step_func(x_scaled, ismear)


def epsilon_imag(
    cder: NDArray,
    eigs: NDArray,
    kweights: ArrayLike,
    efermi: float,
    nedos: int,
    deltae: float,
    ismear: int,
    sigma: float,
    idir: int,
    jdir: int,
    mask: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Replicate the EPSILON_IMAG function of VASP.

    Args:
        cder: The data written to the WAVEDER (nbands, nbands, nkpoints, nspin, diri, dirj)
        eigs: The eigenvalues (nbands, nkpoints, nspin)
        kweights: The kpoint weights (nkpoints)
        efermi: The fermi energy
        nedos: The sampling of the energy values
        deltae: The energy grid spacing
        ismear: The smearing parameter used by the ``step_func``.
        sigma: The width of the smearing
        idir: The first direction of the dielectric tensor
        jdir: The second direction of the dielectric tensor
        mask: Mask for the bands/kpoint/spin index to include in the calculation

    Returns:
        np.ndarray: Array of size `nedos` with the imaginary part of the dielectric function.
    """
    egrid = np.linspace(0, nedos * deltae, nedos, endpoint=False)
    epsdd = np.zeros_like(egrid, dtype=np.complex128)

    cderm = cder * mask if mask is not None else cder
    try:
        _, _, weight, decel, A = _get_transitions(cderm, eigs, kweights, efermi, sigma, ismear, idir, jdir)
    except ValueError as exc:
        if "zero-size array" in str(exc):
            return egrid, epsdd
        raise

    # Reproduce the `SLOT` function calls in VASP:
    # CALL SLOT( REAL(DECEL,q), ISMEAR, SIGMA, NEDOS, DELTAE,  WEIGHT*A*CONST, EPSDD)
    # The conjugate part is not needed since we are running over all pairs of ib, jb
    # vasp just does the conjugate trick to save loop time.
    # Transitions with zero weight or matrix element contribute nothing, so skip them.
    amplitude = (weight * A).ravel()
    nonzero = amplitude != 0
    amplitude, decel = amplitude[nonzero], decel.ravel()[nonzero]

    # Smear all transitions at once, in chunks to bound the (n_transitions, nedos) work arrays.
    # This is the vectorized equivalent of summing get_delta(decel, ...) * weight * A.
    chunk_size = max(1, 2**20 // nedos)
    for start in range(0, len(amplitude), chunk_size):
        x0 = decel[start : start + chunk_size, None]
        x_scaled = (egrid - x0 + (deltae / 2)) / sigma
        sfun = step_func(x_scaled.ravel(), ismear).reshape(x_scaled.shape)
        dfun = np.zeros_like(x_scaled)
        dfun[:, 1:] = (sfun[:, 1:] - sfun[:, :-1]) / deltae
        epsdd += amplitude[start : start + chunk_size] @ dfun
    return egrid, epsdd


def _get_transitions(
    cderm: NDArray,
    eigs: NDArray,
    kweights: ArrayLike,
    efermi: float,
    sigma: float,
    ismear: int,
    idir: int,
    jdir: int,
) -> tuple[int, int, NDArray, NDArray, NDArray]:
    """Get the occupation weights, transition energies and matrix elements of all transitions.

    The first two (band) indices are limited to the range in which ``cderm`` is nonzero.

    Args:
        cderm: The (masked) data written to the WAVEDER (nbands, nbands, nkpoints, nspin, diri, dirj)
        eigs: The eigenvalues (nbands, nkpoints, nspin)
        kweights: The kpoint weights (nkpoints)
        efermi: The fermi energy
        sigma: The width of the smearing
        ismear: The smearing parameter used by the ``step_func``.
        idir: The first direction of the dielectric tensor
        jdir: The second direction of the dielectric tensor

    Raises:
        ValueError: With a "zero-size array" message if ``cderm`` is all zero.

    Returns:
        tuple: The first included band index along each of the two band axes, and the weight,
            transition energy and matrix element arrays, each of shape (nband0, nband1, nkpoints, nspin).
    """
    nonzero = np.nonzero(cderm)
    min_band0, max_band0 = np.min(nonzero[0]), np.max(nonzero[0])
    min_band1, max_band1 = np.min(nonzero[1]), np.max(nonzero[1])
    bands0 = slice(min_band0, max_band0 + 1)
    bands1 = slice(min_band1, max_band1 + 1)

    kweights = np.array(kweights)
    norm_kweights = kweights / np.sum(kweights)
    rspin = 3 - cderm.shape[3]
    eigs_shifted = eigs - efermi
    fermi_w = step_func((eigs_shifted / sigma).ravel(), ismear).reshape(eigs.shape)

    # For the transition between two bands at one kpoint the contribution is:
    #  (fermi[band_j] - fermi[band_i]) * rspin * normalized_kpoint_weight
    weight = (fermi_w[None, bands1] - fermi_w[bands0, None]) * rspin * norm_kweights[None, None, :, None]
    decel = eigs[None, bands1] - eigs[bands0, None]
    A = cderm[bands0, bands1, :, :, idir] * np.conjugate(cderm[bands0, bands1, :, :, jdir])
    return min_band0, min_band1, weight, decel, A


def kramers_kronig(
    eps: NDArray,
    nedos: int,
    deltae: float,
    cshift: float = 0.1,
) -> NDArray:
    """Perform the Kramers-Kronig transformation.

    Perform the Kramers-Kronig transformation exactly as VASP does it.
    The input eps should be complex and the imaginary part of the dielectric function
    should be stored as the real part of the complex input array.
    The output should be the complex dielectric function.

    Args:
        eps: The dielectric function with the imaginary part stored as the real part
            and nothing in the imaginary part.
        nedos: The sampling of the energy values
        deltae: The energy grid spacing
        cshift: The shift of the imaginary part of the dielectric function.

    Returns:
        np.ndarray: Array of size `nedos` with the complex dielectric function.
    """
    egrid = np.linspace(0, deltae * nedos, nedos)
    csfhit = cshift * 1.0j
    cdiff = np.subtract.outer(egrid, egrid) + csfhit
    csum = np.add.outer(egrid, egrid) + csfhit
    vals = -0.5 * ((eps / cdiff) - (np.conj(eps) / csum))
    return np.sum(vals, axis=1) * 2 / np.pi * deltae
