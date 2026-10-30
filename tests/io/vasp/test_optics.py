from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.special
from numpy.testing import assert_allclose
from pytest import approx

from pymatgen.io.vasp.optics import (
    DielectricFunctionCalculator,
    delta_func,
    delta_methfessel_paxton,
    epsilon_imag,
    get_delta,
    step_func,
)
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

__author__ = "Jimmy-Xuan Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__email__ = "jmmshn@gmail.com"

TEST_DIR = f"{TEST_FILES_DIR}/io/vasp"


class TestVasprun(MatSciTest):
    def test_optics(self):
        eps_data_path = f"{TEST_DIR}/fixtures/reproduce_eps"
        vrun = Vasprun(f"{eps_data_path}/vasprun.xml")
        dfc = DielectricFunctionCalculator.from_directory(eps_data_path)
        egrid, eps = dfc.get_epsilon(0, 0)

        assert egrid[0] == approx(0)
        assert egrid[-1] == approx(59.3802)
        assert len(egrid) == len(eps) == 3000

        _, eps_real_ref, eps_imag_ref = vrun.dielectric
        eps_real_ref = np.array(eps_real_ref)[:, 0]
        eps_imag_ref = np.array(eps_imag_ref)[:, 0]
        assert np.max(np.abs(eps_real_ref - np.real(eps))) / np.max(np.abs(np.real(eps))) < 0.01

        # masking with all zeros
        mask = np.zeros_like(dfc.cder, dtype=float)
        _, eps = dfc.get_epsilon(0, 0, mask=mask)
        assert np.max(np.imag(eps)) < 1e-10

        # masking with all ones
        mask = np.ones_like(dfc.cder, dtype=float)
        _, eps = dfc.get_epsilon(0, 0, mask=mask)
        assert np.max(np.abs(eps_real_ref - np.real(eps))) / np.max(np.abs(np.real(eps))) < 0.01

        _, eps = dfc.get_epsilon(0, 1)
        assert np.max(np.abs(eps)) < 0.1

        # plotting
        mask = np.ones_like(dfc.cder, dtype=float)
        x_val, y_val, text = dfc.plot_weighted_transition_data(0, 0, mask=mask, min_val=0.001)
        assert len(x_val) == len(y_val) == len(text)


def test_delta_func():
    x = np.array([0, 1, 2, 3, 4, 5])

    # ismear < -1
    with pytest.raises(ValueError, match="Delta function not implemented for ismear < -1"):
        delta_func(x, -2)

    # ismear == -1
    assert np.all(delta_func(x, -1) == step_func(x, -1) * (1 - step_func(x, -1)))

    # ismear == 0
    assert np.all(delta_func(x, 0) == np.exp(-(x * x)) / np.sqrt(np.pi))

    # ismear > 0
    for ismear in [1, 2, 3]:
        assert np.all(delta_func(x, ismear) == delta_methfessel_paxton(x, ismear))


def test_step_func():
    # array of positive values
    x = np.array([1, 2, 3, 4, 5])
    assert_allclose(step_func(x, -1), 1 / (1.0 + np.exp(-x)))

    # array of negative values
    x = np.array([-1, -2, -3, -4, -5])
    assert_allclose(step_func(x, -1), 1 / (1.0 + np.exp(-x)))

    # array that includes zero
    x = np.array([-1, 0, 1])
    assert_allclose(step_func(x, -1), 1 / (1.0 + np.exp(-x)))

    # ismear == 0
    x = np.array([1, 2, 3, 4, 5])
    assert_allclose(step_func(x, 0), 0.5 + 0.5 * scipy.special.erf(x))


def _make_random_transitions(nspin: int = 2):
    rng = np.random.default_rng(42)
    nband, nk = 6, 3
    cder = rng.normal(size=(nband, nband, nk, nspin, 3)) + 1j * rng.normal(size=(nband, nband, nk, nspin, 3))
    # Zero out the first band entirely so the band window is restricted
    cder[0] = 0
    cder[:, 0] = 0
    eigs = np.sort(rng.uniform(-5, 5, size=(nband, nk, nspin)), axis=0)
    kweights = rng.uniform(1, 2, size=nk)
    return cder, eigs, kweights


@pytest.mark.parametrize("ismear", [-1, 0, 2])
@pytest.mark.parametrize("nspin", [1, 2])
def test_epsilon_imag_matches_reference_loop(ismear, nspin):
    """The vectorized epsilon_imag must reproduce VASP's per-transition SLOT accumulation."""
    cder, eigs, kweights = _make_random_transitions(nspin)
    efermi, nedos, deltae, sigma, idir, jdir = 0.3, 200, 0.06, 0.2, 0, 1

    norm_kweights = kweights / np.sum(kweights)
    rspin = 3 - nspin
    ref = np.zeros(nedos, dtype=np.complex128)
    nband, _, nk = cder.shape[:3]
    for ib, jb, ik, ispin in itertools.product(range(1, nband), range(1, nband), range(nk), range(nspin)):
        fermi_w_i = step_func(np.array([(eigs[ib, ik, ispin] - efermi) / sigma]), ismear)[0]
        fermi_w_j = step_func(np.array([(eigs[jb, ik, ispin] - efermi) / sigma]), ismear)[0]
        weight = (fermi_w_j - fermi_w_i) * rspin * norm_kweights[ik]
        decel = eigs[jb, ik, ispin] - eigs[ib, ik, ispin]
        A = cder[ib, jb, ik, ispin, idir] * np.conjugate(cder[ib, jb, ik, ispin, jdir])
        ref += get_delta(x0=decel, sigma=sigma, nx=nedos, dx=deltae, ismear=ismear) * weight * A

    egrid, epsdd = epsilon_imag(cder, eigs, kweights, efermi, nedos, deltae, ismear, sigma, idir, jdir)
    assert_allclose(egrid, np.linspace(0, nedos * deltae, nedos, endpoint=False))
    assert_allclose(epsdd, ref, rtol=1e-12, atol=1e-12 * np.max(np.abs(ref)))

    # An all-zero mask gives an all-zero spectrum
    _, epsdd = epsilon_imag(
        cder, eigs, kweights, efermi, nedos, deltae, ismear, sigma, idir, jdir, mask=np.zeros(cder.shape)
    )
    assert not np.any(epsdd)


def test_plot_weighted_transition_data_matches_reference_loop():
    cder, eigs, kweights = _make_random_transitions()
    dfc = DielectricFunctionCalculator(
        cder_real=cder.real,
        cder_imag=cder.imag,
        eigs=eigs,
        kweights=kweights,
        nedos=100,
        deltae=0.1,
        ismear=0,
        sigma=0.5,
        efermi=0.3,
        cshift=0.1,
        ispin=2,
        volume=10.0,
    )
    min_val = 0.05
    norm_kweights = kweights / np.sum(kweights)
    ref_x, ref_y, ref_text = [], [], []
    nband, _, nk, nspin = cder.shape[:4]
    for ib, jb, ik, ispin in itertools.product(range(1, nband), range(1, nband), range(nk), range(nspin)):
        fermi_w_i = step_func((eigs[ib, ik, ispin] - dfc.efermi) / dfc.sigma, dfc.ismear)
        fermi_w_j = step_func((eigs[jb, ik, ispin] - dfc.efermi) / dfc.sigma, dfc.ismear)
        weight = (fermi_w_j - fermi_w_i) * (3 - nspin) * norm_kweights[ik]
        A = cder[ib, jb, ik, ispin, 0] * np.conjugate(cder[ib, jb, ik, ispin, 0])
        decel = eigs[jb, ik, ispin] - eigs[ib, ik, ispin]
        matrix_el = np.abs(A) * float(weight)
        if matrix_el > min_val:
            ref_x.append(decel)
            ref_y.append(matrix_el)
            ref_text.append(f"s:{ispin}, k:{ik}, {ib} -> {jb} ({decel:.2f})")

    x_val, y_val, text = dfc.plot_weighted_transition_data(0, 0, min_val=min_val)
    assert len(ref_x) > 0
    assert text == ref_text
    assert_allclose(x_val, ref_x)
    assert_allclose(y_val, ref_y, rtol=1e-12)

    with pytest.raises(ValueError, match="No matrix elements found"):
        dfc.plot_weighted_transition_data(0, 0, mask=np.zeros(cder.shape))
