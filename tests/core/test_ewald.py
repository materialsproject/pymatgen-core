from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta
from itertools import combinations

import numpy as np
import pytest
from pytest import approx

from pymatgen.core import Lattice, Species
from pymatgen.core.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.core.structure import Structure
from pymatgen.util.testing import VASP_IN_DIR


class TestEwaldSummation:
    def setup_method(self):
        filepath = f"{VASP_IN_DIR}/POSCAR"
        self.original_struct = Structure.from_file(filepath)
        self.struct = self.original_struct.copy()
        self.struct.add_oxidation_state_by_element({"Li": 1, "Fe": 2, "P": 5, "O": -2})

    def test_init(self):
        ham = EwaldSummation(self.struct, compute_forces=True)
        assert ham.real_space_energy == approx(-502.23549897772602, abs=1e-4)
        assert ham.reciprocal_space_energy == approx(6.1541071599534654, abs=1e-4)
        assert ham.point_energy == approx(-620.22598358035918, abs=1e-4)
        assert ham.total_energy == approx(-1123.00766, abs=1e-1)
        assert ham.forces[0, 0] == approx(-1.98818620e-01, abs=1e-4)
        assert sum(sum(abs(ham.forces))) == approx(915.925354346, abs=1e-4), "Forces incorrect"
        assert sum(sum(ham.real_space_energy_matrix)) == approx(ham.real_space_energy, abs=1e-4)
        assert sum(sum(ham.reciprocal_space_energy_matrix)) == approx(ham.reciprocal_space_energy, abs=1e-4)
        assert sum(ham.point_energy_matrix) == approx(ham.point_energy, abs=1e-4)
        assert sum(sum(ham.total_energy_matrix)) + ham._charged_cell_energy == approx(ham.total_energy, abs=1e-2)

        with pytest.raises(
            ValueError,
            match="Ewald summation can only be performed on structures that are either oxidation state decorated",
        ):
            EwaldSummation(self.original_struct)
        # try sites with charge.
        charges = []
        for site in self.original_struct:
            if site.specie.symbol == "Li":
                charges.append(1)
            elif site.specie.symbol == "Fe":
                charges.append(2)
            elif site.specie.symbol == "P":
                charges.append(5)
            else:
                charges.append(-2)

        self.original_struct.add_site_property("charge", charges)
        ham2 = EwaldSummation(self.original_struct)
        assert ham2.real_space_energy == approx(-502.23549897772602, abs=1e-4)

    def test_from_dict(self):
        ham = EwaldSummation(self.struct, compute_forces=True)
        ham2 = EwaldSummation.from_dict(ham.as_dict())
        assert ham._real is None
        assert not ham._initialized
        assert ham2._real is None
        assert not ham2._initialized
        assert np.array_equal(ham.total_energy_matrix, ham2.total_energy_matrix)
        # check lazy eval
        assert ham.total_energy == approx(-1123.00766, abs=1e-1)
        assert ham._real is not None
        assert ham._initialized
        ham2 = EwaldSummation.from_dict(ham.as_dict())
        assert ham2._real is not None
        assert ham2._initialized
        assert np.array_equal(ham.total_energy_matrix, ham2.total_energy_matrix)

    def test_as_dict(self):
        ham = EwaldSummation(self.struct, compute_forces=True)
        dct = ham.as_dict()
        assert dct["compute_forces"]
        assert dct["eta"] == ham._eta
        assert dct["acc_factor"] == ham._acc_factor
        assert dct["real_space_cut"] == ham._rmax
        assert dct["recip_space_cut"] == ham._gmax
        assert ham.as_dict() == EwaldSummation.from_dict(dct).as_dict()

    @pytest.mark.parametrize("pbc", [(True, True, False), (False, False, False)])
    def test_non_3d_periodic_lattice(self, pbc):
        # The Ewald sum is only defined for 3D periodic systems (materialsproject/pymatgen#4718)
        struct = Structure(Lattice.cubic(4, pbc=pbc), ["Na+", "Cl-"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        with pytest.raises(ValueError, match="EwaldSummation requires a 3D periodic lattice"):
            EwaldSummation(struct)

    def test_charged_cell_with_site_charges(self):
        # materialsproject/pymatgen#4719: the neutralizing-background correction must also apply
        # when charges are given via the "charge" site property rather than oxidation states.
        lattice = Lattice.cubic(4)
        coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
        site_charged = Structure(lattice, ["Na", "Cl"], coords, site_properties={"charge": [1.0, 0.0]})
        oxi_charged = Structure(lattice, [Species("Na", 1), Species("Cl", 0)], coords)
        # Simple-cubic point-charge lattice in a neutralizing background (Madelung const. 2.837297)
        expected = -2.837297 / 2 * EwaldSummation.CONV_FACT / 4
        for eta in (0.22, 1.0, 2.2):
            # Charges given only via the "charge" site property are not checked against Structure.charge
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                e_site = EwaldSummation(site_charged, eta=eta).total_energy
            e_oxi = EwaldSummation(oxi_charged, eta=eta).total_energy
            assert e_site == approx(e_oxi, abs=1e-4)
            assert e_site == approx(expected, abs=1e-3)

    def test_charge_mismatch_warning(self):
        coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
        struct = Structure(Lattice.cubic(4), [Species("Na", 1), Species("Cl", -1)], coords, charge=1)
        neutral = Structure(Lattice.cubic(4), [Species("Na", 1), Species("Cl", -1)], coords)
        with (
            pytest.warns(UserWarning, match="Structure charge"),
            pytest.warns(UserWarning, match="inconsistent with structure.charge"),
        ):
            ewald = EwaldSummation(struct)
        # The site charges sum to zero, so no charged-cell correction is applied
        assert ewald.total_energy == approx(EwaldSummation(neutral).total_energy)


class TestEwaldMinimizer:
    def test_init(self):
        matrix = np.array(
            [
                [-3.0, 3.0, 4.0, -0.0, 3.0, 3.0, 1.0, 14.0, 9.0, -4.0],
                [1.0, -3.0, -3.0, 12.0, -4.0, -1.0, 5.0, 11.0, 1.0, 12.0],
                [14.0, 7.0, 13.0, 15.0, 13.0, 5.0, -5.0, 10.0, 14.0, -2.0],
                [9.0, 13.0, 4.0, 1.0, 3.0, -4.0, 7.0, 0.0, 6.0, -4.0],
                [4.0, -4.0, 6.0, 1.0, 12.0, -4.0, -2.0, 13.0, 0.0, 6.0],
                [13.0, 7.0, -4.0, 12.0, -2.0, 9.0, 8.0, -5.0, 3.0, 1.0],
                [8.0, 1.0, 10.0, -4.0, -2.0, 4.0, 13.0, 12.0, -3.0, 13.0],
                [2.0, 11.0, 8.0, 1.0, -1.0, 5.0, -3.0, 4.0, 5.0, 0.0],
                [-0.0, 14.0, 4.0, 3.0, -1.0, -5.0, 7.0, -1.0, -1.0, 3.0],
                [2.0, -2.0, 10.0, 1.0, 6.0, -5.0, -3.0, 12.0, 0.0, 13.0],
            ]
        )

        m_list = [[0.9, 4, [1, 2, 3, 4, 8], "a"], [-1, 2, [5, 6, 7], "b"]]

        e_min = EwaldMinimizer(matrix, m_list, 50)

        assert len(e_min.output_lists) == 15, "Wrong number of permutations returned"
        assert e_min.minimized_sum == approx(111.63, abs=1e-3), "Returned wrong minimum value"
        assert len(e_min.best_m_list) == 6, "Returned wrong number of permutations"

    def test_site(self):
        """Test that uses an uncharged structure."""
        filepath = f"{VASP_IN_DIR}/POSCAR"
        struct = Structure.from_file(filepath)
        struct = struct.copy()
        struct.add_oxidation_state_by_element({"Li": 1, "Fe": 3, "P": 5, "O": -2})

        # Comparison to LAMMPS result
        ham = EwaldSummation(struct, compute_forces=True)
        assert ham.total_energy == approx(-1226.3335, abs=1e-3)
        assert ham.get_site_energy(0) == approx(-45.8338, abs=1e-3)
        assert ham.get_site_energy(8) == approx(-27.2978, abs=1e-3)


@pytest.fixture
def nacl() -> Structure:
    """Conventional rocksalt NaCl decorated with oxidation states (8 sites)."""
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.69), [Species("Na", 1), Species("Cl", -1)], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


class TestEwaldSummationExtra:
    @pytest.mark.parametrize(
        "prop",
        [
            "real_space_energy",
            "real_space_energy_matrix",
            "reciprocal_space_energy",
            "reciprocal_space_energy_matrix",
            "point_energy",
            "point_energy_matrix",
            "total_energy_matrix",
        ],
    )
    def test_lazy_properties(self, nacl, prop):
        ewald = EwaldSummation(nacl)
        assert not ewald._initialized
        value = getattr(ewald, prop)
        assert ewald._initialized
        assert np.allclose(value, getattr(EwaldSummation(nacl), prop))

    def test_lazy_forces_and_site_energy(self, nacl):
        ewald = EwaldSummation(nacl, compute_forces=True)
        forces = ewald.forces
        assert ewald._initialized
        assert forces.shape == (len(nacl), 3)
        # Every ion sits on an inversion center, so the net forces vanish
        assert np.allclose(forces, 0, atol=1e-6)

        ewald2 = EwaldSummation(nacl)
        site_energy = ewald2.get_site_energy(0)
        assert ewald2._initialized
        assert sum(ewald2.get_site_energy(idx) for idx in range(len(nacl))) == approx(ewald2.total_energy)
        assert site_energy == approx(ewald2.total_energy / len(nacl))

    def test_forces_not_computed(self, nacl):
        ewald = EwaldSummation(nacl)
        with pytest.raises(AttributeError, match="Forces are available only if compute_forces is True"):
            _ = ewald.forces
        # The energy terms are still computed lazily before raising
        assert ewald._initialized

    def test_nacl_madelung(self, nacl):
        # Rocksalt Madelung constant 1.747565 referenced to the nearest-neighbor distance a/2
        expected = -1.747565 * EwaldSummation.CONV_FACT / (5.69 / 2) * len(nacl) / 2
        assert EwaldSummation(nacl).total_energy == approx(expected, rel=1e-5)

    def test_eta(self, nacl):
        assert EwaldSummation(nacl, eta=0.5).eta == 0.5
        expected = (len(nacl) / 2**0.5 / nacl.volume**2) ** (1 / 3) * np.pi
        assert EwaldSummation(nacl).eta == approx(expected)

    @pytest.mark.parametrize("compute_forces", [True, False])
    def test_str(self, nacl, compute_forces):
        ewald = EwaldSummation(nacl, compute_forces=compute_forces)
        lines = str(ewald).split("\n")
        assert lines[0] == f"Real = {ewald.real_space_energy}"
        assert lines[1] == f"Reciprocal = {ewald.reciprocal_space_energy}"
        assert lines[2] == f"Point = {ewald.point_energy}"
        assert lines[3] == f"Total = {ewald.total_energy}"
        assert lines[4] == ("Forces:" if compute_forces else "Forces were not computed")

    def test_get_site_energy_charged_warns(self):
        struct = Structure(Lattice.cubic(4), [Species("Na", 1)], [[0, 0, 0]])
        ewald = EwaldSummation(struct)
        with pytest.warns(UserWarning, match="Per atom energies for charged structures not supported"):
            site_energy = ewald.get_site_energy(0)
        assert site_energy == approx(ewald.total_energy - ewald._charged_cell_energy)

    def test_no_neighbors_within_real_space_cut(self):
        # A single ion with a real-space cutoff shorter than the lattice parameter has no
        # real-space partners, so the real-space term is identically zero.
        struct = Structure(Lattice.cubic(10), [Species("Na", 1)], [[0, 0, 0]])
        ewald = EwaldSummation(struct, real_space_cut=1.0, compute_forces=True)
        assert ewald.real_space_energy == 0
        assert np.allclose(ewald.forces, 0)

    def test_compute_partial_energy(self, nacl):
        ewald = EwaldSummation(nacl)
        matrix = ewald.total_energy_matrix
        assert ewald.compute_partial_energy([]) == approx(matrix.sum())
        removed = [0, 4]
        keep = [idx for idx in range(len(nacl)) if idx not in removed]
        assert ewald.compute_partial_energy(removed) == approx(matrix[np.ix_(keep, keep)].sum())
        # The cached matrix must not be modified
        assert np.allclose(ewald.total_energy_matrix, matrix)

    def test_compute_sub_structure(self, nacl):
        ewald = EwaldSummation(nacl)
        assert ewald.compute_sub_structure(nacl) == approx(ewald.total_energy_matrix.sum())

        # Removing sites is equivalent to zeroing their rows/columns
        sub = nacl.copy()
        sub.remove_sites([0, 4])
        assert ewald.compute_sub_structure(sub) == approx(ewald.compute_partial_energy([0, 4]))

        # Halving the charge of one site scales its row and column by 0.5
        scaled = nacl.copy()
        scaled.replace(0, {Species("Na", 1): 0.5})
        scale = np.ones(len(nacl))
        scale[0] = 0.5
        expected = (ewald.total_energy_matrix * np.outer(scale, scale)).sum()
        assert ewald.compute_sub_structure(scaled) == approx(expected)

    def test_compute_sub_structure_missing_sites(self, nacl):
        ewald = EwaldSummation(nacl)
        sub = nacl.copy()
        sub.translate_sites([0], [0.1, 0, 0])
        with pytest.raises(ValueError, match=r"Missing sites\.\nunmatched = "):
            ewald.compute_sub_structure(sub)


def brute_force_removal(matrix, n_remove, candidates):
    """Lowest matrix sum obtained by zeroing out n_remove of the candidate indices."""
    best = np.inf
    for removed in combinations(candidates, n_remove):
        keep = [idx for idx in range(len(matrix)) if idx not in removed]
        best = min(best, matrix[np.ix_(keep, keep)].sum())
    return best


class TestEwaldMinimizerExtra:
    @pytest.fixture
    def matrix(self):
        rng = np.random.default_rng(0)
        mat = rng.uniform(-5, 15, size=(8, 8))
        return (mat + mat.T) / 2

    @pytest.mark.parametrize("algo", [EwaldMinimizer.ALGO_FAST, EwaldMinimizer.ALGO_BEST_FIRST])
    def test_removal_matches_brute_force(self, matrix, algo):
        candidates = [0, 1, 2, 3, 4, 5]
        e_min = EwaldMinimizer(matrix, [[0, 3, list(candidates), "X"]], num_to_return=1, algo=algo)
        assert len(e_min.output_lists) == 1
        assert len(e_min.best_m_list) == 3
        assert {idx for idx, _ in e_min.best_m_list} <= set(candidates)
        assert {sp for _, sp in e_min.best_m_list} == {"X"}
        assert e_min.minimized_sum == approx(brute_force_removal(matrix, 3, candidates))

    def test_best_first_stops_early(self, matrix):
        candidates = list(range(8))
        fast = EwaldMinimizer(matrix, [[0, 4, list(candidates), "X"]], num_to_return=3)
        best_first = EwaldMinimizer(
            matrix, [[0, 4, list(candidates), "X"]], num_to_return=3, algo=EwaldMinimizer.ALGO_BEST_FIRST
        )
        assert best_first._finished
        assert len(fast.output_lists) == len(best_first.output_lists) == 3
        assert fast.minimized_sum == approx(brute_force_removal(matrix, 4, candidates))
        # Best-first returns the first num_to_return solutions found, which can only be worse or equal
        assert best_first.minimized_sum >= fast.minimized_sum - 1e-10
        # The fast algorithm keeps a sorted list of the lowest sums
        sums = [out[0] for out in fast.output_lists]
        assert sums == sorted(sums)

    def test_invalid_fraction(self, matrix):
        with pytest.raises(ValueError, match="multiplication fractions must be <= 1"):
            EwaldMinimizer(matrix, [[1.5, 1, [0, 1], "X"]])

    def test_complete_algo_not_implemented(self, matrix):
        with pytest.raises(NotImplementedError, match="Complete algo not yet implemented"):
            EwaldMinimizer(matrix, [[0, 1, [0, 1], "X"]], algo=EwaldMinimizer.ALGO_COMPLETE)

    @pytest.mark.xfail(
        strict=True,
        raises=IndexError,
        reason="minimize_matrix() skips the search for ALGO_TIME_LIMIT, leaving output_lists empty",
    )
    def test_time_limit_algo(self, matrix):
        e_min = EwaldMinimizer(matrix, [[0, 2, [0, 1, 2, 3], "X"]], algo=EwaldMinimizer.ALGO_TIME_LIMIT)
        assert e_min.minimized_sum == approx(brute_force_removal(matrix, 2, [0, 1, 2, 3]))

    def test_best_case_time_limit(self, matrix):
        m_list = [[0, 2, [0, 1, 2, 3], "X"]]
        e_min = EwaldMinimizer(matrix, [[0, 2, [0, 1, 2, 3], "X"]])
        rigorous = e_min.best_case(e_min._matrix, m_list, set(range(8)))
        # The rigorous bound must not exceed the true minimum
        assert rigorous <= e_min.minimized_sum + 1e-10
        # At t ~ 0 the time-limited bound reduces to the rigorous bound
        e_min._algo = EwaldMinimizer.ALGO_TIME_LIMIT
        e_min._start_time = datetime.now(tz=UTC)
        assert e_min.best_case(e_min._matrix, m_list, set(range(8))) == approx(rigorous, abs=1e-3)
        # After the 30 min budget the bound is purely the averaged estimate
        e_min._start_time = datetime.now(tz=UTC) - timedelta(minutes=30)
        assert e_min.best_case(e_min._matrix, m_list, set(range(8))) != approx(rigorous)

    def test_add_m_list_initializes_none(self, matrix):
        e_min = EwaldMinimizer(matrix, [[0, 1, [0, 1], "X"]])
        e_min._output_lists = None
        e_min.add_m_list(1.0, [[0, "X"]])
        assert e_min.output_lists == [[1.0, [[0, "X"]]]]
        assert e_min._current_minimum == 1.0

    def test_get_next_index(self, matrix):
        sums = matrix.sum(axis=1)
        sub = [0, 1, 2, 3]
        # Removal (f < 1) picks the index with the largest row sum
        assert EwaldMinimizer.get_next_index(matrix, [0, 1, sub, "X"], set(sub)) == sub[np.argmax(sums[sub])]
        # f == 1 picks the index with the smallest row sum
        assert EwaldMinimizer.get_next_index(matrix, [1, 1, sub, "X"], set(sub)) == sub[np.argmin(sums[sub])]
