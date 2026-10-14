from __future__ import annotations

import warnings

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
