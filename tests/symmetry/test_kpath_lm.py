from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.kpath import KPathLatimerMunro
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest


def _path_labels(kpath) -> set[str]:
    return {label for segment in kpath.kpath["path"] for label in segment}


class TestKPathLatimerMunroMagnetic:
    @pytest.fixture
    def afm_cscl(self):
        """G-type AFM on a simple-cubic cell: body-center translation is combined with time reversal."""
        struct = Structure(Lattice.cubic(3), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        struct.add_site_property("magmom", [[0, 0, 1], [0, 0, -1]])
        return struct

    def test_type_3_4_afm(self, afm_cscl):
        kpath = KPathLatimerMunro(afm_cscl, has_magmoms=True)
        assert kpath.mag_type == "3/4"
        assert set(kpath.kpath["kpoints"]) == {"a", "c", "d", "e", "g", "Γ"}
        assert _path_labels(kpath) == set(kpath.kpath["kpoints"])
        assert kpath.kpath["kpoints"]["Γ"] == approx([0, 0, 0])
        # vector magmoms given as lists are converted to arrays
        assert all(isinstance(m, np.ndarray) for m in kpath.structure.site_properties["magmom"])

    def test_scalar_magmoms_default_axis(self):
        struct = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        struct.add_site_property("magmom", [2.0])
        with pytest.warns(UserWarning, match="Defaulted to z\\+ spinor"):
            kpath = KPathLatimerMunro(struct, has_magmoms=True)
        assert kpath.mag_type == "3/4"
        assert kpath.structure.site_properties["magmom"][0] == approx([0, 0, 2])

    def test_scalar_magmoms_with_axis(self):
        struct = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        struct.add_site_property("magmom", [2.0])
        z_path = KPathLatimerMunro(struct.copy(), has_magmoms=True, magmom_axis=[0, 0, 1])
        x_path = KPathLatimerMunro(struct.copy(), has_magmoms=True, magmom_axis=[1, 0, 0])
        assert x_path.structure.site_properties["magmom"][0] == approx([2, 0, 0])
        # the spin axis lowers the symmetry differently, giving distinct k-paths
        assert set(z_path.kpath["kpoints"]) != set(x_path.kpath["kpoints"])

    def test_type_2_all_zero_magmoms(self):
        struct = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        struct.add_site_property("magmom", [np.zeros(3)])
        kpath = KPathLatimerMunro(struct, has_magmoms=True)
        assert kpath.mag_type == "2"
        nonmag = KPathLatimerMunro(Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]]))
        assert nonmag.mag_type == "0"
        # grey group: same k-point labels as the non-magnetic path
        assert set(kpath.kpath["kpoints"]) == set(nonmag.kpath["kpoints"])

    def test_type_1_triclinic(self):
        struct = Structure(Lattice.from_parameters(3, 4, 5, 80, 85, 95), ["Fe"], [[0, 0, 0]])
        struct.add_site_property("magmom", [np.array([0.3, 0.2, 1.0])])
        kpath = KPathLatimerMunro(struct, has_magmoms=True)
        assert kpath.mag_type == "1"
        # P-1 with inversion (no time reversal): every segment starts or ends at Gamma
        assert all("Γ" in segment for segment in kpath.kpath["path"])

    def test_has_magmoms_without_magmom_property(self):
        struct = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        with pytest.warns(UserWarning, match="All magnetic moments are being set to zero"):
            kpath = KPathLatimerMunro(struct, has_magmoms=True)
        assert kpath.mag_type == "2"

    def test_convert_magmoms_missing_property(self):
        kpath = KPathLatimerMunro(Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]]))
        with pytest.warns(UserWarning, match="All magnetic moments are being set to zero"):
            struct = kpath._convert_all_magmoms_to_vectors([0, 0, 1], axis_specified=False)
        assert struct.site_properties["magmom"][0] == approx([0, 0, 0])

    def test_magmom_ignored_warning(self):
        struct = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        struct.add_site_property("magmom", [2.0])
        with pytest.warns(UserWarning, match="has_magmoms is False, but site_properties contains the key magmom"):
            kpath = KPathLatimerMunro(struct)
        assert kpath.mag_type == "0"
        assert "magmom" not in kpath.structure.site_properties

    def test_magmom_ignored_does_not_mutate_input(self):
        struct = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        struct.add_site_property("magmom", [2.0])
        with pytest.warns(UserWarning, match="has_magmoms is False"):
            KPathLatimerMunro(struct)
        assert struct.site_properties["magmom"] == [2.0]


class TestKPathLatimerMunroHelpers:
    @pytest.fixture
    def kpath(self):
        return KPathLatimerMunro(Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]]))

    def test_reducible_cell_warning(self):
        struct = Structure(Lattice([[3, 0, 0], [4, 3, 0], [0, 0, 3]]), ["Fe"], [[0, 0, 0]])
        with pytest.warns(UserWarning, match="not fully reduced"):
            kpath = KPathLatimerMunro(struct)
        # primed/duplicate labels get numbered subscripts
        assert any("_{" in label for label in kpath.kpath["kpoints"])

    @pytest.mark.parametrize(
        ("pos1", "pos2", "expected"),
        [
            ([0.0, 0.5, 0.25], [0.0, 0.5, 0.25], True),
            ([0.0, 0.5, 0.999999], [0.0, 0.5, 0.0], True),
            ([1.0, -0.5, 2.25], [0.0, 0.5, 0.25], True),
            ([0.0, 0.5, 0.3], [0.0, 0.5, 0.0], False),
            ([0.0, 0.5], [0.0, 0.5, 0.0], False),
        ],
    )
    def test_closewrapped(self, pos1, pos2, expected):
        assert KPathLatimerMunro._closewrapped(np.array(pos1), np.array(pos2), np.ones(3) * 1e-5) is expected

    def test_closewrapped_does_not_mutate(self):
        op = SymmOp.from_rotation_and_translation(np.eye(3), [1.25, 0, 0])
        KPathLatimerMunro._closewrapped(op.translation_vector, np.zeros(3), np.ones(3) * 1e-5)
        assert op.translation_vector == approx([1.25, 0, 0])

    def test_apply_op_to_magmom(self):
        magmom = np.array([1.0, 2.0, 3.0])
        c2z = np.diag([-1, -1, 1])
        assert KPathLatimerMunro._apply_op_to_magmom(c2z, magmom) == approx([-1, -2, 3])
        # magnetic moments are axial vectors: inversion leaves them unchanged
        assert KPathLatimerMunro._apply_op_to_magmom(-np.eye(3), magmom) == approx(magmom)
        mirror_z = np.diag([1, 1, -1])
        assert KPathLatimerMunro._apply_op_to_magmom(mirror_z, magmom) == approx([-1, -2, 3])

    def test_get_coset_factor(self, kpath):
        identity = SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 0])
        inversion = SymmOp.from_rotation_and_translation(-np.eye(3), [0, 0, 0])
        assert kpath._get_coset_factor([identity, inversion], [identity]) == inversion
        assert kpath._get_coset_factor([identity], [identity]) == "No coset factor found."
        # candidate whose product with H lands back in H is rejected
        c2z = SymmOp.from_rotation_and_translation(np.diag([-1, -1, 1]), [0, 0, 0])
        assert kpath._get_coset_factor([identity, c2z], [c2z]) == "No coset factor found."

    def test_magnetic_symmetry_operations(self, kpath):
        struct = Structure(Lattice.cubic(3), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        struct.add_site_property("magmom", [np.array([0, 0, 1]), np.array([0, 0, -1])])
        identity = SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 0])
        body_center = SymmOp.from_rotation_and_translation(np.eye(3), [0.5, 0.5, 0.5])
        ops = kpath._get_magnetic_symmetry_operations(struct, [identity, body_center], atol=1e-5)
        assert [op.time_reversal for op in ops] == [1, -1]


class TestKPathLatimerMunro(MatSciTest):
    def test_kpath_generation(self):
        triclinic = [1, 2]
        monoclinic = range(3, 16)
        orthorhombic = range(16, 75)
        tetragonal = range(75, 143)
        rhombohedral = range(143, 168)
        hexagonal = range(168, 195)
        cubic = range(195, 231)

        species = ["K", "La", "Ti"]
        coords = [[0.345, 5, 0.77298], [0.1345, 5.1, 0.77298], [0.7, 0.8, 0.9]]
        for sg_num in range(1, 231):
            if sg_num in triclinic:
                lattice = Lattice(
                    [
                        [3.0233057319441246, 1, 0],
                        [0, 7.9850357844548681, 1],
                        [0, 1.2, 8.1136762279561818],
                    ]
                )
            elif sg_num in monoclinic:
                lattice = Lattice.monoclinic(2, 9, 1, 99)
            elif sg_num in orthorhombic:
                lattice = Lattice.orthorhombic(2, 9, 1)
            elif sg_num in tetragonal:
                lattice = Lattice.tetragonal(2, 9)
            elif sg_num in rhombohedral:
                lattice = Lattice.hexagonal(2, 95)
            elif sg_num in hexagonal:
                lattice = Lattice.hexagonal(2, 9)
            elif sg_num in cubic:
                lattice = Lattice.cubic(2)

            struct = Structure.from_spacegroup(sg_num, lattice, species, coords)
            _ = KPathLatimerMunro(struct)  # Throws error if something doesn't work, causing test to fail.

        struct_file_path = f"{TEST_FILES_DIR}/cif/AgO_kpath_test.cif"
        struct = Structure.from_file(struct_file_path)
        _ = KPathLatimerMunro(struct)  # Throws error if something doesn't work, causing test to fail.

    def test_kpath_acentered(self):
        species = ["K", "La", "Ti"]
        coords = [[0.345, 5, 0.77298], [0.1345, 5.1, 0.77298], [0.7, 0.8, 0.9]]
        lattice = Lattice.orthorhombic(2, 9, 1)
        struct = Structure.from_spacegroup(38, lattice, species, coords)
        sga = SpacegroupAnalyzer(struct)
        struct_prim = sga.get_primitive_standard_structure(international_monoclinic=False)
        kpath = KPathLatimerMunro(struct_prim)

        kpoints = kpath._kpath["kpoints"]
        labels = list(kpoints)

        assert sorted(labels) == sorted(["a", "b", "c", "d", "d_{1}", "e", "f", "q", "q_{1}", "Γ"])

        assert kpoints["a"] == approx([0.0, 0.5, 0.0])

        assert kpoints["f"] == approx([-0.5, 0.5, 0.5])

        assert kpoints["c"] == approx([0.0, 0.0, 0.5])

        assert kpoints["b"] == approx([-0.5, 0.5, 0.0])

        assert kpoints["Γ"] == approx([0, 0, 0])

        assert kpoints["e"] == approx([0.0, 0.5, 0.5])

        assert kpoints["d_{1}"] == approx([0.2530864197530836, 0.25308641975308915, 0.0]) or kpoints["d"] == approx(
            [0.2530864197530836, 0.25308641975308915, 0.0]
        )

        assert kpoints["q_{1}"] == approx([0.2530864197530836, 0.25308641975308915, 0.5]) or kpoints["q"] == approx(
            [0.2530864197530836, 0.25308641975308915, 0.5]
        )

    # TODO: Fix this magnetism test.
    # def test_magnetic_kpath_generation(self):
    #     from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
    #     struct_file_path = f"{TEST_FILES_DIR}/io/cif/mcif/LaMnO3_magnetic.mcif"
    #     struct = Structure.from_file(struct_file_path)
    #     mga = CollinearMagneticStructureAnalyzer(struct)
    #     col_spin_orig = mga.get_structure_with_spin()
    #     col_spin_orig.add_spin_by_site([0.0] * 20)
    #     col_spin_sym = SpacegroupAnalyzer(col_spin_orig)
    #     col_spin_prim = col_spin_sym.get_primitive_standard_structure(international_monoclinic=False)

    #     magmom_vec_list = [np.zeros(3) for site in col_spin_prim]
    #     magmom_vec_list[4:8] = [
    #         np.array([3.87, 3.87, 0.0]),
    #         np.array([3.87, 3.87, 0.0]),
    #         np.array([-3.87, -3.87, 0.0]),
    #         np.array([-3.87, -3.87, 0.0]),
    #     ]
    #     col_spin_prim.add_site_property("magmom", magmom_vec_list)

    #     kpath = KPathLatimerMunro(col_spin_prim, has_magmoms=True)

    #     kpoints = kpath._kpath["kpoints"]
    #     assert sorted(kpoints) == sorted(["a", "b", "c", "d", "d_{1}", "e", "f", "g", "g_{1}", "Γ"])

    #     assert kpoints["e"] == approx([-0.5, 0.0, 0.5])

    #     assert kpoints["g"] == approx([-0.5, -0.5, 0.5])

    #     assert kpoints["a"] == approx([-0.5, 0.0, 0.0])

    #     assert kpoints["g_{1}"] == approx([0.5, -0.5, 0.5])

    #     assert kpoints["f"] == approx([0.0, -0.5, 0.5])

    #     assert kpoints["c"] == approx([0.0, 0.0, 0.5])

    #     assert kpoints["b"] == approx([0.0, -0.5, 0.0])

    #     assert kpoints["Γ"] == approx([0, 0, 0])

    #     assert kpoints["d_{1}"] == approx([-0.5, -0.5, 0.0]) or kpoints["d"] == approx([-0.5, -0.5, 0.0])
