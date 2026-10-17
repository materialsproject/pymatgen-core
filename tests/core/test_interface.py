from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest import approx

from pymatgen.core.interface import GrainBoundary, GrainBoundaryGenerator, Interface, count_layers, label_termination
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/core/grain_boundary"


class TestGrainBoundary(MatSciTest):
    def setup_method(self):
        self.Cu_conv = Structure.from_file(f"{TEST_DIR}/Cu_mp-30_conventional_standard.cif")
        GB_Cu_conv = GrainBoundaryGenerator(self.Cu_conv)
        self.Cu_GB1 = GB_Cu_conv.gb_from_parameters(
            [1, 2, 3],
            123.74898859588858,
            expand_times=4,
            vacuum_thickness=1.5,
            ab_shift=[0.0, 0.0],
            plane=[1, 3, 1],
            rm_ratio=0.0,
        )
        self.Cu_GB2 = GB_Cu_conv.gb_from_parameters(
            [1, 2, 3],
            123.74898859588858,
            expand_times=4,
            vacuum_thickness=1.5,
            ab_shift=[0.2, 0.2],
            rm_ratio=0.0,
        )

    def test_init(self):
        assert self.Cu_GB1.rotation_angle == approx(123.74898859588858)
        assert self.Cu_GB1.vacuum_thickness == approx(1.5)
        assert self.Cu_GB2.rotation_axis == (1, 2, 3)
        assert_allclose(self.Cu_GB1.ab_shift, [0.0, 0.0])
        assert_allclose(self.Cu_GB2.ab_shift, [0.2, 0.2])
        assert self.Cu_GB1.gb_plane == (1, 3, 1)
        assert self.Cu_GB2.gb_plane == (1, 2, 3)
        assert_allclose(self.Cu_GB1.init_cell.lattice.matrix, self.Cu_conv.lattice.matrix)

    def test_copy(self):
        Cu_GB1_copy = self.Cu_GB1.copy()
        assert Cu_GB1_copy is not self.Cu_GB1
        assert Cu_GB1_copy == self.Cu_GB1
        assert Cu_GB1_copy.sigma == approx(self.Cu_GB1.sigma)
        assert Cu_GB1_copy.rotation_angle == approx(self.Cu_GB1.rotation_angle)
        assert Cu_GB1_copy.rotation_axis == self.Cu_GB1.rotation_axis
        assert Cu_GB1_copy.gb_plane == self.Cu_GB1.gb_plane
        assert_allclose(Cu_GB1_copy.init_cell.lattice.matrix, self.Cu_GB1.init_cell.lattice.matrix)
        assert_allclose(
            Cu_GB1_copy.oriented_unit_cell.lattice.matrix,
            self.Cu_GB1.oriented_unit_cell.lattice.matrix,
        )
        assert_allclose(Cu_GB1_copy.lattice.matrix, self.Cu_GB1.lattice.matrix)

    def test_sigma(self):
        assert self.Cu_GB1.sigma == approx(9)
        assert self.Cu_GB2.sigma == approx(9)

    def test_top_grain(self):
        assert len(self.Cu_GB1) == len(self.Cu_GB1.top_grain) * 2
        assert_allclose(self.Cu_GB1.lattice.matrix, self.Cu_GB1.top_grain.lattice.matrix)

    def test_bottom_grain(self):
        assert len(self.Cu_GB1) == len(self.Cu_GB1.bottom_grain) * 2
        assert_allclose(self.Cu_GB1.lattice.matrix, self.Cu_GB1.bottom_grain.lattice.matrix)

    def test_coincidents(self):
        assert len(self.Cu_GB1) / self.Cu_GB1.sigma == len(self.Cu_GB1.coincidents)
        assert len(self.Cu_GB2) / self.Cu_GB2.sigma == len(self.Cu_GB2.coincidents)

    def test_as_dict_and_from_dict(self):
        d1 = self.Cu_GB1.as_dict()
        d2 = self.Cu_GB2.as_dict()
        Cu_GB1_new = GrainBoundary.from_dict(d1)
        Cu_GB2_new = GrainBoundary.from_dict(d2)
        assert Cu_GB1_new.sigma == approx(self.Cu_GB1.sigma)
        assert Cu_GB1_new.rotation_angle == approx(self.Cu_GB1.rotation_angle)
        assert Cu_GB1_new.rotation_axis == self.Cu_GB1.rotation_axis
        assert Cu_GB1_new.gb_plane == self.Cu_GB1.gb_plane
        assert_allclose(Cu_GB1_new.init_cell.lattice.matrix, self.Cu_GB1.init_cell.lattice.matrix)
        assert_allclose(
            Cu_GB1_new.oriented_unit_cell.lattice.matrix,
            self.Cu_GB1.oriented_unit_cell.lattice.matrix,
            atol=1e-9,
        )
        assert_allclose(Cu_GB1_new.lattice.matrix, self.Cu_GB1.lattice.matrix)
        assert Cu_GB2_new.sigma == approx(self.Cu_GB2.sigma)
        assert Cu_GB2_new.rotation_angle == approx(self.Cu_GB2.rotation_angle)
        assert Cu_GB2_new.rotation_axis == self.Cu_GB2.rotation_axis
        assert Cu_GB2_new.gb_plane == self.Cu_GB2.gb_plane
        assert_allclose(Cu_GB2_new.init_cell.lattice.matrix, self.Cu_GB2.init_cell.lattice.matrix)
        assert_allclose(
            Cu_GB2_new.oriented_unit_cell.lattice.matrix,
            self.Cu_GB2.oriented_unit_cell.lattice.matrix,
        )
        assert_allclose(Cu_GB2_new.lattice.matrix, self.Cu_GB2.lattice.matrix)

    def test_str(self):
        lines = str(self.Cu_GB1).splitlines()
        assert lines[0] == f"Gb Summary ({self.Cu_GB1.composition.formula})"
        assert lines[1] == "Reduced Formula: Cu"
        assert lines[2] == "Rotation axis: (1, 2, 3)"
        assert lines[4] == "GB plane: (1, 3, 1)"
        assert lines[6] == "vacuum thickness: 1.5"
        assert lines[10] == f"Sites ({len(self.Cu_GB1)})"
        # One line per site after the header
        assert len(lines) == 11 + len(self.Cu_GB1)
        assert lines[11].startswith("1 Cu ")

    def test_get_sorted_structure(self):
        sorted_gb = self.Cu_GB1.get_sorted_structure(key=lambda site: site.frac_coords[2], reverse=True)
        assert isinstance(sorted_gb, GrainBoundary)
        assert len(sorted_gb) == len(self.Cu_GB1)
        assert sorted_gb.sigma == self.Cu_GB1.sigma
        assert sorted_gb.gb_plane == self.Cu_GB1.gb_plane
        c_coords = sorted_gb.frac_coords[:, 2]
        assert np.all(np.diff(c_coords) <= 1e-8)

    def test_sigma_from_site_prop(self):
        assert self.Cu_GB1.sigma_from_site_prop == 9
        assert self.Cu_GB2.sigma_from_site_prop == 9

        merged = self.Cu_GB1.copy()
        merged.add_site_property("grain_label", [None, *merged.site_properties["grain_label"][1:]])
        with pytest.raises(ValueError, match="Sites were merged, this property does not work"):
            _ = merged.sigma_from_site_prop


class TestGrainBoundaryGenerator(MatSciTest):
    @classmethod
    def setup_class(cls):
        cls.Cu_prim = Structure.from_file(f"{TEST_DIR}/Cu_mp-30_primitive.cif")
        cls.GB_Cu_prim = GrainBoundaryGenerator(cls.Cu_prim)
        cls.Cu_conv = Structure.from_file(f"{TEST_DIR}/Cu_mp-30_conventional_standard.cif")
        cls.GB_Cu_conv = GrainBoundaryGenerator(cls.Cu_conv)
        cls.Be = Structure.from_file(f"{TEST_DIR}/Be_mp-87_conventional_standard.cif")
        cls.GB_Be = GrainBoundaryGenerator(cls.Be)
        cls.Pa = Structure.from_file(f"{TEST_DIR}/Pa_mp-62_conventional_standard.cif")
        cls.GB_Pa = GrainBoundaryGenerator(cls.Pa)
        cls.Br = Structure.from_file(f"{TEST_DIR}/Br_mp-23154_conventional_standard.cif")
        cls.GB_Br = GrainBoundaryGenerator(cls.Br)
        cls.Bi = Structure.from_file(f"{TEST_DIR}/Bi_mp-23152_primitive.cif")
        cls.GB_Bi = GrainBoundaryGenerator(cls.Bi)

    def test_gb_from_parameters(self):
        # from fcc primitive cell,axis[1,2,3],sigma 9.
        gb_cu_123_prim1 = self.GB_Cu_prim.gb_from_parameters([1, 2, 3], 123.74898859588858, expand_times=2)
        lat_mat1 = gb_cu_123_prim1.lattice.matrix
        c_vec1 = np.cross(lat_mat1[0], lat_mat1[1]) / np.linalg.norm(np.cross(lat_mat1[0], lat_mat1[1]))
        c_len1 = np.dot(lat_mat1[2], c_vec1)
        vol_ratio = gb_cu_123_prim1.volume / self.Cu_prim.volume
        assert vol_ratio == approx(9 * 2 * 2, abs=1e-8)
        # test expand_times and vacuum layer
        gb_cu_123_prim2 = self.GB_Cu_prim.gb_from_parameters(
            [1, 2, 3], 123.74898859588858, expand_times=4, vacuum_thickness=1.5
        )
        lat_mat2 = gb_cu_123_prim2.lattice.matrix
        c_vec2 = np.cross(lat_mat2[0], lat_mat2[1]) / np.linalg.norm(np.cross(lat_mat2[0], lat_mat2[1]))
        c_len2 = np.dot(lat_mat2[2], c_vec2)
        assert (c_len2 - 1.5 * 2) / c_len1 == approx(2)

        # test normal
        gb_cu_123_prim3 = self.GB_Cu_prim.gb_from_parameters([1, 2, 3], 123.74898859588858, expand_times=2, normal=True)
        lat_mat3 = gb_cu_123_prim3.lattice.matrix
        c_vec3 = np.cross(lat_mat3[0], lat_mat3[1]) / np.linalg.norm(np.cross(lat_mat3[0], lat_mat3[1]))
        ab_len3 = np.linalg.norm(np.cross(lat_mat3[2], c_vec3))
        assert ab_len3 == approx(0)

        # test normal in tilt boundary
        # The 'finfo(np.float32).eps' is the smallest representable positive number in float32,
        # which has been introduced because comparing to just zero or one failed the test by rounding errors.
        gb_cu_010_conv1 = self.GB_Cu_conv.gb_from_parameters(
            rotation_axis=[0, 1, 0],
            rotation_angle=36.8698976458,
            expand_times=1,
            vacuum_thickness=1.0,
            ab_shift=[0.0, 0.0],
            rm_ratio=0.0,
            plane=[0, 0, 1],
            normal=True,
        )
        assert np.all(-np.finfo(np.float32).eps <= gb_cu_010_conv1.frac_coords)
        assert np.all(1 + np.finfo(np.float32).eps >= gb_cu_010_conv1.frac_coords)

        # from fcc conventional cell,axis [1,2,3], siamg 9
        gb_cu_123_conv1 = self.GB_Cu_conv.gb_from_parameters(
            [1, 2, 3], 123.74898859588858, expand_times=4, vacuum_thickness=1.5
        )
        lat_mat1 = gb_cu_123_conv1.lattice.matrix
        assert np.dot(lat_mat1[0], [1, 2, 3]) == approx(0)
        assert np.dot(lat_mat1[1], [1, 2, 3]) == approx(0)
        # test plane
        gb_cu_123_conv2 = self.GB_Cu_conv.gb_from_parameters(
            [1, 2, 3],
            123.74898859588858,
            expand_times=2,
            vacuum_thickness=1.5,
            normal=False,
            plane=[1, 3, 1],
        )
        lat_mat2 = gb_cu_123_conv2.lattice.matrix
        assert np.dot(lat_mat2[0], [1, 3, 1]) == approx(0)
        assert np.dot(lat_mat2[1], [1, 3, 1]) == approx(0)

        # from hex cell,axis [1,1,1], sigma 21
        gb_Be_111_1 = self.GB_Be.gb_from_parameters(
            [1, 1, 1],
            147.36310249644626,
            ratio=[5, 2],
            expand_times=4,
            vacuum_thickness=1.5,
            plane=[1, 2, 1],
        )
        lat_priv = self.Be.lattice.matrix
        lat_mat1 = np.matmul(gb_Be_111_1.lattice.matrix, np.linalg.inv(lat_priv))
        assert np.dot(lat_mat1[0], [1, 2, 1]) == approx(0)
        assert np.dot(lat_mat1[1], [1, 2, 1]) == approx(0)
        # test volume associated with sigma value
        gb_Be_111_2 = self.GB_Be.gb_from_parameters([1, 1, 1], 147.36310249644626, ratio=[5, 2], expand_times=4)
        vol_ratio = gb_Be_111_2.volume / self.Be.volume
        assert vol_ratio == approx(19 * 2 * 4)
        # test ratio = None, axis [0,0,1], sigma 7
        gb_Be_111_3 = self.GB_Be.gb_from_parameters([0, 0, 1], 21.786789298261812, ratio=[5, 2], expand_times=4)
        gb_Be_111_4 = self.GB_Be.gb_from_parameters([0, 0, 1], 21.786789298261812, ratio=None, expand_times=4)
        assert gb_Be_111_3.lattice.abc == gb_Be_111_4.lattice.abc
        assert gb_Be_111_3.lattice.angles == gb_Be_111_4.lattice.angles
        gb_Be_111_5 = self.GB_Be.gb_from_parameters([3, 1, 0], 180.0, ratio=[5, 2], expand_times=4)
        gb_Be_111_6 = self.GB_Be.gb_from_parameters([3, 1, 0], 180.0, ratio=None, expand_times=4)
        assert gb_Be_111_5.lattice.abc == gb_Be_111_6.lattice.abc
        assert gb_Be_111_5.lattice.angles == gb_Be_111_6.lattice.angles

        # gb from tetragonal cell, axis[1,1,1], sigma 15
        gb_Pa_111_1 = self.GB_Pa.gb_from_parameters(
            [1, 1, 1], 151.92751306414706, ratio=[2, 3], expand_times=4, max_search=10
        )
        vol_ratio = gb_Pa_111_1.volume / self.Pa.volume
        assert vol_ratio == approx(17 * 2 * 4)

        # gb from orthorhombic cell, axis[1,1,1], sigma 83
        gb_Br_111_1 = self.GB_Br.gb_from_parameters(
            [1, 1, 1],
            131.5023374652235,
            ratio=[21, 20, 5],
            expand_times=4,
            max_search=10,
        )
        vol_ratio = gb_Br_111_1.volume / self.Br.volume
        assert vol_ratio == approx(83 * 2 * 4)

        # gb from rhombohedra cell, axis[1,2,0], sigma 63
        gb_Bi_120_1 = self.GB_Bi.gb_from_parameters(
            [1, 2, 0], 63.310675060280246, ratio=[19, 5], expand_times=4, max_search=5
        )
        vol_ratio = gb_Bi_120_1.volume / self.Bi.volume
        assert vol_ratio == approx(59 * 2 * 4)

    def test_get_ratio(self):
        # hexagnal
        Be_ratio = self.GB_Be.get_ratio(max_denominator=2)
        assert Be_ratio == [5, 2]
        Be_ratio = self.GB_Be.get_ratio(max_denominator=5)
        assert Be_ratio == [12, 5]
        # tetragonal
        Pa_ratio = self.GB_Pa.get_ratio(max_denominator=5)
        assert Pa_ratio == [2, 3]
        # orthorhombic
        Br_ratio = self.GB_Br.get_ratio(max_denominator=5)
        assert Br_ratio == [21, 20, 5]
        # orthorhombic
        Bi_ratio = self.GB_Bi.get_ratio(max_denominator=5)
        assert Bi_ratio == [19, 5]

    def test_enum_sigma_cubic(self):
        true_100 = [5, 13, 17, 25, 29, 37, 41]
        true_110 = [3, 9, 11, 17, 19, 27, 33, 41, 43]
        true_111 = [3, 7, 13, 19, 21, 31, 37, 39, 43, 49]
        sigma_100 = list(GrainBoundaryGenerator.enum_sigma_cubic(50, [1, 0, 0]))
        sigma_110 = list(GrainBoundaryGenerator.enum_sigma_cubic(50, [1, 1, 0]))
        sigma_111 = list(GrainBoundaryGenerator.enum_sigma_cubic(50, [1, 1, 1]))
        sigma_222 = list(GrainBoundaryGenerator.enum_sigma_cubic(50, [2, 2, 2]))
        sigma_888 = list(GrainBoundaryGenerator.enum_sigma_cubic(50, [8, 8, 8]))

        assert sorted(true_100) == sorted(sigma_100)
        assert sorted(true_110) == sorted(sigma_110)
        assert sorted(true_111) == sorted(sigma_111)
        assert sorted(true_111) == sorted(sigma_222)
        assert sorted(true_111) == sorted(sigma_888)

    def test_enum_sigma_hex(self):
        true_100 = [17, 18, 22, 27, 38, 41]
        true_001 = [7, 13, 19, 31, 37, 43, 49]
        true_210 = [10, 11, 14, 25, 35, 49]
        sigma_100 = list(GrainBoundaryGenerator.enum_sigma_hex(50, [1, 0, 0], [8, 3]))
        sigma_001 = list(GrainBoundaryGenerator.enum_sigma_hex(50, [0, 0, 1], [8, 3]))
        sigma_210 = list(GrainBoundaryGenerator.enum_sigma_hex(50, [2, 1, 0], [8, 3]))
        sigma_420 = list(GrainBoundaryGenerator.enum_sigma_hex(50, [4, 2, 0], [8, 3]))
        sigma_840 = list(GrainBoundaryGenerator.enum_sigma_hex(50, [8, 4, 0], [8, 3]))

        assert sorted(true_100) == sorted(sigma_100)
        assert sorted(true_001) == sorted(sigma_001)
        assert sorted(true_210) == sorted(sigma_210)
        assert sorted(true_210) == sorted(sigma_420)
        assert sorted(true_210) == sorted(sigma_840)

    def test_enum_sigma_tet(self):
        true_100 = [5, 37, 41, 13, 3, 15, 39, 25, 17, 29]
        true_331 = [9, 3, 21, 39, 7, 31, 43, 13, 19, 37, 49]
        sigma_100 = list(GrainBoundaryGenerator.enum_sigma_tet(50, [1, 0, 0], [9, 1]))
        sigma_331 = list(GrainBoundaryGenerator.enum_sigma_tet(50, [3, 3, 1], [9, 1]))

        assert sorted(true_100) == sorted(sigma_100)
        assert sorted(true_331) == sorted(sigma_331)

    def test_enum_sigma_ort(self):
        true_100 = [41, 37, 39, 5, 15, 17, 13, 3, 25, 29]
        sigma_100 = list(GrainBoundaryGenerator.enum_sigma_ort(50, [1, 0, 0], [270, 30, 29]))

        assert sorted(true_100) == sorted(sigma_100)

    def test_enum_sigma_rho(self):
        true_100 = [7, 11, 43, 13, 41, 19, 47, 31]
        sigma_100 = list(GrainBoundaryGenerator.enum_sigma_rho(50, [1, 0, 0], [15, 4]))

        assert sorted(true_100) == sorted(sigma_100)

    def test_enum_possible_plane_cubic(self):
        all_plane = GrainBoundaryGenerator.enum_possible_plane_cubic(4, [1, 1, 1], 60)
        assert len(all_plane["Twist"]) == 1
        assert len(all_plane["Symmetric tilt"]) == 6
        assert len(all_plane["Normal tilt"]) == 12

    def test_get_trans_mat(self):
        mat1, mat2 = GrainBoundaryGenerator.get_trans_mat(
            [1, 1, 1],
            95.55344419565849,
            lat_type="o",
            ratio=[10, 20, 21],
            surface=[21, 20, 10],
            normal=True,
        )
        assert np.dot(mat1[0], [21, 20, 10]) == approx(0)
        assert np.dot(mat1[1], [21, 20, 10]) == approx(0)
        assert np.linalg.det(mat1) == approx(np.linalg.det(mat2))
        ab_len1 = np.linalg.norm(np.cross(mat1[2], [1, 1, 1]))
        assert ab_len1 == approx(0)

    def test_get_rotation_angle_from_sigma(self):
        true_angle = [12.680383491819821, 167.3196165081802]
        angle = GrainBoundaryGenerator.get_rotation_angle_from_sigma(41, [1, 0, 0], lat_type="o", ratio=[270, 30, 29])
        assert_allclose(true_angle, angle)
        close_angle = [36.86989764584403, 143.13010235415598]
        with pytest.warns(UserWarning, match=r"Sigma 6 is not possible .* Returning the rotation angles of sigma 5"):
            angle = GrainBoundaryGenerator.get_rotation_angle_from_sigma(
                6, [1, 0, 0], lat_type="o", ratio=[270, 30, 29]
            )
        assert_allclose(close_angle, angle)

    @pytest.mark.parametrize(
        ("angles", "lattice_type"),
        [((81, 86, 97), "triclinic"), ((90, 101, 90), "monoclinic")],
    )
    def test_init_rejects_unsupported_lattices(self, angles, lattice_type):
        # The lattice type used to be the first letter of get_lattice_type(), so triclinic was treated as tetragonal.
        structure = Structure(Lattice.from_parameters(3.1, 4.3, 5.2, *angles), ["Si"], [[0, 0, 0]])
        with pytest.raises(ValueError, match=f"but the structure is {lattice_type}"):
            GrainBoundaryGenerator(structure)

    def test_imprecise_rotation_angle_error(self):
        exact = GrainBoundaryGenerator.get_rotation_angle_from_sigma(3, [1, 1, 0])[1]
        assert self.GB_Cu_prim.gb_from_parameters([1, 1, 0], exact, plane=[1, 1, 2]).sigma == 3
        with pytest.raises(RuntimeError, match="rotation angle is probably not precise enough"):
            self.GB_Cu_prim.gb_from_parameters([1, 1, 0], round(exact, 2), plane=[1, 1, 2])

    def test_vec_to_surface_negative_components(self):
        assert GrainBoundaryGenerator.vec_to_surface([-100, 3, 2]) == (-100, 3, 2)

    def test_vec_to_surface_invalid_fractional_approximation(self):
        with pytest.raises(ValueError, match="Cannot convert vector"):
            GrainBoundaryGenerator.vec_to_surface([1, 3.141592653589793, 0])

    @pytest.mark.parametrize(
        "perm",
        [
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],  # alpha = 120
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],  # beta = 120
        ],
    )
    def test_init_reorients_hexagonal_c_axis(self, perm):
        hex_struct = Structure(Lattice.hexagonal(3, 5), ["Mg"], [[0, 0, 0]])
        hex_struct.make_supercell(perm)
        assert hex_struct.lattice.gamma == approx(90)

        generator = GrainBoundaryGenerator(hex_struct)
        assert generator.lat_type == "h"
        assert_allclose(generator.initial_structure.lattice.angles, (90, 90, 120))
        assert_allclose(generator.initial_structure.lattice.abc, (3, 3, 5))

    def test_gb_from_parameters_bcc_primitive(self):
        bcc_prim = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]]).get_primitive_structure()
        assert len(bcc_prim) == 1
        angle = GrainBoundaryGenerator.get_rotation_angle_from_sigma(3, [1, 1, 0])[0]
        gb = GrainBoundaryGenerator(bcc_prim).gb_from_parameters([1, 1, 0], angle, expand_times=1)
        assert gb.sigma == 3
        assert gb.volume / bcc_prim.volume == approx(3 * 2)

    @pytest.mark.parametrize(
        ("gb_name", "axis", "ratio", "match"),
        [
            ("GB_Pa", [1, 1, 1], [2, 3, 4], "Tetragonal system needs correct c2/a2 ratio"),
            ("GB_Br", [1, 1, 1], None, "CSL does not exist if all axial ratios are irrational"),
            ("GB_Br", [1, 1, 1], [2, 3], "Orthorhombic system needs correct c2:b2:a2 ratio"),
            ("GB_Be", [0, 0, 1], [2, 3, 4], "Hexagonal system needs correct c2/a2 ratio"),
            ("GB_Bi", [1, 1, 1], [2, 3, 4], r"Rhombohedral system needs correct \(1\+2\*cos\(alpha\)"),
            ("GB_Cu_conv", [1, 1, 0, 1], None, "Lattice type is not implemented"),
        ],
    )
    def test_gb_from_parameters_invalid_ratio(self, gb_name, axis, ratio, match):
        with pytest.raises(RuntimeError, match=match):
            getattr(self, gb_name).gb_from_parameters(axis, 60, ratio=ratio)

    def test_gb_from_parameters_invalid_axis_length(self):
        with pytest.raises(ValueError, match="Invalid length of rotation axis"):
            self.GB_Cu_conv.gb_from_parameters([1, 1, 1, 1, 1], 60)

    def test_gb_from_parameters_unknown_lattice_type(self):
        generator = GrainBoundaryGenerator(self.Cu_conv)
        generator.lat_type = "x"
        with pytest.raises(RuntimeError, match="Lattice type not implemented"):
            generator.gb_from_parameters([1, 1, 1], 60)
        with pytest.raises(RuntimeError, match="Lattice type not implemented"):
            generator.get_ratio()

    def test_gb_from_parameters_non_reduced_axis(self):
        gb_222 = self.GB_Cu_conv.gb_from_parameters([2, 2, 2], 60, expand_times=1)
        gb_111 = self.GB_Cu_conv.gb_from_parameters([1, 1, 1], 60, expand_times=1)
        assert gb_222.rotation_axis == (1, 1, 1)
        assert gb_222.gb_plane == (1, 1, 1)
        assert gb_222.sigma == gb_111.sigma == 3
        assert len(gb_222) == len(gb_111)

    def test_gb_from_parameters_four_index_hex(self):
        angle = 21.786789298261812
        gb_3idx = self.GB_Be.gb_from_parameters([0, 0, 1], angle, ratio=[5, 2], expand_times=1)
        gb_4idx = self.GB_Be.gb_from_parameters([0, 0, 0, 1], angle, ratio=[5, 2], expand_times=1, plane=[0, 0, 0, 1])
        assert gb_4idx.rotation_axis == gb_3idx.rotation_axis == (0, 0, 1)
        assert gb_4idx.gb_plane == gb_3idx.gb_plane == (0, 0, 1)
        assert gb_4idx.sigma == gb_3idx.sigma == 7
        assert_allclose(gb_4idx.lattice.abc, gb_3idx.lattice.abc)

    @pytest.mark.parametrize("ratio", [[15, 4], None])
    def test_gb_from_parameters_rhombohedral(self, ratio):
        angle = GrainBoundaryGenerator.get_rotation_angle_from_sigma(7, [1, 1, 1], lat_type="r", ratio=[15, 4])[0]
        gb_4idx = self.GB_Bi.gb_from_parameters([0, 0, 0, 1], angle, ratio=ratio, expand_times=1, max_search=5)
        gb_3idx = self.GB_Bi.gb_from_parameters([1, 1, 1], angle, ratio=ratio, expand_times=1, max_search=5)
        assert gb_4idx.rotation_axis == gb_3idx.rotation_axis == (1, 1, 1)
        assert gb_4idx.gb_plane == gb_3idx.gb_plane == (1, 1, 1)
        assert gb_4idx.sigma == gb_3idx.sigma == 7
        assert len(gb_4idx) == len(gb_3idx)

    def test_gb_from_parameters_tetragonal_irrational(self):
        gb = self.GB_Pa.gb_from_parameters([0, 0, 1], 36.86989764584402, ratio=None, expand_times=1)
        assert gb.rotation_axis == (0, 0, 1)
        assert gb.gb_plane == (0, 0, 1)
        assert gb.sigma == 5

    @pytest.mark.parametrize(
        ("axis", "ratio"),
        [([1, 0, 0], [21, 20, None]), ([0, 0, 1], [None, 20, 5]), ([0, 1, 0], [21, None, 5])],
    )
    def test_gb_from_parameters_orthorhombic_irrational_axis(self, axis, ratio):
        gb = self.GB_Br.gb_from_parameters(axis, 180.0, ratio=ratio, expand_times=1)
        assert gb.rotation_axis == tuple(axis)
        assert gb.gb_plane == tuple(axis)
        assert gb.join_plane == tuple(axis)
        # A 180 degree twist about a principal axis of an orthorhombic cell is a symmetry operation
        assert gb.sigma == 1

    @pytest.mark.parametrize(
        ("index_none", "expected"),
        [(0, [None, 4, 1]), (1, [21, None, 5]), (2, [1, 1, None])],
    )
    def test_get_ratio_orthorhombic_index_none(self, index_none, expected):
        assert self.GB_Br.get_ratio(index_none=index_none) == expected

    def test_get_ratio_cubic(self):
        assert self.GB_Cu_conv.get_ratio() is None

    @pytest.mark.parametrize(
        ("lat_type", "axis_4", "axis_3", "ratio"),
        [("h", [0, 0, 0, 1], [0, 0, 1], [5, 2]), ("r", [0, 0, 0, 1], [1, 1, 1], [15, 4])],
    )
    def test_get_trans_mat_four_index(self, lat_type, axis_4, axis_3, ratio):
        angle = GrainBoundaryGenerator.get_rotation_angle_from_sigma(7, axis_3, lat_type=lat_type, ratio=ratio)[0]
        t1_4, t2_4 = GrainBoundaryGenerator.get_trans_mat(axis_4, angle, lat_type=lat_type, ratio=ratio)
        t1_3, t2_3 = GrainBoundaryGenerator.get_trans_mat(axis_3, angle, lat_type=lat_type, ratio=ratio)
        assert_allclose(t1_4, t1_3)
        assert_allclose(t2_4, t2_3)
        assert abs(np.linalg.det(t1_4)) == approx(7)

    @pytest.mark.parametrize(
        ("lat_type", "axis", "ratio", "reduced_axis", "reduced_ratio"),
        [
            ("c", [2, 2, 0], None, [1, 1, 0], None),
            ("h", [0, 0, 1], [10, 4], [0, 0, 1], [5, 2]),
            ("r", [1, 1, 1], [30, 8], [1, 1, 1], [15, 4]),
            ("o", [2, 0, 0], [42, 40, 10], [1, 0, 0], [21, 20, 5]),
        ],
    )
    def test_get_trans_mat_reduces_axis_and_ratio(self, lat_type, axis, ratio, reduced_axis, reduced_ratio):
        t1, t2 = GrainBoundaryGenerator.get_trans_mat(axis, 180.0, lat_type=lat_type, ratio=ratio)
        t1_ref, t2_ref = GrainBoundaryGenerator.get_trans_mat(
            reduced_axis, 180.0, lat_type=lat_type, ratio=reduced_ratio
        )
        assert_allclose(t1, t1_ref)
        assert_allclose(t2, t2_ref)

    def test_get_trans_mat_orthorhombic_non_coprime_ratio(self):
        kwargs = {"lat_type": "o", "surface": [21, 20, 10], "normal": True}
        t1, t2 = GrainBoundaryGenerator.get_trans_mat([1, 1, 1], 95.55344419565849, ratio=[20, 40, 42], **kwargs)
        t1_ref, t2_ref = GrainBoundaryGenerator.get_trans_mat(
            [1, 1, 1], 95.55344419565849, ratio=[10, 20, 21], **kwargs
        )
        assert_allclose(t1, t1_ref)
        assert_allclose(t2, t2_ref)

    @pytest.mark.parametrize(
        ("lat_type", "axis", "ratio"),
        [
            ("h", [1, 0, -1, 0], None),
            ("t", [0, 0, 1], None),
            ("t", [0, 0, 1], [2, 3]),
            ("o", [1, 0, 0], [None, 20, 5]),
            ("o", [0, 1, 0], [21, None, 5]),
            ("o", [0, 0, 1], [21, 20, None]),
        ],
    )
    def test_get_trans_mat_default_surface(self, lat_type, axis, ratio):
        # A 180 degree twist about a principal axis maps the lattice onto itself, so |det(t1)| == sigma == 1
        t1, t2 = GrainBoundaryGenerator.get_trans_mat(axis, 180.0, lat_type=lat_type, ratio=ratio)
        assert abs(np.linalg.det(t1)) == approx(1)
        assert abs(np.linalg.det(t2)) == approx(1)

    @pytest.mark.parametrize(
        ("axis", "ratio", "surface"),
        [
            ([1, 0, 0], [None, 20, 5], [1, 0, 0]),
            ([0, 1, 0], [21, None, 5], [0, 1, 0]),
            ([0, 0, 1], [21, 20, None], [0, 0, 1]),
        ],
    )
    def test_get_trans_mat_orthorhombic_irrational_with_surface(self, axis, ratio, surface):
        t1, t2 = GrainBoundaryGenerator.get_trans_mat(axis, 180.0, lat_type="o", ratio=ratio, surface=surface)
        assert abs(np.linalg.det(t1)) == approx(1)
        assert abs(np.linalg.det(t2)) == approx(1)

    @pytest.mark.parametrize(
        ("lat_type", "axis", "ratio", "surface", "exc", "match"),
        [
            ("h", [1, 0, 1], None, None, RuntimeError, r"For irrational c2/a2, CSL only exist"),
            ("r", [1, 0, 0], None, None, RuntimeError, "For irrational ratio_alpha, CSL only exist"),
            ("t", [1, 0, 1], None, None, RuntimeError, r"For irrational c2/a2, CSL only exist"),
            ("o", [1, 0, 0], None, None, ValueError, "Invalid ratio=None for orthorhombic system"),
            ("x", [1, 0, 0], None, None, RuntimeError, "Lattice type has not implemented"),
            ("o", [1, 1, 1], [None, 20, 5], [1, 1, 1], RuntimeError, "For irrational c2, CSL only exist"),
            ("o", [1, 1, 1], [21, None, 5], [1, 1, 1], RuntimeError, "For irrational b2, CSL only exist"),
            ("o", [1, 1, 1], [21, 20, None], [1, 1, 1], RuntimeError, "For irrational a2, CSL only exist"),
            ("o", [1, 0, 0], [None, None, 5], [1, 0, 0], RuntimeError, "No CSL exist for two irrational numbers"),
        ],
    )
    def test_get_trans_mat_errors(self, lat_type, axis, ratio, surface, exc, match):
        with pytest.raises(exc, match=match):
            GrainBoundaryGenerator.get_trans_mat(axis, 180.0, lat_type=lat_type, ratio=ratio, surface=surface)

    def test_get_trans_mat_four_index_surface(self):
        angle = 21.786789298261812
        t1_4, t2_4 = GrainBoundaryGenerator.get_trans_mat(
            [0, 0, 1], angle, lat_type="h", ratio=[5, 2], surface=[0, 0, 0, 2]
        )
        t1_3, t2_3 = GrainBoundaryGenerator.get_trans_mat(
            [0, 0, 1], angle, lat_type="h", ratio=[5, 2], surface=[0, 0, 1]
        )
        assert_allclose(t1_4, t1_3)
        assert_allclose(t2_4, t2_3)
        # Twist boundary: a and b lie in the (001) plane
        assert t1_4[0][2] == t1_4[1][2] == 0

    def test_get_trans_mat_normal_not_found(self):
        with pytest.warns(UserWarning, match="Cannot find the perpendicular c vector, please increase max_search"):
            t1, _ = GrainBoundaryGenerator.get_trans_mat([1, 1, 1], 60, surface=[1, 2, 3], normal=True, max_search=1)
        # Falls back to a c vector that is not normal to the (123) plane
        assert np.linalg.norm(np.cross(t1[2], [1, 2, 3])) > 1e-8

    @pytest.mark.parametrize("surface", [[1, 2, 3], [3, 1, 0]])
    def test_get_trans_mat_normal(self, surface):
        # For (310), no normal c vector is found in the initial search range, so the range is extended
        t1, t2 = GrainBoundaryGenerator.get_trans_mat([1, 1, 1], 60, surface=surface, normal=True)
        assert_allclose(np.cross(t1[2], surface), 0)
        assert np.dot(t1[0], surface) == np.dot(t1[1], surface) == 0
        assert abs(np.linalg.det(t1)) == approx(abs(np.linalg.det(t2)))

    def test_reduce_mat_not_reduced(self):
        mat = np.eye(3, dtype=int)
        with pytest.warns(UserWarning, match="Matrix reduction not performed"):
            reduced = GrainBoundaryGenerator.reduce_mat(mat.copy(), 2, np.eye(3))
        assert_allclose(reduced, mat)

    def test_enum_sigma_hex_variants(self):
        ref = GrainBoundaryGenerator.enum_sigma_hex(30, [0, 0, 1], [5, 2])
        assert sorted(ref) == [7, 13, 19]
        # Four-index notation and non-coprime ratio give the same result
        assert GrainBoundaryGenerator.enum_sigma_hex(30, [0, 0, 0, 1], [5, 2]) == ref
        assert GrainBoundaryGenerator.enum_sigma_hex(30, [0, 0, 1], [10, 4]) == ref
        # Irrational c2/a2 is fine for [001] (twist about c does not involve the c/a ratio)
        irrational = GrainBoundaryGenerator.enum_sigma_hex(30, [0, 0, 1], None)
        assert sorted(irrational) == sorted(ref)
        for sigma, angles in ref.items():
            assert_allclose(sorted(irrational[sigma]), sorted(angles))
        # For [u, v, 0] only m = 0 (180 degree rotation) is allowed, which gives sigma 1 for [100]
        assert GrainBoundaryGenerator.enum_sigma_hex(30, [1, 0, 0], None) == {}
        with pytest.raises(RuntimeError, match=r"For irrational c2/a2, CSL only exist"):
            GrainBoundaryGenerator.enum_sigma_hex(30, [1, 0, 1], None)

    def test_enum_sigma_rho_variants(self):
        ref = GrainBoundaryGenerator.enum_sigma_rho(30, [1, 1, 1], [15, 4])
        assert sorted(ref) == [3, 7, 13, 19, 21]
        assert ref[3] == approx([180.0, 60.0])
        assert GrainBoundaryGenerator.enum_sigma_rho(30, [0, 0, 0, 1], [15, 4]) == ref
        assert GrainBoundaryGenerator.enum_sigma_rho(30, [2, 2, 2], [30, 8]) == ref
        irrational = GrainBoundaryGenerator.enum_sigma_rho(30, [1, 1, 1], None)
        assert sorted(irrational) == sorted(ref)
        assert GrainBoundaryGenerator.enum_sigma_rho(30, [1, -1, 0], None) == {}
        with pytest.raises(RuntimeError, match="For irrational ratio_alpha, CSL only exist"):
            GrainBoundaryGenerator.enum_sigma_rho(30, [1, 0, 0], None)

    def test_enum_sigma_tet_variants(self):
        ref = GrainBoundaryGenerator.enum_sigma_tet(30, [1, 0, 0], [9, 1])
        assert GrainBoundaryGenerator.enum_sigma_tet(30, [2, 0, 0], [18, 2]) == ref
        irrational = GrainBoundaryGenerator.enum_sigma_tet(30, [0, 0, 1], None)
        assert sorted(irrational) == [5, 13, 17, 25, 29]
        assert irrational[5] == approx([53.13010235415599, 36.86989764584402, 126.86989764584403, 143.13010235415598])
        assert GrainBoundaryGenerator.enum_sigma_tet(30, [1, 0, 0], None) == {}
        with pytest.raises(RuntimeError, match=r"For irrational c2/a2, CSL only exist"):
            GrainBoundaryGenerator.enum_sigma_tet(30, [1, 0, 1], None)

    def test_enum_sigma_ort_variants(self):
        ref = GrainBoundaryGenerator.enum_sigma_ort(30, [1, 0, 0], [270, 30, 29])
        assert sorted(ref) == [3, 5, 13, 15, 17, 25, 29]
        # Non-reduced axis and ratio
        doubled = GrainBoundaryGenerator.enum_sigma_ort(30, [2, 0, 0], [540, 60, 58])
        assert sorted(doubled) == sorted(ref)
        # a2 does not enter a rotation about [100], so an irrational a2 gives the same sigmas
        irrational_a = GrainBoundaryGenerator.enum_sigma_ort(30, [1, 0, 0], [270, 30, None])
        assert sorted(irrational_a) == sorted(ref)
        # Irrational c2 with [100] only allows m = 0 -> sigma 1 -> nothing within cutoff
        assert GrainBoundaryGenerator.enum_sigma_ort(30, [1, 0, 0], [None, 60, 58]) == {}
        assert GrainBoundaryGenerator.enum_sigma_ort(30, [0, 1, 0], [270, None, 29]) == {}
        # For principal axes only the two in-plane axial ratios matter
        assert sorted(GrainBoundaryGenerator.enum_sigma_ort(30, [0, 1, 0], [270, 30, 29])) == sorted(
            GrainBoundaryGenerator.enum_sigma_ort(30, [0, 1, 0], [270, 1, 29])
        )
        assert sorted(GrainBoundaryGenerator.enum_sigma_ort(30, [0, 0, 1], [270, 30, 29])) == sorted(
            GrainBoundaryGenerator.enum_sigma_ort(30, [0, 0, 1], [1, 30, 29])
        )

    @pytest.mark.parametrize(
        ("ratio", "match"),
        [
            ([None, 30, 29], "For irrational c2, CSL only exist"),
            ([270, None, 29], "For irrational b2, CSL only exist"),
            ([270, 30, None], "For irrational a2, CSL only exist"),
            ([None, None, 29], "No CSL exist for two irrational numbers"),
        ],
    )
    def test_enum_sigma_ort_errors(self, ratio, match):
        with pytest.raises(RuntimeError, match=match):
            GrainBoundaryGenerator.enum_sigma_ort(30, [1, 1, 1], ratio)

    @pytest.mark.parametrize(
        ("lat_type", "axis", "ratio", "sigma", "expected"),
        [
            ("t", [0, 0, 1], [2, 3], 5, [36.86989764584402, 53.13010235415599, 126.86989764584403, 143.13010235415598]),
            ("h", [0, 0, 1], [5, 2], 7, [21.786789298261812, 38.21321070173819, 81.78678929826182, 98.2132107017382]),
            ("r", [1, 1, 1], [15, 4], 7, [38.21321070173819, 81.78678929826181, 158.2132107017382]),
        ],
    )
    def test_get_rotation_angle_from_sigma_lattices(self, lat_type, axis, ratio, sigma, expected):
        angles = GrainBoundaryGenerator.get_rotation_angle_from_sigma(sigma, axis, lat_type=lat_type, ratio=ratio)
        assert_allclose(sorted(angles)[: len(expected)], expected)

    @pytest.mark.parametrize(
        ("lat_type", "axis", "ratio", "exc", "match"),
        [
            ("c", [0, 0, 0, 1], None, ValueError, "expect r_axis length 3 for selected lattice system, got 4"),
            ("o", [1, 0, 0], [1, 2], RuntimeError, "Orthorhombic system needs correct c2:b2:a2 ratio"),
            ("t", [0, 0, 1], None, RuntimeError, "System needs correct c2/a2 ratio"),
            ("h", [0, 0, 1], [1, 2, 3], RuntimeError, "System needs correct c2/a2 ratio"),
            ("r", [1, 1, 1], None, RuntimeError, r"Rhombohedral system needs correct \(1\+2\*cos\(alpha\)"),
            ("x", [1, 1, 1], None, RuntimeError, "Lattice type not implemented"),
            ("c", [1, 0, 0], None, RuntimeError, "This is a wrong sigma value, and no sigma exists smaller"),
        ],
    )
    def test_get_rotation_angle_from_sigma_errors(self, lat_type, axis, ratio, exc, match):
        with pytest.raises(exc, match=match):
            GrainBoundaryGenerator.get_rotation_angle_from_sigma(3, axis, lat_type=lat_type, ratio=ratio)


class TestInterface(MatSciTest):
    def setup_method(self):
        self.interface: Interface = self.get_structure("Si_SiO2_Interface")

    def test_basic_props(self):
        interface = self.interface
        assert isinstance(interface, Interface)

        assert len(interface.substrate_indices) == 14
        assert len(interface.film_indices) == 36
        assert len(interface.film_sites) == len(interface.film_indices)
        assert len(interface.substrate_sites) == len(interface.substrate_indices)
        assert interface.gap == approx(2)
        assert_allclose(interface.in_plane_offset, [0, 0])
        assert interface.vacuum_over_film == approx(20)
        assert interface.film_termination == "O2_P6/mmm_4"
        assert interface.substrate_termination == "Si_P6/mmm_7"
        assert interface.film_layers == 6
        assert interface.substrate_layers == 2

        iface_dict = interface.as_dict()
        expected_keys = {
            "lattice",
            "sites",
            "in_plane_offset",
            "gap",
            "vacuum_over_film",
            "interface_properties",
        }
        assert expected_keys <= {*iface_dict}
        assert isinstance(interface.from_dict(iface_dict), Interface)

    def test_gap_setter(self):
        interface = self.interface

        assert_allclose(interface.gap, 2.0)

        max_sub_c = np.max(np.array([site.frac_coords for site in interface.substrate])[:, 2])
        min_film_c = np.min(np.array([site.frac_coords for site in interface.film])[:, 2])
        gap = (min_film_c - max_sub_c) * interface.lattice.c
        assert_allclose(interface.gap, gap)

        interface.gap += 1

        assert_allclose(interface.gap, 3.0)

        max_sub_c = np.max(np.array([site.frac_coords for site in interface.substrate])[:, 2])
        min_film_c = np.min(np.array([site.frac_coords for site in interface.film])[:, 2])
        gap = (min_film_c - max_sub_c) * interface.lattice.c
        assert_allclose(interface.gap, gap)

    def test_in_plane_offset_setter(self):
        interface = self.interface
        init_coords = np.array(self.interface.frac_coords)
        interface.in_plane_offset = np.array([0.2, 0.2])

        assert_allclose(interface.in_plane_offset, [0.2, 0.2])

        test_coords = np.array(init_coords)
        for idx in interface.film_indices:
            test_coords[idx] += [0.2, 0.2, 0]
        assert_allclose(np.mod(test_coords, 1.0), np.mod(interface.frac_coords, 1.0))

    def test_vacuum_over_film_setter(self):
        interface = self.interface
        init_coords = self.interface.cart_coords

        assert interface.vacuum_over_film == 20

        interface.vacuum_over_film += 10

        assert interface.vacuum_over_film == 30
        assert_allclose(init_coords, interface.cart_coords)

    def test_get_shifts_based_on_adsorbate_sites(self):
        # Only testing two tolerances as there appears to be significant numerical noise in this method
        assert len(self.interface.get_shifts_based_on_adsorbate_sites()) == 42
        assert len(self.interface.get_shifts_based_on_adsorbate_sites(tolerance=20.0)) == 1

    def test_from_slabs(self):
        si_struct = self.get_structure("Si")
        sio2_struct = self.get_structure("SiO2")
        si_conventional = SpacegroupAnalyzer(si_struct).get_conventional_standard_structure()
        sio2_conventional = SpacegroupAnalyzer(sio2_struct).get_conventional_standard_structure()

        si_slab = SlabGenerator(si_conventional, (1, 1, 1), 5, 10, reorient_lattice=True).get_slab()
        sio2_slab = SlabGenerator(sio2_conventional, (1, 0, 0), 5, 10, reorient_lattice=True).get_slab()

        interface = Interface.from_slabs(film_slab=si_slab, substrate_slab=sio2_slab)
        assert isinstance(interface, Interface)

    def test_init_requires_interface_label(self):
        with pytest.raises(RuntimeError, match="Must provide labeling of substrate and film sites"):
            Interface(Lattice.cubic(3), ["Si"], np.array([[0, 0, 0]]), site_properties={})

    @pytest.mark.parametrize(
        ("attr", "value", "match"),
        [
            ("in_plane_offset", [0.1, 0.2, 0.3], "In-plane shifts require two floats"),
            ("gap", -1, "Can't reduce interface gap below 0"),
            ("vacuum_over_film", -1, "The vacuum over the film can not be less then 0"),
        ],
    )
    def test_setter_errors(self, attr, value, match):
        with pytest.raises(ValueError, match=match):
            setattr(self.interface, attr, value)

    def test_update_c_error(self):
        with pytest.raises(ValueError, match="New c-length must be greater than 0"):
            self.interface._update_c(0)

    def test_copy(self):
        iface_copy = self.interface.copy()
        assert isinstance(iface_copy, Interface)
        assert iface_copy is not self.interface
        assert iface_copy.gap == approx(self.interface.gap)
        assert iface_copy.vacuum_over_film == approx(self.interface.vacuum_over_film)
        assert iface_copy.film_indices == self.interface.film_indices
        assert_allclose(iface_copy.cart_coords, self.interface.cart_coords)

    def test_get_sorted_structure(self):
        sorted_struct = self.interface.get_sorted_structure(key=lambda site: site.frac_coords[2])
        assert type(sorted_struct) is Structure
        assert len(sorted_struct) == len(self.interface)
        assert np.all(np.diff(sorted_struct.frac_coords[:, 2]) >= 0)
        assert sorted_struct.composition == self.interface.composition

    def test_from_slabs_invalid_angles(self):
        oblique = Structure(Lattice.from_parameters(3, 3, 10, 60, 90, 90), ["Si"], [[0, 0, 0.5]])
        orthogonal = Structure(Lattice.tetragonal(3, 10), ["Si"], [[0, 0, 0.5]])
        with pytest.raises(ValueError, match="alpha or beta of the film slab are not approximately 90"):
            Interface.from_slabs(substrate_slab=orthogonal, film_slab=oblique)
        with pytest.raises(ValueError, match="alpha or beta of the substrate slab are not approximately 90"):
            Interface.from_slabs(substrate_slab=oblique, film_slab=orthogonal)

    def test_from_slabs_left_handed_substrate(self):
        substrate = Structure(
            Lattice([[3, 0, 0], [0, 3, 0], [0, 0, -10]]),
            ["Si", "Si"],
            [[0, 0, 0.1], [0.5, 0.5, 0.2]],
            site_properties={"magmom": [0, 0]},
        )
        film = Structure(
            Lattice.tetragonal(3, 10),
            ["Ge", "Ge"],
            [[0, 0, 0.1], [0.5, 0.5, 0.2]],
            site_properties={"magmom": [1, 1], "film_only": [1, 1]},
        )
        iface = Interface.from_slabs(substrate_slab=substrate, film_slab=film, gap=2.0, vacuum_over_film=5.0)
        # The substrate c vector is flipped so the lattice is right-handed
        assert np.linalg.det(iface.lattice.matrix) > 0
        assert iface.gap == approx(2.0)
        assert iface.vacuum_over_film == approx(5.0)
        # Only site properties present in both slabs are kept
        assert set(iface.site_properties) == {"magmom", "interface_label"}
        assert iface.substrate.composition.reduced_formula == "Si"
        assert iface.film.composition.reduced_formula == "Ge"
        # c = film thickness (1 A) + substrate thickness (1 A) + gap + vacuum
        assert iface.lattice.c == approx(1 + 1 + 2 + 5)


def test_label_termination():
    single = Structure(Lattice.tetragonal(3, 10), ["Si"], [[0, 0, 0.5]])
    assert label_termination(single) == "Si_P4/mmm_1"

    slab = Structure(
        Lattice.tetragonal(3, 10),
        ["Si", "O", "O"],
        [[0, 0, 0.2], [0, 0, 0.6], [0.5, 0.5, 0.6]],
    )
    # The topmost plane holds the two O sites
    assert label_termination(slab) == "O2_P4/mmm_2"
    assert label_termination(slab, t_idx=3) == "3_O2_P4/mmm_2"


def test_count_layers():
    single = Structure(Lattice.tetragonal(3, 10), ["Si", "O"], [[0, 0, 0.5], [0.5, 0.5, 0.2]])
    assert count_layers(single) == 1
    assert count_layers(single, Element("O")) == 1

    layered = Structure(
        Lattice.tetragonal(3, 20),
        ["Si", "Si", "Si", "O"],
        [[0, 0, 0.1], [0.5, 0.5, 0.1], [0, 0, 0.4], [0, 0, 0.7]],
    )
    assert count_layers(layered, Element("Si")) == 2
