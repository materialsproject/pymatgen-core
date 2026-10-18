from __future__ import annotations

import os

import numpy as np
import orjson
import pytest
from numpy.testing import assert_allclose
from pytest import approx

import pymatgen.core
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.core.local_env import VoronoiNN
from pymatgen.core.structure_matcher import StructureMatcher
from pymatgen.core.surface import (
    RECONSTRUCTIONS_ARCHIVE,
    ReconstructionGenerator,
    Slab,
    SlabGenerator,
    center_slab,
    generate_all_slabs,
    get_d,
    get_slab_regions,
    get_symmetrically_distinct_miller_indices,
    get_symmetrically_equivalent_miller_indices,
    hkl_transformation,
    miller_index_from_sites,
    uvw_transformation,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

PMG_CORE_DIR = os.path.dirname(pymatgen.core.__file__)


class TestSlab(MatSciTest):
    def setup_method(self):
        zno1 = Structure.from_file(f"{TEST_FILES_DIR}/surfaces/ZnO-wz.cif", primitive=False)
        zno55 = SlabGenerator(zno1, [1, 0, 0], 5, 5, lll_reduce=False, center_slab=False).get_slab()

        Ti = Structure(
            Lattice.hexagonal(4.6, 2.82),
            ["Ti", "Ti", "Ti"],
            [
                [0, 0, 0],
                [0.333333, 0.666667, 0.5],
                [0.666667, 0.333333, 0.5],
            ],
        )

        Ag_fcc = Structure(
            Lattice.cubic(4.06),
            ["Ag", "Ag", "Ag", "Ag"],
            [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
        )

        lattice = Lattice([[3.913449, 0, 0], [0, 3.913449, 0], [0, 0, 5.842644]])
        frac_coords = [
            [0.5, 0, 0.222518],
            [0, 0.5, 0.777482],
            [0, 0, 0],
            [0, 0, 0.5],
            [0.5, 0.5, 0],
        ]
        non_laue = Structure(lattice, ["Nb", "Nb", "N", "N", "N"], frac_coords)

        self.ti = Ti
        self.ag_fcc = Ag_fcc
        self.zno1 = zno1
        self.zno55 = zno55
        self.non_laue = non_laue
        self.hydrogen = Structure(Lattice.cubic(3), ["H"], [[0, 0, 0]])
        self.li_bcc = Structure(Lattice.cubic(3.51004), ["Li", "Li"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    def test_init(self):
        zno_slab = Slab(
            self.zno55.lattice,
            self.zno55.species,
            self.zno55.frac_coords,
            self.zno55.miller_index,
            self.zno55.oriented_unit_cell,
            0,
            self.zno55.scale_factor,
        )
        matrix = self.zno55.lattice.matrix
        area = np.linalg.norm(np.cross(matrix[0], matrix[1]))
        assert zno_slab.surface_area == approx(area)
        assert zno_slab.lattice.parameters == self.zno55.lattice.parameters
        assert zno_slab.oriented_unit_cell.composition == self.zno1.composition
        assert len(zno_slab) == 8

        # check reorient_lattice. get a slab not oriented and check that orientation
        # works even with Cartesian coordinates.
        zno_not_or = SlabGenerator(
            self.zno1,
            [1, 0, 0],
            5,
            5,
            lll_reduce=False,
            center_slab=False,
            reorient_lattice=False,
        ).get_slab()
        zno_slab_cart = Slab(
            zno_not_or.lattice,
            zno_not_or.species,
            zno_not_or.cart_coords,
            zno_not_or.miller_index,
            zno_not_or.oriented_unit_cell,
            0,
            zno_not_or.scale_factor,
            coords_are_cartesian=True,
            reorient_lattice=True,
        )
        assert_allclose(zno_slab.frac_coords, zno_slab_cart.frac_coords, atol=1e-12)
        c = zno_slab_cart.lattice.matrix[2]
        assert_allclose([0, 0, np.linalg.norm(c)], c)

    def test_add_adsorbate_atom(self):
        zno_slab = Slab(
            self.zno55.lattice,
            self.zno55.species,
            self.zno55.frac_coords,
            self.zno55.miller_index,
            self.zno55.oriented_unit_cell,
            0,
            self.zno55.scale_factor,
        )
        returned = zno_slab.add_adsorbate_atom([1], "H", 1)
        assert returned == zno_slab

        assert len(zno_slab) == 9
        assert str(zno_slab[8].specie) == "H"
        assert zno_slab.get_distance(1, 8) == approx(1.0)
        assert zno_slab[8].c > zno_slab[0].c
        matrix = self.zno55.lattice.matrix
        area = np.linalg.norm(np.cross(matrix[0], matrix[1]))
        assert zno_slab.surface_area == approx(area)
        assert zno_slab.lattice.parameters == self.zno55.lattice.parameters

    def test_get_sorted_structure(self):
        species = [str(site.specie) for site in self.zno55.get_sorted_structure()]
        assert species == ["Zn2+"] * 4 + ["O2-"] * 4

    def test_methods(self):
        # Test various structure methods
        self.zno55.get_primitive_structure()

    def test_as_from_dict(self):
        dct = self.zno55.as_dict()
        obj = Slab.from_dict(dct)
        assert obj.miller_index == (1, 0, 0)

    def test_dipole_and_is_polar(self):
        assert_allclose(self.zno55.dipole, [0, 0, 0], atol=1e-9)
        assert not self.zno55.is_polar()
        cscl = self.get_structure("CsCl")
        cscl.add_oxidation_state_by_element({"Cs": 1, "Cl": -1})
        slab = SlabGenerator(
            cscl,
            [1, 0, 0],
            5,
            5,
            reorient_lattice=False,
            lll_reduce=False,
            center_slab=False,
        ).get_slab()
        assert_allclose(slab.dipole, [-4.209, 0, 0])
        assert slab.is_polar()

    @pytest.mark.parametrize("center_slab", [True, False])
    def test_surface_sites_and_symmetry(self, center_slab):
        # test if surfaces are equivalent by using
        # Laue symmetry and surface site equivalence

        # We will also set the slab to be centered and
        # off centered in order to test the center of mass
        slab_gen = SlabGenerator(self.ag_fcc, (3, 1, 0), 10, 10, center_slab=center_slab)
        slab = slab_gen.get_slabs()[0]
        surf_sites_dict = slab.get_surface_sites()
        assert len(surf_sites_dict["top"]) == len(surf_sites_dict["bottom"])
        total_surf_sites = sum(len(surf_sites_dict[key]) for key in surf_sites_dict)
        assert slab.is_symmetric()
        assert total_surf_sites == 8

        # Test if the ratio of surface sites per area is
        # constant, i.e. are the surface energies the same?
        r1 = total_surf_sites / (2 * slab.surface_area)
        slab_gen = SlabGenerator(self.ag_fcc, (3, 1, 0), 10, 10, primitive=False)
        slab = slab_gen.get_slabs()[0]
        surf_sites_dict = slab.get_surface_sites()
        total_surf_sites = sum(len(surf_sites_dict[key]) for key in surf_sites_dict)
        r2 = total_surf_sites / (2 * slab.surface_area)
        assert_allclose(r1, r2)

    def test_symmetrization(self):
        # Restricted to primitive_elemental materials due to the risk of
        # broken stoichiometry. For compound materials, use is_polar()

        # Get all slabs for P6/mmm Ti and Fm-3m Ag up to index of 2

        all_Ti_slabs = generate_all_slabs(
            self.ti,
            2,
            10,
            10,
            bonds=None,
            tol=1e-3,
            max_broken_bonds=0,
            lll_reduce=False,
            center_slab=False,
            primitive=True,
            max_normal_search=2,
            symmetrize=True,
        )

        all_Ag_fcc_slabs = generate_all_slabs(
            self.ag_fcc,
            2,
            10,
            10,
            bonds=None,
            tol=1e-3,
            max_broken_bonds=0,
            lll_reduce=False,
            center_slab=False,
            primitive=True,
            max_normal_search=2,
            symmetrize=True,
        )

        all_slabs = [all_Ti_slabs, all_Ag_fcc_slabs]

        for slabs in all_slabs:
            asymmetric_count = symmetric_count = 0

            for slab in slabs:
                sg = SpacegroupAnalyzer(slab)

                # Check if a slab is symmetric
                if not sg.is_laue():
                    asymmetric_count += 1
                else:
                    symmetric_count += 1

            # Check if slabs are all symmetric
            assert asymmetric_count == 0
            assert symmetric_count == len(slabs)

        # Check if we can generate symmetric slabs from bulk with no inversion
        all_non_laue_slabs = generate_all_slabs(self.non_laue, 1, 15, 15, symmetrize=True)
        assert len(all_non_laue_slabs) > 0

    def test_get_symmetric_sites(self):
        # Check if we get an equivalent site on one
        # surface if we add a new site to the other surface

        all_Ti_slabs = generate_all_slabs(
            self.ti,
            2,
            10,
            10,
            bonds=None,
            tol=1e-3,
            max_broken_bonds=0,
            lll_reduce=False,
            center_slab=False,
            primitive=True,
            max_normal_search=2,
            symmetrize=True,
        )

        for slab in all_Ti_slabs:
            sorted_sites = sorted(slab, key=lambda site: site.frac_coords[2])
            site = sorted_sites[-1]
            point = np.array(site.frac_coords)
            point[2] += 0.1
            point2 = slab.get_symmetric_site(point)
            slab.append("O", point)
            slab.append("O", point2)

            # Check if slab is all symmetric
            sg = SpacegroupAnalyzer(slab)
            assert sg.is_laue()

    def test_oriented_unit_cell(self):
        # Check if we get the fully reduced oriented unit
        # cell. This will also ensure that the constrain_latt
        # parameter for get_primitive_structure is working properly

        def surface_area(s):
            matrix = s.lattice.matrix
            return np.linalg.norm(np.cross(matrix[0], matrix[1]))

        all_slabs = generate_all_slabs(self.ag_fcc, 2, 10, 10, max_normal_search=3)
        for slab in all_slabs:
            ouc = slab.oriented_unit_cell

            assert surface_area(slab) == approx(surface_area(ouc))
            assert len(slab) >= len(ouc)

    def test_get_slab_regions(self):
        # If a slab layer in the slab cell is not completely inside
        # the cell (noncontiguous), check that get_slab_regions will
        # be able to identify where the slab layers are located

        struct = self.get_structure("LiFePO4")
        slab_gen = SlabGenerator(struct, (0, 0, 1), 15, 15)
        slab = slab_gen.get_slabs()[0]
        slab.translate_sites([idx for idx, site in enumerate(slab)], [0, 0, -0.25])
        bottom_c, top_c = [], []
        for site in slab:
            if site.frac_coords[2] < 0.5:
                bottom_c.append(site.frac_coords[2])
            else:
                top_c.append(site.frac_coords[2])
        ranges = get_slab_regions(slab)
        assert tuple(ranges[0]) == (0, max(bottom_c))
        assert tuple(ranges[1]) == (min(top_c), 1)

    def test_as_dict(self):
        slabs = generate_all_slabs(
            self.ti,
            1,
            10,
            10,
            bonds=None,
            tol=1e-3,
            max_broken_bonds=0,
            lll_reduce=False,
            center_slab=False,
            primitive=True,
        )
        slab = slabs[0]
        dict_str = orjson.dumps(slab.as_dict(), option=orjson.OPT_SERIALIZE_NUMPY).decode()
        d = orjson.loads(dict_str)
        assert slab == Slab.from_dict(d)

        # test initializing with a list scale_factor
        slab = Slab(
            self.zno55.lattice,
            self.zno55.species,
            self.zno55.frac_coords,
            self.zno55.miller_index,
            self.zno55.oriented_unit_cell,
            0,
            self.zno55.scale_factor,
        )
        dict_str = orjson.dumps(slab.as_dict()).decode()
        d = orjson.loads(dict_str)
        assert slab == Slab.from_dict(d)

    def _ag_100_slab(self) -> Slab:
        """Six-layer centered Ag(100) slab with one site per layer."""
        return SlabGenerator(self.ag_fcc, (1, 0, 0), 10, 10, center_slab=True).get_slab()

    def test_str(self):
        lines = str(self.zno55).splitlines()
        assert lines[0] == "Slab Summary (Zn4 O4)"
        assert lines[1] == "Reduced Formula: ZnO"
        assert lines[2] == "Miller index: (1, 0, 0)"
        assert lines[3].startswith("Shift: 0.0000, Scale Factor: ")
        assert lines[-9] == "Sites (8)"
        site_lines = lines[-8:]
        for idx, (line, site) in enumerate(zip(site_lines, self.zno55, strict=True)):
            label, species, *coords = line.split()
            assert int(label) == idx + 1
            assert species == site.species_string
            assert_allclose([float(x) for x in coords], site.frac_coords, atol=1e-6)

    def test_add_adsorbate_atom_deprecated_specie(self):
        slab = self.zno55.copy()
        with pytest.warns(DeprecationWarning, match="'specie' is deprecated"):
            slab.add_adsorbate_atom([1], species="O", distance=1, specie="H")
        assert len(slab) == len(self.zno55) + 1
        assert slab[-1].species_string == "H"
        assert slab.get_distance(1, len(slab) - 1) == approx(1.0)

    def test_symmetrically_add_atom_deprecated_specie(self):
        slab = self._ag_100_slab()
        n_sites = len(slab)
        point = np.array([0.5, 0.5, slab.frac_coords[:, 2].max() + 0.05])
        with pytest.warns(DeprecationWarning, match="'specie' is deprecated"):
            slab.symmetrically_add_atom(species="O", point=point, specie="H")
        assert len(slab) == n_sites + 2
        assert [site.species_string for site in slab[-2:]] == ["H", "H"]
        assert_allclose(slab[-2].frac_coords, point)
        # The equivalent site sits on the opposite surface, mirrored about the slab center
        assert slab[-1].frac_coords[2] % 1 == approx(1 - point[2], abs=1e-6)
        assert slab.is_symmetric()

    def test_get_symmetric_site_no_symmetry_ops(self, monkeypatch):
        monkeypatch.setattr(SpacegroupAnalyzer, "get_symmetry_operations", lambda self, cartesian=False: [])
        with pytest.raises(RuntimeError, match="Failed to get symmetric site"):
            self.zno55.get_symmetric_site(np.array([0.5, 0.5, 0.5]))

    def test_get_surface_sites_tag(self):
        slab = self._ag_100_slab()
        surf_sites = slab.get_surface_sites(tag=True)
        z_coords = slab.frac_coords[:, 2]
        assert [idx for _, idx in surf_sites["top"]] == [int(np.argmax(z_coords))]
        assert [idx for _, idx in surf_sites["bottom"]] == [int(np.argmin(z_coords))]
        is_surf = slab.site_properties["is_surf_site"]
        assert len(is_surf) == len(slab)
        assert {idx for idx, flag in enumerate(is_surf) if flag} == {
            int(np.argmax(z_coords)),
            int(np.argmin(z_coords)),
        }

    def test_get_surface_sites_voronoi_error(self, monkeypatch):
        # A RuntimeError from VoronoiNN on the slab is treated as a surface site
        original_get_cn = VoronoiNN.get_cn

        def get_cn(self, structure, n, **kwargs):
            if not isinstance(structure, Slab):
                return original_get_cn(self, structure, n, **kwargs)
            raise RuntimeError("pathological Voronoi tessellation")

        monkeypatch.setattr(VoronoiNN, "get_cn", get_cn)
        slab = self._ag_100_slab()
        surf_sites = slab.get_surface_sites(tag=True)
        com_z = slab.center_of_mass[2]
        assert {idx for _, idx in surf_sites["top"]} == {
            idx for idx, site in enumerate(slab) if site.frac_coords[2] > com_z
        }
        assert {idx for _, idx in surf_sites["bottom"]} == {
            idx for idx, site in enumerate(slab) if site.frac_coords[2] <= com_z
        }
        assert all(slab.site_properties["is_surf_site"])

    def test_get_tasker2_slabs_odd_sites(self):
        # Ag(100) has a single atom per surface layer, so it cannot be split
        slab = SlabGenerator(self.ag_fcc, (1, 0, 0), 10, 10).get_slab()
        with pytest.warns(UserWarning, match="Odd number of sites to divide"):
            assert slab.get_tasker2_slabs() == []

    def test_symmetrically_remove_atoms(self):
        slab = self._ag_100_slab()
        z_coords = slab.frac_coords[:, 2]
        top_idx = int(np.argmax(z_coords))
        slab.symmetrically_remove_atoms([top_idx])
        assert len(slab) == 4
        assert_allclose(sorted(slab.frac_coords[:, 2]), sorted(z_coords)[1:-1])
        assert slab.is_symmetric()

    def test_symmetrically_remove_atoms_bottom_site(self):
        slab = self._ag_100_slab()
        z_coords = slab.frac_coords[:, 2]
        bottom_idx = int(np.argmin(z_coords))
        slab.symmetrically_remove_atoms([bottom_idx])
        assert len(slab) == 4
        assert slab.is_symmetric()

    def test_symmetrically_remove_atoms_multiple_sites(self):
        slab = self._ag_100_slab()
        z_coords = slab.frac_coords[:, 2]
        top_two = [int(idx) for idx in np.argsort(z_coords)[-2:]]
        slab.symmetrically_remove_atoms(top_two)
        # the two top layers and their two equivalent bottom layers are removed
        assert len(slab) == 2
        assert slab.is_symmetric()

    def test_symmetrically_remove_atoms_no_equivalent(self):
        mgo = Structure(
            Lattice.cubic(3.01),
            ["Mg"] * 4 + ["O"] * 4,
            [
                [0, 0, 0],
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0],
                [0.5, 0, 0],
                [0.5, 0.5, 0.5],
                [0, 0, 0.5],
                [0, 0.5, 0],
            ],
        )
        # Tasker 3 (111) slab: the two surfaces are terminated by different species
        slab = SlabGenerator(mgo, (1, 1, 1), 10, 10, max_normal_search=1).get_slab()
        assert not slab.is_symmetric()
        n_sites = len(slab)
        top_idx = int(np.argmax(slab.frac_coords[:, 2]))
        with pytest.warns(UserWarning, match="Equivalent sites could not be found"):
            slab.symmetrically_remove_atoms([top_idx])
        assert len(slab) == n_sites

    def test_center_slab_thin_cell(self):
        # When the c lattice parameter is smaller than the neighbor cutoff,
        # sites "spill over" and get shifted before the final centering
        struct = Structure(Lattice.tetragonal(2.87, 4.0), ["Ag"], [[0, 0, 0.1]])
        centered = center_slab(struct)
        assert_allclose(centered.frac_coords, [[0, 0, 0.5]], atol=1e-8)

    def test_center_slab(self):
        slab = self._ag_100_slab()
        z_coords = sorted(slab.frac_coords[:, 2])
        slab.translate_sites(list(range(len(slab))), [0, 0, -0.1])
        centered = center_slab(slab)
        assert_allclose(sorted(centered.frac_coords[:, 2]), z_coords, atol=1e-8)
        assert np.mean(centered.frac_coords[:, 2]) == approx(0.5)

    def test_center_slab_wrapped_across_boundary(self):
        slab = self._ag_100_slab()
        z_coords = sorted(slab.frac_coords[:, 2])
        slab.translate_sites(list(range(len(slab))), [0, 0, 0.5])
        centered = center_slab(slab)
        assert_allclose(sorted(centered.frac_coords[:, 2]), z_coords, atol=1e-6)

    def test_get_slab_regions_contiguous(self):
        slab = self._ag_100_slab()
        z_coords = slab.frac_coords[:, 2]
        assert_allclose(get_slab_regions(slab), [(z_coords.min(), z_coords.max())])


class TestSlabGenerator(MatSciTest):
    def setup_method(self):
        lattice = Lattice.cubic(3.010)
        frac_coords = [
            [0.00000, 0.00000, 0.00000],
            [0.00000, 0.50000, 0.50000],
            [0.50000, 0.00000, 0.50000],
            [0.50000, 0.50000, 0.00000],
            [0.50000, 0.00000, 0.00000],
            [0.50000, 0.50000, 0.50000],
            [0.00000, 0.00000, 0.50000],
            [0.00000, 0.50000, 0.00000],
        ]
        species = ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"]
        self.MgO = Structure(lattice, species, frac_coords)
        self.MgO.add_oxidation_state_by_element({"Mg": 2, "O": -6})

        lattice_Dy = Lattice.hexagonal(3.58, 25.61)
        frac_coords_Dy = [
            [0.00000, 0.00000, 0.00000],
            [0.66667, 0.33333, 0.11133],
            [0.00000, 0.00000, 0.222],
            [0.66667, 0.33333, 0.33333],
            [0.33333, 0.66666, 0.44467],
            [0.66667, 0.33333, 0.55533],
            [0.33333, 0.66667, 0.66667],
            [0.00000, 0.00000, 0.778],
            [0.33333, 0.66667, 0.88867],
        ]
        species_Dy = ["Dy", "Dy", "Dy", "Dy", "Dy", "Dy", "Dy", "Dy", "Dy"]
        self.Dy = Structure(lattice_Dy, species_Dy, frac_coords_Dy)

    def test_get_slab(self):
        struct = self.get_structure("LiFePO4")
        gen = SlabGenerator(struct, [0, 0, 1], 10, 10)
        struct = gen.get_slab(0.25)
        assert struct.lattice.abc[2] == approx(20.820740000000001)

        fcc = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        gen = SlabGenerator(fcc, [1, 1, 1], 10, 10, max_normal_search=1)
        slab = gen.get_slab()
        assert len(slab) == 6
        gen = SlabGenerator(fcc, [1, 1, 1], 10, 10, primitive=False, max_normal_search=1)
        slab_non_prim = gen.get_slab()
        assert len(slab_non_prim) == len(slab) * 4

        # Some randomized testing of cell vectors
        rng = np.random.default_rng()
        for spg_int in rng.integers(1, 230, 10):
            sg = SpaceGroup.from_int_number(spg_int)
            if sg.crystal_system == "hexagonal" or (
                sg.crystal_system == "trigonal"
                and (
                    sg.hexagonal
                    or sg.int_number
                    in (
                        143,
                        144,
                        145,
                        147,
                        149,
                        150,
                        151,
                        152,
                        153,
                        154,
                        156,
                        157,
                        158,
                        159,
                        162,
                        163,
                        164,
                        165,
                    )
                )
            ):
                lattice = Lattice.hexagonal(5, 10)
            else:
                # Cubic lattice is compatible with all other space groups.
                lattice = Lattice.cubic(5)
            struct = Structure.from_spacegroup(spg_int, lattice, ["H"], [[0, 0, 0]])
            miller = (0, 0, 0)
            while miller == (0, 0, 0):
                miller = tuple(rng.integers(0, 6, size=3, endpoint=True))
            gen = SlabGenerator(struct, miller, 10, 10)
            a_vec, b_vec, _c_vec = gen.oriented_unit_cell.lattice.matrix
            assert np.dot(a_vec, gen._normal) == approx(0)
            assert np.dot(b_vec, gen._normal) == approx(0)

    def test_get_slab_oriented_unit_cell_obeys_constrained_lattice(self):
        bulk = Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(4.194),
            ["Mg", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        slab = SlabGenerator(
            initial_structure=bulk,
            miller_index=(1, 1, 1),
            min_slab_size=10,
            min_vacuum_size=20,
            primitive=True,
            lll_reduce=False,
            center_slab=False,
            max_normal_search=None,
        ).get_slab(shift=0)

        slab_l = slab.lattice
        ouc_l = slab.oriented_unit_cell.lattice
        assert ouc_l.a == approx(slab_l.a)
        assert ouc_l.b == approx(slab_l.b)
        assert ouc_l.alpha == approx(slab_l.alpha)
        assert ouc_l.beta == approx(slab_l.beta)
        assert ouc_l.gamma == approx(slab_l.gamma)

    def test_normal_search(self):
        fcc = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        for miller in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1)]:
            gen = SlabGenerator(fcc, miller, 10, 10)
            gen_normal = SlabGenerator(fcc, miller, 10, 10, max_normal_search=max(miller))
            slab = gen_normal.get_slab()
            assert slab.lattice.alpha == 90
            assert slab.lattice.beta == 90
            assert len(gen_normal.oriented_unit_cell) >= len(gen.oriented_unit_cell)

        graphite = self.get_structure("Graphite")
        for miller in [(1, 0, 0), (1, 1, 0), (0, 0, 1), (2, 1, 1)]:
            gen = SlabGenerator(graphite, miller, 10, 10)
            gen_normal = SlabGenerator(graphite, miller, 10, 10, max_normal_search=max(miller))
            assert len(gen_normal.oriented_unit_cell) >= len(gen.oriented_unit_cell)

        sc = Structure(
            Lattice.hexagonal(3.32, 5.15),
            ["Sc", "Sc"],
            [[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]],
        )
        gen = SlabGenerator(sc, (1, 1, 1), 10, 10, max_normal_search=1)
        assert gen.oriented_unit_cell.lattice.angles[1] == approx(90)

    def test_get_slabs(self):
        gen = SlabGenerator(self.get_structure("CsCl"), [0, 0, 1], 10, 10)

        # Test orthogonality of some internal variables.
        a_len, b, _c = gen.oriented_unit_cell.lattice.matrix
        assert np.dot(a_len, gen._normal) == approx(0)
        assert np.dot(b, gen._normal) == approx(0)

        assert len(gen.get_slabs()) == 1

        struct = self.get_structure("LiFePO4")
        gen = SlabGenerator(struct, [0, 0, 1], 10, 10)
        assert len(gen.get_slabs()) == 5

        assert len(gen.get_slabs(bonds={("P", "O"): 3})) == 2

        # There are no slabs in LFP that does not break either P-O or Fe-O
        # bonds for a miller index of [0, 0, 1].
        assert len(gen.get_slabs(bonds={("P", "O"): 3, ("Fe", "O"): 3})) == 0

        # If we allow some broken bonds, there are a few slabs.
        assert len(gen.get_slabs(bonds={("P", "O"): 3, ("Fe", "O"): 3}, max_broken_bonds=2)) == 2

        # At this threshold, only the origin and center Li results in
        # clustering. All other sites are non-clustered. So the of
        # slabs is of sites in LiFePO4 unit cell - 2 + 1.
        assert len(gen.get_slabs(tol=1e-4, ftol=1e-4)) == 15

        LiCoO2 = Structure.from_file(f"{TEST_FILES_DIR}/surfaces/icsd_LiCoO2.cif", primitive=False)
        gen = SlabGenerator(LiCoO2, [0, 0, 1], 10, 10)
        lco = gen.get_slabs(bonds={("Co", "O"): 3})
        assert len(lco) == 1
        a_len, b, _c = gen.oriented_unit_cell.lattice.matrix
        assert np.dot(a_len, gen._normal) == approx(0)
        assert np.dot(b, gen._normal) == approx(0)

        scc = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3), ["Fe"], [[0, 0, 0]])
        gen = SlabGenerator(scc, [0, 0, 1], 10, 10)
        slabs = gen.get_slabs()
        assert len(slabs) == 1
        gen = SlabGenerator(scc, [1, 1, 1], 10, 10, max_normal_search=1)
        slabs = gen.get_slabs()
        assert len(slabs) == 1

        # Test whether using units of hkl planes instead of Angstroms for
        # min_slab_size and min_vac_size will give us the same number of atoms
        n_atoms = []
        for a_len in [1, 1.4, 2.5, 3.6]:
            struct = Structure.from_spacegroup("Im-3m", Lattice.cubic(a_len), ["Fe"], [[0, 0, 0]])
            slab_gen = SlabGenerator(struct, (1, 1, 1), 10, 10, in_unit_planes=True, max_normal_search=2)
            n_atoms.append(len(slab_gen.get_slab()))
        # Check if the number of atoms in all slabs is the same
        for n_a in n_atoms:
            assert n_atoms[0] == n_a

    def test_triclinic_TeI(self):
        # Test case for a triclinic structure of TeI. Only these three
        # Miller indices are used because it is easier to identify which
        # atoms should be in a surface together. The closeness of the sites
        # in other Miller indices can cause some ambiguity when choosing a
        # higher tolerance.
        n_slabs = {(0, 0, 1): 5, (0, 1, 0): 3, (1, 0, 0): 7}
        TeI = Structure.from_file(f"{TEST_FILES_DIR}/surfaces/icsd_TeI.cif", primitive=False)
        for k, v in n_slabs.items():
            triclinic_TeI = SlabGenerator(TeI, k, 10, 10)
            TeI_slabs = triclinic_TeI.get_slabs()
            assert v == len(TeI_slabs)

    def test_get_orthogonal_c_slab(self):
        TeI = Structure.from_file(f"{TEST_FILES_DIR}/surfaces/icsd_TeI.cif", primitive=False)
        triclinic_TeI = SlabGenerator(TeI, (0, 0, 1), 10, 10)
        TeI_slabs = triclinic_TeI.get_slabs()
        slab = TeI_slabs[0]
        norm_slab = slab.get_orthogonal_c_slab()
        assert norm_slab.lattice.angles[0] == approx(90)
        assert norm_slab.lattice.angles[1] == approx(90)

    def test_get_orthogonal_c_slab_site_props(self):
        TeI = Structure.from_file(f"{TEST_FILES_DIR}/surfaces/icsd_TeI.cif", primitive=False)
        triclinic_TeI = SlabGenerator(TeI, (0, 0, 1), 10, 10)
        TeI_slabs = triclinic_TeI.get_slabs()
        slab = TeI_slabs[0]
        # Add site property to slab
        selective_dynamics = [[True, True, True] for _ in slab]
        new_sp = slab.site_properties
        new_sp["selective_dynamics"] = selective_dynamics
        slab_with_site_props = slab.copy(site_properties=new_sp)

        # Get orthogonal slab
        norm_slab = slab_with_site_props.get_orthogonal_c_slab()

        # Check if site properties is consistent (or kept)
        assert slab_with_site_props.site_properties == norm_slab.site_properties

    def test_get_tasker2_slabs(self):
        # The uneven distribution of ions on the (111) facets of Halite type
        # slabs are typical examples of Tasker 3 structures. We will test
        # this algo to generate a Tasker 2 structure instead
        slab_gen = SlabGenerator(self.MgO, (1, 1, 1), 10, 10, max_normal_search=1)
        # We generate the Tasker 3 structure first
        slab = slab_gen.get_slabs()[0]
        assert not slab.is_symmetric()
        assert slab.is_polar()
        # Now to generate the Tasker 2 structure, we must
        # ensure there are enough ions on top to move around
        slab.make_supercell([2, 1, 1])
        slabs = slab.get_tasker2_slabs()
        # Check if our Tasker 2 slab is nonpolar and symmetric
        for slab in slabs:
            assert slab.is_symmetric()
            assert not slab.is_polar()

    def test_non_stoichiometric_symmetrized_slab(self):
        # For the (111) halite slab, sometimes a non-stoichiometric
        # system is preferred over the stoichiometric Tasker 2.
        slab_gen = SlabGenerator(self.MgO, (1, 1, 1), 10, 10, max_normal_search=1)
        slabs = slab_gen.get_slabs(symmetrize=True)

        # We should end up with two terminations, one with
        # an Mg rich surface and another O rich surface
        assert len(slabs) == 2
        for slab in slabs:
            assert slab.is_symmetric()

        # For a low symmetry primitive_elemental system such as
        # R-3m, there should be some non-symmetric slabs
        # without using non-stoichiometric_symmetrized_slab
        slabs = generate_all_slabs(self.Dy, 1, 30, 30, center_slab=True, symmetrize=True)
        for s in slabs:
            assert s.is_symmetric()
            assert len(s) > len(self.Dy)

    def test_move_to_other_side(self):
        # Tests to see if sites are added to opposite side
        struct = self.get_structure("LiFePO4")
        slab_gen = SlabGenerator(struct, (0, 0, 1), 10, 10, center_slab=True)
        slab = slab_gen.get_slab()
        surface_sites = slab.get_surface_sites()

        # check if top sites are moved to the bottom
        top_index = [ss[1] for ss in surface_sites["top"]]
        slab = slab_gen.move_to_other_side(slab, top_index)
        all_bottom = [slab[i].frac_coords[2] < slab.center_of_mass[2] for i in top_index]
        assert all(all_bottom)

        # check if bottom sites are moved to the top
        bottom_index = [ss[1] for ss in surface_sites["bottom"]]
        slab = slab_gen.move_to_other_side(slab, bottom_index)
        all_top = [slab[i].frac_coords[2] > slab.center_of_mass[2] for i in bottom_index]
        assert all(all_top)

    def test_bonds_broken(self):
        # Querying the Materials Project database for Si
        struct = self.get_structure("Si")

        # Conventional unit cell is supplied to ensure miller indices
        # correspond to usual crystallographic definitions
        conv_bulk = SpacegroupAnalyzer(struct).get_conventional_standard_structure()
        slab_gen = SlabGenerator(conv_bulk, [1, 1, 1], 10, 10, center_slab=True)

        # Setting a generous estimate for max_broken_bonds
        # so that all terminations are generated. These slabs
        # are ordered by ascending number of bonds broken
        # which is assigned to Slab.energy
        slabs = slab_gen.get_slabs(bonds={("Si", "Si"): 2.40}, max_broken_bonds=30)

        # Looking at the two slabs generated in VESTA, we
        # expect 2 and 6 bonds broken so we check for this.
        # Number of broken bonds are floats due to primitive
        # flag check and subsequent transformation of slabs.
        assert slabs[0].energy == approx(2.0)
        assert slabs[1].energy == approx(6.0)

    def test_disordered_slab(self):
        # Test that generation of a disordered Slab is possible
        comp = Composition({"Na": 0.5, "K": 0.5})
        struct = Structure([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [comp], np.zeros((1, 3), dtype=np.float64))
        slabs = SlabGenerator(struct, (0, 0, 1), 1, 0).get_slabs()
        assert slabs[0].sites[0].species == comp

    def test_get_slab_lll_mapping_failure(self, monkeypatch):
        slab_gen = SlabGenerator(self.MgO, (1, 1, 1), 10, 10, lll_reduce=True)
        monkeypatch.setattr(Lattice, "find_mapping", lambda *args, **kwargs: None)
        with pytest.raises(RuntimeError, match="LLL reduction has failed"):
            slab_gen.get_slab()

    def test_gen_possible_terminations_empty(self):
        slab_gen = SlabGenerator(self.MgO, (1, 0, 0), 10, 10)
        slab_gen.oriented_unit_cell = Structure(slab_gen.oriented_unit_cell.lattice, [], [])
        assert slab_gen.gen_possible_terminations() == []

    def test_get_slabs_no_filter(self):
        slab_gen = SlabGenerator(self.MgO, (1, 1, 1), 10, 10, max_normal_search=1)
        filtered = slab_gen.get_slabs()
        unfiltered = slab_gen.get_slabs(filter_out_sym_slabs=False)
        assert len(filtered) == 1
        assert len(unfiltered) == len(slab_gen.gen_possible_terminations()) == 6
        # Unfiltered Slabs are all equivalent to the single filtered one
        matcher = StructureMatcher(ltol=0.1, stol=0.1, primitive_cell=False, scale=False)
        assert len(matcher.group_structures(unfiltered)) == 1

    def test_move_to_other_side_in_unit_planes(self):
        slab_gen = SlabGenerator(self.MgO, (1, 0, 0), 3, 3, in_unit_planes=True, center_slab=True)
        slab = slab_gen.get_slab()
        z_coords = slab.frac_coords[:, 2]
        layers = np.unique(np.round(z_coords, 6))
        top_idx = int(np.argmax(z_coords))
        moved = slab_gen.move_to_other_side(slab, [top_idx])
        # The top site moves to the next (empty) layer below the bottom surface
        assert moved[top_idx].frac_coords[2] == approx(layers[0] - (layers[1] - layers[0]), abs=1e-5)
        others = [idx for idx in range(len(slab)) if idx != top_idx]
        assert_allclose(moved.frac_coords[others], slab.frac_coords[others])

    def test_repair_broken_bonds_reversed_pair(self):
        # The reference element is chosen by coordination, so the order of the pair should not matter
        slab_gen = SlabGenerator(self.get_structure("LiFePO4"), (0, 0, 1), 10, 10, center_slab=True)
        slab = slab_gen.get_slab()
        repaired = slab_gen.repair_broken_bonds(slab.copy(), {("P", "O"): 3})
        repaired_rev = slab_gen.repair_broken_bonds(slab.copy(), {("O", "P"): 3})
        assert not np.allclose(repaired.frac_coords, slab.frac_coords)
        assert_allclose(repaired_rev.frac_coords, repaired.frac_coords)


class TestReconstructionGenerator(MatSciTest):
    def setup_method(self):
        lattice = Lattice.cubic(3.51)
        species = ["Ni"]
        coords = [[0, 0, 0]]
        self.Ni = Structure.from_spacegroup("Fm-3m", lattice, species, coords)
        lattice = Lattice.cubic(2.819000)
        species = ["Fe"]
        coords = [[0, 0, 0]]
        self.Fe = Structure.from_spacegroup("Im-3m", lattice, species, coords)
        self.Si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.430500), ["Si"], [(0, 0, 0.5)])

        with open(f"{PMG_CORE_DIR}/reconstructions_archive.json", "rb") as data_file:
            self.rec_archive = orjson.loads(data_file.read())

    def test_build_slab(self):
        # First lets test a reconstruction where we only remove atoms
        recon = ReconstructionGenerator(self.Ni, 10, 10, "fcc_110_missing_row_1x2")
        slab = recon.get_unreconstructed_slabs()[0]
        recon_slab = recon.build_slabs()[0]
        assert recon_slab.reconstruction
        assert len(slab) == len(recon_slab) + 2
        assert recon_slab.is_symmetric()

        # Test if the ouc corresponds to the reconstructed slab
        recon_ouc = recon_slab.oriented_unit_cell
        ouc = slab.oriented_unit_cell
        assert ouc.lattice.b * 2 == recon_ouc.lattice.b
        assert len(ouc) * 2 == len(recon_ouc)

        # Test a reconstruction where we simply add atoms
        recon = ReconstructionGenerator(self.Ni, 10, 10, "fcc_111_adatom_t_1x1")
        slab = recon.get_unreconstructed_slabs()[0]
        recon_slab = recon.build_slabs()[0]
        assert len(slab) == len(recon_slab) - 2
        assert recon_slab.is_symmetric()

        # If a slab references another slab, make sure it is properly generated
        recon = ReconstructionGenerator(self.Ni, 10, 10, "fcc_111_adatom_ft_1x1")
        slab = recon.build_slabs()[0]
        assert slab.is_symmetric

        # Test a reconstruction where it works on a specific termination (Fd-3m (111))
        recon = ReconstructionGenerator(self.Si, 10, 10, "diamond_111_1x2")
        slab = recon.get_unreconstructed_slabs()[0]
        recon_slab = recon.build_slabs()[0]
        assert len(slab) == len(recon_slab) - 8
        assert recon_slab.is_symmetric()

        # Test a reconstruction where terminations give
        # different reconstructions with a non-primitive_elemental system

    def test_get_d(self):
        # Ensure that regardless of the size of the vacuum or slab
        # layer, the spacing between atomic layers should be the same

        recon = ReconstructionGenerator(self.Si, 10, 10, "diamond_100_2x1")

        recon2 = ReconstructionGenerator(self.Si, 20, 10, "diamond_100_2x1")
        s1 = recon.get_unreconstructed_slabs()[0]
        s2 = recon2.get_unreconstructed_slabs()[0]
        assert get_d(s1) == approx(get_d(s2))

    def test_get_d_single_layer(self):
        lattice = Lattice.tetragonal(2.5, 10)
        ouc = Structure(Lattice.cubic(2.5), ["Fe"], [[0, 0, 0]])
        slab = Slab(lattice, ["Fe", "Fe"], [[0, 0, 0.5], [0.5, 0.5, 0.5]], (0, 0, 1), ouc, 0, np.eye(3))
        with pytest.raises(RuntimeError, match="Cannot identify any layer"):
            get_d(slab)

    def test_invalid_reconstruction_name(self):
        with pytest.raises(KeyError, match="does not exist in the archive"):
            ReconstructionGenerator(self.Ni, 10, 10, "fcc_999_nonexistent")

    def test_base_reconstruction_points_to_remove(self, monkeypatch):
        # A derived reconstruction only specifying points_to_remove replaces both
        # the points_to_remove and points_to_add of its base reconstruction
        points_to_remove = [[0.25, 0.25, 0]]
        monkeypatch.setitem(
            RECONSTRUCTIONS_ARCHIVE,
            "diamond_100_derived",
            {"base_reconstruction": "diamond_100_2x1", "points_to_remove": points_to_remove},
        )
        base = RECONSTRUCTIONS_ARCHIVE["diamond_100_2x1"]
        assert "points_to_add" in base
        assert "points_to_remove" in base

        recon = ReconstructionGenerator(self.Si, 10, 10, "diamond_100_derived")
        assert recon.reconstruction_json["points_to_remove"] == points_to_remove
        assert "points_to_add" not in recon.reconstruction_json
        assert recon.reconstruction_json["miller_index"] == base["miller_index"]
        assert recon.trans_matrix == base["transformation_matrix"]
        # The archive entry of the base reconstruction is left untouched
        assert "points_to_add" in RECONSTRUCTIONS_ARCHIVE["diamond_100_2x1"]

    def test_generate_all_slabs_skips_high_index_reconstructions(self):
        # All Fm-3m reconstructions in the archive have a max Miller index of 1
        assert generate_all_slabs(self.Ni, 0, 10, 10, include_reconstructions=True) == []

    @pytest.mark.xfail(reason="This test relies on neighbor orders and is hard coded. Disable temporarily")
    def test_previous_reconstructions(self):
        # Test to see if we generated all reconstruction types correctly and nothing changes

        match = StructureMatcher()
        for idx in self.rec_archive:
            if "base_reconstruction" in self.rec_archive[idx]:
                arch = self.rec_archive[self.rec_archive[idx]["base_reconstruction"]]
                sg = arch["spacegroup"]["symbol"]
            else:
                sg = self.rec_archive[idx]["spacegroup"]["symbol"]
            if sg == "Fm-3m":
                rec = ReconstructionGenerator(self.Ni, 20, 20, idx)
                el = self.Ni[0].species_string
            elif sg == "Im-3m":
                rec = ReconstructionGenerator(self.Fe, 20, 20, idx)
                el = self.Fe[0].species_string
            elif sg == "Fd-3m":
                rec = ReconstructionGenerator(self.Si, 20, 20, idx)
                el = self.Si[0].species_string

            slabs = rec.build_slabs()
            struct = Structure.from_file(f"{TEST_FILES_DIR}/surfaces/reconstructions/{el}_{idx}.cif")
            assert any(len(match.group_structures([struct, slab])) == 1 for slab in slabs)


class TestMillerIndexFinder(MatSciTest):
    def setup_method(self):
        self.cscl = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.2), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.Fe = Structure.from_spacegroup("Im-3m", Lattice.cubic(2.82), ["Fe"], [[0, 0, 0]])
        mg_lattice = Lattice.from_parameters(3.2, 3.2, 5.13, 90, 90, 120)
        self.Mg = Structure(mg_lattice, ["Mg", "Mg"], [[1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]])
        self.lifepo4 = self.get_structure("LiFePO4")
        self.tei = Structure.from_file(f"{TEST_FILES_DIR}/surfaces/icsd_TeI.cif", primitive=False)
        self.LiCoO2 = Structure.from_file(f"{TEST_FILES_DIR}/surfaces/icsd_LiCoO2.cif", primitive=False)

        self.p1 = Structure(
            Lattice.from_parameters(3, 4, 5, 31, 43, 50),
            ["H", "He"],
            [[0, 0, 0], [0.1, 0.2, 0.3]],
        )
        self.graphite = self.get_structure("Graphite")
        self.trig_bi = Structure(
            Lattice.from_parameters(3, 3, 10, 90, 90, 120),
            ["Bi", "Bi", "Bi", "Bi", "Bi", "Bi"],
            [
                [0.3333, 0.6666, 0.39945113],
                [0.0000, 0.0000, 0.26721554],
                [0.0000, 0.0000, 0.73278446],
                [0.6666, 0.3333, 0.60054887],
                [0.6666, 0.3333, 0.06611779],
                [0.3333, 0.6666, 0.93388221],
            ],
        )

    def test_get_symmetrically_distinct_miller_indices(self):
        # Tests to see if the function obtains the known number of unique slabs

        indices = get_symmetrically_distinct_miller_indices(self.cscl, 1)
        # Check that the positive versions were returned
        # and that an earlier index is preferred ((100) over (010)/(001))
        # and that positive indices are preferred ((111) over (1-1-1) etc.)
        # and that the indices are sorted by absolute sum
        cubic_exp = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
        assert indices == cubic_exp
        indices = get_symmetrically_distinct_miller_indices(self.cscl, 2)
        assert len(indices) == 6

        # For a P... space group, all modes returning conventional indices should
        # return the same indices (as long as the input cell is standard)
        p_conv = get_symmetrically_distinct_miller_indices(self.cscl, 2, cell="conventional")
        p_conv_np = get_symmetrically_distinct_miller_indices(self.cscl, 2, cell="conv_np")
        p_prim = get_symmetrically_distinct_miller_indices(self.cscl, 2, cell="primitive")
        p_prim2conv = get_symmetrically_distinct_miller_indices(self.cscl, 2, cell="prim_2conv")
        assert p_conv == p_conv_np == p_prim == p_prim2conv

        assert len(get_symmetrically_distinct_miller_indices(self.lifepo4, 1)) == 7

        # The TeI P-1 structure should have 13 unique millers (only inversion
        # symmetry eliminates pairs)
        indices = get_symmetrically_distinct_miller_indices(self.tei, 1)
        assert len(indices) == 13

        # P1 and P-1 should have the same # of miller indices since surfaces
        # always have inversion symmetry.
        indices = get_symmetrically_distinct_miller_indices(self.p1, 1)
        assert len(indices) == 13

        indices = get_symmetrically_distinct_miller_indices(self.graphite, 2)
        assert len(indices) == 12

        # Now try a trigonal system and hkil.
        # Note that a trigonal cell must use cell="conventional" to restore the old behaviour
        # of using the primitive symmetry.
        indices = get_symmetrically_distinct_miller_indices(self.trig_bi, 2, return_hkil=True, cell="conventional")
        assert len(indices) == 17
        assert all(len(hkl) == 4 for hkl in indices)
        assert all(i == -h - k for h, k, i, _ in indices)

        # Test to see if the output with max_index i is a subset of the output with max_index i+1
        # Also check that the order of the smaller index values remains unchanged
        prev: list[tuple[int, int, int]] = []
        for idx in range(1, 5):
            new = get_symmetrically_distinct_miller_indices(self.trig_bi, idx)
            assert prev == new[: len(prev)]
            prev = new

        # Using the conventional symmetry ops will lead to less distinct indices (as there is centering):
        # (the input cell is conventional, so input==conv_np)
        indices_i = get_symmetrically_distinct_miller_indices(self.trig_bi, 2, cell="input")
        assert len(indices_i) == 12
        indices_np = get_symmetrically_distinct_miller_indices(self.trig_bi, 2, cell="conv_np")
        assert indices_i == indices_np

        # With primitive symmetry, the count *can* be different (depends on the lattice)
        indices_p = get_symmetrically_distinct_miller_indices(self.trig_bi, 2, cell="primitive")
        assert len(indices_p) == 13
        # Count has to be the same with converted indices, but the indices are different

        # Test a non-conventional cell: Supercell of CsCl
        super_cscl = self.cscl.make_supercell((2, 1, 1), in_place=False)
        # CsCl is cubic, but this supercell is only tetragonal -> "input" and "conventional" differ
        # Also, it is expanded in a, not in c - so a is the unique axis!
        tetra_nonconv_exp = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
        assert get_symmetrically_distinct_miller_indices(super_cscl, 1) == tetra_nonconv_exp
        assert get_symmetrically_distinct_miller_indices(super_cscl, 1, cell="conventional") == cubic_exp

        # With a conventional tetragonal cell, return the planes for a c-unique cell
        tetra_conv_exp = [(1, 0, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
        tetra = Structure(Lattice.tetragonal(2, 3), ["H"], [[0, 0, 0]])
        assert get_symmetrically_distinct_miller_indices(tetra, 1) == tetra_conv_exp

    def test_hkl_uvw_transformation(self):
        """Test hkl/uvw_transformation for reasonable results."""
        # hkls should be reduced
        matrix = np.eye(3)
        assert hkl_transformation(matrix, (2, 2, 2)) == (1, 1, 1)
        # hkl_transformation should use the correct determinant
        matrix[0, 0] = 2 / 3
        assert hkl_transformation(matrix, (3, 1, 0)) == (2, 1, 0)

        # Test a realistic scenario
        struct = Structure.from_spacegroup(
            167, Lattice.hexagonal(5, 7), ("Ca", "C", "O"), ((0, 0, 0), (0, 0, 0.25), (0.25, 0, 0.25))
        )

        matrix = SpacegroupAnalyzer(struct).get_conventional_to_primitive_transformation_matrix()

        # Both have gcd 3: (6, 3, 3) and [3, -3, -3]
        assert hkl_transformation(matrix, (1, 0, 4)) == (2, 1, 1)
        assert uvw_transformation(matrix, (4, 2, -1)) == (1, -1, -1)

    def test_get_symmetrically_equivalent_miller_indices(self):
        # Tests to see if the function obtains all equivalent hkl for cubic (100)
        indices001 = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (0, 0, -1),
            (0, -1, 0),
            (-1, 0, 0),
        ]
        indices = get_symmetrically_equivalent_miller_indices(self.cscl, (1, 0, 0))
        assert all(hkl in indices for hkl in indices001)

        # Tests to see if it captures expanded Miller indices in the family e.g. (001) == (002)
        hcp_indices_100 = get_symmetrically_equivalent_miller_indices(self.Mg, (1, 0, 0))
        hcp_indices_200 = get_symmetrically_equivalent_miller_indices(self.Mg, (2, 0, 0))
        assert len(hcp_indices_100) * 2 == len(hcp_indices_200)
        assert len(hcp_indices_100) == 6
        assert all(len(hkl) == 4 for hkl in hcp_indices_100)

    def test_generate_all_slabs(self):
        slabs = generate_all_slabs(self.cscl, 1, 10, 10)
        # Only three possible slabs, one each in (100), (110) and (111).
        assert len(slabs) == 3

        # make sure it generates reconstructions
        slabs = generate_all_slabs(self.Fe, 1, 10, 10, include_reconstructions=True)

        # Four possible slabs, (100), (110), (111) and the zigzag (100).
        assert len(slabs) == 4

        slabs = generate_all_slabs(self.cscl, 1, 10, 10, bonds={("Cs", "Cl"): 4})
        # No slabs if we don't allow broken Cs-Cl
        assert len(slabs) == 0

        slabs = generate_all_slabs(self.cscl, 1, 10, 10, bonds={("Cs", "Cl"): 4}, max_broken_bonds=100)
        assert len(slabs) == 3

        slabs2 = generate_all_slabs(self.lifepo4, 1, 10, 10, bonds={("P", "O"): 3, ("Fe", "O"): 3})
        assert len(slabs2) == 0

        # There should be only one possible stable surfaces, all of which are
        # in the (001) oriented unit cell
        slabs3 = generate_all_slabs(self.LiCoO2, 1, 10, 10, bonds={("Co", "O"): 3})
        assert len(slabs3) == 1
        mill = (0, 0, 1)
        for s in slabs3:
            assert s.miller_index == mill

        slabs1 = generate_all_slabs(self.lifepo4, 1, 10, 10, tol=0.1, bonds={("P", "O"): 3})
        assert len(slabs1) == 4

        # Now we test this out for repair_broken_bonds()
        slabs1_repair = generate_all_slabs(self.lifepo4, 1, 10, 10, tol=0.1, bonds={("P", "O"): 3}, repair=True)
        assert len(slabs1_repair) > len(slabs1)

        # Lets see if there are no broken PO4 polyhedrons
        miller_list = get_symmetrically_distinct_miller_indices(self.lifepo4, 1)
        all_miller_list = []
        for slab in slabs1_repair:
            hkl = tuple(slab.miller_index)
            if hkl not in all_miller_list:
                all_miller_list.append(hkl)
            broken = []
            for site in slab:
                if site.species_string == "P":
                    neighbors = slab.get_neighbors(site, 3)
                    cn = 0
                    for nn in neighbors:
                        cn += 1 if nn[0].species_string == "O" else 0
                    broken.append(cn != 4)
            assert not any(broken)

        # check if we were able to produce at least one
        # termination for each distinct Miller _index
        assert len(miller_list) == len(all_miller_list)

    def test_computed_scale_factor(self):
        """Test that the scale factor of a Slab matches the lattice transformation."""
        # With no reorientation, it should be the exact transformation
        struct_lattice = Lattice.hexagonal(5, 17)
        # Approximately calcite (spacegroup R-3c)
        struct = Structure.from_spacegroup(
            167, struct_lattice, ("Ca", "C", "O"), ((0, 0, 0), (0, 0, 0.25), (0.25, 0, 0.25))
        )

        # Basic SlabGenerator
        slabgen = SlabGenerator(struct, (1, 0, 4), 1, 1, in_unit_planes=True, primitive=False, reorient_lattice=False)
        # Should work with and without bonds and with all settings
        for slab in slabgen.get_slabs(bonds={("C", "O"): 3}):
            # Slab matrix should result from scale_factor and source lattice
            computed_matrix = slab.scale_factor @ struct_lattice.matrix
            assert np.allclose(slab.lattice.matrix, computed_matrix)
            # Also holds for the OUC, as long as primitivization is disabled
            # Note that the OUC is shorter in c (only slab-part, no vacuum)
            computed_matrix[2] /= 2
            assert np.allclose(slab.oriented_unit_cell.lattice.matrix, computed_matrix)
        # Enable settings that modify the matrices
        slabgen.primitive = True
        slabgen.reorient_lattice = True
        slabgen.lll_reduce = True
        for slab in slabgen.get_slabs():
            # slab matrix is rotated - only determinant will be correct
            assert np.isclose(np.linalg.det(slab.scale_factor), slab.lattice.volume / struct.lattice.volume)

    def test_smaller_than_ouc_slab(self):
        """Test for potential Slabs smaller than the OUC."""
        # Approximately calcite (spacegroup R-3c)
        struct = Structure.from_spacegroup(
            167, Lattice.hexagonal(5, 17), ("Ca", "C", "O"), ((0, 0, 0), (0, 0, 0.25), (0.25, 0, 0.25))
        )
        # Most stable surface is (104) - represented as [42-1] x [010] (or equivalent directions)
        slabgen = SlabGenerator(struct, (1, 0, 4), 1, 1, in_unit_planes=True, primitive=False)

        # Non-primitive Slab will be [010] x [-401] (essentially integer scale factor)
        # Note the long b-side and the 67.6° gamma angle
        # c is [002] because it includes the vacuum
        np_slab = slabgen.get_slab(0.125)
        assert np.allclose(np_slab.scale_factor, [[0, 1, 0], [-4, 0, 1], [0, 0, 2]])
        # Check expected lattice params a/b/gamma as well for safety
        expected = (5.0, 26.248809496813376, 67.60624150257428)
        lat = np_slab.lattice
        assert np.allclose((lat.a, lat.b, lat.gamma), expected)

        # We actually want [010] x [42-1], which have a 90° gamma angle:
        # Due to the rhombohedral centering, the [42-1] vector can be shortened to a third of the full [42-1]
        # The primitive slab will diverge in the slab and OUC lattice because .get_primitive_structure struggles
        # with this primitivization (requires a "good" shift like 0.125, not a "bad" one like 0)
        # Thus allow_smaller_than_ouc must be enabled
        slabgen.primitive = True
        slabgen.allow_smaller_than_ouc = True
        p_slab = slabgen.get_slab(0.125)
        # We get [010] x [-4-21]/3 from this configuration (note that c also changes to be nearly orthogonal)
        assert np.allclose(p_slab.scale_factor, [[0, 1, 0], [-4 / 3, -2 / 3, 1 / 3], [4, 2, 1]])
        # Check expected lattice params a/b/gamma again
        expected = (5.0, 8.08977406634106, 90.0)
        lat = p_slab.lattice
        assert np.allclose((lat.a, lat.b, lat.gamma), expected)

    def test_miller_index_from_sites(self):
        """Test surface miller index convenience function."""
        # test on a cubic system
        cubic = Lattice.cubic(1)
        s1 = np.array([0.5, -1.5, 3])
        s2 = np.array([0.5, 3.0, -1.5])
        s3 = np.array([2.5, 1.5, -4.0])
        assert miller_index_from_sites(cubic, [s1, s2, s3]) == (2, 1, 1)

        # test casting from matrix to Lattice
        matrix = [
            [2.319, -4.01662582, 0.0],
            [2.319, 4.01662582, 0.0],
            [0.0, 0.0, 7.252],
        ]

        s1 = np.array([2.319, 1.33887527, 6.3455])
        s2 = np.array([1.1595, 0.66943764, 4.5325])
        s3 = np.array([1.1595, 0.66943764, 0.9065])
        hkl = miller_index_from_sites(matrix, [s1, s2, s3])
        assert hkl == (2, -1, 0)

    @pytest.mark.parametrize(
        ("struct_name", "miller_index", "system", "n_expected"),
        [
            ("cscl", (1, 0, 0), "cubic", 6),
            ("cscl", (1, 1, 1), "cubic", 8),
            ("LiCoO2", (1, 0, 4), "trigonal", 12),
        ],
    )
    def test_get_symmetrically_equivalent_miller_indices_given_system(
        self, struct_name, miller_index, system, n_expected
    ):
        struct = getattr(self, struct_name)
        # Supplying the crystal system skips the SpacegroupAnalyzer call but gives the same result
        given = get_symmetrically_equivalent_miller_indices(struct, miller_index, system=system)
        detected = get_symmetrically_equivalent_miller_indices(struct, miller_index)
        assert len(given) == n_expected
        assert set(given) == set(detected)

    def test_get_symmetrically_distinct_miller_indices_invalid_cell(self):
        with pytest.raises(ValueError, match="Unsupported cell type invalid"):
            get_symmetrically_distinct_miller_indices(self.cscl, 1, cell="invalid")

    def test_get_symmetrically_distinct_miller_indices_prim_2conv(self):
        # Primitive bcc indices up to 1 correspond to conventional {100}, {110}, {111} and {112}
        indices = get_symmetrically_distinct_miller_indices(self.Fe, 1, cell="prim_2conv")
        assert [tuple(int(i) for i in hkl) for hkl in indices] == [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 2)]
        # For a non-centered (P) lattice, prim_2conv is the same as primitive
        assert get_symmetrically_distinct_miller_indices(
            self.cscl, 1, cell="prim_2conv"
        ) == get_symmetrically_distinct_miller_indices(self.cscl, 1, cell="primitive")

    def test_get_symmetrically_distinct_miller_indices_hkil_non_hexagonal(self):
        with pytest.warns(UserWarning, match="hkil miller indices .* the cubic family"):
            indices = get_symmetrically_distinct_miller_indices(self.cscl, 1, return_hkil=True, cell="conventional")
        assert indices == [(1, 0, -1, 0), (1, 1, -2, 0), (1, 1, -2, 1)]

    @pytest.mark.parametrize(
        ("transf", "hkl", "expected"),
        [
            (np.eye(3), (-1, 1, 0), (1, -1, 0)),
            (-np.eye(3), (1, 2, 0), (1, 2, 0)),
            (np.eye(3), (0, -2, 1), (0, 2, -1)),
        ],
    )
    def test_hkl_transformation_positive_first_index(self, transf, hkl, expected):
        assert hkl_transformation(transf, hkl) == expected
