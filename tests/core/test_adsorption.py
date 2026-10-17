from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymatgen.core.adsorption import (
    AdsorbateSiteFinder,
    generate_all_slabs,
    get_mi_vec,
    get_rot,
    plot_slab,
    put_coord_inside,
    reorient_z,
)
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.testing import MatSciTest


class TestAdsorbateSiteFinder(MatSciTest):
    @classmethod
    def setup_class(cls):
        cls.structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.5), ["Ni"], [[0, 0, 0]])
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
        cls.MgO = Structure(lattice, species, frac_coords)

        slabs = generate_all_slabs(
            cls.structure,
            max_index=2,
            min_slab_size=6.0,
            min_vacuum_size=15.0,
            max_normal_search=1,
            center_slab=True,
        )
        cls.slab_dict = {"".join(str(i) for i in slab.miller_index): slab for slab in slabs}
        cls.asf_211 = AdsorbateSiteFinder(cls.slab_dict["211"])
        cls.asf_100 = AdsorbateSiteFinder(cls.slab_dict["100"])
        cls.asf_111 = AdsorbateSiteFinder(cls.slab_dict["111"])
        cls.asf_110 = AdsorbateSiteFinder(cls.slab_dict["110"])
        cls.asf_struct = AdsorbateSiteFinder(Structure.from_sites(cls.slab_dict["111"].sites))

    def test_init(self):
        AdsorbateSiteFinder(self.slab_dict["100"])
        AdsorbateSiteFinder(self.slab_dict["111"])

    def test_from_bulk_and_miller(self):
        # Standard site finding
        asf = AdsorbateSiteFinder.from_bulk_and_miller(self.structure, (1, 1, 1))
        sites = asf.find_adsorption_sites()
        assert len(sites["hollow"]) == 2
        assert len(sites["bridge"]) == 1
        assert len(sites["ontop"]) == 1
        assert len(sites["all"]) == 4
        asf = AdsorbateSiteFinder.from_bulk_and_miller(self.structure, (1, 0, 0))
        sites = asf.find_adsorption_sites()
        assert len(sites["all"]) == 3
        assert len(sites["bridge"]) == 2
        asf = AdsorbateSiteFinder.from_bulk_and_miller(self.structure, (1, 1, 0), undercoord_threshold=0.1)
        assert len(asf.surface_sites) == 1
        # Subsurface site finding
        asf = AdsorbateSiteFinder.from_bulk_and_miller(self.structure, (1, 1, 1))
        sites = asf.find_adsorption_sites(positions=["ontop", "subsurface", "bridge"])
        assert len(sites["all"]) == 5
        assert len(sites["subsurface"]) == 3

    def test_find_adsorption_sites(self):
        sites = self.asf_100.find_adsorption_sites()
        assert len(sites["all"]) == 3
        assert len(sites["hollow"]) == 0
        assert len(sites["bridge"]) == 2
        assert len(sites["ontop"]) == 1
        sites = self.asf_111.find_adsorption_sites()
        assert len(sites["all"]) == 4
        sites = self.asf_110.find_adsorption_sites()
        assert len(sites["all"]) == 4
        sites = self.asf_211.find_adsorption_sites()
        # Test on structure
        sites = self.asf_struct.find_adsorption_sites()

    def test_generate_adsorption_structures(self):
        atol = 1e-9  # required for comparing values ~0 using assert_allclose

        co = Molecule("CO", [[0, 0, 0], [0, 0, 1.23]])
        structures = self.asf_111.generate_adsorption_structures(co, repeat=[2, 2, 1])
        assert len(structures) == 4
        sites = self.asf_111.find_adsorption_sites()
        # Check repeat functionality
        assert sum(site.properties["surface_properties"] != "adsorbate" for site in structures[0]) == 4 * len(
            self.asf_111.slab
        )
        for n, structure in enumerate(structures):
            assert_allclose(structure[-2].coords, sites["all"][n], atol=atol)
        find_args = {"positions": ["hollow"]}
        structures_hollow = self.asf_111.generate_adsorption_structures(co, find_args=find_args)
        assert len(structures_hollow) == len(sites["hollow"])
        for structure in structures_hollow:
            assert in_coord_list(sites["hollow"], structure[-2].coords, 1e-4)
        # Check molecule not changed after rotation when added to surface
        co = Molecule("CO", [[1.0, -0.5, 3], [0.8, 0.46, 3.75]])
        structures = self.asf_211.generate_adsorption_structures(co)
        assert co == Molecule("CO", [[1.0, -0.5, 3], [0.8, 0.46, 3.75]])
        # Check translation
        sites = self.asf_211.find_adsorption_sites()
        ads_site_coords = sites["all"][0]
        c_site = structures[0][-2]
        assert str(c_site.specie) == "C"
        assert_allclose(c_site.coords, sites["all"][0], atol=atol)
        # Check no translation
        structures = self.asf_111.generate_adsorption_structures(co, translate=False)
        assert co == Molecule("CO", [[1.0, -0.5, 3], [0.8, 0.46, 3.75]])
        sites = self.asf_111.find_adsorption_sites()
        ads_site_coords = sites["all"][0]
        c_site = structures[0][-2]
        assert_allclose(c_site.coords, ads_site_coords + np.array([1.0, -0.5, 3]), atol=atol)

    def test_adsorb_both_surfaces(self):
        # Test out for monatomic adsorption
        oxi = Molecule("O", [[0, 0, 0]])
        ad_slabs = self.asf_100.adsorb_both_surfaces(oxi)
        ad_slabs_one = self.asf_100.generate_adsorption_structures(oxi)
        assert len(ad_slabs) == len(ad_slabs_one)
        for ad_slab in ad_slabs:
            sg = SpacegroupAnalyzer(ad_slab)
            sites = sorted(ad_slab, key=lambda site: site.frac_coords[2])
            assert sites[0].species_string == "O"
            assert sites[-1].species_string == "O"
            assert sg.is_laue()

        # Test out for molecular adsorption
        oh = Molecule(["O", "H"], [[0, 0, 0], [0, 0, 1]])
        ad_slabs = self.asf_100.adsorb_both_surfaces(oh)
        ad_slabs_one = self.asf_100.generate_adsorption_structures(oh)
        assert len(ad_slabs) == len(ad_slabs_one)
        for ad_slab in ad_slabs:
            sg = SpacegroupAnalyzer(ad_slab)
            sites = sorted(ad_slab, key=lambda site: site.frac_coords[2])
            assert sites[0].species_string in ["O", "H"]
            assert sites[-1].species_string in ["O", "H"]
            assert sg.is_laue()

    def test_generate_substitution_structures(self):
        # Test this for a low miller index halite structure
        slabs = generate_all_slabs(self.MgO, 1, 10, 10, center_slab=True, max_normal_search=1)
        for slab in slabs:
            site_finder = AdsorbateSiteFinder(slab)

            sub_structs = site_finder.generate_substitution_structures("Ni")
            # There should be 2 configs (sub O and sub
            # Mg) for (110) and (100), 1 for (111)
            if tuple(slab.miller_index) != (1, 1, 1):
                assert len(sub_structs) == 2
            else:
                assert len(sub_structs) == 1

            # Test out whether it can correctly dope both
            # sides. Avoid (111) because it is not symmetric
            if tuple(slab.miller_index) != (1, 1, 1):
                sub_structs = site_finder.generate_substitution_structures(
                    "Ni", sub_both_sides=True, target_species=["Mg"]
                )
                # Test if default parameters dope the surface site
                for idx, site in enumerate(sub_structs[0]):
                    if site_finder.slab[idx].surface_properties == "surface" and site.species_string == "Mg":
                        assert sub_structs[0][idx].surface_properties == "substitute"

                assert sub_structs[0].is_symmetric()
                # Correctly dope the target species
                assert sub_structs[0].composition.as_dict()["Mg"] == slab.composition.as_dict()["Mg"] - 2
                # There should be one config (sub Mg)
                assert len(sub_structs) == 1

    def test_functions(self):
        slab = self.slab_dict["111"]
        get_rot(slab)
        reorient_z(slab)

    def test_init_mi_vec(self):
        mi_vec = [0.0, 0.0, 1.0]
        asf = AdsorbateSiteFinder(self.slab_dict["100"], mi_vec=mi_vec)
        assert asf.mvec is mi_vec
        assert_allclose(asf.mvec, get_mi_vec(self.slab_dict["100"]), atol=1e-12)

    def test_init_selective_dynamics(self):
        asf = AdsorbateSiteFinder(self.slab_dict["100"], selective_dynamics=True)
        surf_props = asf.slab.site_properties["surface_properties"]
        sel_dyn = asf.slab.site_properties["selective_dynamics"]
        assert len(sel_dyn) == len(asf.slab)
        for prop, flags in zip(surf_props, sel_dyn, strict=True):
            assert flags == ([False] * 3 if prop == "subsurface" else [True] * 3)
        assert sum(flags == [True] * 3 for flags in sel_dyn) == len(asf.surface_sites) == 1

        # Adsorbates added to a selective-dynamics slab are fully relaxable
        oxi = Molecule("O", [[0, 0, 0]])
        ad_slab = asf.add_adsorbate(oxi, [0, 0, 20])
        assert len(ad_slab) == len(asf.slab) + 1
        assert ad_slab[-1].properties["selective_dynamics"] == [True, True, True]
        assert ad_slab[-1].properties["surface_properties"] == "adsorbate"
        assert_allclose(ad_slab[-1].coords, [0, 0, 20], atol=1e-9)

    def test_assign_selective_dynamics(self):
        slab = self.asf_111.slab
        new_slab = AdsorbateSiteFinder.assign_selective_dynamics(slab)
        assert "selective_dynamics" not in slab.site_properties
        expected = [
            [False] * 3 if prop == "subsurface" else [True] * 3 for prop in slab.site_properties["surface_properties"]
        ]
        assert new_slab.site_properties["selective_dynamics"] == expected

    def test_from_bulk_and_miller_invalid_index(self):
        # Non-reduced Miller indices are never generated by generate_all_slabs
        with pytest.raises(ValueError, match="Miller index not in slab dict"):
            AdsorbateSiteFinder.from_bulk_and_miller(self.structure, (2, 2, 2), max_normal_search=1)

    @pytest.mark.parametrize("cartesian", [True, False])
    def test_ensemble_center(self, cartesian):
        sites = self.slab_dict["100"].sites
        indices = [0, 1]
        attr = "coords" if cartesian else "frac_coords"
        assert_allclose(
            AdsorbateSiteFinder.ensemble_center(sites, indices, cartesian=cartesian),
            np.mean([getattr(sites[idx], attr) for idx in indices], axis=0),
        )

    def test_subsurface_sites(self):
        asf = self.asf_100
        subsurface = asf.subsurface_sites()
        assert len(subsurface) + len(asf.surface_sites) == len(asf.slab)
        assert all(site.properties["surface_properties"] == "subsurface" for site in subsurface)

    def test_generate_substitution_structures_bottom_surface(self):
        # Pre-assigned surface properties are kept as-is; mark the bottom site as
        # the surface so substitution is done relative to the lowest site
        slab = self.slab_dict["100"]
        bottom_idx = int(np.argmin(slab.frac_coords[:, 2]))
        props = ["surface" if idx == bottom_idx else "subsurface" for idx in range(len(slab))]
        asf = AdsorbateSiteFinder(slab.copy(site_properties={"surface_properties": props}))
        assert asf.slab.site_properties["surface_properties"] == props

        sub_structs = asf.generate_substitution_structures("Cu")
        assert len(sub_structs) == 1
        cu_sites = [site for site in sub_structs[0] if site.species_string == "Cu"]
        assert len(cu_sites) == 1
        assert cu_sites[0].properties["surface_properties"] == "substitute"
        assert cu_sites[0].frac_coords[2] == pytest.approx(slab.frac_coords[bottom_idx, 2])

    def test_get_mi_vec(self):
        for key, slab in self.slab_dict.items():
            mi_vec = get_mi_vec(slab)
            assert np.linalg.norm(mi_vec) == pytest.approx(1)
            # Normal to both in-plane lattice vectors
            assert_allclose(slab.lattice.matrix[:2] @ mi_vec, [0, 0], atol=1e-10, err_msg=key)

    def test_put_coord_inside(self):
        lattice = Lattice.cubic(3.0)
        assert_allclose(put_coord_inside(lattice, [4.0, -1.0, 7.5]), [1.0, 2.0, 1.5])
        assert_allclose(put_coord_inside(lattice, [1.0, 2.0, 1.5]), [1.0, 2.0, 1.5])

    def test_plot_slab(self):
        import matplotlib.pyplot as plt

        slab = self.slab_dict["111"]
        repeat = 5
        _, ax = plt.subplots()
        ret = plot_slab(slab, ax, repeat=repeat)
        assert ret is ax
        # Two circles (white backing + colored) per site, plus the unit cell patch
        assert len(ax.patches) == 2 * len(slab) * repeat**2 + 1
        # One line of adsorption-site markers
        assert len(ax.lines) == 1
        n_ads_sites = len(AdsorbateSiteFinder(slab).find_adsorption_sites()["all"])
        assert len(ax.lines[0].get_xdata()) == n_ads_sites
        assert ax.get_aspect() == 1.0

        _, ax = plt.subplots()
        plot_slab(slab, ax, repeat=0, draw_unit_cell=False, adsorption_sites=False)
        assert len(ax.patches) == 2 * len(slab)
        assert len(ax.lines) == 0

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason="plot_slab(inverse=True) calls np.array(reversed(...)), producing a 0-d object array",
    )
    def test_plot_slab_inverse(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        plot_slab(self.slab_dict["111"], ax, inverse=True)
