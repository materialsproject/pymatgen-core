from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest import approx

from pymatgen.core import Composition, Element, Lattice, PeriodicSite, Site, Species, Structure
from pymatgen.core.sites import _lazy_site_dict, _LazySiteDict, _sites_pairwise_unequal
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.testing import MatSciTest


class TestSite(MatSciTest):
    def setup_method(self):
        self.ordered_site = Site("Fe", [0.25, 0.35, 0.45])
        self.disordered_site = Site({"Fe": 0.5, "Mn": 0.5}, [0.25, 0.35, 0.45])
        self.propertied_site = Site("Fe2+", [0.25, 0.35, 0.45], {"magmom": 5.1, "charge": 4.2})
        self.propertied_magmom_vec_site = Site(
            "Fe2+",
            [0.25, 0.35, 0.45],
            {"magmom": Magmom([2.6, 2.6, 3.5]), "charge": 4.2},
        )
        self.dummy_site = Site("X", [0, 0, 0])

    def test_properties(self):
        assert not self.disordered_site.is_ordered
        # TODO (janosh): this doesn't raise the expected error (match="specie property only works for ordered sites")
        with pytest.raises(AttributeError, match="attr='specie' not found on Site"):
            _ = self.disordered_site.specie
        assert isinstance(self.ordered_site.specie, Element)
        assert self.propertied_site.properties["magmom"] == approx(5.1)
        assert self.propertied_site.properties["charge"] == approx(4.2)

    def test_as_from_dict(self):
        dct = self.disordered_site.as_dict()
        site = Site.from_dict(dct)
        assert site == self.disordered_site
        assert site != self.ordered_site
        dct = self.propertied_site.as_dict()
        site = Site.from_dict(dct)
        assert site.properties["magmom"] == approx(5.1)
        assert site.properties["charge"] == approx(4.2)
        dct = self.propertied_magmom_vec_site.as_dict()
        site = Site.from_dict(dct)
        assert site.properties["magmom"] == Magmom([2.6, 2.6, 3.5])
        assert site.properties["charge"] == approx(4.2)
        dct = self.dummy_site.as_dict()
        site = Site.from_dict(dct)
        assert site.species == self.dummy_site.species

    def test_hash(self):
        assert hash(self.ordered_site) == 26
        assert hash(self.disordered_site) == 51

    def test_gt_lt(self):
        assert self.ordered_site > self.disordered_site
        assert self.disordered_site < self.ordered_site

    def test_distance(self):
        ord_site = self.ordered_site
        assert np.linalg.norm([0.25, 0.35, 0.45]) == ord_site.distance_from_point([0, 0, 0])
        assert ord_site.distance(self.disordered_site) == 0

    def test_pickle(self):
        dump = pickle.dumps(self.propertied_site)
        assert pickle.loads(dump) == self.propertied_site

    def test_setters(self):
        self.disordered_site.species = "Cu"
        assert self.disordered_site.species == Composition("Cu")
        self.disordered_site.x = 1.25
        self.disordered_site.y = 1.35
        assert self.disordered_site.coords[0] == approx(1.25)
        assert self.disordered_site.coords[1] == approx(1.35)

        with pytest.raises(ValueError, match="Species occupancies sum to more than 1"):
            self.disordered_site.species = {"Cu": 0.5, "Gd": 0.6}

    def test_from_dict_does_not_alias_properties(self):
        dct = self.propertied_site.as_dict()
        site = Site.from_dict(dct)
        site.properties["magmom"] = 0
        assert self.propertied_site.properties["magmom"] == approx(5.1)
        assert dct["properties"]["magmom"] == approx(5.1)


class TestPeriodicSite(MatSciTest):
    def setup_method(self):
        self.lattice = Lattice.cubic(10.0)
        self.si = Element("Si")
        self.site = PeriodicSite("Fe", [0.25, 0.35, 0.45], self.lattice)
        self.site2 = PeriodicSite({"Si": 0.5}, [0, 0, 0], self.lattice)
        assert self.site2.species == Composition({Element("Si"): 0.5}), "Inconsistent site created!"
        self.propertied_site = PeriodicSite(
            Species("Fe", 2),
            [0.25, 0.35, 0.45],
            self.lattice,
            properties={"magmom": 5.1, "charge": 4.2},
        )
        self.labeled_site = PeriodicSite("Fe", [0.25, 0.35, 0.45], self.lattice, label="site label")
        self.dummy_site = PeriodicSite("X", [0, 0, 0], self.lattice)

    def test_properties(self):
        """Test the properties for a site."""
        assert self.site.a == approx(0.25)
        assert self.site.b == approx(0.35)
        assert self.site.c == approx(0.45)
        assert self.site.x == approx(2.5)
        assert self.site.y == approx(3.5)
        assert self.site.z == approx(4.5)
        assert self.site.is_ordered
        assert self.site.label == "Fe"
        assert not self.site2.is_ordered
        assert self.propertied_site.properties["magmom"] == approx(5.1)
        assert self.propertied_site.properties["charge"] == approx(4.2)
        assert self.labeled_site.label == "site label"

    def test_distance(self):
        other_site = PeriodicSite("Fe", np.array([0, 0, 0]), self.lattice)
        assert self.site.distance(other_site) == approx(6.22494979899)

    def test_distance_from_point(self):
        assert self.site.distance_from_point([0.1, 0.1, 0.1]) == approx(6.0564015718906887)

    def test_distance_and_image(self):
        other_site = PeriodicSite("Fe", np.array([1, 1, 1]), self.lattice)
        distance, image = self.site.distance_and_image(other_site)
        assert distance == approx(6.22494979899)
        assert (image == [-1, -1, -1]).all()
        distance, image = self.site.distance_and_image(other_site, [1, 0, 0])
        assert distance == approx(19.461500456028563)
        # Test that old and new distance algo give the same answer for
        # "standard lattices"
        lattice = Lattice(np.eye(3))
        site1 = PeriodicSite("Fe", np.array([0.01, 0.02, 0.03]), lattice)
        site2 = PeriodicSite("Fe", np.array([0.99, 0.98, 0.97]), lattice)
        assert get_distance_and_image_old(site1, site2)[0] == approx(site1.distance_and_image(site2)[0])
        lattice = Lattice.from_parameters(1, 0.01, 1, 10, 10, 10)
        site1 = PeriodicSite("Fe", np.array([0.01, 0.02, 0.03]), lattice)
        site2 = PeriodicSite("Fe", np.array([0.99, 0.98, 0.97]), lattice)
        assert get_distance_and_image_old(site1, site2)[0] > site1.distance_and_image(site2)[0]
        site2 = PeriodicSite("Fe", np.random.default_rng().random(3), lattice)
        dist_old, jimage_old = get_distance_and_image_old(site1, site2)
        dist_new, jimage_new = site1.distance_and_image(site2)
        assert dist_old - dist_new > -1e-8, "New distance algo should give smaller answers!"
        assert not (abs(dist_old - dist_new) < 1e-8) ^ (jimage_old == jimage_new).all(), (
            "If old dist == new dist, images must be the same!"
        )
        lattice = Lattice.from_parameters(3.0, 3.1, 10.0, 2.96, 2.0, 1.0)
        site = PeriodicSite("Fe", [0.1, 0.1, 0.1], lattice)
        site2 = PeriodicSite("Fe", [0.99, 0.99, 0.99], lattice)
        dist, img = site.distance_and_image(site2)
        assert dist == approx(0.15495358379511573)
        assert list(img) == [-11, 6, 0]

    def test_is_periodic_image(self):
        other = PeriodicSite("Fe", np.array([1.25, 2.35, 4.45]), self.lattice)
        assert self.site.is_periodic_image(other), "This other site should be a periodic image."
        other = PeriodicSite("Fe", np.array([1.25, 2.35, 4.46]), self.lattice)
        assert not self.site.is_periodic_image(other), "This other site should not be a periodic image."
        other = PeriodicSite("Fe", np.array([1.25, 2.35, 4.45]), Lattice.rhombohedral(2, 60))
        assert not self.site.is_periodic_image(other), "Different lattices should not be periodic images."

    def test_equality(self):
        assert self.site == self.site
        other_site = PeriodicSite("Fe", np.array([1, 1, 1]), self.lattice)
        assert other_site != self.site

    def test_equality_with_label(self):
        site = PeriodicSite("Fe", [0.25, 0.35, 0.45], self.lattice, label="Fe3")
        assert site == self.site

        assert self.labeled_site.label != site.label
        assert self.labeled_site == site

    def test_equality_prop_with_np_array(self):
        """Some property (e.g. selective dynamics for POSCAR) could be numpy arrays,
        use "==" for equality check might fail in these cases.
        """
        site_0 = PeriodicSite(
            "Fe",
            [0.25, 0.35, 0.45],
            self.lattice,
            properties={"selective_dynamics": np.array([True, True, False])},
        )
        assert site_0 == site_0  # noqa: PLR0124

        site_1 = PeriodicSite(
            "Fe",
            [0.25, 0.35, 0.45],
            self.lattice,
            properties={"selective_dynamics": np.array([True, False, False])},
        )
        assert site_0 != site_1

    def test_from_dict_does_not_alias_properties(self):
        site = PeriodicSite("Fe", [0, 0, 0], self.lattice, properties={"magmom": 5})
        dct = site.as_dict()
        new_site = PeriodicSite.from_dict(dct)
        new_site.properties["magmom"] = 0
        assert site.properties["magmom"] == 5
        assert dct["properties"]["magmom"] == 5

    def test_init_does_not_alias_coords(self):
        frac_coords = np.array([0.1, 0.2, 0.3])
        site = PeriodicSite("Fe", frac_coords, self.lattice)
        frac_coords[0] = 0.9
        assert_allclose(site.frac_coords, [0.1, 0.2, 0.3])
        site.a = 0.5
        assert_allclose(frac_coords, [0.9, 0.2, 0.3])

    def test_as_from_dict(self):
        dct = self.site2.as_dict()
        site = PeriodicSite.from_dict(dct)
        assert site == self.site2
        assert site != self.site
        assert site.label == self.site2.label

        dct = self.propertied_site.as_dict()
        site3 = PeriodicSite({"Si": 0.5, "Fe": 0.5}, [0, 0, 0], self.lattice)
        dct = site3.as_dict()
        site = PeriodicSite.from_dict(dct)
        assert site.species == site3.species
        assert site.label == site3.label

        dct = self.dummy_site.as_dict()
        site = PeriodicSite.from_dict(dct)
        assert site.species == self.dummy_site.species
        assert site.label == self.dummy_site.label

    def test_to_unit_cell(self):
        site = PeriodicSite("Fe", np.array([1.25, 2.35, 4.46]), self.lattice)
        site.to_unit_cell(in_place=True)
        val = [0.25, 0.35, 0.46]
        assert_allclose(site.frac_coords, val)

        lattice_pbc = Lattice(self.lattice.matrix, pbc=(True, True, False))
        site = PeriodicSite("Fe", np.array([1.25, 2.35, 4.46]), lattice_pbc)
        site.to_unit_cell(in_place=True)
        val = [0.25, 0.35, 4.46]
        assert_allclose(site.frac_coords, val)

    def test_setters(self):
        site = self.propertied_site
        site.species = "Cu"
        assert site.species == Composition("Cu")
        site.x = 1.25
        site.y = 1.35
        assert site.coords[0] == approx(1.25)
        assert site.coords[1] == approx(1.35)
        assert site.a == approx(0.125)
        assert site.b == approx(0.135)
        site.lattice = Lattice.cubic(100)
        assert site.x == approx(12.5)

        with pytest.raises(ValueError, match="Species occupancies sum to more than 1"):
            site.species = {"Cu": 0.5, "Gd": 0.6}

        site.frac_coords = [0, 0, 0.1]
        assert_allclose(site.coords, [0, 0, 10])
        site.coords = [1.5, 3.25, 5]
        assert_allclose(site.frac_coords, [0.015, 0.0325, 0.05])

    def test_property_setattr_routes_to_properties(self):
        """Setting an arbitrary attribute on a site should store it in site.properties,
        not as a bare instance attribute, so it survives from_sites / get_sorted_structure."""

        site = PeriodicSite("Fe", [0.25, 0.35, 0.45], self.lattice)

        # Assignment via attribute syntax must land in .properties (deprecated, but still supported)
        with pytest.warns(DeprecationWarning, match="Setting attribute magmom on PeriodicSite is deprecated"):
            site.magmom = 5
        assert site.properties["magmom"] == 5, "magmom should be stored in site.properties, not as a bare attribute"

        # Round-trip through from_sites must preserve it
        reconstructed = PeriodicSite.from_dict(site.as_dict())
        assert reconstructed.magmom == 5, "magmom lost after as_dict/from_dict round-trip"

        # Survival through get_sorted_structure is the real footgun
        from pymatgen.core import Structure

        struct = Structure(
            self.lattice,
            ["Cr", "Fe"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        for site, magmom in zip(struct.sites, (-5, 5), strict=True):
            with pytest.warns(DeprecationWarning, match="Setting attribute magmom on PeriodicSite is deprecated"):
                site.magmom = magmom
        sorted_struct = struct.get_sorted_structure()
        assert all("magmom" in site.properties for site in sorted_struct.sites), (
            "magmom lost on one or more sites after get_sorted_structure"
        )

    def test_repr(self):
        assert repr(self.propertied_site) == "PeriodicSite: Fe2+ (2.5, 3.5, 4.5) [0.25, 0.35, 0.45]"
        assert repr(self.labeled_site) == "PeriodicSite: site label (Fe) (2.5, 3.5, 4.5) [0.25, 0.35, 0.45]"

    def test_str(self):
        assert str(self.site) == "[2.5 3.5 4.5] Fe"
        assert str(self.site2) == "[0. 0. 0.] Si:0.5"
        assert str(self.propertied_site) == "[2.5 3.5 4.5] Fe2+"
        assert str(self.labeled_site) == "[2.5 3.5 4.5] Fe"

    def test_invalid_coordinate(self):
        """Test that non (3,)-shaped coordinate settings are blocked."""
        with pytest.raises(ValueError, match="Coordinates"):
            PeriodicSite("H", np.zeros((2,)), Lattice.cubic(1))

        # Also block the setters from accepting false shapes
        okay = PeriodicSite("H", np.zeros((3,)), Lattice.cubic(1))
        with pytest.raises(ValueError, match="coordinates"):
            okay.frac_coords = np.zeros((2,))
        okay = PeriodicSite("H", np.zeros((3,)), Lattice.cubic(1))
        with pytest.raises(ValueError, match="coordinates"):
            okay.coords = np.zeros((2,))

    @pytest.mark.parametrize(
        ("pbc", "expected"),
        [((True, True, True), [0.25, 0.65, 0.46]), ((True, False, True), [0.25, -2.35, 0.46])],
    )
    @pytest.mark.parametrize("skip_checks", [False, True])
    def test_init_to_unit_cell(self, pbc, expected, skip_checks):
        lattice = Lattice(self.lattice.matrix, pbc=pbc)
        coords = np.array([1.25, -2.35, 4.46])
        site = PeriodicSite("Fe", coords, lattice, to_unit_cell=True, skip_checks=skip_checks)
        assert_allclose(site.frac_coords, expected)
        assert site.frac_coords.dtype == np.float64
        # Wrapping must not alias or mutate the caller's array
        assert not np.shares_memory(site.frac_coords, coords)
        assert_allclose(coords, [1.25, -2.35, 4.46])
        # Integer input coords are wrapped into a float array too
        int_site = PeriodicSite("Fe", [1, -2, 4], lattice, to_unit_cell=True)
        assert_allclose(int_site.frac_coords, [0, 0 if pbc[1] else -2, 0])
        assert int_site.frac_coords.dtype == np.float64

    @pytest.mark.parametrize(
        "offset",
        [0, 1e-6, 9.9e-6, 1e-5, 1.1e-5, 1e-4, -1e-4, 1.2, np.nan, np.inf, -np.inf],
    )
    def test_equality_matches_allclose(self, offset):
        """Site equality must keep the np.allclose(atol=Site.position_atol) coordinate semantics."""
        for coords in ([0.0, 3.5, 4.5], [np.inf, 3.5, 4.5], [np.nan, 3.5, 4.5]):
            ref = PeriodicSite("Fe", coords, Lattice.cubic(1))
            other = PeriodicSite("Fe", np.add(coords, [offset, 0, 0]), Lattice.cubic(1))
            expected = bool(np.allclose(ref.coords, other.coords, atol=Site.position_atol))
            assert (ref == other) is expected
            assert (other == ref) is bool(np.allclose(other.coords, ref.coords, atol=Site.position_atol))

            site_ref = Site("Fe", coords)
            site_other = Site("Fe", np.add(coords, [offset, 0, 0]))
            assert (site_ref == site_other) is bool(np.allclose(site_ref.coords, site_other.coords, atol=1e-5))

        # Coordinates match but species or lattice differ
        assert PeriodicSite("Fe", [0, 0, 0], Lattice.cubic(1)) != PeriodicSite("Co", [0, 0, 0], Lattice.cubic(1))
        assert PeriodicSite("Fe", [0, 0, 0], Lattice.cubic(1)) != PeriodicSite("Fe", [0, 0, 0], Lattice.cubic(2))
        assert Site("Fe", [0, 0, 0]) != Site("Co", [0, 0, 0])

    def test_hash_consistent_with_tolerant_eq(self):
        # Sites differing only by float noise compare equal, so they must hash equal and find each other in dicts
        lattice = Lattice.cubic(5)
        site = PeriodicSite("Fe", [0.1, 0.2, 0.3], lattice, properties={"magmom": 2})
        noisy = PeriodicSite("Fe", [0.1 + 1e-9, 0.2 - 1e-9, 0.3], lattice, properties={"magmom": 2})
        assert site == noisy
        assert hash(site) == hash(noisy)
        assert {site: 1}[noisy] == 1
        assert Site("Fe", [0, 0, 1]) == Site("Fe", [0, 0, 1 + 1e-9])
        assert hash(Site("Fe", [0, 0, 1])) == hash(Site("Fe", [0, 0, 1 + 1e-9]))


class TestLazySiteDict:
    def setup_method(self):
        self.struct = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]]) * (3, 3, 3)
        self.struct.append("O", [0.5, 0.5, 0.5])
        self.values = [{"idx": idx} for idx in range(len(self.struct))]

    def test_sites_pairwise_unequal(self):
        sites = list(self.struct)
        assert _sites_pairwise_unequal(sites)
        assert _sites_pairwise_unequal(sites[:1])
        assert _sites_pairwise_unequal([])
        # The same site twice, or an equal site within the coordinate tolerance
        assert not _sites_pairwise_unequal([*sites, sites[5]])
        near = PeriodicSite("Si", sites[5].frac_coords + 1e-8, self.struct.lattice)
        assert near == sites[5]
        assert not _sites_pairwise_unequal([*sites, near])
        # Same position but different species or properties are unequal sites
        other_species = PeriodicSite("Ge", sites[5].frac_coords, self.struct.lattice)
        assert _sites_pairwise_unequal([*sites, other_species])
        # Non-finite coordinates are conservatively reported as possibly equal
        assert not _sites_pairwise_unequal([Site("Fe", [np.inf, 0, 0]), Site("Fe", [0, 0, 0])])
        # Equality is only one-directional near the relative-tolerance edge: still detected
        site_a, site_b = Site("Fe", [1e3, 0, 0]), Site("Fe", [1e3 + 0.01001005, 0, 0])
        assert (site_a == site_b) != (site_b == site_a)
        assert not _sites_pairwise_unequal([site_a, site_b])

    def test_matches_dict(self):
        sites = list(self.struct)
        ref = dict(zip(sites, self.values, strict=True))
        lazy = _lazy_site_dict(self.struct, self.values)
        assert isinstance(lazy, _LazySiteDict)
        assert len(lazy) == len(ref)
        assert list(lazy) == list(ref)
        assert all(key is ref_key for key, ref_key in zip(lazy, ref, strict=True))
        assert list(lazy.items()) == list(ref.items())
        assert list(lazy.values()) == list(ref.values())
        for site in sites:
            assert lazy[site] is ref[site]
        assert lazy._dict is None  # identity lookups do not build the dict

        # Lookups by equal but non-identical sites (float noise) give the same values as the plain dict
        copies = [PeriodicSite(s.species, s.frac_coords + 1e-10, s.lattice) for s in sites]
        assert all(lazy[copy] is ref[copy] for copy in copies)
        missing = PeriodicSite("Fe", [0.1, 0.1, 0.1], self.struct.lattice)
        with pytest.raises(KeyError):
            lazy[missing]
        assert missing not in lazy

        # Once the dict is handed out it is the source of truth, and it is built only once
        dct = lazy.to_dict()
        assert dct == ref
        assert lazy.to_dict() is dct
        dct[sites[3]] = "new"
        del dct[sites[4]]
        assert lazy[sites[3]] == "new"
        assert sites[4] not in lazy
        assert len(lazy) == len(ref) - 1
        assert dict(lazy.items()) == dct

    def test_fallback_to_dict(self):
        sites = list(self.struct)
        # Duplicate sites collapse exactly as in a plain dict
        dup = _lazy_site_dict([*sites, sites[2]], [*self.values, "dup"])
        assert type(dup) is dict
        assert dup == dict(zip([*sites, sites[2]], [*self.values, "dup"], strict=True))
        # Non-site keys
        assert _lazy_site_dict(["a", "b"], [1, 2]) == {"a": 1, "b": 2}
        with pytest.raises(ValueError, match="Got 2 sites but 1 values"):
            _lazy_site_dict(sites[:2], [1])

    def test_pickle_and_copy(self):
        lazy = _lazy_site_dict(self.struct, self.values)
        lazy[self.struct[0]]  # build the identity index
        for clone in (pickle.loads(pickle.dumps(lazy)), copy.deepcopy(lazy)):
            assert clone._index is None
            assert list(clone.items()) == list(lazy.items())
            for site in clone:
                assert clone[site] == lazy[site]


def get_distance_and_image_old(site1, site2, jimage=None):
    """
    Gets distance between two sites assuming periodic boundary conditions.
    If the index jimage of two sites atom j is not specified it selects the
    j image nearest to the i atom and returns the distance and jimage
    indices in terms of lattice vector translations. If the index jimage of
    atom j is specified it returns the distance between the i atom and the
    specified jimage atom, the given jimage is also returned.

    Args:
        other:
            other site to get distance from.
        jimage:
            specific periodic image in terms of lattice translations,
            e.g. [1,0,0] implies to take periodic image that is one
            a-lattice vector away. If jimage is None, the image that is
            nearest to the site is found.

    Returns:
        tuple[float, np.array]: distance and periodic lattice translations of the other site
            for which the distance applies.

    Note:
        Assumes the primitive cell vectors are sufficiently not skewed such
        that the condition \\|a\\|cos(ab_angle) < \\|b\\| for all possible cell
        vector pairs. ** this method does not check this condition **
    """
    if jimage is None:
        # Old algorithm
        jimage = -np.array(np.around(site2.frac_coords - site1.frac_coords), int)
    mapped_vec = site1.lattice.get_cartesian_coords(jimage + site2.frac_coords - site1.frac_coords)
    dist = np.linalg.norm(mapped_vec)
    return dist, jimage
