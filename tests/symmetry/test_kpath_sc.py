from __future__ import annotations

import math

import numpy as np
import pytest
from pytest import approx

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.kpath import KPathSetyawanCurtarolo
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/symmetry/space_group_structs"


class TestBandStructureSC(MatSciTest):
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
        for idx in range(230):
            sg_num = idx + 1
            if sg_num in triclinic:
                lattice = Lattice(np.diag((3.0233057319441246, 7.9850357844548681, 8.1136762279561818)))
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
            _ = KPathSetyawanCurtarolo(struct)  # Throws error if something doesn't work, causing test to fail.

        struct_file_path = f"{TEST_DIR}/ICSD_170.cif"
        struct = Structure.from_file(struct_file_path)
        hkp = KPathSetyawanCurtarolo(struct)
        assert hkp.name == "MCLC5"

    def test_kpath_acentered(self):
        species = ["K", "La", "Ti"]
        coords = [[0.345, 5, 0.77298], [0.1345, 5.1, 0.77298], [0.7, 0.8, 0.9]]
        lattice = Lattice.orthorhombic(2, 9, 1)
        struct = Structure.from_spacegroup(38, lattice, species, coords)
        kpath = KPathSetyawanCurtarolo(struct)

        kpoints = kpath._kpath["kpoints"]
        labels = list(kpoints)

        assert sorted(labels) == sorted(["\\Gamma", "A", "A_1", "R", "S", "T", "X", "X_1", "Y", "Z"])

        assert list(kpoints["\\Gamma"]) == [0, 0, 0]

        assert kpoints["A"] == approx([0.25308642, 0.25308642, 0.5])

        assert kpoints["A_1"] == approx([-0.25308642, 0.74691358, 0.5])

        assert kpoints["R"] == approx([0, 0.5, 0.5])

        assert kpoints["S"] == approx([0, 0.5, 0])

        assert kpoints["T"] == approx([-0.5, 0.5, 0.5])

        assert kpoints["X"] == approx([0.25308642, 0.25308642, 0])

        assert kpoints["X_1"] == approx([-0.25308642, 0.74691358, 0])

        assert kpoints["Y"] == approx([-0.5, 0.5, 0])

        assert kpoints["Z"] == approx([0, 0, 0.5])


def _one_atom(sg_num: int, lattice: Lattice) -> Structure:
    return Structure.from_spacegroup(sg_num, lattice, ["Si"], [[0, 0, 0]])


@pytest.mark.parametrize(
    ("struct", "name", "labels"),
    [
        (_one_atom(221, Lattice.cubic(3)), "CUB", {"\\Gamma", "X", "R", "M"}),
        (_one_atom(225, Lattice.cubic(4)), "FCC", {"\\Gamma", "K", "L", "U", "W", "X"}),
        (_one_atom(229, Lattice.cubic(3)), "BCC", {"\\Gamma", "H", "P", "N"}),
        (_one_atom(123, Lattice.tetragonal(3, 4)), "TET", {"\\Gamma", "A", "M", "R", "X", "Z"}),
        (_one_atom(139, Lattice.tetragonal(3, 2)), "BCT1", {"\\Gamma", "M", "N", "P", "X", "Z", "Z_1"}),
        (
            _one_atom(139, Lattice.tetragonal(2, 5)),
            "BCT2",
            {"\\Gamma", "N", "P", "\\Sigma", "\\Sigma_1", "X", "Y", "Y_1", "Z"},
        ),
        (_one_atom(47, Lattice.orthorhombic(2, 3, 4)), "ORC", {"\\Gamma", "R", "S", "T", "U", "X", "Y", "Z"}),
        (
            _one_atom(69, Lattice.orthorhombic(2, 4, 5)),
            "ORCF1",
            {"\\Gamma", "A", "A_1", "L", "T", "X", "X_1", "Y", "Z"},
        ),
        (
            _one_atom(69, Lattice.orthorhombic(4, 5, 6)),
            "ORCF2",
            {"\\Gamma", "C", "C_1", "D", "D_1", "L", "H", "H_1", "X", "Y", "Z"},
        ),
        (
            _one_atom(69, Lattice.orthorhombic(3, 3.75, 5)),  # 1/a^2 == 1/b^2 + 1/c^2
            "ORCF3",
            {"\\Gamma", "A", "A_1", "L", "T", "X", "X_1", "Y", "Z"},
        ),
        (
            _one_atom(71, Lattice.orthorhombic(2, 3, 4)),
            "ORCI",
            {"\\Gamma", "L", "L_1", "L_2", "R", "S", "T", "W", "X", "X_1", "Y", "Y_1", "Z"},
        ),
        (
            _one_atom(65, Lattice.orthorhombic(2, 3, 4)),
            "ORCC",
            {"\\Gamma", "A", "A_1", "R", "S", "T", "X", "X_1", "Y", "Z"},
        ),
        (_one_atom(191, Lattice.hexagonal(3, 5)), "HEX", {"\\Gamma", "A", "H", "K", "L", "M"}),
        (
            _one_atom(166, Lattice.hexagonal(2, 5)),
            "RHL1",
            {"\\Gamma", "B", "B_1", "F", "L", "L_1", "P", "P_1", "P_2", "Q", "X", "Z"},
        ),
        (_one_atom(166, Lattice.hexagonal(2, 1)), "RHL2", {"\\Gamma", "F", "L", "P", "P_1", "Q", "Q_1", "Z"}),
        (
            _one_atom(12, Lattice.monoclinic(8, 4, 3, 110)),
            "MCLC1",
            {"\\Gamma", "N", "N_1", "F", "F_1", "F_2", "I", "I_1", "L", "M", "X", "X_1", "X_2", "Y", "Y_1", "Z"},
        ),
        (
            _one_atom(12, Lattice.monoclinic(4, 5, 6, 100)),
            "MCLC3",
            {"\\Gamma", "F", "F_1", "F_2", "H", "H_1", "H_2", "I", "M", "N", "N_1", "X", "Y", "Y_1", "Y_2", "Y_3", "Z"},
        ),
        (
            Structure(Lattice.from_parameters(4.91, 3.81, 3.12, 70.7, 102.5, 106.5), ["Si"], [[0, 0, 0]]),
            "TRI1a",
            {"\\Gamma", "L", "M", "N", "R", "X", "Y", "Z"},
        ),
        (
            Structure(Lattice.from_parameters(3, 4, 5, 80, 85, 95), ["Si"], [[0, 0, 0]]),
            "TRI1b",
            {"\\Gamma", "L", "M", "N", "R", "X", "Y", "Z"},
        ),
    ],
)
def test_bravais_lattice_variants(struct, name, labels):
    kpath = KPathSetyawanCurtarolo(struct)
    assert kpath.name == name
    assert set(kpath.kpath["kpoints"]) == labels
    assert kpath.kpath["kpoints"]["\\Gamma"] == approx([0, 0, 0])
    # every label used in the path must be defined
    assert {label for segment in kpath.kpath["path"] for label in segment} <= labels


def test_bct1_kpoint_coords():
    kpath = KPathSetyawanCurtarolo(_one_atom(139, Lattice.tetragonal(3, 2)))
    a, _, c = kpath.conventional.lattice.abc
    eta = (1 + c**2 / a**2) / 4
    assert kpath.kpath["kpoints"]["Z"] == approx([eta, eta, -eta])
    assert kpath.kpath["kpoints"]["Z_1"] == approx([-eta, 1 - eta, eta])


def test_rhl2_kpoint_coords():
    kpath = KPathSetyawanCurtarolo(_one_atom(166, Lattice.hexagonal(2, 1)))
    alpha = np.radians(kpath.prim.lattice.angles[0])
    assert alpha > np.pi / 2
    eta = 1 / (2 * np.tan(alpha / 2) ** 2)
    nu = 3 / 4 - eta / 2
    assert kpath.kpath["kpoints"]["Q"] == approx([eta, eta, eta])
    assert kpath.kpath["kpoints"]["P"] == approx([1 - nu, -nu, 1 - nu])


def test_orcf2_kpoint_coords():
    kpath = KPathSetyawanCurtarolo(_one_atom(69, Lattice.orthorhombic(4, 5, 6)))
    a, b, c = kpath.conventional.lattice.abc
    phi = (1 + c**2 / b**2 - c**2 / a**2) / 4
    eta = (1 + a**2 / b**2 - a**2 / c**2) / 4
    delta = (1 + b**2 / a**2 - b**2 / c**2) / 4
    kpts = kpath.kpath["kpoints"]
    assert kpts["C"] == approx([0.5, 0.5 - eta, 1 - eta])
    assert kpts["D"] == approx([0.5 - delta, 0.5, 1 - delta])
    assert kpts["H"] == approx([1 - phi, 0.5 - phi, 0.5])


@pytest.mark.parametrize(
    ("method", "name", "last_segment"),
    [("mclc2", "MCLC2", ["N", "\\Gamma", "M"]), ("mclc4", "MCLC4", ["M", "\\Gamma"])],
)
def test_mclc_edge_cases_direct(method, name, last_segment):
    """MCLC2/MCLC4 require exact equalities, so exercise the path builders directly."""
    kpath = KPathSetyawanCurtarolo(_one_atom(12, Lattice.monoclinic(4, 5, 6, 100)))
    result = getattr(kpath, method)(4, 5, 6, np.radians(70))
    assert kpath.name == name
    assert result["path"][-1] == last_segment
    assert result["kpoints"]["Z"] == approx([0, 0, 0.5])
    assert {label for segment in result["path"] for label in segment} <= set(result["kpoints"])


def test_magmom_warning():
    struct = _one_atom(221, Lattice.cubic(3))
    struct.add_site_property("magmom", [1.0])
    with pytest.warns(UserWarning, match="'magmom' entry found in site properties"):
        kpath = KPathSetyawanCurtarolo(struct)
    assert kpath.name == "CUB"


def test_non_standard_input_warning():
    struct = _one_atom(229, Lattice.cubic(3))  # conventional, not primitive
    with pytest.warns(UserWarning, match="does not match the expected standard primitive"):
        KPathSetyawanCurtarolo(struct)


@pytest.mark.parametrize(
    ("struct", "spg_symbol"),
    [
        (_one_atom(123, Lattice.tetragonal(3, 4)), "F4/mmm"),
        (_one_atom(12, Lattice.monoclinic(4, 5, 6, 100)), "I2/m"),
    ],
)
def test_unexpected_space_group_symbol(monkeypatch, struct, spg_symbol):
    """Centerings that are crystallographically impossible for the lattice type fall through with a warning."""
    monkeypatch.setattr(SpacegroupAnalyzer, "get_space_group_symbol", lambda _self: spg_symbol)
    with pytest.warns(UserWarning, match="Unexpected value for spg_symbol"):
        kpath = KPathSetyawanCurtarolo(struct)
    assert kpath.kpath is None


def test_unknown_lattice_type(monkeypatch):
    monkeypatch.setattr(SpacegroupAnalyzer, "get_lattice_type", lambda _self: "quasicrystal")
    with pytest.warns(UserWarning, match="Unknown lattice type quasicrystal"):
        kpath = KPathSetyawanCurtarolo(_one_atom(221, Lattice.cubic(3)))
    assert kpath.kpath is None


class TestKPathBaseHelpers:
    @pytest.fixture
    def kpath(self):
        return KPathSetyawanCurtarolo(_one_atom(221, Lattice.cubic(3)))

    def test_properties(self, kpath):
        assert kpath.structure.lattice == Lattice.cubic(3)
        assert kpath.lattice == Lattice.cubic(3)
        assert kpath.rec_lattice is kpath.prim_rec
        assert kpath.rec_lattice.abc == approx([2 * np.pi / 3] * 3)

    def test_get_kpoints_cartesian_vs_fractional(self, kpath):
        cart, labels = kpath.get_kpoints(line_density=10)
        frac, frac_labels = kpath.get_kpoints(line_density=10, coords_are_cartesian=False)
        assert labels == frac_labels
        assert len(cart) == len(frac) == len(labels)
        assert labels[0] == "\\Gamma"
        assert labels[-1] == "R"
        assert frac[0] == approx([0, 0, 0])
        assert cart[-1] == approx(kpath.rec_lattice.get_cartesian_coords(kpath.kpath["kpoints"]["R"]))
        for kpt, label in zip(frac, labels, strict=True):
            if label:
                assert kpt == approx(kpath.kpath["kpoints"][label])

    def test_get_kpoints_line_density(self, kpath):
        coarse, _ = kpath.get_kpoints(line_density=5)
        fine, _ = kpath.get_kpoints(line_density=40)
        assert len(fine) > len(coarse)
        # Gamma -> X: |X| = 0.5 * 2pi/3 in Cartesian reciprocal space
        _, labels = kpath.get_kpoints(line_density=5)
        n_gx = math.ceil(0.5 * 2 * np.pi / 3 * 5)
        assert labels[: n_gx + 1] == ["\\Gamma", *[""] * (n_gx - 1), "X"]

    def test_get_kpoints_zero_density(self, kpath):
        kpts, labels = kpath.get_kpoints(line_density=0)
        assert kpts == []
        assert labels == []
