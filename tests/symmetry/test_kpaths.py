from __future__ import annotations

import numpy as np
import pytest
from monty.serialization import loadfn

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/electronic_structure/bandstructure"


class TestHighSymmKpath(MatSciTest):
    @pytest.mark.parametrize("path_type", ["setyawan_curtarolo", "latimer_munro"])
    def test_kpath_no_seekpath(self, path_type):
        """Exercise HighSymmKpath paths that don't need seekpath."""
        struct = self.get_structure("Si")
        kpath = HighSymmKpath(struct, path_type=path_type)

        assert kpath.path_type == path_type
        assert kpath.kpath is not None
        assert "kpoints" in kpath.kpath
        assert "path" in kpath.kpath
        # label_index/equiv_labels/path_lengths are only populated for path_type="all"
        assert kpath.label_index is None
        assert kpath.equiv_labels is None
        assert kpath.path_lengths is None

        kpts, labels = kpath.get_kpoints()
        assert len(kpts) == len(labels)
        assert len(kpts) > 0

    def test_path_type_property(self):
        """`path_type` returns whatever was passed in."""
        struct = self.get_structure("Si")
        for pt in ("setyawan_curtarolo", "latimer_munro"):
            assert HighSymmKpath(struct, path_type=pt).path_type == pt

    def test_setyawan_curtarolo_attaches_prim_attrs(self):
        """The SC branch sets `prim`, `conventional`, `prim_rec` on the instance."""
        struct = self.get_structure("Si")
        kpath = HighSymmKpath(struct, path_type="setyawan_curtarolo")
        assert isinstance(kpath.prim, Structure)
        assert isinstance(kpath.conventional, Structure)
        assert isinstance(kpath.prim_rec, Lattice)

    @pytest.mark.usefixtures("seekpath_usable")
    def test_kpath_hinuma(self):
        struct = self.get_structure("Si")
        with pytest.warns(UserWarning, match="K-path from the Hinuma"):
            kpath = HighSymmKpath(struct, path_type="hinuma")
        assert kpath.path_type == "hinuma"
        assert "kpoints" in kpath.kpath

    @pytest.mark.usefixtures("seekpath_usable")
    def test_kpath_all_combines_three(self):
        """`path_type='all'` populates label_index, equiv_labels, and path_lengths."""
        struct = self.get_structure("Si")
        with pytest.warns(UserWarning, match="K-path from the Hinuma"):
            kpath = HighSymmKpath(struct, path_type="all")
        assert kpath.label_index is not None
        assert kpath.equiv_labels is not None
        assert set(kpath.equiv_labels) == {"setyawan_curtarolo", "latimer_munro", "hinuma"}
        # length list has one entry per convention
        assert len(kpath.path_lengths) == 3

    @pytest.mark.usefixtures("seekpath_usable")
    def test_kpath_all_rejects_magmoms(self):
        struct = self.get_structure("Si")
        with pytest.raises(ValueError, match="Cannot select 'all' with non-zero magmoms"):
            HighSymmKpath(struct, path_type="all", has_magmoms=True)

    @pytest.mark.usefixtures("seekpath_usable")
    def test_kpath_generation_across_lattices(self):
        triclinic = [1, 2]
        monoclinic = list(range(3, 16))
        orthorhombic = list(range(16, 75))
        tetragonal = list(range(75, 143))
        rhombohedral = list(range(143, 168))
        hexagonal = list(range(168, 195))
        cubic = list(range(195, 231))

        species = ["K", "La", "Ti"]
        coords = [[0.345, 5, 0.77298], [0.1345, 5.1, 0.77298], [0.7, 0.8, 0.9]]
        rng = np.random.default_rng(seed=0)
        for c in (triclinic, monoclinic, orthorhombic, tetragonal, rhombohedral, hexagonal, cubic):
            sg_num = int(rng.choice(c, 1)[0])
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
            else:
                lattice = Lattice.cubic(2)

            struct = Structure.from_spacegroup(sg_num, lattice, species, coords)
            kpath = HighSymmKpath(struct, path_type="all")
            kpath.get_kpoints()

    def test_continuous_kpath(self):
        bs = loadfn(f"{TEST_DIR}/Cu2O_361_bandstructure.json")
        cont_bs = loadfn(f"{TEST_DIR}/Cu2O_361_bandstructure_continuous.json.gz")
        alt_bs = HighSymmKpath(bs.structure).get_continuous_path(bs)

        assert cont_bs.as_dict() == alt_bs.as_dict()
        assert alt_bs.kpoints[0].label == alt_bs.kpoints[-1].label
