from __future__ import annotations

import copy
import platform

import pytest
from pytest import approx

from pymatgen.core.graphs import MoleculeGraph
from pymatgen.core.molecule_matcher import MoleculeMatcher
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.xyz import XYZ
from pymatgen.util.testing import TEST_FILES_DIR

pybel = pytest.importorskip("openbabel.pybel")


class TestBabelMolAdaptor:
    def setup_method(self):
        coords = [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 1.089000],
            [1.026719, 0.000000, -0.363000],
            [-0.513360, -0.889165, -0.363000],
            [-0.513360, 0.889165, -0.363000],
        ]
        self.mol = Molecule(["C", "H", "H", "H", "H"], coords)

    def test_init(self):
        adaptor = BabelMolAdaptor(self.mol)
        ob_mol = adaptor.openbabel_mol
        pybel = adaptor.pybel_mol
        assert ob_mol.NumAtoms() == 5

        adaptor = BabelMolAdaptor(adaptor.openbabel_mol)
        assert adaptor.pymatgen_mol.formula == "H4 C1"

        adaptor = BabelMolAdaptor(pybel)
        assert adaptor.pymatgen_mol.formula == "H4 C1"

    def test_from_file(self):
        adaptor = BabelMolAdaptor.from_file(f"{TEST_FILES_DIR}/io/babel/Ethane_e.pdb", "pdb")
        mol = adaptor.pymatgen_mol
        assert mol.formula == "H6 C2"

    def test_from_file_return_all_molecules(self):
        adaptors = BabelMolAdaptor.from_file(
            f"{TEST_FILES_DIR}/io/xyz/multiple_frame.xyz",
            "xyz",
            return_all_molecules=True,
        )
        assert len(adaptors) == 302

    def test_from_molecule_graph(self):
        graph = MoleculeGraph.from_empty_graph(self.mol)
        adaptor = BabelMolAdaptor.from_molecule_graph(graph)
        ob_mol = adaptor.openbabel_mol
        assert ob_mol.NumAtoms() == 5
        mol = adaptor.pymatgen_mol
        assert mol.formula == "H4 C1"

    def test_from_str(self):
        xyz = XYZ(self.mol)
        adaptor = BabelMolAdaptor.from_str(str(xyz), "xyz")
        mol = adaptor.pymatgen_mol
        assert mol.formula == "H4 C1"

    def test_localopt(self):
        self.mol[1] = "H", [0, 0, 1.05]
        adaptor = BabelMolAdaptor(self.mol)
        adaptor.localopt()
        opt_mol = adaptor.pymatgen_mol
        for site in opt_mol[1:]:
            assert site.distance(opt_mol[0]) == approx(1.09216, abs=1e-1)

    def test_make3d(self):
        mol_0d = pybel.readstring("smi", "CCCC").OBMol
        adaptor = BabelMolAdaptor(mol_0d)
        adaptor.make3d()
        assert mol_0d.GetDimension() == 3

    def add_hydrogen(self):
        mol_0d = pybel.readstring("smi", "CCCC").OBMol
        assert len(pybel.Molecule(mol_0d).atoms) == 2
        adaptor = BabelMolAdaptor(mol_0d)
        adaptor.add_hydrogen()
        assert len(adaptor.pymatgen_mol) == 14

    @staticmethod
    def _eclipsed_butane():
        """3D n-butane with the C-C-C-C dihedral forced to 0 deg (eclipsed, high energy)."""
        mol = pybel.readstring("smi", "CCCC")
        mol.make3D()
        ob_mol = mol.OBMol
        ob_mol.SetTorsion(*(ob_mol.GetAtom(idx) for idx in (1, 2, 3, 4)), 0.0)
        assert ob_mol.GetTorsion(1, 2, 3, 4) == approx(0, abs=1e-6)
        return ob_mol

    @pytest.mark.parametrize(
        ("algo", "rotor_args"),
        [
            ("WeightedRotorSearch", (250, 50)),
            ("SystematicRotorSearch", (200,)),
            ("RandomRotorSearch", (250, 50)),
        ],
    )
    def test_rotor_conformer(self, algo, rotor_args):
        adaptor = BabelMolAdaptor(self._eclipsed_butane())
        # Leave the process-global forcefield set up on an unrelated molecule. Without
        # ff.Setup(self._ob_mol) in rotor_conformer, the search would run on this one.
        pybel.readstring("smi", "CCO").make3D()

        adaptor.rotor_conformer(*rotor_args, algo=algo)
        ob_mol = adaptor.openbabel_mol
        assert ob_mol.NumAtoms() == 14
        # The search must relax the eclipsed rotor to the anti conformer.
        assert abs(ob_mol.GetTorsion(1, 2, 3, 4)) == approx(180, abs=5)

    def test_rotor_conformer_methane(self):
        mol = copy.deepcopy(self.mol)
        mol[1] = "H", [0, 0, 1.05]
        adaptor = BabelMolAdaptor(mol)
        adaptor.rotor_conformer(250, 50, algo="WeightedRotorSearch")
        opt_mol = adaptor.pymatgen_mol
        assert opt_mol.formula == "H4 C1"

    def test_rotor_conformer_invalid_forcefield(self):
        adaptor = BabelMolAdaptor(self._eclipsed_butane())
        pybel.readstring("smi", "CCO").make3D()
        with pytest.warns(UserWarning, match="not supported in openbabel"):
            adaptor.rotor_conformer(50, 10, forcefield="not_a_forcefield")
        assert abs(adaptor.openbabel_mol.GetTorsion(1, 2, 3, 4)) == approx(180, abs=5)

    def test_confab_conformers_freeze_atoms(self):
        adaptor = BabelMolAdaptor(self._eclipsed_butane())
        conformers = adaptor.confab_conformers(freeze_atoms=[1, 2, 3, 4])
        assert len(conformers) >= 1
        assert all(len(conf) == 14 for conf in conformers)

        # Constraints must not leak into later calls via the process-global forcefield.
        mol = pybel.readstring("smi", "CCCC").OBMol
        adaptor = BabelMolAdaptor(mol)
        adaptor.make3d()
        n_free = len(adaptor.confab_conformers())
        assert n_free > 1

    @pytest.mark.xfail(platform.system() == "Windows", reason="Tests for openbabel failing on Win")
    def test_confab_conformers(self):
        mol = pybel.readstring("smi", "CCCC").OBMol
        adaptor = BabelMolAdaptor(mol)
        adaptor.make3d()
        conformers = adaptor.confab_conformers()
        assert adaptor.openbabel_mol.NumRotors() == 1
        assert len(conformers) >= 1
        if len(conformers) > 1:
            assert MoleculeMatcher().get_rmsd(conformers[0], conformers[1]) != approx(0)
