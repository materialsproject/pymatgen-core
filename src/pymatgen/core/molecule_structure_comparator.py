"""
This module provides classes to comparison the structures of the two
molecule. As long as the two molecule have the same bond connection tables,
the molecules are deemed to be same. The atom in the two molecule must be
paired accordingly.
This module is supposed to perform rough comparisons with the atom order
correspondence prerequisite, while molecule_matcher is supposed to do exact
comparisons without the atom order correspondence prerequisite.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from monty.json import MSONable
from scipy.spatial import cKDTree

from pymatgen.util.due import Doi, due

if TYPE_CHECKING:
    from typing import ClassVar, Self

__author__ = "Xiaohui Qu"
__copyright__ = "Copyright 2011, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Xiaohui Qu"
__email__ = "xhqu1981@gmail.com"
__status__ = "Experimental"
__date__ = "Jan 22, 2014"


@due.dcite(Doi("10.1039/b801115j"), description="Covalent radii revisited")
class CovalentRadius:
    """
    Covalent radius of the elements.

    Beatriz C. et al. Dalton Trans. 2008, 2832-2838. https://doi.org/10.1039/b801115j
    """

    radius: ClassVar = {
        "H": 0.31,
        "He": 0.28,
        "Li": 1.28,
        "Be": 0.96,
        "B": 0.84,
        "C": 0.73,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "Ne": 0.58,
        "Na": 1.66,
        "Mg": 1.41,
        "Al": 1.21,
        "Si": 1.11,
        "P": 1.07,
        "S": 1.05,
        "Cl": 1.02,
        "Ar": 1.06,
        "K": 2.03,
        "Ca": 1.76,
        "Sc": 1.70,
        "Ti": 1.60,
        "V": 1.53,
        "Cr": 1.39,
        "Mn": 1.50,
        "Fe": 1.42,
        "Co": 1.38,
        "Ni": 1.24,
        "Cu": 1.32,
        "Zn": 1.22,
        "Ga": 1.22,
        "Ge": 1.20,
        "As": 1.19,
        "Se": 1.20,
        "Br": 1.20,
        "Kr": 1.16,
        "Rb": 2.20,
        "Sr": 1.95,
        "Y": 1.90,
        "Zr": 1.75,
        "Nb": 1.64,
        "Mo": 1.54,
        "Tc": 1.47,
        "Ru": 1.46,
        "Rh": 1.42,
        "Pd": 1.39,
        "Ag": 1.45,
        "Cd": 1.44,
        "In": 1.42,
        "Sn": 1.39,
        "Sb": 1.39,
        "Te": 1.38,
        "I": 1.39,
        "Xe": 1.40,
        "Cs": 2.44,
        "Ba": 2.15,
        "La": 2.07,
        "Ce": 2.04,
        "Pr": 2.03,
        "Nd": 2.01,
        "Pm": 1.99,
        "Sm": 1.98,
        "Eu": 1.98,
        "Gd": 1.96,
        "Tb": 1.94,
        "Dy": 1.92,
        "Ho": 1.92,
        "Er": 1.89,
        "Tm": 1.90,
        "Yb": 1.87,
        "Lu": 1.87,
        "Hf": 1.75,
        "Ta": 1.70,
        "W": 1.62,
        "Re": 1.51,
        "Os": 1.44,
        "Ir": 1.41,
        "Pt": 1.36,
        "Au": 1.36,
        "Hg": 1.32,
        "Tl": 1.45,
        "Pb": 1.46,
        "Bi": 1.48,
        "Po": 1.40,
        "At": 1.50,
        "Rn": 1.50,
        "Fr": 2.60,
        "Ra": 2.21,
        "Ac": 2.15,
        "Th": 2.06,
        "Pa": 2,
        "U": 1.96,
        "Np": 1.90,
        "Pu": 1.87,
        "Am": 1.80,
        "Cm": 1.69,
    }


class MoleculeStructureComparator(MSONable):
    """Check whether the connection tables of the two molecules are the
    same. The atom in the two molecule must be paired accordingly.
    """

    ionic_element_list = (
        "Na",
        "Mg",
        "Al",
        "Sc",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Rb",
        "Sr",
    )
    halogen_list = ("F", "Cl", "Br", "I")

    def __init__(
        self,
        bond_length_cap=0.3,
        covalent_radius=CovalentRadius.radius,
        priority_bonds=(),
        priority_cap=0.8,
        ignore_ionic_bond=True,
        bond_13_cap=0.05,
    ):
        """Initialize a MoleculeStructureComparator.

        Args:
            bond_length_cap: The ratio of the elongation of the bond to be
                acknowledged. If the distance between two atoms is less than (
                empirical covalent bond length) X (1 + bond_length_cap), the bond
                between the two atoms will be acknowledged.
            covalent_radius: The covalent radius of the atoms.
                dict (element symbol -> radius)
            priority_bonds: The bonds that are known to be existed in the initial
                molecule. Such bonds will be acknowledged in a loose criteria.
                The index should start from 0.
            priority_cap: The ratio of the elongation of the bond to be
                acknowledged for the priority bonds.
            ignore_ionic_bond (bool): Whether to ignore ionic bonds.
            bond_13_cap (float): The ratio of the elongation of the 1-3 bond to be
                acknowledged.
        """
        self.bond_length_cap = bond_length_cap
        self.covalent_radius = covalent_radius
        self.priority_bonds = [tuple(sorted(b)) for b in priority_bonds]
        self.priority_cap = priority_cap
        self.ignore_ionic_bond = ignore_ionic_bond
        self.ignore_halogen_self_bond = True
        self.bond_13_cap = bond_13_cap

    def are_equal(self, mol1, mol2) -> bool:
        """
        Compare the bond table of the two molecules.

        Args:
            mol1: first molecule. pymatgen Molecule object.
            mol2: second molecules. pymatgen Molecule object.
        """
        b1 = set(self._get_bonds(mol1))
        b2 = set(self._get_bonds(mol2))
        return b1 == b2

    @staticmethod
    def get_13_bonds(priority_bonds):
        """Get the 1-3 bonds implied by the given priority bonds.

        Args:
            priority_bonds (list[tuple]): 12 bonds

        Returns:
            tuple: 13 bonds
        """
        all_bond_pairs = list(itertools.combinations(priority_bonds, r=2))
        all_2_bond_atoms = [set(b1 + b2) for b1, b2 in all_bond_pairs]
        all_13_bond_atoms = [a for a in all_2_bond_atoms if len(a) == 3]
        all_2_and_13_bonds = {
            tuple(sorted(b)) for b in itertools.chain(*(itertools.combinations(p, 2) for p in all_13_bond_atoms))
        }
        bonds_13 = all_2_and_13_bonds - {tuple(b) for b in priority_bonds}
        return tuple(sorted(bonds_13))

    def _get_bonds(self, mol):
        """Find all bonds in a molecule.

        Args:
            mol: pymatgen Molecule object.

        Returns:
            List[tuple[int, int]]: bonded atom pairs.
        """
        n_atoms = len(mol)
        if self.ignore_ionic_bond:
            covalent_atoms = [idx for idx, sp in enumerate(mol.species) if sp.symbol not in self.ionic_element_list]
        else:
            covalent_atoms = list(range(n_atoms))

        unavailable_elements = set(mol.composition.as_dict()) - set(self.covalent_radius)
        if unavailable_elements:
            raise ValueError(f"The covalent radius for element {unavailable_elements} is not available")

        coords = mol.cart_coords[covalent_atoms]

        max_radius = max(self.covalent_radius.values())
        max_cap = max(
            self.bond_length_cap,
            self.priority_cap,
            self.bond_13_cap,
        )
        max_cutoff = 2 * max_radius * (1 + max_cap)

        # Use a KDTree to avoid N^2 operations in building the pair list
        tree = cKDTree(coords)
        pairs = np.asarray(
            sorted(tree.query_pairs(max_cutoff)),
            dtype=np.int32,
        )

        if pairs.size == 0:
            return []

        atoms = np.asarray(covalent_atoms)

        pair_dists = np.linalg.norm(
            coords[pairs[:, 0]] - coords[pairs[:, 1]],
            axis=1,
        )

        bond_13 = self.get_13_bonds(self.priority_bonds)

        bonds = []

        for (i_local, j_local), dist in zip(pairs, pair_dists, strict=True):
            i = atoms[i_local]
            j = atoms[j_local]
            pair = (i, j)

            cap = self.covalent_radius[mol.sites[i].specie.symbol] + self.covalent_radius[mol.sites[j].specie.symbol]

            if pair in self.priority_bonds:
                cap *= 1 + self.priority_cap
            elif pair in bond_13:
                cap *= 1 + self.bond_13_cap
            else:
                cap *= 1 + self.bond_length_cap

            if (
                self.ignore_halogen_self_bond
                and pair not in self.priority_bonds
                and mol.sites[i].specie.symbol in self.halogen_list
                and mol.sites[j].specie.symbol in self.halogen_list
            ):
                cap *= 0.1

            if dist <= cap:
                bonds.append(tuple(int(x) for x in pair))

        return bonds

    def as_dict(self):
        """Get MSONable dict."""
        return {
            "version": __version__,
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "bond_length_cap": self.bond_length_cap,
            "covalent_radius": self.covalent_radius,
            "priority_bonds": self.priority_bonds,
            "priority_cap": self.priority_cap,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Reconstruct MoleculeStructureComparator from its MSONable dict representation.

        Args:
            dct (dict): Dict representation.

        Returns:
            MoleculeStructureComparator
        """
        return cls(
            bond_length_cap=dct["bond_length_cap"],
            covalent_radius=dct["covalent_radius"],
            priority_bonds=dct["priority_bonds"],
            priority_cap=dct["priority_cap"],
        )
