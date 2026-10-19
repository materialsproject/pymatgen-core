from __future__ import annotations

import copy
import re

import numpy as np
import pytest
from monty.serialization import dumpfn
from numpy.testing import assert_allclose

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.qchem.outputs import QCOutput
from pymatgen.io.vasp.outputs import Xdatcar
from pymatgen.util.testing import TEST_FILES_DIR, VASP_IN_DIR, VASP_OUT_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/core/trajectory"


class TestTrajectory(MatSciTest):
    @classmethod
    def setup_class(cls):
        cls._xdatcar_structures = Xdatcar(f"{VASP_OUT_DIR}/XDATCAR_traj").structures

        out = QCOutput(f"{TEST_FILES_DIR}/io/qchem/new_qchem_files/ts.out")
        last_mol = out.data["molecule_from_last_geometry"]
        cls._species = last_mol.species
        cls._coords = out.data["geometries"]
        cls._charge = int(last_mol.charge)
        cls._spin = int(last_mol.spin_multiplicity)
        cls._molecules = [
            Molecule(cls._species, coord, charge=cls._charge, spin_multiplicity=cls._spin) for coord in cls._coords
        ]

    def setup_method(self):
        # Fresh per-test Trajectory objects (tests mutate via to_displacements/to_positions
        # and copy.deepcopy). Structures/molecules are not mutated so they can be shared.
        self.traj = Trajectory.from_file(f"{VASP_OUT_DIR}/XDATCAR_traj")
        self.structures = self._xdatcar_structures
        self.molecules = self._molecules
        self.traj_mols = Trajectory(
            species=self._species,
            coords=self._coords,
            charge=self._charge,
            spin_multiplicity=self._spin,
        )

    def _check_traj_equality(self, traj_1, traj_2):
        if traj_1.lattice is not None and not np.allclose(traj_1.lattice, traj_2.lattice):
            return False

        if traj_1.species != traj_2.species:
            return False

        return all(frame1 == frame2 for frame1, frame2 in zip(self.traj, traj_2, strict=False))

    def _get_lattice_species_and_coords(self):
        lattice = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        species = ["Si", "Si"]
        coords = [
            [[0, 0, 0], [0.5, 0.5, 0.5]],
            [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]],
            [[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]],
        ]
        return lattice, species, coords

    def _get_species_and_coords(self):
        species = ["C", "O"]
        coords = [
            [[1.5709474478, -0.16099953, 0.0], [1.9291378639, -1.2161950538, 0.0]],
            [[1.5688628148, -0.1548583957, 0.0], [1.9312224969, -1.2223361881, 0.0]],
            [[1.5690858055, -0.1555153055, 0.0], [1.9309995062, -1.2216792783, 0.0]],
        ]
        return species, coords, 0, 1

    def test_single_index_slice(self):
        assert all(self.traj[idx] == self.structures[idx] for idx in range(0, len(self.structures), 19))
        assert all(self.traj_mols[idx] == self.molecules[idx] for idx in range(len(self.molecules)))

    def test_slice(self):
        sliced_traj = self.traj[2:99:3]
        sliced_traj_from_structs = Trajectory.from_structures(self.structures[2:99:3])

        assert len(sliced_traj) == len(sliced_traj_from_structs), (
            f"{len(sliced_traj)=} != {len(sliced_traj_from_structs)=}"
        )
        assert all(sliced_traj[i] == sliced_traj_from_structs[i] for i in range(len(sliced_traj)))

        sliced_traj = self.traj[:-4:2]
        sliced_traj_from_structs = Trajectory.from_structures(self.structures[:-4:2])

        assert len(sliced_traj) == len(sliced_traj_from_structs), (
            f"{len(sliced_traj)=} != {len(sliced_traj_from_structs)=}"
        )
        assert all(sliced_traj[idx] == sliced_traj_from_structs[idx] for idx in range(len(sliced_traj)))

        sliced_traj = self.traj_mols[:2]
        sliced_traj_from_mols = Trajectory.from_molecules(self.molecules[:2])

        assert len(sliced_traj) == len(sliced_traj_from_mols), f"{len(sliced_traj)=} != {len(sliced_traj_from_mols)=}"
        assert all(sliced_traj[i] == sliced_traj_from_mols[i] for i in range(len(sliced_traj)))

        sliced_traj = self.traj_mols[:-2]
        sliced_traj_from_mols = Trajectory.from_molecules(self.molecules[:-2])

        assert len(sliced_traj) == len(sliced_traj_from_mols), f"{len(sliced_traj)=} != {len(sliced_traj_from_mols)=}"
        assert all(sliced_traj[i] == sliced_traj_from_mols[i] for i in range(len(sliced_traj)))

    def test_list_slice(self):
        sliced_traj = self.traj[[10, 30, 70]]
        sliced_traj_from_structs = Trajectory.from_structures([self.structures[i] for i in [10, 30, 70]])

        assert len(sliced_traj) == len(sliced_traj_from_structs), (
            f"{len(sliced_traj)=} != {len(sliced_traj_from_structs)=}"
        )
        assert all(sliced_traj[i] == sliced_traj_from_structs[i] for i in range(len(sliced_traj)))

        sliced_traj = self.traj_mols[[1, 3]]
        sliced_traj_from_mols = Trajectory.from_molecules([self.molecules[i] for i in [1, 3]])

        assert len(sliced_traj) == len(sliced_traj_from_mols), f"{len(sliced_traj)=} != {len(sliced_traj_from_mols)=}"
        assert all(sliced_traj[i] == sliced_traj_from_mols[i] for i in range(len(sliced_traj)))

    def test_conversion(self):
        # Convert to displacements and back, and then check structures.
        self.traj.to_displacements()
        self.traj.to_positions()

        assert all(struct == self.structures[idx] for idx, struct in enumerate(self.traj))

        self.traj_mols.to_displacements()
        self.traj_mols.to_positions()

        assert all(mol == self.molecules[idx] for idx, mol in enumerate(self.traj_mols))

    def test_site_properties(self):
        lattice, species, coords = self._get_lattice_species_and_coords()

        props = [
            {
                "selective_dynamics": [[True, True, True], [False, False, False]],
                "magmom": [5, 5],
            },
            {
                "selective_dynamics": [[False, False, False], [False, False, False]],
                "magmom": [6, 6],
            },
            {
                "selective_dynamics": [[True, True, True], [False, False, False]],
                "magmom": [5, 5],
            },
        ]
        traj = Trajectory(lattice=lattice, species=species, coords=coords, site_properties=props)

        # compare the overall site properties list
        assert traj.site_properties == props

        # compare the site properties after slicing
        assert traj[0].site_properties == props[0]
        assert traj[1:].site_properties == props[1:]

        species, coords, charge, spin = self._get_species_and_coords()

        props = [
            {"test": [[True, True, True], [False, False, False]]},
            {"test": [[False, False, False], [False, False, False]]},
            {"test": [[True, True, True], [False, False, False]]},
        ]
        traj = Trajectory(
            species=species,
            coords=coords,
            charge=charge,
            spin_multiplicity=spin,
            site_properties=props,
        )

        # compare the overall site properties list
        assert traj.site_properties == props

        # compare the site properties after slicing
        assert traj[0].site_properties == props[0]
        assert traj[1:].site_properties == props[1:]

    def test_frame_properties(self):
        lattice, species, coords = self._get_lattice_species_and_coords()

        props = [{"energy_per_atom": ene} for ene in [-3.0001, -3.0971, -3.0465]]

        traj = Trajectory(lattice=lattice, species=species, coords=coords, frame_properties=props)

        # compare the overall site properties
        assert traj.frame_properties == props

        # compare the site properties after slicing
        expected = props[1:]
        assert traj[1:].frame_properties == expected

        species, coords, charge, spin = self._get_species_and_coords()

        props = [
            {"SCF_energy_in_the_final_basis_set": ene} for ene in [-113.3256885788, -113.3260019471, -113.326006415]
        ]

        traj = Trajectory(
            species=species,
            coords=coords,
            charge=charge,
            spin_multiplicity=spin,
            frame_properties=props,
        )

        # compare the overall site properties
        assert traj.frame_properties == props

        # compare the site properties after slicing
        expected = props[1:]
        assert traj[1:].frame_properties == expected

        # test that the frame properties are set correctly when indexing an individual structure/molecule
        expected = props[0]
        assert traj[0].properties == expected

    def test_extend(self):
        traj = copy.deepcopy(self.traj)

        # Case of compatible trajectories
        compatible_traj = Trajectory.from_file(f"{VASP_OUT_DIR}/XDATCAR_traj_combine_test_1")
        traj.extend(compatible_traj)

        full_traj = Trajectory.from_file(f"{VASP_OUT_DIR}/XDATCAR_traj_combine_test_full")
        compatible_success = self._check_traj_equality(self.traj, full_traj)

        # Case of incompatible trajectories
        traj = copy.deepcopy(self.traj)
        incompatible_traj = Trajectory.from_file(f"{VASP_OUT_DIR}/XDATCAR_traj_combine_test_2")
        incompatible_test_success = False
        try:
            traj.extend(incompatible_traj)
        except Exception:
            incompatible_test_success = True

        assert compatible_success
        assert incompatible_test_success

        traj = copy.deepcopy(self.traj_mols)

        # Case of compatible trajectories
        compatible_traj = Trajectory(
            species=traj.species,
            coords=[
                [
                    [-1.46958173, -0.47370158, -0.03391061],
                    [-0.79757102, 0.48588802, 0.94508206],
                    [0.50256405, 0.8947604, 0.47698504],
                    [1.56101382, 0.13356272, 0.79931048],
                    [1.43897567, -0.8642765, 1.56363034],
                    [2.66882238, 0.48431336, 0.30635727],
                    [-2.72606146, -0.81552889, 0.39696593],
                    [3.307822, -1.01132269, 1.26654957],
                    [-0.81092724, -1.35590014, -0.1458541],
                    [-1.48634516, 0.02121279, -1.02465009],
                    [-0.71212347, 0.03008471, 1.93272477],
                    [-1.37888759, 1.40819443, 1.02143913],
                    [-4.79241099, 0.80275103, -0.39852432],
                    [-4.28509927, -1.03484764, 0.86348452],
                ]
            ],
            charge=0,
            spin_multiplicity=2,
        )
        traj.extend(compatible_traj)

        assert len(traj) == 5
        assert traj[-2] == traj[-1]

        # Case of incompatible trajectories
        species, coords, charge, spin = self._get_species_and_coords()

        traj = copy.deepcopy(self.traj)
        incompatible_traj = Trajectory(species=species, coords=coords, charge=charge, spin_multiplicity=spin)
        incompatible_test_success = False
        try:
            traj.extend(incompatible_traj)
        except Exception:
            incompatible_test_success = True

        assert incompatible_test_success

    def test_extend_site_props(self):
        lattice, species, coords = self._get_lattice_species_and_coords()
        n_frames = len(coords)

        props_1 = {
            "selective_dynamics": [[False, False, False], [False, False, False]],
            "magmom": [5, 5],
        }
        props_2 = {
            "selective_dynamics": [[True, True, True], [False, False, False]],
            "magmom": [5, 5],
        }
        props_3 = [
            {
                "selective_dynamics": [[False, True, True], [False, False, False]],
                "magmom": [5, 5],
            },
            {
                "selective_dynamics": [[True, False, True], [False, False, False]],
                "magmom": [5, 5],
            },
            {
                "selective_dynamics": [[True, True, False], [False, False, False]],
                "magmom": [5, 5],
            },
        ]

        traj_1 = Trajectory(lattice=lattice, species=species, coords=coords, site_properties=props_1)

        traj_2 = Trajectory(lattice=lattice, species=species, coords=coords, site_properties=props_2)

        traj_3 = Trajectory(lattice=lattice, species=species, coords=coords, site_properties=props_3)

        traj_4 = Trajectory(lattice=lattice, species=species, coords=coords, site_properties=None)

        # const & const (both constant and the same site properties)
        traj_combined = copy.deepcopy(traj_1)
        traj_combined.extend(traj_1)
        expected_site_props = props_1
        assert traj_combined.site_properties == expected_site_props

        # const & const (both constant but different site properties)
        traj_combined = copy.deepcopy(traj_1)
        traj_combined.extend(traj_2)
        expected_site_props = [props_1] * n_frames + [props_2] * n_frames
        assert traj_combined.site_properties == expected_site_props

        # const & changing
        traj_combined = copy.deepcopy(traj_1)
        traj_combined.extend(traj_3)
        expected_site_props = [props_1] * n_frames + props_3
        assert traj_combined.site_properties == expected_site_props

        # const & none
        traj_combined = copy.deepcopy(traj_1)
        traj_combined.extend(traj_4)
        expected_site_props = [props_1] * n_frames + [None] * n_frames
        assert traj_combined.site_properties == expected_site_props

        # changing & const
        traj_combined = copy.deepcopy(traj_3)
        traj_combined.extend(traj_1)
        expected_site_props = props_3 + [props_1] * n_frames
        assert traj_combined.site_properties == expected_site_props

        # changing & changing
        traj_combined = copy.deepcopy(traj_3)
        traj_combined.extend(traj_3)
        expected_site_props = props_3 + props_3
        assert traj_combined.site_properties == expected_site_props

        # changing & none
        traj_combined = copy.deepcopy(traj_3)
        traj_combined.extend(traj_4)
        expected_site_props = props_3 + [None] * n_frames
        assert traj_combined.site_properties == expected_site_props

        # none & const
        traj_combined = copy.deepcopy(traj_4)
        traj_combined.extend(traj_1)
        expected_site_props = [None] * n_frames + [props_1] * n_frames
        assert traj_combined.site_properties == expected_site_props

        # none & changing
        traj_combined = copy.deepcopy(traj_4)
        traj_combined.extend(traj_3)
        expected_site_props = [None] * n_frames + props_3
        assert traj_combined.site_properties == expected_site_props

        # none & none
        traj_combined = copy.deepcopy(traj_4)
        traj_combined.extend(traj_4)
        expected_site_props = None
        assert traj_combined.site_properties == expected_site_props

    def test_extend_frame_props(self):
        lattice, species, coords = self._get_lattice_species_and_coords()

        energy_1 = [-3, -3.9, -4.1]
        energy_2 = [-4.2, -4.25, -4.3]
        pressure_2 = [2, 2.5, 2.5]

        # energy only properties
        props_1 = [{"energy": ene} for ene in energy_1]
        traj_1 = Trajectory(lattice=lattice, species=species, coords=coords, frame_properties=props_1)

        # energy and pressure properties
        props_2 = [{"energy": e, "pressure": p} for e, p in zip(energy_2, pressure_2, strict=True)]
        traj_2 = Trajectory(lattice=lattice, species=species, coords=coords, frame_properties=props_2)

        # no properties
        traj_3 = Trajectory(lattice=lattice, species=species, coords=coords, frame_properties=None)

        # test combining two with different properties
        traj_combined = copy.deepcopy(traj_1)
        traj_combined.extend(traj_2)
        expected_props = props_1 + props_2
        assert traj_combined.frame_properties == expected_props

        # test combining two where one has properties and the other does not
        traj_combined = copy.deepcopy(traj_1)
        traj_combined.extend(traj_3)
        expected_props = props_1 + [None] * len(coords)
        assert traj_combined.frame_properties == expected_props

        # test combining two both of which have no properties
        traj_combined = copy.deepcopy(traj_3)
        traj_combined.extend(traj_3)
        assert traj_combined.frame_properties is None

    def test_length(self):
        assert len(self.traj) == len(self.structures)
        assert len(self.traj_mols) == len(self.molecules)

    def test_displacements(self):
        structures = [Structure.from_file(f"{VASP_IN_DIR}/POSCAR")]
        displacements = np.zeros((11, *np.shape(structures[-1].frac_coords)))

        rng = np.random.default_rng()
        for idx in range(10):
            displacement = rng.random(np.shape(structures[-1].frac_coords)) / 20
            new_coords = displacement + structures[-1].frac_coords
            structures.append(Structure(structures[-1].lattice, structures[-1].species, new_coords))
            displacements[idx + 1, :, :] = displacement

        traj = Trajectory.from_structures(structures, constant_lattice=True)
        traj.to_displacements()

        assert_allclose(traj.coords, displacements)

    def test_variable_lattice(self):
        structure = self.structures[0]

        # Generate structures with different lattices
        structures = []
        rng = np.random.default_rng()
        for _ in range(10):
            new_lattice = np.dot(structure.lattice.matrix, np.diag(1 + rng.random(3) / 20))
            temp_struct = structure.copy()
            temp_struct.lattice = Lattice(new_lattice)
            structures.append(temp_struct)

        traj = Trajectory.from_structures(structures, constant_lattice=False)

        # Check if lattices were properly stored
        assert all(
            np.allclose(struct.lattice.matrix, structures[idx].lattice.matrix) for idx, struct in enumerate(traj)
        )

        # Check if the file is written correctly when lattice is not constant.
        traj.write_Xdatcar(filename=f"{self.tmp_path}/traj_test_XDATCAR")

        # Load trajectory from written XDATCAR and compare to original
        written_traj = Trajectory.from_file(f"{self.tmp_path}/traj_test_XDATCAR", constant_lattice=False)
        self._check_traj_equality(traj, written_traj)

    def test_as_from_dict(self):
        dct = self.traj.as_dict()
        traj = Trajectory.from_dict(dct)
        assert isinstance(traj, Trajectory)

        dct = self.traj_mols.as_dict()
        traj = Trajectory.from_dict(dct)
        assert isinstance(traj, Trajectory)

    def test_xdatcar_write(self):
        self.traj.write_Xdatcar(filename=f"{self.tmp_path}/traj_test_XDATCAR")

        # Load trajectory from written XDATCAR and compare to original
        written_traj = Trajectory.from_file(f"{self.tmp_path}/traj_test_XDATCAR")
        self._check_traj_equality(self.traj, written_traj)

    def test_from_file(self):
        try:
            traj = Trajectory.from_file(f"{TEST_DIR}/LiMnO2_chgnet_relax.traj")
            assert isinstance(traj, Trajectory)

            # Check length of the trajectory
            assert len(traj) == 2

            # Check composition of the first frame of the trajectory
            assert traj[0].formula == "Li2 Mn2 O4"

            # Check that ASE calculator properties are converted to frame properties
            assert all(
                all(frame.get(k) is not None for k in ("energy", "forces", "stress")) for frame in traj.frame_properties
            )

        except ImportError:
            with pytest.raises(
                ImportError,
                match=re.escape("ASE is required to read .traj files. pip install ase"),
            ):
                Trajectory.from_file(f"{TEST_DIR}/LiMnO2_chgnet_relax.traj")

    def test_index_error(self):
        with pytest.raises(IndexError, match="index=100 out of range, trajectory only has 100 frames"):
            self.traj[100]
        with pytest.raises(
            TypeError,
            match=re.escape("bad index='test', expected one of [int, slice, list[int], numpy.ndarray]"),
        ):
            self.traj["test"]

    def test_incorrect_dims(self):
        # Good Inputs
        const_lattice = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        species = ["C", "O"]
        coords = [
            [[1.5, -0, 0], [1.9, -1.2, 0]],
            [[1.5, -0, 0], [1.9, -1.2, 0]],
            [[1.5, -0, 0], [1.9, -1.2, 0]],
        ]

        # Problematic Inputs
        short_lattice = [
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ]
        unphysical_lattice = [
            ((1, 0, 0), (0, 1, 0)),
            ((1, 0, 0), (0, 1, 0)),
            ((1, 0, 0), (0, 1, 0)),
        ]
        extra_coords = [
            [[1.5, -0, 0], [1.9, -1.2, 0], [1.9, -1.2, 0]],
            [[1.5, -0, 0], [1.9, -1.2, 0], [1.9, -1.2, 0]],
            [[1.5, -0, 0], [1.9, -1.2, 0], [1.9, -1.2, 0]],
        ]
        unphysical_coords = [
            [[1.5, -0, 0, 1], [1.9, -1.2, 0, 1]],
            [[1.5, -0, 0, 1], [1.9, -1.2, 0, 1]],
            [[1.5, -0, 0, 1], [1.9, -1.2, 0, 1]],
        ]
        wrong_dim_coords = [[1.5, -0, 0, 1], [1.9, -1.2, 0, 1]]

        with pytest.raises(ValueError, match=re.escape("lattice must have shape (M, 3, 3)!")):
            Trajectory(species=species, coords=coords, lattice=short_lattice)
        with pytest.raises(ValueError, match=re.escape("lattice must have shape (3, 3) or (M, 3, 3)")):
            Trajectory(species=species, coords=coords, lattice=unphysical_lattice)
        with pytest.raises(ValueError, match="must have the same number of sites!"):
            Trajectory(species=species, coords=extra_coords, lattice=const_lattice)
        with pytest.raises(ValueError, match=re.escape("coords must have shape (M, N, 3)")):
            Trajectory(species=species, coords=unphysical_coords, lattice=const_lattice)
        with pytest.raises(ValueError, match="coords must have 3 dimensions!"):
            Trajectory(species=species, coords=wrong_dim_coords, lattice=const_lattice)

    def test_to_ase_traj(self):
        traj = Trajectory.from_file(f"{TEST_DIR}/LiMnO2_chgnet_relax.json.gz")

        try:
            ase_traj = traj.to_ase()

            assert len(ase_traj) == len(traj)

            # Ensure all frame properties and the magmoms are populated correctly
            assert all(
                all(atoms.calc.get_property(k) is not None for k in ("energy", "forces", "stress", "magmoms"))
                for atoms in ase_traj
            )

        except ImportError:
            with pytest.raises(
                ImportError,
                match=re.escape("ASE is required to write .traj files. pip install ase"),
            ):
                ase_traj = traj.to_ase()


LATTICE = np.eye(3) * 3.0
SI_SPECIES = ["Si", "Si"]
SI_COORDS = [
    [[0, 0, 0], [0.5, 0.5, 0.5]],
    [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]],
    [[0.2, 0.2, 0.2], [0.7, 0.7, 0.7]],
]
CO_SPECIES = ["C", "O"]
CO_COORDS = [
    [[0, 0, 0], [0, 0, 1.1]],
    [[0, 0, 0], [0, 0, 1.2]],
    [[0, 0, 0], [0, 0, 1.3]],
]


@pytest.fixture
def struct_traj() -> Trajectory:
    return Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=LATTICE)


@pytest.fixture
def mol_traj() -> Trajectory:
    return Trajectory(species=CO_SPECIES, coords=CO_COORDS, charge=0, spin_multiplicity=1)


class TestTrajectoryInit:
    def test_molecule_requires_charge(self):
        with pytest.raises(ValueError, match="charge must be provided for a Molecule-based Trajectory"):
            Trajectory(species=CO_SPECIES, coords=CO_COORDS)

    def test_molecule_without_spin(self):
        traj = Trajectory(species=CO_SPECIES, coords=CO_COORDS, charge=0)
        assert traj.charge == 0
        assert traj.spin_multiplicity is None
        assert traj.lattice is None
        assert traj.constant_lattice is None
        mol = traj[1]
        assert isinstance(mol, Molecule)
        assert mol.spin_multiplicity == 1
        assert_allclose(mol.cart_coords, CO_COORDS[1])

    def test_lattice_object(self):
        traj = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=Lattice(LATTICE))
        assert isinstance(traj.lattice, np.ndarray)
        assert_allclose(traj.lattice, LATTICE)
        assert traj[2].lattice == Lattice(LATTICE)

    def test_list_of_lattice_objects(self):
        lattices = [Lattice.cubic(3.0 + 0.1 * idx) for idx in range(len(SI_COORDS))]
        traj = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=lattices, constant_lattice=False)
        assert traj.lattice.shape == (3, 3, 3)
        for idx, struct in enumerate(traj):
            assert struct.lattice.a == pytest.approx(3.0 + 0.1 * idx)

    def test_single_lattice_not_constant_warns(self):
        with pytest.warns(UserWarning, match="Get constant_lattice=False, but only get a single lattice"):
            traj = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=LATTICE, constant_lattice=False)
        assert traj.lattice.shape == (len(SI_COORDS), 3, 3)
        assert_allclose(traj.lattice[-1], LATTICE)
        assert traj.constant_lattice is False

    def test_displacements_without_base_positions_warns(self):
        with pytest.warns(UserWarning, match="Without providing an array of starting positions"):
            traj = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=LATTICE, coords_are_displacement=True)
        assert traj.base_positions is None
        assert traj.coords_are_displacement

    def test_displacements_with_base_positions(self):
        base = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        disp = np.array([np.zeros((2, 3)), np.full((2, 3), 0.1), np.full((2, 3), 0.1)])
        traj = Trajectory(
            species=SI_SPECIES,
            coords=disp,
            lattice=LATTICE,
            coords_are_displacement=True,
            base_positions=base,
        )
        assert_allclose(traj.base_positions, base)
        # Indexing converts displacements back to positions
        assert_allclose(traj[2].frac_coords, SI_COORDS[2])
        assert not traj.coords_are_displacement
        assert_allclose(traj.coords, SI_COORDS)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"site_properties": [{"magmom": [1, 1]}] * 2},
                "Size of the site properties 2 does not equal the number of frames 3",
            ),
            (
                {"site_properties": {"magmom": [1, 1, 1]}},
                r"Size of site property magmom 3\) does not equal the number of sites in the structure 2",
            ),
            (
                {"site_properties": [{"magmom": [1, 1]}, {"magmom": [1, 1]}, {"magmom": [1]}]},
                r"Size of site property magmom 1\) does not equal the number of sites in the structure 2",
            ),
            (
                {"frame_properties": [{"energy": -1.0}]},
                "Size of the frame properties 1 does not equal the number of frames 3",
            ),
        ],
    )
    def test_invalid_props(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=LATTICE, **kwargs)


class TestTrajectoryMethods(MatSciTest):
    def test_constant_site_properties_dict(self):
        props = {"magmom": [2, 2]}
        traj = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=LATTICE, site_properties=props)
        assert traj[1].site_properties == props
        sliced = traj[[0, 2]]
        assert sliced.site_properties == props
        assert len(sliced) == 2

    def test_get_site_props_errors(self):
        props = [{"magmom": [idx, idx]} for idx in range(3)]
        traj = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=LATTICE, site_properties=props)
        assert traj._get_site_props([0, 2]) == [props[0], props[2]]
        with pytest.raises(ValueError, match="Unexpected frames type"):
            traj._get_site_props(slice(0, 2))

        traj.site_properties = ("not", "valid")
        with pytest.raises(ValueError, match="Unexpected site_properties type"):
            traj._get_site_props(0)

    def test_list_index_out_of_range(self, struct_traj):
        with pytest.raises(IndexError, match=re.escape("index=[5] out of range, trajectory only has 3 frames")):
            struct_traj[[0, 5]]

    def test_list_index_equal_to_length_message(self, struct_traj):
        with pytest.raises(IndexError, match=re.escape("index=[3] out of range")):
            struct_traj[[0, 3]]

    def test_ndarray_index(self, struct_traj):
        sliced = struct_traj[np.array([2, 0])]
        assert isinstance(sliced, Trajectory)
        assert_allclose(sliced.coords, np.array(SI_COORDS)[[2, 0]])

    def test_get_structure(self, struct_traj, mol_traj):
        struct = struct_traj.get_structure(1)
        assert isinstance(struct, Structure)
        assert_allclose(struct.frac_coords, SI_COORDS[1])
        with pytest.raises(TypeError, match="Cannot return Structure for Molecule-based Trajectory"):
            mol_traj.get_structure(0)

    def test_get_molecule(self, struct_traj, mol_traj):
        mol = mol_traj.get_molecule(2)
        assert isinstance(mol, Molecule)
        assert_allclose(mol.cart_coords, CO_COORDS[2])
        with pytest.raises(TypeError, match="Cannot return Molecule for Structure-based Trajectory"):
            struct_traj.get_molecule(0)

    def test_to_displacements_idempotent(self, struct_traj):
        struct_traj.to_displacements()
        disp = struct_traj.coords.copy()
        assert_allclose(disp[1], np.full((2, 3), 0.1))
        struct_traj.to_displacements()
        assert_allclose(struct_traj.coords, disp)
        # Positions are reconstructed when indexing
        assert_allclose(struct_traj[2].frac_coords, SI_COORDS[2])

    def test_to_displacements_molecule_no_pbc_wrap(self):
        coords = [[[0, 0, 0], [0, 0, 1.0]], [[0, 0, 0], [0, 0, 1.9]]]
        traj = Trajectory(species=CO_SPECIES, coords=coords, charge=0)
        traj.to_displacements()
        # No rounding for non-periodic trajectories
        assert_allclose(traj.coords[1], [[0, 0, 0], [0, 0, 0.9]])

    def test_extend_time_step_mismatch(self):
        traj_1 = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=LATTICE, time_step=1.0)
        traj_2 = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=LATTICE, time_step=2.0)
        with pytest.raises(ValueError, match=re.escape("Time steps of the trajectories are incompatible: 1.0 and 2.0")):
            traj_1.extend(traj_2)

    def test_extend_species_mismatch(self, struct_traj):
        other = Trajectory(species=["Ge", "Ge"], coords=SI_COORDS, lattice=LATTICE)
        with pytest.raises(ValueError, match="Species in the trajectories are incompatible"):
            struct_traj.extend(other)

    def test_extend_molecule_with_structure(self, struct_traj, mol_traj):
        with pytest.raises(ValueError, match="Cannot combine Molecule- and Structure-based Trajectory"):
            mol_traj.extend(struct_traj)

    @pytest.mark.parametrize("variable_first", [True, False])
    def test_extend_variable_lattice(self, variable_first):
        lattices = np.array([np.eye(3) * (4.0 + 0.1 * idx) for idx in range(len(SI_COORDS))])
        const = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=LATTICE)
        variable = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=lattices, constant_lattice=False)
        first, second = (variable, const) if variable_first else (const, variable)
        first.extend(second)

        tiled = np.tile(LATTICE, (len(SI_COORDS), 1, 1))
        expected = np.concatenate([lattices, tiled] if variable_first else [tiled, lattices])
        assert len(first) == 2 * len(SI_COORDS)
        assert first.constant_lattice is False
        assert_allclose(first.lattice, expected)
        assert first[4].lattice.a == pytest.approx(expected[4][0][0])

    def test_extend_frame_props_none_first(self):
        props = [{"energy": float(idx)} for idx in range(3)]
        traj_1 = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=LATTICE)
        traj_2 = Trajectory(species=SI_SPECIES, coords=SI_COORDS, lattice=LATTICE, frame_properties=props)
        traj_1.extend(traj_2)
        assert traj_1.frame_properties == [None] * 3 + props

    @pytest.mark.parametrize(("prop1", "prop2", "bad"), [("bad", None, "prop1"), (None, 42, "prop2")])
    def test_combine_site_props_invalid(self, prop1, prop2, bad):
        with pytest.raises(ValueError, match=f"{bad} should be None, list or dict"):
            Trajectory._combine_site_props(prop1, prop2, 1, 1)

    def test_write_xdatcar_molecule(self, mol_traj):
        with pytest.raises(TypeError, match="write_Xdatcar can only be used with Structure-based Trajectory"):
            mol_traj.write_Xdatcar(f"{self.tmp_path}/XDATCAR")

    def test_write_xdatcar_options(self, struct_traj):
        filename = f"{self.tmp_path}/XDATCAR"
        struct_traj.to_displacements()
        struct_traj.write_Xdatcar(filename, system="my Si", significant_figures=3)
        with open(filename, encoding="utf-8") as file:
            lines = file.read().splitlines()
        assert lines[0] == "my Si"
        assert lines[7] == "Direct configuration=     1"
        assert lines[-1] == "0.700 0.700 0.700 Si"
        assert sum(line.startswith("Direct configuration") for line in lines) == 3

        written = Trajectory.from_file(filename)
        assert_allclose(written.coords, SI_COORDS, atol=1e-3)

    def test_json_round_trip(self, struct_traj):
        filename = f"{self.tmp_path}/traj.json"
        struct_traj.frame_properties = [{"energy": float(idx)} for idx in range(3)]
        dumpfn(struct_traj, filename)
        loaded = Trajectory.from_file(filename)
        assert isinstance(loaded, Trajectory)
        assert_allclose(loaded.coords, SI_COORDS)
        assert_allclose(loaded.lattice, LATTICE)
        assert loaded.frame_properties == struct_traj.frame_properties

    def test_from_file_unsupported(self):
        with pytest.raises(ValueError, match="Expect file to be one of"):
            Trajectory.from_file(f"{self.tmp_path}/POSCAR")

    def test_from_file_vasprun(self):
        traj = Trajectory.from_file(f"{VASP_OUT_DIR}/vasprun.etest1.xml.gz")
        assert len(traj) > 0

    def test_from_file_traj_without_ase(self, monkeypatch):
        monkeypatch.setattr("pymatgen.core.trajectory.NO_ASE_ERR", ImportError("no ase"))
        with pytest.raises(ImportError, match=re.escape("ASE is required to read .traj files")):
            Trajectory.from_file(f"{TEST_DIR}/LiMnO2_chgnet_relax.traj")

    def test_to_ase_without_ase(self, monkeypatch, struct_traj):
        monkeypatch.setattr("pymatgen.core.trajectory.NO_ASE_ERR", ImportError("no ase"))
        with pytest.raises(ImportError, match=re.escape("ASE is required to write .traj files")):
            struct_traj.to_ase()


def _write_ase_traj(filename, pbc: bool, scales=(3.0, 3.1), calc: bool = True) -> None:
    ase_traj = pytest.importorskip("ase.io.trajectory")
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    with ase_traj.Trajectory(filename, "w") as traj:
        for idx, scale in enumerate(scales):
            if pbc:
                atoms = Atoms(
                    "Si2",
                    scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
                    cell=np.eye(3) * scale,
                    pbc=True,
                    velocities=np.full((2, 3), 0.01),
                )
            else:
                atoms = Atoms("CO", positions=[[0, 0, 0], [0, 0, scale / 2.5]])
            if calc:
                atoms.calc = SinglePointCalculator(
                    atoms, energy=-1.0 - idx, forces=np.zeros((2, 3)), stress=np.zeros(6)
                )
            traj.write(atoms)


class TestTrajectoryAse(MatSciTest):
    def setup_method(self):
        pytest.importorskip("ase")

    def test_from_ase_additional_fields(self):
        filename = f"{self.tmp_path}/si.traj"
        _write_ase_traj(filename, pbc=True)
        traj = Trajectory.from_ase(filename, constant_lattice=False)

        assert len(traj) == 2
        assert traj.constant_lattice is False
        assert traj[1].lattice.a == pytest.approx(3.1)
        assert [frame["energy"] for frame in traj.frame_properties] == [-1.0, -2.0]
        assert all(frame["temperature"] > 0 for frame in traj.frame_properties)
        assert_allclose(traj[0].site_properties["velocities"], np.full((2, 3), 0.01))

    def test_from_ase_property_map(self):
        filename = f"{self.tmp_path}/si.traj"
        _write_ase_traj(filename, pbc=True, scales=(3.0, 3.0))
        traj = Trajectory.from_ase(filename, property_map={"energy": "e_0_energy"}, additional_fields=None)

        assert traj.constant_lattice is True
        assert traj.frame_properties == [{"e_0_energy": -1.0}, {"e_0_energy": -2.0}]

    def test_from_ase_molecule(self):
        filename = f"{self.tmp_path}/co.traj"
        _write_ase_traj(filename, pbc=False)
        traj = Trajectory.from_ase(filename, constant_lattice=True, additional_fields=None)

        assert traj.lattice is None
        assert isinstance(traj[0], Molecule)
        assert traj[1][1].coords[2] == pytest.approx(3.1 / 2.5)
        assert [frame["energy"] for frame in traj.frame_properties] == [-1.0, -2.0]

    def test_from_ase_infer_variable_lattice(self):
        filename = f"{self.tmp_path}/si.traj"
        _write_ase_traj(filename, pbc=True)
        traj = Trajectory.from_ase(filename)
        assert traj.constant_lattice is False

    def test_from_ase_molecule_infer_lattice(self):
        filename = f"{self.tmp_path}/co.traj"
        _write_ase_traj(filename, pbc=False)
        traj = Trajectory.from_ase(filename, additional_fields=None)
        assert traj.lattice is None

    def test_from_ase_without_calculator(self):
        filename = f"{self.tmp_path}/si.traj"
        _write_ase_traj(filename, pbc=True, calc=False)
        traj = Trajectory.from_ase(filename, constant_lattice=False)
        assert len(traj) == 2

    def test_to_ase_round_trip(self):
        traj = Trajectory(
            species=SI_SPECIES,
            coords=SI_COORDS,
            lattice=LATTICE,
            site_properties={"final_magmom": [1.0, -1.0]},
            frame_properties=[
                {"energy": -1.0 * idx, "forces": np.zeros((2, 3)), "stress": np.zeros(6)} for idx in range(3)
            ],
        )
        ase_file = f"{self.tmp_path}/out.traj"
        ase_traj = traj.to_ase(ase_traj_file=ase_file)

        assert len(ase_traj) == 3
        assert [atoms.get_potential_energy() for atoms in ase_traj] == [0.0, -1.0, -2.0]
        assert_allclose(ase_traj[1].calc.get_property("magmoms"), [1.0, -1.0])
        assert_allclose(ase_traj[2].get_scaled_positions(), SI_COORDS[2])

        roundtrip = Trajectory.from_file(ase_file)
        assert_allclose(roundtrip.coords, SI_COORDS)
        assert [frame["energy"] for frame in roundtrip.frame_properties] == [0.0, -1.0, -2.0]

    def test_to_ase_without_magmoms(self, struct_traj):
        ase_traj = struct_traj.to_ase(ase_traj_file=f"{self.tmp_path}/out.traj")
        assert len(ase_traj) == 3

    def test_to_ase_unrecognized_property(self, struct_traj):
        with pytest.raises(ValueError, match="Unrecognized ASE calculator properties:\nnot_a_property"):
            struct_traj.to_ase(property_map={"not_a_property": "energy"})
