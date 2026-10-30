from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from itertools import islice
from typing import TYPE_CHECKING, cast

import numpy as np

from pymatgen.electronic_structure.core import Spin
from pymatgen.io.lobster.future.core import LobsterInteractionsHolder
from pymatgen.io.lobster.future.versioning import version_processor

if TYPE_CHECKING:
    from typing import ClassVar, Literal

    from numpy.typing import NDArray

    from pymatgen.io.lobster.future.types import LobsterInteractionData


class COXXCAR(LobsterInteractionsHolder):
    """Reader for COXXCAR-style files (COOPCAR, COHPCAR, COBICAR).

    Parses LOBSTER's COXXCAR outputs and organizes bond and orbital-resolved interaction data.

    Attributes:
        filename (PathLike): Input file path.
        num_bonds (int): Number of bond interactions reported.
        num_data (int): Number of energy/data points.
        efermi (float): Fermi energy from the file.
        spins (list[Spin]): Present spin channels.
        interactions (list[dict]): Parsed interaction metadata.
        data (np.ndarray): Raw numerical table parsed from the file.
    """

    interactions_regex: ClassVar[str] = (
        r"(?i)([a-z]+\d*(?:\_\d+)?)(?:\[(\-?\d+\s+\-?\d+\s+\-?\d+)\])?(?:\[([^]\s]*)\])?(?:\(([^)]*)\))?"
    )
    coxxcar_type: ClassVar[str]

    @property
    def energies(self) -> NDArray[np.floating]:
        """Return the energy grid.

        Returns:
            NDArray[np.floating]: Energies (first column of self.data).
        """
        return self.data[:, 0]

    def parse_header(self) -> None:
        """Parse the file header and set metadata attributes.

        Args:
            lines (list[str]): Lines of the COXXCAR file.
        """
        data = list(islice(self.iterate_lines(), 2))[1].split()

        self.num_bonds = int(data[0])
        self.num_data = int(data[2])
        self.efermi = float(data[-1])

        if int(data[1]) == 2:
            self.spins = [Spin.up, Spin.down]
        else:
            self.spins = [Spin.up]

    def parse_bonds(self) -> None:
        """Parse the bonds/interactions header block.

        Args:
            lines (list[str]): Lines of the COXXCAR file.
        """
        self.interactions = []

        self.parse_header()

        lines_generator = islice(self.iterate_lines(), 2, self.num_bonds + 2)

        for line in lines_generator:
            if "Average" in line:
                self.interactions.append(
                    {
                        "index": 0,
                        "centers": ["Average"],
                        "orbitals": [None],
                        "cells": [[]],
                        "length": None,
                    }
                )
                continue

            bond_index, bond_data = line.split(":", 1)

            if bond_regex_results := re.search(r"No\.(\d+)", bond_index):
                bond_index = bond_regex_results.group(1)
            else:
                raise ValueError(f"Could not parse bond index from line: {line}")

            length = None

            bond_tmp: dict[str, list] = defaultdict(list)

            centers = bond_data.split("->")
            for center in centers:
                if m := re.search(self.interactions_regex, center):
                    grps = m.groups()
                    bond_tmp["centers"].append(grps[0])
                else:
                    raise ValueError(f"Could not parse interaction from line: {line}")

                if grps[1]:
                    bond_tmp["cells"].append([int(x) for x in grps[1].split()])
                else:
                    bond_tmp["cells"].append([])

                bond_tmp["orbitals"].append(grps[2])

                if grps[3]:
                    length = float(grps[3])

            bond = {
                "index": int(bond_index),
                "centers": bond_tmp["centers"],
                "cells": bond_tmp["cells"],
                "orbitals": bond_tmp["orbitals"],
                "length": length,
            }

            self.interactions.append(cast("LobsterInteractionData", bond))

    def parse_data(self) -> None:
        """Parse the numerical data block into `self.data` and validate shape.

        Args:
            lines (list[str]): Lines of the COXXCAR file.

        Raises:
            ValueError: If the parsed data array shape does not match the expected shape.
        """
        self.data = np.genfromtxt(
            self.iterate_lines(),
            dtype=np.float64,
            skip_header=self.num_bonds + 2,
            loose=False,
        )

        if self.spins is None:
            raise ValueError("COXXCAR spin channels were not set before reading numerical data.")

        if self.data.shape != (self.num_data, self.num_bonds * 2 * len(self.spins) + 1):
            raise ValueError(
                f"Data shape {self.data.shape} does not match expected shape "
                f"({self.num_data}, {self.num_bonds * 2 * len(self.spins) + 1})."
            )

        self.process_data_into_interactions()

    def process_data_into_interactions(self) -> None:
        """Populate each interaction dict with 'coxx' and 'icoxx' views.

        Assigns numpy views into `self.data` for each spin channel.
        """
        if self.spins is None:
            raise ValueError("COXXCAR spin channels were not set before populating interaction data.")

        for i, interaction in enumerate(self.interactions):
            real_indices = self.interaction_indices_to_data_indices_mapping(
                i,
                spins=self.spins,
            )

            interaction["coxx"] = {}
            interaction["icoxx"] = {}

            if len(self.spins) == 1:
                interaction["coxx"][Spin.up] = self.data[:, real_indices[0]]
                interaction["icoxx"][Spin.up] = self.data[:, real_indices[1]]
            else:
                interaction["coxx"][Spin.up] = self.data[:, real_indices[0]]
                interaction["icoxx"][Spin.up] = self.data[:, real_indices[1]]
                interaction["coxx"][Spin.down] = self.data[:, real_indices[2]]
                interaction["icoxx"][Spin.down] = self.data[:, real_indices[3]]

    @version_processor(min_version="5.1")
    def parse_file(self) -> None:
        """Parse the full COXXCAR file (header and data)."""
        self.parse_bonds()
        self.parse_data()

    def get_data_indices_by_properties(
        self,
        indices: list[int] | None = None,
        centers: list[str] | None = None,
        cells: list[list[int]] | None = None,
        orbitals: list[str] | None = None,
        length: tuple[float, float] | None = None,
        spins: list[Spin] | None = None,
        data_type: Literal["coxx", "icoxx"] | None = None,
    ) -> list[int]:
        """Return data-column indices matching the provided interaction properties.

        Args:
            indices (list[int] | None): Interaction indices to filter.
            centers (list[str] | None): Atom centers to filter.
            cells (list[list[int]] | None): Unit cell indices to filter.
            orbitals (list[str] | None): Orbitals to filter.
            length (tuple[float, float] | None): Length range to filter.
            spins (list[Spin] | None): Spins to include.
            data_type (Literal["coxx", "icoxx"] | None): Restrict column type.

        Returns:
            list[int]: Sorted list of data column indices that match the filters.
        """
        effective_spins = spins if spins is not None else self.spins
        return self.interaction_indices_to_data_indices_mapping(
            sorted(
                self.get_interaction_indices_by_properties(
                    indices,
                    centers,
                    cells,
                    orbitals,
                    length,
                )
            ),
            spins=effective_spins,
            data_type=data_type,
        )

    def get_data_by_properties(
        self,
        indices: list[int] | None = None,
        centers: list[str] | None = None,
        cells: list[list[int]] | None = None,
        orbitals: list[str] | None = None,
        length: tuple[float, float] | None = None,
        spins: list[Spin] | None = None,
        data_type: Literal["coxx", "icoxx"] | None = None,
    ) -> NDArray[np.floating]:
        """Return the data columns matching the provided interaction properties.

        Args:
            indices (list[int] | None): Interaction indices to filter.
            centers (list[str] | None): Atom centers to filter.
            cells (list[list[int]] | None): Unit cell indices to filter.
            orbitals (list[str] | None): Orbitals to filter.
            length (tuple[float, float] | None): Length range to filter.
            spins (list[Spin] | None): Spins to include.
            data_type (Literal["coxx", "icoxx"] | None): Restrict column type.

        Returns:
            np.ndarray: Array with shape (n_energies, n_selected_columns).
        """
        bond_indices = self.get_interaction_indices_by_properties(indices, centers, cells, orbitals, length)
        effective_spins = spins if spins is not None else self.spins

        return self.data[
            :,
            self.interaction_indices_to_data_indices_mapping(
                bond_indices,
                spins=effective_spins,
                data_type=data_type,
            ),
        ]

    def interaction_indices_to_data_indices_mapping(
        self,
        interaction_indices: int | list[int],
        spins: Spin | int | Sequence[Spin | int] | None = None,
        data_type: Literal["coxx", "icoxx"] | None = None,
    ) -> list[int]:
        """Map interaction indices to column indices in `self.data`.

        Args:
            interaction_indices (int | list[int]): Single index or list of interaction indices.
            spins (Spin | int | Sequence[Spin | int] | None): Spin(s) to include.
            data_type (Literal["coxx", "icoxx"] | None): Select columns of that type.

        Returns:
            list[int]: Sorted list of integer column indices into `self.data`.

        Raises:
            ValueError: If an invalid Spin is requested.
        """
        if spins is None:
            spins_list = list(self.spins) if self.spins is not None else []
        elif isinstance(spins, Sequence) and not isinstance(spins, (str, bytes)):
            spins_list = []
            for s in spins:
                if isinstance(s, Spin):
                    spins_list.append(s)
                elif isinstance(s, int) and not isinstance(s, bool):
                    spins_list.append(Spin(s))
                else:
                    raise TypeError(f"Expected Spin or int spin channel, got {type(s).__name__}")
        elif isinstance(spins, int) and not isinstance(spins, bool):
            spins_list = [Spin(spins)]
        elif isinstance(spins, Spin):
            spins_list = [spins]
        else:
            raise TypeError(f"Invalid spins argument type: {type(spins).__name__}")

        if not spins_list:
            raise ValueError("No spin channels available for column mapping.")

        if isinstance(interaction_indices, int):
            interaction_indices = [interaction_indices]

        file_spins = self.spins or []
        if set(spins_list) - set(file_spins):
            raise ValueError(f"Requested `Spin` {spins_list} is not valid. Valid `Spin`s are: {file_spins}.")

        # Column layout: energy, then (coxx, icoxx) per interaction for spin up, then the same block for spin down.
        spin_offsets = {0 if spin == Spin.up else self.num_bonds for spin in spins_list}
        column_shifts = {"coxx": (1,), "icoxx": (2,)}.get(data_type or "", (1, 2))

        return sorted(
            (spin_offset + bond_index) * 2 + shift
            for bond_index in set(interaction_indices)
            if 0 <= bond_index < self.num_bonds
            for spin_offset in spin_offsets
            for shift in column_shifts
        )


class COBICAR(COXXCAR):
    """Reader for COBICAR.lobster files.

    Attributes:
        coxxcar_type (str): Type of COXXCAR file ("COBICAR").
        is_lcfo (bool): Whether the file is in LCFO format.
    """

    coxxcar_type: ClassVar[str] = "COBICAR"

    is_lcfo: ClassVar[bool] = False

    @classmethod
    def get_default_filename(cls) -> str:
        """Return the default filename for COBICAR."""
        return "COBICAR.LCFO.lobster" if cls.is_lcfo else "COBICAR.lobster"


class COHPCAR(COXXCAR):
    """Reader for COHPCAR.lobster files.

    Attributes:
        coxxcar_type (str): Type of COXXCAR file ("COHPCAR").
        is_lcfo (bool): Whether the file is in LCFO format.
    """

    coxxcar_type: ClassVar[str] = "COHPCAR"

    is_lcfo: ClassVar[bool] = False

    @classmethod
    def get_default_filename(cls) -> str:
        """Return the default filename for COOPCAR."""
        return "COHPCAR.LCFO.lobster" if cls.is_lcfo else "COHPCAR.lobster"


class COOPCAR(COXXCAR):
    """Reader for COOPCAR.lobster files.

    Attributes:
        coxxcar_type (str): Type of COXXCAR file ("COOPCAR").
    """

    coxxcar_type: ClassVar[str] = "COOPCAR"

    @classmethod
    def get_default_filename(cls) -> str:
        """Return the default filename for COOPCAR."""
        return "COOPCAR.lobster"


class COHPCAR_LCFO(COHPCAR):
    """Reader for COHPCAR.LCFO.lobster files.

    Attributes:
        is_lcfo (bool): Always True for LCFO format.
    """

    is_lcfo: ClassVar[bool] = True


class COBICAR_LCFO(COBICAR):
    """Reader for COBICAR.LCFO.lobster files.

    Attributes:
        is_lcfo (bool): Always True for LCFO format.
    """

    is_lcfo: ClassVar[bool] = True
