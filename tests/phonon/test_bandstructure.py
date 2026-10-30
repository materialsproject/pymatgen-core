from __future__ import annotations

import copy

import numpy as np
import orjson
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx

from pymatgen.core import Lattice, Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine, estimate_band_connection
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/electronic_structure/bandstructure"


class TestPhononBandStructureSymmLine(MatSciTest):
    def setup_method(self):
        with open(f"{TEST_DIR}/NaCl_phonon_bandstructure.json", "rb") as file:
            dct = orjson.loads(file.read())
        self.bs = PhononBandStructureSymmLine.from_dict(dct)

        with open(f"{TEST_DIR}/Si_phonon_bandstructure.json", "rb") as file:
            dct = orjson.loads(file.read())
        self.bs2 = PhononBandStructureSymmLine.from_dict(dct)

    def test_repr(self):
        assert repr(self.bs) == "PhononBandStructureSymmLine(bands=(6, 204), labels=['Gamma', 'X', 'Y', 'Z'])"
        assert repr(self.bs2) == (
            r"PhononBandStructureSymmLine(bands=(6, 130), labels=['$\\Gamma$', 'X', 'W', 'K', 'L', 'U'])"
        )

    def test_eq(self):
        assert self.bs == self.bs
        assert self.bs == copy.deepcopy(self.bs)
        assert self.bs2 == self.bs2
        assert self.bs != self.bs2

    def test_basic(self):
        assert self.bs.bands[1][10] == approx(0.7753555184)
        assert self.bs.bands[5][100] == approx(5.2548379776)
        assert_array_equal(self.bs.bands.shape, (6, 204))
        assert_array_equal(self.bs.eigendisplacements.shape, (6, 204, 2, 3))
        assert_allclose(
            self.bs.eigendisplacements[3][50][0],
            [0.0 + 0.0j, 0.14166569 + 0.04098339j, -0.14166569 - 0.04098339j],
        )
        assert self.bs.has_eigendisplacements

        assert_allclose(self.bs.asr_breaking(), [-0.0370089502, -0.0370089502, -0.0221388897])

        assert self.bs.nb_bands == 6
        assert self.bs.nb_qpoints == 204

        assert_allclose(self.bs.qpoints[1].frac_coords, [0.01, 0, 0])

    def test_has_imaginary_freq(self):
        for tol in (0, 0.01, 0.02, 0.03, 0.04, 0.05):
            assert self.bs.has_imaginary_freq(tol=tol) == (tol < 0.04), f"{tol=}"

        for tol in (0, 0.01, 0.02, 0.03, 0.04, 0.05):
            assert self.bs2.has_imaginary_freq(tol=tol) == (tol < 0.01), f"{tol=}"

        # test Gamma point imaginary frequency detection
        for tol in (0, 0.01, 0.02, 0.03, 0.04, 0.05):
            assert self.bs.has_imaginary_gamma_freq(tol=tol) == (tol in (0.01, 0.02, 0.03)), f"{tol=}"

        for tol in (0, 0.01, 0.02, 0.03, 0.04, 0.05, 1, 10):
            assert self.bs2.has_imaginary_gamma_freq(tol=tol) is False, f"{tol=}"

    def test_has_imaginary_gamma_freq_repeated_gamma(self):
        # Gamma appears twice on the path; only the second occurrence has an imaginary mode.
        # Each occurrence must be checked against its own frequencies.
        qpoints = [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.25, 0, 0], [0, 0, 0]]
        frequencies = np.ones((3, len(qpoints)))
        frequencies[0, -1] = -1
        bs = PhononBandStructureSymmLine(
            qpoints,
            frequencies,
            Lattice.cubic(4).reciprocal_lattice,
            labels_dict={"Gamma": [0, 0, 0], "X": [0.5, 0, 0]},
        )
        assert bs.has_imaginary_gamma_freq() is True
        frequencies[0, -1] = 1
        assert bs.has_imaginary_gamma_freq() is False

    def test_labels_and_distance(self):
        assert_allclose(
            [self.bs.distance[idx] for idx in (0, 50, 51, 52, -1)],
            [0, 0.15219323466, 0.15219323466, 0.15523709935, 0.63231731398],
        )
        labelled = [(idx, q_pt.label) for idx, q_pt in enumerate(self.bs.qpoints) if q_pt.label]
        assert labelled == [
            (0, "Gamma"),
            (50, "X"),
            (51, "X"),
            (101, "Y"),
            (102, "Y"),
            (152, "Gamma"),
            (153, "Gamma"),
            (203, "Z"),
        ]
        assert_allclose(
            [direction for direction, _ in self.bs.nac_frequencies],
            [[1, 0, 0], [0.5**0.5, 0.5**0.5, 0], [3**-0.5, 3**-0.5, 3**-0.5]],
        )

        # two labels on the same qpoint: both end up in labels_dict, the qpoint takes the last one
        qpoints = [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [0.5, 0.25, 0], [0.5, 0.5, 0]]
        labels = {"G": [0, 0, 0], "X": [0.5, 0, 0], "X'": [0.5, 0, 0], "M": [0.5, 0.5, 0]}
        bs = PhononBandStructureSymmLine(
            qpoints, np.zeros((3, 6)), Lattice.cubic(4).reciprocal_lattice, labels_dict=labels
        )
        assert [q_pt.label for q_pt in bs.qpoints] == ["G", None, "X'", "X'", None, "M"]
        assert list(bs.labels_dict) == ["G", "X", "X'", "M"]
        assert_allclose(bs.distance, np.pi / 8 * np.array([0, 1, 2, 2, 3, 4]))
        assert bs.branches == [
            {"start_index": 0, "end_index": 2, "name": "G-X'"},
            {"start_index": 3, "end_index": 5, "name": "X'-M"},
        ]

    def test_estimate_band_connection(self):
        # ties go to the highest band index, taken bands are skipped
        assert estimate_band_connection(np.eye(3), np.ones((3, 3)), [0, 1, 2]) == [2, 1, 0]
        # no positive overlap left falls back to band 0
        assert estimate_band_connection(np.eye(3), np.zeros((3, 3)), [0, 1, 2]) == [0, 0, 0]
        permutation = np.eye(3)[[2, 0, 1]]
        assert estimate_band_connection(np.eye(3), permutation, [0, 1, 2]) == [2, 0, 1]
        assert estimate_band_connection(np.eye(3), permutation, [2, 0, 1]) == [1, 2, 0]

    def test_nac(self):
        assert self.bs.has_nac
        assert not self.bs2.has_nac
        assert self.bs.get_nac_frequencies_along_dir([1, 1, 0])[3] == approx(4.6084532143)
        assert self.bs.get_nac_frequencies_along_dir([0, 1, 1]) is None
        assert self.bs2.get_nac_frequencies_along_dir([0, 0, 1]) is None
        assert_allclose(
            self.bs.get_nac_eigendisplacements_along_dir([1, 1, 0])[3][1],
            [(0.1063906409128248 + 0j), 0j, 0j],
        )
        assert self.bs.get_nac_eigendisplacements_along_dir([0, 1, 1]) is None
        assert self.bs2.get_nac_eigendisplacements_along_dir([0, 0, 1]) is None

    def test_branches(self):
        assert self.bs.branches[0]["end_index"] == 50
        assert self.bs.branches[1]["start_index"] == 51
        assert self.bs.branches[2]["name"] == "Y-Gamma"
        assert self.bs.get_branch(10)[0]["name"] == "Gamma-X"
        assert len(self.bs.branches) == 4

    def test_dict_methods(self):
        dct = self.bs.as_dict()
        assert dct is not None
        assert orjson.dumps(dct).decode() is not None
        dct = self.bs2.as_dict()
        assert dct is not None
        assert orjson.dumps(dct).decode() is not None
        dct = self.bs2.as_phononwebsite()
        assert dct is not None
        assert orjson.dumps(dct, option=orjson.OPT_SERIALIZE_NUMPY).decode() is not None
        self.assert_msonable(self.bs)
        self.assert_msonable(self.bs2)

    def test_write_methods(self):
        self.bs2.write_phononwebsite(f"{self.tmp_path}/test.json")

    def test_min_max_freq(self):
        min_qpoint, min_freq = self.bs.min_freq()
        assert isinstance(min_qpoint, Kpoint)
        assert list(min_qpoint.frac_coords) == [0, 0, 0]
        assert min_freq == approx(-0.03700895020)

        max_qpoint, max_freq = self.bs.max_freq()
        assert isinstance(max_qpoint, Kpoint)
        assert list(max_qpoint.frac_coords) == [0, 0, 0]
        assert max_freq == approx(7.391425798)

        min_qpoint2, min_freq2 = self.bs2.min_freq()
        assert list(min_qpoint2.frac_coords) == [0, 0, 0]
        assert min_freq2 == approx(-0.0072257889)

        max_qpoint2, max_freq2 = self.bs2.max_freq()
        assert list(max_qpoint2.frac_coords) == [0, 0, 0]
        assert max_freq2 == approx(15.2873634264)

    def test_band_reorder(self):
        # one atom, three modes polarized along x, y, z. The x and y branches cross, so
        # sorting frequencies at each q-point (as phonopy does) swaps them halfway.
        n_q = 6
        freq_x, freq_y, freq_z = np.linspace(1, 3, n_q), np.linspace(3, 1, n_q), np.full(n_q, 5.0)
        frequencies = np.sort([freq_x, freq_y, freq_z], axis=0)
        polarization = np.eye(3)
        eigendisplacements = np.zeros((3, n_q, 1, 3), dtype=complex)
        for q_idx in range(n_q):
            x_first = freq_x[q_idx] < freq_y[q_idx]
            eigendisplacements[0, q_idx, 0] = polarization[0 if x_first else 1]
            eigendisplacements[1, q_idx, 0] = polarization[1 if x_first else 0]
            eigendisplacements[2, q_idx, 0] = polarization[2]

        lattice = Lattice.cubic(5).reciprocal_lattice
        bs = PhononBandStructureSymmLine(
            np.linspace([0, 0, 0], [0.5, 0, 0], n_q),
            frequencies,
            lattice,
            eigendisplacements=eigendisplacements,
            labels_dict={"Gamma": [0, 0, 0], "X": [0.5, 0, 0]},
            structure=Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]]),
        )
        bs.band_reorder()

        assert_allclose(bs.bands, [freq_x, freq_y, freq_z])
        for band_idx in range(3):
            assert_allclose(np.abs(bs.eigendisplacements[band_idx, :, 0]), np.tile(polarization[band_idx], (n_q, 1)))

    def test_get_gamma_point(self):
        for bs in (self.bs, self.bs2):
            g_point = bs.get_gamma_point()
            assert isinstance(g_point, Kpoint)
            assert list(g_point.frac_coords) == [0, 0, 0]
            assert g_point.label in ("Gamma", "$\\Gamma$")

    def test_width(self):
        assert self.bs.width() == approx(7.3227137833)
        assert self.bs2.width() == approx(14.7108925878)

        assert self.bs.width(with_imaginary=True) == approx(7.4284347482)
        assert self.bs2.width(with_imaginary=True) == approx(15.2945892153)
