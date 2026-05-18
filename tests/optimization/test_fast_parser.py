from __future__ import annotations

from io import BytesIO, StringIO

import numpy as np
import pytest

from pymatgen.optimization.fast_parser import parse_N_doubles


def assert_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == np.float64
    assert expected.dtype == np.float64
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_parse_n_doubles_reads_binary_file() -> None:
    data = b"header 1.5 2.5 3.5 tail"
    file = BytesIO(data)
    file.seek(len(b"header "))
    out = np.empty(3, dtype=np.float64)

    parsed = parse_N_doubles(file, out)

    assert parsed == 3
    assert file.tell() == len(b"header 1.5 2.5 3.5")
    assert file.read(1) == b" "
    assert_exact(out, np.array([1.5, 2.5, 3.5], dtype=np.float64))


def test_parse_n_doubles_respects_nelem() -> None:
    data = b"1 2 3 4"
    file = BytesIO(data)
    out = np.empty(4, dtype=np.float64)

    parsed = parse_N_doubles(file, out, nelem=2)

    assert parsed == 2
    assert file.tell() == len(b"1 2")
    assert file.read(1) == b" "
    assert_exact(out[:parsed], np.array([1.0, 2.0], dtype=np.float64))


def test_parse_n_doubles_rejects_text_file() -> None:
    file = StringIO("1 2 3")
    out = np.empty(3, dtype=np.float64)

    with pytest.raises(TypeError):
        parse_N_doubles(file, out)


def test_parse_n_doubles_handles_values_larger_than_buffer() -> None:
    data = b" ".join(b"1.25" for _ in range(300_000)) + b" label"
    file = BytesIO(data)
    out = np.empty(300_000, dtype=np.float64)

    parsed = parse_N_doubles(file, out)

    assert parsed == 300_000
    assert file.tell() == len(data) - len(b" label")
    assert file.read(1) == b" "
    assert np.all(out == 1.25)


def test_parse_n_doubles_rejects_oversized_nelem() -> None:
    file = BytesIO(b"1 2 3")
    out = np.empty(2, dtype=np.float64)

    with pytest.raises(ValueError, match="nelem exceeds output length"):
        parse_N_doubles(file, out, nelem=3)
