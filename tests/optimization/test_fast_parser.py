from __future__ import annotations

from io import BytesIO, StringIO

import numpy as np
import pytest

from pymatgen.optimization.fast_parser import parse_n_doubles


def assert_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == np.float64
    assert expected.dtype == np.float64
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_parse_n_doubles_reads_binary_file() -> None:
    data = b"header 1.5 2.5 3.5 tail"
    file = BytesIO(data)
    file.seek(len(b"header "))
    out = np.empty(3, dtype=np.float64)

    parsed = parse_n_doubles(file, out)

    assert parsed == 3
    assert file.tell() == len(b"header 1.5 2.5 3.5")
    assert file.read(1) == b" "
    assert_exact(out, np.array([1.5, 2.5, 3.5], dtype=np.float64))


def test_parse_n_doubles_respects_nelem() -> None:
    data = b"1 2 3 4"
    file = BytesIO(data)
    out = np.empty(4, dtype=np.float64)

    parsed = parse_n_doubles(file, out, nelem=2)

    assert parsed == 2
    assert file.tell() == len(b"1 2")
    assert file.read(1) == b" "
    assert_exact(out[:parsed], np.array([1.0, 2.0], dtype=np.float64))


def test_parse_n_doubles_rejects_text_file() -> None:
    file = StringIO("1 2 3")
    out = np.empty(3, dtype=np.float64)

    with pytest.raises(TypeError, match="binary file"):
        parse_n_doubles(file, out)

    # The probe is a zero-byte read, so the cursor must not move.
    assert file.tell() == 0


def test_parse_n_doubles_handles_values_larger_than_buffer() -> None:
    data = b" ".join(b"1.25" for _ in range(300_000)) + b" label"
    file = BytesIO(data)
    out = np.empty(300_000, dtype=np.float64)

    parsed = parse_n_doubles(file, out)

    assert parsed == 300_000
    assert file.tell() == len(data) - len(b" label")
    assert file.read(1) == b" "
    assert np.all(out == 1.25)


def test_parse_n_doubles_rejects_oversized_nelem() -> None:
    file = BytesIO(b"1 2 3")
    out = np.empty(2, dtype=np.float64)

    with pytest.raises(ValueError, match="nelem exceeds output length"):
        parse_n_doubles(file, out, nelem=3)


def test_parse_n_doubles_short_file_returns_partial_count() -> None:
    """EOF reached before ``nelem`` values: return partial count without raising."""
    data = b"1.0 2.0 3.0"
    file = BytesIO(data)
    out = np.empty(5, dtype=np.float64)

    parsed = parse_n_doubles(file, out, nelem=5)

    assert parsed == 3
    # Cursor should land right after the last successfully parsed number.
    assert file.tell() == len(data)
    assert_exact(out[:parsed], np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_parse_n_doubles_buffer_ends_at_newline() -> None:
    """A buffer terminating exactly on a whitespace byte is fully consumable."""
    data = b"1.0 2.0 3.0\n"
    file = BytesIO(data)
    out = np.empty(3, dtype=np.float64)

    parsed = parse_n_doubles(file, out)

    assert parsed == 3
    assert_exact(out, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_parse_n_doubles_stops_on_invalid_token() -> None:
    """An invalid token before ``nelem`` is reached aborts parsing without consuming it."""
    data = b"1.0 2.0 not_a_number 3.0"
    file = BytesIO(data)
    out = np.empty(4, dtype=np.float64)

    parsed = parse_n_doubles(file, out, nelem=4)

    assert parsed == 2
    assert_exact(out[:parsed], np.array([1.0, 2.0], dtype=np.float64))
    # Cursor should sit at (or before) the invalid token so the caller can recover.
    remaining = file.read()
    assert b"not_a_number" in remaining


def test_parse_n_doubles_token_longer_than_buffer() -> None:
    """A single huge token (no embedded whitespace) is still parsed correctly."""
    # Build a token like "1." + many zeros + "5" that exceeds the 1 MiB buffer.
    huge_token = b"1." + b"0" * (1_500_000) + b"5"
    data = huge_token + b" 2.5"
    file = BytesIO(data)
    out = np.empty(2, dtype=np.float64)

    parsed = parse_n_doubles(file, out)

    assert parsed == 2
    # The huge token's value is mathematically 1.0...05 == 1.0 at double precision.
    assert out[0] == pytest.approx(1.0, abs=1e-300)
    assert out[1] == 2.5
