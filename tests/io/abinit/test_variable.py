from __future__ import annotations

import pytest

from pymatgen.io.abinit.variable import InputVariable


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([[1, 2], [3, 40]], "\n  1  2\n  3 40"),
        ([[0.0, 0.5, 0.0], [0.5, 0.5, 0.5]], "\n    0.0    0.5    0.0\n    0.5    0.5    0.5"),
        ([["a", "bc"]], "\n  a bc"),
    ],
)
def test_format_list2d(values, expected):
    assert InputVariable.format_list2d(values) == expected


def test_str_list_of_lists():
    # A list of lists (e.g. kptbounds, shiftk) is written one row per line.
    var = InputVariable("kptbounds", [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    lines = str(var).splitlines()
    assert lines[0] == " kptbounds"
    assert [[float(tok) for tok in line.split()] for line in lines[1:]] == [[0, 0, 0], [0.5, 0.5, 0.5]]
