from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pymatgen.io.jdftx._output_utils import _get_orb_label_list, _get_u_to_oa_map
from pymatgen.io.jdftx.outputs import JDFTXOutputs, _disambiguate_paths, implemented_store_vars

from .outputs_test_utils import (
    n2_ex_calc_dir,
    n2_ex_calc_dir_bandprojections_metadata,
    n2_ex_calc_dir_known_paths,
    nh3_ex_calc_dir,
    nh3_ex_calc_dir_bandprojections_metadata,
    nh3_ex_calc_dir_known_paths,
)
from .shared_test_utils import assert_same_value

if TYPE_CHECKING:
    from pathlib import Path

_implemented_store_vars = [v for v in implemented_store_vars if v != "bandstructure"]


@pytest.mark.parametrize(
    ("calc_dir", "known_paths"),
    [(n2_ex_calc_dir, n2_ex_calc_dir_known_paths), (nh3_ex_calc_dir, nh3_ex_calc_dir_known_paths)],
)
def test_known_paths(calc_dir: Path, known_paths: dict):
    """Test that the known paths are correct."""
    jo = JDFTXOutputs.from_calc_dir(calc_dir)
    for path_type, path in known_paths.items():
        assert jo.paths[path_type] == path


@pytest.mark.parametrize(
    ("calc_dir", "store_vars"),
    [
        (n2_ex_calc_dir, ["bandProjections", "eigenvals", "kpts"]),
        (n2_ex_calc_dir, ["bandProjections", "eigenvals"]),
        (n2_ex_calc_dir, ["bandProjections"]),
        (n2_ex_calc_dir, []),
    ],
)
def test_store_vars(calc_dir: Path, store_vars: list[str]):
    """Test that the stored variables are correct."""
    jo = JDFTXOutputs.from_calc_dir(calc_dir, store_vars=store_vars)
    for var in _implemented_store_vars:
        assert hasattr(jo, var)
        if var in store_vars:
            assert getattr(jo, var) is not None
        else:
            assert getattr(jo, var) is None


@pytest.mark.parametrize(
    ("calc_dir", "store_vars"),
    [
        (n2_ex_calc_dir, ["bandProjections", "eigenvals", "kpts", "bandstructure"]),
        (n2_ex_calc_dir, ["bandProjections", "bandstructure"]),
        (n2_ex_calc_dir, ["bandstructure"]),
    ],
)
def test_bandstructure_construction(calc_dir: Path, store_vars: list[str]):
    """Test that the stored variables are correct."""
    jo = JDFTXOutputs.from_calc_dir(calc_dir, store_vars=store_vars)
    assert hasattr(jo, "bandstructure")
    required_for_bandstructure = ["eigenvals", "kpts"]
    for var in required_for_bandstructure:
        assert hasattr(jo, var)
    if "bandProjections" in store_vars:
        assert len(jo.bandstructure.projections)
    else:
        assert not len(jo.bandstructure.projections)


@pytest.mark.parametrize(
    ("calc_dir", "known_metadata"),
    [
        (n2_ex_calc_dir, n2_ex_calc_dir_bandprojections_metadata),
        (nh3_ex_calc_dir, nh3_ex_calc_dir_bandprojections_metadata),
    ],
)
def test_store_bandprojections(calc_dir: Path, known_metadata: dict):
    """Test that the stored band projections are correct."""
    jo = JDFTXOutputs.from_calc_dir(calc_dir, store_vars=["bandProjections"])
    for var in ["orb_label_list"]:
        assert hasattr(jo, var)
        assert_same_value(getattr(jo, var), known_metadata[var])
    assert_same_value(jo.bandProjections.shape, known_metadata["shape"])
    assert pytest.approx(jo.bandProjections[0, 0, 0]) == known_metadata["first val"]


@pytest.mark.parametrize("calc_dir", [n2_ex_calc_dir, nh3_ex_calc_dir])
def test_store_eigenvals(calc_dir: Path):
    jo = JDFTXOutputs.from_calc_dir(calc_dir, store_vars=["eigenvals"])
    assert hasattr(jo, "eigenvals")
    eigenvals = jo.eigenvals
    assert eigenvals is not None


def test_bandstructure_projections_from_complex_bandprojections():
    """Complex band projections must be converted to |<orb|band>|^2 (not their real part) for BandStructure."""
    jo = JDFTXOutputs.from_calc_dir(n2_ex_calc_dir, store_vars=["bandProjections", "bandstructure"])
    assert np.iscomplexobj(jo.bandProjections)
    projections = jo.bandstructure.projections
    total = sum(proj.sum() for proj in projections.values())
    assert all((proj >= 0).all() for proj in projections.values())
    assert total == pytest.approx(np.sum(np.abs(jo.bandProjections) ** 2), rel=1e-5)


def test_check_store_vars_does_not_mutate_input():
    store_vars = ["bandstructure"]
    jo = JDFTXOutputs.from_calc_dir(n2_ex_calc_dir, store_vars=store_vars)
    assert store_vars == ["bandstructure"]
    assert jo.bandstructure is not None


def test_disambiguate_paths_with_prefix(tmp_path: Path):
    for name in ("jdft.eigenvals", "other.eigenvals", "eigenvals"):
        (tmp_path / name).touch()
    paths = sorted(tmp_path.glob("*eigenvals"))
    assert _disambiguate_paths(paths, "eigenvals", "jdft") == tmp_path / "jdft.eigenvals"
    assert _disambiguate_paths(paths, "eigenvals", "$VAR") == tmp_path / "eigenvals"
    assert _disambiguate_paths(paths, "eigenvals", None) == tmp_path / "eigenvals"
    with pytest.raises(FileNotFoundError, match="none match the prefix"):
        _disambiguate_paths(paths, "eigenvals", "missing")


def _write_f_orbital_bandprojections(path: Path) -> None:
    nproj = 16
    lines = [
        f"1 states, 1 bands, {nproj} orbital-projections, 1 species",
        "# Symbol nAtoms nOrbitalsPerAtom lMax nShells(l=0) ... nShells(l=lMax)",
        f"Pt 1 {nproj} 3 1 1 1 1",
        "#     0  [ +0.0000000 +0.0000000 +0.0000000 ]  1.000000000  spin +1; |projection|^2 per orbital:",
        " ".join(f"{0.01 * (i + 1):.4f}" for i in range(nproj)),
    ]
    path.write_text("\n".join(lines) + "\n")


def test_f_orbital_projections(tmp_path: Path):
    """Projections onto f orbitals must map onto the 16 pymatgen Orbitals without errors."""
    bandfile = tmp_path / "bandProjections"
    _write_f_orbital_bandprojections(bandfile)
    u_to_oa_map = _get_u_to_oa_map(bandfile)
    assert len(u_to_oa_map) == 16
    assert sorted(orb for orb, _ in u_to_oa_map) == list(range(16))
    assert all(atom == 0 for _, atom in u_to_oa_map)
    orb_label_list = _get_orb_label_list(bandfile)
    assert JDFTXOutputs._get_lmax(SimpleNamespace(orb_label_list=orb_label_list)) == ("f", 16)
