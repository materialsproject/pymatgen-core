from __future__ import annotations

import io
import os.path
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from pymatgen.core import Lattice, Structure
from pymatgen.io.abinit import pseudos as pseudos_mod
from pymatgen.io.abinit.pseudos import (
    Hint,
    NcAbinitHeader,
    PawAbinitHeader,
    PawXmlSetup,
    Pseudo,
    PseudoParseError,
    PseudoParser,
    PseudoTable,
    RadialFunction,
    _dict_from_lines,
    _int_from_str,
    _read_nlines,
    l2str,
    str2l,
)
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/io/abinit"


class TestPseudo(MatSciTest):
    def setup_method(self):
        nc_pseudo_fnames = defaultdict(list)
        nc_pseudo_fnames["Si"] = [f"{TEST_DIR}/{file}" for file in ("14si.pspnc", "14si.4.hgh", "14-Si.LDA.fhi")]

        self.nc_pseudos = defaultdict(list)
        self.paw_pseudos = defaultdict(list)

        for symbol, file_names in nc_pseudo_fnames.items():
            for file_name in file_names:
                _root, ext = os.path.splitext(file_name)
                pseudo = Pseudo.from_file(file_name)
                self.nc_pseudos[symbol].append(pseudo)

                # Save the pseudo as instance attribute whose name
                # is constructed with the rule: symbol_ppformat
                attribute = f"{symbol}_{ext[1:]}"
                if hasattr(self, attribute):
                    raise RuntimeError(f"self has already {attribute=}")

                setattr(self, attribute, pseudo)

    def test_nc_pseudos(self):
        """Test norm-conserving pseudopotentials."""
        for symbol, pseudos in self.nc_pseudos.items():
            for pseudo in pseudos:
                assert repr(pseudo)
                assert str(pseudo)
                assert pseudo.isnc
                assert not pseudo.ispaw
                assert pseudo.Z == 14
                assert pseudo.symbol == symbol
                assert pseudo.Z_val == 4
                assert pseudo.nlcc_radius >= 0.0

                # Test pickle
                self.serialize_with_pickle(pseudo)

                # Test MSONable
                self.assert_msonable(pseudo)

        # HGH pseudos
        pseudo = self.Si_hgh
        assert not pseudo.has_nlcc
        assert pseudo.l_max == 1
        assert pseudo.l_local == 0
        assert not pseudo.supports_soc
        assert self.Si_hgh.md5 is not None
        assert self.Si_hgh == self.Si_hgh

        # TM pseudos
        pseudo = self.Si_pspnc
        assert pseudo.has_nlcc
        assert pseudo.l_max == 2
        assert pseudo.l_local == 2
        assert not pseudo.supports_soc
        assert self.Si_hgh != self.Si_pspnc

        # FHI pseudos
        pseudo = self.Si_fhi
        assert not pseudo.has_nlcc
        assert pseudo.l_max == 3
        assert pseudo.l_local == 2
        assert not pseudo.supports_soc

        # Test PseudoTable
        table = PseudoTable(self.nc_pseudos["Si"])
        assert repr(table)
        assert str(table)
        assert table.allnc
        assert not table.allpaw
        assert len(table) == 3
        assert len(table[14]) == 3
        assert len(table.select_symbols("Si")) == 3
        assert table.zlist == [14]
        assert not table.is_complete()

        # Test pickle
        self.serialize_with_pickle(table, test_eq=False)

    def test_paw_pseudos(self):
        """Test 28ni.paw."""
        file_name = f"{TEST_DIR}/28ni.paw.tar.xz"
        symbol = "Ni"
        with tarfile.open(file_name, mode="r:xz") as t:
            # TODO: remove attr check after only 3.12+
            if hasattr(tarfile, "data_filter"):
                t.extractall(".", filter="data")
            else:
                t.extractall(".")  # noqa: S202
            path = os.path.join(".", "28ni.paw")
            pseudo = Pseudo.from_file(path)

            assert repr(pseudo)
            assert str(pseudo)
            assert not pseudo.isnc
            assert pseudo.ispaw
            assert pseudo.Z == 28
            assert pseudo.symbol == symbol
            assert pseudo.Z_val == 18
            assert pseudo.paw_radius >= 0.0

            assert pseudo.l_max == 2
            assert pseudo.l_local == 0
            assert pseudo.supports_soc
            assert pseudo.md5 is not None

            # Test pickle
            self.serialize_with_pickle(pseudo)

            # Test MSONable
            self.assert_msonable(pseudo)

    def test_pawxml_pseudos(self):
        """Test O.GGA_PBE-JTH-paw.xml."""
        oxygen = Pseudo.from_file(f"{TEST_DIR}/O.GGA_PBE-JTH-paw.xml")
        assert repr(oxygen)
        assert str(oxygen)
        assert isinstance(oxygen.as_dict(), dict)

        assert oxygen.ispaw
        assert oxygen.symbol == "O"
        assert (oxygen.Z, oxygen.core, oxygen.valence) == (8, 2, 6), oxygen.Z_val == 6

        assert oxygen.xc.type == "GGA"
        assert oxygen.xc.name == "PBE"
        assert oxygen.supports_soc
        assert oxygen.md5 is not None
        assert oxygen.paw_radius == approx(1.4146523028)

        # Test pickle
        new_objs = self.serialize_with_pickle(oxygen)
        # Test MSONable
        self.assert_msonable(oxygen)

        for obj in new_objs:
            assert obj.ispaw
            assert obj.symbol == "O"
            assert (obj.Z, obj.core, obj.valence) == (8, 2, 6), obj.Z_val == 6

            assert obj.paw_radius == approx(1.4146523028)

    def test_oncvpsp_pseudo_sr(self):
        """Test the ONCVPSP Ge pseudo (scalar relativistic version)."""
        ger = Pseudo.from_file(f"{TEST_DIR}/ge.oncvpsp")
        assert repr(ger)
        assert str(ger)
        assert isinstance(ger.as_dict(), dict)
        ger.as_tmpfile()

        assert ger.symbol == "Ge"
        assert ger.Z == approx(32.0)  # noqa: SIM300
        assert ger.Z_val == approx(4.0)
        assert ger.isnc
        assert not ger.ispaw
        assert ger.l_max == 2
        assert ger.l_local == 4
        assert ger.rcore is None
        assert not ger.supports_soc

        # Data persistence
        self.serialize_with_pickle(ger)
        self.assert_msonable(ger)

    def test_oncvpsp_pseudo_fr(self):
        """Test the ONCVPSP Pb pseudo (relativistic version with SO)."""
        pb = Pseudo.from_file(f"{TEST_DIR}/Pb-d-3_r.psp8")
        repr(pb)
        str(pb)

        # Data persistence
        self.serialize_with_pickle(pb)
        self.assert_msonable(pb)

        assert pb.symbol == "Pb"
        assert pb.Z == approx(82.0)  # noqa: SIM300
        assert pb.Z_val == approx(14.0)
        assert pb.isnc
        assert not pb.ispaw
        assert pb.l_max == 2
        assert pb.l_local == 4
        assert pb.supports_soc


class TestPseudoTable(MatSciTest):
    def test_methods(self):
        """Test PseudoTable methods."""
        table = PseudoTable([f"{TEST_DIR}/{file}" for file in ("14si.pspnc", "14si.4.hgh", "14-Si.LDA.fhi")])
        assert str(table)
        assert len(table) == 3
        for pseudo in table:
            assert pseudo.isnc
        assert table.allnc
        assert not table.allpaw
        assert table.zlist == [14]

        # Data persistence
        self.serialize_with_pickle(table, test_eq=False)

        dct = table.as_dict()
        PseudoTable.from_dict(dct)
        self.assert_msonable(table)

        selected = table.select_symbols("Si")
        assert len(selected) == len(table)
        assert selected.__class__ is table.__class__

        with pytest.raises(ValueError, match=r"Found multiple occurrences of symbol\(s\) Si"):
            table.pseudos_with_symbols("Si")


GTH_TEXT = """Goedecker-Teter-Hutter  Wed May  8 14:27:44 EDT 1996
1   1   960508                     zatom,zion,pspdat
2   1   0    0    2001    0.       pspcod,pspxc,lmax,lloc,mmax,r2well
0.2000000 -4.0663326  0.6778322 0 0     rloc, c1, c2, c3, c4
0 0 0                              rs, h1s, h2s
0 0                                rp, h1p
  1.36 .2   0.6                    rcutoff, rloc
"""


def _copy_with_lines(src: str | Path, dst: str | Path, replacements: dict[int, str]) -> str:
    """Copy src to dst replacing the lines (0-based index) given in replacements."""
    lines = Path(src).read_text(encoding="utf-8").splitlines(keepends=True)
    for idx, new_line in replacements.items():
        lines[idx] = new_line if new_line.endswith("\n") else f"{new_line}\n"
    Path(dst).write_text("".join(lines), encoding="utf-8")
    return str(dst)


def _write_gth(path: str | Path = "H.gth") -> str:
    Path(path).write_text(GTH_TEXT, encoding="utf-8")
    return str(path)


def _extract_ni_paw() -> str:
    with tarfile.open(f"{TEST_DIR}/28ni.paw.tar.xz", mode="r:xz") as tar:
        # TODO: remove attr check after only 3.12+
        if hasattr(tarfile, "data_filter"):
            tar.extractall(".", filter="data")
        else:
            tar.extractall(".")  # noqa: S202
    return os.path.abspath("28ni.paw")


class _Report(dict):
    """Minimal stand-in for abipy's DojoReport providing `deepcopy`."""

    def deepcopy(self):
        return _Report(self)


class TestHelpers:
    def test_l2str_str2l(self):
        assert [l2str(ang) for ang in range(7)] == list("spdfghi")
        assert "Unknown angular momentum" in l2str(42)
        assert str2l("f") == 3
        with pytest.raises(KeyError):
            str2l("z")

    def test_read_nlines(self):
        Path("five.txt").write_text("".join(f"line{i}\n" for i in range(5)), encoding="utf-8")
        assert _read_nlines("five.txt", 2) == ["line0\n", "line1\n"]
        assert len(_read_nlines("five.txt", -1)) == 5
        assert len(_read_nlines("five.txt", 10)) == 5

    def test_dict_from_lines(self):
        assert _dict_from_lines("1 2 a,b", [2]) == {"a": "1", "b": "2"}
        # Optional keys in brackets are sanitized.
        assert _dict_from_lines(["skip me", "1 2 a[,b]"], [0, 2]) == {"a": "1", "b": "2"}
        assert _dict_from_lines("1 2 : a,b", [2], sep=":") == {"a": "1", "b": "2"}

        with pytest.raises(ValueError, match="key_nums"):
            _dict_from_lines(["1 a", "2 b"], [1])
        with pytest.raises(ValueError, match="Expecting separator :"):
            _dict_from_lines("1 2 a,b", [2], sep=":")
        with pytest.raises(ValueError, match="must equal"):
            _dict_from_lines("1 2 a,b,c", [2])
        with pytest.raises(TypeError):
            _dict_from_lines("1 a", 1)

    def test_int_from_str(self):
        assert _int_from_str("14") == 14
        assert _int_from_str("14.0") == 14
        # Fractional charges are rounded to the nearest integer.
        assert _int_from_str("14.6") == 15


class TestHint:
    def test_hint(self):
        hint = Hint(ecut=10)
        assert hint.pawecutdg == 10
        assert str(hint) == "ecut: 10, pawecutdg: 10"

        hint = Hint(ecut=10, pawecutdg=40)
        dct = hint.as_dict()
        assert dct["@class"] == "Hint"
        new = Hint.from_dict(dct)
        assert (new.ecut, new.pawecutdg) == (10, 40)


NC_HEADER_KWARGS = {
    "zatom": "14",
    "zion": "4",
    "pspdat": "940714",
    "pspcod": "1",
    "pspxc": "1",
    "lmax": "2",
    "r2well": "0",
    "mmax": "2001",
}

PAW_HEADER_KWARGS = {
    "zatom": "28",
    "zion": "18",
    "pspdat": "20061204",
    "pspcod": "7",
    "pspxc": "7",
    "lmax": "2",
    "lloc": "0",
    "mmax": "350",
    "r2well": "0",
    "pspfmt": "paw3",
    "creatorID": "1305",
    "basis_size": "5",
    "lmn_size": "13",
    "orbitals": [0, 0, 1, 1, 2],
    "number_of_meshes": "3",
    "r_cut": "2.3",
    "shape_type": "2",
    "rshape": "0",
}


class TestAbinitHeaders:
    def test_nc_header(self):
        # APE pseudos use `llocal` instead of `lloc`; extra kwargs are kept.
        header = NcAbinitHeader(" summary  ", llocal="2", extension_switch=1, **NC_HEADER_KWARGS)
        assert header.summary == "summary"
        assert header["lloc"] == 2
        assert header.lloc == 2
        assert header["zatom"] == 14
        assert header["rchrg"] == approx(0.0)
        assert header["extension_switch"] == 1
        with pytest.raises(AttributeError):
            _ = header.does_not_exist

    def test_nc_header_errors(self):
        with pytest.raises(RuntimeError, match="Attribute lloc must be specified"):
            NcAbinitHeader("summary", **NC_HEADER_KWARGS)
        with pytest.raises(RuntimeError, match="Conversion Error for key='lloc'"):
            NcAbinitHeader("summary", lloc="abc", **NC_HEADER_KWARGS)

    def test_paw_header(self):
        header = PawAbinitHeader("Ni", **PAW_HEADER_KWARGS)
        assert header["r_cut"] == approx(2.3)
        assert header["orbitals"] == [0, 0, 1, 1, 2]

        kwargs = PAW_HEADER_KWARGS.copy()
        del kwargs["rshape"]
        with pytest.raises(RuntimeError, match="Attribute rshape must be specified"):
            PawAbinitHeader("Ni", **kwargs)

        with pytest.raises(RuntimeError, match="Conversion Error for key='creatorID'"):
            PawAbinitHeader("Ni", **(PAW_HEADER_KWARGS | {"creatorID": "abc"}))

        with pytest.raises(RuntimeError, match="kwargs should be empty"):
            PawAbinitHeader("Ni", foo=1, **PAW_HEADER_KWARGS)


class TestPseudoParser:
    def test_gth(self):
        pseudo = Pseudo.from_file(_write_gth())
        assert pseudo.isnc
        assert pseudo.type == "NcAbinitPseudo"
        assert pseudo.symbol == "H"
        assert (pseudo.Z, pseudo.Z_val, pseudo.l_max, pseudo.l_local) == (1, 1, 0, 0)
        assert pseudo.summary.startswith("Goedecker-Teter-Hutter")
        assert not pseudo.supports_soc
        assert not pseudo.has_nlcc

    def test_fhi_without_nlcc_line(self):
        # The last record of the FHI header (rchrg, fchrg, qchrg) is optional.
        path = _copy_with_lines(f"{TEST_DIR}/14-Si.LDA.fhi", "Si.fhi", {3: "  0.000  0.000   rchrg, fchrg, qchrg"})
        pseudo = Pseudo.from_file(path)
        assert pseudo.Z == 14
        assert pseudo.nlcc_radius == approx(0.0)

    def test_invalid_files(self):
        parser = PseudoParser()

        # Unknown pspcod.
        path = _copy_with_lines(f"{TEST_DIR}/14si.4.hgh", "bad_pspcod.hgh", {2: " 5 1 1 0 2001 0 pspcod,pspxc"})
        with pytest.raises(PseudoParseError, match="Don't know how to handle pspcod=5"):
            parser.parse(path)

        # Line with pspcod cannot be parsed -> None.
        path = _copy_with_lines(f"{TEST_DIR}/14si.4.hgh", "garbage.hgh", {2: "foo bar"})
        assert parser.read_ppdesc(os.path.abspath(path)) is None
        assert parser.parse(path) is None

        # File too short to contain the header -> None.
        Path("short.txt").write_text("one line\n", encoding="utf-8")
        assert parser.parse("short.txt") is None

        # XML is not handled by read_ppdesc.
        with pytest.raises(PseudoParseError, match="XML pseudo not supported"):
            parser.read_ppdesc(f"{TEST_DIR}/O.GGA_PBE-JTH-paw.xml")

        # Errors in the header parsers are wrapped in PseudoParseError.
        path = _copy_with_lines(f"{TEST_DIR}/14si.4.hgh", "bad_header.hgh", {1: " 14 4 zatom,zion,pspdat"})
        with pytest.raises(PseudoParseError, match=r"bad_header\.hgh"):
            parser.parse(path)

    def test_paw_unsupported_format(self):
        path = _copy_with_lines(_extract_ni_paw(), "Ni_paw2.paw", {3: " paw2 1305    : pspfmt,creatorID"})
        assert PseudoParser().read_ppdesc(path).format == "paw2"
        with pytest.raises(PseudoParseError, match="format paw2 not in"):
            Pseudo.from_file(path)

    def test_paw_abinit_rcore(self):
        pseudo = Pseudo.from_file(_extract_ni_paw())
        assert pseudo.rcore == approx(2.3)
        assert pseudo.rcore == pseudo.paw_radius
        assert pseudo.summary.startswith("Paw atomic data for element Ni")
        assert pseudo.header["orbitals"] == [0, 0, 1, 1, 2]
        assert pseudo.header["number_of_meshes"] == 3

    @pytest.mark.parametrize(("switch", "supports_soc"), [(0, False), (1, False), (2, True), (3, True)])
    def test_oncvpsp_extension_switch(self, switch, supports_soc):
        path = _copy_with_lines(f"{TEST_DIR}/ge.oncvpsp", "ge.psp8", {5: f"     {switch}   extension_switch"})
        assert Pseudo.from_file(path).supports_soc is supports_soc

    def test_oncvpsp_invalid_extension_switch(self):
        path = _copy_with_lines(f"{TEST_DIR}/ge.oncvpsp", "ge.psp8", {5: "     5   extension_switch"})
        pseudo = Pseudo.from_file(path)
        with pytest.raises(ValueError, match="Don't know how to handle extension_switch: 5"):
            _ = pseudo.supports_soc

    @staticmethod
    def _make_scan_dir() -> Path:
        dirname = Path("scan")
        dirname.mkdir()
        for fname in ("14si.pspnc", "14si.4.hgh", "O.GGA_PBE-JTH-paw.xml", "gan.cif"):
            shutil.copy(f"{TEST_DIR}/{fname}", dirname)
        (dirname / ".hidden").write_text(GTH_TEXT, encoding="utf-8")
        # Unknown pspcod -> parse raises and the file is skipped.
        _copy_with_lines(f"{TEST_DIR}/14si.4.hgh", dirname / "bad.hgh", {2: " 5 1 1 0 2001 0 pspcod,pspxc"})
        (dirname / "subdir").mkdir()
        return dirname

    def test_scan_directory(self):
        dirname = self._make_scan_dir()

        pseudos = PseudoParser().scan_directory(dirname)
        assert sorted(p.basename for p in pseudos) == ["14si.4.hgh", "14si.pspnc", "O.GGA_PBE-JTH-paw.xml"]

        pseudos = PseudoParser().scan_directory(dirname, exclude_exts=["xml"], exclude_fnames=["14si.4.hgh"])
        assert [p.basename for p in pseudos] == ["14si.pspnc"]

    @pytest.mark.xfail(strict=True, reason="scan_directory uses list.extend(path), storing the characters of each path")
    def test_scan_directory_parsed_paths(self):
        dirname = self._make_scan_dir()
        parser = PseudoParser()
        parser.scan_directory(dirname, exclude_exts=["xml", "cif"])
        assert sorted(parser._parsed_paths) == sorted(str(dirname / fn) for fn in ("14si.pspnc", "14si.4.hgh"))


class TestPseudoBase:
    @pytest.fixture
    def si_pseudo(self):
        shutil.copy(f"{TEST_DIR}/14si.pspnc", ".")
        return Pseudo.from_file("14si.pspnc")

    def test_as_pseudo(self, si_pseudo):
        assert Pseudo.as_pseudo(si_pseudo) is si_pseudo
        assert Pseudo.as_pseudo(si_pseudo.filepath) == si_pseudo

    def test_eq_not_implemented(self, si_pseudo):
        assert si_pseudo.__eq__("not a pseudo") is NotImplemented
        assert si_pseudo != "not a pseudo"

    def test_repr_relpath_failure(self, si_pseudo, monkeypatch):
        def _raise(*args, **kwargs):
            raise ValueError("relpath failed")

        monkeypatch.setattr(pseudos_mod.os.path, "relpath", _raise)
        assert repr(si_pseudo) == f"<NcAbinitPseudo at {si_pseudo.filepath}>"

    def test_from_dict_md5_mismatch(self, si_pseudo):
        dct = si_pseudo.as_dict()
        assert dct["type"] == "NcAbinitPseudo"
        assert dct["md5"] == si_pseudo.compute_md5()
        dct["md5"] = "wrong"
        with pytest.raises(ValueError, match="md5 found in file does not agree"):
            Pseudo.from_dict(dct)

    def test_hints_without_dojo_report(self, si_pseudo):
        assert not si_pseudo.has_dojo_report
        hint = si_pseudo.hint_for_accuracy("normal")
        assert (hint.ecut, hint.pawecutdg) == (0.0, 0.0)
        assert si_pseudo.djrepo_path == os.path.abspath("14si.djrepo")

    @pytest.mark.parametrize("key", ["hints", "ppgen_hints"])
    def test_hints_from_dojo_report(self, si_pseudo, key):
        si_pseudo.dojo_report = {
            key: {acc: {"ecut": ecut} for acc, ecut in zip(("low", "normal", "high"), (10, 20, 30), strict=True)}
        }
        assert si_pseudo.has_dojo_report
        assert si_pseudo.has_hints
        assert si_pseudo.hint_for_accuracy("normal").ecut == 20
        assert si_pseudo.hint_for_accuracy("high").pawecutdg == 30
        assert "hint for high accuracy: ecut: 30, pawecutdg: 30" in si_pseudo.to_str()

    def test_hints_missing(self, si_pseudo):
        si_pseudo.dojo_report = {"other": {}}
        assert si_pseudo.hint_for_accuracy("low").ecut == 0.0

        # Missing accuracy level -> no hints.
        si_pseudo.dojo_report = {"hints": {"low": {"ecut": 10}}}
        assert not si_pseudo.has_hints
        assert "hint for" not in si_pseudo.to_str()

    def test_as_tmpfile_with_dojo_report(self, si_pseudo):
        Path("14si.djrepo").write_text("{}", encoding="utf-8")
        si_pseudo.dojo_report = _Report(hints={"low": {"ecut": 1}})

        tmpdir = Path("tmp_copy")
        tmpdir.mkdir()
        new = si_pseudo.as_tmpfile(tmpdir=str(tmpdir))
        assert new.filepath == str((tmpdir / "14si.pspnc").absolute())
        assert (tmpdir / "14si.djrepo").is_file()
        assert new.dojo_report == si_pseudo.dojo_report
        assert new == si_pseudo


class TestPawXmlSetup:
    @pytest.fixture
    def oxygen(self):
        return PawXmlSetup(f"{TEST_DIR}/O.GGA_PBE-JTH-paw.xml")

    def test_attributes(self, oxygen):
        assert oxygen.paw_setup_version is not None
        assert oxygen.l_max is None
        assert oxygen.l_local is None
        assert oxygen.summary == ""
        assert oxygen.rcore == approx(1.4146523028)
        assert list(oxygen.valence_states) == ["O1", "O2", "O3", "O4"]
        assert float(oxygen.valence_states["O3"].f) == approx(4.0)
        assert sorted(oxygen.rad_grids) == [f"log{i}" for i in range(1, 6)]
        assert len(oxygen.rad_grids["log1"]) == 1311
        assert oxygen.rad_grids["log1"][0] == approx(0.0)
        assert "_root" not in oxygen.__getstate__()

    def test_radial_functions(self, oxygen):
        for rfunc in (oxygen.ae_core_density, oxygen.pseudo_core_density):
            assert isinstance(rfunc, RadialFunction)
            assert len(rfunc.mesh) == len(rfunc.values)

        # AE and pseudo core densities coincide outside the PAW sphere.
        ae, ps = oxygen.ae_core_density, oxygen.pseudo_core_density
        outside = ae.mesh > 2 * oxygen.paw_radius
        assert ae.values[outside] == approx(ps.values[outside], abs=1e-6)

        states = ["O1", "O2", "O3", "O4"]
        for funcs in (oxygen.ae_partial_waves, oxygen.pseudo_partial_waves, oxygen.projector_functions):
            assert list(funcs) == states
            for rfunc in funcs.values():
                assert len(rfunc.mesh) == len(rfunc.values)

    def test_plots(self, oxygen):
        import matplotlib.pyplot as plt

        fig = oxygen.plot_densities(show=False)
        assert len(fig.axes[0].lines) == 2
        fig = oxygen.plot_waves(show=False)
        assert len(fig.axes[0].lines) == 8
        fig = oxygen.plot_projectors(show=False)
        labels = [line.get_label() for line in fig.axes[0].lines]
        assert labels == [f"TPROJ: O{i}" for i in range(1, 5)]
        plt.close("all")

    @pytest.mark.xfail(strict=True, reason="plot_densities swaps the labels of the AE and pseudo core densities")
    def test_plot_densities_labels(self, oxygen):
        import matplotlib.pyplot as plt

        fig = oxygen.plot_densities(show=False)
        labels = [line.get_label() for line in fig.axes[0].lines]
        plt.close("all")
        # Lines are plotted in the order ae_core_density, pseudo_core_density.
        assert labels == ["$n_c$", r"$\tilde{n}_c$"]

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            ({"eq": "r=a*exp(d*i)", "a": "1", "d": "0.5"}, [1, np.exp(0.5), np.e]),
            ({"eq": "r=a*i/(n-i)", "a": "1", "n": "4"}, [0, 1 / 3, 1]),
            ({"eq": "r = a*(exp(d*i)-1)", "a": "2", "d": "1"}, [0, 2 * (np.e - 1), 2 * (np.e**2 - 1)]),
            ({"eq": "r=d*i", "d": "0.1"}, [0, 0.1, 0.2]),
            ({"eq": "r=(i/n+a)^5/a-a^4", "a": "1", "n": "2"}, [0, 1.5**5 - 1, 31]),
        ],
    )
    def test_eval_grid(self, params, expected):
        mesh = PawXmlSetup._eval_grid(params | {"istart": "0", "iend": "2"})
        assert mesh == approx(expected)

    def test_eval_grid_unknown(self):
        with pytest.raises(ValueError, match="Unknown grid type"):
            PawXmlSetup._eval_grid({"eq": "r=i^2", "istart": "0", "iend": "2"})

    @pytest.mark.parametrize(("line_idx", "match"), [(18, "attrib.id='O1'"), (23, "gid='log1'")])
    def test_duplicated_entries(self, line_idx, match):
        lines = Path(f"{TEST_DIR}/O.GGA_PBE-JTH-paw.xml").read_text(encoding="utf-8").splitlines(keepends=True)
        lines.insert(line_idx, lines[line_idx])
        Path("dup.xml").write_text("".join(lines), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            PawXmlSetup("dup.xml")

    def test_missing_paw_radius(self):
        lines = Path(f"{TEST_DIR}/O.GGA_PBE-JTH-paw.xml").read_text(encoding="utf-8").splitlines(keepends=True)
        assert "PAW_radius" in lines[16]
        Path("old.xml").write_text("".join(lines[:16] + lines[17:]), encoding="utf-8")
        assert PawXmlSetup("old.xml").paw_radius is None


class TestPseudoTableExtended(MatSciTest):
    def setup_method(self):
        self.si_files = [f"{TEST_DIR}/{fn}" for fn in ("14si.pspnc", "14si.4.hgh", "14-Si.LDA.fhi")]
        self.files = [
            *self.si_files,
            f"{TEST_DIR}/O.GGA_PBE-JTH-paw.xml",
            f"{TEST_DIR}/ge.oncvpsp",
            f"{TEST_DIR}/Pb-d-3_r.psp8",
        ]
        self.table = PseudoTable(self.files[::-1])

    def test_basic(self):
        table = self.table
        assert table.zlist == [8, 14, 32, 82]
        assert [p.Z for p in table] == [8, 14, 14, 14, 32, 82]
        assert not table.allnc
        assert not table.allpaw
        assert PseudoTable([self.files[3]]).allpaw
        assert len(table.Si) == 3
        assert table.O[0].ispaw

        assert PseudoTable.as_table(table) is table
        assert isinstance(PseudoTable.as_table(self.si_files), PseudoTable)

        # A single pseudo or a single path is also accepted.
        assert len(PseudoTable(table.O[0])) == 1
        assert len(PseudoTable(self.files[-1])) == 1

    def test_getitem(self):
        table = self.table
        assert len(table[26]) == 0
        sub = table[8:15]
        assert isinstance(sub, PseudoTable)
        assert sub.zlist == [8, 14]
        assert len(sub) == 4
        with pytest.raises(ValueError, match=r"Z\.stop is None"):
            table[8:]

    @pytest.mark.xfail(strict=True, reason="PseudoTable.__getitem__ inserts missing Z keys into the defaultdict")
    def test_getitem_does_not_mutate(self):
        table = self.table
        assert len(table[26]) == 0
        assert len(table[1:3]) == 0
        assert not table.is_complete()
        assert table.zlist == [8, 14, 32, 82]

    def test_is_complete(self):
        assert not self.table.is_complete()
        assert not self.table.is_complete(zmax=9)
        assert PseudoTable([_write_gth()]).is_complete(zmax=2)

    def test_symbol_selection(self):
        table = self.table
        combos = table.all_combinations_for_elements(["Si", "O"])
        assert len(combos) == 3
        assert all([p.symbol for p in combo] == ["Si", "O"] for combo in combos)

        assert table.pseudo_with_symbol("O").ispaw
        assert len(table.pseudo_with_symbol("Si", allow_multi=True)) == 3
        with pytest.raises(ValueError, match="Found 3 occurrences of symbol='Si'"):
            table.pseudo_with_symbol("Si")
        with pytest.raises(ValueError, match="Found 0 occurrences of symbol='Fe'"):
            table.pseudo_with_symbol("Fe")

        assert [p.symbol for p in table.pseudos_with_symbols(["O", "Ge"])] == ["O", "Ge"]
        with pytest.raises(ValueError, match=r"Missing data for symbol\(s\) Fe"):
            table.pseudos_with_symbols(["O", "Fe"])

        assert table.select_symbols(["-Si"]).zlist == [8, 32, 82]
        assert [p.symbol for p in table.select_symbols(["-Si", "-Pb"], ret_list=True)] == ["O", "Ge"]
        with pytest.raises(ValueError, match="all strings must start with `-`"):
            table.select_symbols(["-Si", "O"])

    def test_get_pseudos_for_structure(self):
        geo = Structure(Lattice.cubic(4), ["Ge", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        assert {p.symbol for p in self.table.get_pseudos_for_structure(geo)} == {"Ge", "O"}

        si = Structure(Lattice.cubic(5.4), ["Si"], [[0, 0, 0]])
        with pytest.raises(ValueError, match="multiple occurrences"):
            self.table.get_pseudos_for_structure(si)

    def test_to_table(self):
        text = self.table.to_table(filter_function=lambda p: p.symbol == "Si")
        assert "ge.oncvpsp" in text
        assert "14si" not in text
        assert "basename" in text

        stream = io.StringIO()
        self.table.print_table(stream=stream)
        assert "14si.pspnc" in stream.getvalue()
        assert "PawXmlSetup" in stream.getvalue()

    def test_select(self):
        table = self.table
        assert [p.Z for p in table.sort_by_z()] == [8, 14, 14, 14, 32, 82]
        assert table.select(lambda p: p.Z_val > 4).zlist == [8, 82]
        assert table.select_rows(3).zlist == [14]
        assert table.select_rows([2, 4]).zlist == [8, 32]
        assert table.select_family("chalcogen").zlist == [8]

        assert len(table.with_dojo_report()) == 0
        table.Ge[0].dojo_report = {"hints": {}}
        assert table.with_dojo_report().zlist == [32]

    @pytest.mark.xfail(strict=True, reason="PseudoTable.sorted unpacks (index, pseudo) while iterating over pseudos")
    def test_sorted(self):
        new = self.table.sorted("Z_val", reverse=True)
        assert isinstance(new, PseudoTable)
        assert len(new) == len(self.table)

    def test_as_dict_round_trip(self):
        new = PseudoTable.from_dict(self.table.as_dict())
        assert new.zlist == self.table.zlist
        assert [p.basename for p in new] == [p.basename for p in self.table]

    @pytest.mark.xfail(strict=True, reason="PseudoTable.as_dict builds malformed keys for duplicated elements")
    def test_as_dict_keys(self):
        dct = PseudoTable(self.si_files).as_dict()
        assert {k for k in dct if not k.startswith("@")} == {"Silicon", "Silicon#1", "Silicon#2"}

    def test_from_dir(self):
        top = Path("pseudos")
        (top / "_skip").mkdir(parents=True)
        shutil.copy(f"{TEST_DIR}/Pb-d-3_r.psp8", top)
        shutil.copy(f"{TEST_DIR}/ge.oncvpsp", top)
        shutil.copy(f"{TEST_DIR}/Pb-d-3_r.psp8", top / "_skip" / "skipped.psp8")
        # Unknown pspcod -> parse raises and the file is skipped.
        _copy_with_lines(f"{TEST_DIR}/14si.4.hgh", top / "bad.psp8", {2: " 5 1 1 0 2001 0 pspcod,pspxc"})

        table = PseudoTable.from_dir(str(top))
        assert [p.basename for p in table] == ["Pb-d-3_r.psp8"]

        table = PseudoTable.from_dir(str(top), exts=("psp8", "oncvpsp"))
        assert table.zlist == [32, 82]
        assert len(table) == 2

        # Try all the files in top (non-recursive), skipping the invalid ones.
        _copy_with_lines(f"{TEST_DIR}/14si.4.hgh", top / "garbage.txt", {2: "foo bar"})
        shutil.copy(f"{TEST_DIR}/gan.cif", top)
        table = PseudoTable.from_dir(str(top), exts="all_files")
        assert [p.basename for p in table] == ["ge.oncvpsp", "Pb-d-3_r.psp8"]

        empty = Path("empty")
        empty.mkdir()
        shutil.copy(f"{TEST_DIR}/gan.cif", empty)
        assert PseudoTable.from_dir(str(empty), exts="all_files") is None

    @pytest.mark.xfail(
        strict=True, reason="PseudoTable.from_dir appends None for unparsable files and then crashes in __init__"
    )
    def test_from_dir_unparsable_file(self):
        top = Path("pseudos")
        top.mkdir()
        shutil.copy(f"{TEST_DIR}/Pb-d-3_r.psp8", top)
        _copy_with_lines(f"{TEST_DIR}/14si.4.hgh", top / "garbage.psp8", {2: "foo bar"})
        table = PseudoTable.from_dir(str(top))
        assert [p.basename for p in table] == ["Pb-d-3_r.psp8"]
