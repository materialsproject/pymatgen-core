from __future__ import annotations

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest import approx

from pymatgen.core import Element
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.cp2k.inputs import (
    DOS,
    LDOS,
    PBE,
    PDOS,
    QS,
    AtomicMetadata,
    BandStructure,
    BasisFile,
    BasisInfo,
    BrokenSymmetry,
    Cell,
    Coord,
    Cp2kInput,
    DataFile,
    Davidson,
    Dft,
    DftPlusU,
    Diagonalization,
    EDensityCube,
    ForceEval,
    GaussianTypeOrbitalBasisSet,
    Global,
    GthPotential,
    Keyword,
    KeywordList,
    Kind,
    Kpoints,
    KpointSet,
    Mgrid,
    MOCubes,
    OrbitalTransformation,
    PotentialFile,
    PotentialInfo,
    Scf,
    Section,
    SectionList,
    Smear,
    Subsys,
    VHartreeCube,
    XCFunctional,
)
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/io/cp2k"


BASIS_FILE_STR = """
 H  SZV-MOLOPT-GTH SZV-MOLOPT-GTH-q1
 1
 2 0 0 7 1
     11.478000339908  0.024916243200
      3.700758562763  0.079825490000
      1.446884268432  0.128862675300
      0.716814589696  0.379448894600
      0.247918564176  0.324552432600
      0.066918004004  0.037148121400
      0.021708243634 -0.001125195500
"""


class TestBasis(MatSciTest):
    def test_basis_info(self):
        # Ensure basis metadata can be read from string
        basis_info = BasisInfo.from_str("cc-pc-DZVP-MOLOPT-q1-SCAN")
        assert basis_info.valence == 2
        assert basis_info.molopt
        assert basis_info.electrons == 1
        assert basis_info.polarization == 1
        assert basis_info.cc
        assert basis_info.pc
        assert basis_info.xc == "SCAN"

        # Ensure one-way soft-matching works
        basis_info2 = BasisInfo.from_str("cc-pc-DZVP-MOLOPT-q1")
        assert basis_info2.softmatch(basis_info)
        assert not basis_info.softmatch(basis_info2)

        basis_info3 = BasisInfo.from_str("cpFIT3")
        assert basis_info3.valence == 3
        assert basis_info3.polarization == 1
        assert basis_info3.contracted

    def test_basis(self):
        # Ensure cp2k formatted string can be read for data correctly
        mol_opt = GaussianTypeOrbitalBasisSet.from_str(BASIS_FILE_STR)
        assert mol_opt.nexp == [7]
        # Basis file can read from strings
        bf = BasisFile.from_str(BASIS_FILE_STR)
        for obj in (mol_opt, bf.objects[0]):
            assert_allclose(
                obj.exponents[0],
                [
                    11.478000339908,
                    3.700758562763,
                    1.446884268432,
                    0.716814589696,
                    0.247918564176,
                    0.066918004004,
                    0.021708243634,
                ],
            )

        # Ensure keyword can be properly generated
        kw: Keyword = mol_opt.get_keyword()
        assert kw.values[0] == "SZV-MOLOPT-GTH"
        mol_opt.info.admm = True
        kw = mol_opt.get_keyword()
        assert list(kw.values) == ["AUX_FIT", "SZV-MOLOPT-GTH"]
        mol_opt.info.admm = False


class TestPotential(MatSciTest):
    all_hydrogen_str = """
H ALLELECTRON ALL
    1    0    0
    0.20000000    0
"""

    pot_hydrogen_str = """
H GTH-PBE-q1 GTH-PBE
    1
    0.20000000    2    -4.17890044     0.72446331
    0
"""

    def test_potential_info(self):
        # Ensure potential metadata can be read from string
        pot_info = PotentialInfo.from_str("GTH-PBE-q1-NLCC")
        assert pot_info.potential_type == "GTH"
        assert pot_info.xc == "PBE"
        assert pot_info.nlcc

        # Ensure one-way soft-matching works
        pot_info2 = PotentialInfo.from_str("GTH-q1-NLCC")
        assert pot_info2.softmatch(pot_info)
        assert not pot_info.softmatch(pot_info2)

    def test_potentials(self):
        # Ensure cp2k formatted string can be read for data correctly
        h_all_elec = GthPotential.from_str(self.all_hydrogen_str)
        assert h_all_elec.potential == "All Electron"
        pot = GthPotential.from_str(self.pot_hydrogen_str)
        assert pot.potential == "Pseudopotential"
        assert pot.r_loc == approx(0.2)
        assert pot.nexp_ppl == approx(2)
        assert_allclose(pot.c_exp_ppl, [-4.17890044, 0.72446331])

        # Basis file can read from strings
        pot_file = PotentialFile.from_str(self.pot_hydrogen_str)
        assert pot_file.objects[0] == pot

        pot_file_path = self.tmp_path / "potential-file"
        pot_file_path.write_text(self.pot_hydrogen_str, encoding="utf-8")
        pot_from_file = PotentialFile.from_file(pot_file_path)
        assert pot_file != pot_from_file  # unequal because pot_from_file has filename != None

        # Ensure keyword can be properly generated
        kw = pot.get_keyword()
        assert kw.values[0] == "GTH-PBE-q1"
        kw = h_all_elec.get_keyword()
        assert kw.values[0] == "ALL"


class TestCp2kInput(MatSciTest):
    si_struct = Structure(
        lattice=[
            [0, 2.734364, 2.734364],
            [2.734364, 0, 2.734364],
            [2.734364, 2.734364, 0],
        ],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    invalid_struct = Structure(
        lattice=[[-1.0, -10.0, -100.0], [0.1, 0.01, 0.001], [7.0, 11.0, 21.0]],
        species=["H"],
        coords=[[-1, -1, -1]],
    )

    ch_mol = Molecule(species=["C", "H"], coords=[[0, 0, 0], [1, 1, 1]])

    cp2k_input_str = """
&GLOBAL
    RUN_TYPE ENERGY
    PROJECT_NAME CP2K ! default name
&END
"""

    def setup_method(self):
        self.ci = Cp2kInput.from_file(f"{TEST_DIR}/cp2k.inp")

    def test_basic_sections(self):
        cp2k_input = Cp2kInput.from_str(self.cp2k_input_str)
        assert cp2k_input["GLOBAL"]["RUN_TYPE"] == Keyword("RUN_TYPE", "energy")
        assert cp2k_input["GLOBAL"]["PROJECT_NAME"].description == "default name"
        self.assert_msonable(cp2k_input)

    def test_section_list(self):
        sec1 = Section("TEST")
        sec_list = SectionList(sections=[sec1, sec1])
        for s in sec_list:
            assert isinstance(s, Section)
        assert sec_list[0].name == "TEST"
        assert sec_list[1].name == "TEST"
        assert len(sec_list) == 2
        sec_list += sec1
        assert len(sec_list) == 3

    def test_basic_keywords(self):
        kwd = Keyword("TEST1", 1, 2)
        assert kwd.values == (1, 2)
        kwd = Keyword("TEST2", [1, 2, 3])
        assert kwd.values == ([1, 2, 3],)
        kwd = Keyword("TEST3", "xyz", description="testing", units="Ha")
        assert kwd.description == "testing"
        assert "[Ha]" in kwd.get_str()

    def test_coords(self):
        for struct in (self.invalid_struct, self.si_struct, self.ch_mol):
            coords = Coord(struct)
            for val in coords.keywords.values():
                assert isinstance(val, Keyword | KeywordList)

    def test_kind(self):
        for struct in (self.invalid_struct, self.si_struct, self.ch_mol):
            for spec in struct.species:
                assert spec == Kind(spec).specie

    def test_ci_file(self):
        # proper type retrieval
        assert isinstance(self.ci["FORCE_EVAL"]["DFT"]["MGRID"]["NGRIDS"].values[0], int)
        assert isinstance(self.ci["FORCE_EVAL"]["DFT"]["UKS"].values[0], bool)
        assert isinstance(self.ci["FORCE_EVAL"]["DFT"]["QS"]["EPS_DEFAULT"].values[0], float)

        # description retrieval
        assert (
            self.ci["FORCE_EVAL"]["SUBSYS"].description == '"/" used to break input file parser in Section start line'
        )
        assert self.ci["FORCE_EVAL"]["SUBSYS"]["CELL"].description == "Input parameters needed to set up the CELL."
        assert (
            self.ci["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"].description == "Cutoff in [Ry] for finest level of the MG."
        )
        assert self.ci["FORCE_EVAL"]["METHOD"].description is None

    def test_odd_file(self):
        scramble = ""
        rng = np.random.default_rng()
        for string in self.ci.get_str():
            if rng.choice((True, False)):
                if string == "\t":
                    scramble += " "
                elif string == " ":
                    scramble += "  "
                elif string in ("&", "\n"):
                    scramble += string
                elif string.isalpha():
                    scramble += string.lower()
                else:
                    scramble += string
            else:
                scramble += string
        # Can you initialize from jumbled input
        # should be case insensitive and ignore
        # excessive white space or tabs
        ci = Cp2kInput.from_str(scramble)
        assert ci["FORCE_EVAL"]["DFT"]["UKS"] == Keyword("UKS", True)  # noqa: FBT003
        assert [k.name.upper() for k in ci["FORCE_EVAL"]["DFT"]["BASIS_SET_FILE_NAME"]] == [
            "BASIS_SET_FILE_NAME",
            "BASIS_SET_FILE_NAME",
        ]

    def test_preprocessor(self):
        assert self.ci.check("INCLUDE")
        assert self.ci["INCLUDE"]["KEYWORD"] == Keyword("KEYWORD", "VALUE")
        assert self.ci["FORCE_EVAL"]["METHOD"] == Keyword("METHOD", "QS")
        assert self.ci["FORCE_EVAL"]["DFT"]["SCF"]["MAX_SCF"] == Keyword("MAX_SCF", 1)

    def test_mongo(self):
        cp2k_input = Cp2kInput.from_str(self.cp2k_input_str)
        cp2k_input.inc({"GLOBAL": {"TEST": 1}})
        assert cp2k_input["global"]["test"] == Keyword("TEST", 1)

        cp2k_input.unset({"GLOBAL": "RUN_TYPE"})
        assert "RUN_TYPE" not in cp2k_input["global"].keywords

        cp2k_input.set({"GLOBAL": {"SUBSEC": {"TEST2": 2}, "SUBSEC2": {"Test2": 1}}})
        assert cp2k_input.check("global/SUBSEC")
        assert cp2k_input.check("global/subsec2")


class TestDataFile(MatSciTest):
    def test_data_file(self):
        # make temp file with BASIS_FILE_STR
        data_file = self.tmp_path / "data-file"
        data_file.write_text(BASIS_FILE_STR, encoding="utf-8")
        with pytest.raises(NotImplementedError):
            DataFile.from_file(data_file)


O_DZVP_STR = """
O DZVP-MOLOPT-GTH DZVP-MOLOPT-GTH-q6
 1
 2 0 2 7 2 2 1
     12.015954705512 -0.060190841200  0.065738617900  0.036543638800 -0.034210557400  0.014807054400
      5.108150287385 -0.129597923300  0.110885902200  0.120927648700 -0.120619770900  0.068186159300
      2.048398039874  0.118175889400 -0.053732406400  0.251093670300 -0.213719464600  0.290576499200
      0.832381575582  0.462964485000 -0.572670666200  0.352639910300 -0.473674858400  1.063344189500
      0.352316246455  0.450353782600  0.186760006700  0.294708645200  0.484848376400  0.307656114200
      0.142977330880  0.092715833600  0.387201458600  0.173039869300  0.717465919700  0.318346834400
      0.046760918300 -0.000255945800  0.003825849600  0.009726110600  0.032498979400 -0.005771736600
"""

FE_GTH_STR = """
Fe GTH-PBE-q16 GTH-PBE
    4    6    6
     0.36000000    2     6.69227147    -0.70908015
    3
     0.24927442    2     6.66136734    -3.50573466
                                        4.53064101
     0.25919008    2    -2.01467278     1.28419843
                                       -1.51887015
     0.22072392    1   -12.42768059
"""

SI_STRUCT = Structure(
    lattice=[[0, 2.734364, 2.734364], [2.734364, 0, 2.734364], [2.734364, 2.734364, 0]],
    species=["Si", "Si"],
    coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
)


class TestKeyword:
    def test_str_and_eq(self):
        kw = Keyword("CUTOFF", 400, description="cutoff", units="Ry", verbose=True)
        assert str(kw) == "CUTOFF [Ry] 400 ! cutoff"
        kw.verbosity(False)  # noqa: FBT003
        assert kw.get_str() == "CUTOFF [Ry] 400"
        assert kw[0] == 400
        # case-insensitive equality on name and string values; units matter
        assert Keyword("method", "qs") == Keyword("METHOD", "QS")
        assert Keyword("METHOD", "QS") != Keyword("METHOD", "GPW")
        assert Keyword("CUTOFF", 400, units="Ry") != Keyword("CUTOFF", 400)
        assert Keyword("A", 1) != Keyword("B", 1)
        assert Keyword("A", 1).__eq__("A 1") is NotImplemented

    def test_add_makes_keyword_list(self):
        kw_list = Keyword("A", 1) + Keyword("A", 2)
        assert isinstance(kw_list, KeywordList)
        assert [k.values for k in kw_list] == [(1,), (2,)]

    def test_dict_round_trip(self):
        kw = Keyword("A", 1, "x", 2.5, description="desc", units="eV", verbose=True, repeats=True)
        dct = kw.as_dict()
        assert dct["@class"] == "Keyword"
        kw2 = Keyword.from_dict(dct)
        assert kw2 == kw
        assert (kw2.description, kw2.units, kw2.verbose, kw2.repeats) == ("desc", "eV", True, True)

    def test_from_str(self):
        kw = Keyword.from_str("CUTOFF [Ry] 400", description="cut")
        assert kw.name == "CUTOFF"
        assert kw.units == "Ry"
        assert kw.values == (400,)
        assert kw.description == "cut"
        # ELEMENT values are never post-processed (e.g. "F" must not become False)
        assert Keyword.from_str("ELEMENT F", description=None).values == ("F",)
        assert Keyword.from_str("UKS T", description=None).values == (True,)


class TestKeywordList:
    def test_basics(self):
        kw_list = KeywordList([Keyword("A", 1), Keyword("a", 2)])
        assert kw_list.name == "A"
        assert len(kw_list) == 2
        assert kw_list[1] == Keyword("A", 2)
        assert str(kw_list) == "A 1 \na 2"
        assert kw_list.get_str(indent=1) == "\tA 1 \n\ta 2"
        kw_list.append(Keyword("A", 3))
        kw_list.extend([Keyword("A", 4)])
        assert len(kw_list) == 4
        assert KeywordList([]).name is None

        with pytest.raises(ValueError, match="some keyword is invalid"):
            KeywordList([Keyword("A", 1), Keyword("B", 1)])

    def test_eq(self):
        kl1 = KeywordList([Keyword("A", 1), Keyword("A", 2)])
        assert kl1 == KeywordList([Keyword("A", 1), Keyword("A", 2)])
        assert kl1 != KeywordList([Keyword("A", 2), Keyword("A", 1)])
        assert kl1.__eq__(Keyword("A", 1)) is NotImplemented

    def test_eq_different_length(self):
        assert KeywordList([Keyword("A", 1)]) != KeywordList([Keyword("A", 1), Keyword("A", 2)])

    def test_verbosity(self):
        kw_list = KeywordList([Keyword("A", 1, description="d1"), Keyword("A", 2, description="d2")])
        kw_list.verbosity(True)  # noqa: FBT003
        assert kw_list.get_str() == "A 1 ! d1 \nA 2 ! d2"

    def test_add(self):
        kw_list = KeywordList([Keyword("A", 1)])
        new = kw_list + [Keyword("A", 2)]  # noqa: RUF005
        assert isinstance(new, KeywordList)
        assert len(new) == 2


class TestSection:
    @pytest.fixture
    def section(self):
        sub = Section("SUB", keywords={"K1": Keyword("K1", 1)}, subsections={})
        return Section("TOP", subsections={"SUB": sub}, keywords={"K0": Keyword("K0", "a")}, EXTRA=5)

    def test_init_kwargs_and_getitem(self, section):
        assert section["EXTRA"] == Keyword("EXTRA", 5)
        assert section["k0"] == Keyword("K0", "a")
        assert section["sub"]["K1"] == Keyword("K1", 1)
        with pytest.raises(KeyError):
            section["MISSING"]
        assert section.get("MISSING", "default") == "default"
        assert section.get("SUB").name == "SUB"
        assert section.get_section("missing", 1) == 1
        assert section.get_keyword("missing", 2) == 2

    def test_get_str(self, section):
        assert section.get_str() == "&TOP\n\tK0 a\n\tEXTRA 5\n\t&SUB\n\t\tK1 1\n\t&END SUB\n&END TOP\n"
        assert str(section) == section.get_str()

        sec = Section("S", description="A described section", section_parameters=["ON"], verbose=True)
        assert sec.get_str() == "\n! A described section\n&S ON\n&END S\n"
        sec.verbosity(False)  # noqa: FBT003
        assert sec.get_str() == "&S ON\n&END S\n"

    def test_verbosity_recursive(self, section):
        section["SUB"]["K1"].description = "kw desc"
        section.verbosity(True)  # noqa: FBT003
        assert section["SUB"].verbose
        assert "K1 1 ! kw desc" in section.get_str()

    def test_eq_ignores_print(self, section):
        other = copy.deepcopy(section)
        other["SUB"].insert(Section("PRINT", subsections={}))
        assert section == other
        other["SUB"]["K1"] = 2
        assert section != other

    def test_add(self, section):
        section += Keyword("NEW", 1)
        assert section["NEW"] == Keyword("NEW", 1)
        section += Keyword("NEW", 2)
        assert isinstance(section["NEW"], KeywordList)
        assert len(section["NEW"]) == 2
        section += Section("SUB2", subsections={})
        assert section.check("SUB2")
        with pytest.raises(TypeError, match="Can only add sections or keywords"):
            section + 1
        assert section.add(Keyword("NEW2", 3))["NEW2"] == Keyword("NEW2", 3)
        with pytest.raises(TypeError, match="Can only add keywords, not Section"):
            section.add(Section("X"))

    def test_add_to_keyword_list(self):
        sec = Section("S", keywords={"A": KeywordList([Keyword("A", 1), Keyword("A", 2)])})
        sec += Keyword("A", 3)
        assert isinstance(sec["A"], KeywordList)
        assert len(sec["A"]) == 3

    def test_setitem(self, section):
        section["k0"] = "b"
        assert section["K0"] == Keyword("k0", "b")
        assert "K0" not in section.keywords  # replaced by the new key
        section["SUB"] = Section("SUB", keywords={"K2": Keyword("K2", 2)})
        assert section["SUB"].get("K1") is None
        section["NEWSUB"] = Section("NEWSUB")
        assert section.check("NEWSUB")

        # strict: no insertion of new keys
        section.setitem("NOPE", 1, strict=True)
        assert section.get("NOPE") is None
        section.setitem("NOPE", Section("NOPE"), strict=True)
        assert not section.check("NOPE")
        section.setitem("EXTRA", 6, strict=True)
        assert section["EXTRA"] == Keyword("EXTRA", 6)

    def test_delitem_and_sub(self, section):
        del section["sub"]
        assert not section.check("SUB")
        del section["k0"]
        assert section.get("K0") is None
        section - "EXTRA"
        assert section.get("EXTRA") is None
        with pytest.raises(KeyError, match="No section or keyword matching"):
            del section["MISSING"]

    def test_update(self, section):
        out = section.update(
            {
                "K0": "z",
                "KW": Keyword("KW", 1),
                "sub": {"K1": 10, "DEEP": {"K3": 3.0}},
                "NEW": {"K4": True},
                "SEC": Section("SEC", subsections={}),
            }
        )
        assert out is section
        assert section["K0"] == Keyword("K0", "z")
        assert section["KW"] == Keyword("KW", 1)
        assert section["SUB"]["K1"] == Keyword("K1", 10)
        assert section["SUB"]["DEEP"]["K3"] == Keyword("K3", 3.0)
        assert section["NEW"]["K4"] == Keyword("K4", True)  # noqa: FBT003
        assert section.check("SEC")

        with pytest.raises(TypeError, match="Unrecognized type"):
            section.update({"BAD": [1, 2]})

        # | is an alias for update
        section |= {"K0": "y"}
        assert section["K0"] == Keyword("K0", "y")

    def test_safeset_and_set(self, section):
        section.safeset({"K0": "safe", "MISSING": 1, "NOSEC": {"X": 1}, "SEC": Section("SEC"), "SUB": {"K1": 2}})
        assert section["K0"] == Keyword("K0", "safe")
        assert section.get("MISSING") is None
        assert not section.check("NOSEC")
        assert not section.check("SEC")
        assert section["SUB"]["K1"] == Keyword("K1", 2)

        section.set({"MISSING": 1})
        assert section["MISSING"] == Keyword("MISSING", 1)

    def test_unset(self, section):
        section.unset({"SUB": "K1"})
        assert section["SUB"].get("K1") is None
        section["SUB"]["K5"] = 5
        section.unset({"SUB": Keyword("K5", 5)})
        assert section["SUB"].get("K5") is None
        section.insert(Section("OUTER", subsections={"INNER": Section("INNER", keywords={"Q": Keyword("Q", 1)})}))
        section.unset({"OUTER": {"INNER": "Q"}})
        assert section["OUTER"]["INNER"].get("Q") is None
        with pytest.raises(TypeError, match="Can only add sections or keywords"):
            section.unset({"SUB": [1]})

    def test_inc(self, section):
        section.inc({"SUB": {"K1": 2, "LST": [1, 2]}})
        assert isinstance(section["SUB"]["K1"], KeywordList)
        assert section["SUB"]["LST"] == Keyword("LST", [1, 2])
        with pytest.raises(TypeError, match="Can only add sections or keywords"):
            section.inc({"X": None})

    def test_inc_section(self, section):
        section.inc({"NEWSEC": Section("NEWSEC")})
        assert section.check("NEWSEC")

    def test_insert(self, section):
        section.insert(Section("KIND", alias="KIND H", subsections={}))
        assert "KIND H" in section.subsections
        with pytest.raises(TypeError, match="type of d should be Section or SectionList, got Keyword"):
            section.insert(Keyword("A", 1))

    def test_check_and_by_path(self, section):
        section["SUB"].insert(Section("DEEP", subsections={}))
        assert section.check("sub/deep")
        assert not section.check("sub/nope")
        assert section.by_path("TOP/SUB/DEEP").name == "DEEP"
        assert section.by_path("sub/deep").name == "DEEP"

    def test_silence(self, section):
        section["SUB"].insert(Section("PRINT", subsections={}))
        section.insert(Section("PRINT", subsections={}))
        section.silence()
        assert not section.check("PRINT")
        assert not section.check("SUB/PRINT")

    def test_msonable(self, section):
        assert Section.from_dict(section.as_dict()) == section

    def test_eq_with_section_list(self):
        kpt_set = KpointSet(10, [("GAMMA", (0, 0, 0)), ("X", (0.5, 0, 0))])
        band = BandStructure([kpt_set, kpt_set])
        assert band == copy.deepcopy(band)


class TestSectionListExtra:
    def test_section_list(self):
        sec1 = Section("TEST", keywords={"A": Keyword("A", 1)}, alias="TEST 1")
        sec2 = Section("test", keywords={"A": Keyword("A", 2)})
        sec_list = SectionList([sec1, sec2])
        assert sec_list.alias == "TEST 1"
        assert sec_list.get("A") == Keyword("A", 2)
        assert sec_list.get("A", index=0) == Keyword("A", 1)
        assert str(sec_list) == "&TEST\n\tA 1\n&END TEST\n \n&test\n\tA 2\n&END test\n"
        sec_list.extend([Section("TEST")])
        assert len(sec_list) == 3
        sec_list.verbosity(True)  # noqa: FBT003
        assert all(sec.verbose for sec in sec_list)
        assert SectionList([]).name is None

        assert SectionList([sec1]) == SectionList([copy.deepcopy(sec1)])
        assert SectionList([sec1]) != SectionList([sec2])
        assert sec_list.__eq__(sec1) is NotImplemented

        with pytest.raises(ValueError, match="some section name is invalid"):
            SectionList([Section("A"), Section("B")])

    def test_eq_different_length(self):
        assert SectionList([Section("A")]) != SectionList([Section("A"), Section("A")])


class TestCp2kInputExtra:
    input_str = """
&FORCE_EVAL
  METHOD QS
  &SUBSYS
    &KIND H
      ELEMENT H
    &END KIND
    &KIND H
      BASIS_SET A
      BASIS_SET B
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
&MOTION
  &PRINT ON
  &END
&END MOTION
"""

    def test_from_str_section_lists(self):
        ci = Cp2kInput.from_str(self.input_str)
        kinds = ci["FORCE_EVAL"]["SUBSYS"]["KIND H"]
        assert isinstance(kinds, SectionList)
        assert len(kinds) == 2
        assert isinstance(kinds[1]["BASIS_SET"], KeywordList)
        # section parameter ON is a switch, not an alias
        assert ci.check("MOTION/PRINT")
        assert ci["MOTION"]["PRINT"].section_parameters == []

        # a third repeat appends to the existing SectionList
        ci2 = Cp2kInput.from_str("&A\n&B\nX 1\n&END B\n&B\nX 2\n&END B\n&B\nX 3\n&END B\n&END A")
        assert len(ci2["A"]["B"]) == 3
        assert [sec["X"] for sec in ci2["A"]["B"]] == [Keyword("X", 1), Keyword("X", 2), Keyword("X", 3)]

    def test_string_round_trip(self):
        ci = Cp2kInput.from_str(self.input_str)
        ci2 = Cp2kInput.from_str(ci.get_str())
        assert ci2.get_str() == ci.get_str()

    def test_repeated_keyword_order(self):
        ci = Cp2kInput.from_str("&A\nX 1\nX 2\nX 3\n&END A")
        assert [kw.values[0] for kw in ci["A"]["X"]] == [1, 2, 3]

    def test_write_file(self, tmp_path):
        ci = Cp2kInput.from_str(self.input_str)
        ci.write_file(input_filename="test.inp", output_dir=tmp_path / "new_dir")
        ci2 = Cp2kInput.from_file(tmp_path / "new_dir" / "test.inp")
        assert ci2.get_str() == ci.get_str()

    def test_dict_round_trip(self):
        ci = Cp2kInput.from_str(self.input_str)
        dct = ci.as_dict()
        assert Cp2kInput.from_dict(dct).get_str() == ci.get_str()
        ci2 = Cp2kInput._from_dict(dct)
        assert isinstance(ci2, Cp2kInput)
        assert ci2.get_str() == ci.get_str()


class TestConvenienceSections:
    def test_global(self):
        glob = Global(project_name="proj", run_type="GEO_OPT", keywords={"PRINT_LEVEL": Keyword("PRINT_LEVEL", "LOW")})
        assert glob.get_str() == (
            "&GLOBAL\n\tPROJECT_NAME proj\n\tRUN_TYPE GEO_OPT\n\tEXTENDED_FFT_LENGTHS True\n"
            "\tPRINT_LEVEL LOW\n&END GLOBAL\n"
        )
        assert Global.from_dict(glob.as_dict()) == glob

    def test_force_eval_and_subsys(self):
        fe = ForceEval(subsections={"SUBSYS": Subsys()})
        assert fe["METHOD"] == Keyword("METHOD", "QS")
        assert fe["STRESS_TENSOR"] == Keyword("STRESS_TENSOR", "ANALYTICAL")
        assert fe.repeats
        assert fe.check("SUBSYS")

    def test_dft(self):
        dft = Dft(basis_set_filenames=("BASIS_A", "BASIS_B"), uks=False, wfn_restart_file_name="wfn.wfn")
        assert isinstance(dft["BASIS_SET_FILE_NAME"], KeywordList)
        assert [k.values[0] for k in dft["BASIS_SET_FILE_NAME"]] == ["BASIS_A", "BASIS_B"]
        assert dft["UKS"] == Keyword("UKS", False)  # noqa: FBT003
        assert dft["WFN_RESTART_FILE_NAME"] == Keyword("WFN_RESTART_FILE_NAME", "wfn.wfn")
        assert "WFN_RESTART_FILE_NAME" not in Dft().keywords

    def test_qs_scf_mgrid(self):
        qs = QS(method="GAPW", eps_pgf_orb=1e-5)
        assert qs["METHOD"] == Keyword("METHOD", "GAPW")
        assert qs["EPS_PGF_ORB"] == Keyword("EPS_PGF_ORB", 1e-5)
        assert "EPS_PGF_ORB" not in QS().keywords

        scf = Scf(max_scf=10, eps_scf=1e-7, scf_guess="ATOMIC")
        assert (
            scf.get_str() == "&SCF\n\tMAX_SCF 10\n\tEPS_SCF 1e-07\n\tSCF_GUESS ATOMIC\n\tMAX_ITER_LUMO 400\n&END SCF\n"
        )

        mgrid = Mgrid(cutoff=500, rel_cutoff=50, ngrids=4, progression_factor=2)
        assert [mgrid[k].values[0] for k in ("CUTOFF", "REL_CUTOFF", "NGRIDS", "PROGRESSION_FACTOR")] == [500, 50, 4, 2]

    def test_diagonalization_davidson_ot(self):
        diag = Diagonalization(eps_iter=1e-6)
        assert diag["EPS_ITER"] == Keyword("EPS_ITER", 1e-6)
        assert diag.location == "CP2K_INPUT/FORCE_EVAL/DFT/SCF/DIAGONALIZATION"

        dav = Davidson(new_prec_each=10, preconditioner="FULL_ALL")
        assert dav.get_str() == "&DAVIDSON\n\tNEW_PREC_EACH 10\n\tPRECONDITIONER FULL_ALL\n&END DAVIDSON\n"

        ot = OrbitalTransformation(minimizer="DIIS", energy_gap=0.01, rotation=True)
        assert ot.name == "OT"
        assert ot["MINIMIZER"] == Keyword("MINIMIZER", "DIIS")
        assert ot["ENERGY_GAP"] == Keyword("ENERGY_GAP", 0.01)
        assert ot["ROTATION"] == Keyword("ROTATION", True)  # noqa: FBT003

    def test_cell(self):
        cell = Cell(SI_STRUCT.lattice)
        assert_allclose(cell["A"].values, SI_STRUCT.lattice.matrix[0])
        assert_allclose(cell["C"].values, SI_STRUCT.lattice.matrix[2])

    def test_kind(self):
        basis = GaussianTypeOrbitalBasisSet.from_str(O_DZVP_STR)
        pot = GthPotential.from_str(TestPotential.pot_hydrogen_str)
        kind = Kind("O", alias="O_1", basis_set=basis, potential=pot, magnetization=2, ghost=True)
        assert kind.section_parameters == ["O_1"]
        assert kind.alias == "O_1"
        assert kind["BASIS_SET"] == Keyword("BASIS_SET", "DZVP-MOLOPT-GTH")
        assert kind["POTENTIAL"] == Keyword("POTENTIAL", "GTH-PBE-q1")
        assert kind["GHOST"] == Keyword("GHOST", True)  # noqa: FBT003
        assert kind["MAGNETIZATION"] == Keyword("MAGNETIZATION", 2)

        aux = GaussianTypeOrbitalBasisSet(name="cFIT3", info=BasisInfo(admm=True))
        kind = Kind("O", basis_set="DZVP", potential=None, aux_basis=aux)
        assert isinstance(kind["BASIS_SET"], KeywordList)
        assert kind["BASIS_SET"].get_str() == "BASIS_SET DZVP \nBASIS_SET AUX_FIT cFIT3"
        assert kind.get("POTENTIAL") is None

    def test_kind_closed_shell_magnetization(self):
        kind = Kind("Ne", magnetization=2)
        assert kind.magnetization == 0
        assert kind["MAGNETIZATION"] == Keyword("MAGNETIZATION", 0)

    def test_kind_str_aux_basis(self):
        kind = Kind("H", basis_set="DZVP", aux_basis="cFIT3")
        assert kind["BASIS_SET"][1].get_str() == "BASIS_SET AUX_FIT cFIT3"

    def test_dft_plus_u(self):
        dftu = DftPlusU(l=2, u_minus_j=3.0, u_ramping=0.1)
        assert dftu.name == "DFT_PLUS_U"
        assert dftu["L"] == Keyword("L", 2)
        assert dftu["U_MINUS_J"] == Keyword("U_MINUS_J", 3.0)
        assert dftu["U_RAMPING"] == Keyword("U_RAMPING", 0.1)

    def test_dft_plus_u_attributes(self):
        dftu = DftPlusU(eps_u_ramping=1e-3, init_u_ramping_each_scf=True)
        assert dftu.eps_u_ramping == approx(1e-3)
        assert dftu.init_u_ramping_each_scf

    def test_coord_aliases(self):
        coord = Coord(SI_STRUCT, aliases={"Si_1": [0], "Si_2": [1]})
        assert coord.keywords[0].name == "Si_1"
        assert coord.keywords[1].name == "Si_2"
        assert_allclose(coord.keywords[1].values, SI_STRUCT[1].coords)
        assert Coord(SI_STRUCT).keywords[0].name == "Si"

    def test_print_sections(self):
        assert DOS(ndigits=8)["NDIGITS"] == Keyword("NDIGITS", 8)
        pdos = PDOS(nlumo=10)
        assert pdos["NLUMO"] == Keyword("NLUMO", 10)
        assert pdos["COMPONENTS"].values == ()
        ldos = LDOS(index=3, alias="LDOS 3")
        assert ldos["LIST"] == Keyword("LIST", 3)
        assert ldos.alias == "LDOS 3"
        assert VHartreeCube().get_str() == "&V_HARTREE_CUBE\n&END V_HARTREE_CUBE\n"
        mo_cubes = MOCubes(write_cube=True, nhomo=2, nlumo=3)
        assert mo_cubes.get_str() == "&MO_CUBES\n\tWRITE_CUBE True\n\tNHOMO 2\n\tNLUMO 3\n&END MO_CUBES\n"
        e_dens = EDensityCube(keywords={"STRIDE": Keyword("STRIDE", 1)})
        assert e_dens.get_str() == "&E_DENSITY_CUBE\n\tSTRIDE 1\n&END E_DENSITY_CUBE\n"

    def test_smear(self):
        smear = Smear(elec_temp=500, method="GAUSSIAN")
        assert smear.get_str() == (
            "&SMEAR\n\tELEC_TEMP 500\n\tMETHOD GAUSSIAN\n\tFIXED_MAGNETIC_MOMENT -100.0\n&END SMEAR\n"
        )

    def test_broken_symmetry(self):
        bs = BrokenSymmetry(l_alpha=(0, 2), n_alpha=(4, 3), nel_alpha=(-1, 1))
        assert bs["ALPHA"]["L"] == Keyword("L", 0, 2)
        assert bs["ALPHA"]["NEL"] == Keyword("NEL", -1, 1)
        assert bs["BETA"]["L"] == Keyword("L", -1)

        bs = BrokenSymmetry.from_el("Fe", oxi_state=2)
        assert bs["ALPHA"]["L"] == Keyword("L", 0)
        assert bs["ALPHA"]["N"] == Keyword("N", 4)
        assert bs["ALPHA"]["NEL"] == Keyword("NEL", -2)

        bs = BrokenSymmetry.from_el(Element("Fe"), oxi_state=3, spin="high-up")
        assert bs["ALPHA"]["L"] == Keyword("L", 0, 2)
        assert bs["ALPHA"]["N"] == Keyword("N", 4, 3)
        assert bs["ALPHA"]["NEL"] == Keyword("NEL", -2, -1)

        # no change in occupation without oxidation state or spin
        assert BrokenSymmetry.from_el("Fe")["ALPHA"]["L"] == Keyword("L")

    def test_broken_symmetry_spin_aliases(self):
        def nel(bs):
            return bs["ALPHA"]["NEL"].values, bs["BETA"]["NEL"].values

        # Fe3+ leaves 3d5: low spin aliases resolve to +/-1, high spin to 5 % 5 == 0
        assert nel(BrokenSymmetry.from_el("Fe", 3, spin="low-up")) == nel(BrokenSymmetry.from_el("Fe", 3, spin=1))
        assert nel(BrokenSymmetry.from_el("Fe", 3, spin="low-down")) == nel(BrokenSymmetry.from_el("Fe", 3, spin=-1))
        no_spin = nel(BrokenSymmetry.from_el("Fe", 3))
        assert nel(BrokenSymmetry.from_el("Fe", 3, spin="high-up")) == no_spin
        assert nel(BrokenSymmetry.from_el("Fe", 3, spin="high-down")) == no_spin
        # spin only redistributes electrons between the alpha and beta channels
        for spin in (1, -1):
            alpha, beta = nel(BrokenSymmetry.from_el("Fe", 3, spin=spin))
            assert sum(alpha) + sum(beta) == sum(no_spin[0]) + sum(no_spin[1])
            assert alpha != beta

    def test_broken_symmetry_spin_sign_symmetry(self):
        up = BrokenSymmetry.from_el("Fe", 3, spin=1)
        down = BrokenSymmetry.from_el("Fe", 3, spin=-1)
        assert down["ALPHA"]["NEL"] == up["BETA"]["NEL"]
        assert down["BETA"]["NEL"] == up["ALPHA"]["NEL"]

    def test_broken_symmetry_f_orbital(self):
        bs = BrokenSymmetry.from_el("Ce", oxi_state=4)
        assert bs["ALPHA"]["L"] == Keyword("L", 0, 2, 3)

    def test_xc_functional_and_pbe(self):
        xc = XCFunctional(functionals=["PBE", "LYP"])
        assert xc.check("PBE")
        assert xc.check("LYP")
        pbe = PBE(parameterization="PBESOL", scale_x=0.75)
        assert pbe.get_str() == "&PBE\n\tPARAMETRIZATION PBESOL\n\tSCALE_C 1\n\tSCALE_X 0.75\n&END PBE\n"


class TestKpoints:
    def test_explicit(self):
        kpts = Kpoints(kpts=[(0, 0, 0), (0.5, 0, 0)], weights=[1, 3])
        assert isinstance(kpts["KPOINT"], KeywordList)
        assert kpts["KPOINT"].get_str() == "KPOINT 0 0 0 1 \nKPOINT 0.5 0 0 3"
        assert kpts["SCHEME"] == Keyword("SCHEME", "MONKHORST-PACK")

        with pytest.raises(ValueError, match="lengths of kpts 2 and weights 1 mismatch"):
            Kpoints(kpts=[(0, 0, 0), (0.5, 0, 0)], weights=[1])
        with pytest.raises(ValueError, match="No k-points provided"):
            Kpoints(kpts=[])

    def test_single_mesh(self):
        kpts = Kpoints(kpts=[(2, 3, 4)])
        assert kpts["SCHEME"] == Keyword("SCHEME", "MONKHORST-PACK", 2, 3, 4)

    def test_from_kpoints(self):
        kpts = Kpoints.from_kpoints(VaspKpoints.monkhorst_automatic((2, 3, 4)))
        assert kpts["SCHEME"] == Keyword("SCHEME", "MONKHORST-PACK 2 3 4")

        recip = VaspKpoints(
            style=KpointsSupportedModes.Reciprocal, num_kpts=2, kpts=[(0, 0, 0), (0.5, 0, 0)], kpts_weights=[1, 1]
        )
        kpts = Kpoints.from_kpoints(recip)
        assert kpts["SCHEME"] == Keyword("SCHEME", "GENERAL")
        assert kpts["UNITS"] == Keyword("UNITS", "B_VECTOR")
        assert len(kpts["KPOINT"]) == 2

        cart = VaspKpoints(
            style=KpointsSupportedModes.Cartesian, num_kpts=2, kpts=[(0, 0, 0), (0.5, 0, 0)], kpts_weights=[1, 1]
        )
        assert Kpoints.from_kpoints(cart)["UNITS"] == Keyword("UNITS", "CART_ANGSTROM")

        line = VaspKpoints.automatic_linemode(5, HighSymmKpath(SI_STRUCT))
        kpts = Kpoints.from_kpoints(line)
        assert kpts["SCHEME"] == Keyword("SCHEME", "GENERAL")
        assert len(kpts["KPOINT"]) == len(line.kpts)

    def test_from_kpoints_gamma(self):
        gamma = Kpoints.from_kpoints(VaspKpoints.gamma_automatic((1, 1, 1)), structure=SI_STRUCT)
        assert gamma["SCHEME"] == Keyword("SCHEME", "GAMMA")

        mesh = Kpoints.from_kpoints(VaspKpoints.gamma_automatic((2, 2, 2)), structure=SI_STRUCT)
        assert mesh["SCHEME"] == Keyword("SCHEME", "GENERAL")
        assert sum(kw.values[-1] for kw in mesh["KPOINT"]) == 8

        with pytest.raises(ValueError, match="A structure is required"):
            Kpoints.from_kpoints(VaspKpoints.gamma_automatic((2, 2, 2)))

    def test_kpoint_set_and_band_structure(self):
        kpt_set = KpointSet(npoints=10, kpoints=[("\\Gamma", (0, 0, 0)), ("X", (0.5, 0, 0))])
        assert kpt_set["SPECIAL_POINT"].get_str() == "SPECIAL_POINT Gamma 0 0 0 \nSPECIAL_POINT X 0.5 0 0"
        assert kpt_set["NPOINTS"] == Keyword("NPOINTS", 10)

        band = BandStructure([kpt_set], filename="bands.bs", added_mos=5)
        assert band["FILE_NAME"] == Keyword("FILE_NAME", "bands.bs")
        assert band["ADDED_MOS"] == Keyword("ADDED_MOS", 5)
        assert isinstance(band["KPOINT_SET"], SectionList)
        assert "&KPOINT_SET" in band.get_str()

        line = VaspKpoints(
            style=KpointsSupportedModes.Line_mode,
            num_kpts=10,
            kpts=[(0, 0, 0), (0.5, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0)],
            labels=["G", "X", "X", "M"],
            coord_type="Reciprocal",
        )
        band = BandStructure.from_kpoints(line, kpoints_line_density=15)
        assert len(band["KPOINT_SET"]) == 2
        assert band["KPOINT_SET"][1]["SPECIAL_POINT"].get_str() == (
            "SPECIAL_POINT X 0.5 0 0 \nSPECIAL_POINT M 0.5 0.5 0"
        )
        assert band["KPOINT_SET"][0]["NPOINTS"] == Keyword("NPOINTS", 15)

        recip = VaspKpoints(
            style=KpointsSupportedModes.Reciprocal,
            num_kpts=2,
            kpts=[(0, 0, 0), (0.5, 0, 0)],
            kpts_weights=[1, 1],
            coord_type="Reciprocal",
        )
        band = BandStructure.from_kpoints(recip)
        assert len(band["KPOINT_SET"]) == 1
        assert band["KPOINT_SET"][0]["UNITS"] == Keyword("UNITS", "B_VECTOR")
        assert band["KPOINT_SET"][0]["NPOINTS"] == Keyword("NPOINTS", 1)

        with pytest.raises(ValueError, match="Unsupported k-point style"):
            BandStructure.from_kpoints(VaspKpoints.gamma_automatic())


class TestBasisAndPotentialExtra:
    def test_basis_info_variants(self):
        info = BasisInfo.from_str("LRI-DZVP-MOLOPT-GTH-AUG")
        assert info.lri
        assert info.valence == 2
        assert info.diffuse == 1

        info = BasisInfo.from_str("TZV2PX-MOLOPT-GTH")
        assert (info.valence, info.polarization, info.diffuse) == (3, 2, 1)
        assert info.core is None

        info = BasisInfo.from_str("ADMM-cpFIT3")
        assert info.admm
        assert info.contracted
        assert info.valence == 3

        assert BasisInfo.from_str("DZVP-2X").diffuse == 2

        assert not info.softmatch("not a basis info")
        assert not PotentialInfo().softmatch(info)

    def test_atomic_metadata(self):
        meta = AtomicMetadata(name="X")
        assert meta.get_str() == str(meta)
        assert meta.get_hash() == AtomicMetadata(name="x").get_hash()  # hash is case-insensitive
        # names are soft-matched against the other object's aliases too
        pot = GthPotential(info=PotentialInfo(), name="A", alias_names=["B"])
        assert GthPotential(info=PotentialInfo(), name="B", alias_names=["A"]).softmatch(pot)

    def test_gto_round_trip(self):
        gto = GaussianTypeOrbitalBasisSet.from_str(O_DZVP_STR)
        assert gto.nshell == [{0: 2, 1: 2, 2: 1}]
        assert gto.info.electrons == 6
        assert gto.potential == "Pseudopotential"
        assert gto.coefficients[0][0][2][0] == approx(0.0148070544)
        assert gto.get_str().splitlines()[:3] == ["O DZVP-MOLOPT-GTH DZVP-MOLOPT-GTH-q6", "1", "2 0 2 7 2 2 1"]
        assert GaussianTypeOrbitalBasisSet.from_str(gto.get_str()) == gto
        assert GaussianTypeOrbitalBasisSet.from_dict(gto.as_dict()) == gto
        assert len(gto.get_hash()) == 32

        with pytest.raises(ValueError, match="No name attribute"):
            GaussianTypeOrbitalBasisSet().get_keyword()
        with pytest.raises(ValueError, match="Must have all attributes defined"):
            GaussianTypeOrbitalBasisSet().get_str()

    def test_gto_all_electron(self):
        gto = GaussianTypeOrbitalBasisSet(
            element=Element("O"), name="ALLELECTRON", info=BasisInfo(), potential="All Electron"
        )
        assert gto.name == "ALL"
        assert gto.info.electrons == 8

        gto = GaussianTypeOrbitalBasisSet.from_str("H 6-31G ALL\n 1\n 1 0 0 1 1\n 0.5 1.0\n")
        assert gto.potential == "All Electron"
        assert gto.info.electrons == 1

    def test_gto_softmatch(self):
        gto = GaussianTypeOrbitalBasisSet.from_str(O_DZVP_STR)
        assert GaussianTypeOrbitalBasisSet(info=BasisInfo(molopt=True), element=Element("O")).softmatch(gto)
        assert GaussianTypeOrbitalBasisSet(name="DZVP-MOLOPT-GTH-q6").softmatch(gto)
        assert not GaussianTypeOrbitalBasisSet(info=BasisInfo(cc=True)).softmatch(gto)
        assert not GaussianTypeOrbitalBasisSet(element=Element("H")).softmatch(gto)
        assert not GaussianTypeOrbitalBasisSet(potential="All Electron").softmatch(gto)
        assert not GaussianTypeOrbitalBasisSet(name="other").softmatch(gto)
        assert not GaussianTypeOrbitalBasisSet().softmatch(GthPotential(info=PotentialInfo()))

    def test_gth_nonlocal_round_trip(self):
        pot = GthPotential.from_str(FE_GTH_STR)
        assert pot.n_elecs == {0: 4, 1: 6, 2: 6}
        assert pot.info.electrons == 16
        assert pot.nprj == 3
        assert pot.nprj_ppnl == {0: 2, 1: 2, 2: 1}
        assert pot.hprj_ppnl[0] == {0: {0: approx(6.66136734), 1: approx(-3.50573466)}, 1: {0: approx(4.53064101)}}
        assert pot.radii[2] == approx(0.22072392)
        assert GthPotential.from_str(pot.get_str()) == pot
        assert GthPotential.from_dict(pot.as_dict()) == pot
        assert GthPotential.from_section(pot.get_section()) == pot
        assert pot.get_section().name == "GTH-PBE-q16"

        assert GthPotential(info=PotentialInfo(xc="PBE"), element=Element("Fe"), name="GTH-PBE").softmatch(pot)
        assert not GthPotential(info=PotentialInfo(), potential="All Electron").softmatch(pot)

    def test_gth_errors_and_all_electron(self):
        pot = GthPotential(info=PotentialInfo(), element=Element("Fe"), potential="All Electron", name="ALLELECTRON")
        assert pot.name == "ALL"
        assert pot.info.electrons == 26
        empty = GthPotential(info=PotentialInfo())
        with pytest.raises(ValueError, match="Must initialize all attributes"):
            empty.get_str()
        with pytest.raises(ValueError, match="Cannot get keyword without name"):
            empty.get_keyword()
        with pytest.raises(ValueError, match="Cannot get section without name"):
            empty.get_section()

    def test_data_files(self, tmp_path):
        basis_file = BasisFile.from_str(BASIS_FILE_STR + O_DZVP_STR)
        assert [obj.element for obj in basis_file.objects] == [Element("H"), Element("O")]
        basis_file.write_file(tmp_path / "BASIS")
        basis_file2 = BasisFile.from_file(tmp_path / "BASIS")
        assert all(obj.filename == tmp_path / "BASIS" for obj in basis_file2.objects)
        assert [obj.get_str() for obj in basis_file2.objects] == [obj.get_str() for obj in basis_file.objects]
        assert str(basis_file) == basis_file.get_str()

        pot_file = PotentialFile.from_str(TestPotential.pot_hydrogen_str + FE_GTH_STR)
        assert [obj.name for obj in pot_file.objects] == ["GTH-PBE-q1", "GTH-PBE-q16"]
        assert PotentialFile.from_str(pot_file.get_str()).objects == pot_file.objects
        assert DataFile().get_str() == ""
