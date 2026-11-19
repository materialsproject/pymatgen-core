from __future__ import annotations

import re
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pymatgen.core import DummySpecies, Lattice, Species
from pymatgen.core.structure import Site, Structure
from pymatgen.core.units import bohr_to_ang
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.jdftx.generic_tags import JDFTXOptionWarning, MultiformatTag
from pymatgen.io.jdftx.inputs import (
    JDFTXInfile,
    JDFTXStructure,
    _get_nanmaps,
    _infile_to_pmg_lattice,
    _remove_at_path,
    clean_infile_of_nans,
    selective_dynamics_site_prop_to_jdftx_interpretable,
)
from pymatgen.io.jdftx.jdftxinfile_default_inputs import antoinePvap, default_inputs
from pymatgen.io.jdftx.jdftxinfile_master_format import get_tag_object
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.io.vasp import Poscar

from .inputs_test_utils import (
    assert_equiv_jdftxstructure,
    assert_identical_jif,
    ex_in_files_dir,
    ex_infile1_fname,
    ex_infile1_knowns,
    ex_infile2_fname,
    ex_infile3_fname,
)
from .outputs_test_utils import pcm_params_outfile_path
from .shared_test_utils import assert_same_value

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.io.jdftx.generic_tags import TagContainer
    from pymatgen.util.typing import PathLike


def test_jdftxinfile_structuregen():
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    jdftxstruct = jif.to_jdftxstructure(jif)
    assert isinstance(jdftxstruct, JDFTXStructure)


@pytest.mark.parametrize(
    ("infile_fname", "bool_func"),
    [
        (ex_infile1_fname, lambda jif: all(jif["kpoint-folding"][x] == 1 for x in jif["kpoint-folding"])),
        (ex_infile1_fname, lambda jif: jif["elec-n-bands"]["n"] == 15),
    ],
)
def test_JDFTXInfile_known_lambda(infile_fname: str, bool_func: Callable[[JDFTXInfile], bool]):
    jif = JDFTXInfile.from_file(infile_fname)
    assert bool_func(jif)


def JDFTXInfile_self_consistency_tester(jif: JDFTXInfile, tmp_path: PathLike):
    """Create an assortment of JDFTXinfile created from the same data but through different methods, and test that
    they are all equivalent through "assert_identical_jif" """
    dict_jif = jif.as_dict()
    jif2 = JDFTXInfile.get_dict_representation(JDFTXInfile._from_dict(dict_jif))
    str_list_jif = jif.get_text_list()
    str_jif = "\n".join(str_list_jif)
    jif3 = JDFTXInfile.from_str(str_jif)
    tmp_fname = tmp_path / "tmp.in"
    jif.write_file(tmp_fname)
    jif4 = JDFTXInfile.from_file(tmp_fname)
    jifs = [jif, jif2, jif3, jif4]
    for i in range(len(jifs)):
        for j in range(i + 1, len(jifs)):
            assert_identical_jif(jifs[i], jifs[j])


def test_JDFTXInfile_from_dict(tmp_path) -> None:
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    jif_dict = jif.as_dict()
    # Test that dictionary can be modified and that _from_dict will fix set values
    jif_dict["elec-cutoff"] = 20
    jif2 = JDFTXInfile.from_dict(jif_dict)
    JDFTXInfile_self_consistency_tester(jif2, tmp_path)


@pytest.mark.parametrize("infile_fname", [ex_infile3_fname, ex_infile1_fname, ex_infile2_fname])
def test_JDFTXInfile_self_consistency_fromfile(infile_fname: PathLike, tmp_path) -> None:
    """Test that JDFTXInfile objects with different assortments of tags survive inter-conversion done within
    "JDFTXInfile_self_consistency_tester"""
    jif = JDFTXInfile.from_file(infile_fname)
    JDFTXInfile_self_consistency_tester(jif, tmp_path)


@pytest.mark.parametrize(
    ("val_key", "val"),
    [
        ("lattice", np.eye(3)),
        ("fluid-solvent", "H2O 0.5"),
        ("fluid-solvent", "H2O"),
        ("latt-scale", "1 1 1"),
        ("latt-scale", ["1", "1", "1"]),
        ("latt-scale", [1, 1, 1]),
        ("latt-scale", {"s0": 1, "s1": 1, "s2": 1}),
        ("elec-cutoff", {"Ecut": 20.0, "EcutRho": 100.0}),
        ("elec-cutoff", "20 100"),
        ("elec-cutoff", [20, 100]),
        ("elec-cutoff", 20),
    ],
)
def test_JDFTXInfile_set_values(val_key: str, val: Any, tmp_path) -> None:
    """Test value setting for various tags"""
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    jif[val_key] = val
    # Test that the JDFTXInfile object is still consistent
    JDFTXInfile_self_consistency_tester(jif, tmp_path)


@pytest.mark.parametrize(
    ("val_key", "val"),
    [
        ("fluid-solvent", "H2O"),
        ("dump", "End DOS"),
        ("dump", "End DOS BandEigs"),
        ("dump-interval", "Electronic 1"),
        ("ion", "Fe 1 1 1 0"),
    ],
)
def test_JDFTXInfile_append_values(val_key: str, val: Any, tmp_path) -> None:
    """Test the append_tag method"""
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    val_old = None if val_key not in jif else deepcopy(jif[val_key])
    jif.append_tag(val_key, val)
    val_new = jif[val_key]
    assert val_old != val_new
    # Test that the append_tag does not break the JDFTXInfile object
    JDFTXInfile_self_consistency_tester(jif, tmp_path)


def test_JDFTXInfile_expected_exceptions():
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    with pytest.raises(KeyError):
        jif["barbie"] = "ken"
    # non-repeating tags raise value-errors when appended
    tag = "initial-state"
    with pytest.raises(ValueError, match=re.escape(f"The tag '{tag}' cannot be repeated and thus cannot be appended")):
        jif.append_tag(tag, "$VAR")
    # Phonon and Wannier tags raise value-errors at _preprocess_line
    with pytest.raises(ValueError, match="Phonon functionality has not been added!"):
        jif._preprocess_line("phonon idk")
    with pytest.raises(ValueError, match="Wannier functionality has not been added!"):
        jif._preprocess_line("wannier idk")
    # Tags not in MASTER_TAG_LIST raise value-errors at _preprocess_line
    err_str = f"The barbie tag in {['barbie', 'ken allan']} is not in MASTER_TAG_LIST and is not a comment, "
    "something is wrong with this input data!"
    with pytest.raises(ValueError, match=re.escape(err_str)):
        jif._preprocess_line("barbie ken allan")
    # include tags raise warnings if the file cannot be found
    _filename = "barbie"
    err_str = f"The include file {_filename} ({_filename}) does not exist!"
    with pytest.warns(UserWarning, match=re.escape(err_str)):
        JDFTXInfile.from_str(f"include {_filename}\n", dont_require_structure=True)
    # With a path_parent, the file is also looked for there, and the warning names that path
    filename = ex_in_files_dir / "barbie"
    err_str = f"The include file {_filename} ({filename}) does not exist!"
    with pytest.warns(UserWarning, match=re.escape(err_str)):
        JDFTXInfile.from_str(f"include {_filename}\n", path_parent=ex_in_files_dir, dont_require_structure=True)
    # JDFTXInfile cannot be constructed without any structure tags (lattice, ion, ion-species)
    with pytest.raises(ValueError, match="This input file is missing required structure tags"):
        JDFTXInfile.from_str("dump End DOS\n")
    # "barbie" here is supposed to be "list-to-dict" or "dict-to-list"
    with pytest.raises(ValueError, match="Conversion type barbie is not 'list-to-dict' or 'dict-to-list'"):
        jif._needs_conversion("barbie", ["ken"])
    # Setting tags with unfixable values immediately raises an error
    tag = "exchange-params"
    value = {"blockSize": 1, "nOuterVxx": "barbie"}
    err_str = str(f"The {tag} tag with value:\n{value}\ncould not be fixed!")
    with pytest.raises(ValueError, match=re.escape(err_str)):
        # Implicitly tests validate_tags
        jif[tag] = value
    # `copy` will end up trying to validate the tags, so we need to remove the offending tag first
    jif.pop(tag)
    jif2 = jif.copy()
    # Setting tags with unfixable values through "update" side-steps the error, but will raise it once
    # "validate_tags" is inevitably called
    jif2.update({tag: value})
    with pytest.raises(ValueError, match=re.escape(err_str)):
        jif2.validate_tags(try_auto_type_fix=True)
    # The inevitable error can be reduced to a warning if you tell it not to try to fix the values
    with pytest.warns(UserWarning, match="The exchange-params tag with value"):
        jif2.validate_tags(try_auto_type_fix=False)
    # Setting a non-string tag raises an error within the JDFTXInfile object
    err_str = str(f"{1.2} is not a string!")
    with pytest.raises(TypeError, match=err_str):
        jif[1.2] = 3.4


def test_include_splices_lines(tmp_path):
    """Included lines are read in place of the include line, as JDFTx does, so repeatable tags are added to."""
    (tmp_path / "main.in").write_text(
        "lattice Cubic 10\nion-species GBRV/$ID_pbe.uspp\nion H 0 0 0 1\ninclude ions.in\nion H 0 0 1.5 1\n"
    )
    (tmp_path / "ions.in").write_text("ion-species SG15/$ID_ONCV_PBE.upf\nion O 0 1.5 0 1\ninclude more_ions.in\n")
    # Includes within included files are also found next to the main file
    (tmp_path / "more_ions.in").write_text("ion H 0 0 3 1\n")
    jif = JDFTXInfile.from_file(tmp_path / "main.in")
    assert jif["ion-species"] == ["GBRV/$ID_pbe.uspp", "SG15/$ID_ONCV_PBE.upf"]
    assert [(ion["species-id"], ion["x0"], ion["x1"], ion["x2"]) for ion in jif["ion"]] == [
        ("H", 0, 0, 0),
        ("O", 0, 1.5, 0),
        ("H", 0, 0, 3),
        ("H", 0, 0, 1.5),
    ]
    assert "include" not in jif
    # An include line for a missing file is kept
    with pytest.warns(UserWarning, match="does not exist"):
        jif = JDFTXInfile.from_str("include barbie\n", dont_require_structure=True)
    assert jif["include"] == ["barbie"]


def test_include_errors(tmp_path):
    """Inputs JDFTx refuses to read because of their include files raise errors."""
    # A tag that cannot repeat is given in both files
    (tmp_path / "main.in").write_text("lattice Cubic 10\nelec-cutoff 20\ninclude cutoff.in\n")
    (tmp_path / "cutoff.in").write_text("elec-cutoff 30\n")
    with pytest.raises(ValueError, match="The 'elec-cutoff' tag appears multiple times"):
        JDFTXInfile.from_file(tmp_path / "main.in", dont_require_structure=True)
    # Cyclic includes
    (tmp_path / "a.in").write_text("include b.in\n")
    (tmp_path / "b.in").write_text("include a.in\n")
    with pytest.raises(ValueError, match=re.escape("(in)directly includes itself")):
        JDFTXInfile.from_file(tmp_path / "a.in", dont_require_structure=True)


def test_JDFTXInfile_strip_structure():
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    jif["latt-scale"] = {"s0": 2.0, "s1": 2.0, "s2": 1.0}
    structural_tags = ["lattice", "ion", "coords-type", "latt-scale"]
    assert all(tag in jif for tag in structural_tags)
    jif.strip_structure_tags()
    assert all(tag not in jif for tag in structural_tags)


def test_JDFTXInfile_niche_cases():
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    tag_object, tag, value = jif._preprocess_line("dump-only")
    assert value == ("")
    tag = "elec-ex-corr"
    tag_object = get_tag_object(tag)
    value = "gga"
    params = jif.as_dict()
    err_str = f"The '{tag}' tag appears multiple times in this input when it should not!"
    with pytest.raises(ValueError, match=err_str):
        jif._store_value(params, tag_object, tag, value)
    struct = jif.to_pmg_structure(jif)
    assert isinstance(struct, Structure)
    noneout = jif.validate_tags(return_list_rep=True)
    assert noneout is None
    jif["fluid-solvent"] = {"name": "H2O", "concentration": 0.5}
    assert len(jif["fluid-solvent"]) == 1
    jif.append_tag("fluid-solvent", {"name": "H2O", "concentration": 0.5})
    assert len(jif["fluid-solvent"]) == 2


def test_JDFTXInfile_add_method():
    """Test the __add__ method"""
    # No new values are being assigned in jif2, so jif + jif2 should be the same as jif
    # Since the convenience of this method would be lost if the user has to pay special attention to duplicating
    # repeatable values, entries of a repeatable tag that both objects have are not added a second time
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    jif2 = jif.copy()
    assert jif2 is not jif  # Testing robustness of copy method while we are at it
    jif3 = jif + jif2
    assert_identical_jif(jif, jif3)
    # If a non-repeatable tag is in both objects, the value of the second one is used
    key = "elec-ex-corr"
    val_old = deepcopy(jif[key])
    val_new = "lda"
    assert val_old != val_new
    jif2[key] = val_new
    jif4 = jif + jif2
    assert_same_value(jif4[key], val_new)  # Make sure addition chooses second value for non-repeatable tags
    del jif4
    jif2.append_tag("dump", "Fluid State")
    jif.append_tag("dump", "Fluid Berry")
    jif4 = jif + jif2
    assert len(jif4["dump"]) == len(jif2["dump"]) + 1
    assert {"Fluid": {"State": True}} in jif4["dump"]
    assert {"Fluid": {"State": True}} not in jif["dump"]
    assert {"Fluid": {"Berry": True}} in jif4["dump"]
    assert {"Fluid": {"Berry": True}} not in jif2["dump"]
    # Normal expected behavior
    key_add = "target-mu"
    val_add = 0.5
    assert key_add not in jif
    jif2 = jif.copy()
    jif2[key_add] = val_add
    jif3 = jif + jif2
    assert jif3[key_add]["mu"] == pytest.approx(val_add)


@pytest.mark.parametrize(("infile_fname", "knowns"), [(ex_infile1_fname, ex_infile1_knowns)])
def test_JDFTXInfile_knowns_simple(infile_fname: PathLike, knowns: dict):
    """Test that known values that can be tested with assert_same_value are correct"""
    jif = JDFTXInfile.from_file(infile_fname)
    for key, val in knowns.items():
        assert_same_value(jif[key], val)


def test_jdftxstructure():
    """Test the JDFTXStructure object associated with the JDFTXInfile object"""
    jif = JDFTXInfile.from_file(ex_infile2_fname)
    jstruct = jif.to_jdftxstructure(jif)
    assert isinstance(jstruct, JDFTXStructure)
    struc_str = str(jstruct)
    assert isinstance(struc_str, str)
    newstruct = JDFTXStructure.from_str(struc_str)
    assert isinstance(newstruct, JDFTXStructure)
    # Double checking I got the column/row order right
    assert_same_value(jstruct.structure.lattice, newstruct.structure.lattice)
    assert jstruct.natoms == 16
    with open(ex_infile2_fname, encoding="utf-8") as f:
        lines = list.copy(list(f))
    # Test different ways of creating a JDFTXStructure object create the same object if data is the same
    data = "\n".join(lines)
    struct2 = JDFTXStructure.from_str(data)
    assert_equiv_jdftxstructure(jstruct, struct2)
    struct3 = JDFTXStructure.from_dict(jstruct.as_dict())
    assert_equiv_jdftxstructure(jstruct, struct3)


def test_disordered_structures():
    """Test that disordered structures are handled correctly"""
    jif = JDFTXInfile.from_file(ex_infile2_fname)
    struct = jif.structure
    # Making a disordered structure by replacing some sites with partial occupancies
    struct.sites[1] = Site(species={"Si": 0.5, "Ge": 0.5}, coords=struct.sites[1].frac_coords)
    with pytest.raises(
        ValueError, match="Disordered structure with partial occupancies cannot be converted into JDFTXStructure!"
    ):
        JDFTXStructure(structure=struct)


def test_jdftxstructure_infile_site_properties_conversion():
    """Test conversions related to going back and forth between JDFTXInfile and JDFTXStructure/Structure"""
    jif = JDFTXInfile.from_file(ex_infile2_fname)
    jif["ion"][0]["moveScale"] = 0.0
    jif["ion"][1]["moveScale"] = 1.0
    for i in range(len(jif["ion"])):
        jif["ion"][i]["v"] = {"vx0": 0.0, "vx1": 0.0, "vx2": 0.0}
    jif["ion"][3]["v"] = {"vx0": 1.0, "vx1": 2.0, "vx2": 3.0}
    jstruct = jif.to_jdftxstructure(jif)
    struct = jif.structure
    # moveScale is stored as the movescale site property, and gives the selective dynamics (fixed only for 0)
    assert struct.site_properties["movescale"][:2] == [0.0, 1.0]
    assert struct.site_properties["selective_dynamics"][:2] == [[False] * 3, [True] * 3]
    jif_from_jstruc = JDFTXInfile.from_jdftxstructure(jstruct)
    # Since velocities and moveScale are compatible to be stored as site properties, exporting to
    # `Structure` will save them in `Structure.site_properties`. Initializing a new JDFTXInfile
    # from this `Structure` should yield the same result as `jif_from_jstruc`.
    jif_from_struc = JDFTXInfile.from_structure(struct)
    # Without moveScale, selective dynamics give it (all 0 or 1 here), given per direction (as in
    # Structure.site_properties) or per site
    selective_dynamics = struct.site_properties["selective_dynamics"]
    struct.remove_site_property("movescale")
    jif_from_struc_2 = JDFTXInfile.from_structure(struct, selective_dynamics=selective_dynamics)
    per_site = [int(all(sd)) for sd in selective_dynamics]
    jif_from_struc_3 = JDFTXInfile.from_structure(struct, selective_dynamics=per_site)
    for jif2 in [jif_from_jstruc, jif_from_struc, jif_from_struc_2, jif_from_struc_3]:
        assert isinstance(jif2, JDFTXInfile)
        assert_identical_jif(jif["ion"], jif2["ion"])


def test_pmg_struc():
    jif = JDFTXInfile.from_file(ex_infile2_fname)
    struc1 = jif.to_pmg_structure(jif)
    struc2 = jif.structure
    for s in [struc1, struc2]:
        assert isinstance(s, Structure)
    assert_identical_jif(struc1.as_dict(), struc2.as_dict())


def test_jdftxtructure_naming():
    """Test the naming of the JDFTXStructure object.

    Test to make sure reading from a Structure with labels not exactly matching the element names
    (ie Si0, Si1, or Si+2) will still be read correctly.
    """
    struct = Structure.from_file(ex_in_files_dir / "Si.cif")
    jstruct = JDFTXStructure(structure=struct)
    JDFTXInfile.from_jdftxstructure(jstruct)
    JDFTXInfile.from_structure(struct)


@pytest.mark.parametrize(
    ("value_str", "expected_dict"),
    [
        (
            "H 1.0 1.0 1.0 0",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 0.0,
            },
        ),
        (
            "H 1.0 1.0 1.0 v 1.0 1.0 1.0 0",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "v": {
                    "vx0": 1.0,
                    "vx1": 1.0,
                    "vx2": 1.0,
                },
                "moveScale": 0.0,
            },
        ),
        (
            "H 1.0 1.0 1.0 0.5",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 0.5,
            },
        ),
        (
            "H 1.0 1.0 1.0 1 Linear 1.0 1.0 1.0",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 1.0,
                "constraint type": "Linear",
                "d0": 1.0,
                "d1": 1.0,
                "d2": 1.0,
            },
        ),
        (
            "H 1.0 1.0 1.0 1 HyperPlane 1.0 1.0 1.0 g1 HyperPlane 1.0 1.0 1.0 g1",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 1.0,
                "HyperPlane": [
                    {
                        "d0": 1.0,
                        "d1": 1.0,
                        "d2": 1.0,
                        "group": "g1",
                    },
                    {
                        "d0": 1.0,
                        "d1": 1.0,
                        "d2": 1.0,
                        "group": "g1",
                    },
                ],
            },
        ),
    ],
)
def test_ion_reading(value_str: str, expected_dict: dict):
    ion_tag: MultiformatTag = get_tag_object("ion")
    i = ion_tag.get_format_index_for_str_value("ion", value_str)
    tag_object: TagContainer = ion_tag.format_options[i]
    parsed_tag = tag_object.read("ion", value_str)
    assert_same_value(parsed_tag, expected_dict)


@pytest.mark.parametrize(
    ("expected_out", "stored_dict"),
    [
        (
            "ion H 1.000000000000000 1.000000000000000 1.000000000000000 0",
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 0,
            },
        ),
        (
            (
                "ion H 1.000000000000000 1.000000000000000 1.000000000000000 v "
                "1.000000000000000 1.000000000000000 1.000000000000000 0"
            ),
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "v": {
                    "vx0": 1.0,
                    "vx1": 1.0,
                    "vx2": 1.0,
                },
                "moveScale": 0,
            },
        ),
        (
            (
                "ion H 1.000000000000000 1.000000000000000 1.000000000000000 1 Linear "
                "1.000000000000000 1.000000000000000 1.000000000000000"
            ),
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 1,
                "constraint type": "Linear",
                "d0": 1.0,
                "d1": 1.0,
                "d2": 1.0,
            },
        ),
        (
            (
                "ion H 1.000000000000000 1.000000000000000 1.000000000000000 1 HyperPlane "
                "1.000000000000000 1.000000000000000 1.000000000000000 g1 "
                "HyperPlane 1.000000000000000 1.000000000000000 1.000000000000000 g1"
            ),
            {
                "species-id": "H",
                "x0": 1.0,
                "x1": 1.0,
                "x2": 1.0,
                "moveScale": 1,
                "HyperPlane": [
                    {
                        "d0": 1.0,
                        "d1": 1.0,
                        "d2": 1.0,
                        "group": "g1",
                    },
                    {
                        "d0": 1.0,
                        "d1": 1.0,
                        "d2": 1.0,
                        "group": "g1",
                    },
                ],
            },
        ),
    ],
)
def test_ion_writing(expected_out: str, stored_dict: dict):
    ion_tag: MultiformatTag = get_tag_object("ion")
    i, _ = ion_tag._determine_format_option("ion", stored_dict)
    tag_object: TagContainer = ion_tag.format_options[i]
    output = tag_object.write("ion", stored_dict)
    assert_same_value(output.strip().split(), expected_out.strip().split())


@pytest.mark.parametrize("chain_length", [1, 2, 5])
def test_stat_velocities_take_any_number_of_values(chain_length: int):
    """JDFTx reads as many values as are given: chainLengthT thermostat velocities, and 6 strain rates plus
    chainLengthP lattice thermostat velocities for the barostat.
    """
    thermostat = [0.1 * (i + 1) for i in range(chain_length)]
    barostat = [0.5] * (6 + chain_length)
    jif = JDFTXInfile.from_str(
        f"thermostat-velocity {' '.join(map(str, thermostat))}\nbarostat-velocity {' '.join(map(str, barostat))}\n",
        dont_require_structure=True,
    )
    assert jif["thermostat-velocity"] == {"v": thermostat}
    assert jif["barostat-velocity"] == {"v": barostat}
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True) == jif


def test_stat_velocities_older_dict_form():
    """Dicts from older pymatgen, with one key per value ({"v1": ..., "v2": ..., ...}), still load."""
    jif = JDFTXInfile.from_dict(
        {
            "thermostat-velocity": {"v1": 0.1, "v2": 0.2, "v3": 0.3},
            "barostat-velocity": {f"v{i + 1}": 0.5 for i in range(9)},
        }
    )
    assert jif["thermostat-velocity"] == {"v": [0.1, 0.2, 0.3]}
    assert jif["barostat-velocity"] == {"v": [0.5] * 9}


def test_fractional_move_scale():
    """moveScale is a double in JDFTx: any nonzero value lets the ion move (and scales its step)."""
    jif = JDFTXInfile.from_str(
        "lattice Cubic 10.0\nion H 0 0 0 0.5\nion H 0.5 0.5 0.5 1.5 Linear 1 0 0\nion O 0.2 0.2 0.2 0\n"
    )
    assert [ion["moveScale"] for ion in jif["ion"]] == [0.5, 1.5, 0.0]
    # Written values read back unchanged
    assert [ion["moveScale"] for ion in JDFTXInfile.from_str(str(jif))["ion"]] == [0.5, 1.5, 0.0]
    # Integer values are accepted and stored as floats
    jif["ion"] = ["H 0 0 0 1", {"species-id": "O", "x0": 0.1, "x1": 0.1, "x2": 0.1, "moveScale": 0}]
    assert [ion["moveScale"] for ion in jif["ion"]] == [1.0, 0.0]
    # The values survive a trip through a pymatgen Structure (as the movescale site property)
    struct = JDFTXInfile.from_str("lattice Cubic 10.0\nion H 0 0 0 0.5\nion O 0.2 0.2 0.2 0\n").structure
    assert struct.site_properties["movescale"] == [0.5, 0.0]
    assert [ion["moveScale"] for ion in JDFTXInfile.from_structure(struct)["ion"]] == [0.5, 0.0]


def test_jdftxstructure_movescale_and_selective_dynamics():
    """movescale and selective_dynamics fill each other and are synced with the site properties: a site free in every
    direction goes with any non-zero moveScale, and a site fixed in every direction with 0.
    """
    structure = Structure(Lattice.cubic(5.0), ["H", "H", "O"], [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]])
    free_fixed_free = [[True] * 3, [False] * 3, [True] * 3]
    jstruct = JDFTXStructure(structure, movescale=[0.5, 0, None])  # A site with neither can move
    assert jstruct.structure.site_properties == {"movescale": [0.5, 0.0, 1.0], "selective_dynamics": free_fixed_free}
    assert structure.site_properties == {}  # The given structure is left as it was
    assert JDFTXStructure(jstruct.structure).movescale == [0.5, 0.0, 1.0]
    assert JDFTXStructure(structure, selective_dynamics=free_fixed_free).movescale == [1.0, 0.0, 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        JDFTXStructure(structure, movescale=[0.5, 0, 2], selective_dynamics=free_fixed_free)
    with pytest.warns(UserWarning, match=re.escape("disagree for sites [0, 1]")):
        jstruct = JDFTXStructure(structure, movescale=[0, 1, 1], selective_dynamics=free_fixed_free)
    assert jstruct.movescale == [0.0, 0.0, 1.0]  # Fixed if either marks it fixed
    # A site fixed in only some directions is refused instead of being frozen, also as a site property
    partly_fixed = [[True] * 3, [True, False, True], [True] * 3]
    with pytest.raises(NotImplementedError, match=re.escape("not supported yet (sites [1])")):
        JDFTXStructure(structure, selective_dynamics=partly_fixed)
    structure.add_site_property("selective_dynamics", partly_fixed)
    with pytest.raises(NotImplementedError, match=re.escape("not supported yet (sites [1])")):
        JDFTXInfile.from_structure(structure)
    assert JDFTXInfile.from_structure(structure, selective_dynamics=free_fixed_free)["ion"][1]["moveScale"] == 0.0


@pytest.mark.parametrize("sd_as_site_property", [False, True], ids=["sd_arg", "sd_site_property"])
@pytest.mark.parametrize("movescale_as_site_property", [False, True], ids=["movescale_arg", "movescale_site_property"])
@pytest.mark.parametrize(
    ("selective_dynamics", "movescale", "expected", "disagree"),
    [
        ([True] * 3, 0.5, 0.5, False),
        ([True] * 3, 0.0, 0.0, True),
        ([False] * 3, 0.5, 0.0, True),
        ([True] * 3, None, 1.0, False),
        ([False] * 3, None, 0.0, False),
    ],
    ids=["free_nonzero", "free_zero", "fixed_nonzero", "free_unset", "fixed_unset"],
)
def test_from_structure_movescale_and_selective_dynamics(
    selective_dynamics: list[bool],
    movescale: float | None,
    expected: float,
    disagree: bool,
    movescale_as_site_property: bool,
    sd_as_site_property: bool,
):
    """An ion is fixed if either movescale or selective_dynamics marks it fixed (warning where the other marks it free),
    and else has the given movescale (1 if none), whether each is given as an argument or as a site property. The one
    exception is a movescale argument beside a selective_dynamics site property: the argument replaces it, as an
    intentional change, without a warning.
    """
    structure = Structure(Lattice.cubic(10.0), ["H"], [[0, 0, 0]])
    kwargs = {}
    for name, value, as_site_property in (
        ("selective_dynamics", selective_dynamics, sd_as_site_property),
        ("movescale", movescale, movescale_as_site_property),
    ):
        if value is not None and as_site_property:
            structure.add_site_property(name, [value])
        elif value is not None:
            kwargs[name] = [value]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        jif = JDFTXInfile.from_structure(structure, **kwargs)
    replaced = movescale is not None and not movescale_as_site_property and sd_as_site_property
    expected = movescale if replaced else expected
    assert any("disagree" in str(w.message) for w in record) is (disagree and not replaced)
    assert JDFTXInfile.from_str(str(jif))["ion"][0]["moveScale"] == expected


@pytest.mark.parametrize(("movescale", "selective_dynamics"), [(0.5, [True] * 3), (0.0, [False] * 3)])
def test_to_pmg_structure_selective_dynamics(movescale: float, selective_dynamics: list[bool]):
    """Structures from JDFTx inputs keep moveScale, with selective dynamics fixed only for a moveScale of 0."""
    structure = JDFTXInfile.to_pmg_structure(JDFTXInfile.from_str(f"lattice Cubic 10.0\nion H 0 0 0 {movescale}\n"))
    assert structure.site_properties["movescale"] == [movescale]
    assert structure.site_properties["selective_dynamics"] == [selective_dynamics]


@pytest.mark.parametrize(
    ("movescale_in_file", "edits", "expected", "disagree"),
    [
        (1, {"selective_dynamics": [False] * 3}, 0.0, True),
        (1, {"movescale": 0.0}, 0.0, True),
        (1, {"movescale": 0.5}, 0.5, False),
        (0, {"selective_dynamics": [True] * 3}, 0.0, True),
        (0, {"movescale": 0.5}, 0.0, True),
        (0, {"selective_dynamics": [True] * 3, "movescale": 0.5}, 0.5, False),
    ],
    ids=["freeze_by_sd", "freeze_by_movescale", "rescale", "unfreeze_by_sd", "unfreeze_by_movescale", "unfreeze_both"],
)
def test_editing_site_properties_of_structure_read_from_jdftx(
    movescale_in_file: float, edits: dict, expected: float, disagree: bool
):
    """A structure read from JDFTx has both movescale and selective_dynamics, so editing one of them can't be told apart
    from a stale other one: an ion is held fixed if either marks it fixed. Freezing works through either, and unfreezing
    needs both.
    """
    structure = JDFTXInfile.from_str(f"lattice Cubic 10.0\nion H 0 0 0 {movescale_in_file}\n").structure
    for name, value in edits.items():
        structure.add_site_property(name, [value])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        jif = JDFTXInfile.from_structure(structure)
    assert any("disagree" in str(w.message) for w in record) is disagree
    assert jif["ion"][0]["moveScale"] == expected


def _one_h_structure() -> Structure:
    return Structure(Lattice.cubic(10.0), ["H"], [[0, 0, 0]])


def test_movescale_argument_replaces_selective_dynamics_site_property():
    """A movescale argument wins over the selective_dynamics site property, which is set to match without a warning,
    as an intentional change (e.g. fixing an ion of a structure read from JDFTx).
    """
    structure = JDFTXInfile.from_str("lattice Cubic 10.0\nion H 0 0 0 1\n").structure
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jstruct = JDFTXStructure(structure, movescale=[0])
        jif = JDFTXInfile.from_structure(structure, movescale=[0])
    assert jstruct.structure.site_properties["selective_dynamics"] == [[False] * 3]
    assert jif["ion"][0]["moveScale"] == 0.0


def test_appended_site_movescale():
    """A site appended to a structure read from JDFTx (None for every site property) can move, unless its selective
    dynamics fix it.
    """
    structure = JDFTXInfile.from_str("lattice Cubic 10.0\nion H 0 0 0 0.5\n").structure
    structure.append("H", [0.5, 0.5, 0.5])
    assert structure.site_properties["movescale"] == [0.5, None]
    assert [ion["moveScale"] for ion in JDFTXInfile.from_structure(structure)["ion"]] == [0.5, 1.0]
    structure[1].properties["selective_dynamics"] = [False] * 3
    assert [ion["moveScale"] for ion in JDFTXInfile.from_structure(structure)["ion"]] == [0.5, 0.0]


def test_jdftxstructure_dict_round_trip_does_not_warn_again():
    """Sites that movescale and selective_dynamics disagree on are fixed with a warning and both are set to match, so
    a saved JDFTXStructure loads back the same without warning again.
    """
    structure = Structure(Lattice.cubic(10.0), ["H", "H"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    with pytest.warns(UserWarning, match="disagree"):
        jstruct = JDFTXStructure(structure, movescale=[0.5, None], selective_dynamics=[[False] * 3, [True] * 3])
    assert jstruct.movescale == [0.0, 1.0]
    assert jstruct.selective_dynamics == [[False] * 3, [True] * 3]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = JDFTXStructure.from_dict(jstruct.as_dict())
    assert loaded.movescale == jstruct.movescale
    assert loaded.selective_dynamics == jstruct.selective_dynamics


def test_from_structure_ignore_site_properties_movescale():
    """With ignore_site_properties, only a movescale argument counts."""
    structure = JDFTXInfile.from_str("lattice Cubic 10.0\nion H 0 0 0 0.5\n").structure
    assert JDFTXInfile.from_structure(structure, ignore_site_properties=True)["ion"][0]["moveScale"] == 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jif = JDFTXInfile.from_structure(structure, movescale=[0], ignore_site_properties=True)
    assert jif["ion"][0]["moveScale"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"selective_dynamics": np.array([[1, 1, 1]])}, 1.0),
        ({"selective_dynamics": np.array([[0, 0, 0]])}, 0.0),
        ({"selective_dynamics": [0]}, 0.0),  # One value per site
        ({"selective_dynamics": [True]}, 1.0),
        ({"selective_dynamics": np.array([1])}, 1.0),
        ({"movescale": np.array([0.5])}, 0.5),
        ({"movescale": [2]}, 2.0),
        ({"movescale": [True]}, 1.0),
        ({"movescale": [False]}, 0.0),
    ],
)
def test_from_structure_movescale_input_forms(kwargs: dict, expected: float):
    jif = JDFTXInfile.from_structure(_one_h_structure(), **kwargs)
    assert JDFTXInfile.from_str(str(jif))["ion"][0]["moveScale"] == expected


def test_constraint_on_site_fixed_by_selective_dynamics():
    """A constraint on a site that selective dynamics fix (moveScale 0) is rejected, as JDFTx rejects it."""
    structure = _one_h_structure()
    for name, value in (("constraint_types", "Linear"), ("constraint_vectors", [0, 0, 1])):
        structure.add_site_property(name, [value])
    with pytest.raises(ValueError, match="moveScale of 0"):
        JDFTXInfile.from_structure(structure, selective_dynamics=[[False] * 3])


@pytest.mark.parametrize("name", ["movescale", "selective_dynamics"])
def test_movescale_wrong_length(name: str):
    with pytest.raises(ValueError, match=f"{name} has 2 values, but the structure has 1 sites"):
        JDFTXInfile.from_structure(_one_h_structure(), **{name: [1, 1]})


def test_negative_movescale_is_rejected():
    with pytest.raises(ValueError, match=re.escape("moveScale cannot be negative (sites [0])")):
        JDFTXInfile.from_structure(_one_h_structure(), movescale=[-1])
    with pytest.raises(ValueError, match="moveScale cannot be negative"):
        _ = JDFTXInfile.from_str("lattice Cubic 10.0\nion H 0 0 0 -1\n").structure


FIXED_AND_FREE_TEXT = "lattice Cubic 10.0\nion H 0 0 0 0\nion H 0.5 0.5 0.5 0.5\n"


def test_fixed_ions_reach_poscar():
    """Structures read from JDFTx keep their fixed ions as selective dynamics, which other formats read."""
    poscar = Poscar.from_str(Poscar(JDFTXInfile.from_str(FIXED_AND_FREE_TEXT).structure).get_str())
    assert np.array(poscar.structure.site_properties["selective_dynamics"]).tolist() == [[False] * 3, [True] * 3]


def test_fixed_ions_reach_ase():
    pytest.importorskip("ase")
    atoms = AseAtomsAdaptor.get_atoms(JDFTXInfile.from_str(FIXED_AND_FREE_TEXT).structure)
    assert [constraint.get_indices().tolist() for constraint in atoms.constraints] == [[0]]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # Bare van-der-waals (DFT-D2 with the default scale; how JDFTx echoes a scale of 0) is a scale of 0
        ("van-der-waals", 0.0),
        ("van-der-waals ", 0.0),
        ("van-der-waals 0", 0.0),
        ("van-der-waals 0.75", 0.75),
        ("van-der-waals D3", "D3"),
        ("van-der-waals D3BJ", "D3BJ"),
    ],
)
def test_van_der_waals(line: str, expected: float | str):
    jif = JDFTXInfile.from_str(f"{line}\n", dont_require_structure=True)
    assert jif["van-der-waals"] == expected
    # Written values read back unchanged
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)["van-der-waals"] == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("rhoExternal rho.bin", {"filename": "rho.bin"}),
        ("rhoExternal rho.bin yes", {"filename": "rho.bin", "includeSelfEnergy": True}),
        ("rhoExternal rho.bin no", {"filename": "rho.bin", "includeSelfEnergy": False}),
    ],
)
def test_rho_external(line: str, expected: dict):
    """includeSelfEnergy is a yes/no flag in JDFTx."""
    jif = JDFTXInfile.from_str(f"{line}\n", dont_require_structure=True)
    assert jif["rhoExternal"] == expected
    assert str(jif).split() == line.split()
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)["rhoExternal"] == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Vexternal V.bin", {"filename": "V.bin"}),
        # Spin-polarized potential
        ("Vexternal Vup.bin Vdn.bin", {"filenameUp": "Vup.bin", "filenameDn": "Vdn.bin"}),
    ],
)
def test_vexternal(line: str, expected: dict):
    jif = JDFTXInfile.from_str(f"{line}\n", dont_require_structure=True)
    assert jif["Vexternal"] == expected
    assert str(jif).split() == line.split()
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)["Vexternal"] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("debug Ecomponents", ["Ecomponents"]),
        ("debug Ecomponents\ndebug Forces", ["Ecomponents", "Forces"]),
        # Several options per line, as JDFTx allows
        ("debug Ecomponents Forces", ["Ecomponents", "Forces"]),
        # JDFTx echoes all options merged into the first debug line and the others bare, and runs "Symmetries" into
        # the option before it
        ("debug  Ecomponents ForcesSymmetries\ndebug ", ["Ecomponents", "Forces", "Symmetries"]),
        ("debug Symmetries", ["Symmetries"]),
    ],
)
def test_debug(text: str, expected: list[str]):
    jif = JDFTXInfile.from_str(f"{text}\n", dont_require_structure=True)
    assert jif["debug"] == expected
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)["debug"] == expected


def test_debug_setting():
    # A bare debug line has no options
    assert "debug" not in JDFTXInfile.from_str("debug\n", dont_require_structure=True)
    with pytest.raises(ValueError, match="must be one of"):
        JDFTXInfile.from_str("debug Ecomponents Barbie\n", dont_require_structure=True)
    jif = JDFTXInfile.from_str("debug Ecomponents\n", dont_require_structure=True)
    jif["debug"] = "MuSearch Fluid"
    assert jif["debug"] == ["MuSearch", "Fluid"]
    jif["debug"] = ["Forces", "Ecomponents KpointsBasis"]
    assert jif["debug"] == ["Forces", "Ecomponents", "KpointsBasis"]
    jif.append_tag("debug", "EigsFillings Symmetries")
    assert jif["debug"] == ["Forces", "Ecomponents", "KpointsBasis", "EigsFillings", "Symmetries"]


@pytest.mark.filterwarnings("error:The following boundary errors")
@pytest.mark.parametrize("concentration", ["55.338", "bulk"], ids=["concentration", "bulk"])
def test_multiformat_tag_bounds(concentration: str):
    """Bounds on the subtags of a multiformat tag (here both fluid-solvent formats) are checked, as JDFTx does."""
    line = f"fluid-solvent H2O {concentration} ScalarEOS epsInf 0.5\n"  # JDFTx: "epsInf must be >= 1"
    with pytest.warns(UserWarning, match=re.escape("Value '0.5' for tag 'epsInf' is not within bounds")):
        JDFTXInfile.from_str(line, dont_require_structure=True)
    jif = JDFTXInfile.from_str(line, dont_require_structure=True, validate_value_boundaries=False)
    assert jif["fluid-solvent"][0]["epsInf"] == 0.5
    JDFTXInfile.from_str(line.replace("0.5", "1.77"), dont_require_structure=True)
    # A nan (unset value) is not a bounds error
    JDFTXInfile.from_str(line.replace("0.5", "nan"), dont_require_structure=True)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("perturb-incommensurate-wavefunctions wfns", {"filename": "wfns"}),
        # EcutOld is a double in JDFTx (echoed with %g)
        ("perturb-incommensurate-wavefunctions wfns 20.5", {"filename": "wfns", "EcutOld": 20.5}),
        ("perturb-incommensurate-wavefunctions wfns 20", {"filename": "wfns", "EcutOld": 20.0}),
    ],
)
def test_perturb_incommensurate_wavefunctions(line: str, expected: dict):
    tag = "perturb-incommensurate-wavefunctions"
    jif = JDFTXInfile.from_str(f"{line}\n", dont_require_structure=True)
    assert jif[tag] == expected
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)[tag] == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # Bulk defect: Eps is the bulk dielectric constant
        ("Dtot 2.5 1 2", {"DtotFile": "Dtot", "Eps": 2.5, "rMin": 1.0, "rSigma": 2.0}),
        # Surface defect: Eps is a slab-epsilon output file (with the Slab override for periodic Coulomb)
        ("Dtot slabEps 1 2", {"DtotFile": "Dtot", "Eps": "slabEps", "rMin": 1.0, "rSigma": 2.0}),
        (
            "Slab 001 Dtot slabEps 1 2",
            {"Slab": {"dir": "001"}, "DtotFile": "Dtot", "Eps": "slabEps", "rMin": 1.0, "rSigma": 2.0},
        ),
        # With the Slab override, JDFTx always reads Eps as a file name, even one that looks like a number
        (
            "Slab 010 Dtot 1e3 1 2",
            {"Slab": {"dir": "010"}, "DtotFile": "Dtot", "Eps": "1e3", "rMin": 1.0, "rSigma": 2.0},
        ),
    ],
)
def test_charged_defect_correction(line: str, expected: dict):
    """The forms JDFTx accepts (and echoes unchanged) are read, written and read back."""
    tag = "charged-defect-correction"
    jif = JDFTXInfile.from_str(f"{tag} {line}\n", dont_require_structure=True)
    assert jif[tag] == expected
    assert str(jif).split() == [tag, *line.replace(" 1 2", " 1.0 2.0").split()]
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)[tag] == expected
    # Set from a dict
    jif = JDFTXInfile.from_str("lattice Cubic 10\n", dont_require_structure=True)
    jif[tag] = expected
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)[tag] == expected


@pytest.mark.parametrize(
    "line",
    [
        "Slab Dtot slabEps 1 2",  # JDFTx requires <dir> after Slab
        "Slab 011 Dtot slabEps 1 2",
        "Dtot 2.5 1",
        "Dtot 2.5 1 2 3",
    ],
)
def test_charged_defect_correction_invalid(line: str):
    with pytest.raises(ValueError, match="No valid read format"):
        JDFTXInfile.from_str(f"charged-defect-correction {line}\n", dont_require_structure=True)


def test_charged_defect_correction_slab_needs_eps_file():
    # With the Slab override, Eps is a slab-epsilon file, not a dielectric constant
    jif = JDFTXInfile.from_str("lattice Cubic 10\n", dont_require_structure=True)
    value = {"Slab": {"dir": "001"}, "DtotFile": "Dtot", "Eps": 2.5, "rMin": 1.0, "rSigma": 2.0}
    with pytest.raises(ValueError, match="could not be determined"):
        jif["charged-defect-correction"] = value


# JDFTx's echo of "density-of-states Total Occupied Total Orbital H 1 s Complete Orbital H 2 s Etol 1e-5 Total"
dos_echo = """density-of-states Etol 1.000000e-05 Esigma 0.000000e+00 \\
		Complete \\
	Total \\
		Occupied \\
	Total \\
	Orbital H 1   s \\
		Complete \\
	Orbital H 2   s \\
	Total
"""
dos_echo_known = [
    {"Etol": 1e-5},
    {"Esigma": 0.0},
    {"Complete": True},
    {"Total": True},
    {"Occupied": True},
    {"Total": True},
    {"Orbital": {"species": "H", "atomIndex": 1, "orbDesc": "s"}},
    {"Complete": True},
    {"Orbital": {"species": "H", "atomIndex": 2, "orbDesc": "s"}},
    {"Total": True},
]


def test_density_of_states_keeps_order():
    """Occupied/Complete apply to the weights after them, and weights can repeat, so the order must survive."""
    tag = "density-of-states"
    jif = JDFTXInfile.from_str(dos_echo, dont_require_structure=True)
    assert_same_value(jif[tag], dos_echo_known)
    keywords = [next(iter(entry)) for entry in dos_echo_known]
    assert [line.split()[0] for line in str(jif).splitlines()[1:] if line] == keywords
    for jif2 in (JDFTXInfile.from_str(str(jif), dont_require_structure=True), jif.copy()):
        assert_same_value(jif2[tag], dos_echo_known)
        assert not jif.get_differing_tags(jif2)
    # The same subcommands in another order are a different input
    jif2 = jif.copy()
    jif2[tag] = [*dos_echo_known[:2], *dos_echo_known[4:6], *dos_echo_known[2:4], *dos_echo_known[6:]]
    assert jif.get_differing_tags(jif2) == [tag]
    # A bare density-of-states (total DOS only) is kept
    jif = JDFTXInfile.from_str(f"{tag}\n", dont_require_structure=True)
    assert jif[tag] == []
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)[tag] == []


def test_density_of_states_dict_form():
    """A dict of subcommands (the representation before order was kept) is read in the dict's order."""
    tag = "density-of-states"
    known = [
        {"Esigma": 0.001},
        {"Occupied": True},
        {"Total": True},
        {"OrthoOrbital": {"species": "Si", "atomIndex": 1, "orbDesc": "s"}},
        {"OrthoOrbital": {"species": "Si", "atomIndex": 2, "orbDesc": "p"}},
    ]
    value = {"Esigma": 0.001, "Occupied": True, "Total": True, "OrthoOrbital": [["Si", 1, "s"], ["Si", 2, "p"]]}
    jif = JDFTXInfile.from_str("lattice Cubic 10\n", dont_require_structure=True)
    jif[tag] = value
    assert_same_value(jif[tag], known)
    # Also when the value is never validated (the constructor skips __setitem__)
    jif = JDFTXInfile({tag: {**value, "OrthoOrbital": [{"species": "Si", "atomIndex": 1, "orbDesc": "s"}]}})
    assert_same_value(JDFTXInfile.from_str(str(jif), dont_require_structure=True)[tag], known[:4])


@pytest.mark.parametrize(
    ("line", "error_match"),
    [
        ("Barbie", "'Barbie' is not a valid subcommand"),
        ("Total 5", "'5' is not a valid subcommand"),
        ("Total Orbital H 1", "Subcommand 'Orbital' of 'density-of-states' is missing values"),
    ],
)
def test_density_of_states_read_errors(line: str, error_match: str):
    with pytest.raises(ValueError, match=error_match):
        JDFTXInfile.from_str(f"density-of-states {line}\n", dont_require_structure=True)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("elec-ex-corr hyb-HSE06", "hyb-HSE06"),
        ("elec-ex-corr hyb-hse06", "hyb-HSE06"),  # JDFTx's names are case-insensitive: stored as its spelling
    ],
)
def test_hse06_is_accepted_with_a_warning(line: str, expected: str | list[str]):
    """JDFTx's built-in hyb-HSE06 is read and written back, warning once that it differs from LibXC's HSE06."""
    tag = line.split(maxsplit=1)[0]
    with pytest.warns(JDFTXOptionWarning, match="hyb-gga-hse06") as record:
        jif = JDFTXInfile.from_str(f"{line}\n", dont_require_structure=True)
    assert len([w for w in record if issubclass(w.category, JDFTXOptionWarning)]) == 1
    assert jif[tag] == expected
    assert str(jif).split() == [tag, "hyb-HSE06"]
    jif = JDFTXInfile.from_str("lattice Cubic 10\n", dont_require_structure=True)
    with pytest.warns(JDFTXOptionWarning, match="hyb-gga-hse06") as record:
        jif[tag] = expected
    assert len([w for w in record if issubclass(w.category, JDFTXOptionWarning)]) == 1


@pytest.mark.parametrize("functional", ["gga-PBE", "hyb-PBE0", "hyb-gga-hse06"])
def test_other_functionals_do_not_warn(functional: str):
    with warnings.catch_warnings():
        warnings.simplefilter("error", JDFTXOptionWarning)
        jif = JDFTXInfile.from_str(f"elec-ex-corr {functional}\n", dont_require_structure=True)
        jif["elec-ex-corr"] = functional


# JDFTx's echo of "pcm-params lMax 3 sigma 0.7 cavityTension 5e-06" (fluid LinearPCM, fluid-solvent H2O)
PCM_PARAMS_ECHO = (
    "pcm-params  \\\n\tlMax 3 \\\n\tnc 0.00037 \\\n\tsigma 0.7 \\\n\tcavityTension 5e-06 \\\n\tcavityPressure 0 \\\n"
    "\tcavityScale 1 \\\n\tionSpacing 0 \\\n\tcavityFile  \\\n\tzMask0 0 \\\n\tzMaskH 0 \\\n\tzMaskIonH 0 \\\n"
    "\tzMaskSigma 0.5 \\\n\trhoMin 0 \\\n\trhoMax 0 \\\n\trhoDelta 0 \\\n\teta_wDiel 0 \\\n\tsqrtC6eff 0 \\\n"
    "\tpCavity 0 \\\n\tZtot 0 \\\n\tphiCavity 0 \\\n\tscreenOverride 0\n"
)


def test_pcm_params_echo_is_read_and_written_back():
    """JDFTx's echo prints every key: an empty cavityFile, and zeros for unset keys that JDFTx refuses as input."""
    unset = ["rhoMin", "rhoMax", "rhoDelta", "Ztot", "screenOverride"]
    # Reading only warns about the refused zeros, also from a stored dict (where no flag can be passed)
    with pytest.warns(UserWarning, match="boundary errors"):
        jif = JDFTXInfile.from_str(PCM_PARAMS_ECHO, dont_require_structure=True)
    with pytest.warns(UserWarning, match="boundary errors"):
        assert JDFTXInfile.from_dict(jif.as_dict()) == jif
    with pytest.raises(ValueError, match="boundary errors"):
        jif.validate_boundaries()
    params = jif["pcm-params"]
    assert "cavityFile" not in params
    assert params["lMax"] == 3
    assert isinstance(params["lMax"], int)
    assert params["sigma"] == pytest.approx(0.7)
    assert params["phiCavity"] == 0.0
    assert all(params[key] == 0.0 for key in unset)
    # Written back, the line is valid JDFTx input: no empty cavityFile, no refused zeros, and lMax written as "3"
    written = str(jif)
    tokens = written.split()
    assert "cavityFile" not in tokens
    assert not any(key in tokens for key in unset)
    assert tokens[tokens.index("lMax") + 1] == "3"
    assert JDFTXInfile.from_str(written, dont_require_structure=True)["pcm-params"] == {
        key: value for key, value in params.items() if key not in unset
    }


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("pcm-params cavityFile cavity.bin nc 0.001", {"cavityFile": "cavity.bin", "nc": 0.001}),
        ("pcm-params cavityFile", {}),
        # JDFTx master builds from 2024-02 to 2026-08 have Zcenter, which phiCavity replaced
        ("pcm-params Zcenter 0.5 nc 0.001", {"nc": 0.001, "Zcenter": 0.5}),
    ],
)
def test_pcm_params_read(line: str, expected: dict[str, Any]):
    assert JDFTXInfile.from_str(f"{line}\n", dont_require_structure=True)["pcm-params"] == expected


# JDFTx's echo of "fluid-site-params H2O O sigmaNuc 0.5" and "fluid-site-params H2O H aPol 0.4" (fluid ClassicalDFT)
FLUID_SITE_PARAMS_ECHO = (
    "fluid-site-params H2O O \\\n\tZnuc 6 \\\n\tsigmaNuc 0.5 \\\n\tZelec 6.826 \\\n\taElec 0.36935 \\\n"
    "\tsigmaElec 0.36823 \\\n\trcElec 0.51523 \\\n\talpha 3.73 \\\n\taPol 0.32 \\\n\tRhs 2.57003 \\\n"
    "\telecFilename  \\\n\telecFilenameG \n"
    "fluid-site-params H2O H \\\n\tZnuc 1 \\\n\tsigmaNuc 0.377945 \\\n\tZelec 0.587 \\\n\taElec 0.34641 \\\n"
    "\tsigmaElec 0.390882 \\\n\trcElec 0 \\\n\talpha 3.3 \\\n\taPol 0.4 \\\n\tRhs 0 \\\n"
    "\telecFilename  \\\n\telecFilenameG \n"
    "fluid-site-params \n"
)


def test_fluid_site_params_echo_is_read_and_written_back():
    """JDFTx echoes every site, with empty file names and zeros it refuses as input, then a bare line per extra line."""
    with pytest.warns(UserWarning, match="'Rhs' is not within bounds x > 0"):
        JDFTXInfile.from_str(FLUID_SITE_PARAMS_ECHO, dont_require_structure=True)
    jif = JDFTXInfile.from_str(FLUID_SITE_PARAMS_ECHO, dont_require_structure=True, validate_value_boundaries=False)
    entries = jif["fluid-site-params"]
    # The bare line isn't stored, so a restart doesn't gain one more each time
    assert [(entry["component"], entry["siteName"]) for entry in entries] == [("H2O", "O"), ("H2O", "H")]
    assert entries[0]["sigmaNuc"] == 0.5
    assert entries[1]["aPol"] == 0.4
    assert entries[1]["Rhs"] == 0.0
    assert not any("elecFilename" in entry or "elecFilenameG" in entry for entry in entries)
    # Written back as JDFTx reads it (checked with JDFTx: it echoes the same site parameters), without "Rhs 0"
    lines = str(jif).splitlines()
    assert [line.split()[:3] for line in lines if line] == [
        ["fluid-site-params", "H2O", "O"],
        ["fluid-site-params", "H2O", "H"],
    ]
    assert "Rhs" not in lines[1].split()
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)["fluid-site-params"] == [
        entries[0],
        {key: value for key, value in entries[1].items() if key != "Rhs"},
    ]


@pytest.mark.parametrize(
    ("line", "expected", "written"),
    [
        (
            "fluid-site-params H2O O sigmaNuc 0.5",
            {"component": "H2O", "siteName": "O", "sigmaNuc": 0.5},
            "fluid-site-params H2O O sigmaNuc 0.5",
        ),
        (
            "fluid-site-params CustomSolvent C1 Znuc 4 elecFilename n.bin",
            {"component": "CustomSolvent", "siteName": "C1", "elecFilename": "n.bin", "Znuc": 4.0},
            "fluid-site-params CustomSolvent C1 elecFilename n.bin Znuc 4.0",
        ),
        # JDFTx defaults the site (to the first one) when nothing follows the component
        ("fluid-site-params H2O", {"component": "H2O"}, "fluid-site-params H2O"),
    ],
)
def test_fluid_site_params_read(line: str, expected: dict[str, Any], written: str):
    jif = JDFTXInfile.from_str(f"{line}\n", dont_require_structure=True)
    assert jif["fluid-site-params"] == [expected]
    assert " ".join(str(jif).split()) == written


@pytest.mark.parametrize(
    "line",
    [
        # pymatgen used to read and write this form, which JDFTx rejects
        "fluid-site-params component H2O siteName O sigmaNuc 0.5",
        "fluid-site-params O sigmaNuc 0.5",  # JDFTx reads the first word as the component
    ],
)
def test_fluid_site_params_needs_the_component_first(line: str):
    with pytest.raises(ValueError, match="must be one of"):
        JDFTXInfile.from_str(f"{line}\n", dont_require_structure=True)


def test_fluid_site_params_bare_line_is_not_stored():
    # JDFTx reads a bare line as changing nothing
    jif = JDFTXInfile.from_str("fluid-site-params H2O O sigmaNuc 0.5\nfluid-site-params\n", dont_require_structure=True)
    assert jif["fluid-site-params"] == [{"component": "H2O", "siteName": "O", "sigmaNuc": 0.5}]
    assert "fluid-site-params" not in JDFTXInfile.from_str("fluid-site-params\n", dont_require_structure=True)


@pytest.mark.parametrize("algorithm", ["CG", "MINRES"])
def test_perturb_minimize_algorithm(algorithm: str):
    # JDFTx's options are CG|MINRES (checked with JDFTx)
    jif = JDFTXInfile.from_str(f"perturb-minimize algorithm {algorithm} nIterations 10\n", dont_require_structure=True)
    assert jif["perturb-minimize"] == {"algorithm": algorithm, "nIterations": 10}
    assert " ".join(str(jif).split()) == f"perturb-minimize \\ algorithm {algorithm} \\ nIterations 10"


def test_perturb_minimize_algorithm_rejects_cgiminres():
    # pymatgen used to accept CGIMINRES, which JDFTx never had
    with pytest.raises(ValueError, match="must be one of"):
        JDFTXInfile.from_str("perturb-minimize algorithm CGIMINRES\n", dont_require_structure=True)


# The commands JDFTx reads more than once (allowMultiple in jdftx/commands, main-fork master b3c54e1d) that pymatgen
# has, plus include, which JDFTx's parser handles itself
JDFTX_REPEATABLE_TAGS = {
    "box-potential",
    "charged-defect",
    "chargeball",
    "debug",
    "dump",
    "dump-Eresolved-density",
    "dump-fermi-density",
    "dump-interval",
    "elec-ex-corr-compare",
    "fluid-anion",
    "fluid-cation",
    "fluid-mixing-functional",
    "fluid-site-params",
    "fluid-solvent",
    "include",
    "ion",
    "ion-species",
    "ionic-gaussian-potential",
    "kpoint",
    "symmetry-matrix",
    "tau-core",
}


def test_dict_values_are_written_in_jdftx_order():
    # Positional values must be written in the order JDFTx reads them, whatever the order of the dict's keys
    jif = JDFTXInfile.from_str("lattice Cubic 10\n", dont_require_structure=True)
    jif["elec-cutoff"] = {"EcutRho": 100.0, "Ecut": 20.0}
    jif["elec-smearing"] = {"smearingWidth": 0.01, "smearingType": "Fermi"}
    jif["fluid-site-params"] = {"sigmaNuc": 0.5, "component": "H2O", "siteName": "O"}
    lines = [" ".join(line.split()) for line in str(jif).splitlines()]
    assert "elec-cutoff 20.0 100.0" in lines
    assert "elec-smearing Fermi 0.01" in lines
    assert "fluid-site-params H2O O sigmaNuc 0.5" in lines
    back = JDFTXInfile.from_str(str(jif))
    assert back["elec-cutoff"] == {"Ecut": 20.0, "EcutRho": 100.0}
    assert back["elec-smearing"] == {"smearingType": "Fermi", "smearingWidth": 0.01}


@pytest.mark.parametrize(
    ("line", "old_written", "written"),
    [
        (
            "fluid-solvent H2O 55.338 ScalarEOS quad_nBeta 4 quad_nAlpha 2 quad_nGamma 2",
            "fluid-solvent H2O 55.338 ScalarEOS quad_nAlpha 2.0 quad_nBeta 4.0 quad_nGamma 2.0",
            "fluid-solvent H2O 55.338 ScalarEOS quad_nAlpha 2 quad_nBeta 4 quad_nGamma 2",
        ),
        (
            "electron-scattering eta 0.01 computeRange 1 10",
            "electron-scattering eta 0.01 computeRange 1.0 10.0",
            "electron-scattering \\ eta 0.01 \\ computeRange 1 10",
        ),
        (
            "density-of-states Slice 0 0 0 2 0 0 1",
            "density-of-states Slice 0.0 0.0 0.0 2.0 0.0 0.0 1.0",
            "density-of-states \\ Slice 0.0 0.0 0.0 2.0 0 0 1",
        ),
        (
            "density-of-states AtomSlice H 1 2 1 -1 0",
            "density-of-states AtomSlice H 1 2.0 1.0 -1.0 0.0",
            "density-of-states \\ AtomSlice H 1 2.0 1 -1 0",
        ),
    ],
)
def test_integer_parameters_are_written_as_integers(line: str, old_written: str, written: str):
    """JDFTx reads these as integers and rejects the old "4.0" form (checked with JDFTx), which still reads the same."""
    for text in (line, old_written):
        jif = JDFTXInfile.from_str(f"{text}\n", dont_require_structure=True)
        assert " ".join(str(jif).split()) == written


def test_quad_nbeta_zero_is_left_out():
    # JDFTx echoes an unset quad_nBeta as 0 (for ClassicalDFT), but refuses 0 as input
    line = "fluid-solvent H2O 55.338 ScalarEOS quad_nBeta 0 quad_nAlpha 0\n"
    with pytest.warns(UserWarning, match="'quad_nBeta' is not within bounds x > 0"):
        JDFTXInfile.from_str(line, dont_require_structure=True)
    jif = JDFTXInfile.from_str(line, dont_require_structure=True, validate_value_boundaries=False)
    assert " ".join(str(jif).split()) == "fluid-solvent H2O 55.338 ScalarEOS quad_nAlpha 0"


@pytest.mark.filterwarnings("error:The following boundary errors")
def test_vibrations_configuration_range():
    """iConfiguration/nConfigurations (JDFTx since 2026-08) are read and written; JDFTx needs iConfiguration >= -1."""
    line = "vibrations centralDiff yes iConfiguration 2 nConfigurations 3\n"
    expected = {"centralDiff": True, "iConfiguration": 2, "nConfigurations": 3}
    jif = JDFTXInfile.from_str(line, dont_require_structure=True)
    assert jif["vibrations"] == expected
    assert " ".join(str(jif).split()) == "vibrations centralDiff yes iConfiguration 2 nConfigurations 3"
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)["vibrations"] == expected
    # -1 is the default; JDFTx refuses anything lower, but nConfigurations has no bounds
    unset = line.replace("iConfiguration 2", "iConfiguration -1").replace("nConfigurations 3", "nConfigurations -5")
    JDFTXInfile.from_str(unset, dont_require_structure=True)
    with pytest.warns(UserWarning, match=re.escape("Value '-2' for tag 'iConfiguration' is not within bounds x >= -1")):
        JDFTXInfile.from_str(line.replace("iConfiguration 2", "iConfiguration -2"), dont_require_structure=True)


def test_dump_hc():
    """HC (JDFTx since 2025-09) is a dump variable."""
    jif = JDFTXInfile.from_str("dump End HC\n", dont_require_structure=True)
    assert jif["dump"] == [{"End": {"HC": True}}]
    assert " ".join(str(jif).split()) == "dump End HC"


@pytest.mark.parametrize(
    ("echo", "expected"),
    [
        ("chargeball H 1 0.5", [{"species-id": "H", "norm": 1.0, "width": 0.5}]),
        (
            "setVDW \tO 0.7 1.3 \\\n\tH 1.5 2",
            {"species-params": [{"species": "O", "C6": 0.7, "R0": 1.3}, {"species": "H", "C6": 1.5, "R0": 2.0}]},
        ),
        (
            "set-atomic-radius \tO 1.5 \\\n\tH 1.2",
            {"species-radii": [{"species": "O", "Ri": 1.5}, {"species": "H", "Ri": 1.2}]},
        ),
        (
            "metadynamics-bond O 1 H 2 0.1 0.001 bias.dat",
            {
                "species1": "O",
                "atom1": 1,
                "species2": "H",
                "atom2": 2,
                "resolution": 0.1,
                "energy-per-step": 0.001,
                "state-file": "bias.dat",
            },
        ),
    ],
    ids=["chargeball", "setVDW", "set-atomic-radius", "metadynamics-bond"],
)
def test_species_override_and_metadynamics_commands(echo: str, expected: Any):
    """These commands were missing, so inputs with them raised and restarts lost them. The lines are JDFTx's echoes."""
    tag = echo.split(maxsplit=1)[0]
    jif = JDFTXInfile.from_str(echo + "\n", dont_require_structure=True)
    assert jif[tag] == expected
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)[tag] == expected


@pytest.mark.parametrize("tag", ["electric-field", "perturb-electric-field"])
def test_electric_field_components_are_floats(tag: str):
    """The field components are doubles in JDFTx (Eh/e/a0), so fields weaker than 1 must not be truncated to 0."""
    expected = {"Ex": 0.01, "Ey": 0.0, "Ez": -2.5e-3}
    jif = JDFTXInfile.from_str(f"{tag} 0.01 0 -2.5e-3\n", dont_require_structure=True)
    assert_same_value(jif[tag], expected)
    # Written values read back unchanged
    assert_same_value(JDFTXInfile.from_str(str(jif), dont_require_structure=True)[tag], expected)
    # Setting the tag from a dict or a list
    jif[tag] = {"Ex": 0, "Ey": 0.5, "Ez": 0}
    assert_same_value(jif[tag], {"Ex": 0.0, "Ey": 0.5, "Ez": 0.0})
    jif[tag] = [0.0, 0.0, 0.002]
    assert_same_value(JDFTXInfile.get_dict_representation(jif)[tag], {"Ex": 0.0, "Ey": 0.0, "Ez": 0.002})


@pytest.mark.parametrize(
    ("lattice_type", "value_str"),
    [
        ("Rhombohedral", "Rhombohedral 1.0 1.0"),
        ("Triclinic", "Triclinic 1.0 1.0 1.0 1.0 1.0 1.0"),
        ("Hexagonal", "Hexagonal 1.0 1.0"),
        ("Cubic", "Body-Centered Cubic 1.0"),
        ("Cubic", "Cubic 1.0"),
    ],
)
def test_lattice_reading(lattice_type: str, value_str: str):
    mft_lattice_tag = get_tag_object("lattice")
    assert mft_lattice_tag is not None
    i = mft_lattice_tag.get_format_index_for_str_value("lattice", value_str)
    tag_object = mft_lattice_tag.format_options[i]
    parsed_tag = tag_object.read("lattice", value_str)
    assert lattice_type in parsed_tag


@pytest.mark.parametrize(
    ("value_str"),
    [
        ("Rhombohedral 1.0 1.0"),
        ("Triclinic 1.0 1.0 1.0 1.0 1.0 1.0"),
        ("Hexagonal 1.0 1.0"),
        ("Body-Centered Cubic 1.0"),
        ("Cubic 1.0"),
        ("Orthorhombic 1.0 1.0 1.0"),
        ("Base-Centered Orthorhombic 1.0 1.0 1.0"),
        ("Monoclinic 1.0 1.0 1.0 1.0"),
        ("Base-Centered Monoclinic 1.0 1.0 1.0 1.0"),
        ("Tetragonal 1.0 1.0"),
        ("Body-Centered Tetragonal 1.0 1.0"),
    ],
)
def test_lattice_writing(value_str: str):
    mft_lattice_tag = get_tag_object("lattice")
    assert mft_lattice_tag is not None
    i = mft_lattice_tag.get_format_index_for_str_value("lattice", value_str)
    tag_object = mft_lattice_tag.format_options[i]
    parsed_tag = tag_object.read("lattice", value_str)
    output = tag_object.write("lattice", parsed_tag)
    assert_same_value(
        ("lattice " + value_str).strip().split(),
        output.strip().split(),
    )


@pytest.mark.parametrize(
    ("value_str"),
    [
        ("Rhombohedral 1.0 1.0"),
        ("Triclinic 1.0 1.0 1.0 1.0 1.0 1.0"),
        ("Hexagonal 1.0 1.0"),
        ("Body-Centered Cubic 1.0"),
        ("Cubic 1.0"),
        ("Orthorhombic 1.0 1.0 1.0"),
        ("Base-Centered Orthorhombic 1.0 1.0 1.0"),
        ("Monoclinic 1.0 1.0 1.0 1.0"),
        ("Base-Centered Monoclinic 1.0 1.0 1.0 1.0"),
        ("Tetragonal 1.0 1.0"),
        ("Body-Centered Tetragonal 1.0 1.0"),
    ],
)
def test_jdftxstructure_lattice_conversion(value_str: str):
    test_vars = ["a", "b", "c", "alpha", "beta", "gamma"]
    conv_vars = ["a", "b", "c"]
    mft_lattice_tag = get_tag_object("lattice")
    assert mft_lattice_tag is not None
    i = mft_lattice_tag.get_format_index_for_str_value("lattice", value_str)
    tag_object = mft_lattice_tag.format_options[i]
    parsed_tag = tag_object.read("lattice", value_str)
    infile = JDFTXInfile.from_str("lattice " + value_str + "\n ion H 0.0 0.0 0.0 0", dont_require_structure=True)
    if "modification" in parsed_tag:
        with pytest.raises(NotImplementedError):
            _ = infile.to_pmg_structure(infile)
    else:
        structure = infile.to_pmg_structure(infile)
        for var in test_vars:
            if var in parsed_tag:
                should_be = parsed_tag[var]
                if var in conv_vars:
                    should_be *= bohr_to_ang
                assert_same_value(float(getattr(structure.lattice, var)), float(should_be))


def test_jdftxinfile_comparison():
    jif1 = JDFTXInfile.from_file(ex_infile1_fname)
    jif2 = JDFTXInfile.from_file(ex_infile2_fname)
    assert len(jif1.get_differing_tags(jif2))  # At least one tag should be different
    jif1copy = jif1.copy()
    assert not len(jif1.get_differing_tags(jif1copy))  # No tags should be different
    default_test_tag = "davidson-band-ratio"
    default_test_val = default_inputs[default_test_tag]
    jif1[default_test_tag] = default_test_val
    assert not len(jif1.get_differing_tags(jif1copy))  # Even though jif1copy doesn't have the tag,
    # it won't be recognized as a difference since it is in the default_inputs and matches the default value
    jif1copy["elec-n-bands"] = 20001
    assert len(jif1.get_filtered_differing_tags(jif1copy))  # Change in elec-n-bands should be recognized
    assert not len(
        jif1.get_filtered_differing_tags(jif1copy, exclude_tags=["elec-n-bands"])
    )  # Specific tags can be filtered out
    assert not len(
        jif1.get_filtered_differing_tags(jif1copy, exclude_tag_categories=["electronic"])
    )  # Tag categories can be filtered out


def test_get_differing_tags_fills_defaults():
    """An input matches JDFTx's echo of it, which adds every default, including those that depend on the fluid."""
    echo = JDFTXOutfile.from_file(pcm_params_outfile_path).infile
    given = echo.copy()
    for tag in ("ion-width", "fluid-minimize", "lcao-params"):
        del given[tag]
    given["fluid"] = {"type": "LinearPCM"}
    given["fluid-solvent"] = [{"name": "H2O"}]
    assert given.get_differing_tags(echo) == []
    given["fluid"] = {"type": "NonlinearPCM"}  # Changes the fluid-minimize defaults
    given["fluid-solvent"] = [{"name": "H2O", "epsBulk": 70.0}]
    assert sorted(given.get_differing_tags(echo)) == ["fluid", "fluid-minimize", "fluid-solvent"]


def test_antoine_pvap():
    assert_same_value(antoinePvap(298, 7.31549, 1794.88, -34.764), 1.06736e-10)


def test_nan_stripping():
    jif = JDFTXInfile.from_file(ex_infile1_fname)
    # Single optional subtag being nan should only remove that subtag
    jif["elec-cutoff"]["EcutRho"] = np.nan
    jif = clean_infile_of_nans(jif)
    assert "EcutRho" not in jif["elec-cutoff"]
    # Single required subtag being nan should remove the whole tag
    jif["latt-move-scale"]["s0"] = np.nan
    jif = clean_infile_of_nans(jif)
    assert "latt-move-scale" not in jif
    jif.pop("fluid-solvent")
    jif.read_line("fluid-solvent H2O 55.338 ScalarEOS epsBulk nan")
    assert "epsBulk" in jif["fluid-solvent"][0]
    jif = clean_infile_of_nans(jif)
    assert "epsBulk" not in jif["fluid-solvent"][0]
    print(jif)


def _species_values(*groups: tuple[str, list[float]]) -> dict:
    return {"species-values": [{"species-id": sp, "values": vals} for sp, vals in groups]}


@pytest.mark.parametrize(
    ("magmom_value", "oxstate_value", "magmom_expected", "oxstate_expected"),
    [
        (
            "Mn 3 -3 3",
            "Mn +4 O -2",
            _species_values(("Mn", [3.0, -3.0, 3.0])),
            _species_values(("Mn", [4.0]), ("O", [-2.0])),
        ),
        (
            "Mn 3.0 0 0 3 180 90 O 0 0 0",
            "Mn 4",
            _species_values(("Mn", [3.0, 0.0, 0.0, 3.0, 180.0, 90.0]), ("O", [0.0, 0.0, 0.0])),
            _species_values(("Mn", [4.0])),
        ),
        (
            "Mn 1e-1 -2.5E+0 O 0",
            "Mn +4.0 O -2.0",
            _species_values(("Mn", [0.1, -2.5]), ("O", [0.0])),
            _species_values(("Mn", [4.0]), ("O", [-2.0])),
        ),
    ],
)
def test_species_values_tags_infile_roundtrip(
    magmom_value: str, oxstate_value: str, magmom_expected: dict, oxstate_expected: dict
):
    text = f"spintype z-spin\ninitial-magnetic-moments {magmom_value}\ninitial-oxidation-state {oxstate_value}\n"
    first = JDFTXInfile.from_str(text, dont_require_structure=True)
    assert first["initial-magnetic-moments"] == magmom_expected
    assert first["initial-oxidation-state"] == oxstate_expected
    # Writing and re-reading gives back the same data (the text itself may differ, e.g. "3" is written as "3.0")
    second = JDFTXInfile.from_str(str(first), dont_require_structure=True)
    assert second["initial-magnetic-moments"] == magmom_expected
    assert second["initial-oxidation-state"] == oxstate_expected
    assert first.is_comparable_to(second)


def test_initial_magnetic_moments_comparison():
    base = "spintype z-spin\ninitial-magnetic-moments "
    jif = JDFTXInfile.from_str(base + "Mn 3 -3 O 0", dont_require_structure=True)
    # Species groups can come in any order
    assert jif.is_comparable_to(JDFTXInfile.from_str(base + "O 0 Mn 3 -3", dont_require_structure=True))
    # Values within a group are per-ion, so their order matters
    assert not jif.is_comparable_to(JDFTXInfile.from_str(base + "Mn -3 3 O 0", dont_require_structure=True))
    assert not jif.is_comparable_to(JDFTXInfile.from_str(base + "Mn 3 -3 O 1", dont_require_structure=True))


def test_initial_magnetic_moments_setitem():
    jif = JDFTXInfile.from_str("spintype z-spin\n", dont_require_structure=True)
    jif["initial-magnetic-moments"] = "Fe 2 -2 O 0"
    assert jif["initial-magnetic-moments"] == _species_values(("Fe", [2.0, -2.0]), ("O", [0.0]))
    # Integer values given in dict form are converted to floats
    jif["initial-magnetic-moments"] = _species_values(("Fe", [2, -2]))
    assert jif["initial-magnetic-moments"] == _species_values(("Fe", [2.0, -2.0]))
    assert all(isinstance(v, float) for v in jif["initial-magnetic-moments"]["species-values"][0]["values"])
    assert "initial-magnetic-moments Fe 2.0 -2.0 " in str(jif)


ION_LINES = "ion Mn 0.0 0.0 0.0 1\nion Mn 0.5 0.5 0.5 1\nion O 0.25 0.25 0.25 1\n"


@pytest.mark.parametrize(
    "text",
    [
        "spintype z-spin\n",
        "spintype z-spin\ninitial-magnetic-moments Mn 3 -3 O 0\n" + ION_LINES,
        "spintype z-spin\ninitial-magnetic-moments Mn 3 -3\n",
        "spintype vector-spin\ninitial-magnetic-moments Mn 3 0 0 3 180 0\n",
        "spintype vector-spin\ninitial-magnetic-moments Mn 3 0 0 3 180 0 O 0 0 0\n" + ION_LINES,
        # JDFTx's species names are case-insensitive
        "spintype z-spin\ninitial-magnetic-moments mn 3 -3 o 0\n" + ION_LINES,
        # Two Mn species (from pseudopotential files Mn1.upf and Mn2.upf), as for an antiferromagnet
        "spintype z-spin\ninitial-magnetic-moments Mn1 3 Mn2 -3\nion Mn1 0 0 0 1\nion Mn2 0.5 0.5 0.5 1\n",
    ],
)
def test_validate_initial_magnetic_moments_valid(text: str):
    JDFTXInfile.from_str(text, dont_require_structure=True).validate_initial_magnetic_moments()


@pytest.mark.parametrize(
    ("text", "error_match"),
    [
        ("initial-magnetic-moments Mn 3\n", "spintype tag must be specified"),
        ("spintype vector-spin\ninitial-magnetic-moments Mn 3 -3\n", "must have a multiple of 3 values, got 2"),
        # JDFTx: "Cannot specify magnetic moments in an unpolarized calculation"
        ("spintype spin-orbit\ninitial-magnetic-moments Mn 3 0 0\n", "unpolarized calculation"),
        ("spintype no-spin\ninitial-magnetic-moments Mn 3\n", "unpolarized calculation"),
        ("spintype z-spin\ninitial-magnetic-moments Fe 3\n" + ION_LINES, "Fe in initial-magnetic-moments tag is not"),
        ("spintype z-spin\ninitial-magnetic-moments Mn 3 O 0\n" + ION_LINES, "but there are 2 ions"),
        ("spintype vector-spin\ninitial-magnetic-moments Mn 3 0 0\n" + ION_LINES, "values for 1 ions, but there are 2"),
    ],
)
def test_validate_initial_magnetic_moments_invalid(text: str, error_match: str):
    jif = JDFTXInfile.from_str(text, dont_require_structure=True)
    with pytest.raises(ValueError, match=error_match):
        jif.validate_initial_magnetic_moments()


def _u_channel(species: str, orb_desc: str, u_minus_j: float, vexts: list[tuple[int, float]] | None = None) -> dict:
    channel: dict[str, Any] = {"species": species, "orbDesc": orb_desc, "UminusJ": u_minus_j}
    if vexts is not None:
        channel["Vext"] = [{"atom": atom, "V": v} for atom, v in vexts]
    return channel


@pytest.mark.parametrize(
    ("text", "expected_channels"),
    [
        ("add-U Mn d 0.06\n", [_u_channel("Mn", "d", 0.06)]),
        ("add-U Mn d 0.06 O p 0.1\n", [_u_channel("Mn", "d", 0.06), _u_channel("O", "p", 0.1)]),
        # How JDFTx echoes add-U in out files: one channel per continuation line
        ("add-U \tMn d 0.06 \\\n\tO p 0.1\n", [_u_channel("Mn", "d", 0.06), _u_channel("O", "p", 0.1)]),
        ("add-U Mn d 0.06 Vext 1 0.1 Vext 2 -0.1\n", [_u_channel("Mn", "d", 0.06, [(1, 0.1), (2, -0.1)])]),
        (
            "add-U Mn d 0.06 Vext 1 0.1 O 2p 0.1 Vext 3 0.2\n",
            [_u_channel("Mn", "d", 0.06, [(1, 0.1)]), _u_channel("O", "2p", 0.1, [(3, 0.2)])],
        ),
        (
            "add-U Mn d 0.06 O 2p 0.1 Vext 1 0.1\n",
            [_u_channel("Mn", "d", 0.06), _u_channel("O", "2p", 0.1, [(1, 0.1)])],
        ),
    ],
)
def test_add_u_read_write_roundtrip(text: str, expected_channels: list[dict]):
    jif = JDFTXInfile.from_str(text, dont_require_structure=True)
    assert jif["add-U"] == {"U-channels": expected_channels}
    written = str(jif)
    # JDFTx only accepts add-U once, so every channel must be written on a single add-U line
    assert written.count("add-U") == 1
    jif2 = JDFTXInfile.from_str(written, dont_require_structure=True)
    assert jif2["add-U"] == jif["add-U"]
    assert jif.is_comparable_to(jif2)


@pytest.mark.parametrize(
    ("text", "error_match"),
    [
        ("add-U Mn d 0.06\nadd-U O p 0.1\n", "The 'add-U' tag appears multiple times"),
        ("add-U Mn d\n", "The subtag 'U-channels' for tag 'add-U' is not optional"),
        ("add-U Mn d 0.06 Vext 1\n", "The subtag 'U-channels' for tag 'add-U' is not optional"),
        ("add-U Vext 1 0.1\n", "The subtag 'U-channels' for tag 'add-U' is not optional"),
        ("add-U Mn d 0.06 O p\n", re.escape("were not processed: ['O', 'p']")),
    ],
)
def test_add_u_read_errors(text: str, error_match: str):
    with pytest.raises(ValueError, match=error_match):
        JDFTXInfile.from_str(text, dont_require_structure=True)


def test_add_u_comparison():
    def from_str(value: str) -> JDFTXInfile:
        return JDFTXInfile.from_str(f"add-U {value}\n", dont_require_structure=True)

    jif = from_str("Mn d 0.06 Vext 1 0.1 Vext 2 -0.1 O p 0.1")
    # Neither channel order nor Vext order changes the input
    assert jif.is_comparable_to(from_str("O p 0.1 Mn d 0.06 Vext 2 -0.1 Vext 1 0.1"))
    assert not jif.is_comparable_to(from_str("Mn d 0.06 Vext 1 0.1 Vext 2 -0.1 O p 0.2"))
    # A Vext belongs to the channel it follows
    assert not jif.is_comparable_to(from_str("Mn d 0.06 Vext 1 0.1 O p 0.1 Vext 2 -0.1"))


def test_add_u_setitem():
    # Setting a value runs type validation, which must accept channels that differ in having Vext entries
    jif = JDFTXInfile.from_str("spintype z-spin\n", dont_require_structure=True)
    expected = {"U-channels": [_u_channel("Mn", "d", 0.06), _u_channel("O", "p", 0.1, [(1, 0.1)])]}
    jif["add-U"] = "Mn d 0.06 O p 0.1 Vext 1 0.1"
    assert jif["add-U"] == expected
    jif["add-U"] = {"U-channels": [_u_channel("Mn", "d", 0.06), _u_channel("O", "p", 0.1, [(1, 0.1)])]}
    assert jif["add-U"] == expected


def _ion_lines(*ions: tuple[str, str]) -> str:
    return "".join(f"ion {species} {x0} 0.0 0.0 1\n" for species, x0 in ions)


@pytest.mark.parametrize(
    ("text", "expected_species"),
    [
        # A nan in a non-optional ion subtag removes that ion
        (_ion_lines(("Mn", "0.0"), ("Fe", "nan"), ("O", "0.5")), ["Mn", "O"]),
        # Several nan ions are all removed, without later indices shifting onto the wrong ion
        (_ion_lines(("Mn", "0.0"), ("Fe", "nan"), ("Co", "nan"), ("O", "0.5")), ["Mn", "O"]),
        (_ion_lines(("Fe", "nan"), ("Mn", "0.0"), ("Co", "nan")), ["Mn"]),
        # Two nans in the same ion remove only that ion
        ("ion Mn nan nan 0.0 1\nion O 0.5 0.5 0.5 1\n", ["O"]),
        # If every ion is removed, the ion tag is removed instead of being left empty
        (_ion_lines(("Mn", "nan"), ("O", "nan")), None),
    ],
)
def test_nan_stripping_repeated_tag_entries(text: str, expected_species: list[str] | None):
    jif = JDFTXInfile.from_str("spintype z-spin\n" + text, dont_require_structure=True)
    cleaned = clean_infile_of_nans(jif)
    if expected_species is None:
        assert "ion" not in cleaned
    else:
        assert [ion["species-id"] for ion in cleaned["ion"]] == expected_species
    assert cleaned["spintype"] == "z-spin"
    # The original infile is left untouched
    assert len(jif["ion"]) == text.count("ion ")


def test_remove_at_path():
    params: dict[str, Any] = {"a": [1.0, 2.0, 3.0], "b": {"c": [4.0], "d": 5.0}}
    _remove_at_path(params, ["a", 1])
    assert params["a"] == [1.0, 3.0]
    _remove_at_path(params, ["b", "c", 0])  # emptied list is removed from its parent
    assert params["b"] == {"d": 5.0}
    _remove_at_path(params, ["b", "missing", 0])  # paths that no longer exist are ignored
    _remove_at_path(params, ["a", 5])
    assert params == {"a": [1.0, 3.0], "b": {"d": 5.0}}


@pytest.mark.parametrize(
    "lattice_value",
    [
        "Cubic 10.0",
        "Tetragonal 4.0 6.0",
        "Hexagonal 4.0 6.0",
        "Orthorhombic 4.0 5.0 6.0",
        "Rhombohedral 5.0 70.0",
        "Monoclinic 4.0 5.0 6.0 100.0",
        "Triclinic 4.0 5.0 6.0 80.0 85.0 95.0",
    ],
)
def test_named_lattice_latt_scale(lattice_value: str):
    scale = np.array([2.0, 0.5, 1.8897])
    unscaled = _infile_to_pmg_lattice(JDFTXInfile.from_str(f"lattice {lattice_value}\n", dont_require_structure=True))
    scaled = _infile_to_pmg_lattice(
        JDFTXInfile.from_str(
            f"lattice {lattice_value}\nlatt-scale {scale[0]} {scale[1]} {scale[2]}\n", dont_require_structure=True
        )
    )
    # Lattice vector k is scaled by s_k, so its length scales and the angles between vectors are unchanged
    np.testing.assert_allclose(scaled.abc, np.array(unscaled.abc) * scale)
    np.testing.assert_allclose(scaled.angles, unscaled.angles)


def test_latt_scale_named_and_matrix_lattices_agree():
    named = JDFTXInfile.from_str("lattice Cubic 10.0\nlatt-scale 2 1 1.5\n", dont_require_structure=True)
    matrix = JDFTXInfile.from_str(
        "lattice \\\n 10 0 0 \\\n 0 10 0 \\\n 0 0 10\nlatt-scale 2 1 1.5\n", dont_require_structure=True
    )
    np.testing.assert_allclose(_infile_to_pmg_lattice(named).abc, _infile_to_pmg_lattice(matrix).abc)
    np.testing.assert_allclose(_infile_to_pmg_lattice(named).abc, np.array([20.0, 10.0, 15.0]) * bohr_to_ang)


def test_latt_scale_keeps_fractional_values():
    # e.g. an angstrom-to-bohr conversion factor must not be truncated to 1
    jif = JDFTXInfile.from_str("latt-scale 1.8897 1.8897 1.8897\n", dont_require_structure=True)
    assert jif["latt-scale"] == {"s0": 1.8897, "s1": 1.8897, "s2": 1.8897}
    assert JDFTXInfile.from_str(str(jif), dont_require_structure=True)["latt-scale"] == jif["latt-scale"]


def test_line_continuation():
    # Whitespace around a "\" collapses to a single space, and comments are stripped before lines are joined
    joined = JDFTXInfile.from_str(
        "lattice \\\n 10 0 0  \\  # comment\n0 10 0\\\n0 0 10\nelec-cutoff 20\n\\\nelec-ex-corr gga\n",
        dont_require_structure=True,
    )
    single = JDFTXInfile.from_str(
        "lattice 10 0 0 0 10 0 0 0 10\nelec-cutoff 20\nelec-ex-corr gga\n", dont_require_structure=True
    )
    assert joined == single


@pytest.mark.parametrize(
    ("selective_dynamics", "expected"),
    [
        ([[True, True, True], [True, False, True]], [1, 0]),
        (np.array([[1, 1, 1], [0, 0, 0]]), [1, 0]),
        ([(1, 1, 1), np.array([0.0, 1.0, 1.0]), [], True, 0.0, None], [1, 0, 1, 1, 0, 0]),
    ],
)
def test_selective_dynamics_site_prop_to_jdftx_interpretable(selective_dynamics: Any, expected: list[int]):
    converted = selective_dynamics_site_prop_to_jdftx_interpretable(selective_dynamics)
    np.testing.assert_array_equal(converted, expected)
    assert converted.dtype.kind == "i"


def test_get_nanmaps():
    assert _get_nanmaps(np.nan) == [[]]
    assert _get_nanmaps(1.0) == _get_nanmaps("nan") == _get_nanmaps(None) == []
    val = {"a": [1.0, np.nan, [np.nan]], "b": {"c": np.nan, "d": "nan", "e": True}}
    assert _get_nanmaps(val) == [["a", 1], ["a", 2, 0], ["b", "c"]]
    # For a JDFTXInfile, each path starts at its tag name
    jif = JDFTXInfile.from_str("fluid-solvent H2O 55.338 ScalarEOS epsBulk nan\n", dont_require_structure=True)
    assert _get_nanmaps(jif) == [["fluid-solvent", 0, "epsBulk"]]


def test_jdftxstructure_sort_structure():
    """Sites are sorted into JDFTx's order: grouped by species name (ignoring case) in the order the species first
    appear. Each site's moveScale, velocity and constraint move with it, however they were given.
    """
    # Every ion has a velocity and a single-vector constraint, so these form regular arrays and are site properties too
    infile = JDFTXInfile.from_str(
        "lattice \\\n 10 0 0 \\\n 0 10 0 \\\n 0 0 10\n"
        "ion O 0.5 0.5 0.5 v 0.1 0.0 0.0 0.5 Planar 1 0 0\n"
        "ion Fe2 0.1 0.0 0.0 v 0.0 0.2 0.0 1 Linear 0 0 1\n"
        "ion Fe1 0.3 0.0 0.0 v 0.0 0.0 0.3 1 Linear 0 1 0\n"
        "ion o 0.2 0.2 0.2 v 0.4 0.0 0.0 1 Linear 1 1 0\n"
        "ion Fe2 0.7 0.0 0.0 v 0.0 0.5 0.0 2 Planar 0 0 1\n"
    )
    unsorted = JDFTXInfile.to_jdftxstructure(infile)
    ions = JDFTXInfile.from_str(unsorted.get_str())["ion"]
    expected = [ions[i] for i in (0, 3, 1, 4, 2)]
    assert [ion["species-id"] for ion in expected] == ["O", "o", "Fe2", "Fe2", "Fe1"]
    # From an input file (data given both as arguments and as site properties), from arguments only, and from site
    # properties only
    species_names = {"jdftx_species": unsorted.structure.site_properties["jdftx_species"]}
    for jstruct in (
        JDFTXInfile.to_jdftxstructure(infile, sort_structure=True),
        JDFTXStructure(
            unsorted.structure.copy(site_properties=species_names),
            movescale=unsorted.movescale,
            velocities=unsorted.velocities,
            constraint_types=unsorted.constraint_types,
            constraint_vectors=unsorted.constraint_vectors,
            sort_structure=True,
        ),
        JDFTXStructure(unsorted.structure, sort_structure=True),
    ):
        assert_same_value(JDFTXInfile.from_str(jstruct.get_str())["ion"], expected)
    sorted_structure = JDFTXInfile.to_pmg_structure(infile, sort_structure=True)
    assert sorted_structure.site_properties["jdftx_species"] == ["O", "o", "Fe2", "Fe2", "Fe1"]


def test_jdftxstructure_label_fallbacks():
    lattice = Lattice.cubic(5.0)
    # Labels that aren't element symbols fall back to the species string, then to the element of an oxidized species
    struct = Structure(lattice, ["Si", Species("Fe", 2)], [[0, 0, 0], [0.5, 0.5, 0.5]], labels=["Si0", "Fe_a"])
    assert [ion["species-id"] for ion in JDFTXInfile.from_structure(struct)["ion"]] == ["Si", "Fe"]
    with pytest.raises(ValueError, match="Could not correct site label"):
        JDFTXStructure(structure=Structure(lattice, [DummySpecies("X")], [[0, 0, 0]])).get_str()


def test_matrix_lattice_conversion():
    # Each column of the lattice tag is a lattice vector in bohr, scaled by its latt-scale entry
    jif = JDFTXInfile.from_str("lattice 1 2 3 4 5 6 7 8 10\nlatt-scale 1 2 3\n", dont_require_structure=True)
    expected = np.array([[1, 4, 7], [4, 10, 16], [9, 18, 30]]) * bohr_to_ang
    np.testing.assert_allclose(_infile_to_pmg_lattice(jif).matrix, expected)
    # Writing a structure puts its lattice vectors back into columns, converted to bohr
    written = JDFTXInfile.from_str(
        JDFTXStructure(structure=Structure(Lattice(expected), ["Si"], [[0, 0, 0]])).get_str()
    )
    written_matrix = [[written["lattice"][f"R{i}{j}"] for j in range(3)] for i in range(3)]
    np.testing.assert_allclose(written_matrix, expected.T / bohr_to_ang)


def test_jdftxstructure_keeps_structure_species():
    # Species (including oxidation states) on the wrapped structure are left as they are
    struct = Structure(Lattice.cubic(5.0), ["Si", Species("Fe", 2), DummySpecies("X")], np.eye(3) / 2)
    species_before = struct.species
    assert JDFTXStructure(structure=struct).structure.species == species_before


# Two Fe species (from pseudopotential files Fe1.upf and Fe2.upf, as for an antiferromagnet), and one named after O
# Every ion has a velocity, so velocities form a regular array and are a site property
SPECIES_NAMES_TEXT = (
    "lattice Cubic 10\nion Fe1 0 0 0 v 0.1 0 0 0\nion Fe2 0.5 0.5 0.5 v 0 0 0 1\nion O 0.25 0.25 0.25 v 0 0 0 1\n"
)


def _written_species(jif: JDFTXInfile) -> list[str]:
    return [ion["species-id"] for ion in jif["ion"]]


def test_ion_species_names():
    """Species named after their pseudopotential files get their element from the name, and are written back."""
    jif = JDFTXInfile.from_str(SPECIES_NAMES_TEXT)
    assert _written_species(jif) == ["Fe1", "Fe2", "O"]
    structure = jif.structure
    assert [site.specie.symbol for site in structure] == ["Fe", "Fe", "O"]
    assert structure.site_properties["jdftx_species"] == ["Fe1", "Fe2", "O"]
    assert _written_species(JDFTXInfile.from_structure(structure)) == ["Fe1", "Fe2", "O"]
    # The names decide each ion's species, so they are kept without the other site properties too, as are movescale
    # and selective_dynamics (which JDFTXStructure keeps in sync with the site properties)
    assert jif.to_pmg_structure(jif, fill_site_properties=False).site_properties == {
        "jdftx_species": ["Fe1", "Fe2", "O"],
        "movescale": [0.0, 1.0, 1.0],
        "selective_dynamics": [[False] * 3, [True] * 3, [True] * 3],
    }
    without_properties = JDFTXInfile.from_structure(structure, ignore_site_properties=True)
    assert _written_species(without_properties) == ["Fe1", "Fe2", "O"]
    assert "v" not in without_properties["ion"][0]
    assert without_properties["ion"][0]["moveScale"] == 1
    assert "velocities" in structure.site_properties  # The given structure is left as it was


@pytest.mark.parametrize(
    ("species", "element"),
    [("Fe", "Fe"), ("fe", "Fe"), ("FE", "Fe"), ("Fe1", "Fe"), ("Co2", "Co"), ("Hx", "H"), ("Hf", "Hf"), ("H", "H")],
)
def test_ion_species_element(species: str, element: str):
    structure = JDFTXInfile.from_str(f"lattice Cubic 10\nion {species} 0 0 0 1\n").structure
    assert structure[0].specie.symbol == element
    # Only names that aren't their element symbol (ignoring case, as JDFTx does) are kept
    if species.lower() == element.lower():
        assert "jdftx_species" not in structure.site_properties
    else:
        assert structure.site_properties["jdftx_species"] == [species]


@pytest.mark.parametrize("species", ["Xx9", "1H", "Fe_1"])
def test_ion_species_name_must_give_an_element(species: str):
    with pytest.raises(ValueError, match=f"'{species}' for tag 'species-id' is not a species name pymatgen can read"):
        JDFTXInfile.from_str(f"lattice Cubic 10\nion {species} 0 0 0 1\n")


def test_ion_species_name_not_of_the_site_element_is_not_written():
    structure = JDFTXInfile.from_str(SPECIES_NAMES_TEXT).structure
    structure.replace_species({"Fe": "Co"})  # Keeps the jdftx_species names
    structure.append("O", [0.75, 0.75, 0.75])  # Has no name (None)
    with pytest.warns(UserWarning, match=re.escape("species names of sites [0, 1] aren't of their elements")):
        jif = JDFTXInfile.from_structure(structure)
    assert _written_species(jif) == ["Co", "Co", "O", "O"]


def test_site_labels_are_not_species_names():
    # Labels from other sources (e.g. CIF files) usually aren't JDFTx species names, so elements are written
    structure = Structure(Lattice.cubic(5.0), ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]], labels=["Fe1", "O2"])
    assert _written_species(JDFTXInfile.from_structure(structure)) == ["Fe", "O"]


# Non-orthogonal cell (lattice vectors are the columns, in bohr), so lattice and Cartesian constraint vectors differ
CONSTRAINT_R = np.array([[10.0, 2.0, 1.0], [0.0, 9.0, 3.0], [0.0, 0.0, 8.0]])
CONSTRAINT_TEXT = (
    "lattice 10 2 1 0 9 3 0 0 8\n"
    "ion H 0.1 0.2 0.3 1 Linear 1 0 0\n"
    "ion O 0.4 0.5 0.6 v 0.01 0.02 0.03 1 Planar 0 1 1\n"
    "ion H 0.7 0.8 0.9 1 HyperPlane 1 2 0 g1 HyperPlane 0 0 1 g2\n"
    "ion O 0.2 0.4 0.6 1 HyperPlane 0 1 0\n"
    "ion H 0.5 0.5 0.5 1\n"
)


def _assert_same_constraints(jstruct1: JDFTXStructure, jstruct2: JDFTXStructure):
    assert jstruct1.constraint_types == jstruct2.constraint_types
    assert jstruct1.hyperplane_group_names == jstruct2.hyperplane_group_names
    for v1, v2 in zip(jstruct1.constraint_vectors, jstruct2.constraint_vectors, strict=True):
        assert (v1 is None and v2 is None) or np.allclose(v1, v2)


def test_jdftxstructure_constraints_are_cartesian():
    jstruct = JDFTXStructure.from_jdftxinfile(JDFTXInfile.from_str(CONSTRAINT_TEXT))
    assert jstruct.constraint_types == ["Linear", "Planar", "HyperPlane", "HyperPlane", None]
    # A HyperPlane site holds a list of vectors and a list of group names (None for an unlabeled hyperplane)
    assert jstruct.hyperplane_group_names == [None, None, ["g1", "g2"], [None], None]
    vectors = jstruct.constraint_vectors
    # Lattice-coordinate line directions are covariant (d_cart = R d), plane normals are contravariant (R^T d_cart = d)
    np.testing.assert_allclose(vectors[0], CONSTRAINT_R @ [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(CONSTRAINT_R.T @ vectors[1], [0, 1, 1], atol=1e-12)
    np.testing.assert_allclose([CONSTRAINT_R.T @ v for v in vectors[2]], [[1, 2, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose([CONSTRAINT_R.T @ v for v in vectors[3]], [[0, 1, 0]], atol=1e-12)
    assert vectors[4] is None


def test_jdftxstructure_constraints_roundtrip():
    jif = JDFTXInfile.from_str(CONSTRAINT_TEXT)
    jstruct = JDFTXStructure.from_jdftxinfile(jif)
    # Writing in the same coordinates gives back the original ion tags
    assert_same_value(JDFTXInfile.from_str(jstruct.get_str())["ion"], jif["ion"])
    # Writing in Cartesian coordinates converts the constraint vectors along with the positions
    cart_jif = JDFTXInfile.from_str(jstruct.get_str(in_cart_coords=True))
    assert cart_jif["coords-type"] == "Cartesian"
    _assert_same_constraints(JDFTXStructure.from_jdftxinfile(cart_jif), jstruct)
    # Constraints on only some ions (or HyperPlane ions with different numbers of planes) don't form regular arrays, so
    # they aren't site properties (which would break ASE conversion) and don't survive a trip through a Structure
    assert not {"constraint_types", "constraint_vectors", "group_names"} & set(jif.structure.site_properties)
    assert JDFTXStructure.from_jdftxinfile(JDFTXInfile.from_structure(jif.structure)).constraint_types is None
    # Regular ones are site properties, and do
    regular = JDFTXInfile.from_str(
        "lattice 10 2 1 0 9 3 0 0 8\nion H 0.1 0.2 0.3 1 Linear 1 0 0\nion O 0 0 0 1 Planar 0 1 1\n"
    )
    _assert_same_constraints(
        JDFTXStructure.from_jdftxinfile(JDFTXInfile.from_structure(regular.structure)),
        JDFTXStructure.from_jdftxinfile(regular),
    )


def test_ragged_constraints_reach_ase():
    """Structures with constraints on only some ions convert to ASE Atoms (the constraints aren't site properties)."""
    pytest.importorskip("ase")
    atoms = AseAtomsAdaptor.get_atoms(JDFTXInfile.from_str(CONSTRAINT_TEXT).structure)
    assert len(atoms) == 5


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"constraint_types": ["Linear", None]}, "Constraint types are present but constraint vectors are not"),
        ({"constraint_types": ["Twisted", None], "constraint_vectors": [[0, 0, 1], None]}, "Unknown constraint type"),
        (
            {"constraint_types": ["Linear", "Linear"], "constraint_vectors": [[0, 0, 1], None]},
            "has no constraint vector",
        ),
        ({"constraint_types": ["Planar", None], "constraint_vectors": [[0, 0, 0], None]}, "must be a single non-zero"),
        (
            {
                "constraint_types": ["Linear", None],
                "constraint_vectors": [[0, 0, 1], None],
                "selective_dynamics": [0, 1],
            },
            "moveScale of 0",
        ),
        (
            {
                "constraint_types": ["HyperPlane", None],
                "constraint_vectors": [[[1, 0, 0], [0, 1, 0]], None],
                "hyperplane_group_names": [["g1", None], None],
            },
            "needs a group name",
        ),
    ],
)
def test_jdftxstructure_constraint_errors(kwargs: dict, error_match: str):
    jstruct = JDFTXStructure(
        structure=Structure(Lattice.cubic(5.0), ["H", "H"], [[0, 0, 0], [0.5, 0.5, 0.5]]), **kwargs
    )
    with pytest.raises(ValueError, match=error_match):
        jstruct.get_str()


def test_mixed_format_ion_tags():
    # Linear and labeled-HyperPlane ion tags use different ion tag formats in the same list
    text = "lattice Cubic 10.0\nion H 0.1 0.2 0.3 1 Linear 1 0 0\nion O 0.7 0.8 0.9 1 HyperPlane 1 2 0 g1\n"
    jif = JDFTXInfile.from_str(text)
    assert JDFTXInfile.from_str(str(jif))["ion"] == jif["ion"]
    assert jif.get_differing_tags(JDFTXInfile.from_str(text)) == []
    assert jif.get_differing_tags(JDFTXInfile.from_str(text.replace("HyperPlane 1 2 0", "HyperPlane 5 2 0"))) == ["ion"]
    # A nan in one entry only removes that entry
    nan_jif = JDFTXInfile.from_str(text.replace("0.8 0.9 1 HyperPlane", "nan 0.9 1 HyperPlane"))
    assert clean_infile_of_nans(nan_jif)["ion"] == jif["ion"][:1]


def test_copy_is_independent():
    text = "fluid-solvent H2O 55.338\nfluid-solvent CH3CN 0.5 ScalarEOS epsBulk nan\n"
    jif = JDFTXInfile.from_str(text, dont_require_structure=True)
    copied = jif.copy()
    copied["fluid-solvent"][1]["concentration"] = 1.0
    assert jif["fluid-solvent"][1]["concentration"] == 0.5
    # Cleaning nans works on a copy, so the original's nested entries are left untouched
    assert "epsBulk" not in clean_infile_of_nans(jif)["fluid-solvent"][1]
    assert "epsBulk" in jif["fluid-solvent"][1]


def test_jdftxstructure_velocities_are_cartesian():
    # Velocities (bohr or lattice coordinates per Hartree atomic time unit) are stored in Cartesian angstrom/fs
    au_time_fs = 2.4188843265864e-2  # CODATA Hartree atomic unit of time, in fs
    r = np.array([[10.0, 2.0, 1.0], [0.0, 9.0, 3.0], [0.0, 0.0, 8.0]])  # Lattice vectors are the columns, in bohr
    jif = JDFTXInfile.from_str(
        "lattice 10 2 1 0 9 3 0 0 8\nion H 0.1 0.2 0.3 v 0.01 0.02 0.03 1\nion O 0.4 0.5 0.6 1\n"
    )
    jstruct = JDFTXStructure.from_jdftxinfile(jif)
    np.testing.assert_allclose(jstruct.velocities[0], r @ [0.01, 0.02, 0.03] * bohr_to_ang / au_time_fs)
    assert jstruct.velocities[1] is None
    # Writing in the same coordinates gives back the original ion tags
    assert_same_value(JDFTXInfile.from_str(jstruct.get_str())["ion"], jif["ion"])
    # Writing in Cartesian coordinates converts the velocities along with the positions
    cart_jstruct = JDFTXStructure.from_jdftxinfile(JDFTXInfile.from_str(jstruct.get_str(in_cart_coords=True)))
    np.testing.assert_allclose(cart_jstruct.velocities[0], jstruct.velocities[0])


def _jdftx_named_lattice(a, b, c, alpha, beta, gamma):
    """Lattice vectors (columns, bohr) as JDFTx builds a named lattice (GridInfo::setLatticeVectors)."""
    cos_a, cos_b = np.cos(np.radians(alpha)), np.cos(np.radians(beta))
    sin_g, cos_g = np.sin(np.radians(gamma)), np.cos(np.radians(gamma))
    r = np.zeros((3, 3))
    r[:, 0] = [a, 0, 0]
    r[:, 1] = [b * cos_g, b * sin_g, 0]
    r[:, 2] = [c * cos_b, c * (cos_a - cos_b * cos_g) / sin_g, 0]
    r[2, 2] = np.sqrt(c * c - r[0, 2] ** 2 - r[1, 2] ** 2)
    return r


@pytest.mark.parametrize(
    ("lattice_value", "params"),
    [
        ("Cubic 10", (10, 10, 10, 90, 90, 90)),
        ("Hexagonal 8 12", (8, 8, 12, 90, 90, 120)),
        ("Monoclinic 8 9 10 100", (8, 9, 10, 90, 100, 90)),
        ("Rhombohedral 9 70", (9, 9, 9, 70, 70, 70)),
        ("Triclinic 8 9 10 80 85 95", (8, 9, 10, 80, 85, 95)),
    ],
)
def test_named_lattice_matches_jdftx(lattice_value: str, params: tuple):
    # JDFTx puts a along x and b in the xy-plane, which decides where Cartesian ion coordinates end up in the cell
    jif = JDFTXInfile.from_str(f"lattice {lattice_value}\ncoords-type Cartesian\nion H 1 2 3 1\n")
    r = _jdftx_named_lattice(*params)
    np.testing.assert_allclose(jif.structure.lattice.matrix, r.T * bohr_to_ang, atol=1e-12)
    np.testing.assert_allclose(jif.structure.frac_coords[0], np.linalg.solve(r, [1, 2, 3]), atol=1e-12)


def test_structure_values_keep_jdftx_precision():
    # JDFTx prints positions and velocities with 15 decimals, which must survive reading and writing unchanged
    text = (
        "lattice \\\n 12.345678901234567 0 0 \\\n 0 10.000000000000002 0 \\\n 1.5 0 9.876543210987654\n"
        "ion H 0.123456789012345 0.987654321098765 0.5 v 0.000012345678901 -0.000009876543210 0.000000123456789 1\n"
    )
    jif = JDFTXInfile.from_str(text)
    assert JDFTXInfile.from_str(str(jif)) == jif
    # Positions and velocities also survive conversion to and from a JDFTXStructure (angstrom, angstrom/fs)
    assert JDFTXInfile.from_str(jif.to_jdftxstructure(jif).get_str())["ion"] == jif["ion"]


def test_jdftxstructure_lattice_coords_not_scaled():
    """Fractional (lattice) ion coordinates must not be scaled by the Bohr-to-Angstrom factor."""
    struct = Structure(np.eye(3) * 5.0, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    jif = JDFTXInfile.from_structure(struct)
    assert jif["coords-type"] == "Lattice"
    assert jif.structure.frac_coords == pytest.approx(struct.frac_coords)
    jif_cart = JDFTXInfile.from_structure(struct, write_cart_coords=True)
    assert jif_cart["coords-type"] == "Cartesian"
    assert jif_cart.structure.cart_coords == pytest.approx(struct.cart_coords)


def test_jdftxstructure_hyperplane_round_trip():
    """HyperPlane constraints read from an infile can be written back out."""
    text = (
        "lattice \\\n 10 0 0 \\\n 0 10 0 \\\n 0 0 10\ncoords-type Cartesian\n"
        "ion H 1.0 1.0 1.0 1 HyperPlane 1.0 0.0 0.0 g1 HyperPlane 0.0 1.0 0.0 g2\n"
        "ion H 3.0 1.0 1.0 1 HyperPlane 0.0 0.0 1.0 g3\n"
    )
    jif = JDFTXInfile.from_str(text)
    jstruct = jif.to_jdftxstructure(jif)
    assert jstruct.constraint_types == ["HyperPlane", "HyperPlane"]
    assert jstruct.constraint_vectors == [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]]
    assert jstruct.hyperplane_group_names == [["g1", "g2"], ["g3"]]
    jif2 = JDFTXInfile.from_str(jstruct.get_str(in_cart_coords=True))
    for ion1, ion2 in zip(jif["ion"], jif2["ion"], strict=True):
        assert ion2["HyperPlane"] == ion1["HyperPlane"]


def test_get_filtered_differing_tags_does_not_mutate_args():
    jif1 = JDFTXInfile.from_file(ex_infile1_fname)
    jif2 = jif1.copy()
    jif2["elec-n-bands"] = 20001
    exclude_tags = ["elec-n-bands"]
    ensure_include_tags = ["elec-n-bands"]
    jif1.get_filtered_differing_tags(jif2, exclude_tags=exclude_tags, exclude_tag_categories=["electronic"])
    assert exclude_tags == ["elec-n-bands"]
    exclude_tags = ["elec-n-bands"]
    jif1.get_filtered_differing_tags(jif2, exclude_tags=exclude_tags, ensure_include_tags=ensure_include_tags)
    assert exclude_tags == ["elec-n-bands"]


@pytest.mark.parametrize(
    ("exclude_tags", "exclude_tag_categories"),
    [
        (["elec-n-bands"], ["electronic"]),  # named and in an excluded category
        (["elec-n-bands", "elec-n-bands"], None),  # named twice
        (None, ["electronic", "electronic"]),  # category named twice
    ],
)
def test_get_filtered_differing_tags_ensure_include_beats_repeated_exclusion(exclude_tags, exclude_tag_categories):
    """A tag in ensure_include_tags is compared however many times it was excluded."""
    jif1 = JDFTXInfile.from_file(ex_infile1_fname)
    jif2 = jif1.copy()
    jif2["elec-n-bands"] = 20001
    differing_tags = jif1.get_filtered_differing_tags(
        jif2,
        exclude_tags=exclude_tags,
        exclude_tag_categories=exclude_tag_categories,
        ensure_include_tags=["elec-n-bands"],
    )
    assert differing_tags == ["elec-n-bands"]
    assert not jif1.is_comparable_to(
        jif2,
        exclude_tags=exclude_tags,
        exclude_tag_categories=exclude_tag_categories,
        ensure_include_tags=["elec-n-bands"],
    )
