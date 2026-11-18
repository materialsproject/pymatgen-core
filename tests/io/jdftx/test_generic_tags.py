from __future__ import annotations

import re
import warnings

import pytest

from pymatgen.io.jdftx.generic_tags import (
    AbstractTag,
    BoolTag,
    BoolTagContainer,
    FloatTag,
    IntTag,
    JDFTXOptionWarning,
    MultiformatTag,
    StrListTag,
    StrTag,
    SubcommandTagContainer,
    TagContainer,
    element_from_species_name,
)
from pymatgen.io.jdftx.jdftxinfile_master_format import get_dump_tag_container, get_tag_object

from .shared_test_utils import assert_same_value


class Unstringable:
    """Dummy class that cannot be converted to a string"""

    def __str__(self):
        raise ValueError("Cannot convert to string")


class NonIterable:
    """Dummy class that cannot be iterated through"""

    def __iter__(self):
        raise ValueError("Cannot iterate through this object")


def test_abstract_tag():
    with pytest.raises(TypeError):
        # AbstractTag cannot be instantiated directly
        AbstractTag()


def test_stringify():
    str_tag = StrTag(options=["ken"])
    out_str = str(str_tag)
    assert isinstance(out_str, str)
    assert out_str


def test_bool_tag():
    """Expected behavior of BoolTag is tested here"""
    bool_tag = BoolTag(write_value=False)
    # Values with spaces are impossible to interpret as bools
    tag = "barbie"
    value = "this string has spaces"
    with pytest.raises(ValueError, match=f"'{value}' for '{tag}' should not have a space in it!"):
        bool_tag.read(tag, value)
    # Value errors should be raised if value cannot be conveniently converted to a bool
    value = "non-bool-like"
    with pytest.raises(ValueError, match=f"Could not set '{value}' as True/False for tag '{tag}'!"):
        bool_tag.read(tag, value)
    # bool_tag = BoolTag(write_value=True)
    # Only values appearing in the "_TF_options" map can be converted to bool (allows for yes/no to be interpreted
    # as bools)
    value = "not-appearing-in-read-TF-options"
    with pytest.raises(ValueError, match=f"Could not set '{value}' as True/False for tag '{tag}'!"):
        bool_tag.read(tag, value)


def test_abstract_tag_inheritor():
    """Expected behavior of methods inherited from AbstractTag are tested here
    (AbstractTag cannot be directly initiated)"""
    bool_tag = BoolTag(write_value=False)
    tag = "barbie"
    value = "ken"
    # Abstract-tag inheritors should raise the following error for calling get_list_representation unless
    # they have implemented the method (bool_tag does not and should not)
    with pytest.raises(ValueError, match=f"Tag object with tag '{tag}' has no get_list_representation method"):
        bool_tag.get_list_representation(tag, value)
    # Abstract-tag inheritors should raise the following error for calling get_dict_representation unless
    # they have implemented the method (bool_tag does not and should not)
    with pytest.raises(ValueError, match=f"Tag object with tag '{tag}' has no get_dict_representation method"):
        bool_tag.get_dict_representation(tag, value)
    # "_validate_repeat" is only called if "can_repeat" is True, but must raise an error if value is not a list
    with pytest.raises(TypeError):
        bool_tag._validate_repeat(tag, value)


def test_str_tag():
    str_tag = StrTag(options=["ken"])
    # Values with spaces are rejected as spaces are occasionally used as delimiters in input files
    tag = "barbie"
    value = "ken, allan"
    with pytest.raises(ValueError, match=f"'{value}' for '{tag}' should not have a space in it!"):
        str_tag.read(tag, value)
    # Values that cannot be converted to strings have a safety net error message
    value = Unstringable()
    print_value = "(unstringable)"
    with pytest.raises(TypeError, match=re.escape(f"Value '{print_value}' for '{tag}' should be a string!")):
        str_tag.read(tag, value)
    # "str_tag" here was initiated with only "ken" as an option, so "allan" should raise an error
    # (barbie is the tagname, not the value)
    value = "allan"
    with pytest.raises(
        ValueError, match=re.escape(f"The string value '{value}' must be one of {str_tag.options} for tag '{tag}'")
    ):
        str_tag.read(tag, value)
    # Options match ignoring case, as in JDFTx, and read as the option's own spelling
    assert StrTag(options=["Ken"]).read(tag, "kEN") == "Ken"
    # If both the tagname and value are written, two tokens are to be written
    str_tag = StrTag(write_tagname=True, write_value=True)
    assert str_tag.get_token_len() == 2
    # If only the tagname is written, one token is to be written
    str_tag = StrTag(write_tagname=True, write_value=False)
    assert str_tag.get_token_len() == 1
    # If only the value is written, one token is to be written
    str_tag = StrTag(write_tagname=False, write_value=True)
    assert str_tag.get_token_len() == 1
    # If neither the tagname nor the value are written, no tokens are to be written
    # (this is useless, but it is a valid option)
    str_tag = StrTag(write_tagname=False, write_value=False)
    assert str_tag.get_token_len() == 0


def test_int_tag():
    int_tag = IntTag()
    # Values with spaces are rejected as spaces are occasionally used as delimiters in input files
    value = "ken, allan"
    tag = "barbie"
    with pytest.raises(ValueError, match=f"'{value}' for '{tag}' should not have a space in it!"):
        int_tag.read(tag, value)
    # Values passed to "read" must be strings
    value = {}
    with pytest.raises(TypeError, match=f"Value '{value}' for '{tag}' should be a string!"):
        int_tag.read(tag, value)
    # Values must be able to be type-cast to an integer
    value = "ken"  # (ken is not an integer)
    with pytest.raises(ValueError, match=f"Could not set value '{value}' to an int for tag '{tag}'!"):
        int_tag.read(tag, value)


def test_float_tag():
    float_tag = FloatTag()
    tag = "barbie"
    value = "ken, allan"
    with pytest.raises(ValueError, match=f"'{value}' for '{tag}' should not have a space in it!"):
        float_tag.read(tag, value)
    with pytest.raises(TypeError):
        float_tag.read(tag, {})
    value = "ken"  # (ken is not a number)
    with pytest.raises(ValueError, match=f"Could not set value '{value}' to a float for tag '{tag}'!"):
        float_tag.read(tag, value)
    value = "1.2.3"  # (1.2.3 cannot be a float)
    with pytest.raises(ValueError, match=f"Could not set value '{value}' to a float for tag '{tag}'!"):
        float_tag.read("barbie", "1.2.3")
    # A bare tag is only valid if it has a value to read it as
    with pytest.raises(ValueError, match="Could not set value '' to a float for tag 'barbie'!"):
        float_tag.read("barbie", "")
    assert FloatTag(value_if_bare=0.0).read("barbie", "") == 0.0
    assert FloatTag(value_if_bare=0.0).read("barbie", "1.5") == 1.5


@pytest.mark.parametrize(
    ("name", "element"),
    [
        ("Fe", "Fe"),
        ("fe1", "Fe"),
        ("FE1", "Fe"),
        ("Co2", "Co"),  # The longest element symbol the name starts with, not C
        ("Hx", "H"),
        ("Hf", "Hf"),
        ("O", "O"),
        ("Xx9", None),
        ("1H", None),
        ("Fe_1", None),  # JDFTx's species names are letters and digits only
        ("Fé", None),
        ("", None),
    ],
)
def test_element_from_species_name(name: str, element: str | None):
    assert element_from_species_name(name) == element


def test_str_tag_species_name():
    tag = StrTag(species_name=True)
    assert tag.read("barbie", "Fe1") == "Fe1"
    with pytest.raises(ValueError, match="'Xx9' for tag 'barbie' is not a species name pymatgen can read"):
        tag.read("barbie", "Xx9")


def test_multiformat_tag_error_gives_each_format_reason():
    # Formats failing for the same reason are listed together
    tag = MultiformatTag(format_options=[StrTag(species_name=True), StrTag(species_name=True), FloatTag()])
    with pytest.raises(ValueError, match="No valid read format") as err:
        tag.get_format_index_for_str_value("barbie", "Xx9")
    assert "Format 0, 1: 'Xx9' for tag 'barbie' is not a species name" in str(err.value)
    assert "Format 2: Could not set value 'Xx9' to a float for tag 'barbie'!" in str(err.value)


def test_str_tag_option_warnings():
    tag = StrTag(options=["ken", "allan"], option_warnings={"Ken": "Not ken!"})
    with pytest.warns(JDFTXOptionWarning, match="'barbie ken': Not ken!"):
        assert tag.read("barbie", "ken") == "ken"
    with warnings.catch_warnings():
        warnings.simplefilter("error", JDFTXOptionWarning)
        assert tag.read("barbie", "allan") == "allan"
        # Finding the format of a value does not warn: the value is read again with the format found
        assert MultiformatTag(format_options=[tag, FloatTag()]).get_format_index_for_str_value("barbie", "ken") == 0


def test_str_list_tag():
    tag = StrListTag(options=["A", "BC"], can_repeat=True)
    assert tag.read("barbie", "A BC") == ["A", "BC"]
    assert tag.read("barbie", "") == []
    # Two options run together are split, but only into options
    assert tag.read("barbie", "ABC") == ["A", "BC"]
    with pytest.raises(ValueError, match="must be one of"):
        tag.read("barbie", "AB")
    assert StrListTag().read("barbie", "ken allan") == ["ken", "allan"]


def test_multiformat_tag_validate_value_bounds():
    """Bounds are checked with the format option the value matches."""
    tag = MultiformatTag(format_options=[StrTag(options=["D3"]), FloatTag(lb=0.0)])
    assert tag.validate_value_bounds("barbie", 0.5) == (True, "")
    assert tag.validate_value_bounds("barbie", "D3") == (True, "")
    is_valid, err = tag.validate_value_bounds("barbie", -0.5)
    assert not is_valid
    assert "not within bounds" in err
    # A nan is an unset value, not an out-of-bounds one
    assert tag.validate_value_bounds("barbie", float("nan")) == (True, "")
    # Each entry of a repeated tag is checked with its own format option
    tag = MultiformatTag(
        can_repeat=True,
        format_options=[
            TagContainer(can_repeat=True, subtags={"a": FloatTag(write_tagname=False, lb=0.0)}),
            TagContainer(can_repeat=True, subtags={"b": IntTag(write_tagname=False, ub=1)}),
        ],
    )
    assert tag.validate_value_bounds("barbie", [{"a": 1.0}, {"b": 0}]) == (True, "")
    with pytest.warns(UserWarning, match="Invalid value"):
        is_valid, err = tag.validate_value_bounds("barbie", [{"a": 1.0}, {"b": 2}])
    assert not is_valid
    assert "'b'" in err


def test_read_from_tokens_consumes_one_token_for_single_value_tags():
    assert StrTag().read_from_tokens("tag", ["a", "b"]) == ("a", 1)
    assert FloatTag().read_from_tokens("tag", ["1.5", "b"]) == (1.5, 1)
    assert IntTag().read_from_tokens("tag", ["3", "4"]) == (3, 1)


def test_positional_repeatable_subtag_reads_greedily():
    tag = "barbie"
    tagcontainer = TagContainer(
        subtags={
            "name": StrTag(write_tagname=False, optional=False),
            "values": FloatTag(write_tagname=False, can_repeat=True, optional=False),
        }
    )
    assert tagcontainer.read(tag, "ken 1 -2 3e-1") == {"name": "ken", "values": [1.0, -2.0, 0.3]}
    assert tagcontainer.write(tag, {"name": "ken", "values": [1.0, -2.0]}) == "barbie ken 1.0 -2.0 "
    with pytest.raises(ValueError, match="The subtag 'values' for tag 'barbie' is not optional"):
        tagcontainer.read(tag, "ken")
    with pytest.raises(ValueError, match=re.escape("were not processed: ['allan']")):
        tagcontainer.read(tag, "ken 1 allan")
    optional_tagcontainer = TagContainer(
        subtags={
            "name": StrTag(write_tagname=False, optional=False),
            "values": FloatTag(write_tagname=False, can_repeat=True, optional=True),
        }
    )
    assert optional_tagcontainer.read(tag, "ken") == {"name": "ken"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Mn 3", [("Mn", [3.0])]),
        ("Mn 3 -3 3", [("Mn", [3.0, -3.0, 3.0])]),
        ("Mn 3 -3 O 0 Fe 1.5", [("Mn", [3.0, -3.0]), ("O", [0.0]), ("Fe", [1.5])]),
        ("Mn 3.0 0 0 3 180 90", [("Mn", [3.0, 0.0, 0.0, 3.0, 180.0, 90.0])]),
        ("Mn +4 O -2", [("Mn", [4.0]), ("O", [-2.0])]),
        ("Mn 1e-1 -2.5E+0", [("Mn", [0.1, -2.5])]),
        ("Fe_1 2 Fe_2 -2", [("Fe_1", [2.0]), ("Fe_2", [-2.0])]),
        ("Mn   3  -3\t3", [("Mn", [3.0, -3.0, 3.0])]),
    ],
)
@pytest.mark.parametrize("tag", ["initial-magnetic-moments", "initial-oxidation-state"])
def test_species_values_tags_read_write_roundtrip(tag: str, value: str, expected: list[tuple[str, list[float]]]):
    tag_object = get_tag_object(tag)
    first_read = tag_object.read(tag, value)
    assert first_read == {"species-values": [{"species-id": sp, "values": vals} for sp, vals in expected]}
    line = tag_object.write(tag, first_read)
    written_tag, written_value = line.split(maxsplit=1)
    assert written_tag == tag
    second_read = tag_object.read(written_tag, written_value.strip())
    assert second_read == first_read
    assert tag_object.is_equal_to(first_read, tag_object, second_read)
    # Writing again must be stable once the value has been through one round trip
    assert tag_object.write(tag, second_read) == line


@pytest.mark.parametrize(
    ("value", "error_match"),
    [
        ("Mn", "The subtag 'species-values' for tag 'initial-magnetic-moments' is not optional"),
        ("3 Mn", "The subtag 'species-values' for tag 'initial-magnetic-moments' is not optional"),
        ("Mn 3 O", re.escape("were not processed: ['O']")),
    ],
)
def test_species_values_tag_read_errors(value: str, error_match: str):
    tag = "initial-magnetic-moments"
    with pytest.raises(ValueError, match=error_match):
        get_tag_object(tag).read(tag, value)


def test_empty_means_unset_keyword_subtag():
    # JDFTx echoes an unset file name as the bare keyword, e.g. "cavityFile " in pcm-params
    tagcontainer = TagContainer(
        subtags={"file": StrTag(empty_means_unset=True), "ken": FloatTag(), "allan": FloatTag()}
    )
    assert tagcontainer.read("barbie", "ken 1 file allan 2") == {"ken": 1.0, "allan": 2.0}
    assert tagcontainer.read("barbie", "ken 1 file") == {"ken": 1.0}
    assert tagcontainer.read("barbie", "file f.bin ken 1") == {"file": "f.bin", "ken": 1.0}
    assert tagcontainer.read_from_tokens("barbie", ["file", "ken", "1", "next"]) == ({"ken": 1.0}, 3)
    assert tagcontainer.read_from_tokens("barbie", ["file", "f.bin", "next"]) == ({"file": "f.bin"}, 2)
    # Without it, the next keyword is taken as the value
    with pytest.raises(ValueError, match=re.escape("were not processed: ['1']")):
        TagContainer(subtags={"file": StrTag(), "ken": FloatTag()}).read("barbie", "file ken 1")


def test_tag_container_write_leaves_out_unwritten_subtags():
    # An out of bounds value writes nothing, and leaves no empty line behind
    tagcontainer = TagContainer(
        linebreak_nth_entry=1, subtags={"ken": FloatTag(lb=0.0, lb_incl=False), "allan": FloatTag()}
    )
    assert tagcontainer.write("barbie", {"ken": 0.0, "allan": 1.0}) == "barbie \\\n    allan  1.0 "
    assert tagcontainer.write("barbie", {"ken": 2.0, "allan": 1.0}) == "barbie \\\n    ken  2.0 \\\n    allan  1.0 "


def test_tag_container_subtag_aliases():
    # A name JDFTx's echo uses for a subtag is read as that subtag, and written under the subtag's own name
    tagcontainer = TagContainer(subtags={"ken": StrTag(), "allan": FloatTag()}, subtag_aliases={"kenneth": "ken"})
    assert tagcontainer.read("barbie", "kenneth x allan 1") == {"ken": "x", "allan": 1.0}
    assert tagcontainer.read_from_tokens("barbie", ["kenneth", "x", "next"]) == ({"ken": "x"}, 2)
    assert tagcontainer.write("barbie", {"ken": "x"}) == "barbie ken  x "


def test_tag_container_write_follows_subtag_order():
    tagcontainer = TagContainer(
        subtags={"ken": FloatTag(write_tagname=False), "allan": FloatTag(write_tagname=False), "barbie": FloatTag()}
    )
    assert tagcontainer.write("tag", {"barbie": 3.0, "allan": 2.0, "ken": 1.0}) == "tag 1.0 2.0 barbie  3.0 "
    with pytest.raises(KeyError, match="midge"):
        tagcontainer.write("tag", {"midge": 1.0})


def test_read_from_tokens_requires_a_token():
    with pytest.raises(ValueError, match="No value given for 'tag'"):
        StrTag().read_from_tokens("tag", [])


def _keyword_group_tagcontainer() -> TagContainer:
    """Positional repeating group with a repeatable and a non-repeatable keyword subtag."""
    return TagContainer(
        subtags={
            "group": TagContainer(
                write_tagname=False,
                can_repeat=True,
                optional=False,
                subtags={
                    "name": StrTag(write_tagname=False, optional=False),
                    "ken": FloatTag(write_tagname=True, can_repeat=True),
                    "allan": IntTag(write_tagname=True),
                },
            ),
        }
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a", [{"name": "a"}]),
        ("a ken 1 ken 2 allan 3 b allan 4", [{"name": "a", "ken": [1.0, 2.0], "allan": 3}, {"name": "b", "allan": 4}]),
        ("a b ken 1", [{"name": "a"}, {"name": "b", "ken": [1.0]}]),
    ],
)
def test_positional_tagcontainer_reads_keyword_subtags_in_place(value: str, expected: list[dict]):
    tag = "barbie"
    tagcontainer = _keyword_group_tagcontainer()
    read_value = tagcontainer.read(tag, value)
    assert read_value == {"group": expected}
    written_value = tagcontainer.write(tag, read_value).split(maxsplit=1)[1]
    assert tagcontainer.read(tag, written_value) == read_value


@pytest.mark.parametrize(
    ("value", "error_match"),
    [
        # A keyword can't start a group, since positional subtags never read keyword tokens
        ("ken 1", "The subtag 'group' for tag 'barbie' is not optional"),
        # A non-repeatable keyword is only read once per group
        ("a allan 3 allan 4", re.escape("were not processed: ['allan', '4']")),
        # A keyword must come at its place in the subtag order (allan after ken), so "ken 1" is left unread
        ("a allan 3 ken 1", re.escape("were not processed: ['ken', '1']")),
        # A keyword missing its value invalidates the whole group
        ("a ken", "The subtag 'group' for tag 'barbie' is not optional"),
    ],
)
def test_positional_tagcontainer_keyword_subtag_errors(value: str, error_match: str):
    with pytest.raises(ValueError, match=error_match):
        _keyword_group_tagcontainer().read("barbie", value)


def test_ordered_flag():
    unordered = FloatTag(can_repeat=True)
    ordered = FloatTag(can_repeat=True, ordered=True)
    assert unordered.is_equal_to([1.0, 2.0], unordered, [2.0, 1.0])
    assert not ordered.is_equal_to([1.0, 2.0], ordered, [2.0, 1.0])
    assert ordered.is_equal_to([1.0, 2.0], ordered, [1.0, 2.0 + 1e-10])
    assert not ordered.is_equal_to([1.0, 2.0], ordered, [1.0])
    # Species groups may come in any order, but the values within a group are per-ion and must keep their order
    tag = "initial-magnetic-moments"
    tag_object = get_tag_object(tag)
    value = tag_object.read(tag, "Mn 3 -3 O 0")
    assert tag_object.is_equal_to(value, tag_object, tag_object.read(tag, "O 0 Mn 3 -3"))
    assert not tag_object.is_equal_to(value, tag_object, tag_object.read(tag, "Mn -3 3 O 0"))


def test_type_fix_failure_warning_message():
    """A value whose type can't be fixed warns with the real message (once the literal "warning"), unless silenced."""
    message = (
        "Could not fix the typing for tag 'tag' expecting type <class 'int'> of value abc with type <class 'str'>!"
    )
    with pytest.warns(Warning, match=re.escape(message)):
        IntTag().validate_value_type("tag", "abc", try_auto_type_fix=True)
    message = "Could not fix the typing for tag 'tag' expecting type <class 'str'> of value (unstringable value)!"
    with pytest.warns(Warning, match=re.escape(message)):
        StrTag().validate_value_type("tag", Unstringable(), try_auto_type_fix=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IntTag().validate_value_type("tag", "abc", try_auto_type_fix=True, warn_user=False)


def test_tagcontainer_validation():
    tag = "barbie"
    repeatable_str_subtag = "ken"
    non_repeatable_int_subtag = "allan"
    non_intable_value = "nonintable"
    tagcontainer = TagContainer(
        can_repeat=False,
        subtags={
            f"{repeatable_str_subtag}": StrTag(can_repeat=True),
            f"{non_repeatable_int_subtag}": IntTag(can_repeat=False),
        },
    )
    # issues with converting values to the correct type should only raise a warning within validate_value_type
    with pytest.warns(Warning, match="Invalid value"):
        tagcontainer.validate_value_type(
            f"{tag}", {f"{repeatable_str_subtag}": ["1"], f"{non_repeatable_int_subtag}": f"{non_intable_value}"}
        )
    # _validate_single_entry for TagContainer should raise an error if the value is not a dict
    with pytest.raises(TypeError):
        tagcontainer._validate_single_entry("not a dict")  # Not a dict
    tagcontainer.can_repeat = True
    # repeated entries in dict representation may differ in which optional subtags they include
    _, is_valid, _ = tagcontainer.validate_value_type(tag, [{"ken": ["1"]}, {"ken": ["2"], "allan": 3}])
    assert is_valid
    # inhomogeneous values for repeated TagContainers in list representation should raise an error
    # until filling with default values is implemented
    value = [["ken", "1"], ["ken", "2", "allan", "3"]]
    with pytest.raises(
        ValueError,
        match=re.escape(f"The values '{value}' for tag '{tag}' provided in a list of lists have different lengths"),
    ):
        tagcontainer.validate_value_type(tag, value)


def test_tagcontainer_mixed_nesting():
    tag = "barbie"
    str_subtag = "ken"
    int_subtag = "allan"
    tagcontainer = TagContainer(
        can_repeat=False,
        subtags={
            f"{str_subtag}": StrTag(),
            f"{int_subtag}": IntTag(),
        },
    )
    tagcontainer_mixed_nesting_tester(tagcontainer, tag, str_subtag, int_subtag)


def tagcontainer_mixed_nesting_tester(tagcontainer, tag, str_subtag, int_subtag):
    list_of_dicts_and_lists = [{str_subtag: "b"}, [[int_subtag, 1]]]
    list_of_dicts_and_lists_err = f"tag '{tag}' with value '{list_of_dicts_and_lists}' cannot have"
    " nested lists/dicts mixed with bool/str/int/floats!"
    list_of_dicts_of_strs_and_ints = [{str_subtag: "b"}, {int_subtag: 1}]
    list_of_dicts_of_strs_and_ints_err = f"tag '{tag}' with value '{list_of_dicts_of_strs_and_ints}' "
    "cannot have nested dicts mixed with bool/str/int/floats!"
    list_of_lists_of_strs_and_ints = [[str_subtag, "b"], [int_subtag, 1]]
    list_of_lists_of_strs_and_ints_err = f"tag '{tag}' with value '{list_of_lists_of_strs_and_ints}' "
    "cannot have nested lists mixed with bool/str/int/floats!"
    for value, err_str in zip(
        [list_of_dicts_and_lists, list_of_dicts_of_strs_and_ints, list_of_lists_of_strs_and_ints],
        [list_of_dicts_and_lists_err, list_of_dicts_of_strs_and_ints_err, list_of_lists_of_strs_and_ints_err],
        strict=False,
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(err_str),
        ):
            tagcontainer._check_for_mixed_nesting(tag, value)


def test_tagcontainer_read():
    tag = "barbie"
    repeatable_str_subtag = "ken"
    non_repeatable_int_subtag = "allan"
    tagcontainer = TagContainer(
        can_repeat=False,
        subtags={
            f"{repeatable_str_subtag}": StrTag(can_repeat=True),
            f"{non_repeatable_int_subtag}": IntTag(can_repeat=False),
        },
    )
    # non-repeatable subtags that repeat in the value for "read" should raise an error
    value = f"{non_repeatable_int_subtag} 1 {non_repeatable_int_subtag} 2"
    with pytest.raises(
        ValueError,
        match=f"Subtag '{non_repeatable_int_subtag}' for tag '{tag}' is not allowed to repeat "
        f"but repeats value {value}",
    ):
        tagcontainer.read(tag, value)
    # output of "read" should be a dict of subtags, with list values for repeatable subtags and single values for
    # non-repeatable subtags
    assert_same_value(
        tagcontainer.read(f"{tag}", "ken a ken b allan 3"),
        {"ken": ["a", "b"], "allan": 3},
    )
    required_subtag = "ken"
    optional_subtag = "allan"
    tagcontainer = TagContainer(
        can_repeat=True,
        write_tagname=True,
        subtags={
            required_subtag: StrTag(optional=False, write_tagname=True, write_value=True),
            optional_subtag: IntTag(optional=True, write_tagname=True, write_value=True),
        },
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The subtag '{required_subtag}' for tag '{tag}' is not optional but was not populated during the read!"
        ),
    ):
        tagcontainer.read("barbie", "allan 1")
    unread_values = "fgfgfgf"
    value = f"ken a {unread_values}"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Something is wrong in the JDFTXInfile formatting, the following values for tag '{tag}' "
            f"were not processed: {[unread_values]}"
        ),
    ):
        tagcontainer.read(tag, value)


def test_tagcontainer_write():
    tag = "barbie"
    repeatable_str_subtag = "ken"
    non_repeatable_int_subtag = "allan"
    tagcontainer = TagContainer(
        can_repeat=False,
        subtags={
            f"{repeatable_str_subtag}": StrTag(can_repeat=True),
            f"{non_repeatable_int_subtag}": IntTag(can_repeat=False),
        },
    )
    assert_same_value(
        tagcontainer.write(tag, {repeatable_str_subtag: ["a", "b"]}),
        f"{tag} {repeatable_str_subtag} a {repeatable_str_subtag} b ",
    )
    tagcontainer.subtags[repeatable_str_subtag].write_tagname = False
    assert_same_value(tagcontainer.write(tag, {repeatable_str_subtag: ["a", "b"]}), f"{tag} a b ")
    # Lists are reserved for repeatable tagcontainers
    value = [{"ken": 1}]
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"The value '{value}' (of type {type(value)}) for tag '{tag}' must be a dict for this TagContainer!"
        ),
    ):
        tagcontainer.write(tag, value)


def test_tagcontainer_list_dict_conversion():
    top_subtag = "universe"
    bottom_subtag1 = "sun"
    bottom_subtag2 = "moon"
    tagcontainer = TagContainer(
        can_repeat=True,
        allow_list_representation=False,
        subtags={
            top_subtag: TagContainer(
                allow_list_representation=True,
                write_tagname=True,
                subtags={
                    bottom_subtag1: BoolTag(
                        write_tagname=False,
                        allow_list_representation=False,
                    ),
                    bottom_subtag2: BoolTag(
                        write_tagname=True,
                        allow_list_representation=False,
                    ),
                },
            )
        },
    )
    notadict = f"True {bottom_subtag2} False"
    value = {top_subtag: notadict}
    with pytest.raises(
        ValueError, match=re.escape(f"The subtag {top_subtag} is not a dict: '{notadict}', so could not be converted")
    ):
        tagcontainer._make_list(value)
    # Despite bottom_subtag2 have "write_tagname" set to True, it is not written in the list for _make_list
    # (obviously this is dangerous and this method should be avoided)
    assert_same_value(
        tagcontainer._make_list({top_subtag: {bottom_subtag1: "True", bottom_subtag2: "False"}}),
        [top_subtag, "True", "False"],
    )
    value = {top_subtag: {bottom_subtag1: {"True": True}, bottom_subtag2: "False"}}
    with pytest.warns(Warning, match="sun subtag does not allow list"):
        tagcontainer._make_list(value)
    value = [[top_subtag, "True", "False"]]
    assert_same_value(tagcontainer.get_list_representation("barbie", value), value)
    tag = "barbie"
    value = [["universe", "True", "False"], {"universe": {"sun": True, "moon": False}}]
    err_str = f"The tag '{tag}' set to value '{value}' must be a list of dicts when passed to "
    "'get_list_representation' since the tag is repeatable."
    with pytest.raises(ValueError, match=re.escape(err_str)):
        tagcontainer.get_list_representation(tag, value)
    value = {"universe": {"sun": {"True": True}, "moon": "False"}}
    err_str = f"Value '{value}' must be a list when passed to 'get_dict_representation' since "
    f"tag '{tag}' is repeatable."
    with pytest.raises(ValueError, match=re.escape(err_str)):
        tagcontainer.get_dict_representation("barbie", value)
    err_str = f"Value '{value}' must be a list when passed to 'get_list_representation' since "
    f"tag '{tag}' is repeatable."
    with pytest.raises(ValueError, match=re.escape(err_str)):
        tagcontainer.get_list_representation("barbie", value)


def test_dumptagcontainer():
    dtc = get_dump_tag_container()
    tag = "dump"
    unread_value = "barbie"
    value = f"{unread_value} End DOS"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Something is wrong in the JDFTXInfile formatting, the following values for tag '{tag}' "
            f"were not processed: {[unread_value]}"
        ),
    ):
        dtc.read(tag, value)


def test_booltagcontainer():
    tag = "barbie"
    required_subtag = "ken"
    optional_subtag = "allan"
    not_a_subtag = "alan"
    btc = BoolTagContainer(
        subtags={
            required_subtag: BoolTag(optional=False, write_tagname=True, write_value=False),
            optional_subtag: BoolTag(optional=True, write_tagname=True, write_value=False),
        },
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The subtag '{required_subtag}' for tag '{tag}' is not optional but was not populated during the read!"
        ),
    ):
        btc.read(tag, optional_subtag)
    value = f"{required_subtag} {not_a_subtag}"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Something is wrong in the JDFTXInfile formatting, the following values for tag '{tag}' "
            f"were not processed: {[not_a_subtag]}"
        ),
    ):
        btc.read(tag, value)


def test_subcommandtagcontainer():
    stc = SubcommandTagContainer(
        subtags={
            "ken": BoolTag(write_value=False),
            "allan": FloatTag(lb=0.0),
            "alan": TagContainer(subtags={"a": IntTag(write_tagname=False, optional=False)}),
        }
    )
    value = [{"allan": 1.0}, {"ken": True}, {"alan": {"a": 1}}, {"ken": True}, {"alan": {"a": 2}}]
    assert stc.read("barbie", "allan 1 ken alan 1 ken alan 2") == value
    assert stc.write("barbie", value).split() == "barbie \\ allan 1.0 \\ ken \\ alan 1 \\ ken \\ alan 2".split()
    # Types are fixed per subcommand, and a dict of subcommands becomes entries in the dict's order
    assert stc.validate_value_type("barbie", [{"allan": "1"}, {"alan": ["2"]}], try_auto_type_fix=True)[1:] == (
        True,
        [{"allan": 1.0}, {"alan": {"a": 2}}],
    )
    assert stc.validate_value_type("barbie", {"ken": True, "alan": [{"a": 1}, {"a": 2}]})[2] == [
        {"ken": True},
        {"alan": {"a": 1}},
        {"alan": {"a": 2}},
    ]
    assert stc.validate_value_bounds("barbie", value) == (True, "barbie: ")
    is_valid, err = stc.validate_value_bounds("barbie", [{"allan": -1.0}])
    assert not is_valid
    assert "not within bounds" in err
    # Order matters for comparison
    assert stc.is_equal_to(value, stc, [{"allan": 1.0 + 1e-9}, *value[1:]])
    assert not stc.is_equal_to(value, stc, [value[0], value[2], value[1], *value[3:]])
    assert not stc.is_equal_to(value, stc, value[:-1])
    assert not stc.is_equal_to(value, TagContainer(subtags=stc.subtags), value)
    with pytest.raises(TypeError, match="must be a list of single-key dicts"):
        stc.write("barbie", [{"ken": True, "allan": 1.0}])
    with pytest.raises(ValueError, match="'barbie' is not a valid subcommand of 'ken'!"):
        stc.validate_value_type("ken", [{"barbie": True}])


def test_multiformattagcontainer():
    tag = "dump-fermi-density"
    value = "notabool"
    mftg = get_tag_object(tag)
    with pytest.raises(RuntimeError):
        mftg.read(tag, value)
    with pytest.raises(RuntimeError):
        mftg.write(tag, value)
    errormsg = f"No valid read format for tag '{tag}' with value '{value}'\n"
    "Add option to format_options or double-check the value string and retry!\n\n"
    with pytest.raises(ValueError, match=re.escape(errormsg)):
        mftg.get_format_index_for_str_value(tag, value)
    err_str = f"The format for tag '{tag}' with value '{value}' could not be determined from the available options! "
    "Check your inputs and/or MASTER_TAG_LIST!"
    with pytest.raises(ValueError, match=re.escape(err_str)):
        mftg._determine_format_option(tag, value)


def test_boundary_checking():
    # Check that non-numeric tag returns valid always
    tag = "barbie"
    value = "notanumber"
    valtag = StrTag()
    assert valtag.validate_value_bounds(tag, value)[0] is True
    # Check that numeric tags can return False
    value = 0.0
    valtag = FloatTag(lb=1.0)
    assert valtag.validate_value_bounds(tag, value)[0] is False
    valtag = FloatTag(lb=0.0, lb_incl=False)
    assert valtag.validate_value_bounds(tag, value)[0] is False
    valtag = FloatTag(ub=-1.0)
    assert valtag.validate_value_bounds(tag, value)[0] is False
    valtag = FloatTag(ub=0.0, ub_incl=False)
    assert valtag.validate_value_bounds(tag, value)[0] is False
    # Check that numeric tags can return True
    valtag = FloatTag(lb=0.0, lb_incl=True)
    assert valtag.validate_value_bounds(tag, value)[0] is True
    valtag = FloatTag(ub=0.0, ub_incl=True)
    assert valtag.validate_value_bounds(tag, value)[0] is True
    valtag = FloatTag(lb=-1.0, ub=1.0)
    assert valtag.validate_value_bounds(tag, value)[0] is True
    # Check functionality for tagcontainers
    tagcontainer = TagContainer(
        subtags={
            "ken": FloatTag(lb=0.0, lb_incl=True),
            "allan": StrTag(),
            "skipper": FloatTag(ub=1.0, ub_incl=True, lb=-1.0, lb_incl=False),
        },
    )
    valid, errors = tagcontainer.validate_value_bounds(tag, {"ken": -1.0, "allan": "notanumber", "skipper": 2.0})
    assert valid is False
    assert "allan" not in errors
    assert "ken" in errors
    assert "x >= 0.0" in errors
    assert "skipper" in errors
    assert "1.0 >= x > -1.0" in errors
    # Make sure tags will never write a value that is out of bounds
    valtag = FloatTag(lb=-1.0, ub=1.0)
    assert len(valtag.write(tag, 0.0))
    assert not len(valtag.write(tag, 2.0))


def test_tag_is_equal_to():
    strtag1 = StrTag()
    strtag2 = StrTag()
    floattag1 = FloatTag()
    floattag2 = FloatTag()
    inttag1 = IntTag()
    inttag2 = IntTag()
    booltag1 = BoolTag()
    booltag2 = BoolTag()
    tagcont1 = TagContainer(subtags={"strtag": strtag1, "floattag": floattag1, "inttag": inttag1, "booltag": booltag1})
    tagcont2 = TagContainer(subtags={"strtag": strtag2, "floattag": floattag2, "inttag": inttag2, "booltag": booltag2})
    # Make sure tags of different types are not equal
    assert not strtag1.is_equal_to("", floattag1, 1.0)
    assert not strtag1.is_equal_to("", inttag1, 1)
    assert not strtag1.is_equal_to("", booltag1, True)  # noqa: FBT003
    assert not strtag1.is_equal_to("", tagcont1, {"strtag": "a", "floattag": 1.0, "inttag": 1, "booltag": True})
    # Test some str tag equalities
    assert strtag1.is_equal_to(" a", strtag2, "a")
    assert not strtag1.is_equal_to(" a", strtag2, "b")
    # Test some float tag equalities
    assert floattag1.is_equal_to(1.0, floattag2, 1.0)
    assert not floattag1.is_equal_to(1.0, floattag2, 2.0)
    floattag3 = FloatTag(eq_atol=0.1)
    assert floattag3.is_equal_to(1.0, floattag2, 1.01)
    # Test some int tag equalities
    assert inttag1.is_equal_to(1, inttag2, 1)
    assert not inttag1.is_equal_to(1, inttag2, 2)
    # Test some bool tag equalities
    assert booltag1.is_equal_to(True, booltag2, True)  # noqa: FBT003
    assert not booltag1.is_equal_to(True, booltag2, False)  # noqa: FBT003
    # Test some tagcontainer equalities
    assert tagcont1.is_equal_to(
        {"strtag": "a", "floattag": 1.0, "inttag": 1, "booltag": True},
        tagcont2,
        {"strtag": "a", "floattag": 1.0, "inttag": 1, "booltag": True},
    )
    assert not tagcont1.is_equal_to(
        {"strtag": "a", "floattag": 1.0, "inttag": 1, "booltag": True},
        tagcont2,
        {"strtag": "b", "floattag": 1.0, "inttag": 1, "booltag": True},
    )
    assert not tagcont1.is_equal_to(
        {"strtag": "a", "inttag": 1, "booltag": True},
        tagcont2,
        {"strtag": "a", "floattag": 2.0, "inttag": 1, "booltag": True},
    )
    # Test expected error raising
    with pytest.raises(ValueError, match="Values must be in dictionary format for TagContainer comparison"):
        tagcont1.is_equal_to(
            {"strtag": "a", "floattag": 1.0, "inttag": 1, "booltag": True},
            tagcont2,
            ["strtag", "a", "floattag", 1.0, "inttag", 1, "booltag", True],
        )
    with pytest.raises(ValueError, match="Both values must be strings for StrTag comparison"):
        strtag1.is_equal_to(" a", strtag2, 1.0)
    mfgtag = MultiformatTag(
        format_options=[FloatTag(), IntTag()],
    )
    with pytest.raises(NotImplementedError):
        mfgtag.is_equal_to(" a", strtag2, 1.0)
