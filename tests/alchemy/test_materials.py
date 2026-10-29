from __future__ import annotations

from copy import deepcopy

import orjson
import pytest

from pymatgen.alchemy.filters import ContainsSpecieFilter
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.core import SETTINGS
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.transformations.standard_transformations import (
    PartialRemoveSpecieTransformation,
    SubstitutionTransformation,
    SupercellTransformation,
)
from pymatgen.util.provenance import StructureNL
from pymatgen.util.testing import FAKE_POTCAR_DIR, TEST_FILES_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/alchemy"


class TestTransformedStructure(MatSciTest):
    def setup_method(self):
        structure = MatSciTest.get_structure("LiFePO4")
        self.structure = structure
        trafos = [SubstitutionTransformation({"Li": "Na"})]
        self.trans = TransformedStructure(structure, trafos)

    def test_append_transformation(self):
        trafo = SubstitutionTransformation({"Fe": "Mn"})
        self.trans.append_transformation(trafo)
        assert self.trans.final_structure.reduced_formula == "NaMnPO4"
        assert len(self.trans.structures) == 3
        coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        lattice = [
            [3.8401979337, 0, 0],
            [1.9200989668, 3.3257101909, 0],
            [0, -2.2171384943, 3.1355090603],
        ]
        struct = Structure(lattice, ["Si4+", "Si4+"], coords)
        t_struct = TransformedStructure(struct, [])
        t_struct.append_transformation(SupercellTransformation.from_scaling_factors(2, 1, 1))
        alt = t_struct.append_transformation(
            PartialRemoveSpecieTransformation("Si4+", 0.5, algo=PartialRemoveSpecieTransformation.ALGO_COMPLETE),
            5,
        )
        assert len(alt) == 2

    def test_append_transformation_alternatives_history(self):
        struct = Structure(
            [[3.84, 0, 0], [1.92, 3.33, 0], [0, -2.22, 3.14]], ["Si4+", "Si4+"], [[0, 0, 0], [0.75, 0.5, 0.75]]
        )
        struct.make_supercell([2, 1, 1])
        t_struct = TransformedStructure(struct, [])
        trafo = PartialRemoveSpecieTransformation("Si4+", 0.5, algo=PartialRemoveSpecieTransformation.ALGO_COMPLETE)
        alts = t_struct.append_transformation(trafo, 5)
        assert len(alts) == 2
        # the recorded input structure used to be the last alternative instead of the actual input
        assert Structure.from_dict(t_struct.history[-1]["input_structure"]) == struct
        assert t_struct.structures[0] == struct
        for alt in alts:
            assert Structure.from_dict(alt.history[-1]["input_structure"]) == struct
        t_struct.undo_last_change()
        assert t_struct.final_structure == struct

    def test_append_transformation_does_not_mutate_ranked_list(self):
        struct = Structure(
            [[3.84, 0, 0], [1.92, 3.33, 0], [0, -2.22, 3.14]], ["Si4+", "Si4+"], [[0, 0, 0], [0.75, 0.5, 0.75]]
        )
        struct.make_supercell([2, 1, 1])
        ranked_list = [
            {"structure": struct.copy(), "energy": 0.0},
            {"structure": struct.copy(), "energy": 1.0},
        ]
        trafo = PartialRemoveSpecieTransformation("Si4+", 0.5)
        trafo.apply_transformation = lambda *args, **kwargs: ranked_list
        TransformedStructure(struct, []).append_transformation(trafo, 2)
        # popping "structure" used to mutate the dicts returned (and possibly cached) by the transformation
        assert all("structure" in dct for dct in ranked_list)

    def test_append_filter(self):
        f3 = ContainsSpecieFilter(["O2-"], strict_compare=True, AND=False)
        self.trans.append_filter(f3)

    def test_get_vasp_input(self):
        SETTINGS["PMG_VASP_PSP_DIR"] = FAKE_POTCAR_DIR
        potcar = self.trans.get_vasp_input(MPRelaxSet)["POTCAR"]
        assert "\n".join(p.symbol for p in potcar) == "Na_pv\nFe_pv\nP\nO"
        assert len(self.trans.structures) == 2

    def test_final_structure(self):
        assert self.trans.final_structure.reduced_formula == "NaFePO4"
        # https://github.com/materialsproject/pymatgen/pull/3617
        assert isinstance(deepcopy(self.trans), TransformedStructure)

    def test_from_dict(self):
        with open(f"{TEST_DIR}/transformations.json", "rb") as file:
            dct = orjson.loads(file.read())
        dct["other_parameters"] = {"tags": ["test"]}
        t_struct = TransformedStructure.from_dict(dct)
        t_struct.other_parameters["author"] = "Will"
        t_struct.append_transformation(SubstitutionTransformation({"Fe": "Mn"}))
        assert t_struct.final_structure.reduced_formula == "MnPO4"
        assert t_struct.other_parameters == {"author": "Will", "tags": ["test"]}

    def test_undo_and_redo_last_change(self):
        trafos = [
            SubstitutionTransformation({"Li": "Na"}),
            SubstitutionTransformation({"Fe": "Mn"}),
        ]
        t_struct = TransformedStructure(self.structure, trafos)
        assert t_struct.final_structure.reduced_formula == "NaMnPO4"
        t_struct.undo_last_change()
        assert t_struct.final_structure.reduced_formula == "NaFePO4"
        t_struct.undo_last_change()
        assert t_struct.final_structure.reduced_formula == "LiFePO4"
        with pytest.raises(IndexError, match="No more changes to undo"):
            t_struct.undo_last_change()
        t_struct.redo_next_change()
        assert t_struct.final_structure.reduced_formula == "NaFePO4"
        t_struct.redo_next_change()
        assert t_struct.final_structure.reduced_formula == "NaMnPO4"
        with pytest.raises(IndexError, match="No more changes to redo"):
            t_struct.redo_next_change()
        # Make sure that this works with filters.
        f3 = ContainsSpecieFilter(["O2-"], strict_compare=True, AND=False)
        t_struct.append_filter(f3)
        t_struct.undo_last_change()
        t_struct.redo_next_change()

    def test_set_parameter(self):
        trans = self.trans.set_parameter("author", "will")
        assert trans.other_parameters["author"] == "will"
        assert trans is self.trans

    def test_as_dict(self):
        self.trans.set_parameter("author", "will")
        dct = self.trans.as_dict()
        assert "last_modified" in dct
        assert "history" in dct
        assert "author" in dct["other_parameters"]
        assert Structure.from_dict(dct).formula == "Na4 Fe4 P4 O16"

    def test_snl(self):
        self.trans.set_parameter("author", "will")
        with pytest.warns(UserWarning, match="discarded during type conversion to SNL") as warns:
            struct_nl = self.trans.to_snl([("will", "will@test.com")])

        assert len(warns) >= 1, f"Warning not raised on type conversion with other_parameters {len(warns)=}"
        assert (
            str(warns[0].message)
            == "Data in TransformedStructure.other_parameters discarded during type conversion to SNL"
        )

        t_struct = TransformedStructure.from_snl(struct_nl)
        assert t_struct.history[-1]["@class"] == "SubstitutionTransformation"

        hist = ("testname", "testURL", {"test": "testing"})
        struct_nl = StructureNL(t_struct.final_structure, [("will", "will@test.com")], history=[hist])
        t_struct = TransformedStructure.from_snl(struct_nl).to_snl([("notwill", "notwill@test.com")])
        assert t_struct.history == [hist]
        assert t_struct.authors == [("notwill", "notwill@test.com")]

    def test_init_single_transformation(self):
        # A single AbstractTransformation passed as ``transformations`` is unwrapped to a list.
        ts = TransformedStructure(self.structure, SubstitutionTransformation({"Li": "Na"}))
        assert ts.final_structure.reduced_formula == "NaFePO4"
        assert len(ts.history) == 1

    def test_undo_last_change_no_input_structure(self):
        # ``undo_last_change`` raises IndexError if the last history entry has no input_structure.
        t = TransformedStructure(self.structure)
        t.history.append({"@class": "SomeFilter", "output_parameters": {}})
        with pytest.raises(IndexError, match="has no input_structure"):
            t.undo_last_change()

    def test_redo_next_change(self):
        t = TransformedStructure(self.structure, [SubstitutionTransformation({"Li": "Na"})])
        t.undo_last_change()
        t.redo_next_change()
        assert t.final_structure.reduced_formula == "NaFePO4"

    def test_extend_transformations(self):
        t = TransformedStructure(self.structure)
        t.extend_transformations(
            [
                SubstitutionTransformation({"Li": "Na"}),
                SubstitutionTransformation({"Fe": "Mn"}),
            ]
        )
        assert t.final_structure.reduced_formula == "NaMnPO4"
        assert len(t.history) == 2

    def test_str(self):
        ts = TransformedStructure(self.structure, [SubstitutionTransformation({"Li": "Na"})])
        s = str(ts)
        assert "Current structure" in s
        assert "NaFePO4" in s
        # str() used to pop input_structure from the history, breaking undo and structures
        assert "input_structure" in ts.history[-1]
        assert len(ts.structures) == 2
        ts.undo_last_change()
        assert ts.final_structure.reduced_formula == "LiFePO4"

    def test_snl_does_not_mutate_history(self):
        hist = ("testname", "testURL", {"test": "testing"})
        struct_nl = StructureNL(self.structure, [("will", "will@test.com")], history=[hist])
        t_struct = TransformedStructure.from_snl(struct_nl)
        # from_snl used to add "_snl" to the SNL's own history descriptions
        assert struct_nl.history[0].description == {"test": "testing"}
        # to_snl used to pop "_snl" from the TransformedStructure history
        t_struct.to_snl([("notwill", "notwill@test.com")])
        assert t_struct.history[0]["_snl"] == {"url": "testURL", "name": "testname"}
        assert t_struct.to_snl([("notwill", "notwill@test.com")]).history == [hist]

    def test_was_modified(self):
        ts = TransformedStructure(self.structure, [SubstitutionTransformation({"Li": "Na"})])
        assert ts.was_modified is True
        # A transform whose target isn't present must not change the structure.
        ts2 = TransformedStructure(
            self.structure,
            [SubstitutionTransformation({"Au": "Pt"})],
        )
        assert ts2.was_modified is False

    def test_write_vasp_input(self, tmp_path):
        SETTINGS["PMG_VASP_PSP_DIR"] = FAKE_POTCAR_DIR

        ts = TransformedStructure(self.structure, [SubstitutionTransformation({"Li": "Na"})])
        outdir = tmp_path / "vasp"
        ts.write_vasp_input(MPRelaxSet, output_dir=str(outdir))
        # VASP input files should be present
        inputs = outdir / "INCAR"
        assert inputs.exists()

    def test_from_cif_str_uploaded(self, tmp_path):
        # ``from_cif_str`` sets source to "uploaded cif" when the CIF has no ``_database_code_ICSD``.
        struct = MatSciTest.get_structure("LiFePO4")
        from pymatgen.io.cif import CifWriter

        fpath = tmp_path / "test.cif"
        CifWriter(struct).write_file(str(fpath))
        cif = fpath.read_text()
        t = TransformedStructure.from_cif_str(cif)
        assert t.history[-1]["source"] == "uploaded cif"
        assert t.final_structure.formula == struct.formula

    @pytest.mark.filterwarnings("ignore::pymatgen.io.vasp.inputs.BadPoscarWarning")
    def test_from_poscar_str_vasp4_raises(self):
        # ``from_poscar_str`` rejects VASP4 POSCARs (no element symbols).
        vasp4 = "comment\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 3.0\n1\nDirect\n0.0 0.0 0.0"
        with pytest.raises(ValueError, match="proper VASP5"):
            TransformedStructure.from_poscar_str(vasp4)
