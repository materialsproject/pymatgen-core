from __future__ import annotations

import pytest

from pymatgen.alchemy.filters import ContainsSpecieFilter
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.alchemy.transmuters import (
    CifTransmuter,
    PoscarTransmuter,
    StandardTransmuter,
    _apply_transformation,
    batch_write_vasp_input,
)
from pymatgen.core import SETTINGS
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.transformations.advanced_transformations import SuperTransformation
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation,
    RemoveSpeciesTransformation,
    SubstitutionTransformation,
)
from pymatgen.transformations.transformation_abc import AbstractTransformation
from pymatgen.util.testing import FAKE_POTCAR_DIR, TEST_FILES_DIR, VASP_IN_DIR, MatSciTest


class TestCifTransmuter(MatSciTest):
    def test_init(self):
        trafos = [SubstitutionTransformation({"Fe": "Mn", "Fe2+": "Mn2+"})]
        tsc = CifTransmuter.from_filenames([f"{TEST_FILES_DIR}/cif/MultiStructure.cif"], trafos)
        assert len(tsc) == 2
        expected = {"Mn", "O", "Li", "P"}
        for s in tsc:
            els = {el.symbol for el in s.final_structure.elements}
            assert expected == els


class TestPoscarTransmuterDirectInit:
    def test_init(self):
        # ``PoscarTransmuter`` can be constructed directly from a VASP5 POSCAR string.
        with open(f"{VASP_IN_DIR}/POSCAR", encoding="utf-8") as file:
            poscar_str = file.read()
        tsc = PoscarTransmuter(poscar_str)
        assert tsc[0].final_structure is not None


class TestPoscarTransmuter(MatSciTest):
    def test_init(self):
        trafos = [SubstitutionTransformation({"Fe": "Mn"})]
        tsc = PoscarTransmuter.from_filenames([f"{VASP_IN_DIR}/POSCAR", f"{VASP_IN_DIR}/POSCAR"], trafos)
        assert len(tsc) == 2
        expected = {"Mn", "O", "P"}
        for substitution in tsc:
            els = {el.symbol for el in substitution.final_structure.elements}
            assert expected == els

    def test_transmuter(self):
        tsc = PoscarTransmuter.from_filenames([f"{VASP_IN_DIR}/POSCAR"])
        tsc.append_transformation(RemoveSpeciesTransformation("O"))
        assert len(tsc[0].final_structure) == 8

        tsc.append_transformation(SubstitutionTransformation({"Fe": {"Fe2+": 0.25, "Mn3+": 0.75}, "P": "P5+"}))
        tsc.append_transformation(OrderDisorderedStructureTransformation(), extend_collection=50)
        assert len(tsc) == 4

        trafo = SuperTransformation(
            [
                SubstitutionTransformation({"Fe2+": "Mg2+"}),
                SubstitutionTransformation({"Fe2+": "Zn2+"}),
                SubstitutionTransformation({"Fe2+": "Be2+"}),
            ]
        )
        tsc.append_transformation(trafo, extend_collection=True)
        assert len(tsc) == 12
        for x in tsc:
            # should be 4 trans + starting structure
            assert len(x) == 5, "something might be wrong with the number of transformations in the history"

        # test the filter
        tsc.apply_filter(ContainsSpecieFilter(["Zn2+", "Be2+", "Mn4+"], strict_compare=True, AND=False))
        assert len(tsc) == 8
        assert tsc.transformed_structures[0].as_dict()["history"][-1]["@class"] == "ContainsSpecieFilter"

        tsc.apply_filter(ContainsSpecieFilter(["Be2+"]))
        assert len(tsc) == 4

        # Test set_parameter and add_tag.
        tsc.set_parameter("para1", "hello")
        assert tsc.transformed_structures[0].as_dict()["other_parameters"]["para1"] == "hello"
        tsc.add_tags(["world", "universe"])
        assert tsc.transformed_structures[0].as_dict()["other_parameters"]["tags"] == [
            "world",
            "universe",
        ]


class _PoolTrafo(AbstractTransformation):
    """Fake parallelizable transformation to exercise the ``StandardTransmuter`` Pool branch."""

    @property
    def use_multiprocessing(self) -> bool:
        return True

    def apply_transformation(self, structure, **kwargs):
        return structure


class TestStandardTransmuterExtra(MatSciTest):
    def _apply_one(self):
        struct = MatSciTest.get_structure("LiFePO4")
        trafo = SubstitutionTransformation({"Li": "Na"})
        return StandardTransmuter([TransformedStructure(struct, [])], [trafo])

    def test_getattr(self):
        # ``__getattr__`` forwards to the TransformedStructures, returning a list.
        t = StandardTransmuter([TransformedStructure(MatSciTest.get_structure("LiFePO4"))])
        res = t.reduced_formula
        assert isinstance(res, list)
        assert any("Fe" in f for f in res)

    def test_str(self):
        t = StandardTransmuter([TransformedStructure(MatSciTest.get_structure("LiFePO4"))])
        assert "Current structures" in str(t)

    def test_undo_last_change_and_redo_next_change(self):
        t = self._apply_one()
        assert t[0].final_structure.reduced_formula == "NaFePO4"
        t.undo_last_change()
        assert "Na" not in {el.symbol for el in t[0].final_structure.elements}
        t.redo_next_change()
        assert t[0].final_structure.reduced_formula == "NaFePO4"

    def test_extend_transformations(self):
        t = StandardTransmuter([TransformedStructure(MatSciTest.get_structure("LiFePO4"))])
        t.extend_transformations([SubstitutionTransformation({"Fe": "Mn"})])
        assert "Mn" in {el.symbol for el in t[0].final_structure.elements}

    def test_multiprocess_pool(self, monkeypatch):
        struct = MatSciTest.get_structure("LiFePO4")
        t = StandardTransmuter([TransformedStructure(struct, [])], ncores=2)

        # Mock multiprocessing.Pool so the branch is exercised without spawning
        # (a real Pool hangs here because a bare generator is passed to map()).
        from pymatgen.alchemy import transmuters

        class _FakePool:
            def __init__(self, ncores):
                self.ncores = ncores

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def map(self, fn, iterable, chunksize=None):
                return [fn(item) for item in iterable if item]

        monkeypatch.setattr(transmuters, "Pool", _FakePool)
        result = t.append_transformation(_PoolTrafo())
        assert len(t.transformed_structures) == 1
        assert len(result) == 1

    def test_write_vasp_input(self, tmp_path):
        SETTINGS["PMG_VASP_PSP_DIR"] = FAKE_POTCAR_DIR
        struct = MatSciTest.get_structure("LiFePO4")
        t = StandardTransmuter([TransformedStructure(struct, [])], [SubstitutionTransformation({"Li": "Na"})])
        outdir = tmp_path / "out"
        t.write_vasp_input(vasp_input_set=MPRelaxSet, output_dir=str(outdir))
        # batch_write_vasp_input writes one sub-directory per transformed structure
        assert any(outdir.rglob("INCAR"))

    def test_append_transformed_structures(self):
        a = StandardTransmuter([TransformedStructure(MatSciTest.get_structure("LiFePO4"))])
        b = StandardTransmuter([TransformedStructure(MatSciTest.get_structure("LiFePO4"))])
        a.append_transformed_structures(b)
        assert len(a.transformed_structures) == 2

    def test_append_transformed_structures_typeerror(self):
        a = StandardTransmuter([TransformedStructure(MatSciTest.get_structure("LiFePO4"))])
        with pytest.raises(TypeError, match="incorrect type"):
            a.append_transformed_structures([{"not": "a transformed structure"}])

    def test_apply_transformation_helper(self):
        # exercise the module-level ``_apply_transformation`` used by the Pool path
        struct = MatSciTest.get_structure("LiFePO4")
        ts = TransformedStructure(struct, [])
        out = _apply_transformation((ts, SubstitutionTransformation({"Li": "Na"}), False, True))
        assert out[0] is ts


def _subfolder(transformed_structure):
    # callable that names a subdirectory from the transformed structure
    return "group"


class TestBatchWriteVaspInputExtra(MatSciTest):
    def test_batch_write_vasp_input(self, tmp_path):
        SETTINGS["PMG_VASP_PSP_DIR"] = FAKE_POTCAR_DIR
        struct = MatSciTest.get_structure("LiFePO4")
        transformed_structures = [TransformedStructure(struct, [])]
        outdir = tmp_path / "batch"

        batch_write_vasp_input(transformed_structures, MPRelaxSet, output_dir=str(outdir), create_directory=True)
        assert any(outdir.rglob("INCAR"))

        # subfolder branch
        sub = outdir / "sub"
        batch_write_vasp_input(
            transformed_structures,
            MPRelaxSet,
            output_dir=str(sub),
            create_directory=True,
            subfolder=_subfolder,
        )
        assert sub.is_dir()

        # include_cif branch
        cifdir = outdir / "cif"
        batch_write_vasp_input(
            transformed_structures,
            MPRelaxSet,
            output_dir=str(cifdir),
            create_directory=True,
            include_cif=True,
        )
        assert cifdir.is_dir()
