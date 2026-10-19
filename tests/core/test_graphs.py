from __future__ import annotations

import copy
import re
import warnings
from glob import glob
from pathlib import Path
from shutil import which

import networkx as nx
import networkx.algorithms.isomorphism as iso
import pytest
from monty.serialization import loadfn
from pytest import approx

import pymatgen.core.graphs as graphs_module
from pymatgen.core import Lattice, Molecule, Site, Structure
from pymatgen.core.graphs import EL_COLORS, MoleculeGraph, MolGraphSplitError, PeriodicSite, StructureGraph, _isomorphic
from pymatgen.core.local_env import (
    CovalentBondNN,
    CutOffDictNN,
    MinimumDistanceNN,
    MinimumOKeeffeNN,
    OpenBabelNN,
    VoronoiNN,
)
from pymatgen.core.structure import FunctionalGroups
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

try:
    from openbabel import openbabel
except ImportError:
    openbabel = None

try:
    import pygraphviz
except ImportError:
    pygraphviz = None

from pymatgen.command_line.critic2_caller import Critic2Analysis

__author__ = "Matthew Horton, Evan Spotte-Smith"
__version__ = "0.1"
__maintainer__ = "Matthew Horton"
__email__ = "mkhorton@lbl.gov"
__status__ = "Beta"
__date__ = "August 2017"

TEST_DIR = f"{TEST_FILES_DIR}/core/graphs"


class _FakePopen:
    """Stand-in for subprocess.Popen so draw_graph_to_file can run without GraphViz."""

    returncode = 0
    calls: list[list[str]]

    def __init__(self, args, **kwargs):
        self.args = args
        type(self).calls.append(args)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def communicate(self):
        return None, None


@pytest.fixture
def fake_graphviz(monkeypatch):
    """Patch GraphViz lookups so draw_graph_to_file runs its styling logic without binaries.

    Returns a dict collecting the graph handed to write_dot and the GraphViz command lines.
    """
    captured: dict = {"graphs": [], "popen": _FakePopen}

    def fake_write_dot(graph, path):
        captured["graphs"].append(graph)
        Path(path).write_text("digraph {}\n", encoding="utf-8")

    fake_popen = type("FakePopen", (_FakePopen,), {"calls": [], "returncode": 0})
    captured["popen"] = fake_popen
    monkeypatch.setattr(graphs_module, "which", lambda algo: f"/fake/bin/{algo}")
    monkeypatch.setattr(graphs_module, "write_dot", fake_write_dot)
    monkeypatch.setattr(graphs_module.subprocess, "Popen", fake_popen)
    return captured


class TestStructureGraph(MatSciTest):
    def setup_method(self):
        # trivial example, simple square lattice for testing
        structure = Structure(Lattice.tetragonal(5, 50), ["H"], [[0, 0, 0]])
        self.square_sg = StructureGraph.from_empty_graph(structure, edge_weight_name="", edge_weight_units="")
        self.square_sg.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(1, 0, 0))
        self.square_sg.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(-1, 0, 0))
        self.square_sg.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(0, 1, 0))
        self.square_sg.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(0, -1, 0))
        # TODO: decorating still fails because the structure graph gives a CN of 8 for this square lattice
        # self.square_sg.decorate_structure_with_ce_info()

        # body-centered square lattice for testing
        structure = Structure(Lattice.tetragonal(5, 50), ["H", "He"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.bc_square_sg = StructureGraph.from_empty_graph(structure, edge_weight_name="", edge_weight_units="")
        self.bc_square_sg.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(1, 0, 0))
        self.bc_square_sg.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(-1, 0, 0))
        self.bc_square_sg.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(0, 1, 0))
        self.bc_square_sg.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(0, -1, 0))
        self.bc_square_sg.add_edge(0, 1, from_jimage=(0, 0, 0), to_jimage=(0, 0, 0))
        self.bc_square_sg.add_edge(0, 1, from_jimage=(0, 0, 0), to_jimage=(-1, 0, 0))
        self.bc_square_sg.add_edge(0, 1, from_jimage=(0, 0, 0), to_jimage=(-1, -1, 0))
        self.bc_square_sg.add_edge(0, 1, from_jimage=(0, 0, 0), to_jimage=(0, -1, 0))

        # body-centered square lattice for testing
        # directions reversed, should be equivalent to bc_square
        structure = Structure(Lattice.tetragonal(5, 50), ["H", "He"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.bc_square_sg_r = StructureGraph.from_empty_graph(structure, edge_weight_name="", edge_weight_units="")
        self.bc_square_sg_r.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(1, 0, 0))
        self.bc_square_sg_r.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(-1, 0, 0))
        self.bc_square_sg_r.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(0, 1, 0))
        self.bc_square_sg_r.add_edge(0, 0, from_jimage=(0, 0, 0), to_jimage=(0, -1, 0))
        self.bc_square_sg_r.add_edge(0, 1, from_jimage=(0, 0, 0), to_jimage=(0, 0, 0))
        self.bc_square_sg_r.add_edge(1, 0, from_jimage=(-1, 0, 0), to_jimage=(0, 0, 0))
        self.bc_square_sg_r.add_edge(1, 0, from_jimage=(-1, -1, 0), to_jimage=(0, 0, 0))
        self.bc_square_sg_r.add_edge(1, 0, from_jimage=(0, -1, 0), to_jimage=(0, 0, 0))

        # MoS2 example, structure graph obtained from critic2
        # (not ground state, from mp-1023924, single layer)
        stdout_file = f"{TEST_FILES_DIR}/command_line/critic2/MoS2_critic2_stdout.txt"
        with open(stdout_file, encoding="utf-8") as txt_file:
            reference_stdout = txt_file.read()
        self.structure = Structure.from_file(f"{TEST_FILES_DIR}/command_line/critic2/MoS2.cif")
        with pytest.warns(DeprecationWarning, match="Parsing critic2 standard output is deprecated"):
            c2o = Critic2Analysis(self.structure, reference_stdout)
        self.mos2_sg = c2o.structure_graph(include_critical_points=False)

        lattice = Lattice.cubic(4.17)
        species = ["Ni", "O"]
        coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
        self.NiO = Structure.from_spacegroup(225, lattice, species, coords).get_primitive_structure()

        # BCC example.
        self.bcc = Structure(Lattice.cubic(5.0), ["He", "He"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    def test_inappropriate_construction(self):
        # Check inappropriate strategy
        with pytest.raises(ValueError, match="Chosen strategy is not designed for use with structures"):
            StructureGraph.from_local_env_strategy(self.NiO, CovalentBondNN())

    def test_properties(self):
        assert self.mos2_sg.name == "bonds"
        assert self.mos2_sg.edge_weight_name == "bond_length"
        assert self.mos2_sg.edge_weight_unit == "Å"
        assert self.mos2_sg.get_coordination_of_site(0) == 6
        assert len(self.mos2_sg.get_connected_sites(0)) == 6
        assert isinstance(self.mos2_sg.get_connected_sites(0)[0].site, PeriodicSite)
        assert str(self.mos2_sg.get_connected_sites(0)[0].site.specie) == "S"
        assert self.mos2_sg.get_connected_sites(0, jimage=(0, 0, 100))[0].site.frac_coords[2] == approx(100.303027)

        # these two graphs should be equivalent
        for n in range(len(self.bc_square_sg)):
            assert self.bc_square_sg.get_coordination_of_site(n) == self.bc_square_sg_r.get_coordination_of_site(n)

        # test we're not getting duplicate connected sites
        # thanks to Jack D. Sundberg for reporting this bug

        # known example where this bug occurred due to edge weights not being
        # bit-for-bit identical in otherwise identical edges
        nacl_lattice = [
            [3.48543625, 0.0, 2.01231756],
            [1.16181208, 3.28610081, 2.01231756],
            [0.0, 0.0, 4.02463512],
        ]
        nacl = Structure(nacl_lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

        nacl_graph = StructureGraph.from_local_env_strategy(nacl, CutOffDictNN({("Cl", "Cl"): 5.0}))

        assert len(nacl_graph.get_connected_sites(1)) == 12
        assert len(nacl_graph.graph.get_edge_data(1, 1)) == 6

    def test_set_node_attributes(self):
        self.square_sg.set_node_attributes()

        specie = nx.get_node_attributes(self.square_sg.graph, "specie")
        coords = nx.get_node_attributes(self.square_sg.graph, "coords")
        properties = nx.get_node_attributes(self.square_sg.graph, "properties")

        for idx, site in enumerate(self.square_sg.structure):
            assert str(specie[idx]) == str(site.specie)
            assert coords[idx][0] == site.coords[0]
            assert coords[idx][1] == site.coords[1]
            assert coords[idx][2] == site.coords[2]
            assert properties[idx] == site.properties

    def test_edge_editing(self):
        square = copy.deepcopy(self.square_sg)

        square.alter_edge(
            0,
            0,
            to_jimage=(1, 0, 0),
            new_weight=0.0,
            new_edge_properties={"foo": "bar"},
        )
        new_edge = square.graph.get_edge_data(0, 0)[0]
        assert new_edge["weight"] == approx(0.0)
        assert new_edge["foo"] == "bar"

        square.break_edge(0, 0, to_jimage=(1, 0, 0))

        assert len(square.graph.get_edge_data(0, 0)) == 1

    def test_insert_remove(self):
        struct_copy = copy.deepcopy(self.square_sg.structure)
        square_copy = copy.deepcopy(self.square_sg)

        # Ensure that insert_node appropriately wraps Structure.insert()
        struct_copy.insert(1, "O", [0.5, 0.5, 0.5])
        square_copy.insert_node(1, "O", [0.5, 0.5, 0.5])
        assert struct_copy == square_copy.structure

        # Test that removal is also equivalent between Structure and StructureGraph.structure
        struct_copy.remove_sites([1])
        square_copy.remove_nodes([1])
        assert struct_copy == square_copy.structure

        square_copy.insert_node(
            1,
            "O",
            [0.5, 0.5, 0.5],
            edges=[{"from_index": 1, "to_index": 0, "to_jimage": (0, 0, 0)}],
        )
        assert square_copy.get_coordination_of_site(1) == 1

        # Test that StructureGraph.graph is correctly updated
        square_copy.insert_node(
            1,
            "H",
            [0.5, 0.5, 0.75],
            edges=[{"from_index": 1, "to_index": 2, "to_jimage": (0, 0, 0)}],
        )
        square_copy.remove_nodes([1])

        assert square_copy.graph.number_of_nodes() == 2
        assert square_copy.graph.number_of_edges() == 3

    def test_substitute(self):
        structure = Structure.from_file(f"{TEST_FILES_DIR}/cif/Li2O.cif")
        molecule = FunctionalGroups["methyl"]

        structure_copy = copy.deepcopy(structure)
        structure_copy_graph = copy.deepcopy(structure)

        struct_graph = StructureGraph.from_local_env_strategy(structure, MinimumDistanceNN())
        sg_copy = copy.deepcopy(struct_graph)

        # Ensure that strings and molecules lead to equivalent substitutions
        struct_graph.substitute_group(1, molecule, MinimumDistanceNN)
        sg_copy.substitute_group(1, "methyl", MinimumDistanceNN)
        assert struct_graph == sg_copy

        # Ensure that the underlying structure has been modified as expected
        structure_copy.substitute(1, "methyl")
        assert structure_copy == struct_graph.structure

        # Test inclusion of graph dictionary
        graph_dict = {
            (0, 1): {"weight": 0.5},
            (0, 2): {"weight": 0.5},
            (0, 3): {"weight": 0.5},
        }

        sg_with_graph = StructureGraph.from_local_env_strategy(structure_copy_graph, MinimumDistanceNN())
        sg_with_graph.substitute_group(1, "methyl", MinimumDistanceNN, graph_dict=graph_dict)
        edge = sg_with_graph.graph.get_edge_data(11, 13)[0]
        assert edge["weight"] == approx(0.5)

    def test_auto_image_detection(self):
        struct_graph = StructureGraph.from_empty_graph(self.structure)
        struct_graph.add_edge(0, 0)

        assert len(list(struct_graph.graph.edges(data=True))) == 3

    @pytest.mark.xfail(reason="Need someone to fix this, see issue 4206")
    def test_str(self):
        square_sg_str_ref = """Structure Graph
Structure:
Full Formula (H1)
Reduced Formula: H2
abc   :   5.000000   5.000000  50.000000
angles:  90.000000  90.000000  90.000000
Sites (1)
  #  SP      a    b    c
---  ----  ---  ---  ---
  0  H       0    0    0
Graph: bonds
from    to  to_image
----  ----  ------------
   0     0  (1, 0, 0)
   0     0  (-1, 0, 0)
   0     0  (0, 1, 0)
   0     0  (0, -1, 0)
"""

        mos2_sg_str_ref = """Structure Graph
Structure:
Full Formula (Mo1 S2)
Reduced Formula: MoS2
abc   :   3.190316   3.190315  17.439502
angles:  90.000000  90.000000 120.000006
Sites (3)
  #  SP           a         b         c
---  ----  --------  --------  --------
  0  Mo    0.333333  0.666667  0.213295
  1  S     0.666667  0.333333  0.303027
  2  S     0.666667  0.333333  0.123562
Graph: bonds
from    to  to_image      bond_length (A)
----  ----  ------------  ------------------
   0     1  (-1, 0, 0)    2.417e+00
   0     1  (0, 0, 0)     2.417e+00
   0     1  (0, 1, 0)     2.417e+00
   0     2  (0, 1, 0)     2.417e+00
   0     2  (-1, 0, 0)    2.417e+00
   0     2  (0, 0, 0)     2.417e+00
"""

        # change Å to A
        self.mos2_sg.graph.graph["edge_weight_units"] = "A"
        self.assert_str_content_equal(str(self.square_sg), square_sg_str_ref)
        self.assert_str_content_equal(str(self.mos2_sg), mos2_sg_str_ref)

    def test_mul(self):
        square_sg_mul = self.square_sg * (2, 1, 1)

        square_sg_mul_ref_str = """Structure Graph
Structure:
Full Formula (H2)
Reduced Formula: H2
abc   :  10.000000   5.000000  50.000000
angles:  90.000000  90.000000  90.000000
Sites (2)
  #  SP      a    b    c
---  ----  ---  ---  ---
  0  H     0      0    0
  1  H     0.5    0   -0
Graph: bonds
from    to  to_image
----  ----  ------------
   0     0  (0, 1, 0)
   0     0  (0, -1, 0)
   0     1  (0, 0, 0)
   0     1  (-1, 0, 0)
   1     1  (0, 1, 0)
   1     1  (0, -1, 0)
"""
        square_sg_mul_actual_str = str(square_sg_mul)

        # only testing bonds portion,
        # the c frac_coord of the second H can vary from
        # 0 to -0 depending on machine precision
        square_sg_mul_ref_str = "\n".join(square_sg_mul_ref_str.splitlines()[11:])
        square_sg_mul_actual_str = "\n".join(square_sg_mul_actual_str.splitlines()[11:])

        # TODO: below check is failing, see issue 4206
        warnings.warn("part of test_mul is failing, see issue 4206", stacklevel=2)
        # self.assert_str_content_equal(square_sg_mul_actual_str, square_sg_mul_ref_str)

        # test sequential multiplication
        sq_sg_1 = self.square_sg * (2, 2, 1)
        sq_sg_1 *= 2, 2, 1
        sq_sg_2 = self.square_sg * (4, 4, 1)
        assert sq_sg_1.graph.number_of_edges() == sq_sg_2.graph.number_of_edges()
        # TODO: the below test still gives 8 != 4
        # assert self.square_sg.get_coordination_of_site(0) == 4

        mos2_sg_mul = self.mos2_sg * (3, 3, 1)
        for idx in mos2_sg_mul.structure.indices_from_symbol("Mo"):
            assert mos2_sg_mul.get_coordination_of_site(idx) == 6

        mos2_sg_premul = StructureGraph.from_local_env_strategy(self.structure * (3, 3, 1), MinimumDistanceNN())
        assert mos2_sg_mul == mos2_sg_premul

        # test 3D Structure

        nio_struct_graph = StructureGraph.from_local_env_strategy(self.NiO, MinimumDistanceNN())
        nio_struct_graph *= 3

        for n in range(len(nio_struct_graph)):
            assert nio_struct_graph.get_coordination_of_site(n) == 6

    @pytest.mark.skipif(
        pygraphviz is None or not (which("neato") and which("fdp")),
        reason="graphviz executables not present",
    )
    def test_draw(self):
        # draw MoS2 graph
        self.mos2_sg.draw_graph_to_file(
            f"{self.tmp_path}/MoS2_single.pdf",
            image_labels=True,
            hide_image_edges=False,
        )
        mos2_sg = self.mos2_sg * (9, 9, 1)
        mos2_sg.draw_graph_to_file(f"{self.tmp_path}/MoS2.pdf", algo="neato")

        # draw MoS2 graph that's been successively multiplied
        mos2_sg_2 = self.mos2_sg * (3, 3, 1)
        mos2_sg_2 *= 3, 3, 1
        mos2_sg_2.draw_graph_to_file(f"{self.tmp_path}/MoS2_twice_mul.pdf", algo="neato", hide_image_edges=True)

        # draw MoS2 graph that's generated from a pre-multiplied Structure
        mos2_sg_premul = StructureGraph.from_local_env_strategy(self.structure * (3, 3, 1), MinimumDistanceNN())
        mos2_sg_premul.draw_graph_to_file(f"{self.tmp_path}/MoS2_premul.pdf", algo="neato", hide_image_edges=True)

        # draw graph for a square lattice
        self.square_sg.draw_graph_to_file(f"{self.tmp_path}/square_single.pdf", hide_image_edges=False)
        square_sg = self.square_sg * (5, 5, 1)
        square_sg.draw_graph_to_file(
            f"{self.tmp_path}/square.pdf",
            algo="neato",
            image_labels=True,
            node_labels=False,
        )

        # draw graph for a body-centered square lattice
        self.bc_square_sg.draw_graph_to_file(f"{self.tmp_path}/bc_square_single.pdf", hide_image_edges=False)
        bc_square_sg = self.bc_square_sg * (9, 9, 1)
        bc_square_sg.draw_graph_to_file(f"{self.tmp_path}/bc_square.pdf", algo="neato", image_labels=False)

        # draw graph for a body-centered square lattice defined in an alternative way
        self.bc_square_sg_r.draw_graph_to_file(f"{self.tmp_path}/bc_square_r_single.pdf", hide_image_edges=False)
        bc_square_sg_r = self.bc_square_sg_r * (9, 9, 1)
        bc_square_sg_r.draw_graph_to_file(f"{self.tmp_path}/bc_square_r.pdf", algo="neato", image_labels=False)

        # ensure PDF files were created
        pdfs = {path.split("/")[-1] for path in glob(f"{self.tmp_path}/*.pdf")}
        expected_pdfs = {
            "bc_square_r_single.pdf",
            "bc_square_r.pdf",
            "bc_square_single.pdf",
            "bc_square.pdf",
            "MoS2_premul.pdf",
            "MoS2_single.pdf",
            "MoS2_twice_mul.pdf",
            "MoS2.pdf",
            "square_single.pdf",
            "square.pdf",
        }
        assert pdfs == expected_pdfs

    def test_as_from_dict(self):
        dct = self.mos2_sg.as_dict()
        struct_graph = StructureGraph.from_dict(dct)
        d2 = struct_graph.as_dict()
        assert dct == d2

    def test_from_local_env_and_equality_and_diff(self):
        min_dist_nn = MinimumDistanceNN()
        struct_graph = StructureGraph.from_local_env_strategy(self.structure, min_dist_nn)

        assert struct_graph.graph.number_of_edges() == 6

        nn2 = MinimumOKeeffeNN()
        sg2 = StructureGraph.from_local_env_strategy(self.structure, nn2)

        assert struct_graph == sg2
        assert struct_graph == self.mos2_sg

        # TODO: find better test case where graphs are different
        diff = struct_graph.diff(sg2)
        assert diff["dist"] == 0

        assert self.square_sg.get_coordination_of_site(0) == 4

    def test_from_edges(self):
        edges = {
            (0, 0, (0, 0, 0), (1, 0, 0)): None,
            (0, 0, (0, 0, 0), (-1, 0, 0)): None,
            (0, 0, (0, 0, 0), (0, 1, 0)): None,
            (0, 0, (0, 0, 0), (0, -1, 0)): None,
        }

        structure = Structure(Lattice.tetragonal(5.0, 50.0), ["H"], [[0, 0, 0]])

        struct_graph = StructureGraph.from_edges(structure, edges)

        assert struct_graph == self.square_sg

    def test_extract_molecules(self):
        structure_file = f"{TEST_FILES_DIR}/cif/H6PbCI3N_mp-977013_symmetrized.cif"

        struct = Structure.from_file(structure_file)

        min_dist_nn = MinimumDistanceNN()
        struct_graph = StructureGraph.from_local_env_strategy(struct, min_dist_nn)

        molecules = struct_graph.get_subgraphs_as_molecules()
        assert molecules[0].formula == "H3 C1"
        assert len(molecules) == 1

        molecules = self.mos2_sg.get_subgraphs_as_molecules()
        assert len(molecules) == 0

    def test_types_and_weights_of_connections(self):
        types = self.mos2_sg.types_and_weights_of_connections

        assert len(types["Mo-S"]) == 6
        assert types["Mo-S"][0] == approx(2.416931678417331)

    def test_weight_statistics(self):
        weight_statistics = self.mos2_sg.weight_statistics

        assert len(weight_statistics["all_weights"]) == 6
        assert weight_statistics["min"] == approx(2.4169314100201875)
        assert weight_statistics["variance"] == approx(0, abs=1e-10)

    def test_types_of_coordination_environments(self):
        types = self.mos2_sg.types_of_coordination_environments()
        assert types == ["Mo-S(6)", "S-Mo(3)"]

        types_anonymous = self.mos2_sg.types_of_coordination_environments(anonymous=True)
        assert types_anonymous == ["A-B(3)", "A-B(6)"]

    def test_no_duplicate_hops(self):
        test_structure = Structure(
            lattice=[
                [2.990355, -5.149042, 0.0],
                [2.990355, 5.149042, 0.0],
                [0.0, 0.0, 24.51998],
            ],
            species=["Ba"],
            coords=[[0.005572, 0.994428, 0.151095]],
        )

        min_dist_nn = MinimumDistanceNN(cutoff=6, get_all_sites=True)

        struct_graph = StructureGraph.from_local_env_strategy(test_structure, min_dist_nn)

        assert struct_graph.graph.number_of_edges() == 3

    def test_sort(self):
        sg = copy.deepcopy(self.bc_square_sg_r)
        # insert an unsorted edge, don't use sg.add_edge as it auto-sorts
        sg.graph.add_edge(3, 1, to_jimage=(0, 0, 0))
        sg.graph.add_edge(2, 1, to_jimage=(0, 0, 0))

        assert list(sg.graph.edges)[-2:] == [(3, 1, 0), (2, 1, 0)]
        sg.sort()
        assert list(sg.graph.edges)[-2:] == [(1, 3, 0), (1, 2, 0)]

    def test_from_empty_graph_requires_units(self):
        with pytest.raises(ValueError, match="Please specify units associated with your edge weights"):
            StructureGraph.from_empty_graph(self.bcc, edge_weight_name="bond_length")

    def test_init_from_structure_graph(self):
        sg_copy = StructureGraph(self.mos2_sg)
        assert isinstance(sg_copy.structure, Structure)
        assert sg_copy == self.mos2_sg

    def test_init_restores_image_tuples(self):
        sg = StructureGraph.from_empty_graph(self.bcc)
        sg.add_edge(0, 1, to_jimage=(0, 0, 0), edge_properties={"from_jimage": [0, 0, 0]})
        restored = StructureGraph.from_dict(sg.as_dict())
        edge = restored.graph.get_edge_data(0, 1)[0]
        assert edge["to_jimage"] == (0, 0, 0)
        assert edge["from_jimage"] == (0, 0, 0)
        assert isinstance(edge["from_jimage"], tuple)

    def test_from_edges_with_properties(self):
        edges = {
            (0, 1, (0, 0, 0), (0, 0, 0)): {"weight": 4.33, "bond_type": "metallic"},
            (0, 1, (0, 0, 0), (-1, 0, 0)): {"weight": 4.33},
        }
        sg = StructureGraph.from_edges(self.bcc, edges)
        assert sg.edge_weight_name == "weight"
        edge_data = sg.graph.get_edge_data(0, 1)
        assert len(edge_data) == 2
        by_image = {data["to_jimage"]: data for data in edge_data.values()}
        assert by_image[0, 0, 0] == {"to_jimage": (0, 0, 0), "weight": 4.33, "bond_type": "metallic"}
        assert by_image[-1, 0, 0] == {"to_jimage": (-1, 0, 0), "weight": 4.33}
        assert sg.graph.nodes[0]["specie"] == "He"

    @pytest.mark.parametrize(
        ("edges", "match"),
        [
            ({0: None}, r"Edges must be given as \(from_index, to_index, from_image, to_image\) tuples"),
            ({(0, 5, (0, 0, 0), (0, 0, 0)): None}, "Edges cannot be added if nodes are not present"),
        ],
    )
    def test_from_edges_invalid(self, edges, match):
        with pytest.raises(ValueError, match=match):
            StructureGraph.from_edges(self.bcc, edges)

    def test_add_edge_to_itself_is_ignored(self):
        sg = StructureGraph.from_empty_graph(self.bcc)
        with pytest.warns(UserWarning, match="Tried to create a bond to itself"):
            sg.add_edge(0, 0, to_jimage=(0, 0, 0))
        assert sg.graph.number_of_edges() == 0

    def test_add_edge_duplicate_warns(self):
        sg = copy.deepcopy(self.square_sg)
        n_edges = sg.graph.number_of_edges()
        with pytest.warns(UserWarning, match="Trying to add an edge that already exists"):
            sg.add_edge(0, 0, to_jimage=(1, 0, 0))
        assert sg.graph.number_of_edges() == n_edges

    def test_insert_node_invalid_edges(self):
        sg = copy.deepcopy(self.square_sg)
        with pytest.raises(RuntimeError, match="Some edges are invalid"):
            # missing the required "to_jimage" key
            sg.insert_node(1, "O", [0.5, 0.5, 0.5], edges=[{"from_index": 1, "to_index": 0}])

    def test_alter_edge_missing(self):
        with pytest.raises(ValueError, match="Edge between 0 and 1 cannot be altered; no edge exists"):
            self.square_sg.alter_edge(0, 1, new_weight=1.0)

    def test_break_edge_errors_and_reverse(self):
        sg = copy.deepcopy(self.bc_square_sg)
        with pytest.raises(ValueError, match="Image must be supplied, to avoid ambiguity"):
            sg.break_edge(0, 1)

        # edges are stored with from_index < to_index, so (1, 0) only exists in reverse
        with pytest.raises(ValueError, match="Edge cannot be broken between 1 and 0; no edge exists"):
            sg.break_edge(1, 0, to_jimage=(-1, 0, 0))

        assert len(sg.graph.get_edge_data(0, 1)) == 4
        sg.break_edge(1, 0, to_jimage=(-1, 0, 0), allow_reverse=True)
        remaining = {data["to_jimage"] for data in sg.graph.get_edge_data(0, 1).values()}
        assert remaining == {(0, 0, 0), (-1, -1, 0), (0, -1, 0)}

    def test_substitute_group_unknown(self):
        sg = copy.deepcopy(self.square_sg)
        with pytest.raises(RuntimeError, match="Can't find functional group in list"):
            sg.substitute_group(0, "not-a-functional-group", MinimumDistanceNN)

    def test_rmul(self):
        rmul = (2, 1, 1) * self.square_sg
        mul = self.square_sg * (2, 1, 1)
        assert len(rmul) == len(mul) == 2
        assert rmul == mul

    def test_str_repr_with_weights(self):
        mos2_str = str(self.mos2_sg)
        assert mos2_str.startswith("Structure Graph\nStructure: \n")
        assert "Graph: bonds" in mos2_str
        assert "bond_length (Å)" in mos2_str
        assert mos2_str.count("2.417e+00") == 6

        mos2_repr = repr(self.mos2_sg)
        assert mos2_repr.startswith("Structure Graph\nStructure: \n")
        assert repr(self.mos2_sg.structure) in mos2_repr
        assert mos2_repr.count("2.417e+00") == 6

        # graph without weight name and with empty units
        square_str = str(self.square_sg)
        assert "from    to  to_image    \n----  ----  ------------\n" in square_str
        assert "e+00" not in square_str

    def test_eq_other_type(self):
        assert self.square_sg.__eq__("not a graph") is NotImplemented
        assert self.square_sg != "not a graph"

    def test_diff(self):
        with pytest.raises(ValueError, match="Meaningless to compare StructureGraphs"):
            self.square_sg.diff(self.mos2_sg)

        other = copy.deepcopy(self.mos2_sg)
        other.break_edge(0, 1, to_jimage=(0, 0, 0))
        other.add_edge(1, 2, to_jimage=(0, 0, 0))
        diff = self.mos2_sg.diff(other)
        assert diff["self"] == {(0, 1, (0, 0, 0))}
        assert diff["other"] == {(1, 2, (0, 0, 0))}
        assert len(diff["both"]) == 5
        assert diff["dist"] == approx(1 - 5 / 7)

        # non-strict comparison works across different structures, using species labels
        diff_loose = self.bc_square_sg.diff(self.mos2_sg, strict=False)
        assert diff_loose["self"] == {("H", "H"), ("H", "He")}
        assert diff_loose["other"] == {("Mo", "S")}
        assert diff_loose["both"] == set()
        assert diff_loose["dist"] == approx(1.0)

        empty = StructureGraph.from_empty_graph(self.bcc)
        assert empty.diff(copy.deepcopy(empty))["dist"] == approx(0.0)

    def test_get_subgraphs_as_molecules_twice(self):
        sg = copy.deepcopy(self.mos2_sg)
        assert sg.get_subgraphs_as_molecules() == []
        with pytest.raises(RuntimeError, match="Supercell spacegroup is not None"):
            sg.get_subgraphs_as_molecules()

    def test_get_subgraphs_as_molecules_use_weights(self):
        struct = Structure.from_file(f"{TEST_FILES_DIR}/cif/H6PbCI3N_mp-977013_symmetrized.cif")
        struct_graph = StructureGraph.from_local_env_strategy(struct, MinimumDistanceNN(), weights=True)
        molecules = struct_graph.get_subgraphs_as_molecules(use_weights=True)
        assert [mol.formula for mol in molecules] == ["H3 C1"]

    def test_draw_graph_to_file_requires_graphviz(self, monkeypatch):
        monkeypatch.setattr(graphs_module, "which", lambda _algo: None)
        with pytest.raises(RuntimeError, match="requires GraphViz binaries to be in the path"):
            self.mos2_sg.draw_graph_to_file(f"{self.tmp_path}/MoS2.pdf")

    def test_draw_graph_to_file_styling(self, fake_graphviz):
        filename = f"{self.tmp_path}/MoS2.png"
        self.mos2_sg.draw_graph_to_file(
            filename,
            hide_image_edges=False,
            edge_colors=True,
            node_labels=True,
            weight_labels=True,
            image_labels=True,
            color_scheme="Jmol",
            keep_dot=True,
            algo="neato",
        )
        assert fake_graphviz["popen"].calls == [["neato", "-T", "png", f"{self.tmp_path}/MoS2.dot"]]
        assert Path(f"{self.tmp_path}/MoS2.dot").is_file()
        assert Path(filename).is_file()

        (graph,) = fake_graphviz["graphs"]
        assert graph.graph == {"nodesep": 10.0, "dpi": 300, "overlap": "false"}

        r, g, b = EL_COLORS["Jmol"]["Mo"]
        mo_node = graph.nodes[0]
        assert mo_node["label"] == "Mo(0)"
        assert mo_node["fillcolor"] == f"#{r:02x}{g:02x}{b:02x}"
        assert mo_node["fontcolor"] in {"#000000", "#ffffff"}
        assert mo_node["shape"] == "circle"
        assert graph.nodes[1]["label"] == "S(1)"

        edges = list(graph.edges(data=True))
        assert len(edges) == 6
        for u, v, data in edges:
            assert data["color_uv"] == f"{graph.nodes[u]['fillcolor']};0.5:{graph.nodes[v]['fillcolor']};0.5"
            assert data["label"] == "2.42 "
            if data["to_jimage"] == (0, 0, 0):
                assert data["style"] == "solid"
                assert data["headlabel"] == ""
                assert data["arrowhead"] == "none"
            else:
                assert data["style"] == "dashed"
                assert data["headlabel"] == f"to {data['to_jimage']}"
                assert data["arrowhead"] == "normal"

    def test_draw_graph_to_file_hides_image_edges_and_highlights_diff(self, fake_graphviz):
        self.mos2_sg.draw_graph_to_file(f"{self.tmp_path}/MoS2.pdf")
        (graph,) = fake_graphviz["graphs"]
        assert {data["to_jimage"] for _, _, data in graph.edges(data=True)} == {(0, 0, 0)}
        assert graph.number_of_edges() == 2
        assert all(data["color_uv"] == "#000000" for _, _, data in graph.edges(data=True))
        assert all(data["label"] == "" for _, data in graph.nodes(data=True))
        # .dot file is cleaned up unless keep_dot=True
        assert not Path(f"{self.tmp_path}/MoS2.dot").exists()
        assert fake_graphviz["popen"].calls[-1][:3] == ["fdp", "-T", "pdf"]

        other = copy.deepcopy(self.mos2_sg)
        other.break_edge(0, 1, to_jimage=(0, 0, 0))
        self.mos2_sg.draw_graph_to_file(f"{self.tmp_path}/MoS2_diff.pdf", diff=other, hide_image_edges=False)
        graph = fake_graphviz["graphs"][-1]
        colors = {(u, v, data["to_jimage"]): data["color_uv"] for u, v, data in graph.edges(data=True)}
        assert colors.pop((0, 1, (0, 0, 0))) == "#ff0000"
        assert set(colors.values()) == {"#000000"}

    def test_draw_graph_to_file_nonzero_exit(self, fake_graphviz):
        fake_graphviz["popen"].returncode = 1
        with pytest.raises(RuntimeError, match="fdp exited with return code 1"):
            self.mos2_sg.draw_graph_to_file(f"{self.tmp_path}/MoS2.pdf")

    def test_draw_graph_to_file_hide_unconnected_nodes(self, fake_graphviz):
        structure = Structure(Lattice.cubic(5), ["H", "He", "Li"], [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0]])
        sg = StructureGraph.from_empty_graph(structure)
        sg.add_edge(0, 1, to_jimage=(0, 0, 0))
        sg.add_edge(2, 2, to_jimage=(1, 0, 0))
        sg.draw_graph_to_file(f"{self.tmp_path}/graph.pdf", hide_unconnected_nodes=True)
        (graph,) = fake_graphviz["graphs"]
        assert set(graph.nodes) == {0, 1}


class TestMoleculeGraph:
    def setup_method(self):
        cyclohexene_xyz = f"{TEST_DIR}/cyclohexene.xyz"
        cyclohexene = Molecule.from_file(cyclohexene_xyz)
        self.cyclohexene = MoleculeGraph.from_empty_graph(
            cyclohexene, edge_weight_name="strength", edge_weight_units=""
        )
        self.cyclohexene.add_edge(0, 1, weight=1.0)
        self.cyclohexene.add_edge(1, 2, weight=1.0)
        self.cyclohexene.add_edge(2, 3, weight=2.0)
        self.cyclohexene.add_edge(3, 4, weight=1.0)
        self.cyclohexene.add_edge(4, 5, weight=1.0)
        self.cyclohexene.add_edge(5, 0, weight=1.0)
        self.cyclohexene.add_edge(0, 6, weight=1.0)
        self.cyclohexene.add_edge(0, 7, weight=1.0)
        self.cyclohexene.add_edge(1, 8, weight=1.0)
        self.cyclohexene.add_edge(1, 9, weight=1.0)
        self.cyclohexene.add_edge(2, 10, weight=1.0)
        self.cyclohexene.add_edge(3, 11, weight=1.0)
        self.cyclohexene.add_edge(4, 12, weight=1.0)
        self.cyclohexene.add_edge(4, 13, weight=1.0)
        self.cyclohexene.add_edge(5, 14, weight=1.0)
        self.cyclohexene.add_edge(5, 15, weight=1.0)

        butadiene = Molecule.from_file(f"{TEST_DIR}/butadiene.xyz")
        self.butadiene = MoleculeGraph.from_empty_graph(butadiene, edge_weight_name="strength", edge_weight_units="")
        self.butadiene.add_edge(0, 1, weight=2.0)
        self.butadiene.add_edge(1, 2, weight=1.0)
        self.butadiene.add_edge(2, 3, weight=2.0)
        self.butadiene.add_edge(0, 4, weight=1.0)
        self.butadiene.add_edge(0, 5, weight=1.0)
        self.butadiene.add_edge(1, 6, weight=1.0)
        self.butadiene.add_edge(2, 7, weight=1.0)
        self.butadiene.add_edge(3, 8, weight=1.0)
        self.butadiene.add_edge(3, 9, weight=1.0)

        ethylene = Molecule.from_file(f"{TEST_DIR}/ethylene.xyz")
        self.ethylene = MoleculeGraph.from_empty_graph(ethylene, edge_weight_name="strength", edge_weight_units="")
        self.ethylene.add_edge(0, 1, weight=2.0)
        self.ethylene.add_edge(0, 2, weight=1.0)
        self.ethylene.add_edge(0, 3, weight=1.0)
        self.ethylene.add_edge(1, 4, weight=1.0)
        self.ethylene.add_edge(1, 5, weight=1.0)

        self.pc = Molecule.from_file(f"{TEST_DIR}/PC.xyz")
        self.pc_edges = [
            [5, 10],
            [5, 12],
            [5, 11],
            [5, 3],
            [3, 7],
            [3, 4],
            [3, 0],
            [4, 8],
            [4, 9],
            [4, 1],
            [6, 1],
            [6, 0],
            [6, 2],
        ]
        self.pc_frag1 = Molecule.from_file(f"{TEST_DIR}/PC_frag1.xyz")
        self.pc_frag1_edges = [[0, 2], [4, 2], [2, 1], [1, 3]]
        self.tfsi = Molecule.from_file(f"{TEST_DIR}/TFSI.xyz")
        self.tfsi_edges = (
            [14, 1],
            [1, 4],
            [1, 5],
            [1, 7],
            [7, 11],
            [7, 12],
            [7, 13],
            [14, 0],
            [0, 2],
            [0, 3],
            [0, 6],
            [6, 8],
            [6, 9],
            [6, 10],
        )

    def test_bond_dissociation_reference_files_exist(self):
        """The reference JSONs read by test_construction must exist.

        test_construction is importorskip-gated on openbabel, which no CI
        leg installs, so when these files were dropped in the repo split
        (#141) nothing failed. This guard runs independently of openbabel
        so the same class of split-artifact regression cannot recur
        silently.
        """
        for name in ("pc_frag1_mg.json", "pc_mg.json"):
            assert (Path(TEST_FILES_DIR) / "analysis/bond_dissociation" / name).is_file(), (
                f"missing test reference file: analysis/bond_dissociation/{name} "
                f"(needed by test_construction, which CI cannot run without openbabel)"
            )

    def test_construction(self):
        pytest.importorskip("openbabel")
        edges_frag = {(edge[0], edge[1]): {"weight": 1.0} for edge in self.pc_frag1_edges}
        mol_graph = MoleculeGraph.from_edges(self.pc_frag1, edges_frag)
        # dumpfn(mol_graph.as_dict(), f"{TEST_FILES_DIR}/analysis/bond_dissociation/pc_frag1_mg.json")
        ref_mol_graph = loadfn(f"{TEST_FILES_DIR}/analysis/bond_dissociation/pc_frag1_mg.json")
        assert mol_graph == ref_mol_graph
        assert mol_graph.graph.adj == ref_mol_graph.graph.adj
        for node in mol_graph.graph.nodes:
            assert mol_graph.graph.nodes[node]["specie"] == ref_mol_graph.graph.nodes[node]["specie"]
            for ii in range(3):
                assert mol_graph.graph.nodes[node]["coords"][ii] == ref_mol_graph.graph.nodes[node]["coords"][ii]

        edges_pc = {(edge[0], edge[1]): {"weight": 1.0} for edge in self.pc_edges}
        mol_graph = MoleculeGraph.from_edges(self.pc, edges_pc)
        # dumpfn(mol_graph.as_dict(), f"{TEST_FILES_DIR}/analysis/bond_dissociation/pc_mg.json")
        ref_mol_graph = loadfn(f"{TEST_FILES_DIR}/analysis/bond_dissociation/pc_mg.json")
        assert mol_graph == ref_mol_graph
        assert mol_graph.graph.adj == ref_mol_graph.graph.adj
        for node in mol_graph.graph:
            assert mol_graph.graph.nodes[node]["specie"] == ref_mol_graph.graph.nodes[node]["specie"]
            for ii in range(3):
                assert mol_graph.graph.nodes[node]["coords"][ii] == ref_mol_graph.graph.nodes[node]["coords"][ii]

        mol_graph_edges = MoleculeGraph.from_edges(self.pc, edges=edges_pc)
        mol_graph_strat = MoleculeGraph.from_local_env_strategy(self.pc, OpenBabelNN())

        assert mol_graph_edges.isomorphic_to(mol_graph_strat)

        # Check error message on using inappropriate strategy for molecules
        strategy = VoronoiNN()
        with pytest.raises(
            ValueError,
            match=re.escape(f"{strategy=} is not designed for use with molecules! Choose another strategy"),
        ):
            MoleculeGraph.from_local_env_strategy(self.pc, strategy)

    def test_properties(self):
        assert self.cyclohexene.name == "bonds"
        assert self.cyclohexene.edge_weight_name == "strength"
        assert self.cyclohexene.edge_weight_unit == ""
        assert self.cyclohexene.get_coordination_of_site(0) == 4
        assert self.cyclohexene.get_coordination_of_site(2) == 3
        assert self.cyclohexene.get_coordination_of_site(15) == 1
        assert len(self.cyclohexene.get_connected_sites(0)) == 4
        assert isinstance(self.cyclohexene.get_connected_sites(0)[0].site, Site)
        assert str(self.cyclohexene.get_connected_sites(0)[0].site.specie) == "H"

    def test_set_node_attributes(self):
        self.ethylene.set_node_attributes()

        specie = nx.get_node_attributes(self.ethylene.graph, "specie")
        coords = nx.get_node_attributes(self.ethylene.graph, "coords")
        properties = nx.get_node_attributes(self.ethylene.graph, "properties")

        for idx, site in enumerate(self.ethylene.molecule):
            assert str(specie[idx]) == str(site.specie)
            assert coords[idx][0] == site.coords[0]
            assert coords[idx][1] == site.coords[1]
            assert coords[idx][2] == site.coords[2]
            assert properties[idx] == site.properties

    def test_coordination(self):
        molecule = Molecule(["C", "C"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        mg = MoleculeGraph.from_empty_graph(molecule)
        assert mg.get_coordination_of_site(0) == 0

        assert self.cyclohexene.get_coordination_of_site(0) == 4

    def test_edge_editing(self):
        cyclohexene = copy.deepcopy(self.cyclohexene)
        cyclohexene.alter_edge(0, 1, new_weight=0.0, new_edge_properties={"foo": "bar"})
        new_edge = cyclohexene.graph.get_edge_data(0, 1)[0]
        assert new_edge["weight"] == approx(0.0)
        assert new_edge["foo"] == "bar"

        cyclohexene.break_edge(0, 1)
        assert cyclohexene.graph.get_edge_data(0, 1) is None

        # Replace the now-broken edge
        cyclohexene.add_edge(0, 1, weight=1.0)

    def test_insert_remove(self):
        mol_copy = copy.deepcopy(self.ethylene.molecule)
        eth_copy = copy.deepcopy(self.ethylene)

        # Ensure that insert_node appropriately wraps Molecule.insert()
        mol_copy.insert(1, "O", [0.5, 0.5, 0.5])
        eth_copy.insert_node(1, "O", [0.5, 0.5, 0.5])
        assert mol_copy == eth_copy.molecule

        # Test that removal is also equivalent between Molecule and MoleculeGraph.molecule
        mol_copy.remove_sites([1])
        eth_copy.remove_nodes([1])
        assert mol_copy == eth_copy.molecule

        eth_copy.insert_node(
            1,
            "O",
            [0.5, 0.5, 0.5],
            edges=[{"from_index": 1, "to_index": 2}, {"from_index": 1, "to_index": 3}],
        )
        assert eth_copy.get_coordination_of_site(1) == 2

        # Test that MoleculeGraph.graph is correctly updated
        eth_copy.remove_nodes([1, 2])
        assert eth_copy.graph.number_of_nodes() == 5
        assert eth_copy.graph.number_of_edges() == 2

    def test_get_disconnected(self):
        disconnected = Molecule(
            ["C", "H", "H", "H", "H", "He"],
            [
                [0, 0, 0],
                [-0.3633, -0.5138, -0.8900],
                [1.0900, 0, 0],
                [-0.3633, 1.0277, 0],
                [-0.3633, -0.5138, -0.8900],
                [5, 5, 5],
            ],
        )

        no_he = Molecule(
            ["C", "H", "H", "H", "H"],
            [
                [0, 0, 0],
                [-0.3633, -0.5138, -0.8900],
                [1.0900, 0, 0],
                [-0.3633, 1.0277, 0],
                [-0.3633, -0.5138, -0.8900],
            ],
        )

        just_he = Molecule(["He"], [[5, 5, 5]])

        dis_mg = MoleculeGraph.from_empty_graph(disconnected)
        dis_mg.add_edge(0, 1)
        dis_mg.add_edge(0, 2)
        dis_mg.add_edge(0, 3)
        dis_mg.add_edge(0, 4)

        fragments = dis_mg.get_disconnected_fragments()
        fragments_2, index_map = dis_mg.get_disconnected_fragments(return_index_map=True)
        assert list(map(str, fragments)) == list(map(str, fragments_2))
        assert len(fragments) == 2
        assert fragments[0].molecule == no_he
        assert fragments[1].molecule == just_he
        assert index_map == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

        con_mg = MoleculeGraph.from_empty_graph(no_he)
        con_mg.add_edge(0, 1)
        con_mg.add_edge(0, 2)
        con_mg.add_edge(0, 3)
        con_mg.add_edge(0, 4)
        # make sure get_disconnected_fragments() only returns fragments even if return_index_map=True
        # when the graph is weakly connected
        fragments = con_mg.get_disconnected_fragments(return_index_map=True)
        assert len(fragments) == 1

    def test_split(self):
        bonds = [(0, 1), (4, 5)]
        alterations = {
            (2, 3): {"weight": 1.0},
            (0, 5): {"weight": 2.0},
            (1, 2): {"weight": 2.0},
            (3, 4): {"weight": 2.0},
        }
        # Perform retro-Diels-Alder reaction - turn product into reactants
        reactants = self.cyclohexene.split_molecule_subgraphs(bonds, allow_reverse=True, alterations=alterations)
        assert isinstance(reactants, list)

        reactants = sorted(reactants, key=len)
        # After alterations, reactants should be ethylene and butadiene
        assert reactants[0] == self.ethylene
        assert reactants[1] == self.butadiene

        with pytest.raises(
            MolGraphSplitError,
            match=re.escape("Cannot split molecule; MoleculeGraph is still connected."),
        ):
            self.cyclohexene.split_molecule_subgraphs([(0, 1)])

        # Test naive charge redistribution
        hydroxide = Molecule(["O", "H"], [[0, 0, 0], [0.5, 0.5, 0.5]], charge=-1)
        oh_mg = MoleculeGraph.from_empty_graph(hydroxide)

        oh_mg.add_edge(0, 1)

        new_mgs = oh_mg.split_molecule_subgraphs([(0, 1)])
        for mg in new_mgs:
            if str(mg.molecule[0].specie) == "O":
                assert mg.molecule.charge == -1
            else:
                assert mg.molecule.charge == 0

        # Trying to test to ensure that remapping of nodes to atoms works
        diff_species = Molecule(
            ["C", "I", "Cl", "Br", "F"],
            [
                [0.8314, -0.2682, -0.9102],
                [1.3076, 1.3425, -2.2038],
                [-0.8429, -0.7410, -1.1554],
                [1.9841, -1.7636, -1.2953],
                [1.0098, 0.1231, 0.3916],
            ],
        )

        diff_spec_mg = MoleculeGraph.from_empty_graph(diff_species)
        diff_spec_mg.add_edge(0, 1)
        diff_spec_mg.add_edge(0, 2)
        diff_spec_mg.add_edge(0, 3)
        diff_spec_mg.add_edge(0, 4)

        for idx in range(1, 5):
            bond = (0, idx)

            split_mgs = diff_spec_mg.split_molecule_subgraphs([bond])
            for split_mg in split_mgs:
                species = nx.get_node_attributes(split_mg.graph, "specie")

                for j in range(len(split_mg.graph.nodes)):
                    atom = split_mg.molecule[j]
                    assert species[j] == str(atom.specie)

    def test_build_unique_fragments(self):
        edges = {(edge[0], edge[1]): None for edge in self.pc_edges}
        mol_graph = MoleculeGraph.from_edges(self.pc, edges)
        unique_fragment_dict = mol_graph.build_unique_fragments()
        unique_fragments = [fragment for key in unique_fragment_dict for fragment in unique_fragment_dict[key]]
        assert len(unique_fragments) == 295
        nm = iso.categorical_node_match("specie", "ERROR")
        for ii in range(295):
            # Test that each fragment is unique
            for jj in range(ii + 1, 295):
                assert not nx.is_isomorphic(
                    unique_fragments[ii].graph,
                    unique_fragments[jj].graph,
                    node_match=nm,
                )

            # Test that each fragment correctly maps between Molecule and graph
            assert len(unique_fragments[ii].molecule) == len(unique_fragments[ii].graph.nodes)
            species = nx.get_node_attributes(unique_fragments[ii].graph, "specie")
            coords = nx.get_node_attributes(unique_fragments[ii].graph, "coords")

            mol = unique_fragments[ii].molecule
            for idx, site in enumerate(mol):
                assert str(species[idx]) == str(site.specie)
                assert all(coords[idx] == site.coords)

            # Test that each fragment is connected
            assert nx.is_connected(unique_fragments[ii].graph.to_undirected())

    def test_find_rings(self):
        rings = self.cyclohexene.find_rings(including=[0])
        assert sorted(rings[0]) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        no_rings = self.butadiene.find_rings()
        assert no_rings == []

    def test_isomorphic(self):
        ethyl_xyz_path = f"{TEST_DIR}/ethylene.xyz"
        ethylene = Molecule.from_file(ethyl_xyz_path)
        # swap carbons
        ethylene[0], ethylene[1] = ethylene[1], ethylene[0]

        edges = {
            (0, 1): {"weight": 2},
            (1, 2): {"weight": 1},
            (1, 3): {"weight": 1},
            (0, 4): {"weight": 1},
            (0, 5): {"weight": 1},
        }

        ethylene_graph = MoleculeGraph.from_edges(ethylene, edges)
        # If they are equal, they must also be isomorphic
        assert self.ethylene.isomorphic_to(ethylene_graph)
        assert not self.butadiene.isomorphic_to(self.ethylene)

        # check fix in https://github.com/materialsproject/pymatgen/pull/3221
        # by comparing graph with equal nodes but different edges
        edges[1, 4] = {"weight": 2}
        assert not self.ethylene.isomorphic_to(MoleculeGraph.from_edges(ethylene, edges))

    def test_substitute(self):
        molecule = FunctionalGroups["methyl"]
        mol_graph = MoleculeGraph.from_edges(
            molecule,
            {(0, 1): {"weight": 1}, (0, 2): {"weight": 1}, (0, 3): {"weight": 1}},
        )

        eth_mol = copy.deepcopy(self.ethylene)
        eth_str = copy.deepcopy(self.ethylene)
        # Ensure that strings and molecules lead to equivalent substitutions
        eth_mol.substitute_group(5, molecule, MinimumDistanceNN)
        eth_str.substitute_group(5, "methyl", MinimumDistanceNN)
        assert eth_mol == eth_str

        graph_dict = {
            (0, 1): {"weight": 1.0},
            (0, 2): {"weight": 1.0},
            (0, 3): {"weight": 1.0},
        }
        eth_mg = copy.deepcopy(self.ethylene)
        eth_graph = copy.deepcopy(self.ethylene)

        # Check that MoleculeGraph input is handled properly
        eth_graph.substitute_group(5, molecule, MinimumDistanceNN, graph_dict=graph_dict)
        eth_mg.substitute_group(5, mol_graph, MinimumDistanceNN)
        assert eth_graph.graph.get_edge_data(5, 6)[0]["weight"] == approx(1.0)
        assert eth_mg == eth_graph

    def test_replace(self):
        eth_copy_sub = copy.deepcopy(self.ethylene)
        eth_copy_repl = copy.deepcopy(self.ethylene)
        # First, perform a substitution as above
        eth_copy_sub.substitute_group(5, "methyl", MinimumDistanceNN)
        eth_copy_repl.replace_group(5, "methyl", MinimumDistanceNN)
        # Test that replacement on a terminal atom is equivalent to substitution
        assert eth_copy_repl.molecule == eth_copy_sub.molecule
        assert eth_copy_repl == eth_copy_sub

        # Methyl carbon should have coordination 4
        assert eth_copy_repl.get_coordination_of_site(5) == 4
        # Now swap one functional group for another
        eth_copy_repl.replace_group(5, "amine", MinimumDistanceNN)
        assert list(map(str, eth_copy_repl.molecule.species)) == [
            "C",
            "C",
            "H",
            "H",
            "H",
            "N",
            "H",
            "H",
        ]
        assert len(eth_copy_repl.graph.nodes) == 8
        # Amine nitrogen should have coordination 3
        assert eth_copy_repl.get_coordination_of_site(5) == 3

    def test_as_from_dict(self):
        dct = self.cyclohexene.as_dict()
        mg = MoleculeGraph.from_dict(dct)
        dct2 = mg.as_dict()
        assert str(dct) == str(dct2)

    def test_sort(self):
        sg = copy.deepcopy(self.ethylene)
        # insert an unsorted edge, don't use sg.add_edge as it auto-sorts

        assert list(sg.graph.edges) == [
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
        ]
        sg.sort()
        assert list(sg.graph.edges) == [
            (4, 5, 0),
            (0, 4, 0),
            (1, 4, 0),
            (2, 5, 0),
            (3, 5, 0),
        ]

    def test_from_empty_graph_requires_units(self):
        with pytest.raises(ValueError, match="Please specify units associated with your edge weights"):
            MoleculeGraph.from_empty_graph(self.ethylene.molecule, edge_weight_name="strength")

    def test_init_from_molecule_graph(self):
        mg_copy = MoleculeGraph(self.ethylene)
        assert mg_copy == self.ethylene

    def test_init_restores_image_tuples(self):
        mg = copy.deepcopy(self.ethylene)
        mg.alter_edge(0, 1, new_edge_properties={"to_jimage": [0, 0, 0], "from_jimage": [0, 0, 0]})
        restored = MoleculeGraph.from_dict(mg.as_dict())
        edge = restored.graph.get_edge_data(0, 1)[0]
        assert edge["to_jimage"] == (0, 0, 0)
        assert edge["from_jimage"] == (0, 0, 0)
        assert isinstance(edge["to_jimage"], tuple)
        assert edge["weight"] == approx(2.0)

    @pytest.mark.parametrize(
        ("edges", "match"),
        [
            ({0: None}, r"Edges must be given as \(from_index, to_index\) tuples"),
            ({(0, 42): None}, "Edges cannot be added if nodes are not present"),
        ],
    )
    def test_from_edges_invalid(self, edges, match):
        with pytest.raises(ValueError, match=match):
            MoleculeGraph.from_edges(self.ethylene.molecule, edges)

    def test_from_local_env_strategy_without_boxing(self):
        strategy = CovalentBondNN()
        assert not strategy.extend_structure_molecules
        mg = MoleculeGraph.from_local_env_strategy(self.ethylene.molecule, strategy)
        assert mg.edge_weight_name == "weight"
        assert sorted(mg.graph.edges()) == [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
        assert mg.isomorphic_to(self.ethylene)

    def test_from_local_env_strategy_rejects_structure_only_strategy(self):
        with pytest.raises(ValueError, match="is not designed for use with molecules"):
            MoleculeGraph.from_local_env_strategy(self.ethylene.molecule, VoronoiNN())

    def test_add_edge_duplicate_warns(self):
        mg = copy.deepcopy(self.ethylene)
        with pytest.warns(UserWarning, match="Trying to add an edge that already exists from site 0 to site 1"):
            mg.add_edge(1, 0, weight=5.0)
        assert mg.graph.number_of_edges() == 5
        assert mg.graph.get_edge_data(0, 1)[0]["weight"] == approx(2.0)

    def test_insert_node_invalid_edges(self):
        mg = copy.deepcopy(self.ethylene)
        with pytest.raises(RuntimeError, match="Some edges are invalid"):
            mg.insert_node(1, "O", [0.5, 0.5, 0.5], edges=[{"from_index": 1}])

    def test_alter_edge_missing(self):
        with pytest.raises(ValueError, match="Edge between 2 and 3 cannot be altered; no edge exists"):
            self.ethylene.alter_edge(2, 3, new_weight=1.0)

    def test_break_edge_reverse(self):
        mg = copy.deepcopy(self.ethylene)
        with pytest.raises(ValueError, match="Edge cannot be broken between 1 and 0; no edge exists"):
            mg.break_edge(1, 0)
        mg.break_edge(1, 0, allow_reverse=True)
        assert mg.graph.get_edge_data(0, 1) is None
        assert mg.graph.number_of_edges() == 4

    def test_split_alterations_without_weight(self):
        fragments = self.ethylene.split_molecule_subgraphs(
            [(0, 1)], alterations={(0, 2): {"bond_type": "single"}, (1, 4): {"weight": 3.0}}
        )
        assert len(fragments) == 2
        frag_0, frag_1 = fragments
        assert frag_0.graph.get_edge_data(0, 1)[0] == {"weight": 1.0, "bond_type": "single"}
        assert frag_1.graph.get_edge_data(0, 1)[0] == {"weight": 3.0}

    def test_get_disconnected_fragments_site_properties(self):
        mol = Molecule(
            ["C", "H", "He"],
            [[0, 0, 0], [1.09, 0, 0], [5, 5, 5]],
            charge=1,
            site_properties={"magmom": [0.1, 0.2, 0.3]},
        )
        mg = MoleculeGraph.from_empty_graph(mol)
        mg.add_edge(0, 1)
        frag_ch, frag_he = mg.get_disconnected_fragments()
        assert frag_ch.molecule.site_properties == {"magmom": [0.1, 0.2]}
        assert frag_he.molecule.site_properties == {"magmom": [0.3]}
        assert frag_ch.molecule.charge == 1
        assert frag_he.molecule.charge == 0

    def test_get_disconnected_fragments_partial_site_properties(self):
        mol = Molecule(["C", "H", "He"], [[0, 0, 0], [1.09, 0, 0], [5, 5, 5]])
        mol.insert(3, "H", [-1.09, 0, 0], properties={"magmom": 1.0})
        mg = MoleculeGraph.from_empty_graph(mol)
        mg.add_edge(0, 1)
        mg.add_edge(0, 3)
        frag_ch2, _frag_he = mg.get_disconnected_fragments()
        # "magmom" is only set on one of the three sites, so it must be dropped
        assert "magmom" not in frag_ch2.molecule.site_properties

    def test_substitute_group_unknown(self):
        mg = copy.deepcopy(self.ethylene)
        with pytest.raises(RuntimeError, match="Can't find functional group in list"):
            mg.substitute_group(5, "not-a-functional-group", MinimumDistanceNN)

    def test_replace_group_in_ring(self):
        with pytest.raises(RuntimeError, match="cannot occur at an atom within a ring structure"):
            self.cyclohexene.replace_group(0, "methyl", MinimumDistanceNN)

    def test_find_rings_all(self):
        rings = self.cyclohexene.find_rings()
        assert len(rings) == 1
        assert sorted(rings[0]) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

    def test_str_repr_with_weights(self):
        eth_str = str(self.ethylene)
        assert eth_str.startswith("Molecule Graph\nMolecule: \n")
        assert "Graph: bonds" in eth_str
        assert "to_image      strength\n" in eth_str
        assert "   0     1  (0, 0, 0)     2.000e+00" in eth_str
        assert eth_str.count("1.000e+00") == 4

        eth_repr = repr(self.ethylene)
        assert eth_repr.startswith("Molecule Graph\nMolecule: \n")
        assert repr(self.ethylene.molecule) in eth_repr
        assert "   0     1  (0, 0, 0)     2.000e+00" in eth_repr

        mg = MoleculeGraph.from_empty_graph(self.ethylene.molecule, edge_weight_name="length", edge_weight_units="Å")
        mg.add_edge(0, 1, weight=1.33)
        assert "length (Å)" in str(mg)
        assert "1.330e+00" in str(mg)

        unweighted = MoleculeGraph.from_empty_graph(self.ethylene.molecule)
        unweighted.add_edge(0, 1)
        assert str(unweighted).endswith("----  ----  ------------\n   0     1  (0, 0, 0)   \n")

    def test_len(self):
        assert len(self.ethylene) == 6
        assert len(self.cyclohexene) == 16

    def test_eq_edge_cases(self):
        assert self.ethylene.__eq__("not a graph") is NotImplemented
        assert self.ethylene != "not a graph"
        # molecules whose sites cannot be mapped onto each other
        assert self.ethylene != self.butadiene

    def test_isomorphic_to_different_formula(self):
        # same number of atoms but different composition
        mol = Molecule(["C", "C", "H", "H", "H", "O"], self.ethylene.molecule.cart_coords)
        mg = MoleculeGraph.from_empty_graph(mol)
        assert not self.ethylene.isomorphic_to(mg)

    def test_isomorphic_helper(self):
        graph = self.ethylene.graph
        assert _isomorphic(graph, copy.deepcopy(graph))

        # different number of nodes
        assert not _isomorphic(graph, self.butadiene.graph)

        # same nodes, different number of edges
        fewer_edges = copy.deepcopy(graph)
        fewer_edges.remove_edge(0, 1)
        assert not _isomorphic(graph, fewer_edges)

        # same node and edge counts, but different species
        other_species = copy.deepcopy(graph)
        other_species.nodes[2]["specie"] = "F"
        assert not _isomorphic(graph, other_species)

    def test_diff_different_molecules(self):
        with pytest.raises(ValueError, match="Meaningless to compare MoleculeGraphs"):
            self.ethylene.diff(self.butadiene)

    def test_diff_strict(self):
        diff = self.ethylene.diff(copy.deepcopy(self.ethylene))
        assert diff["self"] == diff["other"] == set()
        assert diff["dist"] == approx(0.0)

    def test_diff_non_strict(self):
        diff = self.ethylene.diff(self.butadiene, strict=False)
        assert diff["self"] == set()
        assert diff["both"] == {("C", "C"), ("C", "H")}

    def test_draw_graph_to_file_requires_graphviz(self, monkeypatch):
        monkeypatch.setattr(graphs_module, "which", lambda _algo: None)
        with pytest.raises(RuntimeError, match="requires GraphViz binaries to be in the path"):
            self.ethylene.draw_graph_to_file("ethylene.pdf")

    def test_draw_graph_to_file(self, fake_graphviz, tmp_path):
        self.ethylene.draw_graph_to_file(f"{tmp_path}/ethylene.pdf", node_labels=True)
        (graph,) = fake_graphviz["graphs"]
        assert graph.nodes[0]["label"] == "C(0)"
        assert graph.number_of_edges() == 5

    def test_draw_graph_to_file_styling(self, fake_graphviz, monkeypatch, tmp_path):
        filename = f"{tmp_path}/ethylene.svg"
        self.ethylene.draw_graph_to_file(
            filename,
            edge_colors=True,
            node_labels=True,
            weight_labels=True,
            image_labels=True,
            keep_dot=True,
            algo="neato",
        )
        assert fake_graphviz["popen"].calls == [["neato", "-T", "svg", f"{tmp_path}/ethylene.dot"]]
        assert Path(f"{tmp_path}/ethylene.dot").is_file()

        (graph,) = fake_graphviz["graphs"]
        r, g, b = EL_COLORS["VESTA"]["C"]
        assert graph.nodes[0]["fillcolor"] == f"#{r:02x}{g:02x}{b:02x}"
        assert graph.nodes[0]["label"] == "C(0)"
        assert graph.nodes[2]["label"] == "H(2)"
        assert graph.number_of_edges() == 5
        for u, v, data in graph.edges(data=True):
            assert data["style"] == "solid"
            assert data["headlabel"] == ""
            assert data["arrowhead"] == "none"
            assert data["color_uv"] == f"{graph.nodes[u]['fillcolor']};0.5:{graph.nodes[v]['fillcolor']};0.5"
            assert data["label"] == ("2.00 " if (u, v) == (0, 1) else "1.00 ")

        # defaults: unlabeled nodes, black edges, .dot removed; isolated nodes hidden
        mol = Molecule(["C", "O", "He"], [[0, 0, 0], [1.2, 0, 0], [5, 5, 5]])
        mg = MoleculeGraph.from_empty_graph(mol)
        mg.add_edge(0, 1)
        mg.draw_graph_to_file(f"{tmp_path}/co.pdf", hide_unconnected_nodes=False)
        graph = fake_graphviz["graphs"][-1]
        assert set(graph.nodes) == {0, 1, 2}
        assert all(data["label"] == "" for _, data in graph.nodes(data=True))
        assert [data["color_uv"] for _, _, data in graph.edges(data=True)] == ["#000000"]
        assert not Path(f"{tmp_path}/co.dot").exists()

        fake_graphviz["popen"].returncode = 2
        with pytest.raises(RuntimeError, match="fdp exited with return code 2"):
            mg.draw_graph_to_file(f"{tmp_path}/co.pdf")
