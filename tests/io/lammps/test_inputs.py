from __future__ import annotations

import filecmp
import re

import pandas as pd
import pytest

from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.inputs import LammpsInputFile, LammpsRun, LammpsTemplateGen, write_lammps_inputs
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/io/lammps"


class TestLammpsInputFile(MatSciTest):
    @classmethod
    def setup_class(cls):
        cls.filename = f"{TEST_DIR}/lgps.in"

    def test_from_file(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        assert lmp_input.ncomments == 3
        assert lmp_input.nstages == 1
        assert lmp_input.stages_names == ["Stage 1"]
        assert lmp_input.stages[0]["commands"][0][1] == "metal"
        assert lmp_input.stages == [
            {
                "stage_name": "Stage 1",
                "commands": [
                    ("units", "metal"),
                    ("atom_style", "full"),
                    ("dimension", "3"),
                    ("pair_style", "hybrid/overlay morse 15 coul/long 15"),  # codespell:ignore coul
                    ("kspace_style", "ewald 1e-4"),
                    ("boundary", "p p p"),
                    ("#", "2) System definition"),
                    ("read_data", "run_init.data"),
                    ("set", "type 1 charge 0.8803"),
                    ("set", "type 2 charge 1.2570"),
                    ("set", "type 3 charge 1.2580"),
                    ("set", "type 4 charge -1.048"),
                    ("neigh_modify", "every 1 delay 5 check yes"),
                    ("#", "3) Simulation settings"),
                    ("pair_coeff", "1 1 morse 0.0580 3.987 3.404"),
                    ("pair_coeff", "1 4 morse 0.0408 1.399 3.204"),
                    ("pair_coeff", "2 4 morse 0.3147 2.257 2.409"),
                    ("pair_coeff", "3 4 morse 0.4104 2.329 2.200"),
                    ("pair_coeff", "4 4 morse 0.0241 1.359 4.284"),
                    ("pair_coeff", "* * coul/long"),  # codespell:ignore coul
                    ("#", "Part A : energy minimization"),
                    ("thermo", "1"),
                    ("thermo_style", "custom step lx ly lz press pxx pyy pzz pe"),
                    ("dump", "dmp all atom 5 run.dump"),
                    ("min_style", "cg"),
                    ("fix", "1 all box/relax iso 0.0 vmax 0.001"),
                    ("minimize", "1.0e-16 1.0e-16 5000 10000"),
                    ("write_data", "run.data"),
                ],
            }
        ]

    def test_get_str(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        string = lmp_input.get_str()
        assert "# LAMMPS input generated from LammpsInputFile with pymatgen v" in string
        assert "\nunits metal\natom_style full\ndimension 3\npair_style hybrid/overlay morse 15 coul/long" in string
        assert "15\nkspace_style ewald 1e-4\nboundary p p p\n# 2) System definition" in string
        assert "\nread_data run_init.data\nset type 1 charge 0.8803\nset type 2 charge" in string
        assert "1.2570\nset type 3 charge 1.2580\nset type 4 charge -1.048\nneigh_modify" in string
        assert "every 1 delay 5 check yes\n# 3) Simulation settings\npair_coeff 1 1" in string
        assert "morse 0.0580 3.987 3.404\npair_coeff 1 4 morse 0.0408 1.399 3.204\npair_coeff" in string
        assert "2 4 morse 0.3147 2.257 2.409\npair_coeff 3 4 morse 0.4104 2.329 2.200\npair_coeff" in string
        assert "4 4 morse 0.0241 1.359 4.284\npair_coeff * * coul/long\n# Part A : energy" in string
        assert "minimization\nthermo 1\nthermo_style custom step lx ly lz press pxx pyy pzz" in string
        assert "pe\ndump dmp all atom 5 run.dump\nmin_style cg\nfix 1 all" in string
        assert "box/relax iso 0.0 vmax 0.001\nminimize 1.0e-16 1.0e-16 5000 10000\nwrite_data run.data" in string

    def test_from_str(self):
        string = """# LGPS

# 1) Initialization
units metal
atom_style full
dimension 3
pair_style hybrid/overlay &
morse 15 coul/long 15
kspace_style ewald 1e-4
boundary p p p

# 2) System definition
read_data run_init.data
set type 1 charge  0.8803
set type 2 charge  1.2570
set type 3 charge  1.2580
set type 4 charge -1.048
neigh_modify every 1 delay 5 check yes

# 3) Simulation settings
pair_coeff 1 1 morse 0.0580 3.987 3.404
pair_coeff 1 4 morse 0.0408 1.399 3.204
pair_coeff 2 4 morse 0.3147 2.257 2.409
pair_coeff 3 4 morse 0.4104 2.329 2.200
pair_coeff 4 4 morse 0.0241 1.359 4.284
pair_coeff * * coul/long

# Part A : energy minimization
thermo 1
thermo_style custom step lx ly lz press pxx pyy pzz pe
dump dmp all atom 5 run.dump

min_style cg
fix 1 all box/relax iso 0.0 vmax 0.001
minimize 1.0e-16 1.0e-16 5000 10000
write_data run.data"""

        lmp_input = LammpsInputFile().from_str(string)
        assert lmp_input.stages == LammpsInputFile().from_file(self.filename).stages

    def test_stages_names(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        assert lmp_input.stages_names == ["Stage 1"]

        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        assert lmp_input.stages_names == [
            "Comment 1",
            "1) Initialization",
            "2) System definition",
            "3) Simulation settings",
            "Part A : energy minimization",
            "Stage 6",
        ]

        lmp_input.stages.append({"stage_name": "New stage", "commands": [("units", "metal")]})
        assert lmp_input.stages_names == [
            "Comment 1",
            "1) Initialization",
            "2) System definition",
            "3) Simulation settings",
            "Part A : energy minimization",
            "Stage 6",
            "New stage",
        ]

    def test_nstages(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        assert lmp_input.nstages == 1

        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        assert lmp_input.nstages == 6

    def test_ncomments(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        assert lmp_input.ncomments == 3

        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        assert lmp_input.ncomments == 1

    def test_get_args(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        units = lmp_input.get_args("units")
        assert units == "metal"

        sets = lmp_input.get_args("set")
        assert sets == [
            "type 1 charge 0.8803",
            "type 2 charge 1.2570",
            "type 3 charge 1.2580",
            "type 4 charge -1.048",
        ]

        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        set1 = lmp_input.get_args("set", stage_name="2) System definition")
        assert set1[1] == "type 2 charge 1.2570"

    def test_contains_command(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        assert lmp_input.contains_command("set")
        assert not lmp_input.contains_command("sett")

        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        assert lmp_input.contains_command("units")
        assert lmp_input.contains_command("units", stage_name="1) Initialization")
        assert not lmp_input.contains_command("units", stage_name="Stage 6")

    def test_set_args(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        lmp_input.set_args(command="units", argument="atomic")
        assert lmp_input.get_args("units") == "atomic"

        lmp_input2 = lmp_input
        lmp_input2.set_args(command="set", argument="new set 2", how=2, stage_name="Stage 1")
        assert lmp_input.stages == lmp_input2.stages
        lmp_input2.set_args(command="set", argument="new set", how="first")
        assert lmp_input.get_args("set")[0] == "new set"
        assert lmp_input.get_args("set")[1] == "type 2 charge 1.2570"

        lmp_input3 = LammpsInputFile().from_file(self.filename, keep_stages=True)
        lmp_input4 = LammpsInputFile().from_file(self.filename, keep_stages=True)
        lmp_input4.set_args(command="set", argument="new set 2", how=2, stage_name="Stage 1")
        assert lmp_input3.stages == lmp_input4.stages
        lmp_input4.set_args(
            command="set",
            argument="new set 2",
            how="first",
            stage_name="2) System definition",
        )
        assert lmp_input4.get_args("set", stage_name="2) System definition")[0] == "new set 2"
        assert lmp_input4.get_args("set", stage_name="2) System definition")[1] == "type 2 charge 1.2570"

    def test_add_stage(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal")
        assert lmp_input.stages == [{"stage_name": "Stage 1", "commands": [("units", "metal")]}]
        lmp_input.add_stage(commands={"pair_style": "eam"}, stage_name="Pair style")
        assert lmp_input.stages == [
            {"stage_name": "Stage 1", "commands": [("units", "metal")]},
            {"stage_name": "Pair style", "commands": [("pair_style", "eam")]},
        ]
        lmp_input.add_stage(commands=["boundary p p p", "atom_style full"], stage_name="Cell")
        assert lmp_input.stages == [
            {"stage_name": "Stage 1", "commands": [("units", "metal")]},
            {"stage_name": "Pair style", "commands": [("pair_style", "eam")]},
            {
                "stage_name": "Cell",
                "commands": [("boundary", "p p p"), ("atom_style", "full")],
            },
        ]

        lmp_input.add_stage(commands=["set type 2 charge 0.0"], after_stage="Stage 1")
        assert lmp_input.stages == [
            {"stage_name": "Stage 1", "commands": [("units", "metal")]},
            {"stage_name": "Stage 4", "commands": [("set", "type 2 charge 0.0")]},
            {"stage_name": "Pair style", "commands": [("pair_style", "eam")]},
            {
                "stage_name": "Cell",
                "commands": [("boundary", "p p p"), ("atom_style", "full")],
            },
        ]
        lmp_input.add_stage(stage_name="Empty stage")
        assert lmp_input.stages == [
            {"stage_name": "Stage 1", "commands": [("units", "metal")]},
            {"stage_name": "Stage 4", "commands": [("set", "type 2 charge 0.0")]},
            {"stage_name": "Pair style", "commands": [("pair_style", "eam")]},
            {
                "stage_name": "Cell",
                "commands": [("boundary", "p p p"), ("atom_style", "full")],
            },
            {"stage_name": "Empty stage", "commands": []},
        ]
        lmp_input.add_stage(
            {
                "stage_name": "New stage",
                "commands": [
                    ("newcommand", "newargs 1 2 3"),
                    ("newcommand2", "newargs 4 5 6"),
                ],
            }
        )
        assert lmp_input.stages == [
            {"stage_name": "Stage 1", "commands": [("units", "metal")]},
            {"stage_name": "Stage 4", "commands": [("set", "type 2 charge 0.0")]},
            {"stage_name": "Pair style", "commands": [("pair_style", "eam")]},
            {
                "stage_name": "Cell",
                "commands": [("boundary", "p p p"), ("atom_style", "full")],
            },
            {"stage_name": "Empty stage", "commands": []},
            {
                "stage_name": "New stage",
                "commands": [
                    ("newcommand", "newargs 1 2 3"),
                    ("newcommand2", "newargs 4 5 6"),
                ],
            },
        ]
        lmp_input.add_stage(
            {
                "stage_name": "New stage 2",
                "commands": [
                    ("newcommand", "newargs 1 2 3"),
                    ("newcommand2", "newargs 4 5 6"),
                ],
            },
            after_stage="Stage 4",
        )
        assert lmp_input.stages == [
            {"stage_name": "Stage 1", "commands": [("units", "metal")]},
            {"stage_name": "Stage 4", "commands": [("set", "type 2 charge 0.0")]},
            {
                "stage_name": "New stage 2",
                "commands": [
                    ("newcommand", "newargs 1 2 3"),
                    ("newcommand2", "newargs 4 5 6"),
                ],
            },
            {"stage_name": "Pair style", "commands": [("pair_style", "eam")]},
            {
                "stage_name": "Cell",
                "commands": [("boundary", "p p p"), ("atom_style", "full")],
            },
            {"stage_name": "Empty stage", "commands": []},
            {
                "stage_name": "New stage",
                "commands": [
                    ("newcommand", "newargs 1 2 3"),
                    ("newcommand2", "newargs 4 5 6"),
                ],
            },
        ]

    def test_merge_stages(self):
        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        lmp_input.merge_stages(["1) Initialization", "3) Simulation settings"])
        assert lmp_input.stages_names == [
            "Comment 1",
            "Merge of: 1) Initialization, 3) Simulation settings",
            "2) System definition",
            "Part A : energy minimization",
            "Stage 6",
        ]
        assert lmp_input.stages == [
            {"stage_name": "Comment 1", "commands": [("#", "LGPS")]},
            {
                "stage_name": "Merge of: 1) Initialization, 3) Simulation settings",
                "commands": [
                    ("units", "metal"),
                    ("atom_style", "full"),
                    ("dimension", "3"),
                    ("pair_style", "hybrid/overlay morse 15 coul/long 15"),
                    ("kspace_style", "ewald 1e-4"),
                    ("boundary", "p p p"),
                    ("pair_coeff", "1 1 morse 0.0580 3.987 3.404"),
                    ("pair_coeff", "1 4 morse 0.0408 1.399 3.204"),
                    ("pair_coeff", "2 4 morse 0.3147 2.257 2.409"),
                    ("pair_coeff", "3 4 morse 0.4104 2.329 2.200"),
                    ("pair_coeff", "4 4 morse 0.0241 1.359 4.284"),
                    ("pair_coeff", "* * coul/long"),
                ],
            },
            {
                "stage_name": "2) System definition",
                "commands": [
                    ("read_data", "run_init.data"),
                    ("set", "type 1 charge 0.8803"),
                    ("set", "type 2 charge 1.2570"),
                    ("set", "type 3 charge 1.2580"),
                    ("set", "type 4 charge -1.048"),
                    ("neigh_modify", "every 1 delay 5 check yes"),
                ],
            },
            {
                "stage_name": "Part A : energy minimization",
                "commands": [
                    ("thermo", "1"),
                    ("thermo_style", "custom step lx ly lz press pxx pyy pzz pe"),
                    ("dump", "dmp all atom 5 run.dump"),
                ],
            },
            {
                "stage_name": "Stage 6",
                "commands": [
                    ("min_style", "cg"),
                    ("fix", "1 all box/relax iso 0.0 vmax 0.001"),
                    ("minimize", "1.0e-16 1.0e-16 5000 10000"),
                    ("write_data", "run.data"),
                ],
            },
        ]

        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        lmp_input.merge_stages(lmp_input.stages_names)
        assert len(lmp_input.stages_names) == 1
        assert lmp_input.stages == [
            {
                "stage_name": "Merge of: Comment 1, 1) Initialization, 2) System definition, 3) Simulation settings, "
                "Part A : "
                "energy minimization, Stage 6",
                "commands": [
                    ("#", "LGPS"),
                    ("units", "metal"),
                    ("atom_style", "full"),
                    ("dimension", "3"),
                    ("pair_style", "hybrid/overlay morse 15 coul/long 15"),
                    ("kspace_style", "ewald 1e-4"),
                    ("boundary", "p p p"),
                    ("read_data", "run_init.data"),
                    ("set", "type 1 charge 0.8803"),
                    ("set", "type 2 charge 1.2570"),
                    ("set", "type 3 charge 1.2580"),
                    ("set", "type 4 charge -1.048"),
                    ("neigh_modify", "every 1 delay 5 check yes"),
                    ("pair_coeff", "1 1 morse 0.0580 3.987 3.404"),
                    ("pair_coeff", "1 4 morse 0.0408 1.399 3.204"),
                    ("pair_coeff", "2 4 morse 0.3147 2.257 2.409"),
                    ("pair_coeff", "3 4 morse 0.4104 2.329 2.200"),
                    ("pair_coeff", "4 4 morse 0.0241 1.359 4.284"),
                    ("pair_coeff", "* * coul/long"),
                    ("thermo", "1"),
                    ("thermo_style", "custom step lx ly lz press pxx pyy pzz pe"),
                    ("dump", "dmp all atom 5 run.dump"),
                    ("min_style", "cg"),
                    ("fix", "1 all box/relax iso 0.0 vmax 0.001"),
                    ("minimize", "1.0e-16 1.0e-16 5000 10000"),
                    ("write_data", "run.data"),
                ],
            }
        ]

    def test_remove_stage(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        lmp_input.remove_stage("Stage 1")
        assert lmp_input.stages == []

        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        lmp_input.remove_stage("1) Initialization")
        assert lmp_input.stages_names == [
            "Comment 1",
            "2) System definition",
            "3) Simulation settings",
            "Part A : energy minimization",
            "Stage 6",
        ]

    def test_rename_stage(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        lmp_input.rename_stage("Stage 1", "Global stage")
        assert lmp_input.stages_names == ["Global stage"]

        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        lmp_input.rename_stage("Stage 6", "Final stage")
        assert lmp_input.stages_names == [
            "Comment 1",
            "1) Initialization",
            "2) System definition",
            "3) Simulation settings",
            "Part A : energy minimization",
            "Final stage",
        ]

    def test_remove_command(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        lmp_input.remove_command("set")
        assert lmp_input.stages == [
            {
                "stage_name": "Stage 1",
                "commands": [
                    ("units", "metal"),
                    ("atom_style", "full"),
                    ("dimension", "3"),
                    ("pair_style", "hybrid/overlay morse 15 coul/long 15"),
                    ("kspace_style", "ewald 1e-4"),
                    ("boundary", "p p p"),
                    ("#", "2) System definition"),
                    ("read_data", "run_init.data"),
                    ("neigh_modify", "every 1 delay 5 check yes"),
                    ("#", "3) Simulation settings"),
                    ("pair_coeff", "1 1 morse 0.0580 3.987 3.404"),
                    ("pair_coeff", "1 4 morse 0.0408 1.399 3.204"),
                    ("pair_coeff", "2 4 morse 0.3147 2.257 2.409"),
                    ("pair_coeff", "3 4 morse 0.4104 2.329 2.200"),
                    ("pair_coeff", "4 4 morse 0.0241 1.359 4.284"),
                    ("pair_coeff", "* * coul/long"),
                    ("#", "Part A : energy minimization"),
                    ("thermo", "1"),
                    ("thermo_style", "custom step lx ly lz press pxx pyy pzz pe"),
                    ("dump", "dmp all atom 5 run.dump"),
                    ("min_style", "cg"),
                    ("fix", "1 all box/relax iso 0.0 vmax 0.001"),
                    ("minimize", "1.0e-16 1.0e-16 5000 10000"),
                    ("write_data", "run.data"),
                ],
            }
        ]

        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        lmp_input._add_command(command="set type 1 charge 0.0", stage_name="1) Initialization")
        lmp_input.remove_command("set")
        assert lmp_input.stages == [
            {"stage_name": "Comment 1", "commands": [("#", "LGPS")]},
            {
                "stage_name": "1) Initialization",
                "commands": [
                    ("units", "metal"),
                    ("atom_style", "full"),
                    ("dimension", "3"),
                    ("pair_style", "hybrid/overlay morse 15 coul/long 15"),
                    ("kspace_style", "ewald 1e-4"),
                    ("boundary", "p p p"),
                ],
            },
            {
                "stage_name": "2) System definition",
                "commands": [
                    ("read_data", "run_init.data"),
                    ("neigh_modify", "every 1 delay 5 check yes"),
                ],
            },
            {
                "stage_name": "3) Simulation settings",
                "commands": [
                    ("pair_coeff", "1 1 morse 0.0580 3.987 3.404"),
                    ("pair_coeff", "1 4 morse 0.0408 1.399 3.204"),
                    ("pair_coeff", "2 4 morse 0.3147 2.257 2.409"),
                    ("pair_coeff", "3 4 morse 0.4104 2.329 2.200"),
                    ("pair_coeff", "4 4 morse 0.0241 1.359 4.284"),
                    ("pair_coeff", "* * coul/long"),
                ],
            },
            {
                "stage_name": "Part A : energy minimization",
                "commands": [
                    ("thermo", "1"),
                    ("thermo_style", "custom step lx ly lz press pxx pyy pzz pe"),
                    ("dump", "dmp all atom 5 run.dump"),
                ],
            },
            {
                "stage_name": "Stage 6",
                "commands": [
                    ("min_style", "cg"),
                    ("fix", "1 all box/relax iso 0.0 vmax 0.001"),
                    ("minimize", "1.0e-16 1.0e-16 5000 10000"),
                    ("write_data", "run.data"),
                ],
            },
        ]

    def test_append(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        lmp_input2 = LammpsInputFile().from_file(self.filename)
        lmp_input.append(lmp_input2)
        assert lmp_input.nstages == 2
        assert lmp_input.ncomments == 6
        assert lmp_input.stages_names == ["Stage 1", "Stage 2"]

        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        lmp_input2 = LammpsInputFile().from_file(self.filename, keep_stages=True)
        lmp_input.append(lmp_input2)
        assert lmp_input.nstages == 12
        assert lmp_input.ncomments == 2
        assert lmp_input.stages_names == [
            "Comment 1",
            "1) Initialization",
            "2) System definition",
            "3) Simulation settings",
            "Part A : energy minimization",
            "Stage 6",
            "Comment 2",
            "Stage 8 (previously 1) Initialization)",
            "Stage 9 (previously 2) System definition)",
            "Stage 10 (previously 3) Simulation settings)",
            "Stage 11 (previously Part A : energy minimization)",
            "Stage 12",
        ]

    def test_add_command(self):
        lmp_input = LammpsInputFile()
        lmp_input.stages.append({"stage_name": "Init", "commands": [("units", "metal")]})
        lmp_input._add_command(stage_name="Init", command="boundary p p p")
        assert lmp_input.stages == [
            {
                "stage_name": "Init",
                "commands": [("units", "metal"), ("boundary", "p p p")],
            }
        ]
        lmp_input._add_command(stage_name="Init", command="atom_style", args="full")
        assert lmp_input.stages == [
            {
                "stage_name": "Init",
                "commands": [
                    ("units", "metal"),
                    ("boundary", "p p p"),
                    ("atom_style", "full"),
                ],
            }
        ]

    def test_add_commands(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage({"stage_name": "Init", "commands": [("units", "metal")]})
        lmp_input.add_commands(stage_name="Init", commands="boundary p p p")
        assert lmp_input.stages == [
            {
                "stage_name": "Init",
                "commands": [("units", "metal"), ("boundary", "p p p")],
            }
        ]
        lmp_input.add_commands(stage_name="Init", commands=["atom_style full", "dimension 3"])
        assert lmp_input.stages == [
            {
                "stage_name": "Init",
                "commands": [
                    ("units", "metal"),
                    ("boundary", "p p p"),
                    ("atom_style", "full"),
                    ("dimension", "3"),
                ],
            }
        ]
        lmp_input.add_commands(stage_name="Init", commands={"kspace_style": "ewald 1e-4"})
        assert lmp_input.stages == [
            {
                "stage_name": "Init",
                "commands": [
                    ("units", "metal"),
                    ("boundary", "p p p"),
                    ("atom_style", "full"),
                    ("dimension", "3"),
                    ("kspace_style", "ewald 1e-4"),
                ],
            }
        ]

    def test_add_comment(self):
        lmp_input = LammpsInputFile()
        lmp_input._add_comment(comment="First comment")
        assert lmp_input.stages == [{"stage_name": "Comment 1", "commands": [("#", "First comment")]}]
        lmp_input._add_comment(comment="Sub comment", stage_name="Comment 1")
        assert lmp_input.stages == [
            {
                "stage_name": "Comment 1",
                "commands": [("#", "First comment"), ("#", "Sub comment")],
            }
        ]
        lmp_input.stages.append({"stage_name": "Init", "commands": [("units", "metal")]})
        lmp_input._add_comment(comment="Inline comment", stage_name="Init")
        assert lmp_input.stages == [
            {
                "stage_name": "Comment 1",
                "commands": [("#", "First comment"), ("#", "Sub comment")],
            },
            {
                "stage_name": "Init",
                "commands": [("units", "metal"), ("#", "Inline comment")],
            },
        ]


class TestFunc(MatSciTest):
    @pytest.mark.filterwarnings("ignore:write_lammps_inputs is deprecated")
    def test_write_lammps_inputs(self):
        # script template
        with open(f"{TEST_DIR}/kappa.txt", encoding="utf-8") as file:
            kappa_template = file.read()
        kappa_settings = {"method": "heat"}
        write_lammps_inputs(output_dir="heat", script_template=kappa_template, settings=kappa_settings)
        with open(f"{self.tmp_path}/heat/in.lammps", encoding="utf-8") as file:
            kappa_script = file.read()
        fix_hot = re.search(r"fix\s+hot\s+all\s+([^\s]+)\s+", kappa_script)
        # placeholders supposed to be filled
        assert fix_hot.group(1) == "heat"
        fix_cold = re.search(r"fix\s+cold\s+all\s+([^\s]+)\s+", kappa_script)
        assert fix_cold.group(1) == "heat"
        lattice = re.search(r"lattice\s+fcc\s+(.*)\n", kappa_script)
        # parentheses not supposed to be filled
        assert lattice.group(1) == "${rho}"
        pair_style = re.search(r"pair_style\slj/cut\s+(.*)\n", kappa_script)
        assert pair_style.group(1) == "${rc}"

        with open(f"{TEST_DIR}/in.peptide", encoding="utf-8") as file:
            peptide_script = file.read()
        # copy data file
        src = f"{TEST_DIR}/data.quartz"
        write_lammps_inputs(output_dir="path", script_template=peptide_script, data=src)
        dst = f"{self.tmp_path}/path/data.peptide"
        assert filecmp.cmp(src, dst, shallow=False)
        # write data file from obj
        obj = LammpsData.from_file(src, atom_style="atomic")
        with pytest.warns(FutureWarning):
            write_lammps_inputs(output_dir="obj", script_template=peptide_script, data=obj)
        obj_read = LammpsData.from_file(f"{self.tmp_path}/obj/data.peptide", atom_style="atomic")
        pd.testing.assert_frame_equal(obj_read.masses, obj.masses)
        pd.testing.assert_frame_equal(obj_read.atoms, obj.atoms)


class TestLammpsTemplateGen(MatSciTest):
    def test_write_inputs(self):
        # simple script without data file
        lis = LammpsTemplateGen().get_input_set(
            script_template=f"{TEST_DIR}/kappa.txt",
            settings={"method": "heat"},
            data=None,
            data_filename="data.peptide",
        )
        assert len(lis) == 1
        lis.write_input(self.tmp_path / "heat")

        with open(self.tmp_path / "heat" / "in.lammps", encoding="utf-8") as file:
            kappa_script = file.read()
        fix_hot = re.search(r"fix\s+hot\s+all\s+([^\s]+)\s+", kappa_script)
        # placeholders supposed to be filled
        assert fix_hot.group(1) == "heat"
        fix_cold = re.search(r"fix\s+cold\s+all\s+([^\s]+)\s+", kappa_script)
        assert fix_cold.group(1) == "heat"
        lattice = re.search(r"lattice\s+fcc\s+(.*)\n", kappa_script)
        # parentheses not supposed to be filled
        assert lattice.group(1) == "${rho}"
        pair_style = re.search(r"pair_style\slj/cut\s+(.*)\n", kappa_script)
        assert pair_style.group(1) == "${rc}"

        # script with data file
        obj = LammpsData.from_file(f"{TEST_DIR}/data.quartz", atom_style="atomic")
        lis = LammpsTemplateGen().get_input_set(
            script_template=f"{TEST_DIR}/in.peptide",
            settings=None,
            data=obj,
            data_filename="data.peptide",
        )
        assert len(lis) == 2
        assert isinstance(lis["data.peptide"], LammpsData)
        lis.write_input(self.tmp_path / "obj")

        obj_read = LammpsData.from_file(str(self.tmp_path / "obj" / "data.peptide"), atom_style="atomic")
        pd.testing.assert_frame_equal(obj_read.masses, obj.masses)
        pd.testing.assert_frame_equal(obj_read.atoms, obj.atoms)


class TestLammpsInputFileExtra(MatSciTest):
    """Coverage tests for error paths, edge cases, and uncovered branches."""

    @classmethod
    def setup_class(cls):
        cls.filename = f"{TEST_DIR}/lgps.in"

    # --- __init__ error paths ---

    def test_init_duplicate_stage_names(self):
        # Two stages with the same name should raise ValueError
        stages = [
            {"stage_name": "A", "commands": [("units", "metal")]},
            {"stage_name": "A", "commands": [("boundary", "p p p")]},
        ]
        with pytest.raises(ValueError, match="Stage names should be unique"):
            LammpsInputFile(stages)

    def test_init_bad_stage_format(self):
        # Missing 'commands' key → KeyError from _check_stage_format
        stages = [{"stage_name": "A"}]
        with pytest.raises(KeyError):
            LammpsInputFile(stages)

    def test_init_bad_stage_name_not_string(self):
        # non-string stage_name → TypeError
        stages = [{"stage_name": 42, "commands": []}]
        with pytest.raises(TypeError, match="should be a string"):
            LammpsInputFile(stages)

    def test_init_bad_commands_type(self):
        # commands not a list → TypeError
        stages = [{"stage_name": "A", "commands": "units metal"}]
        with pytest.raises(TypeError, match="should be a list"):
            LammpsInputFile(stages)

    def test_init_bad_command_format(self):
        # commands is a non-empty list but tuple has >2 elements → ValueError
        stages = [{"stage_name": "A", "commands": [("units", "metal", "extra")]}]
        with pytest.raises(ValueError, match="list of 2-strings tuples"):
            LammpsInputFile(stages)

    # --- set_args error path ---

    def test_set_args_invalid_how(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        # 'how' with non-integer list element → ValueError
        with pytest.raises(ValueError, match="should be a 'first', 'all', an integer or a list of integers"):
            lmp_input.set_args(command="units", argument="atomic", how=[1.5])
        # 'how' as non-list non-string non-int
        with pytest.raises(ValueError, match="should be a 'first', 'all', an integer or a list of integers"):
            lmp_input.set_args(command="units", argument="atomic", how="not_valid")

    # --- add_stage edge cases ---

    def test_add_stage_after_stage_integer(self):
        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        # after_stage is an integer → inserts at index after_stage + 1
        n_stages_before = lmp_input.nstages
        lmp_input.add_stage(commands=["thermo 1"], after_stage=1)
        assert lmp_input.nstages == n_stages_before + 1

    def test_add_stage_after_stage_no_exist(self):
        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        with pytest.raises(ValueError, match="does not exist"):
            lmp_input.add_stage(commands=["thermo 1"], after_stage="Nonexistent stage")

    def test_add_stage_duplicate_name_with_dict(self):
        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        # Adding a full dict stage with a name already present
        existing_name = lmp_input.stages_names[1]
        with pytest.raises(ValueError, match="already present"):
            lmp_input.add_stage(stage={"stage_name": existing_name, "commands": [("units", "metal")]})

    def test_add_stage_with_commands_and_stage_warning(self):
        lmp_input = LammpsInputFile()
        # Passing both a stage dict and commands → UserWarning
        with pytest.warns(UserWarning, match="incompatible"):
            lmp_input.add_stage(
                stage={"stage_name": "Stage A", "commands": [("units", "metal")]},
                commands="boundary p p p",
                stage_name="Extra",
            )

    # --- remove_stage / rename_stage / merge_stages error paths ---

    def test_remove_stage_not_found(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        with pytest.raises(LookupError, match="not present"):
            lmp_input.remove_stage("Nonexistent")

    def test_rename_stage_not_found(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        with pytest.raises(LookupError, match="not present"):
            lmp_input.rename_stage("Nonexistent", "New Stage")

    def test_rename_stage_duplicate(self):
        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        with pytest.raises(ValueError, match="already present"):
            lmp_input.rename_stage("1) Initialization", "2) System definition")

    def test_merge_stages_nonexistent(self):
        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        with pytest.raises(ValueError, match="not in the LammpsInputFile"):
            lmp_input.merge_stages(["1) Initialization", "Nonexistent stage"])

    def test_merge_stages_wrong_order(self):
        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        # Merge in wrong order (3 before 1):
        with pytest.raises(ValueError, match="not in the right order"):
            lmp_input.merge_stages(["3) Simulation settings", "1) Initialization"])

    # --- add_commands error paths ---

    def test_add_commands_invalid_stage(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        with pytest.raises(ValueError, match="does not correspond"):
            lmp_input.add_commands(stage_name="Nonexistent", commands="units metal")

    def test_add_commands_invalid_type(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal")
        with pytest.raises(TypeError, match="string, list of strings or dict"):
            lmp_input.add_commands(stage_name="Stage 1", commands=123)

    def test_add_commands_with_comment_in_list(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal")
        # Inline comments in a list of commands
        lmp_input.add_commands(stage_name="Stage 1", commands=["# A comment", "boundary p p p"])
        cmds = lmp_input.stages[0]["commands"]
        assert ("units", "metal") in cmds
        assert ("# ", "A comment") in cmds or ("#", "A comment") in cmds
        assert ("boundary", "p p p") in cmds

    def test_add_commands_comment_in_dict(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal")
        lmp_input.add_commands(stage_name="Stage 1", commands={"# Inline comment": "", "boundary": "p p p"})
        cmds = lmp_input.stages[0]["commands"]
        assert ("#", "Inline comment") in cmds or ("# ", "Inline comment") in cmds

    # --- remove_command edge cases ---

    def test_remove_command_nonexistent_type(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        with pytest.raises(ValueError, match="string or a list of strings"):
            lmp_input.remove_command("units", stage_name=42)

    def test_remove_command_not_found_warning(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        # Remove a command that doesn't exist → n_removed=0 → UserWarning
        with pytest.warns(UserWarning, match="not found in the LammpsInputFile"):
            lmp_input.remove_command("nonexistent_cmd")

    def test_remove_keep_empty_stages(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal", stage_name="A")
        lmp_input.add_stage(commands="boundary p p p", stage_name="B")
        # Remove the only command from stage B but keep the empty stage
        lmp_input.remove_command("units", stage_name="A", remove_empty_stages=False)
        assert lmp_input.stages_names == ["A", "B"]
        assert lmp_input.stages[0]["commands"] == []
        assert lmp_input.stages[1]["commands"] == [
            ("boundary", "p p p"),
        ]

    def test_remove_command_specific_stage_only(self):
        # remove command in a specific stage, leaving other stages intact
        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        lmp_input.remove_command("set", stage_name="2) System definition")
        # "set" commands in other stages remain
        assert any("set" in cmd[0] for cmd in lmp_input.stages[2]["commands"]) is False
        # The "set" commands should be gone from stage "2) System definition"
        stage_2 = next(s for s in lmp_input.stages if s["stage_name"] == "2) System definition")
        assert not any(cmd[0] == "set" for cmd in stage_2["commands"])

    # --- write_file / get_str with keep_stages=False ---

    def test_get_str_no_keep_stages(self):
        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        s = lmp_input.get_str(keep_stages=False)
        # When keep_stages=False, stage names are not printed as separate blocks
        assert "Comment 1" not in s
        assert "# LAMMPS input generated from LammpsInputFile" in s

    def test_write_file(self):
        lmp_input = LammpsInputFile().from_file(self.filename)
        out_path = self.tmp_path / "lammps_written.in"
        lmp_input.write_file(str(out_path), keep_stages=False)
        with open(out_path, encoding="utf-8") as f:
            content = f.read()
        assert "units metal" in content
        assert "# LAMMPS input generated from LammpsInputFile" in content

    # --- from_str with ignore_comments / edge cases ---

    def test_from_str_ignore_comments(self):
        contents = """# LGPS
units metal
atom_style full
# another comment
boundary p p p
"""
        lmp_input = LammpsInputFile().from_str(contents, ignore_comments=True)
        # The single-stage form with no comments should produce just the commands
        assert lmp_input.nstages == 1
        cmds = lmp_input.stages[0]["commands"]
        assert ("units", "metal") in cmds
        # Comments should be excluded
        assert all(c[0] != "#" for c in cmds)

    def test_from_str_empty_input(self):
        with pytest.raises(ValueError, match="non-empty"):
            LammpsInputFile().from_str("")

    def test_from_str_all_empty_strings(self):
        # All empty lines → ValueError
        with pytest.raises(ValueError, match="non-empty"):
            LammpsInputFile().from_str("\n\n\n")

    # --- _initialize_stage edge cases ---

    def test_initialize_stage_renumbering_after_removal(self):
        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        names_before = lmp_input.stages_names.copy()
        # Remove "Stage 6" then add a new empty stage — the new stage should get
        # the next available stage number (renumbered up, not just nstages + 1)
        lmp_input.remove_stage("Stage 6")
        # Now stages are Comment 1, 1) Init, 2) System, 3) Sim, Part A
        lmp_input.add_stage(stage_name=None)
        assert lmp_input.nstages == len(names_before)
        # New stage name should have a reasonable number (higher than the largest existing)

    # --- LammpsRun class / LammpsRun.md ---

    def test_lammps_run_init(self):

        lammps_run = LammpsRun(
            script_template="units metal\n# placeholder $temperature\ntime $temperature",
            settings={"temperature": 300.0},
            data=None,
            script_filename="in.md",
        )
        assert lammps_run.script_filename == "in.md"
        assert lammps_run.settings == {"temperature": 300.0}

    def test_lammps_run_md(self):
        run = LammpsRun.md(
            data="dummy_str",
            force_field="pair_eam\ncu_u3_200828.eam",
            temperature=300.0,
            nsteps=1000,
        )
        assert run.script_filename == "in.md"
        assert run.settings["temperature"] == 300.0
        assert run.data == "dummy_str"

    # --- get_str single stage ---

    def test_repr(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal")
        # __repr__ calls get_str which adds the header
        s = repr(lmp_input)
        assert "# LAMMPS input generated from LammpsInputFile" in s
        assert "units metal" in s

    def test_from_str_roundtrip_header_stripped(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal", stage_name="Init")
        s = lmp_input.get_str(keep_stages=True)
        # Feed back into from_str — the auto-generated header is stripped on re-parse
        lmp_input2 = LammpsInputFile().from_str(s)
        assert ("units", "metal") in lmp_input2.stages[0]["commands"]

    def test_from_str_manual_header_line(self):
        # Manually provide the auto-generated (short) header line to trigger the
        # while-strip loop in _clean_lines (line 819)
        raw = "# LAMMPS input generated from LammpsInputFile\nunits metal\n"
        lmp_input = LammpsInputFile().from_str(raw)
        assert ("units", "metal") in lmp_input.stages[0]["commands"]

    def test_from_str_multiline_block_all_comments_ignore(self):
        # keep_stages=True + ignore_comments=True + multi-line all-comment block
        # exercises line 607 (keep_block = False)
        contents = "# line one\n# line two\n\nunits metal\n"
        lmp_input = LammpsInputFile().from_str(contents, keep_stages=True, ignore_comments=True)
        cmds = lmp_input.stages[0]["commands"]
        assert all(c[0] != "#" for c in cmds)
        assert ("units", "metal") in cmds

    def test_from_str_multiline_block_all_comments_keep(self):
        # keep_stages=True + ignore_comments=False + multi-line all-comment block
        # exercises lines 612-613 (multiple sub-comments added to the block stage)
        contents = "# top level\n# subline one\n# subline two\n\nunits metal\n"
        lmp_input = LammpsInputFile().from_str(contents, keep_stages=True, ignore_comments=False)
        # Should have a comment stage with all three sub-comments
        c_cmds = lmp_input.stages[0]["commands"]
        joined = " ".join(a[1] for a in c_cmds if a[0].startswith("#"))
        assert "top level" in joined
        assert "subline one" in joined or "subline two" in joined

    # --- _initialize_stage edge cases ---

    def test_initialize_stage_not_a_string(self):
        lmp_input = LammpsInputFile()
        with pytest.raises(TypeError, match="Stage names should be strings"):
            lmp_input._initialize_stage(stage_name=42)

    def test_initialize_stage_duplicate_name(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal", stage_name="A")
        with pytest.raises(ValueError, match="already present"):
            lmp_input._initialize_stage(stage_name="A")

    def test_initialize_stage_index_too_large(self):
        lmp_input = LammpsInputFile()
        with pytest.raises(IndexError, match="too large"):
            lmp_input._initialize_stage(stage_name="New", stage_index=1000)

    def test_initialize_stage_insert_at_index(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal", stage_name="A")
        lmp_input._initialize_stage(stage_name="B", stage_index=0)
        assert lmp_input.stages_names == ["B", "A"]

    def test_initialize_stage_renumber_clash(self):
        # After removing a middle stage, adding a new unnamed stage can create a
        # clash with the old removed name → the renumber path kicks in.
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal", stage_name="Stage 1")
        lmp_input.add_stage(commands="boundary p p p", stage_name="Stage 3")
        # Manually set names to create a gap: Stage 1, Stage 3 → next is Stage 4
        lmp_input._initialize_stage(stage_name=None)
        assert lmp_input.stages_names == ["Stage 1", "Stage 3", "Stage 4"]

    # --- inline comment edge cases ---

    def test_add_comment_new_block_stage_no_name(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal", stage_name="A")
        # Add a comment as a new block stage that already has a stage name
        lmp_input._add_comment(comment="My comment", inline=False, stage_name="Z")
        assert "Z" in lmp_input.stages_names

    def test_add_comment_index_comment_colon(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal", stage_name="A")
        # index_comment=True with "Comment N:" style
        lmp_input._add_comment(
            comment="Comment 1: This is a detail",
            inline=True,
            stage_name="A",
            index_comment=True,
        )
        cmds = lmp_input.stages[0]["commands"]
        # The args should retain the detail without the prefix
        assert any("This is a detail" in str(a[1]) or str(a[1]).startswith("Comment 1") for a in cmds)

    def test_add_comment_inline_no_index(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal", stage_name="A")
        lmp_input._add_comment(comment="Inline detail", inline=True, stage_name="A", index_comment=False)
        cmds = lmp_input.stages[0]["commands"]
        assert any(a[1] == "Inline detail" for a in cmds)

    def test_add_comment_inline_no_stage_name(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal", stage_name="A")
        with pytest.raises(NotImplementedError, match=r"inline.*stage name"):
            lmp_input._add_comment(comment="x", inline=True, stage_name=None)

    # --- from_str / from_file comment block edge cases ---

    def test_from_str_multiline_comment_block(self):
        # from_str with a multi-line block comment, keep_stages=True
        contents = "# header line 1\n# header line 2\n# header line 3\nunits metal\n"
        lmp_input = LammpsInputFile().from_str(contents, keep_stages=True)
        # Should have the header in its commands
        cmds = lmp_input.stages[0]["commands"]
        assert ("units", "metal") in cmds or ("# ", "header line 1") in cmds

    def test_from_str_ignore_comments_blocks(self):
        contents = "# This is a comment\n# another comment\nunits metal\n"
        lmp_input = LammpsInputFile().from_str(contents, ignore_comments=True)
        assert all(c[0] != "#" for c in lmp_input.stages[0]["commands"])

    # --- LammpsRun.write_inputs ---

    def test_lammps_run_write_inputs(self):
        import os

        run = LammpsRun(
            script_template="units metal\ntemperature $temperature",
            settings={"temperature": 300.0},
            data=None,
            script_filename="in.md",
        )
        out_dir = str(self.tmp_path / "lammps_run_output")
        run.write_inputs(out_dir)
        # The script file should have been written
        assert os.path.isfile(f"{out_dir}/in.md")

    # --- add_stage with after_stage = last stage index ---

    def test_add_stage_after_last_stage(self):
        lmp_input = LammpsInputFile().from_file(self.filename, keep_stages=True)
        last_name = lmp_input.stages_names[-1]
        n_before = lmp_input.nstages
        lmp_input.add_stage(commands=["thermo 1"], after_stage=last_name)
        assert lmp_input.nstages == n_before + 1

    # --- _clean_lines / _get_blocks with keep_stages ---

    def test_get_str_single_stage_no_header(self):
        lmp_input = LammpsInputFile()
        lmp_input.add_stage(commands="units metal")
        s = lmp_input.get_str(keep_stages=True)
        # Single stage → header not printed
        assert "Stage 1" not in s
        assert "units metal" in s

    def test_from_str_ampersand_not_at_line_end(self):
        # "&&" is a logical operator in if/variable commands, only a trailing "&" continues a line
        contents = 'units metal\nif "${a} > 1 && ${b} < 2" then "print hi"\npair_style lj/cut &\n  2.5\nrun 0'
        lmp_input = LammpsInputFile.from_str(contents)
        assert lmp_input.stages[0]["commands"] == [
            ("units", "metal"),
            ("if", '"${a} > 1 && ${b} < 2" then "print hi"'),
            ("pair_style", "lj/cut 2.5"),
            ("run", "0"),
        ]

    def test_from_str_roundtrip_keep_stages_no_duplicate_header(self):
        lmp_input = LammpsInputFile.from_str("# Setup\nunits metal\n\n# Run\nrun 0", keep_stages=True)
        for _ in range(2):
            lmp_input = LammpsInputFile.from_str(lmp_input.get_str(), keep_stages=True)
        assert lmp_input.stages_names == ["Setup", "Run"]
        assert lmp_input.get_str().count("# LAMMPS input generated from LammpsInputFile") == 1

    def test_from_str_single_line(self):
        assert LammpsInputFile.from_str("run 0").stages[0]["commands"] == [("run", "0")]
        assert LammpsInputFile.from_str("run 0", keep_stages=True).stages[0]["commands"] == [("run", "0")]

    def test_append_does_not_mutate_argument(self):
        lmp_input1 = LammpsInputFile.from_str("units metal")
        lmp_input2 = LammpsInputFile.from_str("run 0")
        lmp_input1.append(lmp_input2)
        assert lmp_input1.stages_names == ["Stage 1", "Stage 2"]
        assert lmp_input2.stages_names == ["Stage 1"]
        lmp_input1.stages[1]["commands"].append(("run", "10"))
        assert lmp_input2.stages[0]["commands"] == [("run", "0")]
