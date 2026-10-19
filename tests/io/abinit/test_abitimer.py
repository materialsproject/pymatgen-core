from __future__ import annotations

import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Wedge

from pymatgen.io.abinit.abitimer import AbinitTimerParser, AbinitTimerSection, ParallelEfficiency
from pymatgen.util.testing import TEST_FILES_DIR

TEST_DIR = f"{TEST_FILES_DIR}/io/abinit"


def abiref_file(filename):
    """Return absolute path to filename in ~pymatgen/tests/files/abinit."""
    return f"{TEST_DIR}/{filename}"


class TestAbinitTimer:
    def test_timer_parser(self):
        tparser = AbinitTimerParser()
        parsed = tparser.parse([abiref_file("mgb2_scf.abo")])
        assert len(parsed) == 1

        tparser = AbinitTimerParser()
        parsed = tparser.parse([abiref_file("si_scf_v10.2.7.abo")])
        assert len(parsed) == 1

    def test_timer(self):
        tparser = AbinitTimerParser()
        tparser.parse([abiref_file("mgb2_scf.abo")])

        assert len(tparser.timers()) == 1
        timer = tparser.timers()[0]
        assert math.isclose(timer.cpu_time, 5.1)
        assert timer.get_dataframe() is not None

        assert timer.to_table()

        assert len(timer.get_values("cpu_time")) == 30

        assert timer.sum_sections("cpu_time") == pytest.approx(3.298999999)

        assert timer.cpuwall_histogram(show=False)
        assert timer.pie(show=False)
        assert timer.scatter_hist(show=False)

    def test_timer_10_2_7(self):
        tparser = AbinitTimerParser()
        tparser.parse([abiref_file("si_scf_v10.2.7.abo")])

        assert len(tparser.timers()) == 1
        timer = tparser.timers()[0]
        assert math.isclose(timer.cpu_time, 0.3)
        assert timer.get_dataframe() is not None

        assert timer.to_table()

        assert len(timer.get_values("cpu_time")) == 30

        assert timer.sum_sections("cpu_time") == pytest.approx(0.307)

        assert timer.cpuwall_histogram(show=False)
        assert timer.pie(show=False)
        assert timer.scatter_hist(show=False)


# Synthetic strong-scaling series: section name -> (cpu_time, wall_time, ncalls) at 1 CPU.
# Times are divided by ncpus * efficiency to emulate parallel runs.
_SECTIONS = {
    "fourwf%(pot)": (4.0, 4.4, 100),
    "nonlop(apply)": (2.0, 2.2, 50),
    "pspini": (1.0, 1.1, 1),
    "zero_section": (0.0, 0.0, 3),
}
_EFFICIENCY = {"fourwf%(pot)": 1.0, "nonlop(apply)": 0.5, "pspini": 0.25, "zero_section": 1.0}


def _timer_block(mpi_nprocs, mpi_rank, cpu_time, wall_time, section_lines):
    lines = [
        f"-<BEGIN_TIMER mpi_nprocs = {mpi_nprocs}, omp_nthreads = 1, mpi_rank = {mpi_rank}>",
        f"- cpu_time =  {cpu_time:.1f}, wall_time =  {wall_time:.1f}",
        "-",
        "- routine                        cpu     %       wall     %      number of calls  Gflops",
        "-                                                                  (-1=no count)",
        *section_lines,
        "-<END_TIMER>",
    ]
    return "\n".join(lines) + "\n"


def _write_scaling_file(path, ncpus):
    """Write a fake ABINIT output with a TIMER section for a run with `ncpus` MPI procs."""
    sec_lines = []
    for name, (cpu, wall, ncalls) in _SECTIONS.items():
        eff = 1.0 if ncpus == 1 else _EFFICIENCY[name]
        ctime, wtime = cpu / (ncpus * eff), wall / (ncpus * eff)
        sec_lines.append(f"- {name:<28} {ctime:8.3f}  10.0  {wtime:8.3f}  10.0  {ncalls:8d}  -1.00")
    # Total cpu and wall times with different scaling so that cpu/wall efficiencies differ.
    cpu_total, wall_total = 8.0 / ncpus, 16.0 / ncpus**0.5
    text = "Some ABINIT output\n" + _timer_block(ncpus, 0, cpu_total, wall_total, sec_lines)
    text += _timer_block(ncpus, "world", cpu_total, wall_total, sec_lines)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def scaling_parser(tmp_path):
    paths = [_write_scaling_file(tmp_path / f"run_{ncpus}.abo", ncpus) for ncpus in (2, 1, 4)]
    parser = AbinitTimerParser()
    assert parser.parse(paths) == paths
    return parser


@pytest.fixture
def mgb2_timer():
    parser = AbinitTimerParser()
    parser.parse(abiref_file("mgb2_scf.abo"))
    return parser.timers()[0]


class TestAbinitTimerParser:
    def test_parse_single_str_and_iteration(self):
        parser = AbinitTimerParser()
        fname = abiref_file("mgb2_scf.abo")
        assert parser.parse(fname) == [fname]
        assert parser.filenames == [fname]
        assert len(parser) == 1
        assert list(parser) == [fname]
        # Both MPI ranks are stored.
        (timer_0,) = parser.timers(filename=fname)
        (timer_world,) = parser.timers(filename=fname, mpi_rank="world")
        assert timer_0.mpi_rank == "0"
        assert timer_world.mpi_rank == "world"
        assert timer_world.cpu_time == pytest.approx(10.2)
        assert timer_world.wall_time == pytest.approx(10.3)

    def test_parse_skips_bad_files(self, tmp_path, caplog):
        no_timer = tmp_path / "no_timer.abo"
        no_timer.write_text("nothing to see here\n", encoding="utf-8")

        bad_header = tmp_path / "bad_header.abo"
        # Third line of the block should be empty but contains a section line.
        bad_header.write_text(
            "-<BEGIN_TIMER mpi_nprocs = 1, omp_nthreads = 1, mpi_rank = 0>\n"
            "- cpu_time = 1.0, wall_time = 1.0\n"
            "- fourwf   0.500 50.0  0.500  50.0  10  -1.00\n",
            encoding="utf-8",
        )

        bad_section = tmp_path / "bad_section.abo"
        bad_section.write_text(
            _timer_block(1, 0, 1.0, 1.0, ["- fourwf   0.500 50.0  0.500  50.0  10  -1.00", "- garbage line"]),
            encoding="utf-8",
        )

        good = abiref_file("mgb2_scf.abo")
        files = [str(no_timer), str(bad_header), str(bad_section), str(tmp_path / "missing.abo"), good, good]
        parser = AbinitTimerParser()
        assert parser.parse(files) == [good]
        assert parser.filenames == [good]
        log = caplog.text
        assert "No timer section found" in log
        assert "line should be empty" in log
        assert "Cannot properly split line" in log
        assert "Cannot open file" in log
        assert "Cannot overwrite timer" in log

    def test_parse_v83_columns(self):
        parser = AbinitTimerParser()
        parser.parse(abiref_file("si_scf_v10.2.7.abo"))
        timer = parser.timers()[0]
        sect = timer.get_section("timing timab")
        assert sect.cpu_time == pytest.approx(0.132)
        assert sect.cpu_fract == pytest.approx(38.3)
        assert sect.wall_time == pytest.approx(0.133)
        assert sect.wall_fract == pytest.approx(34.8)
        assert sect.ncalls == 14
        assert sect.gflops == pytest.approx(-1.0)

    def test_walk(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        p1 = _write_scaling_file(tmp_path / "a.abo", 1)
        p2 = _write_scaling_file(sub / "b.abo", 2)
        (tmp_path / "ignored.txt").write_text("not an abo file", encoding="utf-8")
        (sub / "empty.abo").write_text("no timer\n", encoding="utf-8")

        parser, paths, ok_files = AbinitTimerParser.walk(top=str(tmp_path), ext=".abo")
        assert isinstance(parser, AbinitTimerParser)
        assert len(paths) == 3
        assert sorted(ok_files) == sorted([p1, p2])
        assert sorted(parser.filenames) == sorted([p1, p2])

    def test_section_names_and_get_sections(self):
        parser = AbinitTimerParser()
        parser.parse([abiref_file("mgb2_scf.abo"), abiref_file("si_scf_v10.2.7.abo")])
        names = parser.section_names()
        assert len(names) == 30
        assert names[0] == "fourwf%(pot)"
        assert parser.section_names(ordkey="cpu_time")[:3] == ["fourwf%(pot)", "nonlop(apply)", "pspini"]

        sections = parser.get_sections("fourwf%(pot)")
        assert [sect.cpu_time for sect in sections] == pytest.approx([2.204, 0.010])

        sections = parser.get_sections("invars2")
        assert len(sections) == 2
        assert sections[0].cpu_time == pytest.approx(0.150)
        # Section absent in the second timer -> fake section.
        assert sections[1].name == "fake"
        assert sections[1].ncalls == -1

        assert AbinitTimerParser().section_names() == []

    def test_pefficiency(self, scaling_parser):
        peff = scaling_parser.pefficiency()
        assert isinstance(peff, ParallelEfficiency)
        assert peff.filenames == scaling_parser.filenames
        # Reference is the run with a single CPU (index 1).
        assert peff._ref_idx == 1
        assert set(peff) == {"total", *_SECTIONS}

        for name, eff in _EFFICIENCY.items():
            if name == "zero_section":
                # Division by zero -> -1 sentinel.
                assert peff[name]["cpu_time"] == [-1, -1, -1]
                assert peff[name]["wall_time"] == [-1, -1, -1]
                continue
            assert peff[name]["cpu_time"] == pytest.approx([eff, 1.0, eff], abs=2e-3)
            assert peff[name]["wall_time"] == pytest.approx([eff, 1.0, eff], abs=2e-3)
            assert peff[name]["cpu_fract"] == [10.0, 10.0, 10.0]
            assert peff[name]["wall_fract"] == [10.0, 10.0, 10.0]

        assert peff["total"]["cpu_fract"] == [100, 100, 100]
        assert peff["total"]["wall_fract"] == [100, 100, 100]

        # Ordering by efficiency (sections with -1 are excluded).
        # Wall-time efficiencies of the non-reference runs: total [2**-0.5, 0.5], fourwf [1, 1],
        # nonlop [0.5, 0.5], pspini [0.25, 0.25].
        assert peff.good_sections() == ("fourwf%(pot)", "total", "nonlop(apply)", "pspini")
        assert peff.good_sections(nmax=2, criterion="max") == ("fourwf%(pot)", "total")
        assert peff.bad_sections(nmax=1) == ("pspini",)
        worst = peff.bad_sections(criterion="min", nmax=3)
        assert worst[0] == "pspini"
        assert set(worst[1:]) == {"total", "nonlop(apply)"}  # tied at 0.5
        # Total cpu time scales ideally, so its cpu-time efficiency is 1 for every run.
        assert set(peff.good_sections(key="cpu_time", nmax=2)) == {"fourwf%(pot)", "total"}

        table = peff.totable()
        assert table[0][0] == "AbinitTimerSection"
        assert len(table[0]) == 1 + 2 * 3
        assert [row[0] for row in table[1:]] == ["fourwf%(pot)", "total", "nonlop(apply)", "pspini"]
        assert table[-1][1:] == ["0.25", "10.00", "1.00", "10.00", "0.25", "10.00"]
        assert len(peff.totable(stop=2)) == 3
        assert peff.totable(reverse=False)[1][0] == "pspini"

    def test_pefficiency_total_cpu_vs_wall(self, scaling_parser):
        peff = scaling_parser.pefficiency()
        # cpu_total scales ideally (8/ncpus), wall_total as 16/sqrt(ncpus).
        assert peff["total"]["cpu_time"] == pytest.approx([1.0, 1.0, 1.0])
        assert peff["total"]["wall_time"] == pytest.approx([2**-0.5, 1.0, 0.5], abs=1e-2)

    def test_summarize(self, scaling_parser):
        frame = scaling_parser.summarize()
        assert list(frame["tot_ncpus"]) == [2, 1, 4]
        assert frame["peff"].iloc[1] == pytest.approx(1.0)

    def test_plot_efficiency(self, scaling_parser):
        fig = scaling_parser.plot_efficiency(show=False)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Total_NCPUs"
        assert ax.get_ylabel() == "Efficiency"
        # 4 good + 4 bad sections ("total" appears in both).
        legend = [text.get_text() for text in ax.get_legend().get_texts()]
        assert len(ax.get_lines()) == 8
        assert legend.count("total") == 2
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            "MPI=2, OMP=1",
            "MPI=1, OMP=1",
            "MPI=4, OMP=1",
        ]

        _, ax2 = plt.subplots()
        fig = scaling_parser.plot_efficiency(what="good", nmax=1, ax=ax2, linewidth=1, markersize=3, show=False)
        assert fig is ax2.figure
        # "total" is not the best section, so a separate total line is added.
        assert [text.get_text() for text in ax2.get_legend().get_texts()] == ["fourwf%(pot)", "total"]

        _, ax3 = plt.subplots()
        scaling_parser.plot_efficiency(what="bad", nmax=1, ax=ax3, show=False)
        # Worst section plus an extra line for the total.
        assert [text.get_text() for text in ax3.get_legend().get_texts()] == ["pspini", "total"]
        assert ax3.get_lines()[-1].get_linewidth() == 2.0
        plt.close("all")

    def test_plot_pie(self, scaling_parser):
        plt.close("all")
        fig = scaling_parser.plot_pie(show=False)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        for ax, timer in zip(fig.axes, scaling_parser.timers(), strict=True):
            assert ax.get_title() == str(timer)
            assert any(isinstance(patch, Wedge) for patch in ax.patches)
        plt.close("all")

    def test_plot_stacked_hist(self, scaling_parser):
        fig = scaling_parser.plot_stacked_hist(nmax=2, show=False)
        ax = fig.axes[0]
        assert ax.get_ylabel() == "wall_time"
        assert ax.get_title() == "Stacked histogram with the 2 most important sections"
        legend = [text.get_text() for text in ax.get_legend().get_texts()]
        assert legend == ["fourwf%(pot)", "nonlop(apply)", "others (nmax=2)"]
        # 3 groups (2 names + others) x 3 timers.
        assert len(ax.patches) == 9
        # Total stacked height equals the sum of the section wall times of each timer.
        heights = np.zeros(3)
        for idx, patch in enumerate(ax.patches):
            heights[idx % 3] += patch.get_height()
        expected = [timer.sum_sections("wall_time") for timer in scaling_parser.timers()]
        assert heights == pytest.approx(expected)
        plt.close("all")

    def test_plot_all(self, scaling_parser):
        figs = scaling_parser.plot_all(show=False)
        assert len(figs) == 3
        assert all(isinstance(fig, plt.Figure) for fig in figs)
        plt.close("all")


class TestAbinitTimerSection:
    def test_basic(self):
        sect = AbinitTimerSection(" fourwf ", "1.5", "10.0", "2.0", "20.0", "7", "-1.00")
        assert sect.name == "fourwf"
        assert sect.to_tuple() == ("fourwf", 2.0, 20.0, 1.5, 10.0, 7, -1.0)
        assert sect.as_dict() == {
            "name": "fourwf",
            "wall_time": 2.0,
            "wall_fract": 20.0,
            "cpu_time": 1.5,
            "cpu_fract": 10.0,
            "ncalls": 7,
            "gflops": -1.0,
        }
        assert sect.to_csvline() == "fourwf, 2.0, 20.0, 1.5, 10.0, 7, -1.0\n"
        assert sect.to_csvline(with_header=True) == (
            "# name wall_time wall_fract cpu_time cpu_fract ncalls gflops\nfourwf, 2.0, 20.0, 1.5, 10.0, 7, -1.0\n"
        )

    def test_fake(self):
        fake = AbinitTimerSection.fake()
        assert fake.name == "fake"
        assert fake.to_tuple() == ("fake", 0.0, 0.0, 0.0, 0.0, -1, 0.0)

    def test_str(self):
        string = str(AbinitTimerSection("fourwf", 1.5, 10.0, 2.0, 20.0, 7, -1.0))
        assert "name = fourwf" in string
        assert "gflops = -1.0" in string


class TestAbinitTimerMethods:
    def test_attributes(self, mgb2_timer):
        timer = mgb2_timer
        assert timer.mpi_nprocs == 2
        assert timer.omp_nthreads == 1
        assert timer.ncpus == 2
        assert timer.fname == abiref_file("mgb2_scf.abo")
        assert timer.info["mpi_rank"] == "0"
        assert repr(timer) == f"AbinitTimer(file={timer.fname!r}, wall_time=5.2, mpi_nprocs=2, omp_nthreads=1)"
        assert len(timer.section_names) == 30
        assert timer.get_section("pspini").ncalls == 1
        with pytest.raises(ValueError, match="not in tuple"):
            timer.get_section("not_a_section")

    def test_to_csv(self, mgb2_timer, tmp_path):
        path = tmp_path / "timer.csv"
        mgb2_timer.to_csv(str(path))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "# name wall_time wall_fract cpu_time cpu_fract ncalls gflops"
        assert lines[1] == "fourwf%(pot), 2.212, 21.5, 2.204, 21.6, 8689, -1.0"
        assert len(lines) == 31

        buffer = io.StringIO()
        mgb2_timer.to_csv(buffer)
        assert buffer.getvalue().splitlines() == lines

    def test_to_table(self, mgb2_timer):
        table = mgb2_timer.to_table()
        assert table[0] == list(AbinitTimerSection.FIELDS)
        assert len(table) == 31
        assert table[1] == ["fourwf%(pot)", "2.212", "21.5", "2.204", "21.6", "8689", "-1.0"]

        table = mgb2_timer.to_table(sort_key="ncalls", stop=2)
        assert len(table) == 3
        assert table[1][0] == "fourwf%(pot)"
        assert table[1][5] == "8689"

        with pytest.warns(FutureWarning):
            assert mgb2_timer.totable(stop=1) == mgb2_timer.to_table(stop=1)

    def test_get_dataframe(self, mgb2_timer):
        frame = mgb2_timer.get_dataframe(sort_key="cpu_time")
        assert list(frame.columns) == list(AbinitTimerSection.FIELDS)
        assert len(frame) == 30
        assert frame["name"].iloc[0] == "fourwf%(pot)"
        assert frame["cpu_time"].is_monotonic_decreasing
        assert frame.mpi_nprocs == 2
        assert frame.omp_nthreads == 1
        assert frame.mpi_rank == "0"
        assert frame.fname == mgb2_timer.fname

    def test_get_values(self, mgb2_timer):
        cpu, ncalls = mgb2_timer.get_values(["cpu_time", "ncalls"])
        assert cpu[0] == pytest.approx(2.204)
        assert ncalls[0] == 8689
        assert len(cpu) == len(ncalls) == 30

    def test_names_and_values(self, mgb2_timer):
        total = mgb2_timer.sum_sections("wall_time")

        names, values = mgb2_timer.names_and_values("wall_time")
        assert len(names) == 30
        assert values == sorted(values)
        assert names[-1] == "fourwf%(pot)"

        names, values = mgb2_timer.names_and_values("wall_time", sorted=False)
        assert names == list(mgb2_timer.section_names)

        names, values = mgb2_timer.names_and_values("wall_time", minval=0.2)
        assert names == ["pspini", "nonlop(apply)", "below minval 0.2", "fourwf%(pot)"]
        assert sum(values) == pytest.approx(total)

        names, values = mgb2_timer.names_and_values("wall_time", minfract=0.05, sorted=False)
        assert names == ["fourwf%(pot)", "nonlop(apply)", "pspini", "below minfract 0.05"]
        assert values[0] == pytest.approx(2.212)
        assert sum(values) == pytest.approx(total)

        with pytest.raises(ValueError, match="minfract should be None"):
            mgb2_timer.names_and_values("wall_time", minval=0.1, minfract=0.1)

    def test_order_sections(self, mgb2_timer):
        ordered = mgb2_timer.order_sections("ncalls", reverse=False)
        ncalls = [sect.ncalls for sect in ordered]
        assert ncalls == sorted(ncalls)
        assert ordered[-1].ncalls == 8689

    def test_cpuwall_histogram(self, mgb2_timer):
        plt.close("all")
        fig = mgb2_timer.cpuwall_histogram(show=False)
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Time (s)"
        assert len(ax.patches) == 2 * 30
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ["CPU", "Wall"]
        assert ax.patches[0].get_height() == pytest.approx(2.204)
        assert ax.patches[30].get_height() == pytest.approx(2.212)
        plt.close("all")

    def test_cpuwall_histogram_with_ax(self, mgb2_timer):
        _, ax = plt.subplots()
        plt.figure()  # make another figure current
        try:
            fig = mgb2_timer.cpuwall_histogram(ax=ax, show=False)
            assert fig is ax.figure
            assert len(ax.patches) == 2 * 30
        finally:
            plt.close("all")

    def test_scatter_hist_with_ax(self, mgb2_timer):
        _, ax = plt.subplots()
        plt.figure()  # make another figure current
        try:
            fig = mgb2_timer.scatter_hist(ax=ax, show=False)
            assert fig is ax.figure
            assert len(ax.collections) == 1
            assert ax.collections[0].get_offsets().shape == (30, 2)
        finally:
            plt.close("all")

    def test_pie(self, mgb2_timer):
        fig = mgb2_timer.pie(minfract=0.05, show=False)
        ax = fig.axes[0]
        # Three sections above 5% of the total + one wedge for all sections below minfract.
        wedges = [patch for patch in ax.patches if isinstance(patch, Wedge)]
        assert len(wedges) == 4
        assert sum(wedge.theta2 - wedge.theta1 for wedge in wedges) == pytest.approx(360)
        labels = {text.get_text() for text in ax.texts}
        assert {"fourwf%(pot)", "nonlop(apply)", "pspini", "below minfract 0.05"} <= labels
        plt.close("all")

    def test_scatter_hist(self, mgb2_timer):
        plt.close("all")
        fig = mgb2_timer.scatter_hist(show=False)
        # Scatter axes + top and right histograms.
        assert len(fig.axes) >= 3
        scatter_ax = next(ax for ax in fig.axes if ax.collections)
        offsets = scatter_ax.collections[0].get_offsets()
        assert len(offsets) == 30
        assert list(offsets[0]) == pytest.approx([2.204, 2.212])
        plt.close("all")
