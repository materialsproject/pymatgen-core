from __future__ import annotations

import gc
import time
from pathlib import Path

from pymatgen.io.vasp.outputs import Locpot


LOCPOT_PATH = Path(__file__).resolve().parents[2] / "test-files" / "io" / "vasp" / "outputs" / "LOCPOT.gz"


def _measure_locpot_read_time(count: int = 5) -> list[float]:
    timings_ms: list[float] = []
    for _ in range(count):
        gc.collect()
        start_ns = time.perf_counter_ns()
        locpot = Locpot.from_file(LOCPOT_PATH)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        timings_ms.append(elapsed_ms)
        # Touch parsed data so timing includes full materialization.
        _ = locpot.dim, locpot.data["total"].shape
        del locpot
    return timings_ms


def test_print_locpot_read_baseline() -> None:
    timings_ms = _measure_locpot_read_time()
    mean_ms = sum(timings_ms) / len(timings_ms)
    median_ms = sorted(timings_ms)[len(timings_ms) // 2]

    print(f"\nLOCPOT read timings (ms): {timings_ms}")
    print(f"LOCPOT read mean (ms): {mean_ms:.3f}")
    print(f"LOCPOT read median (ms): {median_ms:.3f}")

