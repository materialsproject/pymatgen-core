from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from pymatgen.io.vasp.inputs import Poscar


def parse_locpot_total_fromstring(filename: str | Path) -> tuple[Poscar, np.ndarray]:
    path = Path(filename)
    poscar_lines: list[str] = []

    with path.open("rt", encoding="utf-8") as file:
        while True:
            line = file.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading LOCPOT POSCAR header")
            stripped = line.strip()
            if stripped != "" or len(poscar_lines) == 0:
                poscar_lines.append(stripped)
            else:
                break

        poscar = Poscar.from_str("\n".join(poscar_lines))

        dimline = ""
        while True:
            line = file.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading LOCPOT grid dimensions")
            stripped = line.strip()
            if stripped:
                dimline = stripped
                break

        dims = [int(i) for i in dimline.split()]
        ngridpts = dims[0] * dims[1] * dims[2]

        rest = file.read()

    flat = np.fromstring(rest, sep=" ", count=ngridpts)
    if flat.size != ngridpts:
        raise ValueError(f"Expected {ngridpts} grid values, got {flat.size}")

    data = flat.reshape(dims, order="F")
    return poscar, data


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: uv run python locpot_fromstring_prototype.py /path/to/LOCPOT")
        return 2

    path = Path(sys.argv[1])
    start = time.perf_counter_ns()
    poscar, data = parse_locpot_total_fromstring(path)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6

    print(f"path={path}")
    print(f"formula={poscar.structure.formula}")
    print(f"dim={data.shape}")
    print(f"sample={float(data[0, 0, 0])}")
    print(f"elapsed_ms={elapsed_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
