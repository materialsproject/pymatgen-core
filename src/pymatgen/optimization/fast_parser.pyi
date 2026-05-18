from typing import BinaryIO

from numpy.typing import NDArray

def parse_n_doubles(file: BinaryIO, out: NDArray, nelem: int = -1) -> int: ...
