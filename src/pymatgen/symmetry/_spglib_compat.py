"""Compatibility layer for spglib's switch from return-None to exception-based error handling.

spglib 2.7 deprecated its old error handling: failures return None, and every call (successful or not) emits
"DeprecationWarning: Set OLD_ERROR_HANDLING to false and catch the errors directly". spglib 2.8 makes raising
SpglibError the default and 3.0 removes the old behavior. pymatgen supports spglib >= 2.5, which predates both
SpglibError and the OLD_ERROR_HANDLING switch.

call_spglib opts in to the new behavior only for the duration of a pymatgen call, without changing the setting for
other code using spglib, and converts SpglibError back to None. Callers therefore keep a single "None means failure"
check that works on every supported spglib version.
"""

from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import spglib

try:
    from spglib.error import SpglibError
except ImportError:  # spglib < 2.7 reports failures by returning None

    class SpglibError(Exception):  # type: ignore[no-redef]
        """Placeholder for spglib versions without SpglibError."""


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any, ParamSpec, TypeVar

    P = ParamSpec("P")
    R = TypeVar("R")

# Guards the temporary change to spglib's module-level OLD_ERROR_HANDLING flag.
_OLD_ERROR_HANDLING_LOCK = threading.Lock()


def call_spglib(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R | None:
    """Call a spglib function with exception-based error handling, returning None on failure.

    Args:
        func: The spglib function to call, e.g. spglib.get_symmetry_dataset.
        *args: Positional arguments passed to func.
        **kwargs: Keyword arguments passed to func.

    Returns:
        The result of func, or None if spglib reports an error (by returning None or raising SpglibError).
    """
    error_module: Any = getattr(spglib, "error", None)
    if error_module is None or not hasattr(error_module, "OLD_ERROR_HANDLING"):
        # spglib < 2.7 returns None on failure; spglib >= 3.0 always raises.
        try:
            return func(*args, **kwargs)
        except SpglibError:
            return None

    with _OLD_ERROR_HANDLING_LOCK:
        old_error_handling = error_module.OLD_ERROR_HANDLING
        error_module.OLD_ERROR_HANDLING = False
        try:
            return func(*args, **kwargs)
        except SpglibError:
            return None
        finally:
            error_module.OLD_ERROR_HANDLING = old_error_handling


@contextmanager
def ignore_spglib_error_handling_warning() -> Iterator[None]:
    """Silence spglib's OLD_ERROR_HANDLING DeprecationWarning for third-party code that calls spglib.

    Use this around libraries such as seekpath that call spglib internally and still rely on the old return-None
    error handling, so pymatgen cannot opt them in to exceptions with call_spglib.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Set OLD_ERROR_HANDLING to false", category=DeprecationWarning)
        yield
