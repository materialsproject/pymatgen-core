from __future__ import annotations

import pytest

from pymatgen.util.testing import seekpath_unusable_reason


@pytest.fixture(scope="module")
def seekpath_usable() -> None:
    """Skip the module's seekpath-dependent tests when the probe reports unusable.

    The probe is evaluated lazily here — at test time, once per module — rather
    than at module scope during collection. A probe failure outside the known
    unavailability paths then errors only the tests that need seekpath instead
    of failing collection of the whole file.
    """
    if reason := seekpath_unusable_reason():
        pytest.skip(reason)
