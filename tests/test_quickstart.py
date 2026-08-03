"""The worked example is part of the deliverable, so it is tested.

This is the only end-to-end test that covers the whole chain a lab runs: raw audio ->
annotations -> syllable table -> embedding -> per-bird drift vs that bird's own floor ->
treated-vs-control permutation test. If any link breaks, this fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from quickstart import main  # noqa: E402


def test_quickstart_recovers_the_injected_effect():
    # Returns 0 only when the treated group is detected as drifting more (p < 0.05).
    assert main() == 0
