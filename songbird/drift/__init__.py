"""Within-bird song drift estimation.

Two complementary views of change, neither reducible to the other:

* :mod:`songbird.drift.centroid` -- did the syllable *move*?
* :mod:`songbird.drift.dispersion` -- did it get *sloppier*?
* :mod:`songbird.drift.syntax` -- did the *order* change?

A manipulation can do either without doing the other, so both are reported.
"""

from songbird.drift.centroid import (
    naive_squared_centroid_distance,
    unbiased_squared_centroid_distance,
)
from songbird.drift.dispersion import (
    bootstrap_dispersion_ci,
    log_variance_ratio,
    split_half_dispersion_null,
)
from songbird.drift.inference import bootstrap_drift_ci, split_half_null
from songbird.drift.syntax import (
    bout_sequences,
    split_half_syntax_null,
    syntax_divergence,
    transition_counts,
)

__all__ = [
    "bootstrap_dispersion_ci",
    "bout_sequences",
    "split_half_syntax_null",
    "syntax_divergence",
    "transition_counts",
    "bootstrap_drift_ci",
    "log_variance_ratio",
    "naive_squared_centroid_distance",
    "split_half_dispersion_null",
    "split_half_null",
    "unbiased_squared_centroid_distance",
]
