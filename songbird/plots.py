"""Plotting helpers for drift results.

Two conventions these enforce, because a drift figure is easy to draw dishonestly:

* **The noise floor is always drawn.** A drift value has no meaning without it — the whole
  point of this toolkit is that most day-to-day change is smaller than the within-day
  variability it must be judged against.
* **Negative drift is shown, never clipped.** The unbiased estimator is negative about half
  the time under the null. Clipping at zero would hide the evidence that it is unbiased and
  would make a null result look like a small positive effect.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plot_drift_vs_separation", "plot_power_curve"]

_LABEL = {
    "centroid": "centroid drift (standardised)",
    "dispersion": "dispersion drift (log variance ratio)",
}


def plot_drift_vs_separation(result, metric: str = "centroid", ax=None):
    """Per-bird drift against day separation, with each bird's noise floor."""
    import matplotlib.pyplot as plt

    if metric not in _LABEL:
        raise ValueError(f"metric must be one of {sorted(_LABEL)}, got {metric!r}")

    figure, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.5, 4.2))
    key = ("centroid_drift_standardised" if metric == "centroid"
           else "dispersion_drift_abs")

    for name, bird in result.birds.items():
        separations = [pair["separation_days"] for pair in bird.day_pairs]
        values = [pair[key] for pair in bird.day_pairs]
        if not separations:
            continue
        ax.plot(separations, values, "o", label=name, alpha=0.85)
        floor = (bird.centroid_floor / bird.within_type_variance if metric == "centroid"
                 else bird.dispersion_floor)
        ax.axhline(floor, linestyle="--", linewidth=1, alpha=0.6,
                   label=f"{name} noise floor")

    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xlabel("day separation")
    ax.set_ylabel(_LABEL[metric])
    ax.set_title("Drift only counts when it clears the floor")
    ax.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    return figure


def plot_power_curve(curve: dict, ax=None, label: str | None = None,
                     ylabel: str = "minimum detectable drift (standardised)"):
    """Minimum detectable effect against recording volume in bouts."""
    import matplotlib.pyplot as plt

    if not curve:
        raise ValueError("power curve is empty")

    figure, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.0, 4.0))
    bouts = sorted(curve)
    values = [curve[b] for b in bouts]
    ax.plot(bouts, values, "o-", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bouts per timepoint")
    ax.set_ylabel(ylabel)
    ax.set_title("Recording volume is counted in bouts, not minutes")
    if label:
        ax.legend(fontsize=8)
    figure.tight_layout()
    return figure
