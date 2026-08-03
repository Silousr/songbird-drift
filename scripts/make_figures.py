"""Generate the summary figures from saved result files.

Reads the JSON written by the phase scripts and produces publication-ready panels in
`results/figures/`. Every drift panel draws its noise floor, and none clip negative values.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/figures")
BIRDS = ["gy6or6", "or60yw70", "bl26lb16"]
COLOURS = {"gy6or6": "#2b6cb0", "or60yw70": "#c05621", "bl26lb16": "#2f855a"}


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def figure_drift_and_floor():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for bird in BIRDS:
        name = (f"results/phase3/noise_floor_{bird}_clean.json"
                if bird != "bl26lb16" else f"results/phase3/noise_floor_{bird}.json")
        data = load(name)
        if not data:
            continue
        scale = data["within_type_variance_scale"]
        gaps = [r["separation_days"] for r in data["between_day"]]
        drift = [r["mean_drift_standardised"] for r in data["between_day"]]
        axes[0].scatter(gaps, drift, label=bird, color=COLOURS[bird], zorder=3)
        axes[0].axhline(data["noise_floor"]["p95"] / scale, ls="--", lw=1,
                        color=COLOURS[bird], alpha=0.7)

        disp = load(f"results/phase5/dispersion_{bird}"
                    f"{'' if bird == 'bl26lb16' else ''}.json")
        if disp:
            axes[1].scatter([r["separation_days"] for r in disp["between_day"]],
                            [r["mean_abs_dispersion"] for r in disp["between_day"]],
                            label=bird, color=COLOURS[bird], zorder=3)
            axes[1].axhline(disp["noise_floor"]["abs_p95"], ls="--", lw=1,
                            color=COLOURS[bird], alpha=0.7)

    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].set_xlabel("day separation"); axes[0].set_ylabel("centroid drift (standardised)")
    axes[0].set_title("Centroid drift vs within-day floor (dashed)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("day separation")
    axes[1].set_ylabel("|dispersion drift| (log variance ratio)")
    axes[1].set_title("Dispersion drift vs its floor (dashed)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Unmanipulated adult song: day-to-day change sits below the noise floor",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "01_drift_vs_noise_floor.png", dpi=200)
    plt.close(fig)


def figure_power():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for bird in BIRDS:
        data = load(f"results/phase4/sensitivity_{bird}.json")
        if data:
            cells = [c for c in data["cells"]
                     if c["n_types"] == max(x["n_types"] for x in data["cells"])]
            axes[0].plot([c["n_bouts"] for c in cells], [c["mde"] for c in cells],
                         "o-", label=bird, color=COLOURS[bird])
        disp = load(f"results/phase5/dispersion_power_{bird}.json")
        if disp:
            rows = [r for r in disp["rows"] if not np.isnan(r["median_fold_change"])]
            axes[1].plot([r["n_bouts"] for r in rows],
                         [r["median_fold_change"] for r in rows],
                         "o-", label=bird, color=COLOURS[bird])

    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xticks([5, 10, 20, 40, 80, 160])
    axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_xlabel("bouts per timepoint")
    axes[0].set_ylabel("min detectable centroid drift (standardised)")
    axes[0].axhline(0.041, ls=":", color="0.3", label="noise floor ~0.041")
    axes[0].set_title("Centroid sensitivity")
    axes[0].legend(fontsize=8)

    axes[1].set_xscale("log")
    axes[1].set_xticks([5, 10, 20, 40, 80])
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].set_xlabel("bouts per timepoint")
    axes[1].set_ylabel("min detectable variance fold-change")
    axes[1].axhline(1.25, ls=":", color="0.3", label="noise floor ~1.25x")
    axes[1].set_title("Dispersion sensitivity (the binding constraint)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Precision scales with BOUTS; both curves flatten near the noise floor",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "02_sensitivity_curves.png", dpi=200)
    plt.close(fig)


def figure_validation():
    dev = load("results/validation/developmental_grn394.json")
    disp = load("results/validation/dispersion_grn394.json")
    if not dev:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    rows = dev["distance_to_endpoint"]
    scale = dev["scale_within_day_variance"]
    x = [r["dph"] for r in rows]
    y = [r["distance"] for r in rows]
    lo = [r["distance"] - r["ci_low"] for r in rows]
    hi = [r["ci_high"] - r["distance"] for r in rows]
    axes[0].errorbar(x, y, yerr=[lo, hi], fmt="o", ms=4, lw=0.8, color="#2b6cb0")
    axes[0].set_xlabel("days post hatch")
    axes[0].set_ylabel("distance to crystallised song (standardised)")
    axes[0].set_title("Convergence on adult song (centroid, vs fixed reference)")

    if disp:
        fracs = disp["fractions"]
        labels = [f"{f['fraction']:.0%}" for f in fracs]
        early = [f["early_variance"] for f in fracs]
        late = [f["crystallised_variance"] for f in fracs]
        width = 0.35
        pos = np.arange(len(fracs))
        axes[1].bar(pos - width / 2, early, width, label="juvenile (<70 dph)",
                    color="#c05621")
        axes[1].bar(pos + width / 2, late, width, label="crystallised (>=100 dph)",
                    color="#2f855a")
        axes[1].set_xticks(pos); axes[1].set_xticklabels(labels)
        axes[1].set_xlabel("fraction of sounds kept per mode (matched across days)")
        axes[1].set_ylabel("per-syllable variance")
        axes[1].set_title(f"Juvenile song is "
                          f"{np.exp(disp['median_log_ratio']):.2f}x more variable")
        axes[1].legend(fontsize=8)
    fig.suptitle("Validation: recovering a documented developmental effect", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "03_validation.png", dpi=200)
    plt.close(fig)


def figure_embedding():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dims = [8, 16, 32, 64, 128]
    pca = [0.932, 0.966, 0.983, 0.995, 0.999]
    umap_dims = [2, 8, 16, 32]
    umap = [0.582, 0.723, 0.704, 0.683]
    ax.plot(dims, pca, "o-", label="PCA", color="#2b6cb0")
    ax.plot(umap_dims, umap, "s-", label="UMAP", color="#c05621")
    ax.axhline(0.95, ls=":", color="0.3", label="fidelity gate (0.95)")
    ax.set_xscale("log")
    ax.set_xlabel("embedding dimensions")
    ax.set_ylabel("within-type distance fidelity (Spearman rho)")
    ax.set_title("Drift lives inside a syllable type.\nUMAP discards that geometry; "
                 "more dimensions do not help.")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "04_embedding_fidelity.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure_drift_and_floor()
    figure_power()
    figure_validation()
    figure_embedding()
    for path in sorted(OUT.glob("*.png")):
        print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} KB)")
