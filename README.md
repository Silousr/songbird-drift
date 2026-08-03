# Songbird Vocal Plasticity Measurement Toolkit

Quantifies how much an individual songbird's crystallized song drifts over time, with a
known noise floor and honest uncertainty — a power-analysis tool for planning experiments
that test whether a manipulation reopens the critical period.

Birdsong here is treated as what it is: a learned motor sequence with a syllable inventory
and probabilistic syntax. Not language.

## Status

**Phase 0 (data audit) complete — verdict GO.** See [PHASE0_DATA_AUDIT.md](PHASE0_DATA_AUDIT.md).

**Phase 1 (ingestion + segmentation) complete — gate PASSED.** See
[PHASE1_REPORT.md](PHASE1_REPORT.md). All 18 Bengalese finch bird-days (4 birds, 5.5 GB)
ingested. Amplitude segmentation reaches **F1 = 0.969** against the hand annotations on 14
held-out bird-days (22,324 syllables) with the threshold fitted per bird on a separate day;
TweetyNet recovers hand *labels* at **98.9% frame accuracy / 0.33% syllable error** on a
held-out day two days after training.

Two defects in the published dataset were caught by the loader's guardrails and are
documented in [DECISION_LOG.md](DECISION_LOG.md): a signed 16-bit filename counter that
wraps negative in `bl26lb16/042012`, and 10 files in `gy6or6/032212` that are dated nine
days earlier and belong to a different experimental phase (`washout`).

**Phase 2 (embedding + fidelity gate) complete — gate PASSED for PCA, FAILED for UMAP.**
See [PHASE2_REPORT.md](PHASE2_REPORT.md). On 20,000 hand-labelled syllables, every
representation recovered human labels at ~99% both within-day and across days — so label
recovery alone could not choose between them. The deciding test was **within-type distance
fidelity**, since drift is a within-type phenomenon: PCA-64 scores ρ = 0.996/0.997 across
two birds, UMAP-8 only 0.662/0.450, and raising UMAP's dimensionality does not help.

**Drift will be measured in PCA space; UMAP is for visualisation only.** Silhouette score
turned out to be anti-correlated with what matters here and is not used for selection.

**Phase 3 (drift metric + noise floor) complete.** See [PHASE3_REPORT.md](PHASE3_REPORT.md).
Across three birds, day-to-day drift at 1-day separation (**0.013** standardised) sits well
below the within-day noise floor (**0.041**), with 0 of 26 syllable types exceeding it.
Drift grows ~0.012/day and crosses the floor at **≈3 days** — so that is where the usable
signal window opens.

Two corrections were needed first, each of which would otherwise have produced a
confidently wrong answer: the naive centroid distance manufactures drift out of within-bout
correlation (~10× under-correction, never negative, worst on the quietest days), so the
**bout** is the sampling unit throughout; and two nominally-"baseline" days turned out to
be contaminated by adjacent experimental phases and had to be excluded.

**Phase 4 (sensitivity analysis) complete.** See [PHASE4_REPORT.md](PHASE4_REPORT.md).
Minimum detectable drift falls roughly as 1/N with the number of bouts per timepoint:
0.50 at 5 bouts, 0.075 at 20, **0.020 at 80 (~8 min of song)**, 0.010 at 160. Aggregating
across the whole syllable repertoire buys a further factor of ~1.5–2 over a single type.

**All four phases are complete.** The headline planning result: record **~80 bouts
(~8 min of song) per timepoint**. Below that sensitivity is lost quickly; above it,
measurement precision is already finer than the bird's own unmanipulated day-to-day drift,
so extra recording cannot strengthen a claim — additional power has to come from more birds
or longer separations.

## Scope (all phases complete)

1. **Ingestion + segmentation** — `vak` + TweetyNet. Segmentation F1 0.969, label recovery
   98.9% frame accuracy. [PHASE1_REPORT.md](PHASE1_REPORT.md)
2. **Embedding + fidelity gate** — PCA passes (within-type ρ = 0.996), UMAP fails (0.45–0.66).
   [PHASE2_REPORT.md](PHASE2_REPORT.md)
3. **Drift metric + noise floor** — 1-day drift 0.013 vs floor 0.041; signal window opens at
   ~3 days. [PHASE3_REPORT.md](PHASE3_REPORT.md)
4. **Sensitivity analysis** — MDE vs bouts and syllable count; saturation at ~80 bouts.
   [PHASE4_REPORT.md](PHASE4_REPORT.md)

### Known gap

The drift metric measures a **centroid shift**. A manipulation that increased
rendition-to-rendition *variability* without moving the mean would be invisible to it — and
that is a plausible signature of a reopened critical period. A dispersion-drift metric is
the most valuable next addition.

Statistical guardrails: within-bird longitudinal only; every drift number carries an
uncertainty estimate; the null is established before any signal is claimed; power is
reported, not just p-values.

## Datasets selected in Phase 0

| Role | Dataset | Scale |
|---|---|---|
| Positive control | Duke juvenile zebra finch ([10.7924/r4j38x43h](https://doi.org/10.7924/r4j38x43h)) | 183 bird-days, 163 h |
| Adult null | TweetyNet canary ([10.5061/dryad.xgxd254f4](https://doi.org/10.5061/dryad.xgxd254f4)) | 3 birds, 31 bird-days |
| Adult null | Bengalese Finch Song Repository ([10.6084/m9.figshare.4805749](https://doi.org/10.6084/m9.figshare.4805749)) | 4 birds, 18 bird-days |
| Known-effect validation | Zai et al. deafening ([10.5281/zenodo.14732250](https://doi.org/10.5281/zenodo.14732250)) | 76 birds — interior unverified |

The Koumura BirdsongRecognition dataset has **no recoverable time axis** and cannot support
drift analysis, despite being the most-cited annotated Bengalese finch corpus. It is usable
as a segmentation benchmark only.

## Setup

Requires Python ≥3.12 (`vak` 1.1.0 constraint) and [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
```

## Reproducing the Phase 0 inventory

Downloads ~133 MB of segmentation archives and derives the full longitudinal inventory
without touching the deposit's 290 GB of audio:

```bash
./.venv/bin/python scripts/phase0_duke_inventory.py --download --data-dir ./data/duke --out data/phase0/duke_inventory.json
```

Expected: 183 bird-days, 251,061 wav files, 5,318,154 segmented sounds, 163.1 hours, with
0 date mismatches and 0 unparsed filenames.

## Layout

```
scripts/    standalone Phase 0 audit tooling
data/phase0/    derived inventories + upstream metadata kept for provenance
PHASE0_DATA_AUDIT.md    the audit and its GO/NO-GO
DECISION_LOG.md         every methodological choice and why
```

Bulk data is never committed. Point `--data-dir` wherever you keep it; no paths are
hardcoded.
