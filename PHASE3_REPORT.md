# Phase 3 — Drift Metric and Within-Bird Noise Floor

**Date:** 2026-08-03
**Question:** How much does unmanipulated adult song vary day to day? That is the floor any "reopening" signal must clear.
**Answer:** At 1-day separation, **drift is below the within-day noise floor** — 0.013 vs 0.041 in standardised units, averaged over three birds. Drift reaches the floor at roughly 3 days.

Two things had to be fixed or excluded before that number meant anything, and both are reported below because they would each have produced a confidently wrong answer.

---

## The metric

For each syllable type, drift between two days is the **unbiased squared distance between
that type's acoustic centroids**, in a PCA space (64 components, Phase 2) fitted **once**
on a reference day. Fitting the space per day would move the axes themselves, and that
movement would be indistinguishable from song change.

Reported in two units: raw squared PCA distance, and **standardised** — divided by the
pooled within-type variance, making it a dimensionless effect size ("how far the centroid
moved, in units of how much one rendition naturally varies").

**The noise floor** is the identical statistic computed where there is definitionally no
drift: split a single day's *bouts* into two disjoint halves and compare them. Same
estimator, same data, zero true change.

## Correction 1: the naive estimator manufactures drift, and clustering makes it worse

The distance between two sample means is a **biased** estimate of the distance between the
underlying means:

    E[||x̄a - x̄b||²] = ||μa - μb||² + tr(Σa)/na + tr(Σb)/nb

Both correction terms are positive, so a naive metric reports drift between two samples
from the *same* distribution — and reports **more** of it when samples are small. That is
actively hostile here: recording volume varies several-fold between days in this dataset
(39 to 248 songs/day for one bird), so the naive metric would report the most drift on
exactly the quietest days.

Subtracting the two variance terms fixes it — **but only if the sampling unit is
independent, and syllables are not.** They arrive in song bouts, and renditions within a
bout are correlated. Applying the per-rendition correction to bout-clustered data
**under-corrects by ~10×** on the test fixture, leaving a large positive residual:

| | mean estimate under the null | fraction negative |
|---|---|---|
| correction per rendition (clustered data) | **+1.41** | **0.00** |
| correction per bout | −0.00 | 0.50 |
| correction per rendition (genuinely i.i.d. data) | +0.01 | 0.55 |

Truth is zero in all three rows. The first row is a drift signal manufactured entirely out
of within-bout correlation, and it never goes negative — so it would never look like
noise. **The bout is the sampling unit**, for the variance correction, for the bootstrap,
and for the split-half null.

A consequence for planning: precision is governed by the **number of bouts**, not the
number of syllables. Collecting more syllables inside the same number of bouts buys very
little. "Minutes of song" is the wrong unit for a power calculation — Phase 4 must use
bouts.

The estimate is **negative about half the time under the null and must never be clipped**;
clipping would restore exactly the bias the correction removes.

## Correction 2: two "baseline" days are not baseline

Running the metric across all days surfaced an anomaly: for `gy6or6`, every pair involving
2012-03-22 showed **7.6–8.0× the drift** of any other same-gap pair. `or60yw70` showed the
same pattern on its *last* day, 2012-10-01.

I tested three explanations before accepting any:

| Hypothesis | Test | Result |
|---|---|---|
| Artefact of the PCA reference day | Refit the space on each of the 5 days in turn | **Ruled out.** Ratio stays 7.6–8.0× under every reference; values change in the 3rd decimal |
| Time-of-day confound | Restrict all days to a common time window | **Ruled out.** The anomaly gets *larger*, not smaller |
| A few noisy syllable types | Per-type breakdown | **Ruled out.** Essentially every type moves in the same direction |

On stable consecutive days the same computation gives a textbook null — per-type values
straddling zero (−0.15 to +0.28), **0/11 and 0/8 types exceeding the floor**. So the
estimator is calibrated; those two days genuinely differ.

The most probable cause is experimental proximity. The dataset documentation states these
are *"baseline recordings for behavioral experiments that are not included in this
dataset"*, and `gy6or6/032212` is the very directory that Phase 1 found contains 10 files
templated **`washout`** — a post-perturbation recovery phase — dated nine days earlier.

**Both days are excluded from the noise floor.** Including `gy6or6/032212` alone would have
inflated that bird's apparent day-to-day drift roughly eightfold.

## The noise floor

Three birds, off-schema labels excluded, contaminated days excluded, `gr41rd51` excluded
entirely (Phase 0/1: all 1,836 of its files carry an evTAF template).

| Bird | Types | Days | Floor p95 (standardised) | 1-day drift | 2-day | 3-day | Types over floor |
|---|---|---|---|---|---|---|---|
| `gy6or6` | 11 | 4 | 0.0304 | 0.0096 | 0.0239 | 0.0341 | 4/66 |
| `or60yw70` | 8 | 4 | 0.0372 | 0.0118 | 0.0246 | 0.0364 | 0/48 |
| `bl26lb16` | 7 | 3 | 0.0562 | 0.0180 | 0.0357 | — | 0/21 |
| **mean** | | | **0.0413** | **0.0131** | **0.0281** | **0.0353** | **4/135** |

**Headline: at 1-day separation, day-to-day drift (0.0131) sits well below the within-day
noise floor (0.0413), and 0 of 26 syllable types exceed it in any bird.** Within-day
variability dominates between-day change at short separations.

Drift then grows roughly linearly at ~0.012 standardised units per day, crossing the floor
at **≈3 days**. All 4 exceedances in the whole analysis are `gy6or6` at 2–3 day gaps.

### Null calibration

| Bird | Null mean | SD | Fraction negative |
|---|---|---|---|
| `gy6or6` | −0.00015 | 0.274 | 0.63 |
| `or60yw70` | +0.00746 | 0.361 | 0.62 |
| `bl26lb16` | +0.00379 | 0.488 | 0.62 |

Mean ≈ 0 confirms the estimator is unbiased. The fraction negative sitting at 0.62 rather
than 0.50 is **expected, not a defect**: the statistic is a difference of quadratic forms
and is therefore right-skewed, so its median lies below its mean. Unbiasedness is a
statement about the mean.

## What this means for the experiment being planned

**The usable signal window opens at about 3 days.** Comparing song a day apart cannot
distinguish real change from within-day variability with this metric at this recording
volume. Comparisons should be spaced ≥3 days, or aggregate multiple days per timepoint.

**Any claimed "reopening" effect must exceed ~0.04 standardised units** — the within-day
floor — and should be reported against it explicitly, not against zero.

**A caution the data forced.** Two of five nominally-baseline days in a curated public
repository turned out to be unusable as baseline. A wet lab must expect the same and record
genuine baseline days that are demonstrably distant from any manipulation, with the
recording schedule held constant — including time-of-day coverage, which differed sharply
on the excluded days (2–3 h morning-only vs 10–14 h full-day elsewhere).

## Limitations

- **Three birds, 3–4 clean days each, maximum 3-day separation.** The linear growth of
  drift with separation is fitted over a very short baseline and must not be extrapolated
  to weeks. The canary data (9–11 consecutive days) is the natural next test and would
  roughly triple the observable window.
- **Acoustic drift only.** Syntax drift (changes in transition probabilities) is part of
  what song change means and is not measured here.
- **The floor inherits Phase 1 and 2 constraints**: boundaries good to ±5–10 ms, labeller
  instability ≈0.33% syllable error at 2-day separation, measurement in PCA space (UMAP
  attenuates within-type structure and would have depressed these estimates).
- Days were subsampled to 2,000 syllables each, which preserves bout counts but caps
  per-type sample sizes.

## Reproducing

```bash
python scripts/phase3_noise_floor.py --data results/phase2/gy6or6_syllables.npz --exclude-days 2012-03-22 --out results/phase3/noise_floor_gy6or6_clean.json
python scripts/phase3_noise_floor.py --data results/phase2/or60yw70_syllables.npz --exclude-days 2012-10-01 --out results/phase3/noise_floor_or60yw70_clean.json
python scripts/phase3_noise_floor.py --data results/phase2/bl26lb16_syllables.npz --out results/phase3/noise_floor_bl26lb16.json
```

139 tests, each written before its implementation: `python -m pytest tests/ -q`.
