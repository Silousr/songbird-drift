# Validation — Recovering a Known Effect

**Date:** 2026-08-03
**Question:** Does the drift metric detect a real, independently-documented change in song?
**Answer: partial success.** The metric recovers developmental convergence toward adult song (**2× effect, non-overlapping confidence intervals**) — but **only when applied as distance to a fixed reference.** The consecutive-day drift framing fails this test outright, and that failure is the more useful result.

Data: Duke deposit [10.7924/r4j38x43h](https://doi.org/10.7924/r4j38x43h) (Brudner, Pearson & Mooney), CC0. Bird `grn394`: **53 days, dph 55–114, 1,654,580 sounds** — spanning late sensorimotor learning through crystallisation (~90 dph).

---

## Test 1 — Consecutive-day drift: **NEGATIVE**

The prediction: a juvenile's song changes fast during learning and stabilises after
crystallisation, so consecutive-day drift should be much larger before ~90 dph than after.
Same bird, same rig, same pipeline — only the developmental stage differs.

| Regime | Median consecutive-day drift | n |
|---|---|---|
| Learning (dph < 95) | 0.0072 | 32 |
| Crystallised (dph ≥ 95) | 0.0073 | 17 |
| **Ratio** | **1.0×** | |

**No effect.** I checked whether this was an artefact of measuring the wrong thing:

- **Non-song contamination.** The deposit's latents contain *all* detected sounds; the
  authors kept only ~25% as song, discarding cage noise and calls. I re-ran restricted to
  sounds near the crystallised endpoint's clusters (the authors' own documented labelling
  approach) and computed drift per cluster. Result: 0.0135 learning vs 0.0187 crystallised
  — **ratio 0.72×, still the wrong direction.**
- **Dispersion instead of location.** Total within-day variance was essentially flat
  (8.86 early vs 9.37 late, ratio 0.95×), so the effect is not hiding in spread either.

So the negative is not a bug in how the test was framed.

## Test 2 — Distance to the crystallised endpoint: **POSITIVE**

Instead of comparing each day to the previous day, compare it to a **fixed adult
reference** (dph 110–114 pooled).

| dph | Distance to adult | 95% CI |
|---|---|---|
| 57 | 0.179 | [0.142, 0.223] |
| 58 | 0.253 | [0.207, 0.301] |
| 63 | 0.116 | [0.095, 0.142] |
| 67 | 0.042 | [0.034, 0.053] |
| 100 | 0.060 | [0.047, 0.077] |
| 105 | 0.050 | [0.038, 0.065] |
| 109 | 0.017 | [0.012, 0.026] |

- Mean distance **dph < 70: 0.104** versus **dph ≥ 100: 0.052** → **2.0× closer to adult song**
- The early days' confidence intervals **exclude** the late values entirely
- Linear slope **−0.00111 per day, p = 1.5×10⁻³**

A second, independent signature of the same convergence: the fraction of a day's sounds
falling near the crystallised endpoint's clusters rises from **0.12 at dph 57 to 0.70 at
dph 112**. The repertoire visibly converges on the adult form.

**Caveat on the trend statistic.** Spearman ρ = −0.247 (p = 0.09) is *not* significant,
because the change is concentrated in the first few days (57 → 64) and then roughly flat,
so the relationship is not monotone across the whole range. The defensible claims are the
early-versus-late contrast (non-overlapping CIs) and the linear slope — not a smooth
monotonic decline.

## Why Test 1 failed, and what it means for the experiment

Three contributing factors, in order of importance:

1. **Consecutive-day change is below the noise floor.** This is exactly what Phase 3 found
   in adults: 1-day drift (0.013) sits under the within-day floor (0.041), and only clears
   it at ~3 days. The same holds here — day-to-day developmental change from 55 dph onward
   is simply small relative to sampling variation.
2. **The window starts late.** `grn394` begins at 55 dph. The most dramatic song change in
   zebra finches happens around 30–60 dph, so most of the plastic period is already over
   when this record starts.
3. **Distance to a fixed reference integrates change; consecutive differences do not.**
   Each consecutive comparison sees one day's worth of change buried in noise. A fixed
   reference accumulates all the change since baseline, and the signal grows while the
   noise does not.

**The design recommendation that follows is concrete: compare every timepoint against a
pooled baseline reference, not against the preceding timepoint.** For an experiment asking
whether a manipulation reopens the critical period, this roughly doubles the usable signal
for free, and it is the difference between detecting the known effect here and missing it.

## What this validates, and what it does not

**Validated:** the drift statistic — the bias-corrected, bout-clustered centroid distance
with bootstrap intervals — detects a real, independently-documented change in song when
given an appropriate comparison structure. That is a genuine known-effect recovery on
public data.

**Not validated:**

- **This toolkit's own embedding.** The latents are the original authors' VAE. Validating
  our audio → spectrogram → PCA path end-to-end needs the raw audio (291 GB against 30 GB
  free). What Phases 1–2 do establish is that our path recovers hand labels at ~99% on
  Bengalese finch; it has not been run on these zebra finch recordings.
- **The per-type metric.** Duke's syllable-type labels are locked in MATLAB tables this
  pipeline cannot read, and syllable identity is itself ill-defined mid-development. This
  used the whole-distribution form, with a per-cluster variant as a cross-check.
- **A deafening or pharmacological effect.** Zai et al. (76 birds, 44 deafened) remains
  the strongest available test and is untouched — 49 GB, and Zenodo blocks file bytes from
  this environment.
- **Generality.** One bird. `grn395`, `grn397`, `grn475` and `sil469` are available and
  the script takes any `_proj.zip`.

## Honest summary

The metric recovers a known effect, but the exercise mainly demonstrated a **limitation of
the consecutive-day framing** that Phases 3 and 4 had already hinted at and that would
otherwise have surfaced only after a wet-lab experiment had been run and analysed. That is
what validation is for.

## Reproducing

```bash
curl -O https://duke.tind.io/record/135/files/grn394_proj.zip   # 301 MB, CC0
python scripts/validate_developmental.py --proj grn394_proj.zip --out results/validation/developmental_grn394.json
```

152 tests, each written before its implementation: `python -m pytest tests/ -q`.
