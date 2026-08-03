# Experimental Protocol

**For a lab planning an experiment that tests whether a manipulation reopens the song critical period.**

This is the practical summary. Every number in it was measured, and the phase report that
measured it is cited. If you read nothing else, read §1 and §7.

---

## 1. The seven decisions that matter most

| # | Decision | Why |
|---|---|---|
| 1 | **Record ~80 song bouts per bird per timepoint** (~8 min of song for a Bengalese finch) | Below this, sensitivity falls fast. Above it, precision is already finer than the bird's own day-to-day variability, so extra recording cannot strengthen a claim. |
| 2 | **Space timepoints ≥3 days apart** | Consecutive-day change is smaller than within-day variability. The usable signal window opens at ~3 days. |
| 3 | **Compare every timepoint against a *pooled baseline*, not against the previous timepoint** | On the validation dataset this was the difference between detecting a known effect and missing it entirely. |
| 4 | **Collect ≥5 baseline days per bird, all before any manipulation** | The baseline defines the bird's own noise floor. It cannot be borrowed from another bird — measured floors differ 3-fold between individuals. |
| 5 | **Hold the recording schedule constant, including time of day** | Two days in a public "baseline" dataset were unusable partly because they were morning-only while the rest spanned 10–14 h. |
| 6 | **Report all three drift metrics, each against its own floor** | A manipulation may move a syllable, make it sloppier, or reorder the sequence. They are largely independent, and "no effect" is only interpretable if all three were measured. |
| 7 | **The bird is the unit of replication** | Pooling syllables across birds is pseudo-replication and can inflate significance by orders of magnitude. |

## 2. How much song, and in what unit

**Count bouts, not minutes and not syllables.** The estimator's sampling unit is the song
bout, because syllables within a bout are correlated. Collecting more syllables inside the
same number of bouts buys very little precision.

Minimum detectable change at 80% power ([Phase 4](PHASE4_REPORT.md), [Phase 5](PHASE5_DISPERSION_REPORT.md)):

| Bouts per timepoint | ~Song | Centroid drift | Variance fold-change |
|---|---|---|---|
| 5 | 0.5 min | 0.50 | 2.5–3.3× |
| 20 | 2 min | 0.075 | 1.5–2.3× |
| 40 | 4 min | 0.050 | 1.4–1.6× |
| **80** | **8 min** | **0.020** | **1.3–1.5×** |
| 160 | 17 min | 0.010 | — |

Sensitivity improves as **1/N**, not 1/√N, because the centroid statistic is a squared
distance. Doubling bouts roughly halves the detectable effect.

**Dispersion is the binding constraint.** The 80 bouts that give a 0.020 centroid
sensitivity — comfortably under that metric's 0.041 floor — give only a 1.3× variance
sensitivity, which sits *at* the dispersion floor. Plan against the dispersion curve.

## 3. How many birds

Recording more per bird and adding more birds solve different problems, and past ~80 bouts
only the second one helps. `songbird plan` estimates group size by simulating the
permutation test that will actually be run:

```bash
songbird plan --effect 0.20 --between-bird-sd 0.05
```

Birds **per group** needed at 80% power, one-sided, α = 0.05:

| Effect (standardised drift) | Between-bird SD 0.02 | 0.05 | 0.10 |
|---|---|---|---|
| 0.05 | 4 | 12 | >30 |
| 0.10 | 4 | 5 | 12 |
| 0.20 | 4 | 4 | 5 |
| 0.40 | 4 | 4 | 4 |

Estimate `--between-bird-sd` from your own baseline birds: compute each bird's
baseline-to-baseline drift and take the standard deviation across birds. Do not guess it —
it is the dominant term in the table above. Notice how much more it matters than the effect
size: at SD 0.02 almost anything is detectable with 4 birds per group, while at SD 0.10 a
0.05 effect is out of reach at any realistic colony size.

**A hard floor at n=4 per group.** With 3 birds per group there are only 20 distinct ways to
split the labels, so the smallest achievable one-sided permutation p-value is 1/21 = 0.048.
Three birds per group can technically clear α = 0.05, but only in the single most extreme
arrangement — there is no margin. Four per group is the practical minimum, and that is a
property of the design, not of this software.

## 4. Baseline design, and a warning

**Two of five nominally-baseline days in a curated public repository turned out to be
unusable as baseline** ([Phase 3](PHASE3_REPORT.md)). Each showed 7.6–8.0× the drift of any
other day pair in the same bird. One of them sat in a directory that also contained files
tagged `washout` — a post-perturbation recovery phase — dated nine days earlier.

The failure mode is generic: baseline recordings collected *around* an experiment tend to
be contaminated by it. Guard against it by

- recording baseline days that are demonstrably distant from any manipulation, including
  any prior experiment on the same bird;
- keeping ≥5 baseline days so an outlier day is detectable rather than load-bearing;
- checking the baseline days against each other **before** unblinding — any baseline day
  that drifts more than its siblings should be investigated, not averaged in;
- keeping the recording schedule identical across days.

## 5. Running the analysis

```bash
# 1. Describe where the recordings are (or write the manifest CSV yourself)
songbird manifest --root recordings/ --out manifest.csv \
    --pattern '(?P<bird>\w+)_(?P<timestamp>\d{8}T\d{4})\.wav$' \
    --timestamp-format '%Y%m%dT%H%M'

# 2. Drift and noise floors, per bird
songbird analyse --manifest manifest.csv --out results.json --drift-out drift.csv

# 3. Treated versus control
songbird compare --drift drift.csv --out comparison.json --alternative greater
```

`manifest.csv` needs four columns — `bird`, `timestamp`, `audio_path`, `annot_path` — plus
`group` for the comparison. Annotations can be in any format `crowsetta` reads: `simple-seq`
(3-column CSV), `notmat` (evsonganaly), `raven`, `textgrid` (Praat), `aud-seq` (Audacity).

Check the warnings. `manifest` reports files whose names it could not parse; `analyse`
reports recordings with no annotation. Both are usually a convention you did not know you
had, and both change what the numbers mean.

## 6. Reading the output

```
gy6or6: 8000 syllables, 583 bouts, 4 days, 11 syllable types
  noise floor  centroid 0.3861 (standardised 0.0291)   dispersion 0.2137 (1.24x variance)
  2012-03-23 -> 2012-03-25 (+2d): centroid +0.0254 [2/11 types over floor], ...
```

- **Standardised centroid drift** is centroid movement in units of natural
  rendition-to-rendition variation. **Dispersion drift** is a log variance ratio; `+0.21`
  means renditions became 1.24× more variable.
- **`[2/11 types over floor]`** is the number to look at. It counts syllable types whose
  drift confidence interval clears that bird's own noise floor. Zero means no detected
  change, whatever the point estimate says.
- **Negative drift is normal and must not be clipped.** The estimator is bias-corrected, so
  under no change it is negative about half the time. A negative value means the observed
  separation is smaller than sampling noise alone would produce.

## 6b. The three metrics

| Metric | Question | Statistic | Floor is centred on |
|---|---|---|---|
| **Centroid** | did the syllable *move*? | unbiased squared distance between centroids | zero (bias-corrected) |
| **Dispersion** | did it get *sloppier*? | log ratio of total variance | zero |
| **Syntax** | did the *order* change? | Jensen–Shannon divergence of transition bigrams, in bits | a **positive** value |

The syntax floor is the one that catches people out. Jensen–Shannon divergence is
non-negative by construction, so two finite samples from the *same* syntax still differ,
and the smaller the sample the more they differ. A raw syntax divergence therefore means
nothing on its own — it must always be read against the split-half floor for that bird at
that number of bouts. There is no bias correction that could centre it on zero.

Note also that syntax is computed on the **full** annotation set, never a subsample:
dropping syllables at random splices together transitions the bird never produced.

## 7. What this toolkit will not tell you

- **It measures centroid, total variance, and first-order (bigram) syntax.** A change in
  the *shape* of the rendition cloud that preserved its trace, or a change in longer-range
  sequence structure than adjacent pairs, would be missed.
- **The floors were measured over ≤3-day separations in three Bengalese finches.** Whether
  day-to-day variability stays constant over weeks is untested and cannot be assumed;
  measure your own baseline over the timescale you plan to use.
- **It has been validated against a developmental effect, not a pharmacological one.** No
  public dataset contains a pharmacological critical-period manipulation with annotated
  audio. The nearest available test — deafening-induced deterioration (Zai et al., 76
  birds) — has not been run.
- **The species matters.** Numbers here come from Bengalese finch and one zebra finch. The
  1/N scaling should transfer; the absolute floors will not. Re-run the curves on your own
  baseline.

## 8. Pre-registration checklist

- [ ] Baseline days per bird: ____ (≥5 recommended)
- [ ] Bouts per bird per timepoint: ____ (~80)
- [ ] Timepoint spacing: ____ days (≥3)
- [ ] Birds per group: ____ (from `songbird plan`, using your own between-bird SD)
- [ ] Primary metric declared in advance: centroid / dispersion / both
- [ ] Comparison structure: each timepoint vs **pooled baseline**
- [ ] Noise floor computed **per bird** from its own baseline days
- [ ] Recording schedule, including time of day, held constant across all days
- [ ] Analysis unit for the group test: **one drift value per bird**
- [ ] Threshold for "change": drift CI clears that bird's own floor

---

Reports: [Phase 0](PHASE0_DATA_AUDIT.md) · [Phase 1](PHASE1_REPORT.md) ·
[Phase 2](PHASE2_REPORT.md) · [Phase 3](PHASE3_REPORT.md) · [Phase 4](PHASE4_REPORT.md) ·
[Phase 5](PHASE5_DISPERSION_REPORT.md) · [Validation](VALIDATION_REPORT.md) ·
[Decision log](DECISION_LOG.md)
