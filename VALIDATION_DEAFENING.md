# Validation — Deafening, an Adult Manipulation

**Date:** 2026-08-05
**Question:** Everything so far was validated against a *developmental* effect. Does the metric detect a *manipulation* of *adult crystallized* song — the actual target condition?
**Answer: yes, decisively. 19 of 19 deafened birds drift past their own noise floor, at a median 122× that floor, with a median 4 days to first cross it.**

This was the largest remaining gap in the project and it is now closed.

---

## Why this test and not another

The intended application is an experimental manipulation of crystallized adult song. The
best-documented public example is **deafening**: remove an adult songbird's cochleae and its
song deteriorates over weeks. The effect is large, documented for decades, and it is a loss
of exactly the stability that crystallization established.

Data: **Zai, Stepien, Giret & Hahnloser (2024)**, eLife
[10.7554/eLife.90445](https://doi.org/10.7554/eLife.90445), derived dataset
[10.3929/ethz-b-000670443](https://doi.org/10.3929/ethz-b-000670443), MIT licence, 40 MB.

Each bird contributes the pitch of a target syllable, in Hz, per rendition, timestamped in
days, with the deafening moment marked by a `cochlea removal` annotation. The cohort used
is the group the authors label **"WN deaf control"** — birds that were deafened and given
**no** white-noise training, so the only manipulation is the deafening itself.

## Design — everything within bird

Nothing is borrowed across animals. For each bird:

- **Noise floor** — from that bird's own **pre-deafening** days, split in half by bout.
- **Negative control** — drift between pairs of **pre-deafening** days.
- **Test** — drift from the pooled pre-deafening baseline to each post-deafening day.

Bouts were reconstructed from rendition timestamps (a gap above 60 s starts a new bout),
since this deposit stores a timestamp per rendition rather than one file per bout. That
gave ~19 renditions per bout, which is a plausible zebra finch song bout.

## Result

**19 distinct birds** from 24 recordings — five birds contributed two target syllables each,
and counting those twice would have been exactly the pseudo-replication the group test
exists to prevent, so they were merged per bird before any comparison.

| | Median | Birds above their own floor |
|---|---|---|
| **Negative control** — baseline day vs baseline day | 0.031 | 7 / 19 |
| **Test** — baseline vs post-deafening | **3.589** | **19 / 19** |

- **Median post-deafening drift is 122× the bird's own within-day noise floor.**
- **Median 4 days** post-deafening before drift first clears the floor (range 1–26).
- Paired within-bird comparison, same animals and same estimator:
  difference **+7.95** [+4.28, +12.72], **permutation p = 0.0001**.

### The raw physics agrees

Standardised numbers are only meaningful if the underlying measurement moved, so:

| Bird | Baseline pitch | Final pitch | Shift | Baseline SD | Shift in SDs |
|---|---|---|---|---|---|
| `o10g10` | 620.7 Hz | 690.5 Hz | +69.8 Hz (11.2%) | 9.6 Hz | **7.2** |
| `b10p5` | 734.9 Hz | 797.8 Hz | +62.9 Hz (8.6%) | 11.1 Hz | 5.7 |
| `b8k16` | 525.7 Hz | 562.6 Hz | +36.9 Hz (7.0%) | 8.8 Hz | 4.2 |
| `Deg1` | 833.8 Hz | 868.9 Hz | +35.1 Hz (4.2%) | 13.1 Hz | 2.7 |

Target-syllable pitch moved **3–11%**, or **2.7–7.2 baseline standard deviations**, over the
weeks following deafening. That is the documented deterioration, and it reconciles with the
standardised figures — the centroid statistic is a *squared* distance in units of baseline
variance, so a 7.2 SD shift corresponds to ~52, matching `o10g10`'s 44.9.

## An honest reading of the negative control

The control exceeded its own floor in **7 of 19 birds**, which looks at first like a 37%
false-positive rate. It is not, and the reason matters.

The floor is a **within-day** quantity, while baseline pairs are separated by up to ten
days. Phases 3 and 7 already established that real drift accumulates and crosses the
within-day floor at around three days, in both Bengalese finch and canary. Baseline pairs
several days apart *should* sometimes exceed a within-day floor — that is the same
accumulating baseline drift measured elsewhere in this project, not a defect of the test.

The comparison that carries the weight is therefore the paired one: post-deafening drift is
**~115× the same birds' baseline-to-baseline drift**, p = 0.0001. Both quantities include
whatever baseline drift exists; only one includes deafening.

## Scope — what this does and does not establish

**Establishes:** the drift *statistic* — the bias-corrected, bout-clustered estimator, its
split-half floor, its bootstrap interval, and the per-bird design around them — detects a
real manipulation of adult crystallized song, in every animal tested, with a large margin
and a sensible time course.

**Does not establish:** this toolkit's own audio → spectrogram → PCA path on these birds.
The deposit contains derived pitch, not audio; the raw audio is a separate 49 GB archive
that does not fit on the machine this was run on. The embedding path is validated separately
on Bengalese finch (Phases 1–2) and on zebra finch development (Phase 5).

**Also note:** the measurement is **one-dimensional** — the pitch of one target syllable.
"Centroid drift" here is a shift in mean pitch and "dispersion drift" a change in pitch
variability. A full repertoire-wide analysis would need the audio.

**And:** deafening is a proxy. It acts by removing auditory feedback; a manipulation acting
through a different route need not produce the same signature. Both destabilise crystallized
song, which is what makes deafening a good available test, but they are not interchangeable.

## What it says for a real experiment

- **The metric works on the target condition.** An adult manipulation that destabilises song
  is detected in 19/19 birds at ~122× the noise floor.
- **Expect ~4 days before the effect is visible**, and design the first post-treatment
  timepoint accordingly. One bird took 26 days.
- **The within-bird design is sufficient.** No control group was needed to detect the
  effect — each bird's own baseline supplied the floor. A control group is still needed to
  attribute the effect to the treatment rather than to time in the apparatus.

## Reproducing

```bash
curl -L -o companion.zip "https://www.research-collection.ethz.ch/server/api/core/bitstreams/f40e66b0-4d7a-4672-b193-53c04532d230/content"
unzip companion.zip
python scripts/validate_deafening.py --root "Data Goal-directed vocal planning" \
    --out results/validation/deafening.json
```

Runs in ~17 seconds. Tests: `python -m pytest tests/ -q`.
