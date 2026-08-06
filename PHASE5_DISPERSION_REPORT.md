# Dispersion Drift — Closing the Known Gap

**Date:** 2026-08-03
**Question:** The centroid metric asks whether a syllable *moved*. Does a second metric asking whether it got *sloppier* add anything?
**Answer: yes, and it recovers the developmental effect the centroid metric missed** — juvenile song is **1.43× more variable** than crystallized song, against a noise floor of 1.15×.

---

## Why this exists

Phase 4 recorded the gap plainly: the drift metric measures a **centroid shift**, so a
manipulation that increased rendition-to-rendition variability *without* moving the mean
would be invisible. That is not a hypothetical failure mode — destabilised, variable
renditions are a plausible signature of destabilised song, arguably more plausible
than a clean directional shift.

The statistic is the log ratio of total variance, `log(tr Var(b) / tr Var(a))`:
scale-free, antisymmetric under swapping days, and **exactly zero under a pure
translation** — the property that makes it genuinely complementary rather than a
re-expression of the centroid distance.

There is no closed-form bias correction here, because the quantity of interest is total
rendition-to-rendition variance including the within-bout component that a bout-mean
collapse would discard. Uncertainty comes entirely from a bout-level bootstrap, and the
null from splitting a day's bouts in half — the bout remaining the sampling unit
throughout, for the same reason as everywhere else in this project.

## Noise floor in unmanipulated adult song

Same three birds, same clean days, same exclusions as Phase 3.

| Bird | Floor (95th pct of \|log ratio\|) | As a fold-change in variance | Day pairs exceeding |
|---|---|---|---|
| `gy6or6` | 0.242 | 1.27× | 0 / 66 |
| `or60yw70` | 0.218 | 1.24× | 0 / 48 |
| `bl26lb16` | 0.730 | 2.07× | 0 / 21 |

Between-day dispersion drift in adults runs 0.065–0.102 in |log| for `gy6or6` — well under
its 0.242 floor — and **no syllable type in any bird exceeds its floor at any separation.**
Unmanipulated adult song does not measurably change its variability day to day, which is
what a usable null looks like.

`bl26lb16`'s floor is 3× the others. It has the fewest days and fewest types, so the null
is estimated from less data. Floors are bird-specific and must be computed per bird, not
borrowed.

## The two metrics are largely independent

Spearman correlation between |dispersion drift| and centroid drift, across all
type × day-pair combinations:

| Bird | ρ | p | n |
|---|---|---|---|
| `gy6or6` | +0.051 | 0.69 | 66 |
| `or60yw70` | +0.217 | 0.14 | 48 |
| `bl26lb16` | +0.408 | 0.067 | 21 |

Weak and not significant in any bird, though `bl26lb16` hints at a positive association on
few observations. **The two statistics see substantially different things**, which is the
justification for carrying both: reporting only one leaves a real class of change
undetected.

## Validation: recovering what the centroid metric missed

Same bird as the earlier validation (`grn394`, dph 57–114). Syllable modes defined by
clustering the crystallized endpoint; per-mode variance compared early versus late.

| Selection kept | Early variance (dph < 70) | Crystallized (dph ≥ 100) | Ratio | log |
|---|---|---|---|---|
| nearest 12% | 1.806 | 1.252 | 1.44× | +0.366 |
| nearest 30% | 2.450 | 1.718 | 1.43× | +0.355 |
| nearest 60% | 3.414 | 2.459 | 1.39× | +0.328 |

**Juvenile song is ~1.43× more variable than crystallized song**, stable across selection
intensity, against a dispersion noise floor of 0.136 (1.15×). The effect exceeds the floor
by ~2.6×, and the sign is as predicted.

Recall that on the same bird and the same latents, **consecutive-day centroid drift showed
a ratio of 1.0× — no effect whatsoever.** The dispersion metric recovers a documented
developmental effect that the centroid metric could not see. That is the gap closed.

### Replicated in a second bird

`grn397` (dph 56–94, endpoint taken from dph 88 since this bird was not recorded as far
past crystallisation):

| Bird | Effect | Its own floor | Margin |
|---|---|---|---|
| `grn394` | **1.43×** (log +0.355) | 1.15× | 2.6× |
| `grn397` | **1.41×** (log +0.344) | 1.36× | 1.2× |

Near-identical effect sizes from independent birds, both with the predicted sign. `grn397`
has the smaller margin because its record stops at 94 dph, so its "crystallised" reference
is both closer to the learning period and estimated from fewer days — a shorter record
raises the floor as well as weakening the contrast.

### An artefact that reversed the sign, and how it was caught

The first version selected sounds within a **fixed distance** of each mode. That keeps
~12% of sounds early and ~70% late — a tight core carved from a broad early distribution
versus nearly all of a narrow late one. The result came out at **0.90×, i.e. juvenile song
apparently *less* variable than adult song**, exceeding the noise floor in the wrong
direction and looking like a publishable surprise.

Matching the selection *fraction* across days removes it entirely and flips the sign to the
predicted direction. The fraction is now swept, and the conclusion holds at 12%, 30% and
60%.

**The general lesson:** when a filter's stringency varies systematically with the variable
under study, it manufactures an effect in that variable. Here the filter got looser exactly
as development proceeded. Any per-day quality threshold — amplitude, SNR, cluster distance
— carries the same risk, and it must be checked by sweeping the threshold rather than
trusting one setting.

## Recommended reporting

Report both metrics side by side, each against its own noise floor:

- **Centroid drift** — did the syllable move? Floor ≈ 0.041 standardised (Phase 3).
- **Dispersion drift** — did it get sloppier? Floor ≈ 0.22–0.73 in |log variance ratio|.

A manipulation may move either, both, or neither, and "neither" is only interpretable when
both were measured.

## Sensitivity: how many bouts to detect a change in variability

Injected by scaling each half's deviations from its own mean — variance changes by exactly
the requested factor, the centroid does not move at all. Median across syllable types,
80% power, two-sided at α = 0.05.

| Bouts / timepoint | `gy6or6` | `or60yw70` | `bl26lb16` |
|---|---|---|---|
| 5 | 2.75× | 2.50× | 3.25× |
| 10 | 2.00× | 1.88× | 2.75× |
| 20 | 1.50× | 1.50× | 2.25× |
| 40 | 1.40× | 1.35× | 1.62× |
| 80 | **1.30×** | **1.30×** | 1.45× |

At the ~80 bouts/timepoint recommended by Phase 4, a **1.3× change in rendition variance**
is detectable — which sits right at the measured dispersion noise floor (1.24–1.27×). The
same saturation logic applies: sensitivity has been pushed down to the level of the bird's
own variability, and further recording cannot convert into a stronger claim.

Dispersion is the **less sensitive** of the two metrics at equal volume. Detecting a 1.3×
variance change needs the same 80 bouts that buys a 0.02 standardised centroid shift —
comfortably below that metric's 0.041 floor. An experiment powered for centroid drift is
*not* automatically powered for dispersion, so the dispersion curve should be the one used
when planning, since it is the binding constraint.

## Limitations

- **Adult floors come from three birds over 3–4 clean days each**, with a maximum
  separation of 3 days. Same constraint as Phase 3.
- **The developmental validation uses the authors' VAE latents**, not this toolkit's
  embedding, and clusters rather than hand labels — the same scope limits as the earlier
  validation.
- **Total variance is a scalar summary.** A change in the *shape* or orientation of the
  rendition cloud that preserved total variance would still be missed. The covariance
  structure carries more information than its trace.

## Reproducing

```bash
python scripts/phase5_dispersion_floor.py --data results/phase2/gy6or6_syllables.npz --exclude-days 2012-03-22 --out results/phase5/dispersion_gy6or6.json
python scripts/validate_dispersion_developmental.py --proj grn394_proj.zip --out results/validation/dispersion_grn394.json
```

169 tests, each written before its implementation: `python -m pytest tests/ -q`.
