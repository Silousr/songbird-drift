# Phase 4 — Sensitivity Analysis

**Date:** 2026-08-03
**Question:** Given N bouts of song per timepoint, what is the smallest drift detectable at 80% power?
**Answer:** ~0.5 standardised at 5 bouts, falling roughly as 1/N to ~0.01 at 160 bouts. **Around 80 bouts per timepoint (~8 min of song), measurement precision meets the biological noise floor — past that, more recording stops helping.**

---

## Method

Power is estimated by **injection into real data**, never from a formula. Each draw samples
bouts from real recordings, splits them into two disjoint halves, displaces one half by a
known standardised amount, and asks whether the aggregated drift statistic clears its
critical value. Threshold and power come from independent draws.

A parametric power calculation assuming independent syllables would overstate sensitivity
by the design effect — measured at ~10× in Phase 3 — and would promise an experiment far
more sensitive than it can be.

Because the injected shift enters the statistic analytically,

    stat(s) = ||d − s||² − correction = baseline − 2·s·d + ||s||²

each simulation is run once and every effect size on the grid is evaluated in closed form
from the stored `baseline` and projection `s·d`. Exact, and fast enough to sweep the grid.

**Recording volume is counted in bouts.** Not syllables, not minutes. The estimator's
sampling unit is the bout (Phase 3), so precision scales with bout count; more syllables
inside the same bouts buys very little. Minutes are reported only as a conversion, using
each bird's median annotated song per bout (4.4 s, 6.3 s, 8.9 s).

## Minimum detectable drift vs recording volume

Standardised units — the same as the Phase 3 noise floor. All syllable types aggregated.
80% power, α = 0.05.

| Bouts / timepoint | `gy6or6` | `or60yw70` | `bl26lb16` | median | ~song min |
|---|---|---|---|---|---|
| 5 | 0.500 | 0.300 | 0.500 | **0.500** | 0.5 |
| 10 | 0.200 | 0.150 | 0.300 | **0.200** | 1.1 |
| 20 | 0.075 | 0.075 | 0.150 | **0.075** | 2.1 |
| 40 | 0.050 | 0.030 | 0.075 | **0.050** | 4.2 |
| 80 | 0.020 | 0.020 | 0.030 | **0.020** | 8.4 |
| 160 | 0.010 | 0.010 | 0.020 | **0.010** | 16.9 |

MDE falls roughly as **1/N**, not 1/√N, because the statistic is a *squared* distance —
its sampling noise scales as 1/N, so the detectable squared effect does too. Doubling the
number of bouts roughly halves the detectable effect.

## Minimum detectable drift vs syllable-type count

`gy6or6`, aggregating over K randomly chosen syllable types (median of 9 random subsets;
types differ in variance and bout coverage, so a fixed subset would make K incomparable):

| Bouts | K=1 | K=3 | K=5 | K=11 |
|---|---|---|---|---|
| 20 | 0.150 | 0.100 | 0.100 | 0.075 |
| 40 | 0.075 | 0.050 | 0.075 | 0.050 |
| 80 | 0.030 | 0.020 | 0.030 | 0.020 |
| 160 | 0.020 | 0.020 | 0.020 | 0.010 |

Aggregating across the full repertoire buys roughly a factor of 1.5–2 over a single
syllable — real but modest, and much weaker than the effect of recording more bouts. Most
of the gain arrives by K≈3. **Single-type estimates are noticeably noisier**, so a design
resting on one "indicator" syllable is the worst option available.

## Planning table

Bouts per timepoint needed to detect a given standardised drift at 80% power:

| Target drift | Bouts needed | ~Song |
|---|---|---|
| 0.10 | ≥20 | ~2 min |
| 0.05 | ≥40 | ~4 min |
| **0.041 (= noise floor)** | **≥80** | **~8 min** |
| 0.02 | ≥80 | ~8 min |
| 0.01 | ≥160 | ~17 min |

## The result that matters for experimental design

**There is a saturation point, and it arrives early.**

Phase 3 measured real unmanipulated drift at 0.013 (1 day), 0.028 (2 days), 0.035 (3 days),
against a within-day floor of 0.041. Phase 4 shows measurement precision reaches 0.020 at
80 bouts.

So beyond roughly **80 bouts per timepoint the binding constraint stops being measurement
noise and becomes real biological variability.** Recording 160 bouts instead of 80 halves
the MDE to 0.010 — but the bird's own unmanipulated day-to-day drift is already 0.013, so
that extra precision cannot be converted into a stronger claim about a manipulation. It
just resolves noise more finely.

Practical consequences:

- **Record ~80 bouts (~8 min of song) per timepoint.** Below that, sensitivity is lost
  fast; above it, returns collapse.
- **Spend additional effort on more birds or longer separations, not longer recordings.**
  Once past saturation, statistical power comes from replication and from letting real
  drift accumulate — Phase 3 showed drift grows ~0.012/day and only clears the floor at
  ~3 days.
- **A candidate "reopening" effect must exceed ~0.04 standardised** — the noise floor —
  and be reported against it, not against zero.

## Limitations

**The injected effect is a rigid displacement of the whole distribution in a random
direction.** Real drift may be anisotropic, may change the *spread* of renditions rather
than their centre, or may affect some syllables and not others. These curves therefore
describe sensitivity to a **centroid shift**, which is what the Phase 3 metric measures —
not sensitivity to every possible form of song change. A manipulation that increased
rendition-to-rendition variability without moving the mean would be invisible to this
statistic, and that is a plausible signature of a reopened critical period. **Adding a
variance/dispersion drift metric is the single most valuable extension.**

**Single-bird, single-comparison.** These are within-bird sensitivities. A group design
(treated vs control birds) additionally needs between-bird variance, which three birds
cannot estimate usefully.

**Bengalese finch, ≤4 clean days, 7–11 syllable types.** The 1/N scaling should transfer,
but the absolute numbers are species- and rig-specific; each lab should re-run these curves
on its own baseline recordings. The scripts take a `--data` path and no hardcoded paths.

**Everything inherits the earlier constraints:** boundaries good to ±5–10 ms (Phase 1),
labeller instability ≈0.33% syllable error at 2-day separation (Phase 1), measurement in
PCA space because UMAP attenuates within-type structure (Phase 2), and the exclusion of
`gr41rd51` and two contaminated baseline days (Phase 0/1/3).

## Reproducing

```bash
python scripts/phase4_sensitivity.py --data results/phase2/gy6or6_syllables.npz --exclude-days 2012-03-22 --bfsongrepo-root data/bfsongrepo --n-draws 1500 --n-subsets 9 --out results/phase4/sensitivity_gy6or6.json
```

152 tests, each written before its implementation: `python -m pytest tests/ -q`.
