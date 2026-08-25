# Re-analysis With the Authors' Own Syllable Labels

**Date:** 2026-08-25
**What changed:** the per-rendition syllable labels of the Duke developmental deposit
(doi:10.7924/r4j38x43h) became readable. They had been locked inside MATLAB MCOS table
objects; Sam Brudner exported the deposited tables to CSV (MATLAB R2026a, procedure
documented in his export README), which exposed the real labels for all five birds.

**The real inventories are far smaller than the derived ones.** grn394 and grn397 carry
**two** song syllable types (A, B); grn395, grn475 and sil469 carry three. The earlier
validation, unable to read the labels, had re-derived groupings by k-means clustering the
VAE latents into 12 endpoint modes. Every number in this report uses the authors' labels,
the real bout structure (the `file` column, one wav per bout), and excludes the handful of
rows tagged `Laser On`, which belong to a different experiment.

---

## What survives, what strengthens, what does not survive

### Strengthened: convergence toward crystallised song — now 5 of 5 birds

Per-type distance from each day's bout means to a fixed crystallised reference,
standardised by within-type variance on the reference days:

| Bird | Early window (dph) | Early ÷ pre-reference distance |
|---|---|---|
| `grn394` | 57–69 | **23.8×** |
| `grn395` | 57–69 | 4.9× |
| `grn397` | 62–64 | 3.4× |
| `grn475` | 73–79 | 2.6× |
| `sil469` | 72–79 | 2.0× |

The earlier whole-distribution version of this measure gave 2.0× in one bird. Measured per
real syllable type it is far larger and replicates in every bird. This is now the flagship
developmental validation.

### Survives: the consecutive-day comparison still finds almost nothing

Per-type consecutive-day centroid drift in `grn394`: learning median 0.059, crystallised
median 0.052 — ratio **1.1×**. The methodological claim stands with real labels: comparing
each day to the previous day cannot see development; comparing to a fixed reference can.

### Does not survive everywhere: elevated juvenile within-type variability

| Bird | Ratio (early/crystallised variance) | Own floor | Verdict |
|---|---|---|---|
| `grn394` | **1.34×** (log +0.294) | 1.26× | exceeds, predicted sign |
| `grn475` | 1.15× (log +0.142) | 1.12× | exceeds, marginal |
| `sil469` | 0.94× | 1.21× | within floor |
| `grn395` | 0.99× | 1.10× | within floor |
| `grn397` | 0.90× | 1.12× | within floor, opposite sign |

**The previously reported `grn397` value of 1.41× was an artefact of the derived modes and
is retracted.** The cluster-based measure took variance over *all* detected sounds near the
12 endpoint modes — and the labels show that roughly three-quarters of detected sounds are
not song syllables at any age (27% retained early, 21–23% late). Restricted to the authors'
labels, `grn397` shows no elevation at all; a wider sensitivity window (62–69 dph) gives
0.96×, still null. `grn394`'s effect survives relabelling but with a thinner margin than
the cluster-based 1.43× suggested.

### Why the cluster-based number was wrong, quantified

Refitting the original 12-mode k-means on `grn394`'s endpoint sounds and cross-tabulating
against the real labels:

- **Where labeled song lands, the modes respect the type boundary: 95.4% purity** with
  respect to A/B. The label-recovery path itself was faithful.
- **But only 2 of the 12 modes are actually song.** At the crystallised endpoint, one mode
  is 72% labeled song and one is 92%; the other ten are 0–7% labeled — calls and cage
  noise. Three modes contain no labeled rendition at any age.

The cluster-based dispersion analysis averaged per-mode variance across all twelve modes,
so roughly ten-twelfths of what it averaged was the variability of non-song sounds across
development. That is the artefact: not impure clustering, but computing a song statistic
over a partition in which most cells were never song. Retention is nearly flat across
development (27% of detected sounds labeled early, 21–23% late), so a coverage check alone
could not have caught this — the composition of the unlabeled material is what changes.

### A design limit the labels made visible

`grn394` is the only bird recorded deep past crystallisation (to 114 dph), so its
crystallised window (100–114) is the only one that is unambiguously post-crystallisation.
The other four birds' late windows sit at 85–97 dph, around crystallisation rather than
after it, which compresses any early-versus-crystallised contrast. It is consistent with
this that the one bird with a clean post-100 window is also the one with the clearest
variability effect — but with n=1 that is an observation, not a conclusion.

## Corrected bottom line

- **Convergence** (centroid, fixed reference): robust, 5/5 birds, 2–24×.
- **Consecutive-day blindness**: confirmed with real labels.
- **Juvenile within-type variability**: present in 2 of 5 birds (one marginal); not the
  general property the cluster-based analysis suggested. The dispersion metric itself is
  unchanged — what changed is the grouping it was computed over.

## Reproducing

```bash
python scripts/validate_developmental_labeled.py --csv-dir <dir with {bird}_table.csv> \
    --bird grn394 --out results/validation/labeled_grn394.json
```

The CSVs are exports of the publicly deposited `{bird}_table.mat` files
(doi:10.7924/r4j38x43h, CC0); the export script is MATLAB and documented in the export's
own README. Windows and exclusions are printed by the script and stored in the JSON.
