# Phase 2 — Embedding and Fidelity Gate

**Date:** 2026-08-03
**Question:** Is the embedding faithful enough that distances in it mean something?
**Answer: PASS for PCA. FAIL for UMAP — which is the finding that matters.**

All numbers computed in this session on 20,000 hand-labelled syllables (two birds ×
5 days × 2,000/day), regenerable via `scripts/phase2_build_syllable_set.py` and
`scripts/phase2_fidelity.py`.

---

## The gate, and why it has two halves

The brief sets the criterion: an embedding must *"separate syllable types and recover
human labels at high accuracy"*, because *"if distances in the embedding don't correspond
to real acoustic differences, the drift metric is measuring noise."*

Those are two different requirements, and only the first is commonly checked:

1. **Between-type fidelity** — do syllables of different types sit apart? Measured by
   held-out and cross-day k-NN recovery of human labels.
2. **Within-type fidelity** — do distances *inside* one syllable type track real acoustic
   differences? Measured by `within_type_distance_correlation`: the Spearman correlation
   between pairwise embedding distances and pairwise raw-spectrogram distances, computed
   separately within each label so between-type spread cannot flatter it.

The second is the one that decides whether drift is measurable at all. **Drift in
crystallised song is a within-type phenomenon** — renditions of syllable `b` slowly change
shape while remaining recognisably `b`. An embedding that maps every rendition onto its
type centroid would classify perfectly and carry *zero* drift signal. Label recovery cannot
see that failure. This metric can.

## Result 1: label recovery is saturated and cannot choose an embedding

| Bird | Labels | Chance | Representation | Within-day | Cross-day (4 later days) |
|---|---|---|---|---|---|
| `gy6or6` | 11 | 0.222 | pixels (3200d) | 0.9917 | 0.9946 |
| | | | **PCA (64d)** | **0.9917** | **0.9945** |
| | | | UMAP (8d) | 0.9900 | 0.9934 |
| `or60yw70` | 8 | 0.240 | pixels (3200d) | 0.9967 | 0.9839 |
| | | | **PCA (64d)** | **0.9983** | **0.9850** |
| | | | UMAP (8d) | 0.9950 | 0.9795 |

Every representation recovers human labels at 98–99.8%, and the **cross-day** figures are
as good as the within-day ones (for `gy6or6`, marginally better). The embedding's geometry
is stable across days, which is the prerequisite for comparing day A with day B at all.

But note what this table cannot do: **on `gy6or6` all three representations score
identically to four decimal places.** Sweeping PCA from 8 to 128 dimensions and UMAP from
2 to 32 leaves k-NN accuracy pinned at 0.9917 throughout. Label recovery is saturated —
Bengalese finch syllables are highly stereotyped and there are only 8–11 types. **Used
alone it would have declared every candidate embedding equally good.**

## Result 2: within-type fidelity separates them sharply

| Representation | `gy6or6` ρ | `or60yw70` ρ | Silhouette (`gy6or6`) |
|---|---|---|---|
| pixels (3200d) | 1.000 (tautological) | 1.000 (tautological) | 0.271 |
| **PCA (64d)** | **0.996** | **0.997** | 0.317 |
| UMAP (8d) | **0.662** | **0.450** | 0.355 |

`pixels` scores 1.000 by construction — it *is* the reference being correlated against, so
that entry is a sanity check, not evidence.

**PCA preserves within-type geometry almost perfectly. UMAP destroys a third to a half of
it.** On `or60yw70`, ρ = 0.450 means roughly half the rank-ordering of within-type acoustic
distances is scrambled.

### The distortion is intrinsic to UMAP, not a dimensionality artefact

Sweeping components on `gy6or6` (k-NN accuracy was 0.9917 at *every* row):

| Dimensions | 2 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| PCA ρ | — | 0.932 | 0.966 | 0.983 | 0.995 | 0.999 |
| UMAP ρ | 0.582 | 0.723 | 0.704 | 0.683 | — | — |

PCA improves monotonically with dimension. **UMAP does not improve — it plateaus around
0.70 and slightly declines.** Giving UMAP more room does not recover the lost within-type
structure, because discarding it is what UMAP's objective does: it optimises neighbourhood
topology and cluster separation, not metric faithfulness.

### Silhouette score actively points the wrong way

On `gy6or6`, UMAP has the **best** silhouette of all three (0.355 vs PCA's 0.317) while
having the **worst** within-type fidelity (0.662 vs 0.996). And for PCA, silhouette *falls*
as within-type fidelity rises (0.411 at 16d with ρ=0.966; 0.295 at 128d with ρ=0.999).

Silhouette rewards tight, well-separated clusters — that is, it rewards exactly the
within-type compression that destroys the drift signal. **For this project silhouette is
not merely uninformative, it is anti-correlated with what matters.** It is reported here
only to document that.

## Gate verdict

Criteria set before running, and the outcome:

| Criterion | Threshold | PCA (64d) | UMAP (8d) |
|---|---|---|---|
| Held-out label recovery | ≥ 0.95 | 0.992 / 0.998 ✅ | 0.990 / 0.995 ✅ |
| Cross-day label recovery | ≥ 0.95 | 0.995 / 0.985 ✅ | 0.993 / 0.980 ✅ |
| Within-type distance ρ | ≥ 0.95 | 0.996 / 0.997 ✅ | 0.662 / 0.450 ❌ |

# PASS — using PCA (≥64 components) as the drift measurement space.

**UMAP is rejected as a measurement space** and should be used only for visualisation. This
matters because the brief pointed at the AVGN approach (learned latent + UMAP), and UMAP is
the field-standard choice for songbird repertoire work. It is an excellent tool for *seeing*
repertoire structure; it is the wrong tool for *measuring* within-type change, and a drift
metric computed in UMAP space would have been substantially attenuated with no warning
sign — its label recovery and silhouette both look fine.

## Decisions carried into Phase 3

- **Measure drift in PCA space with ≥64 components** (95.1% variance at 32, 97.3% at 64).
  Report the within-type ρ alongside any drift number.
- **Syllables are time-padded, never time-stretched.** Stretching to a fixed length erases
  duration, which is itself a drift-sensitive property.
- **Amplitude is normalised away** within a 60 dB range below each syllable's peak.
  Recording gain is not song, and gain drifting across days would otherwise read as drift.
- Boundaries remain trustworthy only to ±5–10 ms (Phase 1), and labeller instability
  contributes ≈0.33% syllable error at 2-day separation (Phase 1).

## Limitations

**The reference is the raw spectrogram**, not perceptual or physiological distance. An
embedding that fails to preserve spectrogram distance has certainly lost information; one
that preserves it has not thereby been proven perceptually faithful. Spectrogram distance
is a proxy, and it is the input representation the embedding was built from.

**Label recovery saturated at 99%**, so criterion 1 discriminated nothing here. Canary
(20–30 syllable classes vs 8–11) would be a genuinely harder test and should be run before
this conclusion is generalised beyond Bengalese finch.

**No VAE was trained.** The comparison is pixels / PCA / UMAP. A learned non-linear
embedding might preserve within-type structure better than UMAP while compressing further
than PCA — but on this evidence PCA already reaches ρ = 0.996 with saturated label
recovery, so a VAE would have to justify itself against a linear method that is already
close to lossless for this purpose.

## Reproducing

```bash
python scripts/phase2_build_syllable_set.py --root data/bfsongrepo --bird gy6or6 --out results/phase2/gy6or6_syllables.npz
python scripts/phase2_fidelity.py --data results/phase2/gy6or6_syllables.npz --n-pca 64 --out results/phase2/fidelity_gy6or6.json
```

113 tests, each written before its implementation: `python -m pytest tests/ -q`.
