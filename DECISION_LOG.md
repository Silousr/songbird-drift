# Decision Log

Every methodological choice and why. Newest phase at the bottom.

---

## Phase 0 — Data audit (2026-08-03)

### D0.1 — Audit before building, and verify claims rather than relaying them

Four parallel research agents surveyed figshare, Dryad, Zenodo, OSF, Dataverse, CRCNS and
G-Node. Their findings were then **re-derived locally** wherever a number was load-bearing.

This caught two real errors that would have propagated into planning:

- grn394's raw audio was reported as spanning 55–140 dph across 61 day-directories. Reading
  the ZIP64 central directory directly over HTTP range requests gives **55–114 dph, 57
  directories, 78,476 wav files**. The overstatement was 26 days at the most valuable
  (post-crystallization) end.
- The Koumura deposit is widely cited as 10 birds; it has **11**. The figshare `/files`
  endpoint paginates silently at 10 records, which appears to be the origin.

**Why it matters:** a data audit whose numbers are secondhand is not an audit. Every claim
in `PHASE0_DATA_AUDIT.md` now carries a provenance marker, and the Duke figures are
regenerable via `scripts/phase0_duke_inventory.py`.

### D0.2 — Inventory the Duke deposit from `_segs.zip`, not the raw audio

The deposit is 416 GB, 290.6 GB of it audio. The per-bird segmentation archives total
~133 MB and contain one text file per recording, path-encoded with day-post-hatch and
filename-encoded with a full timestamp.

**Choice:** derive the entire longitudinal inventory (183 bird-days, 251,061 files,
5,318,154 sounds, 163.1 hours) from the segmentation archives alone.

**Why:** a complete time-axis inventory for 0.03% of the download. The alternative —
downloading audio to find out what days exist — was never justified, and would not have
fit on disk anyway.

### D0.3 — Parse the explicit `HH_MM_SS` filename suffix, not the serial fraction

Duke filenames look like `grn475_44260.72173850_03_05_20_02_53.txt`. The integer part is an
Excel-style serial date (epoch 1899-12-30): 44260 = 2021-03-05, matching `_03_05_`. The
**fractional part is not time-of-day** — `0.72173850` implies 17:19 while the suffix says
20:02:53, and the discrepancy varies between files.

**Choice:** parse the explicit suffix; use the integer serial only as a cross-check.
Across all 251,061 filenames the serial and the suffix agree on the date in **100%** of
cases, so the cross-check is a genuine integrity test and is asserted in the script.

**Why:** silently mis-derived timestamps are the single most dangerous failure mode for a
drift metric — they would corrupt the time axis while leaving every number plausible.

### D0.4 — Treat the Duke VAE latents as a prototyping shortcut, not as the pipeline

`_proj.zip` ships 32-D VAE latents and 2-D UMAP coordinates as plain HDF5 (~970 MB for all
five birds), already computed by the original authors.

**Choice:** use them to prototype and sanity-check the drift metric against a known-large
effect early, but **do not** let them stand in for our own embedding. The deliverable is a
toolkit someone runs on *their* birds, which requires our own audio → embedding path.

**Why:** measuring drift in someone else's latent space validates their embedding, not
ours. Phase 2's fidelity gate has to run on an embedding this toolkit produces.

### D0.5 — Do not treat Duke's syllable labels as human annotations

Duke's per-rendition type labels were assigned by **inspecting UMAP clusters**, not by
per-rendition hand-labeling. They are also locked inside MATLAB `table` objects
(`{bird}_table.mat`), serialized as MCOS opaque class data: `scipy.io.loadmat` returns a
`MatlabOpaque` with the payload in a 70 MB `__function_workspace__` blob, and
`pymatreader` fails outright. No MATLAB or Octave available.

**Choice:** Phase 1's "recover the existing hand labels" targets the Bengalese finch and
canary datasets. For Duke, prefer re-deriving labels by clustering the provided latents —
which reproduces the authors' documented method — over writing an MCOS parser.

**Why:** the cluster-inspection route is the documented provenance of those labels, so
reproducing it is faithful rather than a workaround, and it doubles as the Phase 2 fidelity
check. Reserve the MCOS parser as fallback; email the authors for a CSV as the cheap option.

### D0.6 — Three independent noise floors, not one pooled estimate

Adult unmanipulated data available: canary (3 birds × 9–11 d), Bengalese finch (4 × 3–5 d),
and grn394 post-crystallization (1 × 19 d) — **68 adult bird-days across 8 birds**.

**Choice:** estimate the day-to-day noise floor separately in each, and report the three
side by side rather than pooling into one number.

**Why:** they differ in species, recording rig, annotation provenance and sampling rate
(32 kHz vs 44.1 kHz). Divergent floors across the three would be a finding about the
metric's robustness; a pooled number would hide it. This is the load-bearing statistic for
the whole project — it should be stress-tested, not averaged.

### D0.7 — Flag `gr41rd51` as not-provably-baseline

Its Bengalese finch filenames embed an evTAF template (`3part_SYLc_th4191_belowhits`) while
sibling birds read `baseline`. A loaded detection template does not prove white noise was
delivered, but absence was not established either.

**Choice:** exclude from the null until confirmed with the authors; do not pool silently.

**Why:** a bird receiving contingent auditory feedback inside the "unmanipulated" set
inflates the noise floor, which would make the tool look *less* sensitive than it is and
silently raise the minimum detectable effect. Erring toward exclusion is the conservative
direction.

### D0.8 — Enter Phase 1 on the smallest dataset, not the most valuable one

**Choice:** start with the Bengalese Finch Song Repository (7.6 GB) rather than Duke or
canary.

**Why:** it is hand-labeled, 32 kHz, crowsetta-native, and is the dataset the `vak` tutorial
itself uses. Segmentation failures there are attributable to our pipeline rather than to
the data — which is exactly what a first integration needs. Scale to canary, then Duke.

### D0.9 — Python 3.12 via `uv`

`vak` 1.1.0 requires ≥3.12; system Python is 3.9.6. Environment pinned to 3.12.13 in
`.venv` via `uv`.

**Why `uv` over conda:** already present on this machine, lockfile-based, and materially
faster. Revisit only if a dependency needs conda-only binaries.

### D0.10 — GO, with the species-mismatch limitation stated up front

Public data supports within-bird drift analysis, so Phase 1 proceeds. Two limitations are
recorded now so they are not discovered later as if they were results:

1. The longest unmanipulated adult run available anywhere is **11 consecutive days**. A
   power analysis for a weeks-long experiment necessarily extrapolates the noise floor
   beyond its observed window. This must appear as a stated limitation of the sensitivity
   curves, not be buried inside them.
2. The positive control is **zebra finch development**; the intended application is
   pharmacological manipulation of **crystallized adult** song. Recovering a developmental
   effect demonstrates sensitivity, not calibration for adult drift. Both get reported.

No public dataset provides a pharmacological critical-period manipulation in adult
songbirds with annotated audio. Deafening (Zai et al.) is the closest proxy.

---

## Phase 1 — Ingestion and segmentation (2026-08-03)

### D1.1 — Cross-check the two date encodings instead of trusting either

Filenames carry DDMMYY; their directory carries MMDDYY. `parse_recording_filename`
accepts the directory name and **raises** `DateMismatchError` on disagreement rather than
picking a winner.

**Why:** a silently mis-derived recording date corrupts the time axis while leaving every
downstream number plausible. For a project whose entire output is "how much did this bird
change between day A and day B", that is the highest-consequence failure available, and
it is invisible unless checked explicitly. On the real data the two encodings agree for
all 821 files loaded.

### D1.2 — Count unannotated audio rather than dropping it silently

`load_day` records `n_audio_files`, `n_annotated_files` and `n_unannotated_files` in
`table.attrs`.

**Why:** annotation coverage here is partial and uneven — the authors note 882 `gr41rd51`
files carry no annotation. A loader that simply globs the CSVs makes a half-annotated day
indistinguishable from a fully-annotated one, which would silently bias any per-day
statistic. The days loaded so far happen to be at 100% coverage, which is worth knowing
rather than assuming.

### D1.3 — Set segmentation parameters from the ground-truth distribution

Measured from the hand labels before choosing anything: 1st-percentile inter-syllable gap
**9.3 ms**, median 25.6 ms, and **24.8% of gaps fall below 20 ms**; 1st-percentile
syllable duration **32.9 ms**. Chosen: `min_silence_s=0.005`, `min_syllable_s=0.02`.

**Why:** the conventional 20 ms `min_silence` would have merged roughly a quarter of all
adjacent syllable pairs into single segments. That error would have propagated into every
later stage as a systematically reduced syllable count, and it would have looked like a
property of the bird rather than of the parameter.

### D1.4 — Treat the segmentation result as algorithm agreement, not ground truth

Pooled segment F1 **0.978** (precision 0.980, recall 0.975) over 14,681 syllables across
8 bird-days, at 10 ms tolerance, generalising to held-out days and a second bird
(0.943–0.996) with no retuning.

**This number is weaker evidence than it looks and is labelled as such in the script.**
The reference boundaries were themselves drawn with `evsonganaly`, an amplitude-threshold
segmenter. Reproducing them shows this implementation recovers *that algorithm's* output,
not that either finds the acoustically correct syllable edge.

**Why it still matters:** it validates the ingestion path end-to-end on real audio, and it
sets the boundary-precision budget for later stages — agreement collapses from 0.995 to
0.920 to 0.458 as tolerance tightens from 10 to 5 to 2 ms, so **syllable boundaries are
only trustworthy to roughly ±5–10 ms**. Any drift feature that depends on finer timing
than that is measuring segmentation noise. The non-circular test is label recovery.

### D1.5 — Exclude off-schema label characters, and report the cost

`labelset = 'iabcdefghjk'` for `gy6or6`, excluding `0` (n=144), `x` (3), `y` (2), `z` (1).
These mark contact calls and unclear sounds, not song syllables. vak drops whole files
containing them: **9 of 162 files (5.6%)** on the training day, 4 on the evaluation day.

**Why:** mixing "not a song syllable" into a syllable-type inventory would corrupt both
the embedding and the syntax statistics. The cost is recorded here because it is a
selection effect on which files enter the analysis, not a free choice.

### D1.6 — CPU, because vak and Apple Silicon MPS are incompatible

vak rejects `accelerator = 'mps'` (accepts only cpu/gpu/tpu/ipu), and routing through
`'gpu'` — which Lightning maps to MPS on Apple Silicon — fails at the first batch:
`Cannot convert a MPS Tensor to float64 dtype`. vak's spectrograms are float64.

**Choice:** train on CPU. Batch size raised 8 → 32 and epochs cut to 1, since vak slides a
window over *every* timebin: one epoch is ~247k windows (7,722 batches), which is ample.

**Why this is worth recording:** it is a hard constraint on anyone reproducing this on a
Mac, and it sets the practical ceiling on how much model training this project can do
locally. Phases 2–4 should assume CPU-only unless a CUDA machine is available.

### D1.7 — Evaluate the labeller on a *different day*, not just a held-out split

The trained model is evaluated both on a within-day test split and on day 032512, two
days after the training day.

**Why:** this is the load-bearing check for the whole project. If an automatic labeller
degrades on later days, that degradation is a time-varying artefact that would appear as
song drift in a bird that has not changed. Phase 3's noise floor must therefore include
labeller instability, or be bounded by it. A within-day test split cannot reveal this.

### D1.8 — Report the labeller's cross-day gap as a floor, not a footnote

TweetyNet trained on `gy6or6` 032312 and evaluated on 032512 (two days later) gives frame
accuracy 0.9887 vs 0.9888 within-day, and syllable error rate 0.0033 vs 0.0015.

**Choice:** record **0.33% syllable error at 2-day separation** (of which ~0.18 points
comes from crossing days) as an explicit floor that Phase 3 drift must exceed, and refuse
to extrapolate it beyond 2 days.

**Why:** an automatic labeller that degrades over time produces apparent song change in a
bird that has not changed. Here it does not degrade materially, which is the result that
makes automatic labelling usable at all for this project — but the test covers one bird,
11 classes and a 2-day gap, because the Bengalese finch repository spans at most 5 days.
Whether it holds at 10-day (canary) or 60-day (developmental) separations is untested.
Assuming it does would smuggle an unmeasured artefact into the noise floor.

### D1.9 — Boundary timing budget of ±5–10 ms

Segment F1 falls 0.997 / 0.995 / 0.920 / 0.458 at tolerances of 20 / 10 / 5 / 2 ms.

**Choice:** treat ±5–10 ms as the resolution limit of any syllable boundary, and reject
Phase 2 features or Phase 3 statistics that depend on finer timing.

**Why:** below that tolerance the segmenter and the human annotation simply disagree, so a
feature resolving to 2 ms would be reporting segmentation noise as signal. Better to bound
this now than to discover it as an unexplained variance component in Phase 3.

---

## Phase 2 — Embedding and fidelity gate (2026-08-03)

### D2.1 — Test within-type distance fidelity, not just label recovery

The gate has two halves. Between-type fidelity (can human labels be recovered?) is the
one usually reported. Within-type fidelity — do distances *inside* a syllable type track
acoustic differences? — is measured here by `within_type_distance_correlation`.

**Why:** drift in crystallised song is a within-type phenomenon. Renditions of syllable
`b` change shape while staying recognisably `b`. An embedding that maps every rendition
onto its type centroid classifies perfectly and carries no drift signal whatsoever. Label
recovery is structurally blind to that failure.

This was not hypothetical. On these data **k-NN label recovery was 0.9917 for every
representation tested** — pixels, PCA at 8/16/32/64/128 dimensions, UMAP at 2/8/16/32.
Used alone, criterion 1 would have declared them all equally fit.

### D2.2 — Measure drift in PCA space; reject UMAP as a measurement space

Within-type ρ: PCA-64 scores **0.996** (`gy6or6`) and **0.997** (`or60yw70`). UMAP-8
scores **0.662** and **0.450**.

Raising UMAP's dimensionality does not help — ρ goes 0.582 (2d), 0.723 (8d), 0.704 (16d),
0.683 (32d), i.e. it plateaus and then declines, while PCA rises monotonically to 0.999 at
128d. The distortion is intrinsic: UMAP optimises neighbourhood topology and cluster
separation, and within-type metric structure is what it trades away.

**Choice:** compute drift in PCA space with ≥64 components. Use UMAP for visualisation only.

**Why this is worth stating loudly:** the brief points at the AVGN approach (learned latent
+ UMAP), and UMAP is the field-standard representation for songbird repertoire work. A
drift metric computed in UMAP space would have been badly attenuated **with no warning
sign** — its label recovery (0.993) and silhouette (best of all three) both look healthy.
This is precisely the "measuring noise" failure the gate was put there to catch.

### D2.3 — Do not use silhouette score to select an embedding here

On `gy6or6` UMAP has the best silhouette (0.355 vs PCA's 0.317) and the worst within-type
fidelity (0.662 vs 0.996). Within PCA, silhouette *falls* as fidelity rises: 0.411 at 16d
(ρ=0.966) versus 0.295 at 128d (ρ=0.999).

**Choice:** report silhouette for documentation, never as a selection criterion.

**Why:** silhouette rewards tight, well-separated clusters — which is exactly the
within-type compression that destroys the drift signal. For this project it is not merely
uninformative, it is anti-correlated with the objective.

### D2.4 — Time-pad syllables rather than time-stretching them

Syllables are padded to a fixed 200 ms window and truncated beyond it, not rescaled to a
common length.

**Why:** time-stretching normalises duration away, and syllable duration is one of the
acoustic properties most likely to move when song destabilises. Padding keeps duration
visible to the embedding. The cost is that syllables above 200 ms are truncated; at the
measured 99th percentile of 123 ms this affects very few.

### D2.5 — Normalise amplitude away, deliberately

Each syllable is normalised over a fixed 60 dB range below **its own** peak, discarding
absolute amplitude.

**Why:** recording gain is not song. Gain that drifts across days — a mic nudged, a
preamp adjusted — would otherwise be indistinguishable from the drift being measured. The
cost is that genuine changes in song amplitude become invisible; that is the right trade
when the alternative is a confound that mimics the signal.

### D2.6 — Excluded off-schema labels by identity, not by frequency

`0`, `x`, `y`, `z` and similar are dropped explicitly rather than by a rarity threshold
(173 syllables for `gy6or6`, 134 for `or60yw70`).

**Why:** a pure count threshold would silently delete genuinely rare *song* syllables
along with the noise classes. Excluding by identity keeps the criterion about what the
symbol means rather than how often it happens to occur.

---

## Phase 3 — Drift metric and noise floor (2026-08-03)

### D3.1 — Bias-correct the centroid distance, and make the BOUT the sampling unit

The squared distance between two sample means over-estimates the distance between the
distribution means by `tr(Σa)/na + tr(Σb)/nb`. Subtracting those terms removes the bias —
but the correction assumes the sampling unit is independent, and syllables are not: they
arrive in bouts, and renditions within a bout are correlated.

Measured on the clustered test fixture, with true drift of zero:

| correction applied per… | mean estimate | fraction negative |
|---|---|---|
| rendition (clustered data) | **+1.41** | **0.00** |
| bout (clustered data) | −0.00 | 0.50 |

**Choice:** bouts are the unit for the variance correction, the bootstrap, and the null.

**Why:** the first row is drift manufactured entirely from within-bout correlation, and
because it never goes negative it would never look like noise. It would also have been
worst on the quietest days, since the bias grows as samples shrink — and recording volume
varies several-fold between days here (39 to 248 songs/day). That is a bias that
correlates with an experimental variable, which is the most dangerous kind.

**Consequence for Phase 4:** precision scales with the number of *bouts*, not syllables.
More syllables inside the same bouts buys little. "Minutes of song" is the wrong unit for
a power curve.

### D3.2 — Never clip the drift estimate at zero

The unbiased estimate is negative roughly half the time under the null.

**Why:** clipping restores exactly the upward bias the correction removes, and converts a
symmetric null into a one-sided pile-up at zero that no longer supports a calibrated test.
Negative values are information — they say the observed separation is smaller than
sampling noise alone would produce.

### D3.3 — Fit the embedding space once, not per day

PCA is fitted on a single reference day and applied to all days.

**Why:** re-fitting per day moves the axes, and axis movement is indistinguishable from
song movement. Verified empirically: refitting the space on each of the five days in turn
changes drift estimates only in the third decimal, confirming the measurements are not an
artefact of which day defines the space.

### D3.4 — Exclude two nominally-baseline days as contaminated

`gy6or6/2012-03-22` and `or60yw70/2012-10-01` each show 7.6–8.0× the drift of any other
same-gap pair in their bird. Three competing explanations were tested and rejected:
PCA reference day (ruled out by refitting on all five days), time-of-day confound (ruled
out — restricting to a common window makes the anomaly *larger*), and a few noisy syllable
types (ruled out — essentially every type moves together).

**Choice:** exclude both from the noise floor, and report the exclusion prominently.

**Why:** the dataset documents itself as *"baseline recordings for behavioral experiments
that are not included in this dataset"*, and `gy6or6/032212` is the same directory Phase 1
found to contain 10 files templated `washout` — a post-perturbation recovery phase. Keeping
`gy6or6/032212` would have inflated that bird's apparent day-to-day drift roughly
eightfold, which would then have been reported as the noise floor and made the tool look
far less sensitive than it is.

**The general lesson, recorded for the wet lab:** two of five days in a curated public
"baseline" repository were unusable as baseline. Genuine baseline days must be
demonstrably distant from any manipulation and recorded on a constant schedule — including
time-of-day coverage, which differed sharply on both excluded days (2–3 h morning-only
versus 10–14 h elsewhere).

### D3.5 — Report drift standardised by within-type variance

Drift is divided by the pooled within-type variance on the reference day.

**Why:** raw squared PCA distance is in arbitrary units that depend on the spectrogram
scaling and the number of components, so it is not comparable across birds or studies.
Dividing by within-type variance gives a dimensionless effect size — centroid movement in
units of natural rendition-to-rendition variation — which is what a power analysis needs
and what another lab could compare against.

### D3.6 — The 0.62 negative fraction is skew, not residual bias

The null distribution has mean ≈ 0 but is negative ~62% of the time in all three birds.

**Why this is expected:** the statistic is a difference of quadratic forms and is therefore
right-skewed, so its median falls below its mean. Unbiasedness is a claim about the mean,
and the means are −0.0002, +0.0075 and +0.0038. Recorded so the asymmetry is not later
mistaken for a defect and "fixed" by someone re-introducing bias.

---

## Phase 4 — Sensitivity analysis (2026-08-03)

### D4.1 — Estimate power by injection into real data, not from a formula

Each draw samples real bouts, splits them in half, displaces one half by a known
standardised amount, and tests whether the statistic clears its critical value.

**Why:** a parametric power formula assumes independent syllables. Phase 3 measured a ~10x
design effect from within-bout correlation, so a formula-based calculation would promise an
experiment far more sensitive than it can be — the worst possible direction for a tool
whose purpose is planning.

Threshold and power use independent draws, so the critical value is never tuned to the
draws it is applied to.

### D4.2 — Evaluate effect sizes in closed form rather than re-simulating

The injected shift enters the statistic analytically: `stat(s) = baseline − 2·s·d + ||s||²`.
Storing `baseline` and the projection `s·d` per draw lets the whole effect grid be
evaluated exactly from one simulation.

**Why:** re-simulating each grid point was too slow to sweep bouts × types × subsets, and
it added Monte-Carlo noise between grid points that showed up as a non-monotonic MDE curve.
The closed form is exact, not an approximation, and removed the artefact.

### D4.3 — Average over random subsets of syllable types when varying K

For K < all, MDE is the median over 9 random type subsets rather than the first K
alphabetically.

**Why:** syllable types differ in variance and in how many bouts contain them, so a fixed
subset makes the K columns incomparable — the first version produced a K=3 column *worse*
than K=1, which was an artefact of which types happened to be chosen.

### D4.4 — Report recording volume in bouts, with minutes only as a conversion

**Why:** the estimator's sampling unit is the bout, so precision tracks bout count.
"Minutes of song" is what a lab naturally plans in, but two protocols with equal minutes
and different bout counts have very different power. Minutes are derived from each bird's
measured median annotated song per bout (4.4–8.9 s), computed on the **full** annotations —
deriving it from the 2,000/day analysis subsample understated song time several-fold, since
the subsample keeps every bout but only a fraction of each bout's syllables.

### D4.5 — Report the saturation point, not just the curve

Beyond ~80 bouts per timepoint, measurement precision (MDE 0.020) is finer than the bird's
own unmanipulated day-to-day drift (0.013–0.035).

**Why this belongs in the deliverable:** the naive reading of a sensitivity curve is "more
data is always better", which would send a lab into long recording sessions that cannot
strengthen any claim. Past saturation, additional power has to come from more birds or
longer separations. Stating where the curve stops paying is more useful than the curve
alone.

### D4.6 — State plainly what the metric cannot see

The injected effect is a rigid displacement of the distribution. The curves therefore
describe sensitivity to a **centroid shift** only.

**Why it matters:** a manipulation that increased rendition-to-rendition *variability*
without moving the mean would be invisible to this statistic — and increased variability is
a plausible signature of a reopened critical period, arguably more plausible than a clean
directional shift. Recorded as the most valuable extension rather than left for someone to
discover after running the experiment.

---

## Validation — Recovering a known effect (2026-08-03)

### V1 — Report the negative result, and diagnose it rather than reframing until it passes

Consecutive-day drift does **not** distinguish learning from crystallised song in `grn394`
(median 0.0072 vs 0.0073, ratio 1.0x). Two candidate explanations were tested and rejected
before accepting the negative: restricting to song-like sounds via endpoint clustering gave
0.0135 vs 0.0187 (ratio 0.72x, still the wrong direction), and total within-day variance was
flat (ratio 0.95x), so the effect is not hiding in dispersion either.

**Why this is recorded prominently:** it would have been easy to run only the test that
worked (distance to a fixed reference) and report a clean success. The negative is the more
useful half — it shows the consecutive-day framing is underpowered for real developmental
change, which is a property of the *design*, not of this dataset, and which would otherwise
have surfaced only after a wet-lab experiment had been run.

### V2 — Compare each timepoint to a pooled baseline, not to the preceding timepoint

Distance to a fixed crystallised reference recovers the effect: 2.0x closer to adult song
late versus early, with non-overlapping confidence intervals and a significant slope
(-0.00111/day, p=1.5e-3).

**Why:** a consecutive comparison sees one interval's change buried in noise; a fixed
reference accumulates all change since baseline while the noise does not grow with it.
This is consistent with Phase 3, where 1-day drift sat below the noise floor and only
cleared it at ~3 days.

**Carried into the experimental design recommendation:** compare every timepoint against a
pooled baseline. On this data it is the difference between detecting the known effect and
missing it entirely.

### V3 — Use blocks of consecutive batch files as the clustering unit, not reconstructed bouts

Duke's latents are batched 20-per-file and the batch-to-bout mapping cannot be recovered
exactly: AVA writes only complete batches (so most days are truncated to a multiple of 20)
and two days have further gaps beyond that.

**Choice:** cluster by blocks of 10 consecutive batch files rather than attempting to
reconstruct bouts.

**Why:** a block is *coarser* than a bout, so it can only widen confidence intervals — the
safe direction. A reconstructed mapping that was subtly wrong would corrupt precisely the
clustering structure the estimator depends on, which is the failure mode D3.1 exists to
prevent. Better a conservative unit than a clever one that might be wrong.

### V4 — State that this validates the statistic, not the toolkit's own embedding

The analysis uses the original authors' VAE latents.

**Why it still counts:** the drift statistic — bias correction, bout clustering, bootstrap
intervals — is what Phases 3 and 4 built and what a wet lab would rely on, and it is
exercised end-to-end here against an independently documented effect. But validating our
own audio-to-embedding path would need the raw audio (291 GB against 30 GB free), so the
claim is deliberately limited. Phases 1-2 establish that path separately on Bengalese finch.

---

## Phase 5 — Dispersion drift (2026-08-03)

### D5.1 — Add a second metric for variability, not just location

The log variance ratio `log(tr Var(b) / tr Var(a))` is exactly zero under a pure
translation, so it measures something the centroid distance structurally cannot.

**Why:** Phase 4 recorded that a manipulation increasing rendition-to-rendition
variability without moving the mean would be invisible — and destabilised renditions are a
plausible signature of a reopened critical period. The validation then confirmed the
concern was real: on `grn394`, consecutive-day centroid drift gave a learning/crystallised
ratio of 1.0x (no effect), while dispersion gives 1.43x against a 1.15x floor.

Measured independence across three birds: Spearman rho = +0.05, +0.22, +0.41 (p = 0.69,
0.14, 0.067). Weak and non-significant, so the two statistics are largely seeing different
things and both are worth carrying.

### D5.2 — No closed-form bias correction; rely on the bootstrap and split-half null

Unlike the centroid distance, dispersion has no analytic correction applied here.

**Why:** the quantity of interest is *total* rendition-to-rendition variance, which
includes the within-bout component that collapsing to bout means would discard. Rather
than invent a correction, uncertainty comes from a bout-level bootstrap and the null from
splitting a day's bouts in half — both keeping the bout as the sampling unit. The null is
naturally centred on zero because a ratio of two halves of the same day is symmetric about
1 (measured median +0.0001).

### D5.3 — Match selection intensity across days, and sweep it

Restricting to sounds near a syllable mode by a **fixed distance** keeps ~12% of sounds
early in development and ~70% late. Keeping the same *fraction* per day removes this.

**Why this is a decision and not a detail:** the fixed-distance version produced a
result of 0.90x — juvenile song apparently *less* variable than adult song — which
exceeded the noise floor and would have read as a genuine, surprising finding. It is
entirely an artefact of carving a tight core from a broad early distribution while keeping
nearly all of a narrow late one. Matching the fraction flips the sign to the predicted
direction, stable at 12%, 30% and 60%.

**The general rule now applied:** when a filter's stringency varies systematically with the
variable under study, it manufactures an effect in that variable. Any per-day quality
threshold — amplitude, SNR, cluster distance — carries this risk and must be swept rather
than trusted at one setting.

### D5.4 — Report both metrics against their own floors, including when both are null

Centroid floor ~0.041 standardised; dispersion floor ~0.22–0.73 in |log variance ratio|,
and bird-specific (`bl26lb16`'s is 3x the others, from fewer days and types).

**Why:** a manipulation may move either, both, or neither. "Neither" is only interpretable
as evidence of no change if both were actually measured — otherwise it is just the absence
of the one test that was run.
