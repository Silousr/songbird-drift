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
