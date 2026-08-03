# Phase 0 — Public Data Audit

**Date:** 2026-08-03
**Question:** Does any public dataset support within-bird song drift analysis over time?
**Answer: GO** — with a specific dataset assignment and three caveats that change the build order.

---

## Provenance markers

Every quantitative claim below carries its evidence level. This matters because several
widely-repeated numbers turned out to be wrong.

| Marker | Meaning |
|---|---|
| **[COMPUTED]** | I downloaded the file and computed this number in this session. Reproducible via `scripts/phase0_duke_inventory.py`. |
| **[API]** | Read directly from the repository's REST API or embedded record metadata in this session. |
| **[DOC]** | Quoted from the dataset's own documentation file, which I downloaded. |
| **[REPORTED]** | From a research agent's fetch, corroborated by a URL but not re-derived by me. |
| **[UNVERIFIED]** | Could not confirm. Stated as unknown, not assumed. |

Two corrections I made to earlier research during verification:

- grn394's raw audio spans **55–114 dph, 57 day-directories** [COMPUTED, ZIP64 central
  directory read over HTTP range requests], *not* 55–140 dph / 61 directories as first
  reported. A 26-day overstatement at the most valuable end of the trajectory.
- The Koumura dataset is commonly cited as 10 birds; it contains **11** (`Bird0`–`Bird10`)
  [API]. The error appears to come from the figshare `/files` endpoint silently paginating
  at 10 records.

---

## Verdict summary

| Dataset | Longitudinal? | Role in this project |
|---|---|---|
| **Duke juvenile zebra finch** (Brudner/Pearson/Mooney) | **YES — 183 bird-days** | **Primary.** Positive control + in-dataset noise floor |
| **TweetyNet canary** (Cohen et al.) | **YES — 31 bird-days** | Adult null; hand-annotated; Phase 1 label recovery |
| **Bengalese Finch Song Repository** (Nicholson/Queen/Sober) | **YES — 18 bird-days** | Adult null; hand-annotated; smallest, start here |
| **Zai et al. deafening + white noise** | **YES — 76 birds** | Known-effect validation. Unverified interior; see risk |
| **Koumura BirdsongRecognition** | **NO — no time axis at all** | Segmentation benchmark only. Cannot support drift |

---

## 1. Duke — Juvenile zebra finch development ⭐ PRIMARY

**"Data from: Juvenile zebra finch syllables for data-driven analysis of development"**
Brudner, Pearson & Mooney · DOI [10.7924/r4j38x43h](https://doi.org/10.7924/r4j38x43h) ·
**CC0 1.0** [API] · record `research.repository.duke.edu/record/135` ·
paper: [PLoS Comput Biol 19(5):e1011051](https://doi.org/10.1371/journal.pcbi.1011051)

**Total deposit: 416.48 GB across 37 files** [API]. Raw audio alone is 290.6 GB.

### Longitudinal structure — the strongest available, by a wide margin

All figures below **[COMPUTED]** by parsing every segment filename in the five `_segs.zip`
archives (241 MB downloaded, 251,061 files parsed, 0 unparsed, 0 date mismatches):

| Bird | Recorded days | dph range | Calendar span | Wav files | Segmented sounds | Sound minutes |
|---|---|---|---|---|---|---|
| `grn394` | **56** | 56–114 | 2019-10-18 → 2019-12-15 | 78,475 | 1,655,644 | 3,467 |
| `grn395` | 40 | 56–105 | 2019-10-18 → 2019-12-06 | 48,885 | 1,451,017 | 2,311 |
| `grn397` | 36 | 56–94 | 2019-10-18 → 2019-11-25 | 65,859 | 1,145,613 | 2,192 |
| `grn475` | 25 | 72–97 | 2021-02-21 → 2021-03-18 | 24,683 | 472,018 | 781 |
| `sil469` | 26 | 72–97 | 2021-01-27 → 2021-02-21 | 33,159 | 593,862 | 1,032 |
| **Total** | **183 bird-days** | | | **251,061** | **5,318,154** | **163.1 hours** |

Median **37–69 minutes of actual segmented sound per bird per day** [COMPUTED] — an
unusually dense sampling that makes the Phase 4 subsampling curves feasible without
extrapolation. Median syllable duration 84–122 ms [COMPUTED].

Date is recoverable two independent ways and they agree exactly: the directory name is
integer dph, and the filename embeds an Excel-serial date plus `MM_DD_HH_MM_SS`. Across
all 251,061 files, serial-vs-filename date mismatches: **0** [COMPUTED]. Time-of-day is
therefore free, which matters — circadian effects are a known confound in song variability.

Real filename, verbatim: `segs_undir/84/grn475_44260.72173850_03_05_20_02_53.txt`
(serial 44260 = 2021-03-05, matching the `_03_05_`). Note the fractional part of the
serial is **not** time-of-day — it disagrees with the explicit `HH_MM_SS` by hours and
varies. Parse the explicit suffix, not the fraction.

### Why this dataset carries both halves of the design

`grn394` runs from 56 to 114 dph. Zebra finch song crystallizes around ~90 dph. So a
single bird, on one recording rig with one pipeline, gives:

- **dph 56–90 — the positive control.** Full sensorimotor learning through
  crystallization. This is a large, well-documented, directional effect. If the drift
  metric cannot see this, the metric is broken.
- **dph 95–114 — the candidate null (19 recorded days)** [COMPUTED]. Post-crystallization
  song from the same bird, same chamber, same segmentation. This is the cleanest possible
  noise floor: it holds constant every nuisance variable that a cross-dataset comparison
  would confound.

That structure is exactly what Phase 3 requires and I did not expect to find it in one
deposit. **Caveat: only `grn394` has a substantial post-crystallization stretch.** `grn395`
and `grn397` effectively end at 94 dph; their apparent 102–105 dph days hold 1–44 files
[COMPUTED] and are unusable. So the in-dataset adult null is **n=1 bird**. It must be
combined with the canary and Bengalese finch nulls, not used alone.

### Format

- **Sampling rate: 44,100 Hz for 3 birds (SAP), 32,000 Hz for 2 birds (EvTAF)** [DOC].
  Mixed rates within one deposit — the pipeline must not assume a global rate.
- `_segs.zip` — AVA amplitude segmentation, plain text onset/offset pairs in seconds,
  one file per wav. Readable, no MATLAB.
- `_proj.zip` — **32-D VAE latents plus 2-D UMAP coordinates as plain HDF5** [COMPUTED:
  `latent_means` shape (20,32) float64, `latent_mean_umap` shape (20,2) float32, 20
  syllables per file]. ~970 MB for all five birds.
- `_raw_wav.zip` — 290.6 GB, `raw/{dph}/{bird}_{serial}_{MM_DD_HH_MM_SS}.wav`.

### The one real blocker: syllable type labels are locked in MATLAB

Per-rendition syllable **type** labels, along with `age`, `datetime`, `duration`, `pca`
and `ffnn_predicted_age`, live only in `{bird}_table.mat` (429,379 rows for grn394 [DOC]).
I downloaded all five (482 MB) and **they cannot be read outside MATLAB as-is** [COMPUTED]:
they are MATLAB `table` objects serialized as MCOS opaque class data. `scipy.io.loadmat`
returns a `MatlabOpaque` with the payload buried in a 70 MB `__function_workspace__` blob;
`pymatreader` fails outright; no MATLAB or Octave on this machine.

Three ways out, in the order I would try them:
1. **Re-derive labels by clustering the provided latents.** The labels were originally
   *assigned by inspecting UMAP clusters* [DOC] — so reproducing them is the documented
   method, not a workaround. This also doubles as the Phase 2 fidelity check.
2. Write an MCOS parser for the `__function_workspace__` blob. Well-trodden but fiddly.
3. Email the authors for a CSV export.

Note what this means for scope: **these are not per-rendition human annotations.** Phase 1's
"recover the existing hand labels" applies to the canary and Bengalese finch data, not here.

---

## 2. TweetyNet canary — adult null, densely hand-annotated

Cohen et al., *eLife* 2022 · DOI [10.5061/dryad.xgxd254f4](https://doi.org/10.5061/dryad.xgxd254f4) ·
**CC0** [API] · 26.42 GB current version, 7 files [REPORTED via Dryad API]

The `storageSize` field reads 37.83 GB [API] but includes a superseded version; the
current version is 26.42 GB.

| Bird | Days | Span | Annotated files | Syllable classes |
|---|---|---|---|---|
| `llb3` | 11 | 2018-04-23 → 05-03 | 2,655 | 20 |
| `llb11` | 11 | 2018-05-04 → 05-14 | 2,031 | 27 |
| `llb16` | 9 | 2018-05-03 → 05-11 | 1,452 | 30 |

[REPORTED — derived by parsing the authors' own prep CSVs in the `yardencsGitHub/tweetynet`
repo, 6,138 filenames, 0 parse failures. Consistent with an independent reuse paper's file
counts to within 40–52 files.]

≈31.5 hours of syllable-level annotated song across 3 adult males, consecutive days,
timestamps `{bird}_{n}_{YYYY}_{MM}_{DD}_{HH}_{MM}_{SS}.wav`. One annotator and one label
scheme per bird across all days, so labels are commensurable day-to-day without
re-registration — important, and not true of every annotated corpus.

**Open question: sampling rate.** Secondary sources say 44.1 kHz, but the authors' own
spectrogram config (`step_size=64`, timebin 0.0027 s) implies ~23.7 kHz [REPORTED]. These
cannot both be right. **Read a WAV header before relying on either** — a wrong rate
silently rescales every frequency feature.

Manipulation: **none**. Ethics statement: birds "were not used in any other experiments"
[REPORTED, verbatim from eLife article JSON]. Adult, male, ages not recorded.

---

## 3. Bengalese Finch Song Repository — adult null, smallest and cleanest entry point

Nicholson, Queen & Sober · DOI [10.6084/m9.figshare.4805749](https://doi.org/10.6084/m9.figshare.4805749) ·
**CC BY 4.0** · 42 files, **7.577 GB** [API, summed]

**18 bird-days across 4 adult males** [API, from the file manifest — I re-derived this
independently]:

| Bird | Days | Dates (MMDDYY) |
|---|---|---|
| `bl26lb16` | 3 | 041912, 042012, 042112 |
| `gr41rd51` | 5 | 061912 … 062312 |
| `gy6or6` | 5 | 032212 … 032612 |
| `or60yw70` | 5 | 092712 … 100112 |

All strictly consecutive. 32 kHz, 16-bit PCM mono [REPORTED, read from a WAV header].
Ships in two parallel forms: `.cbin`+`.not.mat` (evsonganaly) and `.wav`+`.csv`
(crowsetta `simple-seq`, a 3-column `onset_s,offset_s,label`). The deposit is ~2×
redundant — same song in both formats — so unique content is ~4.26 GB.

Two caveats carried forward:
- **Annotation coverage is partial and uneven.** The authors state 882 `gr41rd51` audio
  files lack annotations [REPORTED, from the dataset docs].
- **`gr41rd51` may not be a clean baseline.** Its filenames embed an evTAF template
  (`3part_SYLc_th4191_belowhits`) while other birds read `baseline` [REPORTED]. A loaded
  detection template does not prove white noise was delivered, but it is not proven absent
  either. **Resolve with the authors before pooling this bird into a null.** This is exactly
  the kind of thing that would silently inflate a noise floor.

Directory dates are `MMDDYY` while filename dates are `DDMMYY` — reversed. Easy to corrupt
a time axis; the loader must handle it explicitly.

---

## 4. Zai et al. — deafening + white-noise reinforcement (the validation target)

"Goal-directed vocal planning in a songbird — raw data" · Zai, Stepien, Giret, Hahnloser ·
DOI [10.5281/zenodo.14732250](https://doi.org/10.5281/zenodo.14732250) · **CC BY 4.0** ·
published 2025-01-28 · **one file, `Data_WAV_goal-directed.zip`, 49,283,772,591 bytes
(49.28 GB)** [API — I verified title, DOI, license, authors, and size directly against the
Zenodo API]

Design [REPORTED, from the eLife paper]: **76 adult zebra finches**, 90–300 dph, 32 kHz;
"at least 3 days" baseline before any manipulation; **44 deafened birds** recorded 13–50
days post-deafening; white-noise reinforcement over 8–15 days.

This is the best available "known effect" for the project's stated validation goal —
deafening-induced deterioration of crystallized adult song is large and well documented,
and it is the closest public analogue to "the window reopened."

**Two risks, both real:**

1. **Interior unverified.** Zenodo returns **HTTP 403 to file-byte requests from this
   environment** [COMPUTED — confirmed by direct `HEAD` and `Range` request], so I could
   not read the ZIP central directory to confirm per-bird/per-day structure. Everything
   about the internal layout is currently an inference from the paper. A 40 MB derived
   companion at [10.3929/ethz-b-000670443](https://doi.org/10.3929/ethz-b-000670443)
   reportedly carries `birdname`, `Ndays`, `timestamps`, `annotations` — **read that first**,
   before committing to the 49 GB.
2. **It does not fit on this machine.** 43 GB free; the archive alone is 49.28 GB
   [COMPUTED]. See "Operational constraints" below.

---

## 5. Koumura BirdsongRecognition — NO-GO for drift, useful as a benchmark

DOI [10.6084/m9.figshare.3470165](https://doi.org/10.6084/m9.figshare.3470165) · CC BY 4.0 ·
1.49 GB · **11 birds**, 2,965 wav files, 15,391 sequences, **215,037 annotated notes** [REPORTED,
from parsing all 11 `Annotation.xml`]

**There is no time coordinate anywhere in this dataset** [REPORTED, established three ways:
the XSD has no date field; the parsed XML for all 11 birds has no date field; wav files are
named `0.wav`–`93.wav`; and the ZIP timestamps are batch-export artifacts — all 94 of
Bird10's files were written inside a 38-second window in 2015].

You cannot date two songs, or even order them. Excellent segmentation/classification
benchmark and a good source of syntax statistics. **Any drift claim built on it would be
unfounded.**

---

## 6. Datasets that look right and are not — the "no audio" trap

Several deposits have exactly the experimental design this project wants and contain no
usable audio [all REPORTED]:

| Dataset | Design | What is actually deposited |
|---|---|---|
| Vellema 2019 ([10.5061/dryad.kb814nh](https://doi.org/10.5061/dryad.kb814nh)) | **Canary testosterone implant → withdrawal → re-implant, ~1 year, 6 birds** — a natural window-reopening timecourse, and the single best-designed study for this project's question | Sound Analysis Pro **SQL feature databases** only. No audio |
| Toutounji et al. 2024 ([10.5061/dryad.3r2280gpp](https://doi.org/10.5061/dryad.3r2280gpp)) | Juvenile pitch-shift learning, 22 birds, 42.89 GB | `.mat` pitch values + timestamps. No audio |
| Moorman, Ahn & Kao 2021 | Chronic LMAN manipulation, multi-day song change | Entire deposit is one 43,520-byte `.xls` |
| Sasahara/Tchernichovski/Okanoya | Bengalese finch song development, 12 birds | SAP MySQL tables only |
| TweetyNet results ([10.5061/dryad.gtht76hk4](https://doi.org/10.5061/dryad.gtht76hk4)) | — | Model checkpoints only |

**Vellema is worth an email.** A year-long hormone-driven plasticity timecourse in canaries
is a closer analogue to "reopening a closed critical period" than anything else found, and
only the derived features were deposited.

### Clean negatives [REPORTED]

No public annotated audio exists for: **delayed auditory feedback**; **pitch-shifted
headphones** (the Sober lab's own repo states "the raw data is not provided");
**syrinx/tracheosyringeal denervation**; **classic deafening studies** (Nordeen, Woolley
& Rubel, Lombardino & Nottebohm — all pre-mandate). Kollmorgen et al. 2020 and Lipkind et
al. 2017 are "available upon reasonable request." G-Node/GIN and IEEE DataPort hold nothing
longitudinal. These are genuine absences, not search failures.

Additional candidates worth a look if more adult data is needed: **Trusel et al. 2026**
(Texas Data Repository, CC0, chronic TeNT + lesions, 220.9 GB, per-bird `.rar`, contents
unverified) and **Medina et al. 2022** (α-synuclein in Area X, 31.84 GB, ~12 recording
days/bird over ~3 months, `BirdID/BirdID_[Pre|Post1-3]_[Day1-3]/*.wav`).

---

## GO / NO-GO

# GO

Public data supports within-bird drift analysis, with a clean division of labor:

- **Positive control (large, directional, guaranteed):** Duke developmental trajectory.
  183 bird-days, 163 hours of segmented sound, CC0, downloadable, and — importantly — the
  VAE latents are already computed and readable without MATLAB, so a drift metric can be
  prototyped against a known-large effect before any audio is processed.
- **Noise floor (unmanipulated day-to-day):** three independent sources, deliberately not
  pooled blindly — canary (3 birds × 9–11 d), Bengalese finch (4 birds × 3–5 d), and
  `grn394` post-crystallization (1 bird × 19 d). **68 adult bird-days across 8 birds.**
  Cross-checking the floor across three species/rigs is a feature: if the metric's noise
  floor is wildly different across them, that is itself a finding about the metric.
- **Known-effect validation:** Zai deafening, subject to verifying the interior first.

### The three caveats that change the build order

1. **The adult noise floor is the binding constraint, not the drift signal.** Longest
   unmanipulated adult run available is 11 consecutive days (canary); the Bengalese finch
   gives 3–5. A wet-lab experiment measuring drift over weeks will have a noise floor
   estimated from ≤11-day windows and extrapolated. **That extrapolation must be stated as
   a limitation of the power analysis, not hidden inside it.** Whether day-to-day variance
   is stationary over longer spans is not answerable from public data.
2. **The developmental data is not hand-labeled per rendition,** and its labels are locked
   in MATLAB tables. Phase 1's "recover the existing hand labels" is a canary/Bengalese
   finch task. For Duke, label recovery *is* the Phase 2 fidelity check.
3. **Species mismatch is real.** The positive control is zebra finch development; the
   intended application is a pharmacological manipulation of crystallized adult song,
   possibly in another species. Recovering a developmental effect proves the metric has
   sensitivity; it does not prove it is calibrated for adult drift. Both must be reported.

### What no public dataset provides

Nothing public offers a **pharmacological critical-period manipulation in adult songbirds
with annotated audio**. The Vellema testosterone timecourse is the nearest analogue and
only its derived features were deposited. The wet lab will be generating genuinely novel
data — which is the point, but it means the tool cannot be end-to-end validated on its
actual target condition before the experiment runs. Deafening is the closest proxy.

---

## Operational constraints found while auditing

- **Disk: 43 GB free** [COMPUTED]. Zai alone is 49.28 GB; Duke raw audio is 290.6 GB;
  Duke in full is 416 GB. **Neither validation dataset fits on this machine as-is.** Needs
  external storage or a streaming/partial-extraction strategy before Phase 1.
- **Zenodo blocks file bytes from this environment (403)** [COMPUTED]. Dryad sits behind
  proof-of-work bot protection [REPORTED]. Both work in a normal browser. Duke and figshare
  serve fine, and **Duke supports HTTP range requests** [COMPUTED] — which is how the 91 GB
  archive's manifest was read without downloading it. Expect to fetch Zai and the canary
  data manually.
- **Python 3.12 required** — `vak` 1.1.0 (released 2026-03-13, actively maintained [REPORTED])
  needs ≥3.12; system Python here is 3.9.6. Project venv is pinned to 3.12.13.

## Recommended Phase 1 entry point

Start with the **Bengalese Finch Song Repository**, not the biggest dataset. It is 7.6 GB,
hand-labeled, 32 kHz, in a crowsetta-native format, and it is the dataset the `vak` tutorial
itself uses — so segmentation failures are attributable to our pipeline rather than to the
data. Recover the hand labels there, then scale to canary, then to Duke.

Do not touch Zai until the 40 MB ETH companion confirms the interior structure.

---

## Downloaded and retained

In `data/phase0/` (small derived artifacts, committed):
`duke_inventory.json`, `duke_durations.json`, and `raw_metadata/` holding the Duke
documentation, the Duke record metadata, the figshare bfsongrepo manifest, and the Zenodo
Zai record.

Outside the repo (bulk, in scratchpad — 745 MB): the five Duke `_segs.zip`, five
`_table.mat`, `grn475_proj.zip`, and five `_hatchdate.mat`.
