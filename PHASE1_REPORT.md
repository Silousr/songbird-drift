# Phase 1 — Ingestion and Segmentation

**Date:** 2026-08-03
**Question:** Can the pipeline ingest real annotated song and recover the existing hand labels?
**Answer: PASS**, with three constraints that carry into Phase 2 and a bound that Phase 3 needs.

All numbers below were computed in this session and are regenerable from the scripts named.

---

## 1. What was ingested

The complete Bengalese Finch Song Repository: **4 birds, 18 bird-days, 5.5 GB**
(`wav`+`csv` half only; the `.cbin` half is the same song in another format). All 18
day-archives were downloaded and size-verified against the figshare manifest.

Loading all four birds through the canonical syllable table gives **61,094 syllables over
8 bird-days** for the two fully-verified birds, and 3,546 audio files across all 18
bird-days. Annotation coverage on the days examined is 100%.

Audio is confirmed **32 kHz, 16-bit PCM mono**, read from a real WAV header — matching what
Phase 0 reported from documentation.

## 2. Two defects in the published dataset

Both were caught by loader guardrails, not by inspection. Neither is documented upstream.

**A signed 16-bit counter overflow.** `bl26lb16/042012` filenames wrap from serial 32745 to
**−32754** partway through the day, affecting **84 of that day's 202 files**. The parser now
accepts negative serials, and `serial` is documented as an opaque identifier rather than an
ordering key — ordering must use the timestamp.

**A different experimental phase filed under a baseline day.** `gy6or6/032212` contains
**10 files dated 2012-03-13** — nine days before their directory date — carrying the
template `washout` while every other file in that bird's directories reads `baseline`. In
Sober-lab terminology "washout" is a post-perturbation recovery period, so these are not
baseline song. The date cross-check raised on them rather than silently folding them in.

**And a confirmation.** Phase 0 flagged `gr41rd51` as not-provably-baseline because its
filenames embed an evTAF detection template. Scanning all 3,546 files shows **all 1,836
`gr41rd51` files, across all five days, carry `_3part_SYLc_th4191_belowhits`** — every
file, every day. That is not an incidental leftover. The bird remains excluded from any
"unmanipulated" set until the authors confirm whether white noise was actually delivered.

## 3. Segmentation: F1 = 0.969 on held-out days

`scripts/phase1_segmentation_eval.py`

The threshold is fitted on each bird's **first** day and reported only on its **remaining**
days, so the reported figure is not circular with respect to fitting.

| Bird | Fitted threshold | Held-out days | Syllables | F1 |
|---|---|---|---|---|
| `bl26lb16` | 0.004 | 2 | 3,755 | 0.924 / 0.953 |
| `gr41rd51` | 0.015 | 4 | 4,498 | 0.911 – 0.965 |
| `gy6or6` | 0.006 | 4 | 6,411 | 0.976 – 0.991 |
| `or60yw70` | 0.010 | 4 | 7,660 | 0.943 – 0.996 |
| **Pooled** | | **14 bird-days** | **22,324** | **0.969** (P 0.968, R 0.970) |

**The threshold must be fitted per bird.** The optimum ranges 0.004–0.015 across four
birds; imposing a single global value costs `bl26lb16` about 0.18 F1 (0.77 vs 0.94).
`tune_threshold()` is part of the library, not a constant in a script, because the
deliverable is a tool other people run on their own birds.

**Parameters came from the data, and the conventional value was wrong.** Measuring the
ground truth first: 1st-percentile inter-syllable gap **9.3 ms**, median 25.6 ms, and
**24.8% of gaps below 20 ms**. The usual 20 ms `min_silence` would have merged roughly a
quarter of all adjacent syllable pairs into single segments — an error that would have
looked like a property of the birds rather than of the parameter.

### How much this result is worth

**Less than the number suggests, and the script says so.** These reference boundaries were
themselves drawn with `evsonganaly`, an amplitude-threshold segmenter. Reproducing them
demonstrates that this implementation recovers *that algorithm's* output — not that either
finds the acoustically correct syllable edge.

What it does establish is a **timing budget**. Agreement degrades sharply as tolerance
tightens:

| Tolerance | 20 ms | 10 ms | 5 ms | 2 ms |
|---|---|---|---|---|
| F1 | 0.997 | 0.995 | 0.920 | 0.458 |

**Syllable boundaries are trustworthy to roughly ±5–10 ms and no better.** Any Phase 2
feature or Phase 3 drift statistic that depends on finer timing than that is measuring
segmentation noise, not song.

## 4. Label recovery: the non-circular test

`scripts/phase1_eval_labels.py`

TweetyNet trained via `vak` on `gy6or6` day 032312 (494 s of song, 11 syllable classes),
early-stopped after 18m15s at **validation frame accuracy 0.989**. Unlike the boundaries,
these labels are *human judgements*, so recovering them is genuine evidence.

Evaluated two ways — a held-out split of the training day, and a **different day two days
later**:

| Metric (post-processed) | Within-day | Cross-day (+2 d) | Δ |
|---|---|---|---|
| Frame accuracy | 0.9888 | 0.9887 | −0.0001 |
| Syllable error rate | 0.0015 | 0.0033 | +0.0018 |
| Segment F-score | 0.9928 | 0.9949 | +0.0022 |
| Edit distance per song | 0.107 | 0.196 | +0.089 |

Raw network output (no majority-vote post-processing) shows the same picture: frame
accuracy 0.9887 vs 0.9885, syllable error rate 0.0150 vs 0.0189.

**The labeller does not meaningfully degrade across two days.** Frame accuracy is
identical to four decimal places and segment F-score is marginally *better* on the held-out
day. Syllable error rate roughly doubles, but from 0.15% to 0.33% — about one error per 300
syllables.

### Why this specific comparison was the point

An automatic labeller that degrades over time injects a time-varying artefact that is
indistinguishable from song drift. Had cross-day accuracy fallen materially, every Phase 3
drift estimate would have been confounded at the source. It did not, so:

> **Labeller instability contributes ≈0.33% syllable error at 2-day separation, of which
> ≈0.18 percentage points is attributable to crossing days. Phase 3's noise floor must be
> measured against this, and any claimed drift must exceed it.**

**This bound is narrower than it looks.** It is one bird, 11 syllable classes, and a
**2-day** separation — the Bengalese finch repository spans at most 5 days, so longer gaps
cannot be tested here. Whether labeller agreement holds at 10-day (canary) or 60-day
(developmental) separations is untested and must not be assumed. Re-measuring this on the
canary data, which offers 9–11 consecutive days, is the first thing Phase 2 should inherit.

## 5. Constraints carried forward

**CPU-only on Apple Silicon.** vak rejects `accelerator = 'mps'`, and routing through
`'gpu'` — which Lightning maps to MPS — fails at the first batch with `Cannot convert a MPS
Tensor to float64`, because vak's spectrograms are float64. Training ran on CPU at ~3.1
batches/s. vak also slides a window over *every* timebin, so one epoch over 494 s of song
is ~247k windows. Budget accordingly, or use a CUDA machine.

**Off-schema labels cost files.** Restricting to the 11 song-syllable classes excludes
files containing `0`/`x`/`y`/`z` (contact calls, unclear sounds): 9 of 162 files on the
training day, 4 on the evaluation day. This is a selection effect on which files enter the
analysis, not a free choice.

**`gr41rd51` is not usable as baseline** until its evTAF template is resolved with the
authors — it is otherwise one of the two largest birds in the repository.

## 6. Gate verdict

# PASS

Segmentation recovers hand-annotated boundaries at F1 0.969 on held-out days, and TweetyNet
recovers hand *labels* at 98.9% frame accuracy with a 0.33% syllable error rate on a
held-out day. The pipeline reproduces expert annotation on real data.

**Proceed to Phase 2 (embedding + fidelity check)** carrying: the ±5–10 ms boundary budget,
the 0.33% labeller-error floor, per-bird parameter fitting, and the exclusion of
`gr41rd51` and the `washout` files.

## Reproducing

```bash
python scripts/phase1_segmentation_eval.py --root data/bfsongrepo --out results/phase1/segmentation_amplitude.json
vak prep configs/tweetynet_gy6or6_032312.toml && vak train configs/tweetynet_gy6or6_032312.toml
python scripts/phase1_eval_labels.py --train-results results/phase1/vak/train --config configs/tweetynet_eval_gy6or6_032512.toml
```

81 tests, each written before its implementation: `python -m pytest tests/ -q`.
