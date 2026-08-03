# Songbird Vocal Plasticity Measurement Toolkit

Quantifies how much an individual songbird's crystallized song drifts over time, with a
known noise floor and honest uncertainty — a power-analysis tool for planning experiments
that test whether a manipulation reopens the critical period.

Birdsong here is treated as what it is: a learned motor sequence with a syllable inventory
and probabilistic syntax. Not language.

## Status

**Phase 0 (data audit) complete — verdict GO.** See [PHASE0_DATA_AUDIT.md](PHASE0_DATA_AUDIT.md).

**Phase 1 (ingestion + segmentation) in progress.** All 18 Bengalese finch bird-days
(4 birds, 5.5 GB) ingested. Amplitude segmentation reaches **F1 = 0.969** against the hand
annotations on 14 held-out bird-days (22,324 syllables), with the threshold fitted per bird
on a separate day. TweetyNet label recovery is running; that is the non-circular half of the
gate and the phase is not complete until it reports.

Two defects in the published dataset were caught by the loader's guardrails and are
documented in [DECISION_LOG.md](DECISION_LOG.md): a signed 16-bit filename counter that
wraps negative in `bl26lb16/042012`, and 10 files in `gy6or6/032212` that are dated nine
days earlier and belong to a different experimental phase (`washout`).

## Planned scope

1. **Ingestion + segmentation** on real data, recovering existing hand labels (`vak` + TweetyNet).
2. **Embedding + fidelity gate.** An embedding must separate syllable types and recover
   human labels at high accuracy before any distance computed in it is trusted.
3. **Drift metric + within-bird noise floor.** Day-to-day variation with nothing
   manipulated. The load-bearing statistic.
4. **Sensitivity analysis.** Minimum detectable drift vs recording volume vs syllable count.

Statistical guardrails: within-bird longitudinal only; every drift number carries an
uncertainty estimate; the null is established before any signal is claimed; power is
reported, not just p-values.

## Datasets selected in Phase 0

| Role | Dataset | Scale |
|---|---|---|
| Positive control | Duke juvenile zebra finch ([10.7924/r4j38x43h](https://doi.org/10.7924/r4j38x43h)) | 183 bird-days, 163 h |
| Adult null | TweetyNet canary ([10.5061/dryad.xgxd254f4](https://doi.org/10.5061/dryad.xgxd254f4)) | 3 birds, 31 bird-days |
| Adult null | Bengalese Finch Song Repository ([10.6084/m9.figshare.4805749](https://doi.org/10.6084/m9.figshare.4805749)) | 4 birds, 18 bird-days |
| Known-effect validation | Zai et al. deafening ([10.5281/zenodo.14732250](https://doi.org/10.5281/zenodo.14732250)) | 76 birds — interior unverified |

The Koumura BirdsongRecognition dataset has **no recoverable time axis** and cannot support
drift analysis, despite being the most-cited annotated Bengalese finch corpus. It is usable
as a segmentation benchmark only.

## Setup

Requires Python ≥3.12 (`vak` 1.1.0 constraint) and [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
```

## Reproducing the Phase 0 inventory

Downloads ~133 MB of segmentation archives and derives the full longitudinal inventory
without touching the deposit's 290 GB of audio:

```bash
./.venv/bin/python scripts/phase0_duke_inventory.py --download --data-dir ./data/duke --out data/phase0/duke_inventory.json
```

Expected: 183 bird-days, 251,061 wav files, 5,318,154 segmented sounds, 163.1 hours, with
0 date mismatches and 0 unparsed filenames.

## Layout

```
scripts/    standalone Phase 0 audit tooling
data/phase0/    derived inventories + upstream metadata kept for provenance
PHASE0_DATA_AUDIT.md    the audit and its GO/NO-GO
DECISION_LOG.md         every methodological choice and why
```

Bulk data is never committed. Point `--data-dir` wherever you keep it; no paths are
hardcoded.
