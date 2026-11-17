# Deadline options and Nafis server artifacts — 17 November

At 11:05 UTC / 18:05 WIB, there were roughly six hours until the competition
deadline. The Kaggle snapshot at **11:22:48 UTC** still showed best **0.88314**
and two submissions on 17 November, nominally leaving **one shared daily slot**.
No new submission was made during this work.

## Teammate artifacts found and replayed

After the user authorized searching the server, exact filename search found
`/home/nafisnaufal1426/akucintakc/work/`. Its CSVs, cached neural probabilities,
calibrated SIFT evidence, inference/training scripts and completed checkpoints
are available. These files remain **read-only**. The separate local repo clone
is on `main`; the working ML files are outside that clone, explaining why the
new pipeline was absent from GitHub branches.

The five stored mid-training member predictions are:
`r1_r18d_384`, `r1_r34d_320`, `r1_r18d_288`, `r1_r18d_352w`,
`r1_fr_r18d_320`. Their geometric mean reproduces `ens_r1mid.npy` **exactly**.
Combining all five individual arrays and `team_best_probs.npy` as **six equal
geometric inputs** reproduces `ens_r1mid_team.npy` **exactly**. Each comparison
has maximum probability difference 0. This establishes the stored probability
composition; intermediate checkpoint training/inference was not rerun.

`infer2.py` applies 100 train-prior balancing iterations, log probability, then
adds the frozen `evidence2.npy` matrix. Original team SIFT matches with ≥5
inliers impose a minimum log-evidence 4 for their matched class. `calib.py`
fits isotonic calibration on an automatically generated synthetic matching
proxy and converts its scores into log evidence. We preserve the saved evidence
rather than refitting it. There are no test→test links in this path.

Those steps reproduce local `sub_mine5_team_mid.csv` **byte-for-byte**, SHA256
`6ff8d89ddfd42c911e12c2cf612c8783eb4c7f3be6ff44fb17db3e8b1a83f179`.
Its filename/timestamp associate it with ref 58453652, public .88314; Kaggle's
listing itself does not expose a hash of the uploaded file. Cached-probability
replay does not constitute a reproduction of all five original training runs.

## Fixed fusion candidate ready

The independent [preparation script](../scripts/prepare_deadline_blend.py) creates
one **90% Nafis best / 10% PVT-B3 geometric blend**, before the exact unchanged
Nafis balancing/evidence rule. It uses the completed PVT-B3 epoch-800 test
probabilities and does not retrain or modify teammate files.

- Config: [DEADLINE-NAFIS-PVT.yaml](../configs/DEADLINE-NAFIS-PVT.yaml).
- Valid 2,010-row CSV, **56 changed predictions** versus the reproduced baseline;
  **zero changed rows** with maximum log-evidence ≥4 in this candidate.
- CSV SHA256:
  `d5178aeb41d5f737a93651c2e92ee7f8d8c4de1d3d58478293a74437ef91a3a5`.
- Local path:
  `outputs/DEADLINE-OPTIONS-2026-11-17/NAFIS-PVT-FUSION/NAFIS90_PVTB3_10_candidate.csv`.
- Two behavior tests passed, source hashes were checked before/after preparation,
  and the source CSV replay was exact. **Not uploaded; no candidate score exists.**

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/nafisnaufal1426/venvs/dq/bin/python -m scripts.prepare_deadline_blend
```

The fixed weight is exploratory, not optimized on public scores. Nafis's new
models trained on all official train photographs, so their predictions cannot
give an honest fold-2 blend comparison. Sibling-derived target/proxy labels also
are not organizer test ground truth or an unseen-scene holdout. Therefore a
claim of improvement over .88314 is not established by preparing this file.
Any public probe must account for the remaining shared submission slot.

## Other screened options

The complete B0 experiment finished **09:41:17 UTC**, exit 0, no OOM. Final B0
IDMM matching-pair primary **.841597**, standalone **.815623**, below B3's
**.865532 / .862227**. Exported EMA, checkpoints, optimizer/scheduler counts,
probabilities and frozen inputs were independently verified.
[B0 verification](../experiments/results/LOCAL-PROP-008-B0.summary.json).

A small **exploratory development-fold** B0/B3 mixture screen found no primary
gain over B3: B0 fractions .1/.2/.25/.5 gave .864116/.864364/.862245/.854498,
versus pure B3 .865532. This is not an independent validation result. It did
not justify spending a public slot on that mixture.

Additional final ResNet and high-resolution predictions exist in Nafis's work
folder and could be ensembled without training. They should be considered with
their source configs and proxy limitations, not substituted based only on a
larger epoch number. No such substitution was submitted here.

The selected training option is [LOCAL-PROP-009-PVT448](../experiments/LOCAL-PROP-009-PVT448.md):
100 source-only epochs continuing B3 at 448px, detached at 11:21 UTC. Benchmark
projection is 2.53 hours including 30% overhead, with matched held-out views and
a final-epoch gate. This is inspired by the paper's extra resolution stage;
it is not a promised public improvement. The current fixed blend remains a
prepared comparison while training runs.

[Machine-readable snapshot](../experiments/results/DEADLINE-OPTIONS-2026-11-17.summary.json)
includes hashes, replay checks, benchmark, blend diagnostics and the Kaggle snapshot.
