# LOCAL-PROP-009-PVT448 — deadline resolution continuation

The user asked for another feasible experiment before 17 Nov 23:59 WIB. The
paper-backed option is a **100-epoch 448px continuation** of our completed B3.
The synthetic effective-batch benchmark measured **6.99 seconds/update at 448px**,
projecting 1.94 compute hours and **2.53 hours with 30% overhead**. This fits the
remaining time at launch. It is a continuation of our own scratch-trained model,
not a new external initialization or a new architecture.

The [IDMM paper](https://arxiv.org/abs/2201.10728) adds 100 supervised epochs at
448px. The inspected [official B0 script](https://github.com/CupidJay/Training-Vision-Transformers-with-only-2040-images/blob/0ddba0dffa1125df387020c11eeb91da7f8d85fd/run_pvtv2_b0_448_flowers.sh)
uses batch 128, LR 5e-5 and decay 1e-3. Our local B3 choice scales LR to 1e-4 for
batch 256, uses our existing flora augmentations and a shorter EMA horizon.
This is an adaptation; paper accuracy does not predict our competition F1.

## Frozen protocol

- Initialize exactly from LOCAL-PROP-008 B3 final epoch-800 EMA, SHA256
  `d6074465968be104f819c57706b9eba718069a6c99466081f4d1fdf4bf5e20c2`.
- Same scene fold 2, seed 44, **2,411 train / 603 validation**. Only source train
  photographs enter the loader. No Nafis/test pseudo-labels enter this training.
- **448px, 100 epochs / 1,000 optimizer updates**. Effective source batch 256,
  microbatch 16; include final partial photograph and microbatches. Retain the
  classifier and use new AdamW state, LR 1e-4, decay .001, warmup 10 epochs,
  cosine to 1e-6, label smoothing .1, gradient clipping 5, drop-path .1.
- Existing native rotation, light-color/corruption-v4, crop, MixUp/CutMix and
  accumulated group-loss implementation are reused without editing frozen code.
  Augmentation/sampling epochs use **801–900**, avoiding reuse of early draws.
- EMA decay **.99 per update**, so the short 1,000-update stage has a shorter
  horizon than the previous .999 EMA. BF16, channels last, four loader workers.
- Keep primary **352px inference**, batch 16, flip TTA, fixed four-condition
  validation caches and R34d fold-2 probabilities. This matches existing B3
  comparison views. It deliberately differs from the paper's 448px evaluation.
- Evaluate epochs 25/50/75 diagnostically; select **final epoch-100 EMA** only.
  The practical comparison includes extra optimization and EMA/LR changes;
  it does not isolate resolution from an equally long 224px continuation.

Local baseline: primary pair **.865532**, clean **.890449**, mean-blur **.727417**.
Final gate requires primary **+0.003**, with clean and mean-blur declines at most
.01. These are fixed matching B3/R34d fold-2 metrics. They do not measure the
new full-train Nafis ensemble, whose best public score at launch was **.88314**.

## Execution

Three behavioral tests passed for source/fold preservation, target/external
initialization rejection, matched evaluation and the final gate. Real-source
GPU smoke used 256 photographs at 448px, passed finite loss/gradient/update and
checkpoint roundtrip checks, and reproduced the initial B3 validation
probabilities **exactly** (max difference 0). Peak allocation **8.18 GiB**.
Smoke weights never initialize official continuation training.

Detached supervisor **PID 916122**, runner **916123**, launched **17 Nov 11:21:26
UTC / 18:21:26 WIB**. Heartbeat 60 seconds, checkpoint every ten complete epochs,
unchanged-batch OOM resume, GPU allocation cap 25% A100. As a deadline buffer,
the runner stops at an epoch boundary after **15:30 UTC / 22:30 WIB**, saving a
checkpoint; a cutoff result is not labeled a completed final-epoch experiment.

```bash
PY=/home/nafisnaufal1426/venvs/dq/bin/python
$PY -m unittest tests.test_pvt_high_resolution -v
$PY scripts/train_pvt_high_resolution.py --prepare
$PY scripts/train_pvt_high_resolution.py --smoke
$PY scripts/launch_pvt_high_resolution.py
```

Config: [LOCAL-PROP-009-PVT448.yaml](../configs/LOCAL-PROP-009-PVT448.yaml).
[Historical launch receipt](results/LOCAL-PROP-009-PVT448.launch.json).
Local artifacts: `outputs/LOCAL-PROP-009-PVT448/`, including pinned spec, smoke,
job/status/heartbeat, checkpoint, evaluation history, probabilities and eventual
result. No automated Kaggle upload is embedded. The launch receipt remains a
historical snapshot; the completed result follows.

## Completed result — 17 November 13:59 UTC

Finished **13:51:38 UTC / 20:51:38 WIB**, **2h30m11s**, exit 0, no OOM restart.
Completed all **100 epochs / 1,000 optimizer updates** before the cutoff.
Selected the preregistered **epoch-100 EMA**, evaluated at the matched **352px**.

| Local macro F1 | Original B3 | B3 after 448px stage | Delta |
| --- | ---: | ---: | ---: |
| Primary, balanced 50:50 B3/R34d pair | .865532 | **.871987** | +.006455 |
| Clean, raw pair | .890449 | **.912333** | +.021885 |
| Mean blur, raw pair | .727417 | **.790971** | +.063554 |
| Primary, B3 alone | .862227 | .859817 | **−.002410** |

The fixed advancement gate **passes**. The benefit is in the matching ensemble,
especially its blur conditions; standalone primary does not improve. Epoch 50
had diagnostic primary .874130, but it was not selected after seeing scores.
The recipe also changes optimizer/LR/EMA and adds compute; resolution alone is
not causally isolated by this comparison.

Independent checks verified frozen code/config/input/environment identities,
the exported model against checkpoint EMA tensor-for-tensor, epoch/phase,
all optimizer steps and scheduler step 1,000. Recomputing all reported metrics
from the four stored probability arrays and frozen R34d reproduced them exactly.
Model SHA256: `9f75d6683211e2ac205e6e3ac4bb0232065e1b7569b7ad2ce173d90cdd156d18`.
[Verification receipt](results/LOCAL-PROP-009-PVT448.summary.json).

**No public score exists for this checkpoint.** The latest team submission
`final_domain.csv`, ref **58470727**, scored **.90780**, and the Kaggle limits
endpoint reports **four submissions today, zero remaining**. The earlier .88314
blend is superseded. No new test inference, blend upload, or training was started
during this status check. Full-train Nafis predictions cannot independently
validate a new blend on our fold-2 queries.
