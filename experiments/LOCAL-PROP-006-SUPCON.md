# LOCAL-PROP-006-SUPCON: source-supervised contrastive learning from scratch

Status: **completed; screening failed**. Both arms finished 600 epochs and 24,000
updates each, 14 Nov 20:05:05–15 Nov 00:34:29 UTC, approximately 4 hours 29 minutes.
Exit code 0, no OOM restarts. No additional folds or Kaggle upload were started.

This experiment tests whether supervised contrastive representation learning can
improve the strongest team's ensemble. [The preceding research](../docs/METHOD_RESEARCH_2026-11-14.md)
found that the source-loss control outperformed target pseudo-label consistency,
and a fold-safe geometric diagnostic added no corrected labels. The next change
therefore uses reliable source labels to learn cross-image class relationships.
No competition gain from SupCon has been observed yet.

## Objective and matched controls

Both arms are ResNet18d, trained at 288 pixels from the same saved random
initialization, with no external pretrained weights. Each has the same encoder,
30-class classifier, and projection MLP (512 hidden, 128 output). The inference
classifier is the encoder's normal linear head; the projection is for training.

- **A:** label-smoothed cross entropy on both source views.
- **B:** the same cross entropy plus supervised contrastive loss on normalized
  projection embeddings. Temperature 0.1, loss coefficient ramps from 0 to 1
  over 20 epochs using a cosine ramp.

SupCon treats every other same-class embedding as a positive and all non-self
embeddings as the denominator. With 30 classes, two photos per class and two
views, every anchor has three positives, including views of a different photo
from a distinct scene group. The comparison set has **120 embeddings in one real
GPU batch**. No microbatch approximation or gradient accumulation substitutes for
the contrastive comparison set. This joint CE+SupCon objective is an adaptation
of the paper, rather than a replay of its principal two-stage training procedure.
[Khosla et al., Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362).

The sampler is class-balanced. Each update draws two images per class, from
different detected scene groups; sampling across updates is with replacement.
An epoch has 40 updates and 2,400 source photograph draws. Each source image
produces paired light-color and corrupt-v4 views after the same native rotation
and random-resized-crop seed. Batch MixUp/CutMix is disabled to preserve discrete
positive class labels. The two views may differ geometrically because v4 itself
contains an affine augmentation.

Both arms share the exact prepared batch and deterministic dropout seed. They
are updated A then B on each batch; both start progressing from epoch 1. This
interleaving reuses source decoding and maintains matched update counts. It is
not two simultaneous GPU forward/backward computations.

The fixed budget is **600 epochs, 24,000 optimizer updates per arm**, AdamW LR
0.002 to 1e-6, three warmup epochs, weight decay 0.05, label smoothing 0.1,
gradient norm clip 5, BF16, channels-last, and constant EMA decay 0.999.
Data sampling, architecture, update schedule, and all optimizer settings are
matched between arms. The primary decision uses final epoch-600 EMA weights.

## Validation and promotion

Use the team's fixed scene fold 2: **2,411 source rows / 603 validation rows**.
All validation images and detected groups are excluded from training. Neither
arm sees test pixels, pseudo-labels, graph edges, or SIFT matches. The full-train
class prior is frozen to match the team's inference convention. Baseline model
selection has used full OOF historically; this is a conditional development
screen, not an independent final test.

Evaluate every 50 epochs and at 600 with student@352, flip TTA, and an equal
ensemble with the fixed same-fold reference R34d@320. Recompute the strongest
team ensemble through that same batch-32 inference path before training and
record its primary, clean, and blur metrics. Primary is macro F1 after the team's
100-iteration balancing, before SIFT. Clean and Gaussian-blur (sigma 0.6/1.0,
kernel 7) guards use raw ensemble F1 on the same held-out rows.

B must exceed both the initial strongest team ensemble and matched A by **0.01
macro F1**. Clean and mean additional-blur F1 must each stay within −0.01 of both
comparators. Report class-level F1 and a class-stratified paired scene-group
bootstrap with 2,000 replicates. Intermediate scores are diagnostic and do not
select the checkpoint. No extra folds or Kaggle upload happen automatically
from this pilot; promotion requires adequate confirmation.

## Reproducibility and background execution

Config: `configs/LOCAL-PROP-006-SUPCON.yaml`.
Runner: `scripts/supervised_contrastive.py`.
Supervisor: `scripts/launch_supervised_contrastive.py`.
Python: `/home/nafisnaufal1426/venvs/dq/bin/python`.

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NO_ALBUMENTATIONS_UPDATE=1
/home/nafisnaufal1426/venvs/dq/bin/python scripts/supervised_contrastive.py --prepare
/home/nafisnaufal1426/venvs/dq/bin/python scripts/supervised_contrastive.py --smoke
/home/nafisnaufal1426/venvs/dq/bin/python scripts/launch_supervised_contrastive.py
```

Preparation pins raw official source files, reference weights/config/caches,
scene metadata, frozen reference code, versions, and new code/config hashes.
The random initialization is saved once and loaded identically by both arms.
Augmentation/sampling and dropout seeds derive deterministically from epoch and
step; a dedicated DataLoader generator avoids consuming the model random stream.

One atomic **paired checkpoint** saves both student/EMA models, optimizers,
schedulers, histories, and random state after each fully completed epoch.
An OOM exits with code 75; the detached supervisor retries after 30 seconds from
the last complete paired epoch, preserving the 120-embedding comparison set.
Partial A/B updates from an interrupted epoch are both discarded. Other errors
stop the job. The supervisor writes heartbeat files every 60 seconds and prevents
duplicate jobs. No other user's process or dependency is modified.

Outputs are ignored by Git under `outputs/LOCAL-PROP-006-SUPCON/`, including
`spec.json`, `initial_random_weights.pt`, source manifest, reference predictions,
`paired_checkpoint.pt`, arm histories, final EMA weights, validation probabilities,
`status.json`, `job.json`, logs, and eventual `result.json`/`completion.json`.
GPU sharing is explicitly authorized; there is no CPU fallback. ETA is reported
only after measuring actual training throughput.

The prelaunch suite has 38 passing tests. New tests check that cross-image
same-class alignment reduces SupCon loss, every anchor needs a non-self positive,
zero SupCon weight preserves the CE gradient, and class-balanced sampling spans
groups while excluding validation images. GPU smoke must additionally show
matched initial A/B losses, a nonzero encoder update difference when the SupCon
coefficient is enabled, finite gradients, identical starting weights, and a full
paired checkpoint/EMA/optimizer/scheduler round trip.

## Verified launch evidence

Preparation pinned all inputs and the actual comparison-set size. The strongest
team ensemble recomputed on **fold 2** scores **0.826344263** primary, 0.822751824
raw primary, 0.871937416 clean and 0.594269235 mean additional blur. These scores
are local fold-2 baseline values; they must be compared on this same fold rather
than against the earlier fold-0 SSL scores or public leaderboard values.

The paired GPU smoke passed at 20:04:28 UTC. Both starting networks produced
identical CE 3.460339 and SupCon loss 4.864398 on the same 120 views. Enabling
the contrastive term changed the encoder weights by up to 3.33786e-5 after the
first update. Finite gradients and checkpoint/EMA/optimizer/scheduler restoration
were verified; peak allocated GPU memory was **3.3036 GiB**. Smoke weights are
discarded, and training starts from the saved random initialization.

Supervisor PID at launch: 4146105; runner PID: 4146107. At 20:06:28 UTC, both arms
had completed **38/40 updates in epoch 1**. No OOM restart was recorded. The job
records status per paired update and checkpoints every completed paired epoch.
PIDs are historical evidence; inspect current `job.json` and process command
lines before using them later. First validation is scheduled for epoch 50;
no candidate validation or leaderboard gain was claimed at launch. These PIDs
are now historical; the job has completed.

## Final results

All values below are local macro F1 on the same **603 fold-2 validation rows**.
Primary is the equal student/frozen-R34d ensemble after fixed prior balancing,
before SIFT. Clean and mean additional-blur guards use raw ensemble F1.

| Metric | Original team baseline | CE control A | CE+SupCon B | B minus A |
| --- | ---: | ---: | ---: | ---: |
| Primary balanced ensemble | 0.826344 | **0.834465** | 0.827229 | −0.007236 |
| Primary raw ensemble | 0.822752 | 0.833777 | 0.837310 | +0.003533 |
| Clean raw ensemble | 0.871937 | 0.875798 | 0.873993 | −0.001805 |
| Mean additional-blur raw ensemble | 0.594269 | 0.627330 | 0.595798 | −0.031532 |
| Primary raw R18d student | 0.810467 | 0.835351 | 0.810198 | −0.025153 |

B gains only **0.000884** against the original primary baseline and is
**0.007236 below A**, failing both +0.01 primary gain requirements. Clean guards
pass, but the mean-blur guard against A also fails (−0.031532 exceeds the allowed
−0.01 loss). On balanced primary labels, B fixes five A errors and breaks eight
correct A predictions; 29 labels change overall, including changes between wrong
classes. Against the original baseline it fixes eight errors and breaks six,
with 35 changed labels.

Conditional class-stratified paired scene-group bootstrap 95% intervals for B's
primary delta are **[−0.012675, +0.013879]** versus the original baseline and
**[−0.020446, +0.004280]** versus A. Both include zero. This tested joint-loss
configuration provides no demonstrated advantage over its matched control.
This conclusion does not reject all supervised contrastive formulations.

A improves primary by **+0.008120** against the original baseline, fixing twelve
errors and breaking seven correct predictions. Its R18d-only raw F1 improves
by +0.024885. This is useful evidence for the alternative CE recipe, but changes
include paired views, class-balanced sampling, and disabled MixUp/CutMix; no
individual cause is isolated. One development fold does not establish a full
OOF or leaderboard improvement, and A is not automatically promoted.

B's training contrastive loss falls to 1.1830, versus 4.2487 in A's unoptimized
projection, yet its primary and blur generalization do not improve over A.
Optimizing the contrastive training objective is therefore not sufficient evidence
of better validation performance. B has a higher raw ensemble F1 than A, while
the chosen balancing reverses their ranking. Calibration/decoding is a possible
separate follow-up; no post-hoc change is used to rescue the preregistered gate.

The strongest logged intermediate primary scores are A 0.837551 and B 0.830705,
both at epoch 550. The decision remains epoch 600; no intermediate checkpoint is
promoted. Existing best team predictions remain unchanged.

## Final verification

Verification recomputed initial/A/B metrics from stored probabilities, confirmed
the exact student/frozen-R34d averages, and checked all pinned code/config/input
hashes. Both final encoder and projection EMA files match the paired epoch-600
checkpoint exactly. Optimizer/scheduler states confirm 24,000 updates per arm
and both histories contain 600 epochs. Peak GPU allocation was **3.3926 GiB**.
This verification used saved artifacts; training and GPU inference were not repeated.

[Machine-readable summary](results/LOCAL-PROP-006-SUPCON.summary.json) records the
final metrics, gates, intervals, trajectories, checkpoint hashes and verification.
Full local artifacts remain under `outputs/LOCAL-PROP-006-SUPCON/`, including
`result.json`, `final_verification.json`, `paired_checkpoint.pt`, and both arms'
final EMA weights and validation predictions.

Follow-up 15 Nov 07:28 UTC: the user explicitly requested a public probe. Final
**CE A** replaced one R18d fold-2 member in EXP-024; SupCon B was not submitted.
Public **0.86464**, below the original **0.87012**. See the separate
[submission report](LOCAL-PROP-006-A-PUBLIC.md) for composition, receipt and limits.
