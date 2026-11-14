# LOCAL-PROP-004-SSL: adaptive self-training, fold-0 screening

Status: **running**, detached supervisor launched 14 Nov 2026 at 09:34:51 UTC.
Control A began real optimizer updates; B will follow sequentially.

The retrieval graph pilot reduced pooled two-fold balanced macro F1 from 0.867045 to
0.853589 (delta −0.013457). This experiment therefore excludes graph propagation
and tests adaptive self-training directly. This is an **AdaMatch-inspired adaptation**,
not an exact reproduction of the paper, and makes no prediction of a 0.99 score.

## Matched comparison

Both arms initialize ResNet18d from kaka's reproduced `R_r18d_i288_e600_f0/model.pt`,
the final EMA after 600 epochs. The original optimizer state is unavailable, so this
is a new training phase with a fresh AdamW optimizer and schedule. It is not a
resumption of the original 600-epoch run. No external pretraining or data is used.

- **A:** source-loss control. It processes the same target inputs through BatchNorm,
  but the target consistency loss has weight zero.
- **B:** the same source loss plus adaptive consistency on the original unlabeled
  test images. Dynamic pseudo-labels come from the current EMA teacher's weak view.

The source branch pairs light-color and `corrupt_v4` views of the same native
rotation, with the same random resized crop seed. Source-only and joint
source/target logits are randomly interpolated before supervised cross entropy.
Batch MixUp/CutMix is disabled in both arms. These source and BatchNorm changes
mean that A is a control for this training phase, not an exact rerun of kaka's recipe.
The initial checkpoint is evaluated separately.

Target weak augmentation is a horizontal flip and a shift of at most four pixels;
the strong view adds mild brightness/contrast and a small cutout. Full synthetic
`corrupt_v4` is not applied a second time to the already corrupted test images.
The target loss uses detached hard labels, source/target marginal alignment
clipped to [0.5, 2], and a threshold of 0.9 times moving source confidence.
EMA statistics decay at 0.99. For epochs 1–10, selection additionally requires
agreement with the two initial fold-0 backbones. A cosine ramp brings target loss
weight from 0 to 1 over the first 20 epochs. Rejected examples still count in the
loss denominator. The weight EMA uses a constant decay of 0.999.

Both arms use seed 44, identical stateless image draws, 100 epochs at 288 pixels,
AdamW with LR 1e-4 → 1e-6 and one warmup epoch, weight decay 0.05, label smoothing
0.1, BF16, and gradient clipping at norm 5. Each optimizer update accumulates eight
microbatches, each containing eight labeled images and 24 unlabeled images.
Each epoch uses 2,368 of the 2,400 source images without replacement (a shuffled
32-image remainder is dropped); target sampling is with replacement, 7,104 draws
per epoch. There are 37 updates per epoch and 3,700 per arm. A runs before B.

## Fold safety and evaluation

The team scene fold contains 2,400 source and 614 held-out rows with no shared
scene groups. Held-out images never enter either adaptation loader. Initial
anchors use only fold-0 R18d and R34d predictions. The all-five-fold test ensemble
and full-gallery SIFT matches are forbidden as training teachers because their
training labels include fold-0 validation examples. The full-train class prior is
frozen to match the team's inference convention; validation labels are never used
for gradients, pseudo-label selection, checkpoint selection, or early stopping.

Every ten epochs, evaluate EMA predictions at 352 pixels with flip TTA. The
primary metric combines the student and a frozen R34d fold-0 checkpoint at 320
pixels with equal weights, then applies the team's 100-iteration prior balancing.
Clean and additional post-resize Gaussian blur (sigma 0.6 and 1.0, kernel 7) are
diagnostics on the same held-out images. Evaluation uses batch size 32, so the
initial weights are recomputed at batch size 32 and rounding differences from
the exact reference replay are recorded explicitly. Scores used for comparisons
all use this same inference path. No SIFT is used in this pilot.

Only the final epoch-100 EMA makes the screening decision. Intermediate scores
are logged and do not select checkpoints. B must gain at least 0.01 macro F1
against both its initial checkpoint and A. Clean and mean blur raw ensemble F1
must each be no more than 0.01 below either comparator. Class-stratified paired
scene-group bootstrap intervals use 2,000 replicates. Single-fold intervals are
conditional diagnostics and do not establish a leaderboard improvement.

Passing this screening supports a later all-five-fold confirmation. This pilot
does not automatically train more folds or upload a Kaggle submission. The user's
conditional submission authorization remains applicable after adequate validation.

## Background execution and artifacts

Configuration: `configs/LOCAL-PROP-004-SSL.yaml`.
Runner: `scripts/adaptive_self_training.py`.
Supervisor: `scripts/launch_adaptive_self_training.py`.
Interpreter: `/home/nafisnaufal1426/venvs/dq/bin/python`, the exact reproduced team environment.

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NO_ALBUMENTATIONS_UPDATE=1
/home/nafisnaufal1426/venvs/dq/bin/python scripts/adaptive_self_training.py --prepare
/home/nafisnaufal1426/venvs/dq/bin/python scripts/adaptive_self_training.py --smoke
/home/nafisnaufal1426/venvs/dq/bin/python scripts/launch_adaptive_self_training.py
```

The output directory is `outputs/LOCAL-PROP-004-SSL/` (ignored by Git). It contains
immutable code/config/input hashes, raw-image hashes, split manifests, derived
caches, initial and frozen-backbone predictions, smoke-test results, per-arm
history and checkpoints, logs, `status.json`, `job.json`, and `heartbeat.json`.
The launcher returns immediately. Its detached supervisor records a heartbeat
every 60 seconds and guards against duplicate runs. There are no automatic chat
notifications, uploads, or background Git changes.

Every completed epoch atomically saves student, EMA, optimizer, scheduler,
pseudo-label statistics, and Python/NumPy/Torch/CUDA random state. On CUDA OOM,
the runner exits with code 75 and the supervisor retries after 30 seconds from
the last full epoch, preserving batch sizes and deterministic image draws. Other
errors stop the job for inspection. Only this job's child processes are managed.
GPU sharing is explicitly authorized; there is no CPU fallback.

Checks cover fold isolation, relative confidence and anchor gates, marginal
alignment, detached masked loss and its denominator, deterministic augmentation,
and RNG checkpoint round trips. A real GPU smoke test additionally executes a full
accumulated optimizer update with target loss active and verifies EMA/checkpoint
restoration before launch.

## Verified launch evidence

All 30 repository tests passed before launch. The initial fold-0 ensemble score
recomputed at batch size 32 is **0.8714644561**, matching the reference's score;
raw primary is 0.8679445544, clean 0.9049602556, and mean additional-blur
0.6080878568. These are initial local validation metrics, not candidate results.

The GPU smoke test passed at 09:34:43 UTC: one full eight-microbatch update with
target loss active, 46/192 target examples selected, finite gradient norm 1.49982,
peak allocated memory 2.1253 GiB, and EMA/checkpoint round-trip checks passed.
The smoke weights are discarded; A and B each start from the original team EMA.

Detached supervisor PID at launch: 3909795; runner PID: 3909797. At 09:35:52 UTC,
A was in epoch 1 with **9/37 optimizer updates completed**, target loss weight 0,
and no OOM restart recorded. PIDs are historical evidence; check `job.json` and
the process command line before using them later. Final scores and runtime are
pending. The supervisor will write `completion.json` when both arms finish.
