# LOCAL-PROP-008-PVT-IDMM — source-only transformers from scratch

The user approved PVTv2 and instance-discrimination pretraining after the
[architecture research](../docs/TRANSFORMER_ARCHITECTURE_RESEARCH_2026-11-15.md).
This experiment asks whether a hierarchical transformer, with or without IDMM,
improves our flora classifier. It prioritizes **PVTv2-B3**; **B0** follows as a
size comparison. It starts new random weights and does not fine-tune the team
ResNets. Frozen R34d probabilities are used for evaluation only.

## Evidence and scope

[Training Vision Transformers with Only 2040 Images](https://arxiv.org/abs/2201.10728)
reports PVTv2-B3 Flowers102 accuracy of 90.5% with supervised learning and 95.9%
with source-only IDMM plus supervised learning at 224px. Its 96.7% result adds a
448px stage, which is **not included here**. These are accuracy measurements on
Flowers102, not predictions of DATERAQUEST macro F1 or a promise of 0.99.

The [official implementation](https://github.com/CupidJay/Training-Vision-Transformers-with-only-2040-images/tree/0ddba0dffa1125df387020c11eeb91da7f8d85fd)
was inspected at commit `0ddba0dffa1125df387020c11eeb91da7f8d85fd`.
Its B0 pretraining script uses six **112px** crops: two with scale 0.14–1.0 and
four with scale 0.05–0.4, effective photograph batch 256, CutMix alpha 0.5,
AdamW decay 1e-3, 800 epochs. The implementation independently mixes each crop
group and adds their mean CE losses. We implement those semantics independently
using the installed `timm` architecture; no external checkpoint is downloaded.

This is an **adaptation**, not an exact reproduction: our fold, 30 species,
2,411 instance IDs, OpenCV/Albumentations color and blur implementation, fixed
ImageNet normalization (std 0.229), BF16, microbatch accumulation, clipping,
standard warmup/cosine, and supervised flora-corruption recipe differ. Supervised
EMA is updated per optimizer step; no EMA is used for the IDMM transfer.

## Frozen experiment

| Architecture | Parameters with 30-class head | Peak LR | Priority |
| --- | ---: | ---: | --- |
| `pvt_v2_b3` | 44,741,086 | 0.0005 | First |
| `pvt_v2_b0` | 3,417,470 | 0.001 | After both B3 supervised arms |

Team scene fold **2**, seed **44**, **2,411 source / 603 validation** photographs.
Compact instance IDs are assigned only after excluding held-out scene groups.
Every epoch shuffles all source photographs without replacement; nine full
256-photo batches and one 107-photo batch give **10 updates/epoch**. There are
no test images, pseudo-labels, external data, or external pretrained weights.

For each architecture:

1. Train IDMM for **800 epochs / 8,000 updates**, six views/photo at **112px**.
   Loss is mean smoothed CutMix CE for the two global views **plus** mean
   smoothed CutMix CE for the four local views. Instance head has 2,411 classes.
2. Train two supervised arms for **800 epochs / 8,000 updates each**, at **224px**.
   **Control** starts from the original scratch backbone; **candidate** starts
   from the final own-IDMM backbone. Discard the instance head and give both
   arms the exact same original random 30-class head. Reset optimizer/scheduler.
3. Within the supervised pair, share the exact source order, augmentations,
   mixed batches, and dropout seeds. Evaluate the final epoch-800 EMA;
   intermediate evaluations at epochs 200/400/600 are diagnostic only.

Supervised views use native-size random rotation ±180° with probability 0.9,
light color or corruption-v4 with equal probability, random resized crop with
scale 0.25–1.0 and ratio 0.5–2.0, and horizontal flip. Each update uses either
CutMix or MixUp with equal probability, alpha 0.5. A single view/photo is used.
This differs from the older ResNet CE A's class-balanced two-view recipe.

Both stages use AdamW, weight decay 0.001, 10-epoch warmup, cosine decay to 1e-6,
label smoothing 0.1, drop-path 0.1, head dropout 0, gradient clipping 5, BF16 and
channels last. Supervised EMA decay is 0.999. Effective photograph batch is 256;
IDMM microbatch is 128 views, supervised microbatch 32. Mixing occurs across
the **whole crop group before slicing**, and gradients are weighted by each
microbatch's actual fraction, including partial final microbatches.

The candidate receives extra pretraining compute. This paired comparison tests
the whole IDMM addition, not a gain at matched total compute. B0 and B3 use
different learning rates, so their comparison is a practical recipe comparison.

## Evaluation and advancement

Use the existing frozen primary, clean, blur0.6 and blur1.0 validation caches,
**352px** candidate inference and **320px** frozen R34d inference, flip TTA.
The 224-to-352 inference change is explicit and common to both supervised arms.
Main metric is local macro F1 of the 50:50 candidate/R34d pair, original
100-iteration prior balancing, before SIFT. Also save raw pair, standalone
primary, clean pair and mean-blur pair scores/probabilities. The legacy balance
prior uses all-train class counts for comparability; it never enters training.
These evaluation settings are development-fold diagnostics, not independent OOF.

Frozen comparison: old team pair **0.826344**, stronger ResNet18d CE A pair
**0.834465**, CE A clean **0.875798**, CE A mean-blur **0.627330**.
Screen the final IDMM candidate for primary gain at least **0.005 versus CE A**
and **0.005 versus its matched PVT supervised control**, with clean/mean-blur
drops no worse than 0.01 versus CE A. Save failed checks as well as passed checks.
A passing gate warrants review and further validation; it does not establish a
public improvement. No additional folds or submission is embedded in this runner.
The strongest observed public result remains EXP-024 **0.87012** at launch.

## Execution and reproducibility

Ten behavioral tests cover held-out exclusion, species-label independence in
IDMM, full-source sampling including the final partial batch, CutMix area,
MixUp/smoothed CE equivalence, accumulated gradients for unequal crop groups,
head replacement, scratch-only factory, foreign-checkpoint rejection, scheduler
budget and advancement gates. Real-source GPU smoke also exercises each phase,
model and optimizer update, checkpoint roundtrip, and 352px inference.
All ten tests passed. Real-source smoke completed at **15 Nov 2026 16:19:04 UTC**,
with finite losses/gradients and changed weights in all six updates. Peak GPU
allocation was **5.31 GiB**. Both supervised heads matched before their updates.
The matching frozen specification SHA256 is
`b29065c810bc5d855120439fe60c9f00c67d850e908f76a488f719162980d602`.

The synthetic warm-update benchmark estimated **39.88 hours of GPU compute**
for all six stages, before data loading/evaluation/checkpoint overhead. It is
an estimate on a busy shared GPU, not a completion guarantee. B3 runs first
(about 30.2 of those hours). Full duration may approach the competition deadline;
the launch record preserves the timestamp and current progress. The longer
800+800 recipe is deliberate: this experiment does not substitute a short
100-epoch result for the paper's substantially longer training schedule.

```bash
PY=/home/nafisnaufal1426/venvs/dq/bin/python
$PY -m unittest tests.test_pvt_idmm -v
$PY scripts/train_pvt_idmm.py --prepare
$PY scripts/train_pvt_idmm.py --smoke
$PY scripts/launch_pvt_idmm.py
```

The detached supervisor writes a heartbeat every 60 seconds and retries CUDA
OOM without changing batches. Checkpoint every **10 complete epochs** includes
paired model/EMA, optimizer, scheduler, RNG, history and specification digest;
an interrupted epoch restarts from the last saved checkpoint. No other user's
process is modified. GPU memory allocation is capped at 25% of the A100 capacity.

Local root: `outputs/LOCAL-PROP-008-PVT-IDMM/`. `spec.json` pins code/config,
installed model implementation, initial random weights, environment and upstream
data/evaluation artifacts. Upstream verification transitively checks all official
training image hashes and split isolation. Each architecture has `idmm/` and
`supervised/` phase checkpoints, histories, final models and completion hashes.
Top-level `status.json`, `job.json`, `heartbeat.json` and `runner.log` track work.
Smoke weights never initialize official training.

Detached launch: **15 Nov 2026 16:19:47 UTC**, supervisor PID **206091**.
Training is in progress; no candidate validation score or new submission exists
at launch. The launch receipt is a historical snapshot, not live telemetry.

Config: [LOCAL-PROP-008-PVT-IDMM.yaml](../configs/LOCAL-PROP-008-PVT-IDMM.yaml).
Historical launch receipt: [launch.json](results/LOCAL-PROP-008-PVT-IDMM.launch.json).
