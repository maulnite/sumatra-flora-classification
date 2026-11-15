# LOCAL-PROP-007-BACKBONES — scratch backbone exploration

The user requested a different backbone after the ResNet-only experiments and
the CE-A public probe scored 0.86464, below the team's best 0.87012. This run
compares **EfficientNetV2 Tiny** and **ConvNeXt Tiny**, both randomly initialized,
against the existing ResNet18d CE A on the same development fold.

## Candidates and evidence

| Candidate | `timm` architecture | Parameters, 30-class head | AdamW peak LR | Warmup |
| --- | --- | ---: | ---: | ---: |
| EfficientNetV2 Tiny | `efficientnetv2_rw_t` | 12,655,138 | 0.002 | 3 epochs |
| ConvNeXt Tiny | `convnext_tiny` | 27,843,198 | 0.0005 | 20 epochs |

[EfficientNetV2](https://arxiv.org/abs/2104.00298) combines MBConv and Fused-MBConv
blocks. The installed `timm` source explicitly identifies `efficientnetv2_rw_t`
as a custom scaled Tiny variant, **not a variant reported in the paper**.
[ConvNeXt](https://arxiv.org/html/2201.03545v2) uses large depthwise kernels,
inverted bottlenecks and LayerNorm; its paper documents AdamW, cosine decay,
warmup and stochastic depth for scratch ImageNet training. Those published
large-dataset results do not establish a gain on our 3,014-image dataset.

The hypothesis is that these different feature extractors may improve accuracy
or complement R34d's errors. Architecture-specific LR/warmup and drop-path 0.1
make this a practical **backbone plus recipe comparison**, not a causal isolation
of architecture. The ConvNeXt LR is a frozen local choice; it is not presented as
the paper's original batch-4096 recipe. Both candidates have dropout 0.2.

## Frozen protocol

- Official source images only; `pretrained=False` is enforced in the factory.
- Team scene fold **2**, **2,411 training / 603 validation**, seed **44**.
- Same source sampler and paired-view implementation as LOCAL-PROP-006 CE A:
  30 classes × two distinct-scene photographs × two views = **120 views/update**.
  The deterministic epoch/step draws, original-size rotation, light-color and
  corruption-v4 views, and shared crop seeds are reused without modification.
- Train at **288px**, **600 epochs / 24,000 updates per backbone**. Each model
  processes each identical source batch in turn. No SupCon or pseudo-label loss.
- AdamW, weight decay 0.05, cosine decay to 1e-6, label smoothing 0.1,
  gradient clipping 5, EMA 0.999, BF16 and channels last. No MixUp/CutMix.
- EMA evaluation at **352px**, original normalization and flip TTA. Reuse the
  previously frozen primary, clean, blur0.6 and blur1.0 fold-2 caches; no new
  corruption draws. Fixed R34d fold-2 probabilities remain unchanged at 320px.
- Main metric: 50:50 candidate/R34d pair, original 100-iteration train-prior
  balancing before SIFT. Raw standalone and ensemble diagnostics also saved.
- Compare final epoch 600 against the original team pair **0.826344** and
  final CE A pair **0.834465**. Evaluations every 50 epochs show learning curves;
  intermediate peaks do not replace the preregistered final-epoch result.
- Gate: primary gain at least **0.010 vs team pair** and **0.005 vs CE A pair**;
  clean and mean-blur raw ensemble drops no worse than 0.01 against either.
  Conditional paired scene-group bootstrap uses 2,000 replicates.

This is one development fold, not full OOF. Other fold checkpoints cannot be
averaged into its validation predictions because they trained on these queries.
No test images are used for this experiment. Additional folds and Kaggle upload
are separate follow-ups; the runner contains no remote submission operation.

## Execution and verification

Eight relevant tests passed, including scratch-only construction, independent
optimizer configuration, wrong-fold/target-use rejection, incompatible-checkpoint
rejection, and the reused source-sampling guards. A **real full 120-view GPU update
for each backbone** passed with finite loss/gradients and changed encoder weights.
Optimizer/scheduler steps and both model/EMA checkpoint roundtrips passed.
Peak smoke allocation **12.17 GiB**. Initial random weights, runner/config,
installed model source, environment versions and upstream input hashes are pinned.

Detached supervisor ran **15 Nov 2026 07:39:44–13:58:26 UTC**, PID **11058**.
Both backbones completed **600 epochs / 24,000 updates**. Wall time was **6h18m43s**,
exit code **0**, with **zero OOM restarts**. The supervisor wrote a heartbeat every
60 seconds and checkpointed both backbones after each complete epoch. Other
users' processes were untouched.

```bash
PY=/home/nafisnaufal1426/venvs/dq/bin/python
$PY scripts/explore_backbones.py --prepare
$PY scripts/explore_backbones.py --smoke
$PY scripts/launch_backbone_exploration.py
```

Config: [LOCAL-PROP-007-BACKBONES.yaml](../configs/LOCAL-PROP-007-BACKBONES.yaml).
Local artifacts: `outputs/LOCAL-PROP-007-BACKBONES/`, including `spec.json`,
`smoke_result.json`, `job.json`, `heartbeat.json`, `status.json`, `runner.log`,
`paired_checkpoint.pt`, per-backbone EMA weights/validation probabilities/history,
and final `result.json`. Cold smoke timings include first-use CUDA overhead and
are not a reliable whole-training ETA.

## Verified final result

All scores below are **fold-2 local macro F1**, final epoch 600. Primary, clean
and blur use the same fixed 50:50 candidate/R34d fold-2 ensemble; primary is
prior-balanced, while clean/blur are raw. The last column evaluates the new
student alone, without R34d or prior balancing.

| Student | Primary balanced pair | Clean raw pair | Mean-blur raw pair | Standalone raw primary |
| --- | ---: | ---: | ---: | ---: |
| Original team ResNet18d | 0.826344 | 0.871937 | 0.594269 | 0.810467 |
| Matched ResNet18d CE A | **0.834465** | **0.875798** | 0.627330 | **0.835351** |
| EfficientNetV2 Tiny | 0.805343 | 0.839179 | **0.662033** | 0.743845 |
| ConvNeXt Tiny | 0.796876 | 0.843716 | 0.650868 | 0.709219 |

EfficientNetV2 primary is **−0.021001** versus the original team pair and
**−0.029121** versus CE A. ConvNeXt is **−0.029468** and **−0.037588**, respectively.
Both fail the primary-gain requirements and clean guards. Both pass mean-blur
guards and improve that diagnostic versus CE A (+0.034703 / +0.023538), but this
does not compensate for the primary and clean losses under the frozen gate.

Conditional paired scene-group bootstrap 95% intervals for primary deltas:

| Candidate | Versus team pair | Versus CE A |
| --- | --- | --- |
| EfficientNetV2 Tiny | [−0.046261, +0.002075] | [−0.054791, −0.006418] |
| ConvNeXt Tiny | [−0.052384, −0.008829] | [−0.065453, −0.013058] |

**Neither tested recipe is promoted to further folds or Kaggle submission.**
The strongest observed public result remains the original EXP-024 **0.87012**;
this experiment did not produce a test/public score. One development fold, seed
and recipe per architecture do not establish intrinsic architecture superiority.
In particular, optimizer settings and stochastic depth differ from CE A, as
declared before training; a different recipe remains an untested hypothesis.

Post-training CPU verification recomputed all five metrics from stored
probabilities, checked that each saved ensemble equals the exact student/R34d
mean, and independently recomputed the promotion gates. Both final model files
match their EMA states in the paired epoch-600 checkpoint exactly. Both optimizer
and scheduler states confirm 24,000 updates, with 600 contiguous history entries.
All pinned upstream data/code/config/model-source hashes and environment versions
remain unchanged. This verification did not rerun training or GPU inference.

[Machine-readable final summary](results/LOCAL-PROP-007-BACKBONES.summary.json)
contains metrics, gates, conditional intervals, execution receipt, checkpoint
hashes and final verification. The earlier `.launch.json` remains a historical
snapshot of the running job at launch.
