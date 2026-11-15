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

Detached supervisor launched **15 Nov 2026 07:39:44 UTC**, PID **11058**. It writes
a heartbeat every 60 seconds and checkpoints both backbones after each complete
epoch. CUDA OOM restarts from the last complete paired epoch without changing
the batch or optimizer recipe. Other users' processes are untouched.

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
and final `result.json`. Training is in progress; no candidate score is available
at launch. Cold smoke timings include first-use CUDA overhead and are not a
reliable whole-training ETA.
