# RESEARCH-010 — BioCLIP 2.5 Huge partial fine-tuning

**Separate pretrained research experiment, not eligible for DATERAQUEST.** The
user explicitly requested trying the hypothetical pretrained option and leaving
it in the background. Audited competition rules require scratch training and
exclude BioCLIP. This pipeline has no test inference, submission CSV generation
or upload operation, and refuses configs marked competition-eligible.

## Model and local validation

[BioCLIP 2.5 Huge](https://huggingface.co/imageomics/bioclip-2.5-vith14), ViT-H/14,
revision `6e3d04e3d6522012c88181085c5ae666e14c45cd`. The public 3,944,517,804-byte
safetensors file was downloaded without account authorization and verified
against its published LFS SHA256:
`ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97`.
Only the image encoder is constructed, using OpenCLIP's official vision factory;
all visual checkpoint keys load strictly. No text model or taxonomy prompts.

Use **scene fold 2, seed 44: 2,411 local train / 603 validation photographs**.
Train and validation groups remain disjoint. Only official training photographs
enter local optimization. The external pretraining dataset's potential overlap
with our local images is unknown; this is a locally held-out transfer experiment.

Primary measure: **standalone macro F1 on the fixed corruption validation view**,
without balancing or SIFT. Report clean, blur06, blur10 and mean blur separately;
equal raw averaging with the frozen matching R34d is a secondary diagnostic.
The same held-out views are resized bicubic from 352px to **224px**, the model's
native resolution. This is a different resolution from scratch-model inference.
Use the checkpoint's **CLIP mean/std**, and horizontal flip probability averaging.
Clean validation is not the public leaderboard.

## Fixed two-stage protocol

1. **Frozen linear probe.** Extract normalized 1,024-dimensional features from
   original and horizontally flipped images. Fit multinomial logistic regression
   with fixed **C=10**, maximum 2,000 iterations. Source features are from 2,411
   photographs with two orientations, not 4,822 independent photographs. Evaluate
   the four held-out views and initialize the new 30-class head from this probe.
2. **Actual partial fine-tuning, 30 epochs / 2,280 updates.** Open **four final
   transformer blocks out of 32**, final LayerNorm/projection, and classifier.
   Update **80,053,790 parameters**; the visual encoder totals **632,076,800**.
   AdamW backbone LR **1e-5**, classifier LR **1e-3**, warmup two epochs, cosine
   to 1% LR, decay .01 excluding scalar/vector parameters, smoothing .05,
   gradient clipping 1, EMA .99. Effective batch **32**, microbatch **8**, BF16.
   Include the final 11 photographs and partial microbatch each epoch.

Reuse the frozen source sampler and flora augmentation code: uniform epoch
permutation, native ±45-degree rotation p=.5, light-color/corruption-v4 equally,
crop scale .5–1 and aspect .75–1.333, horizontal flip. MixUp alpha .2 with p=.3.
Evaluate every five epochs and select **final epoch-30 EMA**, without early
stopping or a public weight search. Save only trainable deltas and optimizer/EMA
state; frozen base weights are recovered from the pinned pretrained checkpoint.

## Verification and background execution

Three behavioral tests passed for research scope, selected/frozen parameter
updates, strict delta restoration and restoration after an EMA evaluation error.
Real-source GPU smoke passed a full 32-photo update, finite loss/gradient,
frozen-parameter preservation, a changed transformer parameter and exact delta
checkpoint roundtrip. Smoke weights do not initialize the official run.
Peak smoke allocation **3.96 GiB**. Measured update **2.03 seconds** projects
**1.28 compute hours**, excluding probe extraction, evaluation and checkpoint
overhead. Initial practical estimate **1.5–2.5 hours**, subject to GPU sharing.

Detached **17 Nov 15:21:51 UTC / 22:21:51 WIB**, supervisor **968744**, runner
**968746**. It first builds the frozen-probe baseline, then fine-tunes. Heartbeat
60 seconds; atomic checkpoint after each complete epoch; same-batch OOM restart
from the last complete epoch. GPU sharing remains authorized, allocation cap 25%
of the shared A100. The launcher prevents duplicate supervisors. The run can
continue after the competition deadline as separate research.

Added **open_clip_torch 3.3.0** and **ftfy 6.3.1** with `--no-deps`. Previously
installed torch/torchvision/timm/Hugging Face/transformers versions were checked
and remain unchanged. [Additional dependencies](../requirements-research-bioclip.txt).

- [Config](../configs/RESEARCH-010-BIOCLIP25-FT.yaml)
- [Pinned downloader](../scripts/download_bioclip_research.py)
- [Runner](../scripts/train_bioclip_research.py)
- [Launcher](../scripts/launch_bioclip_research.py)
- [Historical launch/verification receipt](results/RESEARCH-010-BIOCLIP25-FT.launch.json)
- Local artifacts: `outputs/RESEARCH-010-BIOCLIP25-FT/`.
  Inspect `job.json`, `heartbeat.json`, `status.json`, `linear_probe.json`,
  `history.json`, `checkpoint.pt`, `result.json`, `runner.log`.

```bash
PY=/home/nafisnaufal1426/venvs/dq/bin/python
$PY scripts/download_bioclip_research.py
$PY -m unittest tests.test_bioclip_research -v
$PY scripts/train_bioclip_research.py --prepare
$PY scripts/train_bioclip_research.py --smoke
$PY scripts/launch_bioclip_research.py
```

Initial code/config/data/checkpoint hashes are frozen in the local spec. Do not
rewrite the weights receipt or prepared inputs while the run is active. Existing
verified downloads return without changing that receipt. The launch receipt is a
historical snapshot; subsequent stage results follow below and in output files.

## Frozen-probe result and fine-tuning start

The source-feature linear probe completed **15:24:58 UTC**:

| Local macro F1 | Frozen BioCLIP + trained linear head |
| --- | ---: |
| Primary corruption view, standalone | **.736072** |
| Clean, standalone | **.987217** |
| Mean blur, standalone | **.757493** |
| Matching equal R34d pair, raw primary | **.883058** |

Clean performance is strong, while the synthetic corruption shift still hurts
the frozen encoder. These are local validation measures, not public scores.
They are not directly interchangeable with scratch metrics using a different
resolution, prior balancing or postprocessing. Fine-tuning began at
**15:25:20 UTC / 22:25:20 WIB**, with a verified real optimizer update in epoch1.
[Frozen-probe verification and active-state snapshot](results/RESEARCH-010-BIOCLIP25-FT.probe.json).
The 30-epoch fine-tuning result is pending.
