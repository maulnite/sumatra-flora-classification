# LOCAL-PROP-006-A-PUBLIC — CE control fold-2 replacement

The user explicitly requested a Kaggle submission on 15 November 2026 to inspect
public performance. This public probe uses **CE control A**, which outperformed
CE+SupCon B in the completed [matched experiment](LOCAL-PROP-006-SUPCON.md).
The failed SupCon promotion gates remain failed; this submission does not promote B.

## Frozen candidate

- Scratch ResNet18d, final EMA epoch 600, source-only scene fold 2, seed 44.
- Encoder: `outputs/LOCAL-PROP-006-SUPCON/A/model.pt`, SHA-256
  `b11f38515245a222b4d422920b2bf6537ef78a252c489c5d8bda08a1ee4c0dab`.
- Replace `R_r18d_i288_e600_f2` in the strongest EXP-024 ten-checkpoint ensemble.
  Retain the other four R18d and all five R34d reference members. Each family
  has weight 0.5; each member has total weight 0.1. This is a replacement of
  one existing member, not an extra 90:10 blend with the full original ensemble.
- A test inference: 352px cached official images, batch 32, BF16, channels last,
  original ImageNet normalization and horizontal-flip probability averaging.
- Original train-frequency prior balancing, 100 iterations; original SIFT
  override at at least five inliers. No thresholds or weights searched on public.
- Config: [LOCAL-PROP-006-A-PUBLIC.yaml](../configs/LOCAL-PROP-006-A-PUBLIC.yaml).
  Runner: [prepare_ce_public_probe.py](../scripts/prepare_ce_public_probe.py).

## Validation evidence and scope

On 603 held-out fold-2 queries, **A plus the matching R34d fold-2 model** has
primary balanced macro F1 **0.834465**, versus **0.826344** for the old matching
R18d/R34d pair: **+0.008120**. A alone has raw primary F1 **0.835351**.

These are not the OOF scores of the complete replacement ensemble. The other
folds' checkpoints trained on fold-2 images, so evaluating that complete ensemble
on these queries would leak training examples. A's other four folds have not
been trained. Public performance is a separate measurement; private is unavailable.

## Verification and artifacts

The preparation runner checks pinned training inputs/code/environment, checkpoint
SHA-256, test-cache provenance, ID order and all ten stored reference predictions.
It requires the original family means and baseline CSV to reproduce exactly,
replays A's validation forward pass against saved probabilities, then performs
the actual test forward pass. It validates 2,010 ordered IDs and integer labels
0–29, rejects a candidate identical to baseline, and rechecks input hashes.
It performs no remote upload itself.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1 \
  /home/nafisnaufal1426/venvs/dq/bin/python -m scripts.prepare_ce_public_probe
```

Local artifacts are under `outputs/LOCAL-PROP-006-A-PUBLIC/`: inference log,
raw A probabilities, raw/balanced ensemble probabilities, changed predictions,
exact submission CSV, provenance in `preparation.json`, and Kaggle receipts.

## Verified public result

Submission **58374703**, submitted **2026-11-15 07:28:10 UTC**, status **COMPLETE**:

| Candidate | Public macro F1 | Delta vs team best |
| --- | ---: | ---: |
| EXP-024 original, ref 58278059 | 0.87012 | — |
| CE A fold-2 replacement, ref 58374703 | **0.86464** | **−0.00548** |

The prepared CSV changed **56 / 2,010 labels** versus the best team submission.
It has SHA-256
`d0fd851ddd8a9a121aafd85fb4c2f1c5d327e745cf0b37581ffc2cd0d2330895`.
The 393 accepted SIFT rows changed 30 neural labels in this candidate.
Preparation took **13.67 seconds**; A's validation replay was byte-identical in
probability values (maximum difference 0). The baseline CSV was also identical.

The local fold-2 gain did not translate into a higher public score for this
specific ensemble replacement. The strongest observed public submission remains
EXP-024 at **0.87012**. No additional weights, thresholds, or submissions were
tried after observing this result. Private score remains unavailable. The result
does not establish how a five-fold ensemble trained with A's recipe would score.

[Machine-readable result](results/LOCAL-PROP-006-A-PUBLIC.summary.json).
