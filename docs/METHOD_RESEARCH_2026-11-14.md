# Method research after the adaptive self-training pilot

Recommendation: test **source-supervised contrastive representation learning** next.
The proposed first experiment adds a supervised contrastive loss to cross entropy
under a matched source-only training recipe. After the user authorized immediate
execution, [LOCAL-PROP-006-SUPCON](../experiments/LOCAL-PROP-006-SUPCON.md)
froze the concrete configuration, passed preparation/GPU checks, and started
detached training at 20:05:05 UTC. Both arms start from identical random weights.

## Evidence from this project

The strongest verified team submission remains public F1 0.87012, ref 58278059.
A fresh `kaggle-orie` submission listing and `gh-orie` branch inspection on 14 Nov
found no newer team submission beyond our earlier blur run and no change to `kaka`
commit `b447792e3fc8ded59b98f8889023ab9f9b94c47e`. The completed adaptive
self-training pilot is documented in [LOCAL-PROP-004-SSL](../experiments/LOCAL-PROP-004-SSL.md).
It failed its primary gate against the matched control. Its apparent initial-to-B
gain does not establish a benefit from target pseudo-label consistency.

The strongest baseline's saved fold-0 predictions expose two problems. On 614
validation rows, the raw ensemble misclassifies 83 corrupted views. Of these,
32 are correct on their clean counterpart, but **51 remain wrong even when clean**.
Eight clean-view errors are correct when corrupted. Raw accuracy is 0.864821 on
the primary corrupted images and 0.903909 on clean images. This is an error-overlap
diagnostic, not a causal decomposition or a bound on another model's accuracy.
Clean counterparts must never enter validation-time adaptation or its gallery.
The overlap suggests investigating class representation as well as robustness.

I also checked the simplest test-order hypothesis: adjacent predictions agree
on 3.43% of test pairs versus a permutation mean of 3.88% (SD 0.42%, 200 shuffles).
These are model predictions, not labels. They provide no support for exploiting
contiguous class blocks in test IDs.

Already-tested team directions include rot8 TTA, multi-size averaging,
letterboxing, and inference color/contrast normalization. They are not treated as
new proposals. The historical distillation teacher uses all-fold labels, so its
fold-0 proxy is not accepted as leakage-free evidence for our new method.

## New measured geometric diagnostic

[LOCAL-PROP-005-DIAG config](../configs/LOCAL-PROP-005-DIAG.yaml) fixes fold 1,
thresholds, and a strict gallery before screening. No model is trained.

The teammate matcher applies a first-to-second descriptor ratio across the whole
descriptor database. Nearby descriptors from related images can compete with a
valid nearest match; this is a recall hypothesis, not a demonstrated software bug.
The new diagnostic instead retrieves candidate images using fold-1 neural
embeddings, then applies per-image RootSIFT ratio/mutual checks and homography
verification. It deduplicates correspondence locations, requires at least ten
inliers, checks hull coverage and reprojection error, and rejects insufficient
class margins. Every reference belongs to the training folds, excluding all
validation images and detected scene groups. It reuses the analytic gallery
at 448px/240 keypoints and extracts queries from the exact primary 352px cache;
it is not a complete ASIFT or RIFT implementation, or a replay of the original
matcher at its higher feature budget.

Results on all **607 fold-1 queries** against 2,407 references:

| Diagnostic | Result |
| --- | ---: |
| Mean retrieved candidate images per query | 43.8 |
| At least one candidate has the true class | 97.20% |
| Accepted geometric matches | 20 |
| Correct labels among accepted matches | 20 / 20 |
| Labels changed from neural baseline | 0 |
| Balanced baseline macro F1 | 0.862873 |
| Balanced F1 after accepted overrides | 0.862873 |
| Runtime | 38.67 seconds |

Thus this conservative per-image matcher adds no gain: its accepted matches cover
examples that the neural baseline already gets right. The result does not rule
out larger keypoint budgets, different detectors, affine views, or geometric
matching in general. It is insufficient evidence to select them as the main
next experiment. No test predictions or submissions were generated.
The [summary](../experiments/results/LOCAL-PROP-005-DIAG.summary.json) records the
result. All pinned inputs/code/config remained unchanged during the run, and
four new geometric tests plus the existing suite give **34 passing tests**.

## Methods investigated

### Supervised contrastive learning — primary proposal

SupCon makes different photographs of the same labeled class positives in an
embedding space while separating other classes. It is more than consistency of
two views of the same image: labels supply cross-image positive relationships.
The original paper reports better classification and natural-corruption
robustness than its cross-entropy comparisons. Its ImageNet-C ResNet-50 mCE is
67.2 versus 68.6 for its reported cross-entropy implementation; those benchmark
numbers are not a prediction of our macro F1. Large-batch dependence and changes
in optimization/augmentation are relevant caveats. [Khosla et al., SupCon,
arXiv:2004.11362v5](https://arxiv.org/abs/2004.11362).

Here, labels are the reliable information we already own. The proposed objective
directly rewards same-species structure across distinct scenes and corrupted
views without selecting pseudo-labels for test images. This is a hypothesis
motivated by our errors; no competition gain from SupCon has been measured yet.

### AugMix consistency — separate fallback, not mixed into the first ablation

AugMix combines stochastic augmentation chains with a Jensen-Shannon consistency
loss across clean and augmented predictions. Its CIFAR-100-C reported error drops
from 53.3% to 35.9% in the paper's comparison. The original operation set deliberately
excludes several benchmark corruption types; replacing it with our v4 operations
would be an adaptation requiring a new controlled protocol. This is a stronger
source-supervised robustness objective than merely increasing blur probability.
[Hendrycks et al., AugMix, arXiv:1912.02781v2](https://arxiv.org/abs/1912.02781).

PRIME also mixes spatial, spectral, and color primitives. Its parameters are
selected to preserve semantics, and stronger transforms have an accuracy/robustness
tradeoff. Its full-ImageNet results use pretrained initialization, so that part
of its benchmark setup cannot be copied under our rules; an official-data-only
scratch implementation of the augmentation itself is a different permitted
design, subject to organizer rules. It is not selected ahead of the simpler
objective ablation. [Modas et al., PRIME,
arXiv:2112.13547v2](https://arxiv.org/abs/2112.13547).

### MODS / ASIFT / RIFT2 — lower-priority retrieval research

MODS progressively synthesizes viewpoints and replaces the standard ratio with
the first geometrically inconsistent nearest neighbor (FGINN), then filters
duplicates and verifies geometry. Its evidence explains why simply merging
descriptors from many affine views can harm ordinary ratio tests.
[Mishkin et al., MODS, arXiv:1503.02619](https://arxiv.org/abs/1503.02619).

RIFT2 is a genuinely different analytic matching family: phase congruency and
log-Gabor maximum-index-map descriptors target radiometric variation, with a more
efficient rotation-invariance construction. Its published evidence concerns
multimodal image matching, largely remote sensing, rather than this plant task.
It needs neither external neural weights nor test labels, but still needs
matchable scene structure. Better feature invariance does not create a reference
for a scene absent from the gallery. Based on the small correct-but-redundant
matching coverage in our diagnostic, it remains secondary until a controlled
benchmark demonstrates a larger useful recall. [Li et al., RIFT2,
arXiv:2303.00319v1](https://arxiv.org/abs/2303.00319).

## Concrete next experiment: a controlled SupCon pilot

The points below are the research proposal. Its subsequent frozen executable
implementation and launch evidence are recorded in
[LOCAL-PROP-006-SUPCON](../experiments/LOCAL-PROP-006-SUPCON.md).

1. Use the team scene fold **2** for the new objective screen, excluding its images
   and groups from training. Baseline selection has previously used full OOF;
   this screen is conditional development evidence, not an independent final test.
2. Train a matched pair of **ResNet18d at 288px from identical random initialization**,
   using only official labeled training images outside fold 2. Keep architecture,
   data, update budget, crop/rotation/corruption, EMA, and optimizer matched.
   Do not rely on external pretraining or use the unlabeled test pool.
3. Both arms receive paired weak/strong source views with discrete class labels.
   Disable batch MixUp/CutMix in both arms. Sample multiple photos from distinct
   scene groups per class so the contrastive positives span scenes. A practical
   batch target is 30 classes × 2 photographs × 2 views = 120 embeddings.
4. **A:** cross entropy. **B:** cross entropy plus supervised contrastive loss on
   an L2-normalized projection head. Start with temperature 0.1; freeze the loss
   weight and full training budget before execution. This joint objective is an
   adaptation, not an exact replay of the paper's main two-stage method. Plain
   microbatch gradient accumulation does not supply the negatives of a larger
   contrastive batch; the actual comparison set must be handled explicitly.
5. Use 600 backbone epochs as the planning budget, matching the strong team's
   training duration rather than judging a newly initialized encoder after a
   short fine-tune. Fix the final-EMA selection rule in advance. Measure first-epoch
   runtime and VRAM before giving an ETA or launching detached background jobs.
6. Evaluate student@352 with flip TTA and its equal ensemble with the frozen
   same-fold R34d@320; apply the same fixed prior balancing and clean/blur guards.
   Compare against both the matched CE control and the original strongest team
   ensemble on the same queries. Initial screening target remains **+0.01 macro F1**
   versus both comparators; clean and mean-blur raw ensemble deltas must each be
   at least −0.01 versus both. Report uncertainty and class-level behavior.
7. Promote only on adequate confirmation across the other folds. Preserve the
   current best submission until validation justifies replacement. A 0.99 target
   is not established by the papers or our diagnostics.

The important change is learning the class representation with true cross-image
relationships. AugMix-JSD and new geometric descriptors remain separately testable
alternatives, rather than extra components added to an uninterpretable first run.

## Research artifacts and status

Local evidence is in `outputs/LOCAL-PROP-005-DIAG/`: primary-paper methods/metadata
under `research/`, `preflight.json`, `matches.json`, `result.json`, and
`clean_corrupt_error_overlap.json`. The screening runner is
`scripts/screen_pairwise_geometry.py`. It completed successfully in CPU background;
the diagnostic and earlier SSL pilot have completed. The separate SupCon A/B
training job is active as documented above.

No dependencies were changed, no team files/processes were modified, no external
dataset or pretrained model was downloaded, and no Kaggle resource was changed.
