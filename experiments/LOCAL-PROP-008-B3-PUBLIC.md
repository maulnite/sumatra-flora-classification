# LOCAL-PROP-008-B3-PUBLIC — one PVTv2-B3 replacement

The final source-only IDMM candidate in
[LOCAL-PROP-008](LOCAL-PROP-008-PVT-IDMM.md) passed its frozen development-fold
gate. The user had explicitly authorized a Kaggle submission if the score
increased. One fixed probe was prepared and submitted; no public weight search
was performed.

## Candidate and composition

- PVTv2-B3, all weights initially random; 800-epoch own-source IDMM at 112px,
  then 800-epoch supervised training at 224px; final EMA, scene fold 2, seed 44.
  No test image or external weight initializes the transformer.
- Checkpoint SHA256:
  `d6074465968be104f819c57706b9eba718069a6c99466081f4d1fdf4bf5e20c2`.
- Replace `R_r18d_i288_e600_f2` in the original EXP-024 ensemble. Retain the
  other four R18d and all five R34d members. PVT weight is **0.1**; the other
  nine members retain weight 0.1 each. Original float32 family reduction order,
  train-frequency prior balancing (100 iterations), and SIFT ≥5 inliers remain.
- PVT test inference: cached official images at **352px**, batch **16**, BF16,
  channels last, original normalization and horizontal-flip probability averaging.
- Config: [LOCAL-PROP-008-B3-PUBLIC.yaml](../configs/LOCAL-PROP-008-B3-PUBLIC.yaml).
  Runner: [prepare_pvt_public_probe.py](../scripts/prepare_pvt_public_probe.py).

On 603 held-out fold-2 queries, the equal B3/R34d pair has balanced primary
macro F1 **0.865532**, versus **0.834465** for the stronger CE A/R34d pair,
delta **+0.031067**. B3 alone has raw primary **0.862227**. Clean and blur guards
also pass. These are matching-pair metrics, **not full replacement-ensemble OOF**:
the other fold checkpoints trained on these validation queries. Only one new
PVT fold exists. No inference of the full ensemble on these queries is used.

## Verification and submission

The preparation checks training inputs/code/config/environment and checkpoint
hashes, final epoch and gate, all ten original test probability files, cache
provenance and ID order. The original best CSV reconstructs byte-for-byte.
PVT validation inference also reproduces its saved probabilities **exactly**,
with maximum difference 0 and no changed argmax labels. Two new behavioral tests
check exact one-member replacement, preserved input arrays/normalization, and
invalid family/probability rejection. Original training tests remain unchanged.

Preparation took **46.63 seconds**. The candidate CSV has **2,010 valid ordered
rows**, changes **85 labels** from EXP-024, and has SHA256
`27febd1762cba20185ff09c5b22f34ef85175a384f494bd204219b237b13e02c`.
The fixed SIFT rule accepts 393 rows and changes 26 candidate neural labels.
Input hashes are rechecked after inference. The preparation runner does not
upload; the separate `kaggle-orie` action records a single submission journal.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1 \
  /home/nafisnaufal1426/venvs/dq/bin/python -m scripts.prepare_pvt_public_probe
```

Kaggle **ref 58446643**, submitted **16 Nov 2026 22:37:45 UTC**, **COMPLETE**:

| Candidate | Public macro F1 | Delta vs previous best |
| --- | ---: | ---: |
| Original EXP-024, ref 58278059 | 0.87012 | — |
| **B3 one-fold replacement, ref 58446643** | **0.87197** | **+0.00185** |

This is the highest observed team public score at recording. The gain is modest
and does not establish a private-score improvement or the performance of a
five-fold PVT ensemble. No further weight, threshold or submission was tried
after this probe. The independently scheduled B0 comparison continues.

Local artifacts: `outputs/LOCAL-PROP-008-B3-PUBLIC/`, including test/validation
probabilities, exact CSV, changed predictions, preparation provenance, CLI logs,
submission journal and Kaggle receipt. These data/model artifacts are ignored.
[Machine-readable result and receipt](results/LOCAL-PROP-008-B3-PUBLIC.summary.json).
