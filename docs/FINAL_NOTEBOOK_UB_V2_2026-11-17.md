# Comprehensive UB notebook, version 2

V2 consolidates the submission narrative in **one main notebook**,
`DATERAQUEST_UB_Lengkap.ipynb`, and an offline HTML preview. EDA figures are
embedded in Markdown attachments so they remain visible when code outputs are
collapsed. The original V1 deliverables and all teammate inputs remain intact.

## Content

1. Official raw-data inventory, ordered IDs, dimensions and class mapping.
2. Complete class counts, imbalance, domain flags, and a fixed-seed 400+400 raw
   quality audit: saturation, brightness, sharpness, black fraction and near-gray.
3. Original team EDA: class distribution, all 30 classes, test gallery and
   train/test statistical distributions.
4. Full preprocessing/augmentation source and visual examples; an explicit
   table distinguishes base/teacher augmentation from the student recipe.
5. Random-initialized base architecture and full 600-epoch training source,
   original config/log, and optional retraining command.
6. Teacher pseudo-label construction with its actual ensemble/cache provenance.
7. **Actual new student distillation**, streamed training log and learning curve.
8. Recomputed scene-fold0 teacher-subset metrics, confusion matrix and per-class
   F1 from saved validation probabilities, explicitly scoped to that subset.
9. Fresh raw-JPEG student inference, CSV format/order/labels, 2,010-row agreement
   and exact SHA256 verification against ref 58470727 / public 0.90780.

The deployed classifier is one ResNet34d. It is a distilled model using official
train labels plus teacher pseudo-labels on test. V2 does not describe it as
trained exclusively from raw train labels. The base and teacher are historical
scratch experiments, not freshly retrained in this revision. Student agreement
is not independent validation. The student training is rerun; actual results
and artifact hashes are in the [receipt](../experiments/results/FINAL-NOTEBOOK-UB-V2-2026-11-17.json).

## Execution and deliverables

Executed successfully: **21/21 code cells**, **52 total cells** after embedding
the figures. Fresh distillation reached **2,010/2,010 teacher agreement at
epoch 7** (378 seconds after training/evaluation initialization). The complete
notebook run took **486.42 seconds**. Subsequent raw-JPEG inference took
**6.88 seconds** of model time, and its CSV is byte-identical to the reference:
`59b9e62480311d8dfbdc41cf3329a7e5ec4a5726d619032d746127c2b3b13aa7`.

Fresh student checkpoint SHA256:
`e02211eae4db942092548b59d5e1c98130b35d1fc6e2fe218610b67402dbf8cc`.
The recomputed teacher-subset validation has **614 rows**, macro F1
**0.877528**, accuracy **0.871336**; this is not student validation. All ten
figures plus the crest were decoded and verified in both notebook attachments
and offline HTML. Original inputs were hash-checked unchanged, and archive CRCs
passed. The runtime packages the freshly trained student, not the old weights.
A separate clean-folder Run All with `redistill=False` loaded the runtime from
the delivery copy and independently reproduced the exact CSV with the same
fresh checkpoint; all 21 cells passed again without training a second student.

```bash
python scripts/revise_final_ub_complete.py
python scripts/execute_final_ub_notebook.py \
  outputs/FINAL-UB-V2-2026-11-17/DATERAQUEST_UB_Lengkap.ipynb
python scripts/package_final_ub_complete.py
```

- `pensi_lomba_data_FINAL_V2.zip`: one executed notebook, offline HTML, exact CSV,
  README, environment and execution/verification receipts.
- `pensi_lomba_data_RUNTIME_V2.zip`: one base scratch checkpoint for training,
  one fresh student checkpoint for deployment, teacher/validation probability
  caches, folds, figures, log and manifest. The raw dataset remains separate.
- Local output: `outputs/FINAL-UB-V2-2026-11-17/`.
- Delivery copy: `/home/nafisnaufal1426/akucintakc/final-notebook/revisi_ub_v2/`.

Extract both archives into the same folder for a student-training rerun. Set
`DATERA_DATA_ROOT` and choose the GPU environment in the snapshot. Default
`redistill=True` performs student training; `False` replays inference using the
packaged fresh student. `retrain_base=True` additionally repeats 600 base epochs
and takes several hours. Regenerating all teacher models is not a portable
default of this student notebook. The actual registration ID must be supplied
in the portal filename if required; it was not invented.

GitHub source retains aggregate EDA/evaluation plots but omits raw competition
image galleries and code outputs. Use the local executed notebook or HTML to
review all figures. Only Markdown was added for offline HTML rendering, with
`--no-deps`; no shared package was upgraded/replaced. No Kaggle/portal upload.
