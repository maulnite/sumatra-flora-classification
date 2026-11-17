# Final notebooks — pensi lomba data

Members: **Nafis Naufal Rahman**, **Orie Abyan Maulana**,
**M Kaka Chadafi F**. Cover uses the official Universitas Brawijaya crest,
embedded as an offline attachment. No registration ID was invented.

The user selected preserving **0.90780 with explicit distillation provenance**.
The final notebook does not describe teacher pseudo-labels as official test
labels or imply a fresh raw-train-only model achieved that score.

## Deliverables and validation

Local directory: `outputs/FINAL-UB-2026-11-17/`.
Delivery copy: `/home/nafisnaufal1426/akucintakc/final-notebook/revisi_ub/`.

- `DATERAQUEST_UB_Single_Model.ipynb`: all **18 code cells** freshly executed;
  EDA, preprocessing and formula replay retained, with actual raw-JPEG student
  inference. Historical base/distillation logs are explicitly identified.
- `DATERAQUEST_UB_Ensemble.ipynb`: all **36 original executed code cells** and
  their outputs remain identical to the supplied ensemble notebook; cover and
  introductory navigation are revised. Historical training was not rerun.
- `submission.csv`: **2,010/2,010** labels and every byte match Nafis's
  `final_domain.csv`, Kaggle ref **58470727**, historical public **0.90780**.
- `pensi_lomba_data_final.zip`: two executed notebooks, CSV, README and receipts.
- `pensi_lomba_data_final_lengkap.zip`: additionally includes the portable
  student runtime: one checkpoint, teacher probability cache, logs and figures.
  This enables student replay, not a portable rerun of all 20 teacher trainings.

Fresh student inference: **one ResNet34d**, **21,319,294 parameters**,
448px INTER_AREA resize, ImageNet normalization, original/flip probability
average. Load/model/resize **11.10s**, model inference **7.36s** on the shared
A100. The main notebook finished in **43.11s**. This is the already-distilled
student checkpoint, not newly trained weights. No teacher/SIFT/postprocessing
is called on its deployment inference path.

CSV SHA256:
`59b9e62480311d8dfbdc41cf3329a7e5ec4a5726d619032d746127c2b3b13aa7`.

Agreement establishes identical test predictions; it is not independent student
validation. The public score remains attached to the historical Kaggle CSV.
No new submission or notebook upload was performed. Original archives,
reference notebook, scripts, weights and prediction inputs were hash-checked
unchanged before packaging. Both ZIPs passed CRC checks and embedded CSV hash
verification. All code syntax and notebook schemas passed; no error outputs.

## Reproduction and versioning

See [notebook source and commands](../notebooks/final_ub/README.md),
[build script](../scripts/build_final_ub_notebooks.py),
[execution script](../scripts/execute_final_ub_notebook.py),
[packaging script](../scripts/package_final_ub_notebooks.py), and
[verification receipt](../experiments/results/FINAL-NOTEBOOK-UB-2026-11-17.json).

GitHub notebooks omit competition images/prediction outputs. Full executed
notebooks, CSVs, arrays, ZIPs and weights stay local. Optional retraining flags
are disabled by default; rebuilding the student requires the team's scratch
base checkpoint or repeating its 600-epoch base training. No raw dataset is
redistributed. The original EDA and augmentation implementation is retained.

The supplied Three Makasih notebook informed only cover/narrative structure;
its model code, data, results and identities are not copied. UB crest provenance:
[official brand page](https://www.ub.ac.id/id/about/ub-brand/) and
[original image](https://www.ub.ac.id/wp-content/uploads/2024/08/cropped-favicon-270x270.png).

Installed nbformat/nbclient and missing schema dependencies as new packages
without changing existing shared packages. The tested runtime snapshot is in
both ZIPs. Final filenames use the supplied team name; add the actual registered
team ID before portal upload if its `IDTim_NamaTim.zip` convention is required.
