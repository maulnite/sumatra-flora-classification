# Code map: what each script does, where it lives, and how to run it

This page explains the scripts behind the experiments in `experiments/`. Each entry says what the script does, what it reads and writes, how to run it, and which experiments used it.

> **Status.** The scripts are now in this repository under `code/`, and the two notebooks under `notebooks/`. Their run summaries (configs, per-fold metrics, training logs) are under `experiments/runs/`, `experiments/logs/`, and `experiments/sift/`. Predictions (`*_submission.csv`, probability arrays `*.npy`) and model weights are **not** in the repository by design.
>
> Paths below that start with `~/inahuy/dateruy/` refer to the working folder on the training server. The same scripts live in `code/` here.

## Where the code runs

- **Machine:** a shared multi-user JupyterHub server with one NVIDIA A100 (80 GB). Other users run jobs on the same GPU, so the team keeps its own GPU memory under about 50 GB.
- **Access:** log in to JupyterHub in a browser with the team account. Open a terminal from the Launcher (or File → New → Terminal).
- **Working folder:** `~/inahuy/dateruy/` (the account's home folder, then this folder). All scripts below run from this folder.
- **Python environment:** a virtualenv at `~/venvs/dq`. Activate it in the terminal before any command:
  ```bash
  cd ~/inahuy/dateruy
  source ~/venvs/dq/bin/activate
  ```
- **Data:** the official competition files are in `dateraquest-2-0-machine-learning-competition/` (`train/`, `test/`, `train.csv`, `class_mapping.csv`, `sample_submission.csv`). They are not in git and must not be published.
- **Caches:** pre-computed image arrays live in `cache/`. They are created once and reused by every run.
- **Outputs:** each run writes to `runs/`. Folder names follow `<run>_f<fold>` (per fold) and `<run>_submission.csv` / `_oof.npy` / `_ptest.npy` (combined).

## Scripts

### `train.py` (main training script)

- **Purpose:** trains one model per fold, runs the 5-fold loop, and builds the ensemble for one recipe.
- **Commands:**
  - `python train.py prep`: build the fold file and the validation and test caches (run once).
  - `python train.py fold --recipe R --fold 0`: train one fold.
  - `python train.py run --recipe R --jobs 3 --tag NAME`: train all 5 folds and write the combined submission.
  - `python train.py ensemble --recipe R --tag NAME`: rebuild the combined files from finished folds.
- **Recipes** (defined in the `RECIPES` table): `E`, `F`, `H`, `R`, `L`. Options such as `--arch`, `--attn`, `--pool`, `--img`, `--cmode`, `--split`, `--tta`, `--epochs` override the recipe.
- **Important options:**
  - `--split scene` (default) or `--split phash` (the fold layout used for recipes E and F in the notebook).
  - `--img` is the training resolution (default 224).
  - `--cmode` picks the corruption: `corrupt_v4`, `corrupt_hard`, `corrupt_light`, `corrupt_v5`.
  - `--wandb PROJECT` logs to Weights & Biases. The team account must be logged in on the server first.
- **Reads:** `dateraquest-2-0-machine-learning-competition/`, `cache/`.
- **Writes:** `runs/<tag>_f<k>/` with `model.pt`, `pv.npy` (validation probabilities), `pt.npy` (test probabilities), `preds.npz`, `hist.csv`, `cfg.json`, `done.json`, `log.txt`. The ensemble step also writes `runs/<tag>_submission.csv`, `runs/<tag>_oof.npy`, and `runs/<tag>_ptest.npy`.
- **Used by:** EXP-007 through EXP-020, EXP-025. EXP-004 and EXP-005 used the notebook equivalent (`--split phash`).

### `dq.py` (data and augmentation module)

- **Purpose:** holds the augmentation pipelines (`corrupt`, `corrupt_v4`, `corrupt_hard`), image reading, and the resize helper. It is imported by `train.py` and by the notebook.
- **Reads/writes:** nothing on disk.
- **Used by:** every training run.

### `dateraquest.ipynb` (exploration and first experiments)

- **Purpose:** the original exploration notebook. It contains the data analysis and the first recipe E and F runs (5-fold with pHash groups).
- **Run:** open it in JupyterLab and run the cells top to bottom. Results from the original run are summarised in EXP-004 and EXP-005.
- **Used by:** EXP-004, EXP-005, EXP-006 (fold 0 only).

### `evalres.py` (inference resolution, ensembles, and label transfer)

- **Purpose:** the main evaluation script for the late experiments.
  - `probs`: for each run and each inference resolution, computes out-of-fold and test probabilities and saves them under `runs/probs/`.
  - `search`: greedily picks the best ensemble on OOF, optionally applies prior balance, optionally replaces labels with SIFT matches, and writes one submission.
- **Commands:**
  ```bash
  python evalres.py probs R_r18d_e600 R_r34d_e600 R_r18d_i288_e600 --sizes 224 288 320 352
  python evalres.py search R_r18d_e600 R_r34d_e600 R_r18d_i288_e600 --tag NAME --sift 5
  ```
- **Reads:** `runs/<run>_f*/model.pt`, `cache/corr4r_<size>.npy`, `cache/test_<size>.npy`, `runs/sift_test.csv` (only when `--sift` is set).
- **Writes:** `runs/probs/`, `runs/<tag>_submission.csv`.
- **Used by:** EXP-018 to EXP-024.

### `ens_eval.py` (exhaustive ensemble search, earlier)

- **Purpose:** tries all combinations of 1 to 4 finished runs on OOF and prints the top results. Used before `evalres.py` existed.
- **Reads:** `runs/<run>_oof.npy`, `runs/<run>_ptest.npy`.
- **Writes:** `runs/ens_best_submission.csv`.
- **Used by:** EXP-015.

### `ens320.py` (ensemble at resolution 320)

- **Purpose:** recomputes OOF and test probabilities at inference resolution 320 for three runs and writes the prior-balanced ensemble.
- **Reads:** `cache/corr4r_320.npy`, `cache/test_320.npy`, and the model files of the three runs.
- **Writes:** `runs/ens320_submission.csv` and `runs/ens320_balance_submission.csv`.
- **Used by:** EXP-016 (the 320 OOF value) and EXP-022.

### `sift_match.py` (SIFT matching of test images to train)

- **Purpose:** matches each test image, and each corrupted train image, to the train set with SIFT features and RANSAC. It also prints a precision and coverage table for each inlier threshold.
- **Run:** `python sift_match.py` (takes roughly 10 minutes; uses CPU heavily).
- **Reads:** the train and test images.
- **Writes:** `runs/sift_test.csv` (label and inlier count per test image) and `runs/sift_trainq.csv` (the same for corrupted train images).
- **Used by:** EXP-023.

### `sift_oof_check.py` (OOF proxy for SIFT)

- **Purpose:** applies SIFT label transfer to the OOF predictions, using the corrupted train matches, to estimate the effect on validation.
- **Reads:** `runs/probs/*_oof.npy`, `runs/sift_trainq.csv`.
- **Used by:** EXP-023 (proxy row).

### `manipulasi_gambar.ipynb` (inference image manipulation study)

- **Purpose:** tests colour, contrast, autocrop, and resolution changes at inference (EXP-021), and the prior-balance step (EXP-022). It also contains the RegNet/EfficientNet comparison cells.
- **Run:** open in JupyterLab and run top to bottom. It needs the `cache/` arrays and the finished runs.
- **Used by:** EXP-021, EXP-022.

### Shell helpers: `screen_arch.sh`, `wave2.sh`

- **Purpose:** start several training runs in the background with `nohup`. `screen_arch.sh` runs the fold-0 architecture screen (EXP-017). `wave2.sh` starts the five-run batch from the 2026-11-12 session (EXP-018 to EXP-020, EXP-025).
- **Run:** `bash screen_arch.sh` or `bash wave2.sh`.
- **Check progress:** `tail -n 3 runs/<tag>_f0/log.txt`, or `nvidia-smi` for GPU memory. Stop a run with `pkill -f -- "--tag <tag> "`.

### `train.before-img.py` (backup)

- A copy of `train.py` from before the `--img` and `corrupt_v5` options were added. Same defaults and outputs. Kept so older runs can be reproduced exactly.

## How the scripts map to experiments

| Experiment(s) | Main script(s) |
| --- | --- |
| EXP-004, EXP-005, EXP-006 | notebook `dateraquest.ipynb` (phash folds) |
| EXP-007 to EXP-014 | `train.py` (`--recipe`, `--arch`, `--attn`, `--pool`) |
| EXP-015 | `ens_eval.py` |
| EXP-016, EXP-017 | `train.py`, `screen_arch.sh` |
| EXP-018 to EXP-020, EXP-025 | `train.py`, `wave2.sh`, `evalres.py probs` |
| EXP-021, EXP-022 | `manipulasi_gambar.ipynb`, `ens320.py` |
| EXP-023 | `sift_match.py`, `sift_oof_check.py`, `evalres.py search --sift` |
| EXP-024 | `evalres.py search` |

## Reproducing a result

1. Confirm the file hash in the experiment report matches the copy you have.
2. Build the caches (`train.py prep`), then train the recipe from the experiment's `config` block.
3. Compare your `done.json` and `runs/<tag>_submission.csv` SHA-256 prefix with the report. Small differences in the last digit of a score can come from GPU non-determinism, so compare to 3 decimal places.
