# Sumatra Flora Classification

**DATERAQUEST 2.0, Machine Learning Competition** (HMSD Adyatama, ITERA)

Classify 30 plant species from images, with CNNs trained **from scratch** (no pretrained weights). Metric: **macro F1**. This repo holds our team's code, experiment records, and results.

> **Public repository.** Anyone can read what is pushed here, including other teams. The competition rules forbid sharing code or predictions across teams. Read [docs/COMPETITION_RULES.md](docs/COMPETITION_RULES.md) before adding anything.

## Results at a glance

Public leaderboard (30% of test) for each submission we made. Each step is one change on top of the previous one.

| Step | Public LB | What changed | Experiments |
| --- | ---: | --- | --- |
| 1 | 0.519 | Single ResNet18 fold, corrupt-v4 augmentation | [EXP-004](experiments/EXP-004.md) |
| 2 | 0.677 | 5 folds, MixUp/CutMix, EMA | [EXP-005](experiments/EXP-005.md) |
| 3 | 0.739 | Rotation augmentation (test is rotated 0–360°), CBAM ensemble | [EXP-008](experiments/EXP-008.md), [EXP-010](experiments/EXP-010.md) |
| 4 | 0.80694 | Ensemble of ResNet18d and ResNet18 (600 epochs) | [EXP-015](experiments/EXP-015.md) |
| 5 | 0.82164 | Inference at 320 px, class-prior balancing | [EXP-022](experiments/EXP-022.md) |
| 6 | 0.83262 | SIFT label transfer from train (44 labels) | [EXP-023](experiments/EXP-023.md) |
| **7** | **0.87012** | ResNet18d trained at 288 px, ResNet34d, inference 352/320, balance, SIFT (29 labels) | [EXP-024](experiments/EXP-024.md) |

Best single model on 5-fold out-of-fold (OOF) validation: **0.8449** (ResNet18d, 600 epochs, trained at 288 px, evaluated at 352 px; [EXP-020](experiments/EXP-020.md)).

The full, always-current table is in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

### How to read the numbers

- **OOF** is validation on held-out folds, grouped by scene so near-duplicate photos never cross folds. It is the number we tune on.
- **Public LB** is the only independent check. Private LB is not known until the end.
- OOF runs higher than public LB, and recent gains did not all carry over (for example +0.043 OOF from resolution and balance gave +0.015 public). Treat OOF gains as directional.
- Ensemble members and inference sizes were chosen on the same OOF they are reported with, so those OOF values are slightly optimistic.
- Scores from different validation protocols must not be compared. See [docs/protocols.json](docs/protocols.json).

## What the code does

The main pipeline, in [code/train.py](code/train.py):

1. **Prepare data.** Build validation and test caches once (`prep`). Validation images get the same corruption as the test set (synthetic colour shifts, shadows, JPEG damage, and now rotation).
2. **Train per fold.** Each fold trains one model from random weights, with MixUp/CutMix, EMA, and a cosine schedule. Recipes `E`, `F`, `H`, `R`, `L` set the augmentation; `--arch`, `--img`, `--cmode`, `--attn`, `--pool` change the model and input.
3. **Ensemble.** Average the five fold models' test probabilities into one submission.

On top of that:
- [code/evalres.py](code/evalres.py) evaluates each model at several inference sizes, picks the best ensemble on OOF, applies class-prior balancing, and optionally replaces test labels with SIFT matches from train.
- [code/sift_match.py](code/sift_match.py) matches test and corrupted-train images to train with SIFT and RANSAC, and reports how precise each inlier threshold is.
- [notebooks/manipulasi_gambar.ipynb](notebooks/manipulasi_gambar.ipynb) tests colour, contrast, border, and resolution changes at inference time.

Full description of every script (inputs, outputs, commands, and which experiments used it) is in [docs/CODE_MAP.md](docs/CODE_MAP.md).

## What we learned

**Helped:**
- **Rotation augmentation** (±180°) and rotation in validation. Test images are rotated uniformly over the full circle, and models without it scored much lower.
- **Heavy synthetic corruption** (corrupt-v4). Light augmentation dropped OOF to 0.40 ([EXP-007](experiments/EXP-007.md)).
- **Longer training** (600 vs 300 epochs): ResNet18 0.672 → 0.751 OOF ([EXP-014](experiments/EXP-014.md)).
- **ResNet18d** (deep stem) beat ResNet18, ResNet34, and ResNet50 at 300 epochs ([EXP-013](experiments/EXP-013.md)).
- **Training at 288 px** plus inference at 320–352 px: the best single model ([EXP-020](experiments/EXP-020.md)).
- **Ensembling** two different models: about +0.02 OOF each time ([EXP-010](experiments/EXP-010.md), [EXP-015](experiments/EXP-015.md)).
- **SIFT label transfer** from train to test: +0.011 public for 44 labels ([EXP-023](experiments/EXP-023.md)).

**Did not help:**
- Colour and contrast normalisation at inference (gray-world, CLAHE, stretch) cost 0.05–0.13 OOF ([EXP-021](experiments/EXP-021.md)).
- CBAM attention and GeM pooling: below plain ResNet18 ([EXP-009](experiments/EXP-009.md)).
- RegNet and EfficientNet from scratch: below ResNet18d on the same folds ([EXP-016](experiments/EXP-016.md), [EXP-017](experiments/EXP-017.md)).
- Harder corruption on validation (blur, noise, downscale): worse on fold 0 ([EXP-006](experiments/EXP-006.md)).

**Still open:** two 300-epoch runs (training at 288 px, and a blur/grayscale augmentation) are paused with saved checkpoints ([EXP-025](experiments/EXP-025.md)).

## Repository layout

~~~text
code/                 Training, evaluation, and matching scripts (see docs/CODE_MAP.md)
notebooks/            Exploration and inference-manipulation notebooks
experiments/          One Markdown report + JSON record per experiment (EXP-004 onward)
experiments/runs/     Per-fold configs, metrics, and training history
experiments/logs/     Training logs
experiments/sift/     SIFT precision table for train queries
docs/                 Experiment registry, protocols, code map, rules, submission guide
scripts/registry.py   Validate and render the experiment registry
tests/                Unit tests (synthetic data only)
~~~

Not in the repository, by design: competition data, model weights, cached image arrays, prediction CSVs, and probability arrays.

## Experiment registry

Each experiment has `experiments/EXP-###.json` (structured record) and `EXP-###.md` (report). The table in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) is generated from the JSON files:

~~~bash
python scripts/registry.py check    # validate all experiment records
python scripts/registry.py render   # rebuild the table
~~~

## Running the code

The scripts were run on a single A100 GPU. They look for the official data in `code/dateraquest-2-0-machine-learning-competition/`, and write `cache/` and `runs/` next to the scripts, so run everything from `code/` after placing the data there.

~~~bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # install a PyTorch build that matches your CUDA
cd code                                  # data must be at code/dateraquest-2-0-machine-learning-competition/
python train.py prep                     # once: folds and image caches
python train.py run --recipe R --arch resnet18d --img 288 --epochs 600 --tta flip --jobs 3 --tag NAME
python evalres.py probs NAME --sizes 288 320 352
python evalres.py search NAME --tag FINAL
~~~

`python -m unittest discover -s tests` runs the registry and submission tests without any data.

## Submission format

A valid `submission.csv` has a header plus **2,010 rows**, exactly two columns `id,label`, integer labels `0`–`29`, and the same ID order as `sample_submission.csv`. Check it with `scripts/validate_submission.py` before uploading. Organizer rules are in [docs/SUBMISSION_GUIDE.md](docs/SUBMISSION_GUIDE.md).

## Collaboration

- Work on your own branch, record each experiment with a unique ID, and open a pull request for review.
- Do not commit competition data, model weights, prediction CSVs, or API keys.
- Team workflow and sync instructions: [docs/TEAM_WORKFLOW.md](docs/TEAM_WORKFLOW.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
