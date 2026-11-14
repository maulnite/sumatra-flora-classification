# Sumatra Flora Classification

**DATERAQUEST 2.0 — Machine Learning Competition**  
*Sumatra Flora Vision Challenge | HMSD Adyatama | Institut Teknologi Sumatera (ITERA)*

A collaborative computer vision project to classify **30 plant species** from Kebun Raya ITERA. This three-person team's focus is **training from scratch**, reliable augmentation, and robust evaluation.

> **Status: starter repository.** All experiment results, leaderboard numbers, and member handles are placeholders until verified.  
> **Public visibility:** All code published here can be viewed by anyone, including other competing teams. Sharing code **within our registered team** is allowed; cross-team code/prediction sharing is prohibited. Official rules take precedence.

Local onboarding audit (12 Nov 2026): official data and nine existing Kaggle submissions have been inspected. See [verified competition/data/submission snapshot](docs/COMPETITION_AUDIT_2026-11-12.md). These historical submissions are separate from the placeholder EXP reports below; their training code and local validation metrics are not yet in this repository.

The [deep dataset audit](docs/DEEP_DATASET_AUDIT_2026-11-12.md) records detected scene overlap, provisional group folds, image quality shifts beyond JPEG, and train-label review candidates.

Update 13 Nov 2026: the team's working pipeline and EXP-004–EXP-025 records are on branch `kaka`, with verified best public Macro F1 **0.87012**. See [the team update review](docs/TEAM_UPDATE_2026-11-13.md). [LOCAL-PROP-001 revision 2](experiments/LOCAL-PROP-001.md) completed recipe R / ResNet18d A/B training with added post-resize blur on a local holdout. Mean blur Macro F1 increased **0.3921 → 0.6567**, while primary F1 changed **0.7290 → 0.7235**; the automatic gate failed. The user subsequently requested submission B: **public 0.79702**, ref **58306830**, COMPLETE. Local artifacts are under `outputs/LOCAL-PROP-001/`.

## Competition overview

[LOCAL-PROP-002](experiments/LOCAL-PROP-002.md) uses the team's strongest EXP-024 pipeline as its frozen baseline; its original 0.87012 submission was reconstructed byte-for-byte from stored team probabilities. A fixed 10% addition of the blur model changed shared-holdout F1 **0.8527 → 0.8505**, so the original baseline was retained without another submission. Team artifacts are available at the server path documented in the report.

Update 14 Nov 2026: [checkpoint inference reproduction](experiments/BASELINE-INFERENCE-REPRO-001.md) executed all **10 best-team checkpoints**. All four OOF/test probability arrays and final CSV are **byte-identical** to the reference; maximum probability difference is **0**. Replaying preprocessing from all **5,024 raw images at two resolutions** also reproduces every cache pixel. Balanced neural OOF is **0.852564**; 21 tests pass. Training and SIFT matching were not rerun; final postprocessing uses the original SIFT match table. No new Kaggle submission.

Next research priority: [LOCAL-PROP-004](experiments/LOCAL-PROP-004.md), domain-adaptive self-training on official unlabeled test images, with adaptive pseudo-label confidence and separately evaluated retrieval/graph components. CPU diagnostics show 1,330/1,332 selected OOF predictions correct at backbone agreement plus confidence≥0.8, and 521 selected test images; test pseudo-label accuracy is unknown. Current top public is 0.99645. **Research proposal only; no new training or submission.** The earlier [30-epoch fine-tuning plan](experiments/LOCAL-PROP-003.md) was deferred by the user before execution.

| Item | Details |
| --- | --- |
| Organizer | HMSD Adyatama, ITERA |
| Task | Single-label image classification; 30 species (labels 0–29) |
| Dataset | Official organizer/Kaggle dataset only |
| Test set | 2,010 unlabeled JPG images |
| Evaluation | **Macro F1-Score** |
| Public/private leaderboard | 30% / 70% of test data |
| Models | Architectures of choice, **all weights randomly initialized** |
| Kaggle limit | **3 submissions per day** |
| Preliminary deadline | **17 November 2026, 23:59 WIB** |
| Finalist PPTX deadline | **26 November 2026, 23:59 WIB** |

### Main rules

- Do **not** use externally pretrained weights, transfer learning, external training datasets, manual test labeling, VLM (CLIP, SigLIP, BLIP, etc.), instant AutoML, or inference APIs.
- ResNet, EfficientNet, ConvNeXt, ViT and custom CNN designs are allowed, trained **from scratch** (`weights=None` or `pretrained=False`).
- Original data augmentation and hyperparameter tuning tools such as Optuna are allowed.
- Fix and document seeds, split, model config, hardware, training logs, and reproducible CSV outputs.
- Work only with your officially registered teammates, not other competing teams.

See [competition rule summary](docs/COMPETITION_RULES.md).

## Team members

| Member | GitHub account | Experiment responsibility |
| --- | --- | --- |
| Member 1 | @TBD | TBD |
| Member 2 | @TBD | TBD |
| Member 3 | @TBD | TBD |

Repository owner should invite both teammates via **Settings → Collaborators** (public means anyone can *read*, not *write*).

## Project structure

~~~text
.github/              GitHub Actions and PR/issue templates
configs/              Training config templates
data/                 Local official competition data (ignored)
docs/                 Rules, experiment registry, data and team notes
experiments/          One report per experiment / member
notebooks/            EDA, training and inference notebooks
outputs/              Local predictions and checkpoints (ignored)
reports/              Finalist presentation/report materials
scripts/              Dataset inspection and CSV validation
src/sumatra_flora/
  data/               Data loader, stratified split, transforms
  models/             Original from-scratch architectures
  training/           Training loops, losses, schedulers
  evaluation/         Macro F1 and per-class analysis
  inference/          Test predictions, ensemble methods
tests/                Synthetic utility tests
README.md             Project description and results
CONTRIBUTING.md       Three-member collaboration guide
~~~

## Getting started

~~~bash
git clone https://github.com/maulnite/sumatra-flora-classification.git
cd sumatra-flora-classification
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

Install GPU-specific PyTorch using the [official selector](https://pytorch.org/get-started/locally/) to match your CUDA and driver versions.

Download the **official Kaggle competition dataset through permitted access** into the local `data/raw/` folder (not provided or committed to GitHub):

~~~text
data/raw/
├── train/
│   ├── 0_Albizia_saponaria/
│   └── ... (30 class folders)
├── test/ (test_0001.jpg ... test_2010.jpg)
├── train.csv
├── class_mapping.csv
└── sample_submission.csv
~~~

Inspect the local download:

~~~bash
python scripts/inspect_dataset.py --root data/raw
~~~

Create a new experiment report:

~~~bash
python scripts/new_experiment.py --id EXP-004 --owner your-handle --title "ResNet scratch baseline"
~~~

Validate a model-generated CSV before submitting:

~~~bash
python scripts/validate_submission.py --file outputs/submissions/submission.csv --template data/raw/sample_submission.csv
~~~

## Experiment methodology

1. Inspect class distributions, resolution, duplicates and label mapping on the official training images.
2. Build a reproducible stratified train/validation split; guard against similar-image leakage, and record the seed and exact indices.
3. Try custom CNN/ResNet-style/EfficientNet-style/ConvNeXt-style/ViT models **without pretrained weights**.
4. Compare augmentation, input resolution, losses, schedulers and regularization on the **same validation split**.
5. Use **Macro F1** and per-class F1/confusion matrix; avoid choosing models solely on the public leaderboard.
6. Log code commit, config, hardware, libraries, seed, runtime, errors, artifacts and metric values.

Macro F1 averages the F1 of each of the 30 classes with equal weight:

~~~python
from sklearn.metrics import f1_score
score = f1_score(y_true, y_pred, labels=list(range(30)), average="macro", zero_division=0)
~~~

## Team experiment results

**No experiments are completed yet.** Each member updates their own [experiment report](experiments/) and the verified summary below through pull requests.

| Experiment | Owner | Architecture / change | Seed / split | Validation Macro F1 | Public LB | Private LB | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| [EXP-001](experiments/EXP-001.md) | @TBD | TBD | TBD | — | — | — | Planned |
| [EXP-002](experiments/EXP-002.md) | @TBD | TBD | TBD | — | — | — | Planned |
| [EXP-003](experiments/EXP-003.md) | @TBD | TBD | TBD | — | — | — | Planned |

### Best validated result (placeholder)

| Metadata | Value |
| --- | --- |
| Experiment ID / member | TBD |
| Architecture and parameters | TBD |
| Data split, seed, random initialization | TBD |
| Model config / Git commit SHA | TBD |
| Validation Macro F1 | TBD |
| Public leaderboard Macro F1 | TBD |
| Private leaderboard Macro F1 | Not yet released |
| Training time / hardware | TBD |
| Reproducible notebook / inference command | TBD |
| Local checkpoint path | TBD |

For a fuller cross-experiment log, see [Experiment Registry](docs/EXPERIMENT_LOG.md).

## Submission

A valid `submission.csv` needs **2,011 total lines**: a header plus **2,010 test rows**, exactly 2 columns `id,label`, integer labels from `0` to `29`, and **the exact ID order in the official sample_submission.csv**. No blanks, duplicate IDs, or extra rows.

~~~csv
id,label
test_0001.jpg,0
test_0002.jpg,0
~~~

*The example illustrates CSV format only; it is not a real prediction.*

Before 17 November 2026 23:59 WIB, the team must submit a ZIP (`IDTim_NamaTim.zip`) containing executed `.ipynb` notebook(s) with training/validation results and the matching prediction `.csv` to the organizer portal. A reproducible execution must generate the submitted predictions. See [Submission Guide](docs/SUBMISSION_GUIDE.md).

## Collaboration and safety

- Work in `exp/<handle>/<topic>`, `feat/<handle>/<topic>`, `fix/<handle>/<topic>` or `docs/<handle>/<topic>` branches and request teammate review before merging to `main`.
- Give each experiment a unique ID and notebook filename to avoid merge conflicts.
- **Never commit** organizer images/CSVs, model checkpoints, submission CSVs, Kaggle API keys, secrets, or private artifacts; these are ignored under `.gitignore` where practical.
- This repo is public: publishing active solution code is visible beyond your team. Follow organizer integrity rules and coordinate what to make public.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [Team Workflow](docs/TEAM_WORKFLOW.md).
