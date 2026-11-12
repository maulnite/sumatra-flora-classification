# Experiment Registry

Central, shared index of every experiment on every branch. Humans read the table below; scripts read `experiments/EXP-###.json`.

## How it works

- **One experiment = one JSON file + one Markdown report** in `experiments/` (for example `EXP-024.json` and `EXP-024.md`). The JSON is the machine-readable record; the Markdown explains it.
- **Validation and leaderboard numbers are separate fields.** Never compare across different `protocol` values; definitions live in `docs/protocols.json`.
- **The table below is generated.** Run `python scripts/registry.py render` (add `--all-branches` to include experiments from every `origin/*` branch). Do not edit the generated block by hand.
- **Check before committing:** `python scripts/registry.py check`.
- **IDs:** `EXP-004` onward holds the backfilled history of branch `kaka`. `EXP-001`–`EXP-003` are the original team placeholders, still unclaimed. Before taking a new ID, `git fetch` and look for it on other branches; `render --all-branches` fails when two branches reuse an ID.

## Generated registry

<!-- BEGIN GENERATED REGISTRY -->

| ID | Status | Owner / branch | Protocol | Best OOF macro F1 | Public LB | Note |
| --- | --- | --- | --- | ---: | ---: | --- |
| EXP-004 | completed | kaka / `kaka` | `phash5_v4` | 0.6300 | 0.51900 | ResNet18 224, corrupt_v4, 150 ep, phash folds |
| EXP-005 | completed | kaka / `kaka` | `phash5_v4` | — | 0.67700 | E + MixUp/CutMix 0.5 + EMA, 300 ep, 5-fold |
| EXP-006 | partial | kaka / `kaka` | `phash5_v4` | — | — | Harder corruption, fold 0 only: worse |
| EXP-007 | completed | kaka / `kaka` | `scene5_v4rot` | 0.3990 | — | Light augmentation (no corrupt_v4): much worse |
| EXP-008 | completed | kaka / `kaka` | `scene5_v4rot` | 0.6723 | — | Rotation aug + scene folds + v4+rot validation |
| EXP-009 | completed | kaka / `kaka` | `scene5_v4rot` | 0.6377 | — | R + CBAM + GeM: below plain R |
| EXP-010 | completed | kaka / `kaka` | `scene5_v4rot` | 0.6900 | 0.73900 | Mean of R and R_cbam_gem |
| EXP-011 | completed | kaka / `kaka` | `scene5_v4rot` | 0.6872 | — | ResNet34 300 ep: below ResNet18d |
| EXP-012 | completed | kaka / `kaka` | `scene5_v4rot` | 0.6960 | — | ResNet50 300 ep: below ResNet18d |
| EXP-013 | completed | kaka / `kaka` | `scene5_v4rot` | 0.7524 | — | ResNet18d 300 ep: best 300-epoch model |
| EXP-014 | completed | kaka / `kaka` | `scene5_v4rot` | 0.7509 | — | ResNet18 600 ep (vs 300) |
| EXP-015 | completed | kaka / `kaka` | `scene5_v4rot` | 0.7727 | 0.80694 | ResNet18d + ResNet18 600ep; first 0.80 LB |
| EXP-016 | completed | kaka / `kaka` | `scene5_v4rot` | 0.7218 | — | RegNetY-008 5-fold: weaker than ResNet18d |
| EXP-017 | completed | kaka / `kaka` | `scene_fold0_v4rot` | — | — | Fold-0 screen: RegNet/EffNet below ResNet18d |
| EXP-018 | completed | kaka / `kaka` | `scene5_v4rot` | 0.8361 | — | ResNet34d 600 ep, 224 train; OOF up to 0.836 at inference 320 |
| EXP-019 | completed | kaka / `kaka` | `scene5_v4rot` | 0.8308 | — | ResNet18d 600 ep, 224 train; OOF 0.83 at inference 288 |
| EXP-020 | completed | kaka / `kaka` | `scene5_v4rot` | 0.8449 | — | Train 288 px, 600 ep: best single model (OOF 0.8449 at inference 352) |
| EXP-021 | completed | kaka / `kaka` | `scene5_v4rot` | 0.8100 | — | Colour fixes hurt; inference resolution 256-352 helps |
| EXP-022 | completed | kaka / `kaka` | `scene5_v4rot` | 0.8155 | 0.82164 | Prior balance + res 320 -> LB 0.82164 |
| EXP-023 | completed | kaka / `kaka` | `scene5_v4rot` | — | 0.83262 | SIFT inlier>=5 replaces 44 labels -> LB 0.83262 |
| EXP-024 | completed | kaka / `kaka` | `scene5_v4rot` | 0.8526 | 0.87012 | Best public LB so far: 0.87012 |
| EXP-025 | paused | kaka / `kaka` | `scene5_v4rot` | — | — | Paused for shared-GPU etiquette; checkpoints saved |

### Top 5 by OOF (same protocol only compare within a protocol)

| ID | Protocol | OOF macro F1 | Public LB |
| --- | --- | ---: | ---: |
| EXP-024 | `scene5_v4rot` | 0.8526 | 0.87012 |
| EXP-020 | `scene5_v4rot` | 0.8449 | — |
| EXP-018 | `scene5_v4rot` | 0.8361 | — |
| EXP-019 | `scene5_v4rot` | 0.8308 | — |
| EXP-022 | `scene5_v4rot` | 0.8155 | 0.82164 |

### Protocols

| Protocol | Meaning |
| --- | --- |
| `phash5_v4` | 5-fold OOF; folds = near-duplicate groups (pHash) + StratifiedGroupKFold; validation = v4 corruption, no rotation; evaluated at 224 px. Scores are NOT comparable with scene5_v4rot. |
| `scene5_v4rot` | 5-fold OOF; folds = 237 scene groups (scene_folds, fixed file); validation = corrupt_v4 + uniform random rotation in [-180, 180]; evaluated at the resolution listed per result. Use this protocol for all new comparisons. |
| `scene_fold0_v4rot` | Single held-out fold 0 of scene5 only. Quick screening; never compare against 5-fold OOF numbers directly. |

<!-- END GENERATED REGISTRY -->

## Original placeholders (not started)

| Experiment | Owner | Status |
| --- | --- | --- |
| [EXP-001](../experiments/EXP-001.md) | @TBD | Planned |
| [EXP-002](../experiments/EXP-002.md) | @TBD | Planned |
| [EXP-003](../experiments/EXP-003.md) | @TBD | Planned |

## Reading the numbers

- **Best OOF macro F1** is the 5-fold out-of-fold score under the stated protocol. `scene_fold0_v4rot` entries report one fold only.
- **Public LB** is the Kaggle public score (30% of test). Private is unknown until the competition ends.
- Ensemble and post-processing choices were selected on the same OOF they are reported with. Treat their OOF as **optimistic**; the public score is the only independent check.
