# Experiment Registry

One completed run = one `experiments/EXP-###.md` report. Only enter verified metrics.

Nine historical Kaggle submissions were verified on 12 Nov 2026 (best public Macro F1: **0.84120**). Their exact names, descriptions, scores, and interpretation are recorded in [the onboarding audit](COMPETITION_AUDIT_2026-11-12.md). Authors, training code, configs, and local validation metrics were not exposed by the submission listing, so these runs have not been assigned to the placeholder EXP IDs below.

| Experiment | Owner | Change | Config/split/seed | Validation Macro F1 | Public LB | Status |
| --- | --- | --- | --- | ---: | ---: | --- |
| [EXP-001](../experiments/EXP-001.md) | @TBD | TBD | TBD | — | — | Planned |
| [EXP-002](../experiments/EXP-002.md) | @TBD | TBD | TBD | — | — | Planned |
| [EXP-003](../experiments/EXP-003.md) | @TBD | TBD | TBD | — | — | Planned |

Update 13 Nov 2026: branch `kaka` contains published experiment IDs EXP-004 through EXP-025. The earlier local EXP-004 draft is now [LOCAL-PROP-001 revision 2](../experiments/LOCAL-PROP-001.md), with no team experiment ID assigned. It tests recipe R / ResNet18d control vs mild post-resize blur on a shared local holdout (2,378 train / 634 validation; 2 scene neighbors purged). Both arms completed 300 epochs: primary Macro F1 **0.728991 → 0.723461**, clean **0.844319 → 0.845658**, mean additional-blur Macro F1 **0.392141 → 0.656741**. Automatic gate failed; the user subsequently explicitly requested submission B: **ref 58306830, COMPLETE, public 0.79702**, private unavailable. Its protocol is distinct from team scene237 OOF. See [the reviewed team update](TEAM_UPDATE_2026-11-13.md) for upstream results. The placeholder table above is the original `main` scaffold, not the current registry on `kaka`.

## Ablation / comparison placeholder

| Factor | Run A | Run B | Matched validation split? | Conclusion |
| --- | --- | --- | --- | --- |
| Backbone | TBD | TBD | TBD | TBD |
| Image resolution | TBD | TBD | TBD | TBD |
| Augmentations | TBD | TBD | TBD | TBD |
| Loss/regularization | TBD | TBD | TBD | TBD |
| Macro F1 | — | — | TBD | TBD |

Always distinguish validation results from public/private leaderboard scores.
