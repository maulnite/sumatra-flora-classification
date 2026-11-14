# Experiment Registry

One completed run = one `experiments/EXP-###.md` report. Only enter verified metrics.

Nine historical Kaggle submissions were verified on 12 Nov 2026 (best public Macro F1: **0.84120**). Their exact names, descriptions, scores, and interpretation are recorded in [the onboarding audit](COMPETITION_AUDIT_2026-11-12.md). Authors, training code, configs, and local validation metrics were not exposed by the submission listing, so these runs have not been assigned to the placeholder EXP IDs below.

| Experiment | Owner | Change | Config/split/seed | Validation Macro F1 | Public LB | Status |
| --- | --- | --- | --- | ---: | ---: | --- |
| [EXP-001](../experiments/EXP-001.md) | @TBD | TBD | TBD | — | — | Planned |
| [EXP-002](../experiments/EXP-002.md) | @TBD | TBD | TBD | — | — | Planned |
| [EXP-003](../experiments/EXP-003.md) | @TBD | TBD | TBD | — | — | Planned |

Update 13 Nov 2026: branch `kaka` contains published experiment IDs EXP-004 through EXP-025. The earlier local EXP-004 draft is now [LOCAL-PROP-001 revision 2](../experiments/LOCAL-PROP-001.md), with no team experiment ID assigned. It tests recipe R / ResNet18d control vs mild post-resize blur on a shared local holdout (2,378 train / 634 validation; 2 scene neighbors purged). Both arms completed 300 epochs: primary Macro F1 **0.728991 → 0.723461**, clean **0.844319 → 0.845658**, mean additional-blur Macro F1 **0.392141 → 0.656741**. Automatic gate failed; the user subsequently explicitly requested submission B: **ref 58306830, COMPLETE, public 0.79702**, private unavailable. Its protocol is distinct from team scene237 OOF. See [the reviewed team update](TEAM_UPDATE_2026-11-13.md) for upstream results. The placeholder table above is the original `main` scaffold, not the current registry on `kaka`.

Update lanjutan 13 Nov: [LOCAL-PROP-002](../experiments/LOCAL-PROP-002.md) memakai **EXP-024 best public 0.87012** sebagai baseline. Artefak ditemukan di folder server yang didokumentasikan tim; final CSV berhasil direproduksi byte-for-byte dan full balanced neural OOF **0.852564** direproduksi. Pada 603 query yang tidak terlihat oleh B dan memakai frozen views tim, kontrol **0.852673**, blend tetap 90% best +10% B **0.850512** (delta −0.002160). **Baseline asli dipertahankan, tidak ada submission baru.** Ini screening subset, bukan full OOF kandidat. Tidak ada training ulang; 19 tests lulus.

Update 14 Nov: [BASELINE-INFERENCE-REPRO-001](../experiments/BASELINE-INFERENCE-REPRO-001.md) benar-benar menjalankan ulang inference **10 checkpoint best**. Keempat file OOF/test memiliki SHA-256 identik dengan referensi, selisih probabilitas maksimum **0**, balanced neural OOF **0.852564**, dan final CSV identik byte-for-byte. **5.024 raw images × 2 resolusi** menghasilkan cache pixel-identical. Runtime 140,66 detik; 21 tests lulus. Training dan pencarian match SIFT tidak diulang; final menggunakan tabel SIFT referensi. Tidak ada submission baru. Ini melengkapi checkpoint replay yang belum dilakukan pada LOCAL-PROP-002.

Planned 14 Nov: [LOCAL-PROP-003](../experiments/LOCAL-PROP-003.md) menguji fine-tuning akhir ResNet18d terbaik selama 30 epoch, dengan kontrol MixUp/CutMix p0,5 versus kandidat p0. Kedua arm mulai dari bobot EMA epoch600 identik dengan optimizer baru; corruption v4 dan keluarga ResNet34d tetap. Screening fold0 (baseline balanced pada fold tersebut **0,871464**), cek konsistensi fold1, lalu fold2–4 hanya jika lolos. Gate promosi full OOF minimal +0,002 terhadap baseline dan kontrol, serta clean/blur guards. **Belum training atau submission.**

User revision 14 Nov: LOCAL-PROP-003 deferred before training. [LOCAL-PROP-004](../experiments/LOCAL-PROP-004.md) researches domain-adaptive self-training, adaptive confidence, and retrieval/graph inference. Fresh leaderboard top public **0,99645**. Baseline CPU diagnostics: OOF agreement+confidence≥0,8 gives **1.330/1.332 correct**, with **521 test candidates** of unknown accuracy; confidence≥0,95 selects just one test image. Both backbone top1 predictions are wrong on 399 OOF queries. **No candidate training, implemented method, or new submission.** Primary strict OOF and secondary transductive holdout are explicitly separated; paper links and diagnostic JSON are included.

## Ablation / comparison placeholder

| Factor | Run A | Run B | Matched validation split? | Conclusion |
| --- | --- | --- | --- | --- |
| Backbone | TBD | TBD | TBD | TBD |
| Image resolution | TBD | TBD | TBD | TBD |
| Augmentations | TBD | TBD | TBD | TBD |
| Loss/regularization | TBD | TBD | TBD | TBD |
| Macro F1 | — | — | TBD | TBD |

Always distinguish validation results from public/private leaderboard scores.
