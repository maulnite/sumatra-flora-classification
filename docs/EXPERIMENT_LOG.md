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

Planned 14 Nov: [LOCAL-PROP-003](../experiments/LOCAL-PROP-003.md) menguji fine-tuning akhir ResNet18d terbaik selama 30 epoch, dengan kontrol MixUp/CutMix p0,5 versus kandidat p0. Kedua arm mulai dari bobot EMA epoch600 identik dengan optimizer baru; corruption v4 dan keluarga ResNet34d tetap. Screening fold 0 (baseline balanced pada fold tersebut **0,871464**), cek konsistensi fold 1, lalu fold2–4 hanya jika lolos. Gate promosi full OOF minimal +0,002 terhadap baseline dan kontrol, serta clean/blur guards. **Belum training atau submission.**

User revision 14 Nov: LOCAL-PROP-003 deferred before training. [LOCAL-PROP-004](../experiments/LOCAL-PROP-004.md) researches domain-adaptive self-training, adaptive confidence, and retrieval/graph inference. Fresh leaderboard top public **0,99645**. Baseline CPU diagnostics: OOF agreement+confidence≥0,8 gives **1.330/1.332 correct**, with **521 test candidates** of unknown accuracy; confidence≥0,95 selects just one test image. Both backbone top1 predictions are wrong on 399 OOF queries. **No candidate training, implemented method, or new submission.** Primary strict OOF and secondary transductive holdout are explicitly separated; paper links and diagnostic JSON are included.

Eksekusi 14 Nov 08:59 UTC: [LOCAL-PROP-004-PILOT](../experiments/LOCAL-PROP-004-PILOT.md) menjalankan ekstraksi embedding empat checkpoint serta retrieval/graph pada fold 0/1 sebagai detached background job. Primary query readout dan secondary joint transductive graph dipisahkan. Config/gate dibekukan;24 tests lulus. Monitor otomatis mencatat heartbeat setiap60 detik. Tidak ada backbone training, SIFT/ASIFT baru, atau Kaggle upload; hasil akhir disimpan otomatis di output job.

Hasil pilot 14 Nov 09:04 UTC: pooled 1.221-query balanced F1 baseline **0,867045**, primary graph **0,853589** (delta **−0,013457**, conditional group bootstrap 95%[−0,024140;−0,003413]). Kedua fold turun; 13 error diperbaiki dan 30 prediksi benar rusak. Konfigurasi graph tidak dipromosikan. Runtime 278,64 detik, exit 0. Tidak ada backbone training/submission; AdaMatch/self-training belum dimulai. [Ringkasan hasil](../experiments/results/LOCAL-PROP-004-PILOT.summary.json).

Training 14 Nov 09:34 UTC: [LOCAL-PROP-004-SSL](../experiments/LOCAL-PROP-004-SSL.md)
menjalankan **adaptive self-training tanpa graph**, A/B masing-masing 100 epoch
pada strict scene fold 0. Kontrol A berjalan lebih dahulu; kandidat B memakai
dynamic EMA pseudo-label, clipped distribution alignment, relative confidence,
dan weak/strong target consistency pada official test tanpa label. Baseline lokal
awal **0,871464**, 30 tests dan satu full GPU smoke update lulus (peak 2,13 GiB).
Detached supervisor aktif, checkpoint setiap epoch, heartbeat setiap 60 detik,
OOM resume tanpa perubahan batch size. Pada 09:35:52 UTC A telah menjalankan
9 optimizer updates. Belum ada hasil kandidat atau submission; gate final-epoch
mensyaratkan minimal +0,01 terhadap initial dan matched control serta clean/blur guards.

Hasil 14 Nov 15:14 UTC: **LOCAL-PROP-004-SSL selesai**, masing-masing 100 epoch
dan 3.700 updates, runtime total 5 jam 40 menit, exit 0, tanpa OOM. Primary
balanced F1 fold 0: initial **0,871464**, kontrol A **0,876597**, adaptive B
**0,874712**. B naik +0,003248 dari initial tetapi turun −0,001885 dari kontrol;
kedua gate primary +0,01 gagal. Clean/blur guards lolos. Conditional paired-group
95% CI B−A [−0,009917; +0,005350] mencakup nol. Tidak dipromosikan ke fold lain
atau submission. Saved probabilities dan final EMA/checkpoint diverifikasi;
seluruh input/code/config hashes tetap. [Ringkasan hasil](../experiments/results/LOCAL-PROP-004-SSL.summary.json).

Research 14 Nov 19:50 UTC: [LOCAL-PROP-005-DIAG](METHOD_RESEARCH_2026-11-14.md)
screened per-image RootSIFT/homography after fold-safe neural retrieval on 607
fold-1 queries. Twenty accepted matches were all correct but changed no baseline
labels: **0,862873 → 0,862873**. Runtime 38,67 seconds; 34 tests passed. This is
a conservative 448px/240-keypoint gallery diagnostic, not full ASIFT/RIFT.
No training, test prediction, or submission. Main next research recommendation:
matched source-supervised CE versus **CE + supervised contrastive representation
loss** from scratch, with distinct-scene positive pairs. SupCon training has not
started; [research report and proposed protocol](METHOD_RESEARCH_2026-11-14.md).

Execution 14 Nov 20:05 UTC: [LOCAL-PROP-006-SUPCON](../experiments/LOCAL-PROP-006-SUPCON.md)
started detached **from scratch**, matched CE versus CE+SupCon, ResNet18d@288,
600 epochs / 24.000 updates per arm, strict fold 2 (2.411 source / 603 validation).
A/B process the same 120-view batch in turn from the first epoch. Source labels
only, distinct-scene class positives, no target images/pseudo-labels. Strong
team fold-2 baseline **0,826344**. All 38 tests and a paired full-batch GPU smoke
passed; peak smoke VRAM 3,30 GiB. At 20:06:28 UTC both arms had completed 38 updates.
Candidate validation is pending; first evaluation at epoch 50. Paired epoch
checkpoints and a 60-second heartbeat support background monitoring/OOM resume.

Result 15 Nov 00:34 UTC: **LOCAL-PROP-006-SUPCON completed**, 600 epochs / 24.000
updates per arm, runtime 4h29m, exit 0, no OOM. Fold-2 primary balanced F1:
baseline **0,826344**, CE A **0,834465**, CE+SupCon B **0,827229**. B−baseline
+0,000884; B−A **−0,007236**; both primary gates fail. Mean-blur B−A −0,031532
also fails the guard. Conditional paired-group 95% CI B−A [−0,020446; +0,004280]
includes zero. A's +0,008120 versus baseline is a single-fold signal for its
alternative CE recipe, not a confirmed OOF/public improvement. Final probabilities,
paired EMA/checkpoint, 24.000-step optimizer/scheduler states and input hashes
verified. No more folds or submission. [Summary](../experiments/results/LOCAL-PROP-006-SUPCON.summary.json).

## Ablation / comparison placeholder

| Factor | Run A | Run B | Matched validation split? | Conclusion |
| --- | --- | --- | --- | --- |
| Backbone | TBD | TBD | TBD | TBD |
| Image resolution | TBD | TBD | TBD | TBD |
| Augmentations | TBD | TBD | TBD | TBD |
| Loss/regularization | TBD | TBD | TBD | TBD |
| Macro F1 | — | — | TBD | TBD |

Always distinguish validation results from public/private leaderboard scores.
