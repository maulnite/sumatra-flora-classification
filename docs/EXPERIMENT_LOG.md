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

Public probe 15 Nov 07:28 UTC: the user explicitly requested a submission.
[LOCAL-PROP-006-A-PUBLIC](../experiments/LOCAL-PROP-006-A-PUBLIC.md) replaces only
EXP-024's R18d fold-2 member with final CE control A (total weight 0.1), retaining
the other nine checkpoints, prior100 and SIFT5. A validation replay and original
baseline CSV reproduce exactly; candidate changes 56/2,010 test labels. Kaggle
**ref 58374703, COMPLETE, public 0.86464**, delta **−0.00548** from best **0.87012**.
The single-fold local gain did not transfer to this public probe. No full candidate
OOF exists; no further submission or additional folds were run. EXP-024 remains
the strongest observed public result. [Receipt and provenance](../experiments/results/LOCAL-PROP-006-A-PUBLIC.summary.json).

Backbone exploration 15 Nov 07:39 UTC:
[LOCAL-PROP-007-BACKBONES](../experiments/LOCAL-PROP-007-BACKBONES.md) started detached
scratch **EfficientNetV2 Tiny (`efficientnetv2_rw_t`, 12.66M)** and **ConvNeXt Tiny
(27.84M)** on strict fold 2. Same CE-A source draws/paired views, 288px, 600 epochs
/ 24,000 updates each; architecture-specific LR/warmup explicitly recorded.
Both see each identical batch in turn. Frozen 352px evaluation views and R34d
fold-2 predictions are reused. Eight relevant tests and full 120-view GPU update
/ checkpoint roundtrip passed; peak smoke allocation 12.17 GiB. Supervisor PID
11058, heartbeat 60 seconds, same-batch OOM resume. **Training in progress; no
candidate validation result or new Kaggle submission.** Final-epoch gate compares
the team pair (0.826344) and CE A (0.834465), with clean/blur guards. One fold only.

Final backbone result 15 Nov 13:58 UTC: **LOCAL-PROP-007-BACKBONES completed**,
600 epochs / 24,000 updates each, wall time **6h18m43s**, exit 0, no OOM.
Fold-2 balanced primary pair: team **0.826344**, matched ResNet18d CE A **0.834465**,
EfficientNetV2 Tiny **0.805343**, ConvNeXt Tiny **0.796876**. Candidate deltas versus
CE A are **−0.029121 / −0.037588**; both fail primary and clean gates. Mean-blur
improves to **0.662033 / 0.650868**, versus A 0.627330. Final model EMA/checkpoint,
optimizer/scheduler steps, probabilities, recomputed metrics/gates and all pinned
input/code/config/environment hashes verified. **No additional folds or Kaggle
submission**; EXP-024 remains best public **0.87012**. This evaluates the two frozen
recipes on one development fold, not universal architecture superiority.
[Final summary](../experiments/results/LOCAL-PROP-007-BACKBONES.summary.json).

Transformer architecture research 15 Nov:
[RESEARCH-008](TRANSFORMER_ARCHITECTURE_RESEARCH_2026-11-15.md) inspected six primary
papers and exact data/evaluation protocols. Closest evidence is **Training Vision
Transformers with Only 2040 Images**: Flowers102, source-only random initialization,
PVTv2-B3 accuracy 90.5% with supervised training, 95.9% with target-only IDMM plus
supervised training at 224px, 96.7% after an extra 448px stage. These are paper
accuracies, not DATERAQUEST macro F1. Recommendation: **PVTv2-B3 (44.74M)** with
**B0 (3.42M)** as a size comparison; both CPU factories initialize scratch 30-class
models in the installed environment. CCT 97.19% used a different **6,552-image train
split**, per the author's clarification; SCOTT 97.15% used **8,189 unlabeled images
including test**, versus 80.69% with 2,040 source images. Evidence updates the earlier
MobileViT/Swin hypothesis. **Research only; no transformer training or upload.**
[Structured source and decision record](../experiments/results/RESEARCH-008-TRANSFORMER.summary.json).

Transformer experiment 15 Nov 16:19 UTC:
[LOCAL-PROP-008-PVT-IDMM](../experiments/LOCAL-PROP-008-PVT-IDMM.md) launched
detached, supervisor PID **206091**. Source-only scratch **PVTv2-B3 (44.74M)**
runs first: 800-epoch IDMM at 112px with six crops, followed by paired
800-epoch supervised control / IDMM-initialized candidate at 224px. **B0 (3.42M)**
follows with the same stage lengths. Strict team fold 2, 2,411 source / 603
validation, effective photograph batch 256, ten updates/epoch. Only own source
photos initialize the transformers. Matched supervised batches/head/dropout
seeds distinguish the IDMM addition; candidate uses extra total compute.
The official IDMM crop grouping and losses were inspected and independently
implemented, with flora augmentations and other declared adaptations.
Ten behavioral tests and real-source GPU updates/checkpoint/inference smoke
passed; peak allocation **5.31 GiB**. Estimated synthetic GPU compute **39.88h**
excludes loading/evaluation/checkpoint overhead and shared-GPU variation.
Heartbeat every 60 seconds, checkpoints every ten complete epochs, same-batch
OOM resume. **Training in progress, no candidate validation or Kaggle upload.**
[Frozen protocol and launch receipt](../experiments/results/LOCAL-PROP-008-PVT-IDMM.launch.json).

Verified B3 result and public probe, 16 Nov:
[LOCAL-PROP-008 B3](../experiments/LOCAL-PROP-008-PVT-IDMM.md) completed 800 IDMM
epochs and 800 supervised epochs per arm at **21:00:36 UTC**, 8,000 updates per
stage/arm, no OOM/restart. Final matching fold-2 pair: PVT control **0.811614**,
IDMM candidate **0.865532**, stronger ResNet CE A **0.834465**. IDMM standalone
**0.862227**, clean pair **0.890449**, mean-blur pair **0.727417**. All frozen
gates pass; candidate uses extra pretraining compute. Independent final metrics,
EMA exports, optimizer/scheduler counts and pinned inputs/code/environment verified.
Conditional scene-group bootstrap primary delta vs CE A: **[+0.013614, +0.049258]**.
[B3 summary](../experiments/results/LOCAL-PROP-008-B3.summary.json).

Under the user's prior conditional submission authorization,
[LOCAL-PROP-008-B3-PUBLIC](../experiments/LOCAL-PROP-008-B3-PUBLIC.md) replaced one
R18d fold-2 member of EXP-024 with B3; nine other members/postprocessing fixed.
Original best CSV and candidate validation replay reproduce exactly; 85/2,010
test labels change. Kaggle **ref 58446643, COMPLETE, public 0.87197**, submitted
**22:37:45 UTC**, **+0.00185** versus previous best **0.87012**. **New highest
observed team public score**; private is unavailable, full replacement OOF does
not exist, and no public weight search followed. B0 is still training.
[Public receipt and provenance](../experiments/results/LOCAL-PROP-008-B3-PUBLIC.summary.json).

Team update 17 Nov 08:50 UTC:
[Nafis review](TEAM_UPDATE_2026-11-17.md) verified **ref 58453652,
`sub_mine5_team_mid.csv`, COMPLETE, public 0.88314**: **+0.01117** versus PVT
probe 0.87197 and **+0.01302** versus EXP-024. Description: five new scratch
ResNet18d/34d models, 288–384px, full train plus sibling-matched test images,
intermediate EMA, EXP-024 probabilities, geometric mean, prior balancing and
calibrated test→train SIFT with no test→test links. Separate SIFT sibling graph
probe **ref 58450478**, test→train/test→test ≥6 inliers, scored **0.84096**.
Code/config, blend weights, label handling and local metrics are not available.
`NafisNaufal` is a collaborator, but no published Nafis branch/PR was visible;
`kaka` and `main` are unchanged. **New best public baseline: 0.88314**; no new
submission or training was launched by this audit. Existing B0 training continues.
[Snapshot](../experiments/results/TEAM-UPDATE-2026-11-17.summary.json).

Deadline follow-up 17 Nov 11:21 UTC:
[Server search](DEADLINE_OPTIONS_2026-11-17.md) found Nafis's work under
`/home/nafisnaufal1426/akucintakc/work/`. Five mid-training arrays plus EXP-024
as six equal geometric inputs reproduce the saved ensemble exactly; prior100
and frozen calibrated evidence reproduce the associated .88314 local CSV
byte-for-byte. Training and intermediate checkpoint inference were not rerun.
Prepared one unsubmitted **90% Nafis / 10% PVT geometric blend**, 56 changed
labels and no changes on strong-evidence rows; no independent blend score exists.
Kaggle snapshot 11:22 UTC: two submissions today, nominally one shared slot left.

LOCAL-PROP-008 fully completed **09:41:17 UTC**, 41h21m30s, exit 0, no OOM.
B0 IDMM final matching-pair **.841597**, standalone **.815623**, below B3
**.865532 / .862227**. B0 EMA/checkpoint/steps/metrics/input hashes verified;
small exploratory B0/B3 blends did not improve B3's primary. No B0 upload.
[B0 result](../experiments/results/LOCAL-PROP-008-B0.summary.json).

[LOCAL-PROP-009-PVT448](../experiments/LOCAL-PROP-009-PVT448.md) started detached
**11:21:26 UTC**, supervisor **916122**: own B3 checkpoint, 100 source-only
epochs at 448px, 1,000 updates, same fold2/352px validation. Three tests and
full 256-photo GPU smoke passed, initial validation replay exact, peak 8.18 GiB.
Benchmark projects 2.53 hours with 30% overhead. Final-epoch gate requires +.003
primary vs B3, clean/blur drops at most .01. Epoch-boundary cutoff 22:30 WIB;
no automated upload. **Training in progress, no new score or submission.**
[Launch receipt](../experiments/results/LOCAL-PROP-009-PVT448.launch.json).

## Ablation / comparison placeholder

| Factor | Run A | Run B | Matched validation split? | Conclusion |
| --- | --- | --- | --- | --- |
| Backbone | TBD | TBD | TBD | TBD |
| Image resolution | TBD | TBD | TBD | TBD |
| Augmentations | TBD | TBD | TBD | TBD |
| Loss/regularization | TBD | TBD | TBD | TBD |
| Macro F1 | — | — | TBD | TBD |

Always distinguish validation results from public/private leaderboard scores.
