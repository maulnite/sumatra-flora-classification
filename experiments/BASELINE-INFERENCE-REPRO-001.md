# BASELINE-INFERENCE-REPRO-001 — Replay checkpoint terbaik kaka

Status: **selesai; seluruh pemeriksaan lolos**, 14 November 2026 pukul 07:52 UTC. Ini verifikasi reproduksi inference EXP-024, tanpa training baru atau upload Kaggle. Sepuluh checkpoint benar-benar dimuat dan menjalankan forward pass; probabilitas referensi hanya dibaca untuk pembandingan.

## Hasil

| Pemeriksaan | Hasil |
| --- | --- |
| ResNet18d train288/infer352, 5 checkpoint | OOF 3.014 × 30 dan mean test 2.010 × 30 identik persis |
| ResNet34d train224/infer320, 5 checkpoint | OOF 3.014 × 30 dan mean test 2.010 × 30 identik persis |
| Selisih probabilitas maksimum, keempat array | **0**; seluruh SHA-256 file NPY juga identik |
| Label OOF/test yang berbeda dari referensi | **0** |
| Full neural OOF Macro F1, sebelum balancing | 0,849455062621862 |
| Full neural OOF Macro F1, setelah balancing | **0,8525643117483758** |
| Label test yang diganti SIFT ≥5 inliers | 29 |
| Final CSV vs `best_sift_submission.csv` | **Identik byte-for-byte**, 0 perbedaan label |
| Raw files proyek vs raw files tim | 5.024/5.024 SHA-256 identik |
| Preprocessing ulang dari raw, ukuran 320 dan 352 | **10.048/10.048 view pixel-identical** dengan cache tim |
| Runtime utama | 140,66 detik, A100 80 GB yang dibagi |
| Unit tests | **21 lulus, 0 skipped** |

SHA-256 final CSV:

```text
6ca4cb673741edc3ca1e9903ed31dfb3b35924b52797742715a40e667f4d25e9
```

CSV tersebut identik dengan submission tim ref **58278059**, public Macro F1 **0,87012**, sebagaimana diverifikasi pada [review sebelumnya](../docs/TEAM_UPDATE_2026-11-13.md). Tidak ada submission atau skor leaderboard baru pada verifikasi ini.

## Metode dan provenance

Artefak tim dibaca dari `/home/nafisnaufal1426/inahuy/dateruy/`, lokasi yang disebut `docs/CODE_MAP.md` branch `kaka`. Runner memakai model `R_r18d_i288_e600_f0..4` dan `R_r34d_e600_f0..4`, masing-masing checkpoint 600 epoch. Config dan state dict diperiksa; state dict dimuat secara strict, dengan `pretrained=False`.

Environment `/home/nafisnaufal1426/venvs/dq/bin/python` cocok dengan delapan dependency yang dicatat dalam original W&B requirements **seluruh sepuluh run**: torch 2.14.1+cu130, torchvision 0.29.1+cu130, timm 1.0.31, NumPy 2.2.6, pandas 2.3.3, OpenCV headless 5.0.0.93, albumentations 2.0.8, dan scikit-learn 1.7.2. Tidak ada paket bersama yang diganti.

Source `train.py`, `dq.py`, dan `evalres.py` disalin ke snapshot lokal dan di-hash. Fungsi model, normalisasi, prediction, rotation, validation preprocessing, dan balancing dieksekusi dari snapshot tersebut; training launcher tidak diimpor. Dataset metadata, label order, fold IDs, dan seluruh hash input diperiksa sebelum/sesudah proses. File/proses milik tim tidak diubah. Frozen specification menyimpan config dan hash runner; resume menolak perubahan specification atau source.

Fold sizes `[614,607,603,600,590]` berasal dari file tim aktual. Untuk OOF, setiap checkpoint hanya memprediksi query pada fold yang ditahan. Untuk test, mean lima fold dihitung dalam urutan asli. Inference menggunakan batch 128, autocast BF16, channels-last, normalisasi RGB asli, dan rata-rata softmax original/horizontal flip. Backend flags mengikuti reference inference. Seluruh hasil ternyata exact; toleransi pembanding yang dibekukan sebelumnya (`atol=1e-5`, `rtol=1e-4`) tidak diperlukan untuk menyatakan kelulusan.

Forward pass memakai frozen caches tim. Secara terpisah, **seluruh** 3.014 gambar train dan 2.010 gambar test dari dataset proyek di-hash terhadap raw files tim dan diproses ulang pada kedua resolusi. Train memakai rotasi dan corruption v4 dengan seed per indeks asli; test memakai RGB decode dan INTER_AREA squash. Seluruh piksel hasil baru dibandingkan dengan cache input forward pass dan identik. Ini memverifikasi kesetaraan input dari gambar mentah tanpa menjalankan model dua kali pada piksel yang sama.

Hasil baru dua keluarga dirata-ratakan 50:50, memakai train-class prior balancing 100 iterasi, lalu menerapkan tabel `runs/sift_test.csv` asli dengan minimum 5 inliers. **Pencarian match SIFT belum dihitung ulang**; klaim reproduksi ini mencakup neural inference dan penerapan tabel SIFT referensi, bukan pembangunan ulang matcher. Training dari random initialization juga belum diulang. Keterbatasan evaluasi SIFT OOF pada laporan sebelumnya tetap berlaku.

## Menjalankan ulang

Runner: [reproduce_checkpoint_inference.py](../scripts/reproduce_checkpoint_inference.py). Config: [BASELINE-INFERENCE-REPRO-001.yaml](../configs/BASELINE-INFERENCE-REPRO-001.yaml). [Ringkasan JSON](results/BASELINE-INFERENCE-REPRO-001.summary.json) menyimpan hasil, versi, config, source/checkpoint/cache hashes, per-fold comparisons, dan hash keempat probabilitas.

Command membutuhkan environment dan artefak lokal tim yang dicatat di config; clone repository saja belum menyediakan bobot/cache. Semua output ditulis di folder proyek sendiri. GPU sharing sudah diotorisasi; satu model aktif per giliran dengan batas allocator 20% memori GPU, empat CPU threads, dan empat preprocessing workers.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  NO_ALBUMENTATIONS_UPDATE=1 nice -n 10 \
  /home/nafisnaufal1426/venvs/dq/bin/python -u \
  scripts/reproduce_checkpoint_inference.py
```

Artefak lokal ada di `outputs/BASELINE-INFERENCE-REPRO-001/`: `result.json`, `preflight.json`, `specification.json`, frozen sources, per-checkpoint validation/test predictions, empat array neural, raw-image comparison records, dan `recomputed_best_sift_submission.csv`. Dataset, checkpoint, dan prediksi tidak dipublikasikan ke GitHub.

Sebelumnya [LOCAL-PROP-002](LOCAL-PROP-002.md) hanya merekonstruksi CSV dari probabilitas tim yang tersimpan. Verifikasi ini melengkapi tahap **checkpoint → probabilitas** dan **raw image → cache**. Baseline neural sekarang memiliki bukti reproduksi aktual untuk eksperimen lanjutan; skor tetap hasil baseline yang sama.
