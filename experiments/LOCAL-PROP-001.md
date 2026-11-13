# LOCAL-PROP-001 — Recipe R ResNet18d dengan blur ringan setelah resize

> Revision **2** | Status: **Executing A/B**
>
> Owner: `@maulnite` bersama Codex | 13 November 2026
>
> ID eksperimen tim belum dialokasikan. Eksekusi diotorisasi 13 November 2026.

Satu eksperimen dengan satu perubahan: **apakah Gaussian blur ringan setelah resize 224×224 meningkatkan ketahanan recipe R + ResNet18d terhadap kehilangan detail?** [Config](../configs/LOCAL-PROP-001.yaml), split lokal, [runner](../scripts/run_local_experiment.py), serta empat cache validation sudah tersedia. Smoke check CPU dan GPU bf16 berhasil; kontrol A dijalankan sebelum kandidat B.

## Dasar revisi

[Update tim](../docs/TEAM_UPDATE_2026-11-13.md) menunjukkan augmentasi `corrupt_v4` sudah menjadi baseline yang matang. Light-vs-heavy sudah dicoba. EXP-025 memakai `corrupt_v5`, yaitu v4 diikuti blur/downscale pada resolusi asli dan grayscale tambahan; run tersebut masih tercatat paused.

[Audit lokal](../docs/DEEP_DATASET_AUDIT_2026-11-12.md) menemukan median variance Laplacian 2.730 pada train dan 860 pada test setelah resize 256×256. Kontrol JPEG train memberi sekitar 2.654. Ini mendukung investigasi kehilangan detail di luar JPEG, tetapi isi scene/kontras juga memengaruhi ukuran tersebut.

Hipotesis kita lebih spesifik: blur sebelum downsampling dapat memiliki efek kecil pada input akhir. Blur ringan pada resolusi input model memberi perubahan yang bisa diukur langsung. A/B ini menguji efek penambahan operator post-resize; perbandingan langsung posisi native vs post-resize memerlukan eksperimen lanjutan.

## Lanjut memakai protokol lokal

User belum memperoleh lokasi scene split/checkpoint tim dan mengizinkan rencana dilanjutkan. Karena itu **kedua arm dilatih baru pada holdout lokal yang sama**, memakai config recipe R ResNet18d dari branch `kaka`, commit `b447792e3fc8ded59b98f8889023ab9f9b94c47e` sebagai referensi.

Protokol diberi nama **`local_holdout_v4rot224_v1`**. Skor OOF tim 0.7524 atau public 0.87012 menjadi konteks historis; pembandingan utama adalah A vs B lokal. Tidak ada ketergantungan pada scene237 atau checkpoint tim yang belum dibagikan.

## Split yang dibekukan

Mulai dari broad scene groups audit dengan fold 2 sebagai validation tetap. Tambahkan relasi dari corrupted-train SIFT matching tim pada **≥16 inliers**, periksa indeks/label referensi terhadap metadata resmi, lalu gabungkan connected components. Calon train yang satu component dengan validation diberi role `purged_scene_neighbor`.

| Item | Nilai |
| --- | --- |
| Train / validation | **2.378 / 634** |
| Calon train yang dipisahkan karena scene overlap tambahan | **2** gambar |
| Cakupan kelas | Seluruh **30 kelas** pada kedua partisi |
| Relasi SIFT train ≥16 inliers | 342 |
| Expanded scene groups | 2.452 |
| Overlap strict / broad / expanded group | **0 yang terdeteksi** |
| Manifest | `outputs/LOCAL-PROP-001/plan/split_manifest.csv` |
| SHA-256 | `6ef3ac1e78c9cce9bb45a47d3a74d960ddddb3ab2cb95970d4093bb7e7d18d37` |

File/label asli tetap utuh. Artefak SIFT memakai implicit row order dan tidak memuat ukuran coverage; grouping tetap provisional, bukan jaminan seluruh scene sudah teridentifikasi. Threshold 16 dipakai sebagai guard konservatif, selaras dengan minimum inliers audit ketat. Tidak ada label test yang dibuat atau digunakan.

## Setup yang sama pada kedua arm

| Faktor | Nilai |
| --- | --- |
| Model | timm ResNet18d, `pretrained=False`, dropout 0,2, avg pooling |
| Input training / inference | 224 / 224 |
| Training | 300 epoch, batch efektif 64 |
| Optimizer | AdamW, LR 0,002, weight decay 0,05 |
| Scheduler | Warmup 3 epoch, cosine |
| Regularisasi | Label smoothing 0,1, EMA 0,999 |
| View mix | Sekitar 50% clean + 50% corrupt_v4 |
| Geometri | Rotasi ±180° p=0,9, RRC scale 0,25–1 / ratio 0,5–2, flip |
| MixUp / CutMix | p=0,5; alpha 0,2 / 1,0 |
| Normalisasi | Konstanta kanal yang digunakan pipeline tim |

**A:** recipe tersebut tanpa blur tambahan. **B:** operator Gaussian blur setelah resize, kernel 7×7, sigma uniform **0,4–1,0 piksel**, border reflect101, probabilitas **0,25 pada view corrupt**. Diterapkan sebelum normalisasi dan MixUp/CutMix; view clean tidak mendapat blur tambahan. Secara ekspektasi sekitar **12,5% seluruh training view** mendapat operator baru.

Seluruh faktor lain dipertahankan. Parameter merupakan satu rancangan awal yang ditetapkan sebelum melihat skor model. Kedua run memakai initialization dan sampler order yang sama; RNG blur dipisahkan agar penambahan operator tidak menggeser transformasi bersama. Resume perlu menyimpan state RNG.

## Evaluasi yang dibekukan

Semua kondisi memakai **634 gambar validation yang sama**, cache dibekukan sebelum A/B:

| Kondisi | Fungsi |
| --- | --- |
| `v4rot` | **Metrik utama**: bentuk val_rot tim pada 224, seed berdasarkan indeks resmi |
| `clean` | Memantau performa gambar clean |
| `v4rot_blur_06` | View v4rot ditambah blur sigma 0,6 setelah resize |
| `v4rot_blur_10` | View v4rot ditambah blur sigma 1,0 setelah resize |

Macro F1 dihitung atas label 0–29 dengan `zero_division=0`; simpan per-class F1/confusion matrix. TTA **flip** sama pada semua kondisi. Simpan versi library, ID order, dan hash cache. View validation tidak berubah mengikuti arm.

Checkpoint pembanding adalah **EMA epoch 300** untuk keduanya. Evaluasi setiap 25 epoch hanya monitoring. SIFT dan prior correction tidak dimasukkan ke skor neural utama.

`blur_F1` adalah mean Macro F1 dua kondisi blur. Paired bootstrap memakai expanded scene groups di dalam setiap kelas, 2.000 replikasi seed 43; intervalnya diagnostic pada satu holdout lokal.

B layak dilanjutkan jika **blur_F1 naik ≥0,02**, penurunan **v4rot_F1 ≤0,005**, dan penurunan **clean_F1 ≤0,01**. Periksa juga per-class failures dan ketidakpastian. Angka tersebut kriteria operasional, bukan jaminan hasil atau bukti generalisasi ke test. Tidak ada tambahan seed/fold atau ensemble dalam eksperimen awal ini.

Pada 13 November user mengotorisasi eksekusi dan **satu submission otomatis jika skor meningkat**. Gate submit lebih ketat: seluruh kriteria di atas harus lolos **dan v4rot Macro F1 B harus lebih tinggi dari A**. CSV berasal dari checkpoint EMA epoch 300 kandidat yang dilatih pada 2.378 gambar train lokal; tidak ada full-data refit, SIFT, prior correction, atau klaim bahwa skor public akan melampaui 0.87012. Runner memvalidasi 2.010 ID/label, memeriksa kuota tim, dan menyimpan ledger sebelum mencoba upload melalui `kaggle-orie`. Respons upload yang ambigu tidak dicoba ulang otomatis.

## Eksekusi

Source augmentasi tim disalin tanpa perubahan ke `scripts/local_dq.py` dari commit yang dipin. Runner baru mendukung manifest, seed stateless per epoch/ID/view, sampler terpisah, blur setelah crop, cache bersama, checkpoint optimizer/scheduler/EMA/RNG, evaluasi, inference, dan gate upload. Source di `outputs/upstream/kaka` tetap utuh.

Environment `.venv` memakai system-site-packages dan tambahan terisolasi dari [requirements-local-experiment.txt](../requirements-local-experiment.txt): NumPy 1.26.4, OpenCV 4.11.0.86, Albumentations 2.0.8. Torch 1.13.1+cu116, torchvision 0.14.1+cu116, serta timm 1.0.30 yang sudah ada berhasil menjalankan CPU dan GPU smoke. Paket shared tidak diganti. Environment efektif tersimpan dalam `plan/environment-freeze.txt`; ini environment spesifik server, bukan klaim telah memenuhi seluruh generic requirements repo.

User mengizinkan berbagi GPU A100; satu job dijalankan dengan memory fraction cap 0,20 (sekitar 16 GiB) dan prioritas CPU `nice 10`. Proses pengguna lain tidak diubah. Runtime diukur pada sepuluh epoch awal; checkpoint disimpan setiap sepuluh epoch.

Perintah pengendali yang telah dijalankan dalam proses terpisah:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u scripts/run_local_experiment.py auto --allow-gpu-sharing
```

`outputs/LOCAL-PROP-001/automation_process.json` mencatat PID/command; `automation.log` dan `status.json` mencatat progres. Lock proyek menolak duplikasi runner. Setelah selesai, `comparison.json` memuat metrik, delta, paired bootstrap, dan per-class F1. Gate gagal berarti tidak ada submission. Gate lolos menghasilkan CSV, hash model/file, ledger upload, dan respons Kaggle.

Verifikasi sebelum training: **14 unit tests lulus**, CPU smoke finite backward/EMA/RNG roundtrip lulus, GPU bf16 forward/backward/AdamW/EMA lulus. Inisialisasi A/B dibandingkan melalui hash bobot; manifest dan cache diverifikasi sebelum run/resume. Seed/CUDA algorithm tetap membatasi klaim bitwise reproducibility lintas hardware/library.

Referensi implementasi: [Albumentations reproducibility](https://albumentations.ai/docs/4-advanced-guides/reproducibility/). API `Compose.set_random_seed` juga diperiksa langsung pada versi 2.0.8 yang dipasang; Lambda upstream memakai RNG NumPy global sehingga runner men-seed keduanya.

Perbaikan evaluasi SIFT menjadi tindak lanjut terpisah: galeri referensi untuk setiap query validation harus dibatasi ke training partition. Proxy 0.8656 tidak menjadi target eksperimen neural ini.

## Artefak dan hasil

[Script pembuat split](../scripts/prepare_proposal_split.py): `python scripts/prepare_proposal_split.py`. Metadata, class counts, serta daftar purge ada di `outputs/LOCAL-PROP-001/plan/split_metadata.json`. Draft dan manifest lama diarsipkan di `outputs/LOCAL-PROP-001/archive/revision_1_superseded_2026-11-13/`.

| Hasil | A | B | Delta B−A |
| --- | ---: | ---: | ---: |
| v4rot Macro F1 | — | — | — |
| Clean Macro F1 | — | — | — |
| Mean blur Macro F1 | — | — | — |
| Runtime / peak memory | — | — | — |

Config revision 2, runner, dan cache tersedia; eksekusi A/B dimulai. Angka hasil belum diisi sampai run benar-benar selesai. Progres aktual tersedia di `outputs/LOCAL-PROP-001/status.json`.
