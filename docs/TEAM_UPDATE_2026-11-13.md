# Review update tim — 13 November 2026

Pemeriksaan memakai `kaggle-orie` dan `gh-orie`, dimulai sekitar 06:46 UTC. Branch yang diperiksa adalah **`kaka`**, commit **`b447792e3fc8ded59b98f8889023ab9f9b94c47e`**. Branch `main` masih di scaffold `e0189a9`. Salinan terpisah disimpan di `outputs/upstream/kaka/`; working tree riset lokal tidak di-merge atau di-reset.

Sumber utama: [README branch kaka](https://github.com/maulnite/sumatra-flora-classification/blob/b447792e3fc8ded59b98f8889023ab9f9b94c47e/README.md), [registry eksperimen](https://github.com/maulnite/sumatra-flora-classification/blob/b447792e3fc8ded59b98f8889023ab9f9b94c47e/docs/EXPERIMENT_LOG.md), [riwayat submission Kaggle](https://www.kaggle.com/competitions/dateraquest-2-0-machine-learning-competition/submissions). Snapshot dan bukti pemeriksaan lokal ada di `outputs/updates/2026-11-13/`.

## Hasil terbaru yang terverifikasi di Kaggle

Ada **10 submission COMPLETE**, dibanding 9 pada audit sebelumnya. Submission baru:

| Item | Nilai |
| --- | --- |
| Ref | `58278059` |
| File | `best_sift_submission.csv` |
| Waktu | 12 Nov 2026 15:56:58 UTC = 22:56:58 WIB |
| Public Macro F1 | **0.87012** |
| Private | Belum tersedia |
| Kenaikan atas best sebelumnya 0.84120 | **+0.02892** |
| Kenaikan atas submission terakhir 0.83262 | **+0.03750** |
| Tim / rank snapshot | `pensi lomba data`, team ID 17315594, **rank 24** |
| Identifier di deskripsi | `sha=6ca4cb673741edc3` |

Deskripsi asli:

```text
scratch ens4 resnet18d+resnet34d-224 mix-corruptv4 ep600 mixcut0.5 ema999 rot180 scenefold5 mean tta-flip infer352+320 prior-balance sift-inlier5 sha=6ca4cb673741edc3
```

Menurut [EXP-024](https://github.com/maulnite/sumatra-flora-classification/blob/b447792e3fc8ded59b98f8889023ab9f9b94c47e/experiments/EXP-024.json), pipeline akhirnya memakai mean dua keluarga model, masing-masing lima fold:

1. **ResNet18d**, train **288 px**, **600 epoch**, inference **352 px** (`EXP-020`).
2. **ResNet34d**, train **224 px**, **600 epoch**, inference **320 px** (`EXP-018`).

Kemudian diterapkan prior balancing dan SIFT dengan ambang ≥5 inliers. Tercatat **29 prediksi diubah** oleh SIFT; ini berbeda dari jumlah seluruh test yang mempunyai match. Token `ens4` di deskripsi tidak dijadikan bukti empat arsitektur karena record final dan script proxy menunjuk dua keluarga model tersebut.

OOF yang dicatat untuk single ResNet18d@352 adalah **0.8449**; mean ensemble dengan balance, sebelum SIFT, **0.8526**. Angka OOF ini dibaca dari laporan/log, belum direproduksi dari checkpoint lokal. Varian final tanpa SIFT tidak disubmit, sehingga kontribusi marginal SIFT terhadap public 0.87012 belum terukur secara terpisah.

## Definisi `mix-corruptv4` kini tersedia

Kode [train.py](https://github.com/maulnite/sumatra-flora-classification/blob/b447792e3fc8ded59b98f8889023ab9f9b94c47e/code/train.py#L276-L290) menggabungkan dataset view clean dan view corrupt, lalu mengambil sampel secara acak dari gabungan itu. Proporsinya sekitar **50% view clean dan 50% view corrupt_v4**. View clean tetap memakai augmentasi ringan serta geometri yang dipilih recipe.

Campuran view tersebut berbeda dari **MixUp/CutMix**: operator batch ini diaktifkan terpisah dengan probabilitas 0,5, MixUp alpha 0,2, CutMix alpha 1,0, dan switch probability 0,5.

[dq.py bagian corrupt_v4](https://github.com/maulnite/sumatra-flora-classification/blob/b447792e3fc8ded59b98f8889023ab9f9b94c47e/code/dq.py#L72-L93) memuat:

- RGB shift / HSV shift / channel shuffle, dengan kemungkinan blok aktif 0,9 dan bobot relatif 4:2:1.
- Penguatan saturation, grayscale probabilitas 0,1, dan brightness/contrast.
- Affine dengan scale/translation/rotasi kecil serta border hitam.
- Salah satu shadow, haze, sun flare, rain, atau occluder poligonal, probabilitas blok 0,4.
- Black coarse dropout, 1–3 area, probabilitas 0,3.
- JPEG kualitas **10–50**, probabilitas **1,0**, pada gambar sebelum crop/resize training.

Recipe R menambahkan rotasi uniform **−180 sampai +180 derajat** dengan probabilitas 0,9, kemudian corruption dan random resized crop. EMA decay **0,999** terbaca langsung di kode. Model dibuat dengan `timm.create_model(..., pretrained=False)`.

Setup training default mereka: batch 64, AdamW LR **0,002**, weight decay **0,05**, label smoothing **0,1**, warmup tiga epoch lalu cosine, serta normalisasi kanal tetap. Nilai-nilai ini berbeda dari draft lokal kita; mengganti banyak faktor sekaligus akan menyulitkan atribusi hasil.

## Eksperimen yang relevan agar tidak diulang

| ID | Eksperimen / hasil yang dicatat | Implikasi |
| --- | --- | --- |
| EXP-007 | Augmentasi ringan, ResNet18 300 epoch: OOF **0.3990** pada `scene5_v4rot` | Sudah ada pembanding arah light augmentation |
| EXP-008 | Recipe R, ResNet18 300 epoch: OOF **0.6723** pada protokol tersebut | Titik awal yang lebih relevan untuk ablation corruption |
| EXP-006 | `corrupt_hard`, fold 0: **0.626**, dibanding E **0.651** pada protokol pHash lama | Memperberat corruption tidak otomatis memperbaiki model; bukti hanya satu fold |
| EXP-013 | ResNet18d 300 epoch: OOF **0.7524** | Baseline arsitektur yang sudah dikerjakan |
| EXP-014 | ResNet18 600 epoch: OOF **0.7509** | Training lebih panjang sudah diuji |
| EXP-018/020 | ResNet34d dan ResNet18d train 288, 600 epoch | Komponen hasil public terbaru |
| EXP-021 | Normalisasi warna/contrast pada inference menurunkan OOF | Hindari mengulang perubahan tersebut tanpa hipotesis baru |
| EXP-025 | ResNet18d train 288/300 epoch serta `corrupt_v5` 224/300 epoch, **paused** dengan checkpoint | Varian blur/downscale/grayscale sudah direncanakan dan sebagian dijalankan |

`corrupt_v5` di [train.py](https://github.com/maulnite/sumatra-flora-classification/blob/b447792e3fc8ded59b98f8889023ab9f9b94c47e/code/train.py#L42-L48) menambahkan Gaussian/motion blur atau downscale dengan probabilitas 0,35 di atas v4, kemudian grayscale tambahan 0,1. Belum ada hasil selesai untuk varian ini. Catatan paused berasal dari repo; audit ini tidak mengubah proses atau checkpoint pada server tim.

## Temuan validasi dan reproduksi

### Scene split tim berbeda dari grouping audit lokal

Protokol `scene5_v4rot` mencatat **237 scene groups**, dengan ukuran fold `[614, 607, 603, 600, 590]`. Grouping audit kita memiliki 2.457 broad groups dan fold yang berbeda. Skor dari kedua protokol tidak bisa dibandingkan langsung sebagai efek perubahan augmentasi.

`scene_folds.csv` / `.npy` **belum ada di branch yang diperiksa**. Kode hanya mengimpor file scene fold tersebut; builder yang tersedia membuat pHash fold, bukan scene237. Checkpoint, OOF/test probabilities, dan `sift_test.csv` juga tidak disertakan. Ini sesuai kebijakan artefak besar tidak dimasukkan Git, tetapi menghambat reproduksi identik sampai artefak tim tersedia.

Untuk `--img 288`, runner membutuhkan cache `corr4r_288.npy` dan `test_288.npy`. `train.py prep` sendiri hanya menyiapkan cache dasar 224; builder ukuran lain ada di `evalres.py`. Urutan instruksi harus mencakup pembuatan cache ukuran tambahan.

Requirements repo belum mendeklarasikan beberapa import pipeline seperti timm, albumentations, OpenCV, dan imagehash. W&B bersifat opsional. Belum ada instalasi dependency atau uji training pada review ini. Resume checkpoint menyimpan model/optimizer/scheduler/EMA, tetapi kode yang diperiksa tidak menyimpan state RNG untuk melanjutkan urutan stokastik secara persis.

### Proxy SIFT belum mengecualikan validation fold

[sift_match.py](https://github.com/maulnite/sumatra-flora-classification/blob/b447792e3fc8ded59b98f8889023ab9f9b94c47e/code/sift_match.py#L45-L74) membangun galeri dari seluruh train. Untuk query corrupted train, matching memakai `exclude=i`: gambar sumber query sendiri dikeluarkan, tetapi referensi dari seluruh validation fold tetap dapat dipakai. [sift_oof_check.py](https://github.com/maulnite/sumatra-flora-classification/blob/b447792e3fc8ded59b98f8889023ab9f9b94c47e/code/sift_oof_check.py#L11-L24) memasukkan label match tersebut ke prediksi OOF tanpa filtering fold.

Pemeriksaan numerik `experiments/sift/sift_trainq.csv` terhadap `train.csv` lokal mengonfirmasi:

- Pada ≥5 inliers: **754 query / 3.014 (25,0%)** mempunyai match, seluruh label match cocok dengan label query, dan tidak ada self-match.
- **341 dari 754 match** ditandai `grup_sama=True` oleh artefak tim. Dalam protokol scene-group split, referensi satu group berada di fold yang sama dengan query.

Presisi 100% pada proxy tersebut memang cocok dengan CSV yang tersedia, tetapi bukan estimasi generalisasi pada galeri yang hanya berisi training fold. Skor post-SIFT **0.8656** perlu diperlakukan sebagai proxy, belum sebagai OOF bebas leakage. Evaluasi yang benar perlu membatasi reference gallery ke fold training untuk setiap query validation. Untuk inference test, seluruh train berlabel dapat menjadi reference gallery.

Selain itu, ensemble/resolusi dipilih pada OOF yang juga dipakai melaporkan skor; repo telah mencatat optimisme tersebut. Prior balancing memakai asumsi prior kelas test menyerupai prior train, yang perlu divalidasi. Public **0.87012** tetap merupakan skor Kaggle yang terverifikasi; perubahan model/resolusi/balance/SIFT pada hasil final merupakan paket perubahan, bukan ablation setiap faktor.

## Dampak pada rencana lokal

Draft lokal yang sebelumnya disebut EXP-004 belum menjalankan training. Karena branch tim telah memakai **EXP-004 sampai EXP-025**, draft dipindahkan ke **[LOCAL-PROP-001](../experiments/LOCAL-PROP-001.md)**, dengan [config proposal](../configs/LOCAL-PROP-001.yaml) berstatus `needs_revision` dan tanpa ID eksperimen tim. Isi awal dipertahankan sebagai konteks, bukan protokol eksekusi baru.

Sebelum memilih eksperimen berikutnya, fondasi yang relevan adalah recipe R/ResNet18d dan protokol tim yang dibekukan. Perbandingan v4/v5 yang sudah tercatat perlu ditinjau agar pekerjaan tidak duplikat; perbaikan evaluasi SIFT dapat dilakukan sebagai pemeriksaan terpisah setelah fold dan artefaknya tersedia. Draft light-vs-corrupt baru akan banyak mengulang arah yang sudah dicoba tim dan belum mempunyai evaluator yang sebanding.

**Update setelah review:** user mengizinkan rencana tetap dilanjutkan sambil menunggu artefak tim. [Proposal revision 2](../experiments/LOCAL-PROP-001.md) memakai baseline recipe R dan dua run baru pada holdout lokal yang sama, dengan satu tambahan Gaussian blur setelah resize. Protokol lokal diberi nama berbeda dan tidak diklaim sebanding dengan scene237 OOF. Draft awal diarsipkan; config aktif kini berstatus `revised_ready_for_implementation`.

## Pemeriksaan yang dilakukan

- Clone branch terpisah dan pin commit; source tim tidak diedit.
- Delapan file Python pipeline berhasil diparse AST tanpa mengeksekusi training.
- Registry check: **22 experiment records, 0 masalah schema**. Pemeriksaan schema tidak membuktikan angka training sudah direproduksi.
- MD5 `dq.py` dan backup `train.before-img.py` cocok dengan provenance yang dicatat; `train.py` saat ini adalah versi patched yang lebih baru.
- Label/reference index proxy SIFT diperiksa ulang menggunakan label train resmi.
- Training dan inference model, instalasi dependency bersama, push GitHub, serta submission Kaggle tidak dijalankan.
