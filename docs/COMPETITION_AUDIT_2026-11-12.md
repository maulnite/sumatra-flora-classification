# Audit awal DATERAQUEST — 12 November 2026

Audit lanjutan kini tersedia di [audit kemiripan scene, kandidat label, dan corruption](DEEP_DATASET_AUDIT_2026-11-12.md). Keterangan "belum diuji" di snapshot awal berikut merujuk pada kondisi sebelum audit lanjutan tersebut.

Sumber primer: halaman kompetisi, data, riwayat submission, dan leaderboard yang diambil menggunakan `kaggle-orie` pada 12 November 2026 (UTC). Repository diperiksa menggunakan `gh-orie`, lalu di-clone ke workspace proyek. Angka leaderboard merupakan snapshot, bukan hasil validasi lokal.

- Kompetisi: https://www.kaggle.com/competitions/dateraquest-2-0-machine-learning-competition
- Aturan: https://www.kaggle.com/competitions/dateraquest-2-0-machine-learning-competition/rules
- Riwayat tim: https://www.kaggle.com/competitions/dateraquest-2-0-machine-learning-competition/submissions
- Repository: https://github.com/maulnite/sumatra-flora-classification
- Snapshot mentah lokal: `outputs/discovery/competition_pages.json`, `submissions.csv`, `leaderboard.csv`, dan `submission_limits.json` (diabaikan Git).

## Kompetisi dan batasan

Klasifikasi citra single-label **30 spesies tanaman Kebun Raya ITERA**, label integer 0–29. Metrik resmi **Macro F1**: kontribusi setiap kelas setara. Public leaderboard menggunakan 30% test; private 70% dan menentukan seluruh peringkat penyisihan.

Model harus dilatih **dari nol**, memakai data resmi saja. Blueprint ResNet/EfficientNet/ConvNeXt/ViT diperbolehkan; bobot pretrained, dataset eksternal, VLM, API inferensi, AutoML instan, dan pelabelan manual test dilarang. Augmentasi dan tuning hyperparameter diperbolehkan. Seed, notebook, hasil validasi, dan prediksi harus dapat direproduksi. Aturan resmi juga melarang plagiarisme/forking kode publik dan kolaborasi lintas tim.

Deadline penyisihan **17 November 2026 23:59 WIB = 16:59 UTC**. Selain CSV Kaggle, wajib mengumpulkan `IDTim_NamaTim.zip` melalui portal panitia, berisi notebook `.ipynb` yang telah dijalankan lengkap dengan log training/metrik validasi dan CSV yang menghasilkan prediksi sesuai submission. Finalis menyerahkan PPTX pada 26 November 2026 23:59 WIB.

Aturan tertulis membatasi **3 submission/hari**. Endpoint limits saat audit mengembalikan `numToday=2`, `numTotal=9`, `numAllowedNow=2`. Angka allowance API tidak konsisten dengan batas tertulis jika dianggap kuota harian tersisa; jangan menganggapnya izin untuk melewati 3/hari. Audit ini tidak mengirim submission.

## Kondisi repository

Repository ternyata sudah berisi scaffold, bukan kosong. HEAD saat clone: `e0189a9` (`chore: add ML folder scaffold, validation scripts, tests and CI`). Sebelumnya ada dokumentasi aturan, template eksperimen, helper inspeksi/validasi CSV, dan CI. Belum ada implementasi training, notebook hasil eksperimen, checkpoint, probabilitas OOF, atau konfigurasi nyata dari submission yang ditemukan. EXP-001 sampai EXP-003 masih placeholder dan tidak dipetakan secara otomatis ke submission.

Identitas pembuat masing-masing submission tidak tersedia di kolom output CLI. Nilai `sha=...` pada deskripsi dicatat sebagai identifier artefak yang belum diketahui maknanya; tidak diasumsikan sebagai commit Git repository.

Perubahan audit dibuat lokal pada branch `docs/maulnite/competition-audit`. Dataset dan output tetap di folder yang diabaikan Git. Tidak ada push atau perubahan remote repository.

## Dataset yang telah diunduh dan dimuat

Arsip resmi berukuran sekitar **761 MiB**, berisi 5.027 file (5.024 gambar + 3 CSV), total 798.401.845 byte setelah ekstraksi. Data berada di `data/raw/`.

| Komponen | Hasil pemeriksaan |
| --- | --- |
| Train | **3.014 gambar**, 30 folder kelas; CSV `id,label` |
| Test | **2.010 gambar**, tanpa label; template `id,label` |
| Mapping | 30 baris, kolom `label,species,folder_name` |
| Kelas terkecil | Label 19, `Nepenthes_Sumatrana`, **47** gambar |
| Kelas terbesar | Label 12, `Echinodorus_radican`, **169** gambar |
| Rasio terbesar/terkecil | **3,60x** |
| Integritas | Seluruh gambar ada, tidak ada file gambar di luar metadata, semua ID unik dan folder cocok dengan label |
| Decode | Seluruh **5.024 gambar JPEG RGB** berhasil dibaca penuh |
| Duplikat identik | **0 grup**, baik SHA-256 file maupun piksel RGB beserta dimensinya; termasuk pemeriksaan lintas train/test |
| Urutan template | Tepat `test_0001.jpg` sampai `test_2010.jpg` |

**Ada perbedaan encoding JPEG train dan test yang besar.** Dimensi paling umum sama: 608×1080 (train 1.442, test 963 gambar), disusul 768×1024 (507/335). Namun median file train **231.987 byte**, test **29.878,5 byte**. Setelah dinormalisasi luas gambar, median byte/piksel **0,3395 vs 0,04344**, sekitar **7,82x**.

Pemeriksaan header JPEG memperkuat bahwa ini perbedaan kompresi, bukan hanya konten: train mempunyai satu tabel kuantisasi luminance dengan mean **9,25**, sedangkan test mempunyai **32** tabel dengan mean **57,625–196,25** (median **135,625**). Nilai kuantisasi lebih besar menunjukkan kuantisasi lebih kasar. Ini tidak membuktikan semua jenis corruptions yang diterapkan panitia, tetapi menunjukkan domain shift encoding yang nyata. Temuan tersebut konsisten dengan motivasi augmentasi `mix-corruptv4` pada deskripsi eksperimen; isi augmentasinya tetap harus diperiksa dari kode.

Contact sheet train menunjukkan variasi skala objek (daun/bunga hingga tanaman utuh), sudut, pencahayaan, dan latar yang ramai. Ini observasi visual pada sampel train, bukan analisis label test. Kesamaan scene/individu tanaman **belum diuji**; tidak adanya duplikat identik tidak menjamin split acak bebas leakage.

Artefak lokal: `outputs/discovery/dataset_audit.json` (statistik dan checks), `image_inventory.csv` (metadata dan hash per gambar), `jpeg_encoding_summary.json` (header JPEG), dan `train_contact_sheet.jpg` (satu contoh tiap kelas). Distribusi lengkap setiap kelas tercatat dalam JSON. Semua diabaikan Git.

Implikasi untuk eksperimen berikutnya: gunakan Macro F1/per-class F1, validasi dengan augmentasi kompresi yang berasal dari train saja, dan verifikasi group/scene split sebelum mempercayai skor validasi. Jangan mengasumsikan prior label test sama dengan train hanya dari jumlah gambar; prosedur `prior-balance` memerlukan justifikasi dan validasi tersendiri.

## Submission tim yang terverifikasi

Ada **9 submission COMPLETE**; semua private score masih kosong. Leaderboard dan hasil pencarian kompetisi konsisten dengan tim **pensi lomba data** (team ID `17315594`), rank **25 dari 83 tim**, best public **0.84120** saat audit. Pencocokan nama tim berasal dari leaderboard yang skornya sama dengan best submission dan rank akun 25; output `submissions` sendiri tidak menampilkan team ID/author.

Tanggal dan jam berikut dalam **UTC**, urut kronologis. File bernama `submission.csv` dipisahkan berdasarkan ref, bukan nama file.

| Ref | Waktu UTC | File | Public Macro F1 | Ringkasan deskripsi |
| --- | --- | --- | ---: | --- |
| 58148247 | 09 Nov 02:19 | `submission.csv` | 0.58605 | scratch-v1, paired160, CutMix189, SIFT15 |
| 58149138 | 09 Nov 02:58 | `submission.csv` | 0.39263 | scratch neural-control, paired160, CutMix189 |
| 58168024 | 09 Nov 14:52 | `submission_modelE.csv` | 0.51865 | ResNet18-224, mix-corruptv4, 150 epoch, fold0, flip TTA |
| 58182144 | 10 Nov 01:56 | `sub_F_5fold.csv` | 0.67725 | ResNet18-224, 300 epoch, mixcut0.5, ema999, mean 5-fold, flip TTA |
| 58189158 | 10 Nov 06:08 | `submission.csv` | **0.84120** | r18d, group237, fullrefit40, flip, CPUfp32, cascade90, SIFT10 |
| 58207313 | 10 Nov 16:14 | `R_Rcg_ens_submission.csv` | 0.73979 | ens2 ResNet18 + cbam-gem, rot180, scenefold5, flip TTA |
| 58222641 | 11 Nov 02:52 | `ens_best_submission.csv` | 0.80694 | ens2 ResNet18d+ResNet18-224, 300+600 epoch, scenefold5, flip TTA |
| 58262807 | 12 Nov 05:49 | `ens320_balance_submission.csv` | 0.82164 | ens3, infer320, prior-balance |
| 58262886 | 12 Nov 05:53 | `manip_best_sift_submission.csv` | 0.83262 | ens3, infer320, prior-balance, sift-inlier5 |

### Deskripsi asli, tanpa rekonstruksi konfigurasi

```text
58148247: scratch-v1 paired160 CutMix189 SIFT15 sha=90432f664295ca9c
58149138: scratch neural-control paired160 CutMix189 sha=d95c92d0c147a2f9
58168024: scratch resnet18-224 mix-corruptv4 ep150 fold0 tta-flip sha=8a5f4c7bf022ecab
58182144: scratch resnet18-224 mix-corruptv4 ep300 mixcut0.5 ema999 5fold-mean tta-flip sha=bb754fa098a4b4a4
58189158: scratch r18d group237 fullrefit40 flip CPUfp32 cascade90 SIFT10 sha=ff56dc4977e121b5
58207313: scratch ens2 resnet18-224 mix-corruptv4 ep300 mixcut0.5 ema999 rot180 scenefold5 + cbam-gem 5fold-mean tta-flip sha=2ee9e95970e889a2
58222641: scratch ens2 resnet18d+resnet18-224 mix-corruptv4 ep300+ep600 mixcut0.5 ema999 rot180 scenefold5 mean tta-flip sha=ac30067a72cf887f
58262807: scratch ens3 resnet18d+resnet18-224 mix-corruptv4 ep300+ep600 mixcut0.5 ema999 rot180 scenefold5 mean tta-flip infer320 prior-balance sha=69be8bb4b9dd7b74
58262886: scratch ens3 resnet18d+resnet18-224 mix-corruptv4 ep300+ep600 mixcut0.5 ema999 rot180 scenefold5 mean tta-flip infer320 prior-balance sift-inlier5 sha=8a04e7a7ca571ee4
```

## Pembacaan eksperimen

**Jalur neural + SIFT.** `scratch-v1 ... SIFT15` mencapai 0.58605, sedangkan `neural-control` dengan token paired160/CutMix189 yang sama mencapai 0.39263 (selisih 0.19342). Kemudian `r18d ... cascade90 SIFT10` menjadi best 0.84120. Ini mengarah pada hipotesis bahwa fitur lokal/matching SIFT memberikan informasi berguna bagi pipeline. Namun jumlah gambar yang diganti, cara pencocokan, threshold, neural backbone awal, dan apakah kedua run awal benar-benar kontrol berpasangan belum diketahui. Jangan mengartikan `SIFT15`, `SIFT10`, `paired160`, `CutMix189`, `cascade90`, atau `group237` sebagai nilai hyperparameter tertentu tanpa kode.

**Jalur ResNet dan ensemble.** Dari ResNet18 224 fold0/150 epoch (0.51865), deskripsi bergerak ke mean 5-fold/300 epoch plus mixcut/EMA (0.67725), ensemble dengan CBAM/GeM (0.73979), lalu ResNet18d+ResNet18 300/600 epoch (0.80694). Rotasi dan scene-fold tampaknya menjadi bagian strategi robust augmentation/validasi. `CBAM`, `GeM`, dan `EMA` lazim berarti attention, generalized mean pooling, dan moving average bobot; implementasi dan nilainya belum terverifikasi. `ens2`/`ens3` tidak cukup untuk memastikan total checkpoint karena setiap komponen bisa terdiri dari banyak fold.

**Perubahan terbaru.** Ensemble tiga komponen dengan inferensi 320 dan prior-balance mencapai 0.82164; versi berlabel sift-inlier5 mencapai 0.83262, naik **0.01098**. Terhadap ensemble sebelumnya 0.80694, kenaikannya **0.01470** sebelum SIFT. Semua ini perbandingan skor public antarsubmission, bukan estimasi efek kausal satu hyperparameter. Infer320 lazim berarti resolusi inferensi 320; formula prior-balance dan inlier filter belum tersedia.

**Best belum tergantikan.** Submission terbaru 0.83262 masih **0.00858 di bawah best 0.84120**. Jangan menganggap file terbaru sebagai artefak terbaik. Tidak ada validation Macro F1 atau private score untuk menilai generalisasi kedua jalur dari metadata yang tersedia.

Informasi yang masih diperlukan dari artefak internal tim: pemilik setiap run, notebook/config nyata, split/group/seed, validation Macro F1 dan per-class F1, checkpoint serta probabilitas OOF/test, cara integrasi SIFT dan prior correction, dan makna identifier `sha`. Ini adalah batas pemahaman dari metadata, bukan bukti bahwa suatu eksperimen tidak dikerjakan.

## Reproduksi inspeksi lokal

```bash
# Perintah download yang digunakan; data resmi saja.
kaggle-orie competitions download -c dateraquest-2-0-machine-learning-competition -p data/raw -q
# Arsip telah diekstrak di data/raw dengan pemeriksaan path sebelum ekstraksi.
python scripts/inspect_dataset.py --root data/raw
python scripts/audit_dataset.py --root data/raw --output outputs/discovery
python -m unittest discover -s tests -v
```

CLI pages versi ini membutuhkan slug di kedua posisi agar tidak tertimpa default parser:

```bash
kaggle-orie competitions pages dateraquest-2-0-machine-learning-competition list dateraquest-2-0-machine-learning-competition --content --format json
kaggle-orie competitions submissions dateraquest-2-0-machine-learning-competition --page-size 200 --format csv
```

Python aktif yang terverifikasi untuk inspeksi: `/opt/conda/bin/python` 3.10.4, NumPy 1.24.1, pandas 1.5.3, Pillow 9.4.0. `micromamba run -n orie python` juga menemukan executable tersebut; prefix environment orie tidak memiliki interpreter Python sendiri. CLI `kaggle-orie` melaporkan Kaggle 2.2.4 dari environment tool terpisah. Dependency bersama tidak diubah. Daftar versi lengkap disimpan di `outputs/discovery/environment.json`; environment ini belum dinyatakan sebagai environment training yang memenuhi `requirements.txt` repo.
