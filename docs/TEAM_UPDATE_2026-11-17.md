# Update tim — 17 November 2026

**Update terbaru, 13:59 UTC / 20:59 WIB:** best public tim menjadi **0.90780**,
`final_domain.csv`, ref **58470727**, COMPLETE. Kuota API **0 tersisa**.
Komposisi probabilitas dan CSV lokal direplay persis; detail ada pada
[follow-up final](#follow-up-final--1359-utc). Bagian awal berikut adalah
snapshot historis pukul 08:50 UTC.

Diperiksa memakai `kaggle-orie` dan `gh-orie`; snapshot terakhir
**08:50:45 UTC / 15:50:45 WIB**. Pengguna menyebut update berasal dari Nafis.
CLI submission menampilkan nama, deskripsi dan skor, tetapi tidak menampilkan
penulis individual. Angka publik terverifikasi; detail training di bawah masih
berasal dari deskripsi karena kode baru belum ditemukan di repo tim.

## Submission baru

| Ref | Waktu UTC, 17 Nov | File | Public macro F1 |
| --- | --- | --- | ---: |
| 58450478 | 00:55:46 | `sub_teamprobs_graph.csv` | 0.84096 |
| **58453652** | **03:01:40** | **`sub_mine5_team_mid.csv`** | **0.88314** |

Keduanya **COMPLETE**, private score belum tersedia. Best publik yang teramati
sekarang **0.88314**: **+0.01117** versus probe PVT-B3 kita **0.87197**, dan
**+0.01302** versus EXP-024 **0.87012**. Sumber langsung:
[submission terbaik](https://www.kaggle.com/competitions/dateraquest-2-0-machine-learning-competition/submissions?select=58453652),
[probe graph](https://www.kaggle.com/competitions/dateraquest-2-0-machine-learning-competition/submissions?select=58450478).

## Pipeline yang disebut dalam deskripsi

`sub_teamprobs_graph.csv` menggunakan probabilitas EXP-024 dari ResNet18d/34d,
prior balancing, lalu SIFT sibling graph dengan test→train ≥6 inlier dan
test→test ≥6, menggunakan similarity RANSAC. Hasilnya **0.84096**. Ini bukan
hasil graph pilot lokal kita dan tidak boleh dianggap resep graph yang sama.

`sub_mine5_team_mid.csv` menyebut:

- **Lima model scratch ResNet18d/34d baru**, dengan resolusi yang disebut
  **288–384px**. Pembagian backbone, jumlah seed/fold dan resolusi tiap tahap
  belum tersedia.
- **Full train + sibling-matched test images**. Cara pemilihan test, pembuatan
  label/target dan bobotnya belum terverifikasi. Pseudo-label otomatis dari
  matching merupakan kemungkinan, bukan temuan kode.
- **Checkpoint EMA saat training masih berjalan**, digabung dengan
  probabilitas tim **EXP-024** menggunakan **geometric mean**.
- **Prior balancing** dan evidence SIFT test→train yang disebut calibrated;
  deskripsi secara eksplisit menyatakan **tanpa test→test links**.

Artinya, update terbaik yang dijelaskan masih memakai keluarga **ResNet**,
dengan perubahan data training, model ensemble dan pemanfaatan evidence
matching. Deskripsi tidak menyebut PVT/transformer kita. Bobot geometric mean,
temperature, epoch, jumlah sibling, prior dan aturan kalibrasi belum diketahui.

Perbedaan skor kedua submission tidak mengisolasi manfaat menghapus test→test:
model, data dan penggabungan probabilitas juga berubah bersamaan. Checkpoint
yang dilatih pada seluruh train juga tidak dapat dievaluasi pada query fold 2
sebagai holdout bebas training; dibutuhkan prediksi atau run fold-safe terpisah
untuk membandingkan metrik lokal dengan eksperimen PVT kita.

## Hasil pemeriksaan GitHub

Repo [maulnite/sumatra-flora-classification](https://github.com/maulnite/sumatra-flora-classification/branches)
memiliki **13 branch** saat snapshot. Akun **`NafisNaufal`** terdaftar sebagai
collaborator, bersama `Inaka06` dan `maulnite`, tetapi belum ada branch atau PR
Nafis yang terlihat. Tidak ada fork repo tim atau repo publik milik Nafis dengan
nama yang cocok DATERA/flora/Sumatra yang ditemukan dalam pemeriksaan ini.
Ini tidak membuktikan bahwa tidak ada pekerjaan lokal atau repo privat lain.

- `main` tetap `e0189a92254c72955d27d842676af9f01675f38a`.
- `kaka` tetap `b447792e3fc8ded59b98f8889023ab9f9b94c47e`, sama dengan audit
  13 November; commit terakhirnya berasal dari `Inaka06`.
- `exp/maulnite/pvt-idmm` masih `7ef8f321c6d00281ea65e19baac8a50b32ae5f97`
  sebelum catatan ini dipublikasikan.
- `pushed_at` repo pada snapshot adalah **16 Nov 22:41:09 UTC**, yaitu push
  laporan PVT kita. Belum ditemukan push kode Nafis di repo ini.

Karena itu, **pipeline Nafis belum berhasil direproduksi atau diaudit dari kode**.
Untuk menggabungkan PVT dengan best baru, artefak yang masih diperlukan adalah
kode/config, manifest sibling dan cara pemberian target, checkpoint/epoch,
probabilitas test dengan urutan ID, serta rumus/parameter penggabungan dan SIFT.
Baseline publik untuk keputusan berikutnya diperbarui ke **0.88314**; baseline
EXP-024 tetap disimpan sebagai referensi yang sudah berhasil direproduksi.

Tidak ada training baru, perubahan proses rekan, atau submission baru dari audit
ini. Training B0 yang sudah dijadwalkan tetap berjalan. Snapshot mentah lokal:
`outputs/updates/2026-11-17/submissions.csv` dan `snapshot.json`.
[Ringkasan terstruktur](../experiments/results/TEAM-UPDATE-2026-11-17.summary.json).

## Follow-up server setelah pencarian diizinkan pengguna

Pencarian filename menemukan **`/home/nafisnaufal1426/akucintakc/work/`**, berisi
script training/inference, checkpoint, probabilitas dan CSV Nafis. Semua file
dibaca tanpa diubah. Clone Git terpisah ada di `repo/sumatra-flora-classification`
dan masih di `main`; kode ML yang ditemukan berada di luar clone tersebut.

Lima array `ptest_mid.npy` dan probabilitas tim sebagai enam input geometric
mean setara mereproduksi `ens_r1mid_team.npy` **persis**, maksimum selisih 0.
Menggunakan prior balancing dan evidence `infer2.py` mereproduksi CSV lokal
`sub_mine5_team_mid.csv` **byte-for-byte**. Ini menyelesaikan hambatan replay
probabilitas/postprocessing; training dan inference checkpoint intermediate
belum dijalankan ulang. Detail training tetap perlu dibedakan dari replay array.

Candidate fixed geometric blend 90% best Nafis +10% PVT-B3 kini tersedia dan
belum di-upload. Run B0 juga sudah selesai, dan kelanjutan B3 448px telah
diluncurkan sebelum deadline. Lihat [audit server dan opsi deadline](DEADLINE_OPTIONS_2026-11-17.md)
untuk hashes, komposisi, keterbatasan validasi dan status masing-masing.

## Follow-up final — 13:59 UTC

Listing `kaggle-orie` mencatat dua submission tambahan:

| Ref | Waktu UTC | File | Public macro F1 |
| --- | --- | --- | ---: |
| 58467983 | 12:08:18 | `R34_288_sift_submission.csv` | .87461 |
| **58470727** | **13:49:17** | **`final_domain.csv`** | **.90780** |

Best publik naik **+.02466** dari .88314. Private score belum tersedia.
Endpoint `submission-limits` mengembalikan **4 today / 19 lifetime / 0 remaining**.
Audit ini tidak mengirim submission atau mengubah file/proses rekan.
[Submission terbaru](https://www.kaggle.com/competitions/dateraquest-2-0-machine-learning-competition/submissions?select=58470727).

Kode `final.py` dan array di folder kerja Nafis membentuk dua kelompok:

- **A:** enam model whole-image / real-adapted, geometric mean setara:
  `r1_r18d_384`, `r1_r34d_320`, `r1_r18d_288`, `r1_r18d_352w`,
  `r1_fr_r18d_320`, `r2_fr_r18d_320w`.
- **B:** probabilitas tim dengan bobot .4, serta geometric mean empat model
  crop/high-resolution dengan bobot kelompok .6:
  `p_r18d_384_f0` @512, `p_r18d_384_full` @512,
  `p_r34d_352_full` @448, `p_r18d_448_full` @576.
- Pada **1,116 gambar berflag video**, geometric blend **A .5 / B .5**;
  pada **894 foto**, **A .2 / B .8**. Flag mengikuti `test_isvideo.npy` milik
  pipeline, bukan verifikasi manual domain setiap gambar.
- Prior balancing 100 iterasi, lalu log probabilitas ditambah frozen calibrated
  evidence. SIFT tim lama dengan ≥5 inlier memberi log-evidence minimum 4.

Perhitungan independen ini mereproduksi `final_domain.npy` **persis**, max
selisih 0, dan `final_domain.csv` **byte-for-byte**. File deliverable
`submission.npy` dan `submission.csv` juga identik dengan artefak final tersebut.
CSV berisi 2,010 baris dalam urutan ID resmi dan lolos validasi format.
SHA256 CSV: `59b9e62480311d8dfbdc41cf3329a7e5ec4a5726d619032d746127c2b3b13aa7`.
Kaggle tidak mengekspos hash file upload; asosiasi submission menggunakan
filename, waktu dan deskripsi. Training dan inference checkpoint model-model
ini **belum dijalankan ulang**. Angka proxy dari pasangan matching tidak menjadi
holdout independen, terutama untuk model full-train.

[Ringkasan dan hash input](../experiments/results/TEAM-UPDATE-2026-11-17-FINAL.summary.json).
Snapshot CLI dan skrip pemeriksaan lokal ada di
`outputs/DEADLINE-CHECK-2026-11-17/`. Baseline keputusan berikutnya **.90780**;
candidate 90:10 berbasis .88314 tetap tidak di-upload dan kini superseded.

Eksperimen kita [PVT448](../experiments/LOCAL-PROP-009-PVT448.md) juga selesai:
100 epoch, 1,000 update, matching-pair lokal **.871987**, clean **.912333**,
mean blur **.790971**. Gate lulus, tetapi standalone turun ke **.859817**.
Tidak ada public score checkpoint baru; metrik lokal ini tidak bisa dibandingkan
langsung dengan .90780 public milik ensemble tim.
