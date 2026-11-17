# Update tim — 17 November 2026

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
