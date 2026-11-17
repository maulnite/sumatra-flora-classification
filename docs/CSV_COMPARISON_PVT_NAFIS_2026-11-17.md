# Perbandingan CSV PVT dan Nafis final-domain — 17 November

Permintaan pengguna: bandingkan hasil terbaik kita dengan `final_domain.csv`
Nafis. CSV public-best kita yang sudah ada mendapat **public .87197**. Untuk
hasil PVT448 terbaru, inference final epoch-100 EMA dijalankan juga, lalu hanya
checkpoint PVT pada resep CSV sebelumnya yang diganti. Sembilan member ResNet,
bobot total PVT .1, prior100, dan SIFT≥5 tetap sama. Ini bukan blend baru dengan
ensemble Nafis. Kandidat PVT448 **belum memiliki public score**.

Angka **.912333 adalah validation clean** dari pasangan PVT448/R34d, sedangkan
**.871987 adalah validation utama** pasangan tersebut. Keduanya berasal dari
603 query lokal, berbeda dari public Kaggle pada test. Skor **.90780** milik
CSV final-domain Nafis. Jangan memperlakukan .912333 sebagai public score PVT.

## Hasil pada 2,010 ID test

| CSV kita dibanding final-domain Nafis | Prediksi sama | Prediksi berbeda | Persentase sama |
| --- | ---: | ---: | ---: |
| Public-best PVT sebelumnya, public .87197 | 1,680 | 330 | 83.58% |
| Kandidat PVT448 terbaru, public belum tersedia | 1,689 | 321 | 84.03% |

PVT448 mengubah **72** prediksi dari CSV kita sebelumnya: **31** menjadi sama
dengan Nafis, **22** menjadi berbeda, dan **19** berubah tetapi tetap berbeda.
Ini menunjukkan perubahan agreement, bukan perubahan akurasi; label test resmi
tidak tersedia. Menentukan label yang benar atau manfaat ensemble dari jumlah
disagreement saja belum memungkinkan.

Mengikuti flag domain Nafis, kandidat terbaru berbeda pada **121/1,116 video**
(10.84%) dan **200/894 foto** (22.37%). Flag tersebut berasal dari pipeline.
Perbedaan kelas paling sering pada kandidat terbaru: PVT memprediksi
**Echinodorus_radican**, Nafis **Portulaca_Sp**, sebanyak **12 gambar**; lalu
PVT **Nepenthes_reinwardtiana_Miq**, Nafis **Echinodorus_radican**, **8 gambar**.
Ini adalah arah perbedaan prediksi, bukan koreksi ground truth.

## Pemeriksaan dan artefak

Inference **352px, batch16, flip TTA**, sesuai validation. CSV public-best
sebelumnya direplay byte-identik sebelum mengganti model. Inference validation
PVT448 mereproduksi array final tersimpan **persis**, max selisih 0, tanpa label
berubah. Pemeriksaan source hashes sebelum/sesudah inference lulus. Semua CSV
berisi ID resmi lengkap, urutan dan label valid. Tidak ada training atau upload
Kaggle pada pekerjaan ini. Inference/persiapan selesai sekitar **49 detik**.

- Config: [LOCAL-PROP-009-CSV-COMPARE.yaml](../configs/LOCAL-PROP-009-CSV-COMPARE.yaml).
- Script: [compare_pvt448_final_domain.py](../scripts/compare_pvt448_final_domain.py).
- [Receipt terstruktur](../experiments/results/LOCAL-PROP-009-CSV-COMPARE.summary.json)
  mencatat hashes, replay, pasangan kelas dan hitungan domain. Sesudah inference,
  nama kolom rendering diperjelas menjadi `ours` dan `reference`; renderer diuji
  pada ketiga pasangan CSV dengan hitungan identik, tanpa mengulang GPU.
- Kandidat: `outputs/CSV-COMPARE-PVT-NAFIS/PVT448/LOCAL-PROP-009-PVT448_f2replace_candidate.csv`.
  SHA256 `91ca542f377879373afc63b9377695b0bb851c542ac0cc9a146a9c851e9e4639`.
- Daftar 321 perbedaan, termasuk nama spesies:
  `outputs/CSV-COMPARE-PVT-NAFIS/PVT448/PVT448_vs_nafis_differences.csv`.
- Folder yang sama menyimpan semua pasangan label, tabel transisi kelas,
  probabilitas dan log inference. CSV/probabilitas tetap lokal, tidak dipublikasi.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
NO_ALBUMENTATIONS_UPDATE=1 /home/nafisnaufal1426/venvs/dq/bin/python \
  -m scripts.compare_pvt448_final_domain
```
