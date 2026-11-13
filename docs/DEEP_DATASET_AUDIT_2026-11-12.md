# Audit lanjutan dataset — 12 November 2026

Audit lokal pada data resmi di `data/raw/`, melanjutkan [audit awal](COMPETITION_AUDIT_2026-11-12.md). Seluruh 5.024 gambar diperiksa menggunakan fitur analitik, tanpa bobot pretrained atau dataset luar. Data asli, label, dan resource remote tidak diubah. Tidak ada model kompetisi yang dilatih atau submission yang dikirim.

## Temuan utama

1. **Ada foto satu scene/objek yang sangat mirip walaupun tidak ada duplikat byte/piksel identik.** Ambang geometris ketat menemukan 423 hubungan train–train, membentuk 213 kelompok non-singleton yang mencakup **562 dari 3.014 gambar train (18,6%)**. Contoh yang ditinjau visual menunjukkan foto beruntun dengan perubahan posisi, crop, atau sudut kecil.
2. **Split acak berpotensi memberi validasi optimistis.** Pada 10 split stratified 80:20, rata-rata **16,2% gambar validation** mempunyai group terdeteksi yang juga muncul di train. Dengan ambang lebih longgar, angkanya **25,2%**. Ini ukuran overlap scene, bukan besarnya kenaikan skor akibat leakage.
3. **Test mengalami perubahan kualitas selain JPEG.** Terukur 211 gambar hampir grayscale (10,5%) dan 563 kandidat gambar dengan kotak hitam interior (28,0%); sampel visual mendukung adanya occlusion buatan. Blur, perubahan warna/pencahayaan kuat, dan area berwarna seragam juga terlihat pada sampel. Jenis noise dan formula augmentasi asli belum dapat diidentifikasi secara unik.
4. **Belum ada salah label train yang terkonfirmasi.** Tidak ditemukan scene group lintas label pada ambang ketat maupun longgar. Ranking outlier diperiksa dengan konteks gambar; sejumlah kandidat muncul karena kemiripan tekstur/warna. Ini bukan bukti seluruh label botaninya benar.

## Metode pencarian kemiripan

- pHash 63 bit efektif (DC diabaikan), memakai delapan varian rotasi/flip. Hamming distance ≤8 menghasilkan **18 kandidat pasangan**: 7 train–train, 1 train–test, 10 test–test. Angka ini kandidat hash, bukan jumlah duplikat terverifikasi.
- SIFT maksimal sekitar 240 keypoints per gambar pada sisi panjang maksimal 448 piksel, diikuti normalisasi RootSIFT. Semua parameter berasal dari algoritme analitik.
- Retrieval memakai histogram warna, thumbnail, dan vocabulary 128 kata visual yang dipelajari **dari descriptor train saja** (sampel 40.000, seed 42). TF-IDF juga di-fit pada train saja.
- Tahap retrieval memeriksa **171.473 pasangan kandidat** dari seluruh split. Pemeriksaan lanjutan mencakup seluruh **164.253 pasangan train dalam label yang sama**, sehingga hubungan train tidak bergantung hanya pada shortlist retrieval. Kedua angka mencakup pemeriksaan yang beririsan dan tidak boleh dijumlahkan sebagai jumlah pasangan unik.
- Descriptor harus lolos Lowe ratio 0,75 secara mutual, kemudian homography RANSAC dengan threshold 3 piksel. Seed RANSAC ditetapkan deterministik per pasangan.

| Kriteria scene | Ketat | Longgar |
| --- | ---: | ---: |
| Minimum inliers | 16 | 10 |
| Minimum rasio inliers terhadap mutual matches | 0,50 | 0,35 |
| Minimum luas convex hull inliers / luas gambar, pada kedua gambar | 2,5% | 1,0% |
| Maksimum median reprojection error | 1,5 px | 2,0 px |

Threshold adalah pilihan operasional untuk audit ini, bukan ambang ilmiah yang sudah dikalibrasi pada data berlabel scene. Dua contoh sintetis dipakai sebagai sanity check: pasangan sama yang mengalami perspektif/rotasi/JPEG tetap cocok; scene tak berhubungan tidak dianggap cocok.

## Hasil grouping dan risiko validasi

| Hasil pada train | Ketat | Longgar |
| --- | ---: | ---: |
| Pasangan scene terdeteksi | 423 | 772 |
| Jumlah group termasuk singleton | 2.665 | 2.457 |
| Group dengan ≥2 gambar | 213 | 260 |
| Gambar dalam group non-singleton | 562 | 817 |
| Group terbesar | 10 gambar | 33 gambar |
| Group yang memuat lebih dari satu label | 0 | 0 |
| Mean validation yang group-nya juga ada di train, random 80:20 | 16,24% | 25,21% |
| Rentang pada 10 random split | 14,10–18,24% | 23,88–26,37% |

Contoh train yang ditinjau: `12_Echinodorus_radican/12_11.jpg` dengan `12_65.jpg`, serta `18_Monstera_Adansonii/18_73.jpg` dengan `18_104.jpg`. Nomor file yang berjauhan tetap bisa berhubungan; jangan memakai urutan ID sebagai pengganti grouping scene.

Pemeriksaan otomatis juga menemukan **84 hubungan train–test** pada ambang ketat, melibatkan **81 train dan 73 test**. Ini bukti adanya scene overlap pada subset yang terdeteksi; bukan estimasi lengkap overlap dataset. Tidak dilakukan pelabelan manual atau penulisan prediksi test. Galeri review pasangan berlabel hanya dibuat untuk train–train.

Group merupakan connected components: relasi A–B dan B–C bisa menyatukan A/B/C walaupun A–C tidak langsung cocok. Background bersama, pola berulang, atau papan tanaman bisa menyebabkan false positive; perubahan sudut besar, blur, occlusion, atau tekstur minim bisa menyebabkan false negative. Group ini belum mewakili identitas biologis setiap tanaman secara pasti.

### Kandidat split yang disimpan

`outputs/deep_audit/provisional_train_scene_groups.csv` memuat 3.014 baris:

- `id,label`: metadata resmi train.
- `strict_group,broad_group`: grouping dari dua ambang.
- `provisional_fold`: StratifiedGroupKFold lima fold untuk strict group.
- `provisional_broad_fold`: versi broad group untuk pemeriksaan sensitivitas.

Seed 42. Kelima fold strict berisi **609, 603, 598, 608, dan 596** gambar validation; masing-masing mencakup seluruh 30 kelas. Setiap group terdeteksi hanya berada pada satu fold. Pemeriksaan ini hanya menjamin tidak ada group terdeteksi yang terpecah; belum menjamin tidak ada foto satu scene yang terlewat oleh detektor.

## Perubahan kualitas selain JPEG

Pengukuran dilakukan setelah resize 256×256 untuk proxy kualitas, kecuali ukuran blockiness yang dihitung pada resolusi native. Kontrol memakai **300 gambar train**, 10 per kelas, dipilih seed 42. Masing-masing dikompresi ulang dalam memori dengan tabel kuantisasi JPEG dari test yang diambil acak; subsampling 4:2:0. File asli tidak ditimpa.

| Median ukuran | Train asli | Train + JPEG control | Test |
| --- | ---: | ---: | ---: |
| Brightness grayscale, 0–255 | 114,05 | 113,44 | 101,17 |
| Saturation, 0–1 | 0,371 | 0,372 | 0,396 |
| Variance Laplacian | 2.730,13 | 2.653,56 | 860,13 |
| Mean gradient | 113,74 | 112,14 | 76,66 |
| Entropy grayscale | 7,46 | 7,40 | 6,89 |
| Native block excess | 0,34 | 4,52 | 3,08 |

Kompresi kontrol menghasilkan pola blockiness yang lebih mendekati test, tetapi tidak mereproduksi sebagian besar perubahan ketajaman, brightness, dan entropy. Variance Laplacian rendah mendukung indikasi berkurangnya detail/blur; ia juga dipengaruhi isi scene, occlusion, resize, dan kontras. Hasil ini tidak mengisolasi penyebab setiap perbedaan maupun membuktikan mekanisme augmentasi tertentu.

### Deteksi artefak digital

| Flag otomatis | Train | Test |
| --- | ---: | ---: |
| Hampir grayscale: ≥99,5% piksel memiliki selisih max–min kanal RGB ≤2 | 0 | **211 / 2.010 (10,5%)** |
| Gambar dengan kandidat rectangle hitam interior | **1** | **563 / 2.010 (28,0%)** |
| Jumlah rectangle interior kandidat | 1 | 790 |

Rectangle berasal dari connected components piksel dengan seluruh kanal ≤6, minimum sisi 12 piksel, luas minimal 0,015% gambar, fill ratio ≥85%, aspect ratio ≤8, dan tidak menyentuh tepi gambar. Angka 563 adalah hasil detektor dan bukan jumlah cutout yang telah diverifikasi satu per satu. Galeri acak 24 positif (seed 42) ditinjau visual dan menunjukkan kotak buatan. Mask berwarna, rectangle yang menyentuh tepi, atau area sangat terdistorsi bisa terlewat.

Satu flag pada train adalah `7_Cuphea_hyssopifolia_kunth/7_9.jpg`: pemeriksaan visual menunjukkan **elemen UI screenshot telepon**, bukan cutout sintetis. Ini contoh mengapa gambar tidak boleh otomatis dihapus berdasarkan flag. Foto train juga memuat watermark kamera, misalnya `17_Lxora_acuminata/17_18.jpg`; keberadaannya belum dikuantifikasi di seluruh dataset. Statistik komponen hitam yang menyentuh border disimpan, tetapi tidak ditafsirkan sebagai jumlah padding karena bisa berasal dari background gelap alami.

## Pemeriksaan kandidat salah label

Semua train diranking berdasarkan ketidakcocokan tetangga BoW-SIFT dengan label asal, setelah mengecualikan group ketat gambar tersebut. Disimpan informasi tetangga sesama label, lintas label, vote top-10, dan gap similarity. **Ini diagnostic retrieval, bukan classifier spesies yang tervalidasi.**

Review visual mencakup galeri 30 kandidat yang diprioritaskan, konteks sesama/lintas label untuk 12 kelas, seluruh 106 gambar kelas 18 dan 74 gambar kelas 17, serta 12 contoh pasangan scene. Contoh: `18_24.jpg` mendapat banyak tetangga label 22 secara fitur, tetapi konteks sesama label `18_67.jpg` tampak sangat serupa. `13_74.jpg` juga terdorong ke kelas bertekstur ranting, padahal ada konteks sesama label yang serupa. Contoh tersebut memperlihatkan false alarm dari fitur tekstur.

Tidak ada relabel, penghapusan, atau daftar salah label terkonfirmasi. Audit ini belum memverifikasi identitas botani seluruh gambar atau memastikan satu nama kelas hanya berisi satu spesies yang tepat. Daftar kandidat disediakan untuk review ahli atau pemeriksaan OOF model from-scratch berikutnya.

## Implikasi untuk eksperimen

- Gunakan kandidat group split dan bandingkan ambang strict/broad; laporkan per-class F1 dan performa pada holdout train yang dikorupsi secara sintetis. Semua transformasi untuk training/validation berasal dari data resmi.
- Ablation augmentasi sebaiknya mencakup JPEG, grayscale, perubahan warna/brightness, blur, serta occlusion. Besar transformasi dan probabilitasnya tetap perlu dipilih melalui validasi; proporsi flag test tidak otomatis menjadi probabilitas training terbaik.
- Pertahankan baseline tanpa koreksi label. Evaluasi kandidat salah label menggunakan hasil OOF yang tidak melihat scene serupa dalam train sebelum mengambil tindakan.
- Nilai manfaat SIFT dengan kontrol neural-only pada split yang sama. Scene overlap mendukung relevansi investigasi fitur lokal, tetapi tidak menjamin kenaikan skor private.

## Artefak lokal dan reproduksi

Seluruh artefak diabaikan Git di `outputs/deep_audit/`. Artefak utama:

| File | Isi |
| --- | --- |
| `deep_audit_summary.json` | Ringkasan final grouping, kualitas, dan keterbatasan |
| `geometric_pairs.csv` | Evidence numerik 937 pasangan yang disimpan, termasuk yang hanya memenuhi shortlist pHash |
| `provisional_train_scene_groups.csv` | Group dan kandidat fold train |
| `train_scene_examples.jpg` | Contoh scene train–train untuk review |
| `train_visual_outliers.csv`, `train_outlier_contexts.jpg` | Ranking kandidat dan konteks visual train |
| `quality_comparison.json`, `quality_distributions.png` | Statistik dan plot train/JPEG-control/test |
| `digital_artifact_candidates.csv`, `digital_artifact_summary.json` | Flag per gambar dan rekap artefak |
| `test_quality_extremes.jpg`, `test_black_rectangle_candidates.jpg` | Review kualitas test tanpa label spesies |
| `features.npz`, `feature_config.json` | Cache fitur analitik dan konfigurasi |
| `manual_review.json` | Cakupan dan catatan review visual |

Environment yang dipakai: Python 3.10.4 yang telah tersedia, NumPy 1.24.1, scipy/scikit-learn/Pillow, serta OpenCV headless **4.8.1.78** yang diisolasi lokal. Tidak ada dependency bersama yang diganti. Empat worker digunakan; seluruh ekstraksi model-free dan pemeriksaan dilakukan di CPU.

```bash
python -m pip install --target outputs/audit_tools --no-deps opencv-python-headless==4.8.1.78

PYTHONPATH="$PWD/outputs/audit_tools" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python scripts/audit_dataset_deep.py --workers 4

PYTHONPATH="$PWD/outputs/audit_tools" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python scripts/audit_dataset_deep.py --expand-train --workers 4

PYTHONPATH="$PWD/outputs/audit_tools" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python scripts/audit_dataset_deep.py --scan-artifacts --workers 4

PYTHONPATH="$PWD/outputs/audit_tools" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m unittest discover -s tests -v
```

Delapan tes lulus, termasuk pasangan sintetis tertransformasi vs negatif tak berhubungan serta pembedaan rectangle interior, frame, dan grayscale. Gambar resmi bukan bahan tes unit. Statistik dan grouping berasal dari eksekusi penuh pada data resmi, bukan hasil sintetis. Tidak ada klaim skor model atau perubahan leaderboard dari audit ini.
