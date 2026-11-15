# Landasan pemilihan transformer untuk training dari nol

Riset 15 November 2026, setelah LOCAL-PROP-007 selesai. Pertanyaan: arsitektur apa
yang punya bukti eksperimen mendekati skala data kita, dan apakah layak langsung
memakai model besar? Ini tinjauan terarah atas paper dan protokolnya, bukan review
sistematis seluruh literatur. Belum ada training transformer atau submission baru.

**Rekomendasi: prioritaskan keluarga PVTv2. PVTv2-B3 menjadi kandidat kapasitas
besar, dengan PVTv2-B0 sebagai pembanding ukuran; IDMM menjadi kandidat recipe
pretraining yang seluruhnya berasal dari data train resmi.** Ini keputusan
eksperimental berdasarkan kedekatan bukti, belum hasil pada DATERAQUEST.

## Kondisi yang harus dijelaskan oleh paper

Data resmi berisi 3.014 gambar train dan 30 kelas. Pada scene fold 2, yang benar-benar
tersedia untuk melatih model adalah **2.411 gambar**, dari **2.166 scene group**,
dengan **38–135 gambar per kelas**; 603 gambar / 542 group ditahan untuk validasi.
Angka ini dihitung dari `cache/scene_folds.csv` tim. Group terdeteksi bukan jaminan
identitas biologis yang berbeda. Augmentasi tidak menambah scene independen.

Audit sebelumnya menemukan pergeseran JPEG, blur, warna, grayscale, dan occlusion
pada test; lihat [audit](DEEP_DATASET_AUDIT_2026-11-12.md). Aturan yang kita audit
mengharuskan bobot acak dan data resmi saja. Jadi bukti dengan hanya beberapa ribu
**label** tetapi jutaan gambar eksternal tanpa label tetap kurang cocok. Self-supervised
pretraining yang dimulai acak dan memakai foto train resmi saja tetap termasuk
training dari nol dalam pengertian tanpa bobot/data eksternal.

## 1. Bukti paling dekat: Training Vision Transformers with Only 2040 Images

[Cao, Yu, Wu, 2022, arXiv:2201.10728](https://arxiv.org/html/2201.10728), khususnya
Tabel 1 dan 3 serta §4.2. Flowers102 memakai **2.040 training / 6.149 testing,
102 kelas**. Kedua tahap IDMM dilatih hanya pada dataset target. Baris transfer
dari ImageNet atau SIN-10k dipisahkan dari rekomendasi ini.

Angka berikut adalah **accuracy Flowers102 yang dilaporkan paper**, bukan macro F1
kompetisi kita. Kolom 224px memakai 800 epoch supervised; baris IDMM menambah
800 epoch SSL pada 112px. Kolom terakhir menambah 100 epoch supervised di 448px.

| Backbone | Acak → supervised 224px | IDMM → supervised 224px | IDMM + tahap 448px |
| --- | ---: | ---: | ---: |
| DeiT-Tiny | 67,8% | 83,4% | 85,6% |
| DeiT-Base | 67,3% | 88,1% | 90,6% |
| PVTv2-B0 | 90,3% | 94,6% | 95,9% |
| PVTv2-B3 | 90,5% | 95,9% | 96,7% |

IDMM memakai klasifikasi identitas foto, multi-crop, CutMix, dan label smoothing,
kemudian supervised training. Hasilnya mendukung pemilihan **keluarga PVTv2 dan
recipe**, bukan janji bahwa model besar selalu menang. Selisih B0/B3 tanpa SSL
hanya 0,2 poin; setelah IDMM pada 224px selisihnya 1,3 poin. Budget antarkolom
berbeda, sehingga gain SSL tidak mengisolasi metode dari tambahan compute.

## 2. Dasar desain: PVT v2: Improved Baselines with Pyramid Vision Transformer

[Wang dkk., arXiv:2106.13797](https://arxiv.org/html/2106.13797), §3.3–3.5 dan
Tabel 1. Tokenization memakai convolution dengan patch yang overlap; feed-forward
network menyertakan depthwise convolution; resolusi mengecil dan channel bertambah
antarstage. Ini memberi dasar desain untuk menggabungkan informasi lokal dan
attention. Relevansinya untuk tekstur/struktur tanaman adalah **hipotesis kita**.

Paper arsitektur utamanya menguji skala ImageNet; bukti 2.040 gambar berasal dari
paper IDMM di atas. Jangan menyebut semua PVTv2 memiliki linear attention: pada
seri yang diperiksa, **B2-Li** adalah varian linear-SRA; B3 menggunakan SRA biasa.

Factory lokal `timm==1.0.31`, `pretrained=False`, classifier 30 kelas, berhasil
menginisialisasi **B0 3.417.470**, B2 24.865.246, dan **B3 44.741.086 parameter**.
Ini pemeriksaan CPU atas ketersediaan, belum benchmark GPU atau hasil training.
B3 adalah pilihan model lebih besar yang punya baris eksperimen langsung pada
skala data yang relevan; B2 dapat menjadi kompromi compute tetapi merupakan
interpolasi pilihan kita, bukan varian yang dilaporkan pada tabel IDMM tersebut.

## 3. Pembanding Swin: Efficient Training of Visual Transformers with Small Datasets

[Liu dkk., 2021, arXiv:2106.03746](https://arxiv.org/html/2106.03746), Tabel 1 dan 4.
Pada Flowers102 dengan 2.040 training, 6.149 testing, dan **100 epoch dari nol**,
paper melaporkan top-1 berikut:

| Arsitektur | Supervised baseline | Dengan dense relative localization |
| --- | ---: | ---: |
| CvT-13 | 54,29% | 56,29% |
| Swin-T | 34,51% | 39,37% |
| T2T-ViT-14 | 31,73% | 34,35% |

Loss tambahan memprediksi jarak posisi pasangan token. Penulis menunjukkan bahwa
ranking ImageNet tidak otomatis sama dengan ranking pada data kecil. Ini menurunkan
prioritas Swin sebagai pilihan awal kita. Namun 100 epoch dan recipe berbeda dari
IDMM; tabel antarpaper tidak merupakan pembandingan arsitektur yang terkontrol,
dan tidak membuktikan Swin-Base akan gagal. CvT di sini adalah **Convolutional
vision Transformer**, berbeda dari Compact Vision Transformer dalam paper CCT.

## 4. CCT: relevan secara desain, dengan klarifikasi split yang penting

[Hassani dkk., Escaping the Big Data Paradigm with Compact Transformers,
arXiv:2104.05704](https://arxiv.org/html/2104.05704) memberi landasan convolutional
tokenizer dan sequence pooling. Hasil CIFAR berasal dari 50.000 gambar training;
itu tidak setara dengan jumlah data kita.

Untuk Flowers102, tabel paper mencantumkan scratch **97,19%** dan pretrained
ImageNet **99,76%**. Dalam [klarifikasi penulis](https://github.com/SHI-Labs/Compact-Transformers/issues/65#issuecomment-1249532940),
split yang dipakai ternyata memiliki **6.552 train / 818 validation / 819 unlabeled
test**, berbeda dari split standar 2.040 train+val / 6.149 test. Penulis juga
menjelaskan bahwa evaluasi mereka memakai validation split. Oleh karena itu,
97,19% tidak boleh dijadikan bukti akurasi pada 2.040 training images.

Ada pula ketidakkonsistenan nama varian: tabel paper menyebut CCT-14/7×2 untuk
scratch, sedangkan README/config resmi yang diperiksa menyebut CCT-7/7×2 untuk
97,19%. Status ini dicatat, tidak diselesaikan dengan menebak. CCT tetap kandidat
sekunder, dengan kebutuhan reproduksi protokol. README dan config diperiksa pada
commit `cd11f0912749d376f7465cb83534e768b6701d7d`, tanpa mengunduh bobot atau dataset.

## 5. SCOTT + MIM-JEPA: menarik untuk transductive SSL, bukan bukti source-only yang sama

[Vélez-García dkk., Escaping the big data paradigm in self-supervised representation
learning, arXiv:2502.18056v2, 6 Maret 2026](https://arxiv.org/html/2502.18056v2),
khususnya Tabel 11–13. SCOTT memakai sparse convolutional tokenizer dan masked
prediction di latent space dengan EMA teacher yang dipelajari sendiri.

Pada Flowers102, Tabel 11 membedakan:

| Gambar yang tersedia saat SSL | Jumlah | Top-1 attentive probing |
| --- | ---: | ---: |
| Train+val standar | 2.040 | 80,69% |
| Train+val+test, tanpa label test | 8.189 | 97,15% |

Tabel 13 menunjukkan kenaikan saat ukuran model naik dari 6,5M ke 22,4M, tetapi
hasil itu memakai pool 8.189 gambar. Jadi manfaat scaling tersebut belum terbukti
pada hanya 2.040 gambar. Menarik untuk studi transductive terpisah jika protokol
kompetisi mengizinkan; validation kita tetap harus dikeluarkan dari seluruh SSL
untuk primary source-only. Teacher EMA internal bukan teacher pretrained eksternal.

## 6. SPT + LSA: dukungan mekanisme, relevansi jumlah data lebih rendah

[Lee dkk., Vision Transformer for Small-Size Datasets,
arXiv:2112.13492](https://arxiv.org/html/2112.13492), §4.1 dan Tabel 2. SPT memberi
informasi piksel tetangga saat tokenization; LSA mengubah distribusi attention.
Pada Tiny-ImageNet, modified Swin naik **60,87% → 64,95%**. Namun small-size di
sini mencakup CIFAR 50.000 dan Tiny-ImageNet 100.000 gambar, serta model dengan
konfigurasi yang disesuaikan. Ini mendukung pentingnya locality, bukan bukti langsung
untuk Swin standar pada 2.411 gambar.

## Keputusan yang bisa diuji

1. **Keluarga utama: PVTv2. Kandidat besar: B3, kontrol ukuran: B0.** Pasangan ini
   menjawab pertanyaan ukuran dalam keluarga yang sama dan punya bukti langsung
   dari eksperimen 2.040 gambar. Ini memperbarui rekomendasi awal MobileViT/Swin
   yang sebelumnya bertumpu pada mekanisme dan ketersediaan library.
2. Pisahkan pengaruh backbone dari recipe: supervised PVTv2 menjadi kontrol untuk
   PVTv2 + IDMM. Replikasi angka terbaik paper akan membutuhkan dua tahap dan
   budget yang jauh lebih besar; varian pendek harus disebut adaptasi/pilot.
   Kegagalan joint SupCon kita sebelumnya tidak menguji pretraining IDMM.
3. Tetap gunakan scene fold tim. Seluruh tahap, termasuk SSL dan perhitungan
   statistik yang di-fit, hanya boleh melihat source-training groups untuk
   primary validation. Laporkan standalone macro F1, ensemble tetap dengan R34d,
   serta clean/blur diagnostics. Perbedaan optimisasi, augmentation dan jumlah
   updates harus tercatat; samakan budget untuk klaim perbandingan ukuran.
4. Fold 2 adalah development fold yang telah berulang dipakai. Konfirmasi pada
   fold tambahan diperlukan sebelum menganggap gain akan bertahan. Pertahankan
   pembanding terkuat, bukan hanya membandingkan dengan kandidat yang lebih lemah.
5. Benchmark update GPU aktual sebelum menetapkan jumlah epoch penuh. Tidak ada
   training otomatis dari riset ini; tidak ada estimasi wall time yang disamakan
   dengan hardware paper atau janji public 0,99.

Angka paper adalah accuracy/top-1 pada dataset dan split masing-masing. Angka
tersebut **tidak dapat dikonversi menjadi prediksi macro F1 DATERAQUEST**, terlebih
dengan domain shift dan scene-group holdout kita. Rekomendasi di atas adalah
hipotesis yang memiliki landasan eksperimen lebih dekat, bukan jaminan skor.

[Ringkasan terstruktur](../experiments/results/RESEARCH-008-TRANSFORMER.summary.json)
menyimpan sumber, tabel, asumsi, temuan lokal, keputusan dan keterbatasan.
