# LOCAL-PROP-004 — Domain adaptation dengan test tanpa label dan retrieval

Status: **proposal riset; belum training, belum implementasi metode, belum submission**. Menggantikan prioritas LOCAL-PROP-003 setelah user meminta arah dengan potensi dampak lebih besar. Hasil di bawah adalah diagnosis baseline, bukan hasil kandidat.

Update eksekusi: [LOCAL-PROP-004-PILOT](LOCAL-PROP-004-PILOT.md) mengimplementasikan tahap embedding/retrieval/graph dan dimulai di background pada 14 November 08:59 UTC. Kode/config dibekukan sebelum skor;24 tests lulus. Metode training domain adaptation dan geometric matching belum diimplementasikan pada pilot ini. Hasil akhir tersedia otomatis di folder output setelah job selesai.

Rekomendasi utama: mengubah tujuan belajar menjadi **semi-supervised domain adaptation**, menggunakan official test images tanpa label, dengan pseudo-label adaptif, konsistensi weak/strong views, dan pembaruan teacher. Retrieval/geometri serta graph inference diuji sebagai komponen pendukung. Bobot awal dapat memakai checkpoint tim yang seluruhnya dilatih dari scratch; tidak memerlukan pretrained model eksternal.

## Yang benar-benar diketahui

Snapshot 14 November 2026 melalui `kaggle-orie` mengonfirmasi top public **0,99645** (`Tim Bul`, submission 00:22:29 UTC), sementara best tim **0,87012**. Metode tim teratas tidak diketahui. Skor tinggi tersebut bukan bukti bahwa mereka memakai retrieval, pseudo-label, ataupun teknik tertentu. Public hanya mencakup 30% test dan private tetap menentukan hasil akhir.

[Diagnostik CPU](results/LOCAL-PROP-004.research.json) memakai probabilitas baseline yang sudah [direproduksi exact dari checkpoint](BASELINE-INFERENCE-REPRO-001.md):

| Bukti | Hasil | Implikasi / batas klaim |
| --- | ---: | --- |
| Dua backbone sama-sama salah top-1 pada OOF | **399/3.014** | Memilih di antara dua top-1 saja tidak mengatasi semua error |
| Jawaban benar di luar top-5 mean ensemble | **176/3.014** | Perubahan ranking lima kandidat juga mempunyai batas; angka ini bukan Macro F1 |
| OOF: backbone sepakat dan mean confidence≥0,8 | **1.330/1.332 benar, 99,85% precision** | Sumber seed potensial; bukan bukti precision pada test |
| Test dengan filter yang sama | **521/2.010, seluruh 30 kelas terwakili** | Ada target-domain images untuk self-training; pseudo-label belum terverifikasi |
| Test dengan agreement dan confidence≥0,95 | **1 gambar** | Fixed high threshold praktis membuang hampir seluruh target pool |
| SIFT tim: test dengan ≥5 inliers | **393/2.010, 19,55%** | Coverage saat ini terbatas; akurasi test match tidak diketahui |
| SIFT mengubah label pada baseline best | **29 gambar** | Memperkuat matcher saja belum mempunyai bukti cukup untuk menutup gap besar |

OOF menggunakan satu held-out checkpoint per keluarga, sedangkan mean test memakai lima per keluarga. Diagnosis khusus fold 0 juga dihitung: agreement+confidence≥0,8 benar pada **283/284 query validation**, dan memilih **536 test images** menggunakan **hanya teacher fold 0**. Ini menunjukkan bahwa seed pool tidak bergantung pada teacher dari fold yang sudah melihat label validation. Tetap tidak mengonfirmasi label test yang sebenarnya.

## Dasar metode dari sumber primer

**AdaMatch** menyatukan semi-supervised learning dan domain adaptation: prediksi weak target views menjadi pseudo-label untuk strong views, disertai distribution alignment dan relative confidence threshold. Paper juga mengevaluasi pelatihan dari random initialization. Threshold relatif relevan ketika skala confidence train/test berbeda; alignment dapat merugikan jika prior kelas kedua domain berbeda. Ini alasan metodologis untuk mengujinya, tanpa memindahkan angka benchmark paper menjadi proyeksi skor kompetisi. [Berthelot et al., AdaMatch](https://arxiv.org/abs/2106.04732).

**SoftMatch** menyediakan alternatif pembobotan pseudo-label berbasis distribusi confidence, menggunakan truncated Gaussian dan penyeimbangan bobot kelas. Ini kandidat ablation jika hard selection terlalu membatasi coverage; tidak otomatis digabung dengan semua bagian AdaMatch dan tidak diklaim sebagai metode yang sudah diuji di sini. [Chen et al., SoftMatch](https://arxiv.org/abs/2301.10921).

**LaplacianShot** menggabungkan informasi klasifikasi/prototipe dan hubungan antartetangga pada embedding query untuk transductive inference tanpa melatih ulang backbone. Untuk proyek ini, memakai classifier probabilities sebagai unary, seed confidence, atau geometric edges adalah adaptasi yang perlu diuji; bukan reproduksi persis paper. Paper few-shot tersebut tidak membuktikan bahwa graph tanaman kita mempunyai neighborhood yang bersih. [Ziko et al., Laplacian Regularized Few-Shot Learning](https://arxiv.org/abs/2006.15486).

**ASIFT** mensimulasikan affine views sebelum SIFT untuk menangani perubahan sudut pandang yang lebih luas. Layak sebagai pengembangan matcher tanpa pretrained weights. Ia tetap membutuhkan struktur gambar yang dapat dicocokkan; tidak mengidentifikasi spesies baru ketika test tidak mempunyai referensi visual yang sesuai. [Yu & Morel, ASIFT](https://www.ipol.im/pub/art/2011/my-asift/).

## Apa yang berbeda dari eksperimen distillation kaka

Source server `distill.py` untuk `D_r18d_288_f0` sudah memakai seluruh test dengan fixed soft targets, KL loss, serta random rotation/flip. Ini pekerjaan tim yang berguna, tetapi bukan pembuktian kandidat berikut:

| Bagian | Distillation server yang diperiksa | Kandidat yang akan diuji |
| --- | --- | --- |
| Target | Semua fixed soft probabilities | Pseudo-label dipilih/dibobot menurut reliabilitas; diperbarui saat teacher membaik |
| Target views | Meniru prediksi asli pada rotated/flipped test | Weak-view target dan strong-view consistency dengan corruption yang dibatasi |
| Domain / confidence | Tidak ada adaptive confidence mechanism | Relative threshold, domain alignment yang dievaluasi, audit coverage per kelas |
| Informasi antargambar | Tidak dipakai | Graph/retrieval sebagai komponen terpisah bila pilot mendukung |
| Teacher validation fold 0 | Mean semua lima fold | Teacher hanya fold 0 ketika mengevaluasi fold 0 |
| Pembanding | Baseline validation 288 | Baseline train 288/infer 352 dan ensemble R34d320 yang direproduksi, query/protokol sama |

Teacher distillation lama mencakup model fold lain yang dilatih memakai label validation fold 0. Karena itu skor student tersebut tidak dijadikan estimasi leakage-free untuk kandidat baru. Tidak mengubah atau melanjutkan proses/file training tim.

## Rangkaian eksperimen yang diusulkan

**Tahap pertama: diagnosis tanpa training backbone.** Ekstrak embedding train/test dan query validation memakai checkpoint yang benar untuk fold 0 dan1. Ukur nearest-neighbor purity, agreement lintas backbone/transformasi, precision/coverage seed, dan konflik match geometri. Uji graph inference dengan feature kNN dan unary classifier; bandingkan dengan classifier serta nearest-neighbor sederhana. Geometric matches perlu mutual matching, RANSAC, spatial coverage dan margin terhadap kelas alternatif, tidak cukup hitungan inlier saja. Jangan menyalurkan satu seed yang bertentangan ke seluruh connected component.

Candidate retrieval memakai multi-scale RootSIFT dan affine views pada kandidat teratas untuk membatasi biaya. Tidak memakai LoFTR, SuperPoint, LightGlue, DINO, atau encoder pretrained eksternal. Threshold/matcher diuji pada train-derived positives/negatives; label test tidak diperiksa atau diberikan manual. Jika graph/retrieval tidak menunjukkan tambahan yang meyakinkan, keluarkan komponen tersebut dari konfigurasi training pertama.

**Tahap kedua: pilot domain adaptation yang terkontrol.** Bandingkan matched control dengan kandidat adaptive self-training pada fold 0, dengan tujuan pilot **+0,01 Macro F1** sebagai sasaran screening, bukan perkiraan hasil. Kandidat memakai supervised source loss serta confidence-masked target consistency, dengan target-loss warmup. Kontrol mendapat jumlah update dan source/target forward yang sama, tetapi target consistency loss dimatikan; pengaruh mixed BatchNorm harus sama agar gain tidak salah diatribusikan. Baseline EXP-024 asli juga tetap menjadi pembanding.

Point awal konfigurasi yang perlu dibekukan setelah diagnosis: student ResNet18d train 288/infer 352, sumber supervised official train di luar held-out fold, target unlabeled official test, teacher awal dua keluarga milik fold yang sama; sekitar 100 epoch dengan checkpoint akhir yang telah ditetapkan. Relative threshold AdaMatch 0,9 dapat menjadi titik awal berbasis paper, bukan threshold 0,95 absolut. Initial ensemble agreement memberi diagnosis seed; jangan menganggap threshold 0,8 hasil screening ini sebagai parameter yang sudah independen tervalidasi. Pilih satu formulasi utama, catat perubahan terhadap paper, dan hindari mencampur banyak algoritme tanpa ablation.

Target images sudah corrupt. Strong views dibatasi agar tidak menghilangkan identitas lagi; tidak menerapkan corruption v4 penuh dua kali. Supervisi train dan per-class monitoring menjaga model tidak hanya mempelajari kelas yang mudah atau menguatkan pseudo-label yang salah. Class weighting bukan bukti bahwa prior test uniform. Mekanisme confidence/label expansion diputuskan memakai development data dan dibekukan sebelum hasil final.

**Tahap ketiga: konfirmasi dan perluasan.** Jika pilot menunjukkan gain yang cukup besar terhadap kedua pembanding, evaluasi protokol identik pada fold 1 dan kemudian seluruh lima fold. Catat per-class F1, clean/blur guards, pseudo-label churn, calibration/coverage, dan paired group uncertainty. Semua teacher/graph gallery mengikuti aturan fold. Inference test dan submission baru hanya mengikuti hasil serta kriteria promosi yang dibekukan, kuota, dan izin submit bersyarat user yang sudah tersedia. Target 0,99 tetap aspirasi; tidak ada jaminan kenaikan atau klaim hasil kandidat.

## Memisahkan dua bentuk evaluasi

1. **Primary strict group OOF:** seluruh held-out fold images dan labelnya dikeluarkan dari training, termasuk unlabeled adaptation pool. Source-trained teacher untuk fold tersebut juga tidak pernah melihat labelnya. Unlabeled real test dapat dipakai sebagai target domain. Bandingkan dengan baseline pada cache primary yang sama, sebelum SIFT. Full OOF baseline 0,852564 tetap acuan untuk protokol ini.
2. **Secondary transductive holdout:** bila query validation tanpa label ikut adaptation/graph untuk mensimulasikan penggunaan seluruh test pool, beri nama dan laporan terpisah. Label held-out tidak pernah dipakai untuk loss, pseudo-label generation, seed, atau pemilihan checkpoint. Skor ini tidak dipresentasikan sebagai strict image-held-out OOF.

Gallery retrieval validation hanya boleh berisi training fold. **Aturan ini juga berlaku saat membuat pseudo-label untuk real test yang dipakai adaptation pada fold tersebut**: semua sumber label teacher, SIFT, seed, dan graph harus mengecualikan held-out labels/images dari awal indexing. Tabel SIFT test asli dari full-train gallery tidak boleh langsung dipakai untuk adaptation validation karena dapat membawa label held-out secara tidak langsung. Memasukkan original clean dari query yang disintesis akan mengubahnya menjadi source-retrieval benchmark; itu tidak setara dengan klasifikasi scene-held-out. Hanya untuk final test inference di luar evaluasi held-out, seluruh official labeled train adalah gallery yang sah. Validasi kontrol harus memakai jumlah target inputs dan informasi transductive yang sama.

## Aturan, sumber, dan status

Rules dibaca ulang secara terautentikasi melalui `kaggle-orie competitions pages ... --content --format json`. Isinya mewajibkan official dataset, melarang external pretrained weights dan manual test labeling. Tidak ditemukan larangan eksplisit automatic pseudo-labeling pada isi yang diterima; ini interpretasi atas aturan yang tersedia, bukan endorsement khusus panitia. Semua target labels dalam proposal berasal dari algoritme, bukan anotasi manual. Website rules tidak dapat dibaca lewat public scraper; snapshot CLI lokal menjadi bukti yang dipakai.

Catatan paper dan raw API/CLI responses disimpan lokal di `outputs/RESEARCH-004/`; yang dipublikasikan hanya ringkasan diagnosis, rencana, dan links sumber. Tidak ada source pesaing, dataset tambahan, training baru, perubahan dependency, atau submission. Hiperparameter/budget/gate operasional lengkap belum dibekukan; proposal ini memilih arah riset dan langkah pembuktian pertama, bukan klaim runner sudah siap jalan.
