# LOCAL-PROP-004-PILOT — Retrieval dan graph dari embedding baseline

Eksekusi pertama dimulai **14 November 2026, 08:59:56 UTC**, sebagai background job atas izin user. Ini tahap awal [LOCAL-PROP-004](LOCAL-PROP-004.md): **tanpa backbone training, SIFT/ASIFT matching, atau Kaggle submission**. Kode, config, dan input hashes dibekukan sebelum skor graph dilihat. Status aktual dan hasil akhir dicatat otomatis di folder output; laporan ini menjelaskan protokol, bukan klaim hasil kandidat.

## Pertanyaan

Apakah embedding dua model terbaik sudah mempunyai tetangga visual yang cukup bersih sehingga retrieval/graph dapat memperbaiki klasifikasi dan membantu menyediakan pseudo-label? Jika tidak, komponen graph perlu dikeluarkan atau ditinjau ulang sebelum training domain adaptation. Hasil negatif graph tidak membatalkan seluruh gagasan adaptive self-training.

## Fitur dan sumber label

- Gunakan hanya checkpoint `R_r18d_i288_e600_f0/f1` infer 352 dan `R_r34d_e600_f0/f1` infer 320, dengan source dan dependency versions yang telah diverifikasi pada [reference reproduction](BASELINE-INFERENCE-REPRO-001.md).
- Ekstrak global-average-pooled feature sebelum classifier; rata-rata float32 original/horizontal-flip features, lalu L2 normalization. Centering tiap keluarga dipasang pada **training gallery saja**, lalu kedua keluarga digabung dan dinormalisasi dengan bobot sama.
- Source gallery berisi dua view per training image: clean RGB squash dan frozen v4rot cache. Fold0 memakai 2.400 training IDs dan 614 query; fold1 memakai 2.407 training IDs dan 607 query. Source IDs/groups tidak beririsan dengan query.
- Official test menjadi unlabeled graph nodes. Source labels saja yang di-clamp. Test tidak mempunyai ground truth/label manual dan pseudo-labelnya tidak dikunci sebagai kebenaran.
- Seluruh unary query/test probabilities berasal dari file per-fold yang sudah direproduksi, **bukan mean teacher semua lima fold**. Bobot dan source labels held-out fold tidak boleh masuk melalui teacher, matcher, atau seed.
- Ekstraksi batch kecil BF16 dapat mempunyai rounding berbeda dari batch 128 referensi. Perbedaannya dicatat dan pemeriksaan toleransi luas menangkap kesalahan head/normalisasi; unary pembanding selalu memakai array referensi yang exact. Klaim exact probability reproduction tetap merujuk verifikasi sebelumnya.

## Perbandingan yang dibekukan

1. **Baseline neural**: mean dua keluarga model, prior balancing 100 iterasi.
2. **Retrieval murni**: cosine kNN, 15 tetangga dari **source image IDs yang berbeda**, dari kedua gallery views. Softmax similarity weights memakai temperature 0,1.
3. **Neural + retrieval**: campuran probabilitas 50:50, tanpa pencarian bobot.
4. **Primary graph**: graph hanya berisi labeled training views dan unlabeled test. Setiap query validation melakukan readout independen ke graph yang sudah dibangun, tanpa masuk ke graph atau memengaruhi query lain. Readout memakai reciprocal neighbors; query tanpa edge valid mempertahankan unary.
5. **Secondary joint transductive graph**: query validation ikut membangun graph sebagai nodes **tanpa label**. Ini dilaporkan terpisah dari primary image-held-out evaluation.

Graph memakai mutual 15-NN, edge temperature 0,1, row normalization, restart diffusion alpha 0,8 selama maksimum 50 iterasi/tolerance 1e-6. Source labels tetap one-hot pada semua update, isolated nodes mempertahankan unary. Hasil graph dicampur 50:50 dengan unary neural. Parameter ini adalah pilihan pilot yang dibekukan, bukan reproduksi persis LaplacianShot atau hasil tuning validation.

Label query diberikan kepada evaluator **sesudah** graph/retrieval menghasilkan probabilitas. Source-view duplicate IDs ditangani saat menghitung diagnostik retrieval; graph memang mempunyai dua labeled view nodes per source image. Tidak menggunakan stored SIFT test table yang dibangun dari full-train gallery.

## Metrik dan kriteria

Primary metric: balanced ensemble Macro F1, sebelum SIFT. Pada skor per-fold, balancing memakai populasi query fold tersebut yang sama untuk semua pembanding. Skor pooled memakai **1.221 query fold 0+1**, dengan balancing dipasang ulang pada populasi pooled masing-masing sistem; ini bukan full 3.014-image OOF.

Target screening komponen graph: primary pooled gain **≥0,01** terhadap baseline dan delta setiap fold **≥0**. Ini sasaran keputusan pilot, bukan prediksi kenaikan leaderboard. Secondary joint graph tidak menggantikan primary bila hasilnya lebih baik. Tidak melakukan hyperparameter search atau mengubah bobot berdasarkan hasil.

Laporkan raw/balanced Macro F1, nearest-neighbor top1/top5 purity, seed precision/coverage pada agreement+confidence≥0,8 sebagai diagnosis, perubahan prediksi benar/salah, dan paired group bootstrap 95% (2.000 replikasi seed 44). Bootstrap memakai fixed balanced predictions dan tidak mencakup uncertainty training atau selection. Perubahan test labels tidak membuktikan perbaikan karena label test tidak tersedia.

## Background execution dan monitoring

Runner: [pilot_retrieval_graph.py](../scripts/pilot_retrieval_graph.py). Launcher/supervisor: [launch_retrieval_graph.py](../scripts/launch_retrieval_graph.py). Config: [LOCAL-PROP-004-PILOT.yaml](../configs/LOCAL-PROP-004-PILOT.yaml).

```bash
/home/nafisnaufal1426/venvs/dq/bin/python scripts/launch_retrieval_graph.py
```

Launcher memulai detached session dengan stdin tertutup dan log ke file, sehingga proses tidak bergantung pada terminal. Pemanggilan ulang mengembalikan job aktif; supervisor lock mencegah duplicate computation. Runner lock dan frozen specification menjaga resume hanya pada kode/config/input yang sama.

User sudah mengizinkan GPU sharing, termasuk saat GPU hampir penuh. Batch awal 8 dengan satu model aktif per giliran; limit allocator 20% bukan reservasi memori. Jika GPU OOM, batch turun hingga 1 lalu mencoba kembali setiap 20 detik tanpa menghentikan proses lain. Tidak berpindah ke CPU secara otomatis. Empat CPU threads dan empat image-preprocessing workers digunakan. Log mencatat jumlah image yang sudah diproses dan device aktual.

Supervisor memeriksa child process setiap 60 detik dan menulis heartbeat; setelah selesai menulis completion marker dengan exit code dan ringkasan gate. Monitor ini mencatat status di disk dan tidak mengirim pesan chat sendiri. Tidak ada git mutation atau remote upload otomatis dari background job.

Artefak di `outputs/LOCAL-PROP-004-PILOT/`:

- `job.json`, `heartbeat.json`, `status.json`, `runner.log`, `supervisor.log`.
- `specification.json`, clean caches milik proyek, per-checkpoint feature files, feature probability checks.
- `fold0_result.json`, `fold1_result.json`, per-fold predictions, lalu `result.json` dan `completion.json` ketika selesai.

Validasi sebelum launch: **24 unit tests lulus**. Tiga test baru memeriksa duplicate-view retrieval identities, clamped source/isolated graph nodes, dan readout yang tidak dipengaruhi query lain. Tidak ada dataset, checkpoint, feature arrays, atau predictions yang dipublikasikan ke GitHub.
