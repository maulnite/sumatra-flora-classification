# LOCAL-PROP-002 — Best team ensemble plus a fixed blur branch

Status: **completed; original strongest baseline retained; no submission**.

Kontrol adalah **EXP-024**, best public tim **0.87012**: mean ResNet18d train288/infer352 dan ResNet34d train224/infer320, masing-masing lima fold, 600 epoch, prior balancing, lalu SIFT ≥5 inliers. Artefak asli ditemukan di `~/inahuy/dateruy/`, sesuai `docs/CODE_MAP.md` branch `kaka`; source/weights/proses tim tidak diubah. CSV kontrol sudah direkonstruksi **dari probabilitas tim yang tersimpan** secara byte-for-byte dengan SHA-256 `6ca4cb673741edc3ca1e9903ed31dfb3b35924b52797742715a40e667f4d25e9`. Pada tahap LOCAL-PROP-002 ini, checkpoint baseline belum menjalankan inference ulang.

**Update 14 November:** [BASELINE-INFERENCE-REPRO-001](BASELINE-INFERENCE-REPRO-001.md) kemudian menjalankan ulang seluruh sepuluh checkpoint baseline. Keempat probabilitas OOF/test dan final CSV identik persis; preprocessing seluruh raw images juga pixel-identical dengan cache. Training dan pencarian match SIFT belum diulang.

Satu kandidat yang dibekukan: **90% probabilitas neural baseline + 10% model B LOCAL-PROP-001**, kemudian prior balancing dan SIFT yang sama. Ini menguji nilai tambahan model B untuk sistem kuat; tidak melatih ulang atau mengganti dua keluarga backbone best tim. Bobot 10% ditetapkan sebelum membaca skor kandidat dan tidak dicari ulang jika hasil gagal.

## Validation yang sebanding

Baseline memakai probabilitas OOF tim; tiap query diprediksi oleh checkpoint dari fold yang tidak memuat query. Untuk B, gunakan hanya validation lokal sebelumnya yang team group-nya tidak muncul dalam training B: **603 gambar, 30 kelas**, fold counts `[116,119,127,115,126]`. Sebanyak 31 dari 634 gambar lama dikeluarkan. Pada fold file aktual terdapat **2.708 distinct group IDs** dan 259 non-singleton groups; angka tersebut berbeda dari sebutan scene237 di laporan historis. Fold IDs dan ukuran `[614,607,603,600,590]` tetap berasal dari file tim asli.

Cache validation 224 dari eksperimen pertama tidak identik dengan cache tim. Probabilitas B dihitung ulang pada `cache/corr4r_224.npy` milik tim; probabilitas baseline memakai cache inference 352/320 yang telah dibekukan tim. Ini memastikan seluruh model mendapat versi corruption yang berasal dari cache protokol tim. Perbedaan pixel cache lama baru diketahui setelah kedua cache tersedia; hasil lama tetap dicatat sebagai protokol terpisah.

Metrik utama: Macro F1 **setelah prior balancing, sebelum SIFT**, dihitung atas 603 query yang sama. Prior tetap frekuensi kelas seluruh train, algoritme 100 iterasi yang sama, dan population balancing 603 query yang sama untuk A/B. Skor subset ini bukan full OOF 0.852564 tim. Full baseline OOF juga direproduksi sebagai pemeriksaan identitas artefak.

Metrik sekunder menerapkan stored top-one SIFT match hanya jika reference berada pada **fold lain** dari query. Match yang berasal dari validation fold query ditolak; reference alternatif tidak tersedia dalam CSV ini. **Angkanya hanya proxy**: source generator SIFT yang dipublikasikan memakai corruption/rotation seed berbeda dari primary cache, dan CSV tidak menyertakan query-image hashes untuk memastikan provenance view yang dipakai saat file dibuat. Ini belum merupakan evaluasi end-to-end SIFT pada view primary yang sama. Primary tidak memakai label transfer. Untuk test, galeri seluruh train valid sehingga aturan SIFT ≥5 baseline dipertahankan.

Paired bootstrap: team groups di dalam kelas, 2.000 replikasi seed43. Ini screening pada holdout lama, bukan validation baru yang independen. Klaim efektivitas dibatasi pada split dan cache tersebut.

## Kriteria sebelum skor kandidat dilihat

- Primary Macro F1 gain ≥0.002 terhadap kontrol kuat pada subset yang sama.
- Secondary fold-filtered SIFT Macro F1 tidak turun.
- Lower bound paired bootstrap 95% delta utama ≥−0.01; interval dilaporkan sebagai diagnosis, bukan klaim signifikansi otomatis.
- Tidak mengganti bobot, memilih checkpoint baru, atau membuka eksperimen training tambahan berdasarkan hasil.

Runner [evaluate_strong_baseline.py](../scripts/evaluate_strong_baseline.py) memakai [config](../configs/LOCAL-PROP-002.yaml). File tim dibaca saja; snapshot file kecil, manifest, hashes, probabilitas, dan hasil ditulis ke `outputs/LOCAL-PROP-002/`. Model B tetap checkpoint EMA epoch300 dengan SHA-256 `e0359ef82591783ac02c63087d25f724d2cdf929b3fd2170ec15ac7ed6112e30`.

GPU sharing telah diotorisasi; inference satu model memakai batas 20% memori GPU. Tidak ada training baru. Command evaluasi tidak melakukan upload; CSV kandidat hanya dibuat jika kriteria lolos.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u scripts/evaluate_strong_baseline.py
```

## Hasil

Selesai 13 November 2026, 11:36:44 UTC. [Ringkasan JSON](results/LOCAL-PROP-002.summary.json) mencatat metrik, provenance, checks, dan keterbatasan. Tidak ada training baru atau pencarian bobot; 19 unit tests lulus.

| Pemeriksaan baseline | Hasil |
| --- | ---: |
| Full neural OOF raw | 0,849455 |
| Full neural OOF + prior balance | **0,852564** |
| Public baseline yang telah diverifikasi | **0,87012** |
| Reproduksi final submission baseline | **Identik byte-for-byte** |
| Label test yang diganti SIFT di baseline | 29 |

| Macro F1 pada shared holdout 603 gambar | Kontrol best | +10% B | Delta |
| --- | ---: | ---: | ---: |
| **Primary: neural + prior balance** | **0,852673** | 0,850512 | **−0,002160** |
| Fold-filtered SIFT proxy | 0,854148 | 0,852007 | −0,002142 |

Paired group bootstrap 95% delta utama **[−0,007368; +0,002874]**. Interval mencakup nol; satu screening ini belum membuktikan arah efek generalisasi. Namun kandidat tidak mencapai kenaikan ≥0,002 yang dibekukan sebelumnya. **Keputusan: pertahankan EXP-024 asli. Tidak membuat candidate submission CSV atau mengirim submission baru.**

Campuran mengubah enam prediksi validation: satu kesalahan diperbaiki, dua prediksi benar menjadi salah, dan tiga perubahan tetap salah. Akurasi berubah dari 519/603 menjadi 518/603. Pemeriksaan coherence cache pada seluruh 603 gambar menggunakan ringkasan 16×16 menunjukkan korelasi minimum >0,9999 antara cache tim224 dan tim320/352, dengan median absolute difference 0,015625 pada skala piksel 0–255; view antar-resolusi tim konsisten secara visual/statistik. Pemeriksaan ini tidak memakai label untuk memilih gambar.

Model B sendiri pada view tim yang dihitung ulang memperoleh Macro F1 0,710698. Skor ini berbeda dari 0,723461 eksperimen pertama karena validation IDs dan cache berubah. Angka full OOF, shared holdout, dan public leaderboard tidak dipertukarkan sebagai bukti efek augmentasi.

Gabungan akan mengubah 63 label test dari baseline setelah SIFT, tetapi label test tidak tersedia; jumlah perubahan tidak menjadi bukti perbaikan. Bobot tetap 10% setelah hasil diketahui. Probabilitas disimpan untuk audit, sedangkan CSV kandidat tidak dibuat karena gate gagal.

Artefak lokal: `baseline_verified.json`, `evaluation_manifest.csv`, `specification.json`, `inference_identity.json`, `extra_probabilities.npz`, `validation_predictions.npz`, `result.json`, serta `postrun_audit.json` di `outputs/LOCAL-PROP-002/`. Semua file asli tim tetap utuh.
