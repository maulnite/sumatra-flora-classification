# Final notebook Universitas Brawijaya

Tim **pensi lomba data**: Nafis Naufal Rahman, Orie Abyan Maulana,
dan M Kaka Chadafi F.

**Versi utama terbaru:** `DATERAQUEST_UB_Lengkap.ipynb`, satu notebook dengan
EDA visual, audit raw, preprocessing, source base scratch, training student
yang dijalankan ulang, evaluasi dan pemeriksaan CSV identik. Versi beroutput
dan HTML offline tersedia di `outputs/FINAL-UB-V2-2026-11-17/`; lihat
[catatan V2](../../docs/FINAL_NOTEBOOK_UB_V2_2026-11-17.md). Source V2 mempertahankan
grafik agregat, sementara galeri citra kompetisi hanya ada pada paket lokal.

```bash
python scripts/revise_final_ub_complete.py
python scripts/execute_final_ub_notebook.py \
  outputs/FINAL-UB-V2-2026-11-17/DATERAQUEST_UB_Lengkap.ipynb
python scripts/package_final_ub_complete.py
```

File di GitHub adalah **source tanpa output gambar kompetisi**. Versi beroutput,
CSV, runtime student, dan ZIP final disiapkan secara lokal di
`outputs/FINAL-UB-2026-11-17/`. Lambang UB disematkan sebagai attachment agar
sampul tidak memerlukan koneksi internet; sumbernya ada di
`assets/lambang_source.json`.

- `DATERAQUEST_UB_Single_Model.ipynb`: EDA dan preprocessing asli tim, teacher
  replay, riwayat training scratch/distillation, serta inference satu ResNet34d
  langsung dari JPEG raw. Semua 2.010 prediksi dan byte CSV terverifikasi sama
  dengan `final_domain.csv`, public 0,90780, ref 58470727.
- `DATERAQUEST_UB_Ensemble.ipynb`: source notebook teacher 20 model dari tim.
  Versi ZIP mempertahankan seluruh log dan output historis notebook asli.

Training student memakai label train resmi **dan pseudo-label teacher pada
test**, dimulai dari checkpoint scratch tim. Riwayat training dipertahankan
secara eksplisit sebagai hasil historis, bukan diklaim dijalankan ulang saat
revisi. Kesamaan dengan teacher bukan validasi student yang independen.

## Pembuatan pada server tim

```bash
python scripts/build_final_ub_notebooks.py \
  --member 'Nafis Naufal Rahman' \
  --member 'Orie Abyan Maulana' \
  --member 'M Kaka Chadafi F'
python scripts/execute_final_ub_notebook.py \
  outputs/FINAL-UB-2026-11-17/DATERAQUEST_UB_Single_Model.ipynb
python scripts/package_final_ub_notebooks.py
```

Builder membaca arsip dan artefak probabilitas/checkpoint asli di server tim;
path arsip, work directory, dan keluaran dapat dikonfigurasi. Data, checkpoint,
array prediksi, dan ZIP tidak disimpan di GitHub. Paket lengkap lokal menyertakan
satu checkpoint student dan cache turunan untuk menjalankan notebook utama.
Notebook teacher beroutput adalah rekaman eksperimen di environment aslinya.

Runtime utama dapat diarahkan dengan `DATERA_DATA_ROOT`,
`DATERA_ARTIFACT_ROOT`, dan `DATERA_RUN_DIR`. Default tidak mengulang training.
Training ulang tersedia lewat flag notebook dan membutuhkan checkpoint base
scratch atau 600 epoch training base lebih dulu.

Notebook Three Makasih yang diberikan pengguna hanya menjadi referensi susunan
sampul dan alur presentasi, tanpa penyalinan kode model/data/hasilnya.
