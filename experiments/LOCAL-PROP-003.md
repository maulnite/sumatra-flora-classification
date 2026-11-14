# LOCAL-PROP-003 — Fine-tuning akhir dengan MixUp/CutMix dimatikan

Status: **rencana, belum menjalankan training**. Rencana dibuat 14 November 2026 sesudah [reproduksi inference baseline](BASELINE-INFERENCE-REPRO-001.md) berhasil exact. [Config](../configs/LOCAL-PROP-003.yaml) adalah protokol rencana; runner baru belum diimplementasikan. Tidak ada klaim hasil kandidat.

## Pertanyaan dan alasan memilihnya

Apakah ResNet18d terbaik mendapat tambahan manfaat jika fase fine-tuning singkat di akhir tidak lagi mencampur dua gambar/target lewat MixUp/CutMix, sambil tetap belajar dari corruption v4?

Ini **hipotesis**, belum hasil: fase akhir tanpa batch mixing mungkin membantu pemisahan kelas ketika bobot sudah matang. Clean/corrupt image views tetap 50:50, termasuk JPEG, rotasi, color/grayscale, dan occlusion yang relevan dengan audit. Mematikan MixUp/CutMix berbeda dari mematikan `mix-corruptv4`.

Dasar keputusan yang sudah terverifikasi:

- Baseline EXP-024 memakai ResNet18d train 288/infer 352 dan ResNet34d train 224/infer 320, masing-masing lima fold; kedua keluarga dilatih 600 epoch. MixUp/CutMix masih aktif dengan probabilitas 0,5.
- [LOCAL-PROP-001](LOCAL-PROP-001.md) memperbaiki synthetic blur robustness, tetapi primary validation turun dan public submission 0,79702. [LOCAL-PROP-002](LOCAL-PROP-002.md) menambah 10% model tersebut ke best ensemble dan primary shared holdout turun 0,852673 → 0,850512. Tambahan blur/blend yang sama belum memberi alasan untuk diulang.
- Distillation server `D_r18d_288_f0` sekaligus menambah teacher loss dan mematikan mixing. Hasilnya tidak mengisolasi efek mixing, serta validation 288-nya tidak setara dengan infer 352 baseline. Teacher test probabilities juga menggabungkan fold lain yang dilatih pada label validation fold 0. Tidak dipakai sebagai bukti peningkatan yang independen.
- Pemeriksaan `kaggle-orie` pada 14 November masih menunjukkan best **0,87012**, ref 58278059, dan 11 submissions COMPLETE. Branch `kaka` tetap `b447792e3fc8ded59b98f8889023ab9f9b94c47e`. Tidak ada pembaruan branch atau submission baru sejak checkpoint replay.

## Satu perubahan yang diuji

| Komponen | A: kontrol tambahan training | B: kandidat |
| --- | --- | --- |
| Inisialisasi | Copy EMA epoch 600 dari fold yang sama | Identik A |
| Arsitektur/resolusi | ResNet18d, train 288/infer 352 | Sama |
| Fine-tuning | 30 epoch; final EMA epoch 30 | Sama |
| Optimizer | AdamW baru, LR 1e-4, WD 0,05 | Sama |
| Schedule | Warmup 1 epoch, cosine ke 1e-6 | Sama |
| Batch / label smoothing | 64 / 0,1 | Sama |
| EMA / dropout | Pengaturan tim, decay 0,999 / 0,2 | Sama |
| Clean/corrupt views | 50:50, `corrupt_v4`, rotasi ±180°, random resized crop | Sama |
| Probabilitas MixUp/CutMix | **0,5** | **0** |
| ResNet34d dan ensemble | Beku; rata-rata dua keluarga 50:50 | Sama |

`ckpt.pt` berisi optimizer/scheduler telah dihapus oleh runner tim ketika training selesai. Karena itu ini **fine-tuning dari bobot EMA dengan optimizer baru**, bukan exact resume urutan training 600. Kedua arm sama-sama mendapat reset optimizer dan tambahan 30 epoch. Perbandingan B–A mengisolasi perubahan mixing dalam fase baru; perbandingan B–baseline menguji manfaat terhadap sistem yang dipakai tim.

Seed fine-tuning **44+fold** untuk A/B, urutan sample dan seed augmentasi dipasangkan, dengan RNG sampling/augmentasi terpisah dari RNG mixing. Config, initial weight hashes, source, dependency versions, manifest, RNG state, learning curves, final EMA dan probabilitas disimpan di folder kita. Sebelum update optimizer pertama, loading initial checkpoint harus menghasilkan prediksi yang cocok dengan reference replay. Tidak mengubah file/proses tim atau memakai test images untuk training.

## Urutan eksekusi dan keputusan

1. **Screen fold 0:** train 2.400, validation 614, seluruh 30 kelas (minimum 10 per kelas). Jalankan A dan B masing-masing 30 epoch, satu run per giliran. Budget awal **60 epoch / 2.220 optimizer steps**. Ukur ETA setelah tiga epoch pertama; utilisasi shared GPU membuat estimasi historis belum dapat dijadikan janji runtime.
2. **Gate screen:** primary B harus naik **≥0,002** terhadap **baseline asli dan A**, serta lolos clean/blur guards. Jika gagal, hentikan proposal; jangan mengganti LR, epoch, blending weight, atau fold berdasarkan hasil ini. Jika A lebih baik, catat bahwa hipotesis mematikan mixing tidak didukung, tanpa otomatis mempromosikan A.
3. **Cek konsistensi fold 1:** lakukan pasangan A/B yang sama. Primary B harus tidak turun terhadap baseline asli maupun A dan tetap lolos guard. Ini pemeriksaan fold tambahan, bukan holdout yang belum pernah dipakai tim.
4. **Jika kedua tahap lolos:** jalankan A/B pada fold 2–4. Total maksimum **10 run × 30 epoch = 300 epoch tambahan**; tidak ada pengulangan training 600. Semua hasil fold dilaporkan, termasuk yang gagal.
5. **Promosi hanya setelah full 5-fold OOF:** primary B meningkat ≥0,002 terhadap baseline dan A; minimal 3 dari 5 fold membaik terhadap masing-masing pembanding; seluruh guards lolos. Setelah itu baru inference test semua lima checkpoint B dan gabungkan dengan keluarga ResNet34d asli pada bobot 50:50, prior balancing 100 iterasi, serta aturan/tabel SIFT referensi yang sama.

Skor test tidak diketahui sebelum submit. Jika semua gate full OOF lolos, gunakan izin user sebelumnya untuk satu submission bersyarat, setelah validasi CSV dan pemeriksaan kuota/baseline terbaru tim. Jika tim mempunyai baseline lebih kuat yang baru, rencana perlu ditinjau terhadap baseline itu sebelum submission. Satu fold yang naik belum memicu submission.

## Evaluasi yang dibandingkan

Primary adalah **Macro F1 ensemble setelah prior balancing, sebelum SIFT**, pada query dan populasi balancing yang sama untuk baseline, A, dan B. Setiap query OOF hanya memakai checkpoint held-out fold-nya dari masing-masing keluarga; checkpoint fold lain tidak boleh ikut memprediksi query tersebut. Memperbaiki satu fold bukan bukti full OOF naik.

Baseline fold 0 saat balancing dihitung **pada 614 query fold 0 saja** adalah **0,8714644561**. Angka ini berbeda protokol/populasi dari full OOF 3.014 query **0,8525643117** dan bukan public 0,87012. [Diagnostik baseline](results/LOCAL-PROP-003.planning.json) menyimpan skor tiap fold dari probabilitas yang telah direproduksi. Pada full OOF, balancing kembali dilakukan pada seluruh 3.014 query untuk setiap sistem, menggunakan prior train yang sama.

Secondary: single-model R18d F1, raw ensemble F1, per-class F1, pasangan kesalahan yang diperbaiki/prediksi benar yang rusak, dan confusion matrix. Model diuji pada raw clean images di squash 320/352 serta dua view stress: GaussianBlur 7×7 dengan sigma 0,6 dan 1,0, REFLECT_101, diterapkan setelah resize cache v4rot masing-masing model. Semua sistem memakai piksel yang sama; inference R34d pada view tersebut dibekukan. Guard adalah penurunan maksimum **0,01 Macro F1** pada clean dan pada mean kedua blur conditions, masing-masing dibanding baseline dan A. Guards memakai raw ensemble predictions sebelum prior balancing. Stress view ini bukan perkiraan pasti distribusi test.

Laporkan paired group bootstrap 95% dengan 2.000 replikasi seed 44 pada fixed balanced predictions; interval bersifat kondisional pada model dan balancing yang sudah dipasang, serta tidak mencakup training randomness/selection uncertainty. Interval adalah diagnosis, bukan syarat signifikansi atau bukti pasti perbaikan leaderboard. Fold/data ini sudah dipakai dalam eksperimen tim; final test tetap evaluasi eksternal.

Checkpoint yang dievaluasi untuk keputusan selalu **final EMA epoch 30**. Log epoch 0/10/20/30 dapat dipakai melihat kurva, tanpa memilih epoch terbaik. Tidak ada pencarian bobot ensemble, penggantian corruption, arsitektur baru, pseudo-label, atau teacher loss dalam proposal ini. SIFT matching belum dibangun ulang; skor SIFT OOF yang pernah diaudit tetap tidak menjadi primary metric.

Rencana ini melanjutkan bobot backbone terbaik yang sudah terverifikasi. Tambahan training, perubahan batch mixing, dan peningkatan terhadap baseline asli dapat dinilai secara terpisah dengan dua kontrol yang jelas.
