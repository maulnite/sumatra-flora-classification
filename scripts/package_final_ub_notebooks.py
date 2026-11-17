"""Validate and package the revised final notebooks without remote uploads."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import nbformat
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "59b9e62480311d8dfbdc41cf3329a7e5ec4a5726d619032d746127c2b3b13aa7"
NAMES = ["Nafis Naufal Rahman", "Orie Abyan Maulana", "M Kaka Chadafi F"]
NOTEBOOKS = ["DATERAQUEST_UB_Single_Model.ipynb", "DATERAQUEST_UB_Ensemble.ipynb"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/FINAL-UB-2026-11-17")
    parser.add_argument("--copy-to", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    manifest = json.loads((out / "build_manifest.json").read_text())
    execution = json.loads((out / "execution.json").read_text())
    verification = json.loads((out / "run_ub_single/verification.json").read_text())
    assert execution["status"] == "success"
    assert verification["agreement_with_teacher"] == 2010
    assert verification["raw_jpeg_inference_executed"]
    assert verification["submission_sha256"] == EXPECTED
    for filename, expected in manifest["input_sha256"].items():
        assert digest(filename) == expected, f"Original input changed: {filename}"
    shell = InteractiveShell.instance()
    notebooks = {}
    counts = {}
    for name in NOTEBOOKS:
        notebook = nbformat.read(out / name, as_version=4)
        nbformat.validate(notebook)
        assert all(member in notebook.cells[0].source for member in NAMES)
        assert "pensi lomba data" in notebook.cells[0].source
        assert "attachment:lambang_ub.png" in notebook.cells[0].source
        codes = [c for c in notebook.cells if c.cell_type == "code"]
        for cell in codes:
            assert cell.execution_count is not None, (name, cell.source[:100])
            assert not any(o.output_type == "error" for o in cell.outputs), name
            compile(shell.transform_cell(cell.source), name, "exec")
        counts[name] = len(codes)
        notebooks[name] = notebook
    # Preserve the original ensemble's complete code and historical outputs.
    original_archive = next(Path(p) for p in manifest["input_sha256"] if p.endswith("final-notebook.zip"))
    with zipfile.ZipFile(original_archive) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("IDTim_NamaTim.zip"))) as inner:
            original = nbformat.reads(inner.read("dateraquest_final.ipynb").decode(), as_version=4)
    old_codes = [c for c in original.cells if c.cell_type == "code"]
    new_codes = [c for c in notebooks[NOTEBOOKS[1]].cells if c.cell_type == "code"]
    assert len(old_codes) == len(new_codes)
    for old, new in zip(old_codes, new_codes):
        assert old.source == new.source and old.outputs == new.outputs
        assert old.execution_count == new.execution_count
    source_csv = out / "run_ub_single/submission.csv"
    assert digest(source_csv) == EXPECTED
    csv = pd.read_csv(source_csv)
    assert csv.shape == (2010, 2) and list(csv.columns) == ["id", "label"]
    assert csv.id.is_unique and csv.label.between(0, 29).all()
    shutil.copyfile(source_csv, out / "submission.csv")
    shutil.copyfile(out / "run_ub_single/verification.json", out / "verification.json")
    dependencies = ["numpy", "pandas", "opencv-python-headless", "torch", "torchvision",
                    "timm", "albumentations", "scikit-learn", "matplotlib", "nbformat",
                    "nbclient", "ipykernel"]
    versions = {name: importlib.metadata.version(name) for name in dependencies}
    (out / "environment.json").write_text(json.dumps({"python": sys.version, "packages": versions}, indent=2) + "\n")
    (out / "requirements-snapshot.txt").write_text("\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n")
    readme = """# Final DATERAQUEST — pensi lomba data

Universitas Brawijaya

1. Nafis Naufal Rahman
2. Orie Abyan Maulana
3. M Kaka Chadafi F

## Isi dan provenance

- `DATERAQUEST_UB_Single_Model.ipynb`: notebook utama, seluruh sel kode telah
  dieksekusi ulang untuk EDA, preprocessing, replay formula teacher, dan inference
  langsung dari 2.010 JPEG test dengan SATU ResNet34d (21.319.294 parameter).
- `DATERAQUEST_UB_Ensemble.ipynb`: notebook pendamping; kode dan output eksekusi
  asli tim dipertahankan untuk riwayat teacher 20 checkpoint, training, dan
  validasi. Sampul dan pengantar disesuaikan. Sel training ini tidak dijalankan
  ulang saat revisi; jalur server di sel historis mengikuti lingkungan asli tim.
- `submission.csv`: hasil inference student yang benar-benar dijalankan ulang,
  identik byte dengan `final_domain.csv` (Kaggle ref 58470727, public 0,90780).
- `verification.json`: bukti kontrak CSV, agreement, dan hash checkpoint.
- `environment.json` dan `requirements-snapshot.txt`: environment yang diuji.

Student merupakan hasil distillation hard target: checkpoint dasar ResNet34d
scratch tim dilanjutkan dengan kombinasi label train resmi dan pseudo-label
teacher untuk citra test. Pembuatan teacher tetap bagian dari biaya training.
Riwayat 600 epoch base dan 7 epoch student ditampilkan sebagai log historis.
Agreement 2.010/2.010 mengukur reproduksi teacher, bukan validasi independen.
Skor 0,90780 berasal dari submission Kaggle tersebut; tidak ada submission baru
atau klaim evaluasi baru selama revisi notebook.

## Dua paket

- `pensi_lomba_data_final.zip`: dua notebook beroutput, CSV, dan bukti verifikasi.
- `pensi_lomba_data_final_lengkap.zip`: isi yang sama beserta folder
  `single_model_runtime` untuk mengulang EDA dan inference student. Hanya SATU
  checkpoint deployment disertakan; teacher direplay dari probabilitas turunan.

Data kompetisi mentah tidak dimasukkan. Notebook ensemble merupakan rekaman
eksperimen lengkap tim dalam environment aslinya; runtime portabel difokuskan
pada notebook utama, bukan mengulang training 20 checkpoint di mesin lain.

## Mengulang notebook utama

1. Ekstrak paket lengkap, sehingga `single_model_runtime/` sejajar notebook.
2. Sediakan dataset resmi berisi `train.csv`, `class_mapping.csv`,
   `sample_submission.csv`, serta folder `train/` dan `test/`.
3. Gunakan environment Python dengan dependency pada snapshot dan GPU CUDA.
   Snapshot versi mencatat environment aktual, termasuk build CUDA PyTorch.
4. Jika data berada di luar `data/raw` project atau mount input Kaggle,
   set `DATERA_DATA_ROOT` ke folder dataset. Opsional `DATERA_ARTIFACT_ROOT`
   menentukan folder runtime dan `DATERA_RUN_DIR` folder keluaran.
5. Buka notebook utama dan Run All dari folder hasil ekstraksi. Flag
   `retrain_base=False` dan `redistill=False` adalah mode reproduksi default.
   Keluaran berada di `run_ub_single/submission.csv` dan `verification.json`.

Untuk benar-benar mengulang training, aktifkan flag yang dijelaskan dalam
notebook. Distillation ulang membutuhkan checkpoint base scratch tim melalui
`DATERA_BASE_CHECKPOINT`, atau jalankan ulang training base 600 epoch lebih dulu.
Checkpoint base dan seluruh checkpoint teacher tidak masuk paket runtime.
Run ulang training tidak dijanjikan byte-identik pada perangkat/environment lain.

EDA dan preprocessing tim dipertahankan. Lambang UB berasal dari situs resmi
dan disematkan offline sebagai attachment. Struktur penyajian mengacu pada
notebook Three Makasih yang diberikan, tanpa menyalin kode model atau hasilnya.

SHA256 CSV:
`59b9e62480311d8dfbdc41cf3329a7e5ec4a5726d619032d746127c2b3b13aa7`

Nama paket memakai nama tim yang diberikan. Bila portal mewajibkan format
`IDTim_NamaTim.zip`, tambahkan ID registrasi tim yang sebenarnya sebelum unggah;
ID registrasi tidak ditebak. Tidak ada notebook/paket yang diunggah otomatis.
"""
    (out / "README_FINAL.md").write_text(readme)
    common = [*NOTEBOOKS, "submission.csv", "verification.json", "build_manifest.json",
              "execution.json", "environment.json", "requirements-snapshot.txt", "README_FINAL.md"]
    archives = {}
    for complete in (False, True):
        filename = "pensi_lomba_data_final_lengkap.zip" if complete else "pensi_lomba_data_final.zip"
        paths = [out / name for name in common]
        if complete:
            paths += sorted(p for p in (out / "single_model_runtime").rglob("*") if p.is_file())
        with zipfile.ZipFile(out / filename, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in paths:
                archive.write(path, path.relative_to(out).as_posix())
        with zipfile.ZipFile(out / filename) as archive:
            assert archive.testzip() is None
            assert hashlib.sha256(archive.read("submission.csv")).hexdigest() == EXPECTED
        archives[filename] = {"sha256": digest(out / filename), "bytes": (out / filename).stat().st_size}
    receipt = {"team": "pensi lomba data", "members": NAMES, "executed_code_cells": counts,
               "original_ensemble_code_outputs_unchanged": True, "original_inputs_unchanged": True,
               "submission_sha256": EXPECTED, "archives": archives}
    (out / "package_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if args.copy_to:
        target = args.copy_to.resolve()
        target.mkdir(parents=True, exist_ok=True)
        # New deliverables only; never overwrite the supplied original archive.
        for name in [*common, *archives, "package_verification.json"]:
            dest = target / name
            if dest.exists():
                assert digest(dest) == digest(out / name), f"Refusing to overwrite different file: {dest}"
            else:
                shutil.copyfile(out / name, dest)
        receipt["delivery_directory"] = str(target)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
