"""Package one executed review notebook, visible figures and exact predictions."""
from __future__ import annotations

import base64
import copy
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import zipfile

import markdown
import nbformat as nb
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/FINAL-UB-V2-2026-11-17'
EXPECTED='59b9e62480311d8dfbdc41cf3329a7e5ec4a5726d619032d746127c2b3b13aa7'
NAME='DATERAQUEST_UB_Lengkap.ipynb'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def figure_cell(path,caption):
    cell=nb.v4.new_markdown_cell(f'**{caption}**\n\n![{caption}](attachment:{path.name})')
    cell.metadata['final_ub_figure']=True
    cell.attachments={path.name:{'image/png':base64.b64encode(path.read_bytes()).decode()}}
    return cell


def render(notebook):
    body=[]
    for index,cell in enumerate(notebook.cells):
        if cell.cell_type=='markdown':
            text=cell.source.replace('<div align="center">','<div align="center" markdown="1">')
            for filename,mimes in cell.get('attachments',{}).items():
                for mime,data in mimes.items():text=text.replace('attachment:'+filename,f'data:{mime};base64,{data}')
            text=re.sub(r'\$\$.*?\$\$', '<p class="formula">Macro F1 = (1/30) × Σ [2 × precision × recall / (precision + recall)]</p>',text,flags=re.S)
            body.append(markdown.markdown(text,extensions=['tables','fenced_code','md_in_html']))
        elif cell.cell_type=='code':
            body.append(f'<details><summary>Kode sel {index+1} · eksekusi {cell.execution_count}</summary><pre><code>{html.escape(cell.source)}</code></pre></details>')
            for output in cell.outputs:
                if output.output_type=='stream':body.append('<pre class="output">'+html.escape(output.text)+'</pre>')
                elif output.output_type in ('display_data','execute_result'):
                    data=output.data
                    if 'image/png' in data:body.append('<img class="figure" src="data:image/png;base64,'+data['image/png']+'"/>')
                    elif 'text/html' in data:body.append(data['text/html'])
                    elif 'text/plain' in data:body.append('<pre class="output">'+html.escape(data['text/plain'])+'</pre>')
    return '''<!doctype html><html lang="id"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>pensi lomba data — DATERAQUEST UB</title><style>
    body{max-width:1120px;margin:40px auto;padding:0 28px;font:16px/1.65 system-ui,sans-serif;color:#18283a;background:#fff}
    h1,h2,h3{color:#003e77;line-height:1.3}h1{margin-top:48px;border-bottom:3px solid #d6aa48;padding-bottom:12px}
    img{max-width:100%;height:auto}.figure{display:block;margin:20px auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0;display:block;overflow:auto}
    th,td{border:1px solid #dce3e9;padding:8px 11px;text-align:left}th{background:#edf3fa}pre{overflow:auto;white-space:pre-wrap;background:#f5f7fa;padding:16px;font-size:12px;line-height:1.5}
    details{border:1px solid #dce3e9;border-radius:8px;margin:14px 0;padding:10px 14px}summary{cursor:pointer;color:#36566f;font-size:13px}
    .output{border-left:3px solid #d6aa48}.formula{text-align:center;background:#f6f8fb;padding:14px}
    a{color:#005da8}hr{border:0;border-top:1px solid #dce3e9;margin:28px 0}
    </style><body>'''+ '\n'.join(body)+'</body></html>'


def main():
    execution=json.loads((OUT/'execution.json').read_text());assert execution['status']=='success'
    run=OUT/'run_ub_single';verification=json.loads((run/'verification.json').read_text())
    assert verification['training_reexecuted']['distillation']
    assert verification['agreement_with_teacher']==2010 and verification['submission_sha256']==EXPECTED
    assert sha(run/'submission.csv')==EXPECTED
    csv=pd.read_csv(run/'submission.csv');assert csv.shape==(2010,2) and list(csv)==['id','label']
    notebook=nb.read(OUT/NAME,as_version=4)
    shell=InteractiveShell.instance()
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is not None
            assert not any(o.output_type=='error' for o in cell.outputs)
            compile(shell.transform_cell(cell.source),NAME,'exec')
    # Visible Markdown figures survive viewers that collapse code-cell outputs.
    replacements={
      '## 2. Inventaris, label, dan integritas ID':'## 1.2 Inventaris, label, dan integritas ID',
      '## 2.1 Integritas dataset dan ringkasan kelas':'# 2. Eksplorasi Data\n## 2.1 Integritas dataset dan ringkasan kelas',
      '## 3. Modul preprocessing yang digunakan tim':'## Modul pendukung preprocessing',
      '## 4. Implementasi recipe crop dan model scratch':'## Modul pendukung arsitektur dan recipe training',
      '## 5. EDA: kelas, contoh citra, dan statistik train–test':'## 2.4 Distribusi kelas, galeri raw, dan domain shift',
      '## 6. Pipeline augmentasi yang dipertahankan':'## 3.1 Pipeline augmentasi yang dipertahankan',
      '## 7. Sibling matching sebagai sumber evidence otomatis':'## 3.2 Pencocokan citra dan evidence otomatis',
      '## 8. Teacher ensemble dan asal target':'## 3.3 Formula pembentukan pseudo-label',
      '## 9. Base model: riwayat training scratch':'# 4. Base Model dari Bobot Acak',
      '## 11. Apa yang dipelajari student':'## 5.1 Data, target, dan objective student',
      '## 12. Satu model untuk seluruh test':'## 7.1 Satu model untuk seluruh test',
      '## 13. CSV: kontrak, agreement, dan hash':'## 7.2 CSV: kontrak, agreement, dan hash',
      '## 14. Ringkasan artefak dan batas klaim':'## 7.3 Ringkasan hasil yang diverifikasi',
    }
    captions={
      '01_class_dist.png':'EDA 1 — Distribusi 30 kelas train menurut flag dimensi',
      '02_train_samples.png':'EDA 2 — Contoh raw dari setiap kelas train',
      '03_test_samples.png':'EDA 3 — Variasi citra test raw',
      '04_train_vs_test.png':'EDA 4 — Pergeseran statistik kualitas train dan test',
      '05_pipeline.png':'Preprocessing 1 — Urutan transformasi recipe base/teacher',
      '06_aug_views.png':'Preprocessing 2 — Enam belas hasil augmentasi citra yang sama',
      '07_sibling_match.png':'Target training — Contoh pencocokan otomatis historis tim',
      '10_student_learning.png':'Training student — Kurva run baru pada revisi ini',
      '08_confusion.png':'Evaluasi — Confusion matrix subset teacher pada scene-fold0',
      '09_perclass_f1.png':'Evaluasi — F1 per kelas subset teacher pada scene-fold0',
    }
    groups={
      "for figure in ['01_class_dist.png'":list(captions)[:4],
      "for figure in ['05_pipeline.png'":['05_pipeline.png','06_aug_views.png'],
      "display(Image(filename=str(ARTIFACT_ROOT/'07_sibling_match.png')))":['07_sibling_match.png'],
      'import re':['10_student_learning.png'],
      'from sklearn.metrics import f1_score,confusion_matrix,accuracy_score':['08_confusion.png','09_perclass_f1.png'],
    }
    cells=[]
    for cell in notebook.cells:
        if cell.metadata.get('final_ub_figure'):continue
        if cell.cell_type=='markdown':
            for before,after in replacements.items():
                if after not in cell.source:cell.source=cell.source.replace(before,after)
            cell.source=re.sub(r'(?:# 2\. Eksplorasi Data\n)+','# 2. Eksplorasi Data\n',cell.source)
            if cell.source.startswith('### Implikasi EDA'):
                cell.source=cell.source.split('\n\n### Preprocessing base dan student secara eksplisit')[0]
                cell.source+='''

### Preprocessing base dan student secara eksplisit

| Tahap | Base scratch / recipe teacher | Student distillation / deployment |
|---|---|---|
| Pembacaan | JPEG resmi → OpenCV BGR → RGB uint8 | JPEG resmi → RGB uint8 |
| Train resize | RandomResizedCrop, base ResNet34d 352px | Squash INTER_AREA 448×448 |
| Augmentasi train | Rotasi ±180°, corruption-v4/light color, crop dan flip | Horizontal flip p=0,5 |
| Target | Label train resmi + pseudo-label matching otomatis | Label train resmi + pseudo-label teacher |
| Normalisasi | Mean [0,485; 0,456; 0,406], std [0,229; 0,224; 0,225] | Mean/std yang sama, setelah skala 0–1 |
| Inferensi final student | Tidak dipanggil dalam deployment student | Seluruh citra 448px, original + horizontal flip, rata-rata probabilitas |

Konstanta normalisasi tidak memuat bobot pretrained. Training base memakai
MixUp/CutMix dan EMA; distillation student menggunakan cross-entropy dengan
label smoothing 0,05 dan AdamW. Kedua tahap tidak disamakan dalam penjelasan.
'''
        cells.append(cell)
        if cell.cell_type=='code':
            if cell.source.startswith('import re'):
                cell.source=cell.source.replace("else ARTIFACT_ROOT/'training_history.json'", "else ARTIFACT_ROOT/'student_fresh_log.txt'")
                cell.source=cell.source.replace("log_path.read_text() if CFG['redistill'] else history['student_log']", "log_path.read_text() if log_path.is_file() else history['student_log']")
            for prefix,figures in groups.items():
                if cell.source.startswith(prefix):
                    # Figures are directly embedded once in the review copy.
                    cell.outputs=[o for o in cell.outputs if 'image/png' not in o.get('data',{})]
                    for filename in figures:
                        path=(OUT/'single_model_runtime'/filename) if filename.startswith('07_') else run/'figs'/filename
                        cells.append(figure_cell(path,captions[filename]))
    notebook.cells=cells
    notebook.metadata['distribution_variant']='executed complete review notebook with offline figures'
    nb.validate(notebook);nb.write(notebook,OUT/NAME)
    html_path=OUT/'DATERAQUEST_UB_Preview.html';html_path.write_text(render(notebook))
    runtime=OUT/'single_model_runtime'
    fresh=run/'runs/student_r34d_rebuilt'
    shutil.copyfile(fresh/'model.pt',runtime/'student_r34d_model.pt')
    shutil.copyfile(fresh/'log.txt',runtime/'student_fresh_log.txt')
    assert sha(runtime/'student_r34d_model.pt')==verification['student_checkpoint_sha256']
    manifest=json.loads((runtime/'manifest.json').read_text())
    manifest.setdefault('reference_student_weights_sha256',manifest['student_weights_sha256'])
    manifest['student_weights_sha256']=sha(runtime/'student_r34d_model.pt')
    manifest['student_distillation_reexecuted']=True
    manifest['distillation_training_not_reexecuted']=False
    manifest['base_training_not_reexecuted']=True
    manifest['original_outputs_preserved']=False
    manifest['artifact_sha256']={str(p.relative_to(runtime)):sha(p) for p in runtime.rglob('*') if p.is_file() and p.name!='manifest.json'}
    for filename,expected in manifest['input_sha256'].items():assert sha(filename)==expected,filename
    (runtime/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for filename in ['submission.csv','verification.json']:shutil.copyfile(run/filename,OUT/filename)
    readme='''# pensi lomba data — final revisi lengkap

Nafis Naufal Rahman · Orie Abyan Maulana · M Kaka Chadafi F
Universitas Brawijaya

`DATERAQUEST_UB_Lengkap.ipynb` adalah satu notebook utama yang memuat EDA visual,
audit raw data, preprocessing, augmentasi, source arsitektur/training, pembentukan
target otomatis, training student yang dijalankan ulang, evaluasi dan inference.
Sepuluh gambar disematkan sebagai attachment Markdown agar tampak langsung.
`DATERAQUEST_UB_Preview.html` menyediakan preview lengkap offline tanpa Jupyter.

Student menggunakan label train resmi dan pseudo-label dari teacher ensemble
20 checkpoint scratch tim. Bobot awal student berasal dari base ResNet34d scratch
tim. Riwayat base 600 epoch dicantumkan; base dan teacher tidak dilatih ulang
saat revisi ini. Distillation student benar-benar dijalankan ulang. Deployment
hanya menggunakan satu ResNet34d; biaya training tetap mencakup teacher.
Referensi validasi scene-fold0 merupakan subset teacher, bukan validasi student.

CSV yang dihasilkan run baru identik seluruh byte dengan final_domain.csv,
Kaggle ref 58470727, public 0,90780. Agreement bukan evaluasi independen terhadap
label test resmi. Tidak ada submission baru atau unggahan portal otomatis.

ZIP `pensi_lomba_data_FINAL_V2.zip`: satu notebook beroutput, HTML, CSV dan bukti.
ZIP `pensi_lomba_data_RUNTIME_V2.zip`: folder single_model_runtime berisi base
scratch, student dan cache turunan, serta environment snapshot. Ekstrak keduanya
ke folder yang sama untuk Run All. Dataset resmi disediakan terpisah. Set
DATERA_DATA_ROOT ke folder berisi train.csv, class_mapping.csv,
sample_submission.csv, train/ dan test/. Gunakan environment GPU sesuai snapshot.
Default redistill=True menjalankan training student kembali; False mereproduksi
inference cepat dari student baru yang dikemas. retrain_base=True mengulang
600 epoch dari bobot acak dan membutuhkan waktu beberapa jam. Pipeline ini
tidak diklaim sebagai raw-train-label-only: asal pseudo-label dijelaskan di notebook.

CSV SHA256: 59b9e62480311d8dfbdc41cf3329a7e5ec4a5726d619032d746127c2b3b13aa7
Nama paket belum memuat ID registrasi tim yang tidak diberikan; tambahkan ID
sebenarnya bila portal mewajibkan pola IDTim_NamaTim.zip.
'''
    (OUT/'README_FINAL_V2.md').write_text(readme)
    environment=json.loads((ROOT/'outputs/FINAL-UB-2026-11-17/environment.json').read_text())
    environment['preview_renderer']={'Markdown':importlib.metadata.version('markdown')}
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    common=[NAME,html_path.name,'submission.csv','verification.json','execution.json','README_FINAL_V2.md','environment.json']
    archives={}
    for filename,paths in [
        ('pensi_lomba_data_FINAL_V2.zip',[OUT/p for p in common]),
        ('pensi_lomba_data_RUNTIME_V2.zip',[*sorted(p for p in runtime.rglob('*') if p.is_file()),OUT/'environment.json'])]:
        with zipfile.ZipFile(OUT/filename,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for path in paths:archive.write(path,path.relative_to(OUT).as_posix())
        with zipfile.ZipFile(OUT/filename) as archive:assert archive.testzip() is None
        archives[filename]=dict(bytes=(OUT/filename).stat().st_size,sha256=sha(OUT/filename))
    public=copy.deepcopy(notebook)
    for cell in public.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
        for filename in list(cell.get('attachments',{})):
            if filename in ['02_train_samples.png','03_test_samples.png','05_pipeline.png','06_aug_views.png','07_sibling_match.png']:
                del cell.attachments[filename]
                cell.source='*Galeri citra kompetisi tersedia pada notebook final lokal beroutput.*'
    public.metadata['distribution_variant']='source; aggregate figures retained, raw competition image galleries omitted'
    nb.write(public,ROOT/'notebooks/final_ub'/NAME)
    receipt=dict(verification=verification,execution=execution,offline_figures=10,
                 all_original_inputs_unchanged=True,archives=archives)
    (OUT/'package_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'experiments/results/FINAL-NOTEBOOK-UB-V2-2026-11-17.json').write_text(json.dumps(receipt,indent=2)+'\n')
    target=Path('/home/nafisnaufal1426/akucintakc/final-notebook/revisi_ub_v2');target.mkdir(exist_ok=True)
    for name in [*common,*archives,'package_verification.json']:shutil.copyfile(OUT/name,target/name)
    shutil.copytree(runtime,target/'single_model_runtime',dirs_exist_ok=True)
    print(json.dumps(dict(delivery=str(target),**receipt),indent=2))


if __name__=='__main__':main()
