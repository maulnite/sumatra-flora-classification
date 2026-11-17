"""Build a single review notebook with visible EDA and a real distillation rerun."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

import nbformat as nb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/FINAL-UB-2026-11-17'
OUT = ROOT / 'outputs/FINAL-UB-V2-2026-11-17'
WORK = Path('/home/nafisnaufal1426/akucintakc/work')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def md(source):
    return nb.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source):
    return nb.v4.new_code_cell(textwrap.dedent(source).strip())


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = OUT / 'single_model_runtime'
    shutil.copytree(OLD / 'single_model_runtime', runtime, dirs_exist_ok=True)
    base = WORK / 'runs/p_r34d_352_full/model.pt'
    shutil.copyfile(base, runtime / 'base_r34d_model.pt')
    folds = np.load(runtime / 'scene_folds.npy')
    va = np.where(folds == 0)[0]
    np.savez_compressed(runtime / 'validation_reference.npz', indices=va,
        team18=np.load(WORK / 'team_R_r18d_i288_e600_rot2_oof.npy')[va],
        team34=np.load(WORK / 'team_R_r34d_e600_rot2_oof.npy')[va],
        crop=np.load(WORK / 'runs/p_r18d_384_f0/pval_512_rot2.npy'))
    manifest = json.loads((runtime / 'manifest.json').read_text())
    manifest['input_sha256'][str(base)] = digest(base)
    manifest['artifact_sha256'] = {str(p.relative_to(runtime)): digest(p)
        for p in runtime.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    manifest['revision'] = 'v2: raw EDA + actual new student distillation + exact CSV check'
    (runtime / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    notebook = nb.read(OLD / 'DATERAQUEST_UB_Single_Model.ipynb', as_version=4)
    old = copy.deepcopy(notebook.cells)
    old[1] = md('''
    ## Ringkasan metode dan panduan membaca

    Tujuan akhir adalah **satu ResNet34d** yang mengklasifikasikan citra flora
    langsung dari JPEG. Notebook ini memuat inventaris raw data, EDA visual,
    audit kualitas, preprocessing/augmentasi, kode arsitektur dan training,
    distillation, evaluasi, serta pembuatan dan pemeriksaan CSV.

    **Asal target training:** label train resmi digabung dengan pseudo-label
    otomatis teacher ensemble 20 checkpoint scratch milik tim. Student dimulai
    dari base ResNet34d scratch tim. Teacher dipakai pada tahap training;
    deployment hanya memakai satu student. Rincian formula target disertakan
    agar seluruh alur dan asal label dapat diperiksa.

    | Tahap | Data dan keluaran |
    |---|---|
    | 1. Raw data | Train 3.014, test 2.010, 30 kelas, pemeriksaan ID dan dimensi |
    | 2. EDA | Distribusi kelas, contoh 30 kelas, galeri test, statistik kualitas |
    | 3. Preprocessing | RGB, resize, rotasi, corruption, crop, flip, normalisasi |
    | 4. Training | Recipe base scratch dan **training student yang dijalankan ulang** |
    | 5. Evaluasi | Referensi scene-held-out teacher serta agreement student |
    | 6. Submission | Inference satu model dari raw JPEG dan verifikasi byte CSV |

    Mode eksekusi revisi ini mengulang **distillation student**. Training base
    600 epoch dan pembangunan teacher adalah hasil eksperimen tim sebelumnya;
    source/config/log-nya tetap dicantumkan. Skor Kaggle 0,90780 berasal dari
    submission ref 58470727, bukan skor validasi baru dari distillation.
    ''')
    old[2] = md('''
    # 1. Persiapan Environment dan Data Raw

    Dataset resmi tidak dikemas ulang. Runtime menyediakan checkpoint base
    scratch tim, checkpoint student referensi, probabilitas turunan teacher,
    fold, dan log. Dengan `redistill=True`, training student benar-benar berjalan
    kembali; `False` memakai student referensi untuk replay inference cepat.
    `retrain_base=True` tersedia untuk mengulang base 600 epoch dari bobot acak.
    ''')
    old[3].source = old[3].source.replace('redistill=False', 'redistill=True')
    old[3].source = old[3].source.replace("print('Mode:', 'historical training replay + actual raw-JPEG inference')",
        "print('Mode:', 'fresh student distillation + raw-JPEG inference' if CFG['redistill'] else 'reference student inference')")
    old[11].source = old[11].source.removeprefix('%%writefile eda_local.py\n')
    old[11].source = "Path('figs').mkdir(exist_ok=True)\n" + old[11].source
    old[12].source = "for figure in ['01_class_dist.png','02_train_samples.png','03_test_samples.png','04_train_vs_test.png']:\n    display(Image(filename=str(RUN_DIR/'figs'/figure)))"
    old[13].source = old[13].source.replace('# Part II — Dari Preprocessing ke Teacher', '# 3. Preprocessing dan Pembentukan Target Training')
    old[26].source = old[26].source.replace("SZ = 448; out =", "os.makedirs('cache', exist_ok=True)\nSZ = 448; out =")
    old[27] = code('''
    STUDENT_CHECKPOINT=ARTIFACT_ROOT/'student_r34d_model.pt'
    if CFG['redistill']:
        assert BASE_CHECKPOINT.is_file(), 'Base scratch checkpoint required'
        environment=os.environ.copy()
        environment.update(DATERA_BASE_CHECKPOINT=str(BASE_CHECKPOINT),
                           DATERA_TEACHER_CSV=str(RUN_DIR/'teacher_reference.csv'),
                           DATERA_STUDENT_OUT=str(RUN_DIR/'runs/student_r34d_rebuilt'))
        process=subprocess.Popen([sys.executable,'-u','distill_local.py'],env=environment,
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout: print(line,end='',flush=True)
        if process.wait(): raise RuntimeError('Student distillation failed')
        STUDENT_CHECKPOINT=RUN_DIR/'runs/student_r34d_rebuilt/model.pt'
        print('Fresh student checkpoint SHA256:',sha256(STUDENT_CHECKPOINT))
    else:
        assert sha256(STUDENT_CHECKPOINT)==MANIFEST['student_weights_sha256']
        print('Reference checkpoint inference mode; no fresh training claimed.')
    ''')
    old[25].source = old[25].source.replace("print('Historical student training log (retained from the team run):')",
        "print('REFERENCE run log for comparison; the next training cell prints the new run:')")
    old[24].source = old[24].source.replace('# Part III — Distillation ke Satu ResNet34d', '# 5. Training Student ResNet34d')
    old[28].source = old[28].source.replace('# Part IV — Inference Raw JPEG dan Pemeriksaan Submission', '# 7. Inference Raw JPEG dan Pemeriksaan Submission')
    old[33].source = old[33].source.replace("student_training='historical hard-target distillation; log/config retained'",
        "student_training='fresh hard-target distillation' if CFG['redistill'] else 'reference historical student'")
    old[33].source = old[33].source.replace("raw_jpeg_inference_executed=True,teacher_cache_replay_executed=True,",
        "raw_jpeg_inference_executed=True,teacher_cache_replay_executed=True,eda_executed=True,")
    integrity = [md('''
    ## 2.1 Integritas dataset dan ringkasan kelas

    Distribusi kelas dihitung dari seluruh train, tanpa memakai label test.
    Flag video/foto hanya diturunkan dari dimensi gambar. Kegagalan membaca
    gambar, ID berulang, atau label di luar rentang menghentikan eksekusi.
    '''), code('''
    counts=train.groupby('label').size().rename('images')
    class_table=class_mapping.merge(counts,on='label')
    class_table['share_percent']=100*class_table.images/len(train)
    display(class_table)
    print('Minimum/maximum class size:',int(counts.min()),int(counts.max()))
    print('Imbalance ratio:',float(counts.max()/counts.min()))
    print('Train/test filename overlap:',len(set(train.id)&set(sample.id)))
    print('Dimensions train:',pd.Series(list(map(tuple,train_dims))).value_counts().head(8).to_dict())
    print('Dimensions test:',pd.Series(list(map(tuple,test_dims))).value_counts().head(8).to_dict())
    '''), md('''
    ## 2.2 Audit kualitas pada citra raw

    Sampel tetap 400 train + 400 test, seed 0. Saturasi dan grayscale dihitung
    pada resize 256px. `black_fraction` berarti bagian piksel grayscale <8;
    metrik ini mencakup border maupun area dalam gambar dan **tidak** otomatis
    berarti kotak hitam interior. Kandidat hampir grayscale memakai rata-rata
    rentang kanal RGB <3 pada skala 0–255. Threshold ini merupakan indikator
    audit, bukan anotasi manual. Blur memakai varians Laplacian.
    '''), code('''
    def raw_quality(path):
        image=cv2.imread(str(path));assert image is not None,str(path)
        image=cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        channel_range=np.ptp(image.astype(np.float32),axis=2).mean()
        return dict(saturation=float(hsv[...,1].mean()),brightness=float(hsv[...,2].mean()),
                    sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                    black_fraction=float((gray<8).mean()),rgb_channel_range=float(channel_range),
                    near_grayscale=bool(channel_range<3))
    audit_rng=np.random.RandomState(0)
    audit_rows=[]
    for split,paths in [('train',train_paths),('test',test_paths)]:
        chosen=audit_rng.choice(len(paths),400,replace=False)
        with ThreadPoolExecutor(CFG['workers']) as pool:
            values=list(pool.map(raw_quality,[paths[i] for i in chosen]))
        audit_rows.extend(dict(split=split,id=paths[i].name,**value) for i,value in zip(chosen,values))
    quality=pd.DataFrame(audit_rows)
    display(quality.groupby('split').agg(images=('id','size'),saturation_median=('saturation','median'),
        sharpness_median=('sharpness','median'),black_fraction_median=('black_fraction','median'),
        near_grayscale_count=('near_grayscale','sum')))
    quality.to_csv('quality_audit_sample.csv',index=False)
    '''), md('''
    ## 2.3 EDA visual asli tim

    Sel berikut menghitung ulang distribusi kelas, memilih contoh raw setiap
    kelas, menampilkan variasi test, dan membandingkan statistik train–test.
    Enam gambar EDA/preprocessing ditampilkan di notebook dan disematkan offline
    pada versi final sehingga tetap terlihat tanpa menjalankan training.
    ''')]
    training_curve = [md('''
    ## 5.2 Kurva training student yang dijalankan pada revisi ini

    Loss, agreement terhadap teacher, dan akurasi train dihitung dari log run
    student yang dipakai untuk inference berikutnya. Agreement adalah ukuran
    distillation, bukan akurasi pada label test resmi atau validasi independen.
    '''), code('''
    import re
    import matplotlib.pyplot as plt
    log_path=STUDENT_CHECKPOINT.parent/'log.txt' if CFG['redistill'] else ARTIFACT_ROOT/'student_fresh_log.txt'
    log_text=log_path.read_text() if log_path.is_file() else history['student_log']
    records=[]
    for match in re.finditer(r'epoch (\\d+) loss ([\\d.]+) \\| agreement with ensemble submission ([\\d.]+) \\| train acc ([\\d.]+)',log_text):
        epoch,loss,agreement_value,accuracy=map(float,match.groups())
        records.append(dict(epoch=int(epoch),loss=loss,teacher_agreement=agreement_value,train_accuracy=accuracy))
    learning=pd.DataFrame(records);assert len(learning)>0
    display(learning)
    figure,axes=plt.subplots(1,2,figsize=(11,3.5))
    axes[0].plot(learning.epoch,learning.loss,marker='o');axes[0].set(title='Student training loss',xlabel='epoch')
    axes[1].plot(learning.epoch,learning.teacher_agreement,marker='o',label='teacher agreement')
    axes[1].plot(learning.epoch,learning.train_accuracy,marker='.',label='official train accuracy')
    axes[1].set(xlabel='epoch',ylim=(0,1.02));axes[1].legend()
    figure.tight_layout();figure.savefig('figs/10_student_learning.png',dpi=100);plt.close(figure)
    display(Image(filename='figs/10_student_learning.png'))
    ''')]
    validation = [md('''
    # 6. Evaluasi dan Interpretasi

    Referensi scene-fold0 berikut dihitung kembali dari probabilitas tersimpan
    subset teacher crop-recipe (dua model fold0 tim dan satu model crop).
    Referensi ini **bukan** OOF seluruh teacher final dan bukan validasi student
    yang dimulai dari checkpoint full-train. Confusion matrix, macro F1,
    akurasi, dan F1 per kelas dilaporkan dengan cakupan tersebut.
    '''), code('''
    from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
    reference=np.load(ARTIFACT_ROOT/'validation_reference.npz')
    val_index=reference['indices'];val_y=train.label.to_numpy()[val_index]
    team_val=nz(gm([reference['team18'],reference['team34']]))
    val_probability=nz(gm([team_val,reference['crop']]))
    val_prediction=val_probability.argmax(1)
    per_class=f1_score(val_y,val_prediction,labels=np.arange(30),average=None,zero_division=0)
    print('Teacher subset scene-fold0 rows:',len(val_y))
    print('Macro F1:',float(per_class.mean()),'accuracy:',float(accuracy_score(val_y,val_prediction)))
    display(class_mapping.assign(f1=per_class,support=np.bincount(val_y,minlength=30)))
    confusion=confusion_matrix(val_y,val_prediction,labels=np.arange(30))
    normalized=confusion/np.maximum(confusion.sum(1,keepdims=True),1)
    figure,ax=plt.subplots(figsize=(10,8))
    chart=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
    ax.set(xlabel='predicted class',ylabel='true class',title='Teacher subset: scene-fold0 held-out reference')
    ax.set_xticks(range(30));ax.set_yticks(range(30));figure.colorbar(chart,ax=ax)
    figure.tight_layout();figure.savefig('figs/08_confusion.png',dpi=100);plt.close(figure)
    order=np.argsort(per_class);figure,ax=plt.subplots(figsize=(12,4))
    ax.bar(range(30),per_class[order]);ax.axhline(per_class.mean(),ls='--',color='black')
    ax.set_xticks(range(30));ax.set_xticklabels(order);ax.set(xlabel='class (sorted by F1)',ylabel='F1',title='Per-class F1: teacher subset reference')
    figure.tight_layout();figure.savefig('figs/09_perclass_f1.png',dpi=100);plt.close(figure)
    display(Image(filename='figs/08_confusion.png'));display(Image(filename='figs/09_perclass_f1.png'))
    ''')]
    # Preserve the original full preprocessing/model sources and teacher formula.
    cells = old[:6] + integrity[:4] + old[6:10] + integrity[4:] + old[10:22]
    cells += old[24:28] + training_curve + validation + old[28:]
    for cell in cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook.cells = cells
    notebook.metadata['revision'] = 'single comprehensive review notebook; fresh student training enabled'
    path = OUT / 'DATERAQUEST_UB_Lengkap.ipynb'
    nb.validate(notebook)
    nb.write(notebook, path)
    print(json.dumps(dict(notebook=str(path),cells=len(cells),base_sha256=digest(base)),indent=2))


if __name__ == '__main__':
    build()
