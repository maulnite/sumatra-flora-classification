"""Build the UB-branded final notebook without rewriting training provenance.

The ensemble notebook retains the team's historical executed outputs. The main
notebook is newly executed separately to reproduce student inference from raw
JPEGs. Public source copies omit competition images and prediction outputs.
"""
from __future__ import annotations
import argparse
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import textwrap
import zipfile
import nbformat as nbf
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
EXPECTED='59b9e62480311d8dfbdc41cf3329a7e5ec4a5726d619032d746127c2b3b13aa7'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def md(text):return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text):return nbf.v4.new_code_cell(textwrap.dedent(text).strip())

def cover(title,team,members,logo):
    names='\n'.join('- **'+name+'**' for name in members)
    text=f'''<div align="center">\n\n![Lambang Universitas Brawijaya](attachment:lambang_ub.png)\n\n<h3>DATERAQUEST 2.0 · Sumatra Flora Vision Challenge</h3>\n<h1>{title}</h1>\n<p><strong>Universitas Brawijaya</strong></p>\n<p>Tim <strong>{team}</strong></p>\n</div>\n\n---\n\n'''
    if names:text+='### Anggota Tim\n\n'+names+'\n\n---\n\n'
    text+='''### Deskripsi dan tujuan

Mengidentifikasi **30 kelas flora** Kebun Raya ITERA dari **3.014 citra train**
dan **2.010 citra test**. Tantangannya adalah pergeseran kualitas citra: rotasi,
perubahan warna, occlusion, blur, dan kompresi JPEG. Metrik kompetisi adalah
macro F1, sehingga setiap kelas memiliki bobot yang sama:

$$F_{1,\\mathrm{macro}}=\\frac{1}{30}\\sum_{c=0}^{29}\\frac{2P_cR_c}{P_c+R_c}.$$

Sumber data: [DATERAQUEST 2.0, Kaggle](https://www.kaggle.com/competitions/dateraquest-2-0-machine-learning-competition).
Lambang: [situs resmi Universitas Brawijaya](https://www.ub.ac.id/id/about/ub-brand/).
'''
    if logo.is_file():
        logo_data=base64.b64encode(logo.read_bytes()).decode()
    else:
        published=nbf.read(ROOT/'notebooks/final_ub/DATERAQUEST_UB_Single_Model.ipynb',as_version=4)
        logo_data=published.cells[0].attachments['lambang_ub.png']['image/png']
    cell=md(text);cell.attachments={'lambang_ub.png':{'image/png':logo_data}};return cell


def build(args):
    work=args.work_root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    runtime=out/'single_model_runtime';runtime.mkdir(exist_ok=True)
    archive=args.archive.resolve();originals={}
    with zipfile.ZipFile(archive) as outer:
        for filename in outer.namelist():
            if not filename.endswith('.zip'):continue
            with zipfile.ZipFile(io.BytesIO(outer.read(filename))) as inner:
                for name in inner.namelist():
                    if name.endswith('.ipynb'):originals['single' if 'single' in filename else 'ensemble']=nbf.reads(inner.read(name).decode(),as_version=4)
                assert hashlib.sha256(inner.read('submission.csv')).hexdigest()==EXPECTED
    assert digest(work/'final_domain.csv')==EXPECTED
    inputs={str(archive):digest(archive),str(ROOT/'Three Makasih_Notebook Final.ipynb'):digest(ROOT/'Three Makasih_Notebook Final.ipynb')}
    def snapshot(source,target):
        source=Path(source);value=digest(source);target=runtime/target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        assert digest(target)==value and digest(source)==value;inputs[str(source)]=value;return target
    snapshot(work/'runs/student_r34d/model.pt','student_r34d_model.pt')
    snapshot(work/'pairs2.csv','pairs2.csv')
    snapshot(Path('/home/nafisnaufal1426/inahuy/dateruy/cache/scene_folds.npy'),'scene_folds.npy')
    for name in ['07_sibling_match.png','08_confusion.png','09_perclass_f1.png']:snapshot(work/'figs'/name,name)
    family_a=['r1_r18d_384','r1_r34d_320','r1_r18d_288','r1_r18d_352w','r1_fr_r18d_320','r2_fr_r18d_320w']
    family_b={'p_r18d_384_f0':512,'p_r18d_384_full':512,'p_r34d_352_full':448,'p_r18d_448_full':576}
    arrays={};prob_paths=[]
    for i,run in enumerate(family_a):
        p=work/'runs'/run/'ptest_rot2.npy'
        if not p.exists():p=sorted((work/'runs'/run).glob('ptest_*_flip.npy'))[0]
        arrays[f'A{i}']=np.load(p);prob_paths.append(p)
    for i,(run,size) in enumerate(family_b.items()):
        p=work/'runs'/run/f'ptest_{size}_rot2.npy';arrays[f'B{i}']=np.load(p);prob_paths.append(p)
    for key,name in [('team','team_rot2_probs.npy'),('video','test_isvideo.npy'),('evidence','evidence2.npy')]:
        p=work/name;arrays[key]=np.load(p);prob_paths.append(p)
    old_path=Path('/home/nafisnaufal1426/inahuy/dateruy/runs/sift_test.csv');old=pd.read_csv(old_path)
    arrays['old_label']=old.label.to_numpy();arrays['old_inlier']=old.inlier.to_numpy()
    np.savez_compressed(runtime/'teacher_members.npz',**arrays)
    for p in [*prob_paths,old_path]:inputs[str(p)]=digest(p)
    history={}
    for kind,run in [('base','p_r34d_352_full'),('student','student_r34d')]:
        history[kind+'_log']=(work/'runs'/run/'log.txt').read_text();inputs[str(work/'runs'/run/'log.txt')]=digest(work/'runs'/run/'log.txt')
    history['base_config']=json.loads((work/'runs/p_r34d_352_full/cfg.json').read_text())
    history['teacher_public_score']=.90780;history['teacher_submission_ref']=58470727
    history['ensemble_models']=20;history['single_model_training']='official train labels plus automatic teacher labels for official test images'
    (runtime/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
    for filename in ['dq.py','train5.py','eda2.py','distill.py','predict_single.py']:
        inputs[str(work/filename)]=digest(work/filename)
    manifest=dict(input_sha256=inputs,submission_sha256=EXPECTED,A_models=family_a,B_models=family_b,
                  teacher_family_checkpoint_count=20,derived_teacher_arrays='Cached model predictions: 6 familyA, 4 familyB, aggregate of 10 team checkpoints',
                  student_weights_sha256=digest(runtime/'student_r34d_model.pt'),no_external_pretrained_weights=True,
                  source_training_not_reexecuted=True,distillation_training_not_reexecuted=True,original_outputs_preserved=True,
                  reference_usage='Three Makasih notebook: presentation structure only; no modeling code or results copied',
                  artifact_sha256={str(p.relative_to(runtime)):digest(p) for p in runtime.rglob('*') if p.is_file() and p.name!='manifest.json'})
    (runtime/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    logo=ROOT/'notebooks/final_ub/assets/lambang_ub.png'
    ensemble=copy.deepcopy(originals['ensemble'])
    ensemble.cells[0]=cover('Flora Sumatra:<br>Audit Citra dan Teacher Ensemble',args.team,args.member,logo)
    ensemble.cells.insert(1,md('''
    ## Peran notebook pendamping

    Notebook ini mempertahankan **kode, log, gambar EDA, dan output eksekusi asli
    tim** untuk teacher ensemble 20 model. Tampilan sampul dan navigasi diperbarui;
    model dan hasil historis tidak diubah. Jalur server pada sel asli menunjukkan
    lingkungan tempat eksperimen itu benar-benar dijalankan. Sel-sel training
    historis tidak dijalankan ulang selama penyuntingan ini.

    **Alur:** audit citra → RootSIFT/RANSAC → dua keluarga CNN dari bobot acak →
    ensemble menurut domain → prior balancing dan evidence → CSV teacher.
    Public Kaggle **0,90780** terasosiasi dengan `final_domain.csv`, ref **58470727**.

    Notebook utama **`DATERAQUEST_UB_Single_Model.ipynb`** menjelaskan distillation
    dan mengeksekusi ulang inference raw JPEG dengan satu ResNet34d. Agreement
    terhadap teacher dilaporkan sebagai kesamaan prediksi, bukan validasi baru.

    | Bagian | Isi |
    |---|---|
    | 1 | Environment, EDA, dan preprocessing |
    | 2 | Pencocokan sibling dan kalibrasi evidence |
    | 3–4 | Dua keluarga model scratch dan riwayat training |
    | 5 | Pemeriksaan desain pada scene-held-out/proxy |
    | 6 | Penggabungan akhir dan CSV teacher |
    '''))
    ensemble.metadata['revision_provenance']={'original_archive':archive.name,'original_execution_outputs_retained':True,'new_model_training':False,'source':'team supplied notebook'}
    nbf.validate(ensemble);nbf.write(ensemble,out/'DATERAQUEST_UB_Ensemble.ipynb')
    cells=[cover('Flora Sumatra:<br>Dari Audit Citra ke Model Tunggal',args.team,args.member,logo),md('''
    ### Narasi utama

    **Part I** mengukur keseimbangan kelas dan pergeseran kualitas train–test.
    **Part II** menerjemahkan temuan itu menjadi augmentasi serta teacher ensemble
    dari model scratch. **Part III** memindahkan prediksi teacher ke satu ResNet34d
    melalui distillation. **Part IV** mengeksekusi inference langsung dari JPEG raw
    dan memverifikasi CSV akhir.

    Teacher memakai **20 checkpoint**; deployment memakai **satu model, 21,3 juta
    parameter**. Reduksi biaya berlaku pada inference. Pembangunan teacher dan
    pembuatan target pseudo tetap merupakan bagian dari riwayat training.
    Semua bobot dasar tim berasal dari random initialization, tanpa pretrained
    eksternal. Student menggunakan checkpoint scratch milik tim sebagai awal.

    | Bagian | Cakupan |
    |---|---|
    | I | Data raw, integritas, EDA kelas, galeri, dan domain shift |
    | II | Preprocessing, augmentasi, teacher, dan riwayat base model |
    | III | Pembentukan target otomatis dan distillation student |
    | IV | Inference satu model, verifikasi CSV, dan artefak |
    '''),md('''
    # Part I — Pemahaman Data dan Pergeseran Domain
    ## 1. Environment dan kontrak eksekusi

    Eksekusi default mereproduksi **EDA, preprocessing, formula teacher dari
    cache probabilitas, dan inference student dari raw JPEG**. Log training
    historis ditampilkan dengan provenance yang jelas. Untuk benar-benar
    mengulang training, aktifkan flag yang tersedia; base model membutuhkan
    600 epoch dan distillation menggunakan label teacher otomatis.

    Paket `single_model_runtime` menyimpan satu checkpoint deployment, array
    probabilitas turunan teacher, log/config training, dan manifest SHA256.
    Dataset mentah tidak dimasukkan ke paket ini. Data tetap berasal dari
    dataset resmi kompetisi; cache prediksi merupakan artefak turunan tim.
    '''),code('''
    import os, sys, json, hashlib, time, random, subprocess, gc
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import pandas as pd
    import cv2
    import torch
    import timm
    from PIL import Image as PILImage
    from IPython.display import display, Image, Markdown

    ORIGINAL_DIR = Path.cwd()
    def find_data_root():
        candidates = []
        if os.environ.get('DATERA_DATA_ROOT'): candidates.append(Path(os.environ['DATERA_DATA_ROOT']))
        candidates += [p/'data/raw' for p in [ORIGINAL_DIR,*ORIGINAL_DIR.parents]]
        for pattern in ['/kaggle/input/*/train.csv','/kaggle/input/*/*/train.csv']:
            import glob
            candidates += [Path(x).parent for x in glob.glob(pattern)]
        candidates.append(Path('/home/nafisnaufal1426/orie/datera/data/raw'))
        return next(p.resolve() for p in candidates if (p/'train.csv').is_file() and (p/'test').is_dir())
    DATA_ROOT = find_data_root()
    ARTIFACT_ROOT = Path(os.environ.get('DATERA_ARTIFACT_ROOT', str(ORIGINAL_DIR/'single_model_runtime'))).resolve()
    assert (ARTIFACT_ROOT/'student_r34d_model.pt').is_file(), 'Extract single_model_runtime next to this notebook or set DATERA_ARTIFACT_ROOT'
    RUN_DIR = Path(os.environ.get('DATERA_RUN_DIR', str(ORIGINAL_DIR/'run_ub_single'))).resolve()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(RUN_DIR); sys.path.insert(0,str(RUN_DIR))
    os.environ['DATERA_DATA_ROOT']=str(DATA_ROOT)
    os.environ['DATERA_SCENE_ROOT']=str(ARTIFACT_ROOT)
    CFG = dict(seed=0, image_size=448, infer_batch=128, workers=4,
               retrain_base=False, redistill=False,
               expected_submission_sha256='59b9e62480311d8dfbdc41cf3329a7e5ec4a5726d619032d746127c2b3b13aa7')
    random.seed(CFG['seed']); np.random.seed(CFG['seed']); torch.manual_seed(CFG['seed'])
    torch.set_num_threads(4); cv2.setNumThreads(1)
    assert torch.cuda.is_available(), 'GPU required for the full inference/training demonstration'
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    def sha256(path):
        h=hashlib.sha256()
        with open(path,'rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        return h.hexdigest()
    MANIFEST=json.loads((ARTIFACT_ROOT/'manifest.json').read_text())
    for relative,expected in MANIFEST['artifact_sha256'].items():
        assert sha256(ARTIFACT_ROOT/relative)==expected, relative
    print('Data:', DATA_ROOT)
    print('GPU:', torch.cuda.get_device_name(0))
    print('Versions:', dict(torch=torch.__version__,timm=timm.__version__,numpy=np.__version__,opencv=cv2.__version__))
    print('Mode:', 'historical training replay + actual raw-JPEG inference')
    '''),md('''
    ## 2. Inventaris, label, dan integritas ID
    `train.csv` adalah sumber label resmi untuk citra train. Label test tidak
    tersedia dari penyelenggara; target test yang digunakan student akan dibuat
    oleh teacher dan selalu dinamai **pseudo-label**.
    '''),code('''
    train=pd.read_csv(DATA_ROOT/'train.csv')
    sample=pd.read_csv(DATA_ROOT/'sample_submission.csv')
    class_mapping=pd.read_csv(DATA_ROOT/'class_mapping.csv').sort_values('label')
    assert train.id.is_unique and sample.id.is_unique
    assert len(train)==3014 and len(sample)==2010
    assert np.array_equal(class_mapping.label,np.arange(30))
    assert train.label.between(0,29).all()
    assert sample.id.tolist()==[f'test_{i:04d}.jpg' for i in range(1,2011)]
    train_paths=[DATA_ROOT/'train'/name for name in train.id]
    test_paths=[DATA_ROOT/'test'/name for name in sample.id]
    assert all(p.is_file() for p in train_paths+test_paths)
    def dimensions(path):
        with PILImage.open(path) as image:return image.height,image.width
    with ThreadPoolExecutor(CFG['workers']) as pool:
        train_dims=np.array(list(pool.map(dimensions,train_paths)))
        test_dims=np.array(list(pool.map(dimensions,test_paths)))
    is_video=lambda d:((d[:,0]==608)&(d[:,1]==1080))|((d[:,0]==1080)&(d[:,1]==608))
    train_video,test_video=is_video(train_dims),is_video(test_dims)
    np.save('train_dims.npy',train_dims);np.save('test_isvideo.npy',test_video)
    display(pd.DataFrame({'split':['train','test'],'images':[len(train),len(sample)],
                         'video_dimension_flag':[int(train_video.sum()),int(test_video.sum())]}))
    display(class_mapping)
    '''),md('''
    ## 3. Modul preprocessing yang digunakan tim
    Modul berikut mempertahankan `read_rgb`, squash `INTER_AREA`, normalisasi
    ImageNet, `corrupt_v4`, dan light colour jitter. Angka mean/std adalah
    konstanta preprocessing; tidak ada bobot pretrained ImageNet yang dimuat.
    '''),code('%%writefile dq.py\n'+(work/'dq.py').read_text()),md('''
    ## 4. Implementasi recipe crop dan model scratch
    Source training berikut disertakan agar konfigurasi model, augmentasi,
    MixUp/CutMix, dan EMA dapat diperiksa. Perubahan dari file tim hanya lokasi
    data/cache agar mengikuti konfigurasi notebook dan jumlah reader default.
    Menulis modul ini tidak menjalankan 600 epoch.
    ''')]
    train_source=(work/'train5.py').read_text()
    train_source=train_source.replace("R='/home/nafisnaufal1426/orie/datera/data/raw/'; C='/home/nafisnaufal1426/inahuy/dateruy/cache/'; DEV='cuda'", "R=os.environ['DATERA_DATA_ROOT'].rstrip('/')+'/'; C=os.environ['DATERA_SCENE_ROOT'].rstrip('/')+'/'; DEV='cuda'")
    train_source=train_source.replace('def squash_all(paths, size, workers=48):','def squash_all(paths, size, workers=4):')
    cells += [code('%%writefile train5.py\n'+train_source),md('''
    ## 5. EDA: kelas, contoh citra, dan statistik train–test
    EDA mempertahankan analisis notebook tim: distribusi 30 kelas menurut flag
    dimensi video/foto, contoh setiap kelas, galeri test, dan lima statistik
    kualitas pada sampel acak 400 gambar per split. Flag dimensi merupakan
    indikator domain, bukan bukti manual bahwa setiap gambar berasal dari video.
    ''' )]
    eda=(work/'eda2.py').read_text().split('# 6 sibling match example')[0]
    eda=eda.replace("R = '/home/nafisnaufal1426/orie/datera/data/raw/'","R = os.environ['DATERA_DATA_ROOT'].rstrip('/') + '/'")
    cells += [code('%%writefile eda_local.py\n'+eda),code('''
    Path('figs').mkdir(exist_ok=True)
    completed=subprocess.run([sys.executable,'eda_local.py'],capture_output=True,text=True)
    print(completed.stdout)
    if completed.returncode:
        print(completed.stderr);raise RuntimeError('EDA failed')
    for figure in ['01_class_dist.png','02_train_samples.png','03_test_samples.png','04_train_vs_test.png']:
        display(Image(filename=str(RUN_DIR/'figs'/figure)))
    '''),md('''
    ### Implikasi EDA
    Ketidakseimbangan kelas bersifat moderat; macro F1 tetap diperlukan agar kelas
    kecil tidak tertutup kelas besar. Citra test memiliki penurunan sharpness dan
    lebih banyak area gelap. Temuan ini memotivasi augmentasi yang meniru kondisi
    test, bukan pembersihan citra test yang mengubah identitas atau label.

    # Part II — Dari Preprocessing ke Teacher
    ## 6. Pipeline augmentasi yang dipertahankan
    Rotasi ±180° p=.9 → campuran 50% corruption-v4 / 50% light colour →
    RandomResizedCrop (scale .25–1, ratio .5–2) + horizontal flip → normalisasi.
    Training base memakai MixUp/CutMix dan EMA. Inference tidak melakukan crop:
    seluruh citra di-squash ke persegi sehingga tanaman tetap terlihat.
    '''),code('''
    for figure in ['05_pipeline.png','06_aug_views.png']:
        display(Image(filename=str(RUN_DIR/'figs'/figure)))
    '''),md('''
    ## 7. Sibling matching sebagai sumber evidence otomatis
    RootSIFT dan similarity-RANSAC membandingkan test dengan gallery train.
    Label pasangan dan evidence merupakan hasil algoritma, bukan anotasi test
    resmi. Hanya hubungan test→train yang dipakai pada teacher final. Gambar
    berikut adalah hasil historis notebook tim yang dipertahankan.
    '''),code("display(Image(filename=str(ARTIFACT_ROOT/'07_sibling_match.png')))"),md('''
    ## 8. Teacher ensemble dan asal target
    Teacher menggabungkan enam whole-image model, empat crop/high-resolution
    model, dan agregat sepuluh checkpoint ResNet tim: **20 checkpoint**.
    Formula memakai bobot menurut flag domain, prior balancing, dan calibrated
    sibling evidence. Di bawah ini formula final dieksekusi ulang menggunakan
    **cache probabilitas model tim**, tidak dengan membaca CSV sebagai label
    resmi. Pelatihan dan sumber cache didokumentasikan di notebook pendamping.
    '''),code('''
    cache=np.load(ARTIFACT_ROOT/'teacher_members.npz')
    gm=lambda values,weights=None:np.exp(np.average([np.log(np.maximum(p,1e-6)) for p in values],axis=0,weights=weights))
    nz=lambda p:p/p.sum(1,keepdims=True)
    PA=nz(gm([cache[f'A{i}'] for i in range(6)]))
    PB=nz(gm([cache['team'],nz(gm([cache[f'B{i}'] for i in range(4)]))],[1,1.5]))
    assert np.array_equal(cache['video'],test_video)
    teacher_probability=nz(np.where(test_video[:,None],gm([PA,PB],[1.,1.]),gm([PA,PB],[.25,1.])))
    prior=np.bincount(train.label,minlength=30)/len(train)
    balanced=teacher_probability.copy()
    for _ in range(100):
        balanced*= (prior/np.maximum(balanced.mean(0),1e-9))[None]
        balanced/=balanced.sum(1,keepdims=True)
    evidence=cache['evidence'].copy()
    for row,(label,inliers) in enumerate(zip(cache['old_label'],cache['old_inlier'])):
        if inliers>=5:evidence[row,int(label)]=max(evidence[row,int(label)],4.)
    teacher_labels=(np.log(np.maximum(balanced,1e-9))+evidence).argmax(1)
    teacher=sample.copy();teacher['label']=teacher_labels
    teacher.to_csv('teacher_reference.csv',index=False)
    assert sha256('teacher_reference.csv')==CFG['expected_submission_sha256']
    print('Teacher CSV reproduced exactly:',len(teacher),'rows')
    print('Historical Kaggle ref 58470727, public macro F1 = 0.90780')
    print('The public score belongs to the verified submission; this is not a new evaluation.')
    '''),md('''
    ## 9. Base model: riwayat training scratch
    ResNet34d dasar dilatih 600 epoch dari bobot acak pada citra train berlabel,
    ditambah citra test yang memperoleh pseudo-label melalui sibling matching.
    Source, konfigurasi, dan log berikut adalah **riwayat training tim**, bukan
    klaim bahwa 600 epoch dijalankan ulang dalam sesi notebook ini.
    '''),code('''
    history=json.loads((ARTIFACT_ROOT/'training_history.json').read_text())
    display(pd.Series(history['base_config'],name='base model config').to_frame())
    print(history['base_log'])
    '''),code('''
    BASE_CHECKPOINT=Path(os.environ.get('DATERA_BASE_CHECKPOINT',str(ARTIFACT_ROOT/'base_r34d_model.pt')))
    if CFG['retrain_base']:
        command=[sys.executable,'train5.py','--name','base_r34d_full','--arch','resnet34d',
                 '--img','352','--epochs','600','--bs','64','--workers',str(CFG['workers']),
                 '--seed','2','--fold','-1','--pairs',str(ARTIFACT_ROOT/'pairs2.csv'),
                 '--real_p','.15','--eval_sizes','352','416','448']
        subprocess.run(command,check=True)
        BASE_CHECKPOINT=RUN_DIR/'runs/base_r34d_full/model.pt'
    else:
        print('600-epoch base training not re-run; its recorded logs/config are shown above.')
    '''),md('''
    ## 10. Validasi dan batas interpretasi
    Confusion matrix dan F1 per kelas berikut berasal dari **subset crop-recipe
    teacher pada scene fold0 yang di-hold-out**, sesuai notebook asli. Itu bukan
    OOF independen untuk seluruh ensemble 20 model, dan bukan validasi student
    yang telah melihat pseudo-label test. Agreement terhadap teacher pada data
    distillation hanya mengukur reproduksi label.
    '''),code('''
    display(Image(filename=str(ARTIFACT_ROOT/'08_confusion.png')))
    display(Image(filename=str(ARTIFACT_ROOT/'09_perclass_f1.png')))
    '''),md('''
    # Part III — Distillation ke Satu ResNet34d
    ## 11. Apa yang dipelajari student
    Student dimulai dari **checkpoint scratch milik tim**, bukan bobot pretrained
    eksternal. Setiap batch menggabungkan citra train dengan label resmi dan
    citra test dengan pseudo-label teacher. Ini adalah distillation menggunakan
    hard target pada dataset kompetisi, bukan model yang sejak awal dilatih hanya
    dengan `train.csv`. Deployment berikutnya tidak memanggil teacher atau SIFT.
    '''),code('''
    DISTILLATION = dict(size=448,batch_size=64,learning_rate=2e-4,weight_decay=.01,
                        label_smoothing=.05,seed=0,max_epochs=60,
                        target='teacher pseudo-labels + official train labels',
                        augmentation='horizontal flip p=.5',selection='agreement 100% after at least 5 epochs')
    display(pd.Series(DISTILLATION,name='distillation config').to_frame())
    print('Historical student training log (retained from the team run):')
    print(history['student_log'])
    ''')]
    distill=(work/'distill.py').read_text()
    distill=distill.replace("SZ = 448; out = 'runs/student_r34d';", "SZ = 448; out = os.environ.get('DATERA_STUDENT_OUT','runs/student_r34d_rebuilt');")
    distill=distill.replace("pd.read_csv('final_domain.csv')", "pd.read_csv(os.environ['DATERA_TEACHER_CSV'])")
    distill=distill.replace("torch.load('runs/p_r34d_352_full/model.pt', map_location=DEV)","torch.load(os.environ['DATERA_BASE_CHECKPOINT'], map_location=DEV, weights_only=True)")
    cells += [code('%%writefile distill_local.py\n'+distill),code('''
    STUDENT_CHECKPOINT=ARTIFACT_ROOT/'student_r34d_model.pt'
    if CFG['redistill']:
        assert BASE_CHECKPOINT.is_file(), 'Attach the team scratch base checkpoint or enable retrain_base first'
        environment=os.environ.copy()
        environment.update(DATERA_BASE_CHECKPOINT=str(BASE_CHECKPOINT),
                           DATERA_TEACHER_CSV=str(RUN_DIR/'teacher_reference.csv'),
                           DATERA_STUDENT_OUT=str(RUN_DIR/'runs/student_r34d_rebuilt'))
        subprocess.run([sys.executable,'distill_local.py'],env=environment,check=True)
        STUDENT_CHECKPOINT=RUN_DIR/'runs/student_r34d_rebuilt/model.pt'
    else:
        assert sha256(STUDENT_CHECKPOINT)==MANIFEST['student_weights_sha256']
        print('Using the verified student from the historical distillation run. No training rerun claimed.')
    '''),md('''
    # Part IV — Inference Raw JPEG dan Pemeriksaan Submission
    ## 12. Satu model untuk seluruh test
    Citra dibaca ulang langsung dari berkas JPEG resmi, bukan dari cache prediksi.
    Seluruh gambar diubah ke 448×448 dengan `INTER_AREA`, dinormalisasi dengan
    mean/std ImageNet, lalu diprediksi pada original dan horizontal flip.
    Probabilitas dirata-ratakan dan label ditentukan dengan argmax. Tidak ada
    ensemble model, prior balancing, atau SIFT pada jalur inference student.
    '''),code('''
    from train5 import Net, predict, squash_all
    t0=time.perf_counter()
    model=Net('resnet34d',30).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(STUDENT_CHECKPOINT,map_location='cpu',weights_only=True),strict=True)
    model.eval()
    parameter_count=sum(p.numel() for p in model.parameters())
    raw_test=squash_all(test_paths,CFG['image_size'],workers=CFG['workers'])
    assert raw_test.shape==(2010,448,448,3) and raw_test.dtype==np.uint8
    loaded_at=time.perf_counter();torch.cuda.synchronize()
    student_probability=predict(model,raw_test,bs=CFG['infer_batch'],rot180=False)
    torch.cuda.synchronize();done_at=time.perf_counter()
    assert student_probability.shape==(2010,30)
    assert np.isfinite(student_probability).all() and np.allclose(student_probability.sum(1),1,atol=1e-5)
    student_labels=student_probability.argmax(1)
    print(f'One ResNet34d: {parameter_count:,} parameters ({parameter_count/1e6:.2f}M)')
    print(f'Load/model/resize: {loaded_at-t0:.2f}s; flip-TTA model inference: {done_at-loaded_at:.2f}s')
    print(f'Model inference: {(done_at-loaded_at)*1000/len(sample):.2f}ms/image on this shared GPU')
    '''),md('''
    ## 13. CSV: kontrak, agreement, dan hash
    Public macro F1 **0,90780** adalah hasil Kaggle dari CSV teacher. Jika seluruh
    label dan byte CSV student identik, keduanya merepresentasikan prediksi yang
    sama pada test kompetisi. Kesamaan itu tidak mengukur generalisasi student
    pada tanaman atau scene baru yang tidak digunakan untuk distillation.
    '''),code('''
    submission=sample.copy();submission['label']=student_labels.astype(np.int64)
    submission.to_csv('submission.csv',index=False)
    assert list(submission.columns)==['id','label']
    assert submission.id.equals(sample.id) and submission.id.is_unique
    assert submission.label.between(0,29).all() and submission.label.dtype.kind in 'iu'
    agreement=int((student_labels==teacher_labels).sum())
    output_sha=sha256('submission.csv')
    print(f'Identical labels: {agreement}/{len(sample)}')
    print('Student SHA256:',output_sha)
    print('Teacher SHA256:',sha256('teacher_reference.csv'))
    assert agreement==len(sample),'Student differs from teacher; do not inherit the teacher public score'
    assert output_sha==CFG['expected_submission_sha256']
    print('PASS: raw-JPEG single-model CSV is byte-identical to the final-domain submission.')
    display(submission.head())
    '''),md('''
    ## 14. Ringkasan artefak dan batas klaim
    Ringkasan di bawah berasal dari pemeriksaan pada notebook ini. Teacher dan
    base model tetap didokumentasikan dalam notebook ensemble pendamping.
    Tidak ada submission baru dikirim saat notebook direvisi.
    '''),code('''
    summary=dict(models_at_deployment=1,parameters=parameter_count,test_rows=len(sample),
                 raw_jpeg_inference_executed=True,teacher_cache_replay_executed=True,
                 student_training='historical hard-target distillation; log/config retained',
                 base_training='historical team scratch model; log/config retained',
                 training_reexecuted=dict(base=CFG['retrain_base'],distillation=CFG['redistill']),
                 agreement_with_teacher=agreement,submission_sha256=output_sha,
                 historical_teacher_public_score=.90780,historical_teacher_submission_ref=58470727,
                 new_independent_student_validation_score=None,new_kaggle_submission=False,
                 external_pretrained_weights=False,student_checkpoint_sha256=sha256(STUDENT_CHECKPOINT))
    Path('verification.json').write_text(json.dumps(summary,indent=2)+'\\n')
    display(pd.Series(summary,name='verified artifact summary').to_frame())
    print('Output:', RUN_DIR/'submission.csv')
    '''),md('''
    ### Referensi dan provenance
    - Data, label train, dan metrik: DATERAQUEST 2.0, Kaggle.
    - Kode, checkpoint scratch, probabilitas teacher, serta log: artefak tim
      pada notebook ensemble yang diberikan pengguna.
    - Arsitektur ResNet melalui `timm`, selalu dengan `pretrained=False`.
    - Gaya susunan sampul, narasi antarbab, dan pemisahan EDA/model/evaluasi
      mengacu pada notebook **Three Makasih** yang diberikan pengguna. Tidak ada
      kode model, data, hasil, nama, atau identitas tim referensi yang disalin.
    - Lambang Universitas Brawijaya diambil dari situs resmi dan disematkan
      sebagai attachment agar sampul tetap tampil tanpa koneksi internet.
    ''')]
    main=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'},'provenance':{'teacher_submission_ref':58470727,'teacher_public':.90780,'method':'scratch teacher ensemble + single-student hard-target distillation','reference_style_only':True}})
    nbf.validate(main);nbf.write(main,out/'DATERAQUEST_UB_Single_Model.ipynb')
    source_dir=ROOT/'notebooks/final_ub';source_dir.mkdir(exist_ok=True)
    for name,notebook in [('DATERAQUEST_UB_Ensemble.ipynb',ensemble),('DATERAQUEST_UB_Single_Model.ipynb',main)]:
        public=copy.deepcopy(notebook)
        public.metadata['distribution_variant']='source only; outputs omitted to avoid embedding competition images'
        if 'revision_provenance' in public.metadata:
            public.metadata['revision_provenance']['original_execution_outputs_retained']=False
        for cell in public.cells:
            if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
        nbf.write(public,source_dir/name)
    (out/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),main_cells=len(main.cells),ensemble_cells=len(ensemble.cells),runtime_bytes=sum(p.stat().st_size for p in runtime.rglob('*') if p.is_file())),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path('/home/nafisnaufal1426/akucintakc/final-notebook/final-notebook.zip'))
    parser.add_argument('--work-root',type=Path,default=Path('/home/nafisnaufal1426/akucintakc/work'))
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/FINAL-UB-2026-11-17')
    parser.add_argument('--team',default='pensi lomba data');parser.add_argument('--member',action='append',default=[])
    build(parser.parse_args())

if __name__=='__main__':main()
