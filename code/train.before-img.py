#!/usr/bin/env python
"""Training Dateraquest yang dioptimalkan buat server (A100 + 256 core). EDA & cerita lengkapnya ada di dateraquest.ipynb.

Kenapa lebih cepat dari notebook:
- worker kirim batch numpy uint8 lewat pipe, bukan /dev/shm (server cuma punya 1 GB shm) -> worker bisa banyak
- normalisasi + TTA di GPU, autocast bf16 (A100), tanpa GradScaler
- beberapa fold jalan paralel di 1 GPU (`run --jobs`)
- cache fold/val/test disimpan di cache/ sekali, dipakai ulang semua run

Gambar test diputar acak 0-360 derajat, jadi resep R: augmentasi rotasi +-180, validasi v4 + rotasi, TTA 4 putaran x flip.

Pakai (dari folder yang sama dengan dq.py dan folder dataset):
  python train.py prep                                    # sekali: fold + cache val (v4, v4+rotasi) & test
  python train.py run --recipe R --jobs 3                 # 5 fold paralel, lalu ensemble
  python train.py fold --recipe R --fold 0                # satu fold aja
  python train.py ensemble --recipe R                     # OOF F1 + submission dari 5 fold
Override: --attn cbam --pool gem --arch resnet34 --epochs 200 --mixp 0.5 --ema 1 --lr 2e-3 --bs 64 --rot 180 --tta rot8 --split scene --tag nama_run
Split: `scene` (default, cache/scene_folds.npy dari scene_folds.csv tim, dipakai read-only) atau `phash` (fold notebook).
Hasil per fold: runs/<tag>_f<k>/ (model.pt, pv.npy = prob val v4+rotasi, pt.npy = prob test, preds.npz = semua varian TTA, hist.csv, log.txt)
"""
import argparse, functools, hashlib, json, math, os, subprocess, sys, time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F, timm
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler
from sklearn.metrics import f1_score
import albumentations as A, torch.nn as nn
import dq

# augmentasi "ringan" ala kamera: eksposur, white balance, saturasi, blur/downscale, JPEG. Tanpa occluder/tint berat corrupt_v4.
dq.CORRUPTERS["corrupt_light"] = A.Compose([
    A.RandomGamma(gamma_limit=(65, 150), p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=(0.5, 1.5), hue=0.02, p=0.7),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
    A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.Downscale(scale_range=(0.5, 0.9))], p=0.35),
    A.ImageCompression(quality_range=(10, 60), p=0.5),
])

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dateraquest-2-0-machine-learning-competition"
CACHE, RUNS = ROOT / "cache", ROOT / "runs"
SEED, N_FOLDS, DEV = 42, 5, "cuda"

# resep dari notebook (semua mode "mix": tiap sampel 50% bersih, 50% korupsi `cmode`)
RECIPES = {
    "E": dict(epochs=150, cmode="corrupt_v4", mixp=0.0, ema=0),
    "F": dict(epochs=300, cmode="corrupt_v4", mixp=0.5, ema=1),
    "H": dict(epochs=150, cmode="corrupt_hard", mixp=0.0, ema=0),
    "R": dict(epochs=300, cmode="corrupt_v4", mixp=0.5, ema=1, rot=180.0, tta="rot8"),
    "L": dict(epochs=300, cmode="corrupt_light", mixp=0.5, ema=1, rot=180.0, tta="flip"),
}
DEFAULTS = dict(arch="resnet18", lr=2e-3, wd=0.05, bs=64, smooth=0.1, eval_every=25, rot=0.0, rot_p=0.9, tta="flip", split="scene", attn="none", pool="avg")
OVERRIDES = dict(arch=str, epochs=int, cmode=str, mixp=float, ema=int, lr=float, wd=float, bs=int, eval_every=int,
                 rot=float, rot_p=float, tta=str, split=str, attn=str, pool=str)
# split -> (file di cache/, ukuran fold acuan). sklearn beda versi = shuffle fold beda, jadi fold dibekukan sebagai file.
FOLD_FILES = {"phash": ("folds.npy", [603, 604, 602, 603, 602]), "scene": ("scene_folds.npy", [614, 607, 603, 600, 590])}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wb_init(project, name, group, config, d):
    """W&B opsional (--wandb PROJECT). Belum `wandb login` -> otomatis offline (sync nanti: `wandb sync <dir>`)."""
    if not project: return None
    import wandb
    kw = dict(project=project, name=name, group=group, config=config, dir=str(d), reinit="finish_previous")
    try:
        return wandb.init(**kw)
    except Exception as e:  # gak ada API key / gak ada internet: jangan sampai training gagal gara-gara logging
        log(f"wandb online gagal ({type(e).__name__}), pakai mode offline")
        return wandb.init(mode="offline", **kw)


def load_meta():
    train = pd.read_csv(DATA / "train.csv"); sub = pd.read_csv(DATA / "sample_submission.csv")
    nc = len(pd.read_csv(DATA / "class_mapping.csv"))
    return train, sub, nc, [str(DATA / "train" / i) for i in train.id], [str(DATA / "test" / i) for i in sub.id]


def load_folds(split):
    name, sizes = FOLD_FILES[split]
    fold = np.load(CACHE / name)
    assert np.bincount(fold).tolist() == sizes, f"fold {split} beda dari acuan {sizes}!"
    return fold


def save_npy(path, arr):  # tulis ke file sementara dulu, biar run yang mati di tengah gak ninggalin cache setengah jadi
    tmp = path.with_suffix(".tmp.npy"); np.save(tmp, arr); os.replace(tmp, path)


def rot_img(img, angle, reflect):  # putar di resolusi asli; pojok kosong diisi pantulan atau hitam
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101 if reflect else cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------- prep: fold + cache

def phash(p):
    import imagehash
    from PIL import Image
    im = Image.open(p); im.draft("RGB", (im.size[0] // 4, im.size[1] // 4))
    return imagehash.phash(im.convert("RGB"), hash_size=8).hash.flatten()


def make_folds(train, paths, workers):
    # sama persis kayak notebook bagian 6: near-dup (pHash dist <= 8) satu grup, StratifiedGroupKFold seed 42
    from scipy.sparse.csgraph import connected_components
    from sklearn.model_selection import StratifiedGroupKFold
    with ThreadPoolExecutor(workers) as ex:
        H = np.array(list(ex.map(phash, paths)), np.float32)
    D = H @ (1 - H).T + (1 - H) @ H.T; np.fill_diagonal(D, 99)
    _, groups = connected_components(D <= 8, directed=False)
    fold = np.full(len(train), -1)
    for f, (_, va) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, train.label, groups)):
        fold[va] = f
    return fold


def test_img(p):
    return dq.squash(dq.read_rgb(p))


def val_rot(args):
    """(path, seed) -> versi v4 + rotasi acak 360 derajat, 224x224. Sudut, pengisi, dan korupsi tetap per gambar."""
    path, seed = args
    rng = np.random.RandomState(seed + 1_000_003)
    img = rot_img(dq.read_rgb(path), rng.uniform(-180, 180), rng.rand() < 0.5)
    np.random.seed(seed); dq.corrupt_v4.set_random_seed(seed)
    return dq.squash(dq.corrupt_v4(image=img)["image"])


def prep(a):
    CACHE.mkdir(exist_ok=True)
    train, sub, _, trp, tep = load_meta()
    if not (CACHE / "folds.npy").exists():
        t0 = time.time(); save_npy(CACHE / "folds.npy", make_folds(train, trp, 64)); log(f"folds: {time.time() - t0:.0f}s")
    if not (CACHE / "scene_folds.npy").exists() and (CACHE / "scene_folds.csv").exists():
        d = pd.read_csv(CACHE / "scene_folds.csv")
        assert (d.id.values == train.id.values).all() and (d.label.values == train.label.values).all(), "scene_folds.csv beda urutan"
        save_npy(CACHE / "scene_folds.npy", d.fold.values.astype(np.int64))
    for s, (name, _) in FOLD_FILES.items():
        if (CACHE / name).exists(): log(f"fold {s}:", np.bincount(load_folds(s)).tolist())
        else: log(f"PERINGATAN: cache/{name} belum ada (split {s} gak bisa dipakai)")
    ctx = mp.get_context("forkserver")
    if not (CACHE / "corr4.npy").exists():
        t0 = time.time()
        with ctx.Pool(a.workers) as pool:  # seed = index gambar -> sama persis dengan corr4_cache notebook
            pairs = pool.map(dq.val_pair, [(p, k, "corrupt_v4") for k, p in enumerate(trp)], chunksize=4)
        save_npy(CACHE / "clean.npy", np.stack([c for c, _ in pairs])); save_npy(CACHE / "corr4.npy", np.stack([v for _, v in pairs]))
        log(f"cache val v4: {time.time() - t0:.0f}s")
    if not (CACHE / "corr4r.npy").exists():
        t0 = time.time()
        with ctx.Pool(a.workers) as pool:
            save_npy(CACHE / "corr4r.npy", np.stack(pool.map(val_rot, [(p, k) for k, p in enumerate(trp)], chunksize=4)))
        log(f"cache val v4+rotasi: {time.time() - t0:.0f}s")
    if not (CACHE / "test.npy").exists():
        with ThreadPoolExecutor(min(a.workers, 64)) as ex:
            save_npy(CACHE / "test.npy", np.stack(list(ex.map(test_img, tep))))
        log("cache test: ok")


# ---------------------------------------------------------------- data

class TrainNP(Dataset):
    """Kayak dq.TrainDS, tapi balikin numpy uint8 CHW. Batch numpy dikirim worker lewat pipe (pickle), bukan /dev/shm."""
    def __init__(self, paths, labels, mode, rot=0.0, rot_p=0.0):
        self.paths, self.labels, self.mode, self.rot, self.rot_p = paths, labels, mode, rot, rot_p
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = dq.read_rgb(self.paths[i])
        if self.rot and np.random.rand() < self.rot_p:  # test diputar acak 360 derajat, jadi train juga
            img = rot_img(img, np.random.uniform(-self.rot, self.rot), np.random.rand() < 0.5)
        img = dq.CORRUPTERS[self.mode](image=img)["image"] if self.mode in dq.CORRUPTERS else dq.light_color(image=img)["image"]
        return np.ascontiguousarray(dq.rrc(image=img)["image"].transpose(2, 0, 1)), self.labels[i]


def worker_init(i):  # seed semua corrupter per worker, termasuk corrupt_light yang didaftarkan di file ini
    dq.worker_init(i); dq.CORRUPTERS["corrupt_light"].set_random_seed(torch.initial_seed() % 2**32)


class GeM(nn.Module):
    """Generalized mean pooling: rata-rata pangkat p (dipelajari). p=1 -> avg pool, p besar -> mendekati max pool."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p, self.eps = nn.Parameter(torch.tensor(p)), eps
    def forward(self, x):
        return x.float().clamp(min=self.eps).pow(self.p).mean((2, 3)).pow(1.0 / self.p)


def make_model(c, nc):
    kw = dict(block_args=dict(attn_layer=c["attn"])) if c["attn"] != "none" else {}  # se / cbam di tiap blok ResNet
    model = timm.create_model(c["arch"], pretrained=False, num_classes=nc, drop_rate=0.2, **kw)
    if c["pool"] == "gem": model.global_pool = GeM()
    return model


def collate_np(batch):
    return np.stack([x for x, _ in batch]), np.array([y for _, y in batch], np.int64)


@functools.cache
def norm_stats():  # lazy: worker juga meng-import file ini, jangan sampai mereka nyentuh CUDA
    return (torch.tensor(dq.MEAN, device=DEV).view(1, 3, 1, 1), torch.tensor(dq.STD, device=DEV).view(1, 3, 1, 1))


def norm(x):  # uint8 NCHW di GPU -> float ternormalisasi, channels_last
    mean, std = norm_stats()
    return ((x.float() - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model, arr, tta="flip", bs=256):
    """arr: uint8 NHWC (cache). tta: flip = asli + flip horizontal; rot8 = 4 putaran 90 derajat x flip."""
    model.eval(); out = []
    turns = range(4) if tta == "rot8" else [0]
    for i in range(0, len(arr), bs):
        x = norm(torch.from_numpy(np.ascontiguousarray(arr[i:i + bs])).to(DEV).permute(0, 3, 1, 2))
        p = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for k in turns:
                xr = torch.rot90(x, k, (2, 3)) if k else x
                p = p + model(xr).float().softmax(1) + model(xr.flip(3)).float().softmax(1)
        out.append((p / (2 * len(turns))).cpu())
    return torch.cat(out).numpy()


def macro_f1(y, prob):
    return f1_score(y, prob.argmax(1), average="macro")


# ---------------------------------------------------------------- training satu fold

def make_cfg(a):
    c = {**DEFAULTS, **RECIPES[a.recipe]}
    c.update({k: getattr(a, k) for k in OVERRIDES if getattr(a, k) is not None})
    c["name"] = a.tag or a.recipe
    assert c["split"] in FOLD_FILES and c["tta"] in ("flip", "rot8") and c["attn"] in ("none", "se", "cbam") and c["pool"] in ("avg", "gem")
    return c


def train_fold(c, fold, workers, wandb_project=None):
    out = RUNS / f"{c['name']}_f{fold}"; out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        log(f"{out.name}: sudah selesai, skip"); return
    (out / "cfg.json").write_text(json.dumps(c, indent=1))
    wb = wb_init(wandb_project, out.name, c["name"], {**c, "fold": fold, "workers": workers}, out)
    torch.manual_seed(SEED + fold); torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    train, _, nc, trp, _ = load_meta(); folds = load_folds(c["split"]); y = train.label.values
    tr, va = np.where(folds != fold)[0], np.where(folds == fold)[0]; yv = y[va]
    ld = lambda n: np.load(CACHE / n, mmap_mode="r")
    va_clean, va_v4, va_rot, test = ld("clean.npy")[va], ld("corr4.npy")[va], ld("corr4r.npy")[va], np.load(CACHE / "test.npy")

    mk = lambda m: TrainNP([trp[k] for k in tr], y[tr].tolist(), m, c["rot"], c["rot_p"])
    ds = ConcatDataset([mk("clean"), mk(c["cmode"])])
    dl = DataLoader(ds, c["bs"], sampler=RandomSampler(ds, num_samples=len(tr)), num_workers=workers, drop_last=True,
                    collate_fn=collate_np, persistent_workers=True, prefetch_factor=4,
                    worker_init_fn=worker_init, multiprocessing_context="forkserver")

    model = make_model(c, nc).to(DEV).to(memory_format=torch.channels_last)
    gem_p = [p for n, p in model.named_parameters() if n.endswith("global_pool.p")]  # p GeM jangan kena weight decay
    rest = [p for n, p in model.named_parameters() if not n.endswith("global_pool.p")]
    opt = torch.optim.AdamW([{"params": rest}, {"params": gem_p, "weight_decay": 0.0}], lr=c["lr"], weight_decay=c["wd"])
    total, warm = c["epochs"] * len(dl), 3 * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    mixer = timm.data.Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=c["mixp"], switch_prob=0.5,
                            label_smoothing=c["smooth"], num_classes=nc) if c["mixp"] > 0 else None
    ema = timm.utils.ModelEmaV3(model, decay=0.999) if c["ema"] else None

    ckpt, hist, start, t0 = out / "ckpt.pt", [], 1, time.time()
    if ckpt.exists():  # lanjut setelah mati lampu / di-kill
        st = torch.load(ckpt, map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        if ema: ema.load_state_dict(st["ema"])
        hist, start = st["hist"], st["ep"] + 1
        log(f"{out.name}: lanjut dari checkpoint epoch {st['ep']}")
    for ep in range(start, c["epochs"] + 1):
        model.train(); tl = 0.0
        for xb, yb in dl:
            x = norm(torch.from_numpy(xb).to(DEV, non_blocking=True)); yt = torch.from_numpy(yb).to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if mixer: x, yt = mixer(x, yt)
                loss = F.cross_entropy(model(x), yt, label_smoothing=0.0 if mixer else c["smooth"])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if ema: ema.update(model)
            tl += loss.item()
        ev = ema.module if ema else model
        if ep % c["eval_every"] == 0 or ep == c["epochs"]:
            fc, fv, fr = (macro_f1(yv, predict(ev, a, c["tta"])) for a in (va_clean, va_v4, va_rot))
            hist.append(dict(epoch=ep, loss=tl / len(dl), f1_clean=fc, f1_v4=fv, f1_v4rot=fr, sec=time.time() - t0))
            if wb: wb.log(hist[-1], step=ep)
            log(f"[{out.name}] ep {ep:3d} | loss {tl / len(dl):.3f} | F1 bersih {fc:.4f} | v4 {fv:.4f} | v4+rot {fr:.4f} ({c['tta']}) | {time.time() - t0:.0f}s")
        if ep % 10 == 0 and ep < c["epochs"]:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), ep=ep, hist=hist,
                            ema=ema.state_dict() if ema else None), ckpt.with_suffix(".tmp"))
            os.replace(ckpt.with_suffix(".tmp"), ckpt)
    del dl
    ev = ema.module if ema else model
    # semua varian TTA disimpan, biar efek TTA rotasi bisa dibandingkan tanpa inference ulang
    preds = {f"{n}_{t}": predict(ev, a, t) for t in ("flip", "rot8") for n, a in (("v4rot", va_rot), ("v4", va_v4), ("test", test))}
    np.savez(out / "preds.npz", **preds)
    pv, pt = preds[f"v4rot_{c['tta']}"], preds[f"test_{c['tta']}"]
    np.save(out / "pv.npy", pv); np.save(out / "pt.npy", pt); torch.save(ev.state_dict(), out / "model.pt")
    pd.DataFrame(hist).to_csv(out / "hist.csv", index=False)
    res = dict(f1_v4rot=macro_f1(yv, pv), minutes=(time.time() - t0) / 60, test_conf_median=float(np.median(pt.max(1))),
               **{f"f1_{k}": macro_f1(yv, v) for k, v in preds.items() if not k.startswith("test")})
    (out / "done.json").write_text(json.dumps(res, indent=1)); ckpt.unlink(missing_ok=True)
    if wb: wb.summary.update(res); wb.finish()
    log(f">>> {out.name}: F1 v4+rot {res['f1_v4rot']:.4f} ({c['tta']}) | flip {res['f1_v4rot_flip']:.4f} vs rot8 {res['f1_v4rot_rot8']:.4f} | {res['minutes']:.1f} menit")


# ---------------------------------------------------------------- ensemble + submission

def ensemble(c, wandb_project=None):
    train, sub, nc, _, _ = load_meta(); folds = load_folds(c["split"]); y = train.label.values
    oof, pts = np.zeros((len(y), nc), np.float32), []
    for f in range(N_FOLDS):
        d = RUNS / f"{c['name']}_f{f}"
        if not (d / "done.json").exists(): log(f"{d.name} belum selesai, ensemble dibatalkan"); return
        oof[folds == f] = np.load(d / "pv.npy"); pts.append(np.load(d / "pt.npy"))
        log(f"{d.name}: F1 val v4+rot {macro_f1(y[folds == f], oof[folds == f]):.4f}")
    p = np.mean(pts, 0)
    log(f"OOF macro F1 val v4+rot ({c['name']}, split {c['split']}): {macro_f1(y, oof):.4f} | acc {(oof.argmax(1) == y).mean():.4f}")
    s = pd.DataFrame({"id": sub.id, "label": p.argmax(1).astype(int)})
    assert list(s.columns) == ["id", "label"] and len(s) == len(sub) and (s.id.values == sub.id.values).all()
    assert s.notna().all().all() and s.label.between(0, nc - 1).all()
    path = RUNS / f"{c['name']}_submission.csv"; s.to_csv(path, index=False)
    np.save(RUNS / f"{c['name']}_oof.npy", oof); np.save(RUNS / f"{c['name']}_ptest.npy", np.stack(pts))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    log(f"{path} ditulis | sha={sha}")
    wb = wb_init(wandb_project, f"{c['name']}_ensemble", c["name"], c, RUNS)
    if wb:
        wb.summary.update(dict(oof_f1_v4rot=macro_f1(y, oof), oof_acc=float((oof.argmax(1) == y).mean()), submission_sha=sha,
                               **{f"f1_fold{f}": macro_f1(y[folds == f], oof[folds == f]) for f in range(N_FOLDS)}))
        wb.finish()


def run(a, c):
    prep(a)  # cepat kalau cache udah ada; dikerjain di sini biar fold paralel gak rebutan bikin cache
    per = max(2, a.workers // a.jobs)
    extra = sum([[f"--{k}", str(getattr(a, k))] for k in OVERRIDES if getattr(a, k) is not None], [])
    if a.tag: extra += ["--tag", a.tag]
    if a.wandb: extra += ["--wandb", a.wandb]

    def one(f):
        d = RUNS / f"{c['name']}_f{f}"; d.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "fold", "--recipe", a.recipe, "--fold", str(f), "--workers", str(per)] + extra
        with open(d / "log.txt", "a") as lf:
            return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode

    log(f"{len(a.folds)} fold, {a.jobs} paralel, {per} worker/fold | log: runs/{c['name']}_f*/log.txt")
    with ThreadPoolExecutor(a.jobs) as ex:
        rcs = dict(zip(a.folds, ex.map(one, a.folds)))
    bad = [f for f, rc in rcs.items() if rc != 0]
    if bad: log(f"fold gagal: {bad} -> cek log.txt-nya"); sys.exit(1)
    if sorted(a.folds) == list(range(N_FOLDS)): ensemble(c, a.wandb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "fold", "run", "ensemble"])
    ap.add_argument("--recipe", default="R", choices=list(RECIPES))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    ap.add_argument("--jobs", type=int, default=3, help="berapa fold jalan bareng (run)")
    ap.add_argument("--workers", type=int, default=max(4, min(160, (os.cpu_count() or 8) - 16)),
                    help="worker data. `run`: total, dibagi rata ke --jobs")
    ap.add_argument("--tag", default=None, help="nama run (default = nama resep)")
    ap.add_argument("--wandb", default=None, metavar="PROJECT", help="log ke W&B project ini (opsional)")
    for k, t in OVERRIDES.items(): ap.add_argument(f"--{k}", type=t, default=None)
    a = ap.parse_args()
    mp.set_forkserver_preload(["__main__", "dq"])  # worker fork dari proses yang udah import torch/cv2/albumentations
    c = make_cfg(a)
    if a.cmd == "prep": prep(a)
    elif a.cmd == "fold": train_fold(c, a.fold, a.workers, a.wandb)
    elif a.cmd == "run": run(a, c)
    else: ensemble(c, a.wandb)


if __name__ == "__main__":
    main()
