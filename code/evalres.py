#!/usr/bin/env python
"""Evaluasi run di beberapa resolusi inference, lalu cari ensemble terbaik (lanjutan manipulasi_gambar.ipynb Bagian 3).

  python evalres.py probs R_r18d R_e600 --sizes 224 288 320 352   # cache prob OOF/test per (run, resolusi) di runs/probs/
  python evalres.py search R_r18d R_e600 --tag best --sift 5      # ensemble greedy (run@res) + balance (+ label SIFT) -> submission

Catatan: ensemble dipilih di OOF yang sama dengan yang dilaporkan, jadi angka OOF ensemble sedikit optimistis.
"""
import argparse, hashlib, json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
import dq, train as T

PROBS = T.RUNS / "probs"
f1 = lambda y, p: f1_score(y, p.argmax(1), average="macro")


def balance(p, target, iters=100):  # geser bobot per kelas sampai rata-rata prob = prior train; label gak dipakai
    w = np.ones(p.shape[1])
    for _ in range(iters):
        q = p * w; q /= q.sum(1, keepdims=True); w *= target / q.mean(0)
    q = p * w
    return q / q.sum(1, keepdims=True)


def val_rot_size(args):  # sama dengan T.val_rot (seed per gambar), cuma ukuran akhirnya beda
    path, seed, size = args
    rng = np.random.RandomState(seed + 1_000_003)
    img = T.rot_img(dq.read_rgb(path), rng.uniform(-180, 180), rng.rand() < 0.5)
    np.random.seed(seed); dq.corrupt_v4.set_random_seed(seed)
    return dq.squash(dq.corrupt_v4(image=img)["image"], size)


def caches(size, trp, tep):
    cv, ct = T.CACHE / f"corr4r_{size}.npy", T.CACHE / f"test_{size}.npy"
    if not cv.exists():
        with mp.get_context("fork").Pool(48) as pool:
            T.save_npy(cv, np.stack(pool.map(val_rot_size, [(p, k, size) for k, p in enumerate(trp)], chunksize=4)))
        T.log(f"cache val v4+rot {size}px dibuat")
    if not ct.exists():
        with ThreadPoolExecutor(48) as ex:
            T.save_npy(ct, np.stack(list(ex.map(lambda p: dq.squash(dq.read_rgb(p), size), tep))))
    return np.load(cv, mmap_mode="r"), np.load(ct, mmap_mode="r")


def probs(names, sizes):
    train, _, nc, trp, tep = T.load_meta(); y = train.label.values; folds = T.load_folds("scene")
    PROBS.mkdir(exist_ok=True)
    C = {s: caches(s, trp, tep) for s in sizes}  # semua cache dibuat dulu (fork) sebelum CUDA dipakai
    for n in names:
        todo = [s for s in sizes if not (PROBS / f"{n}_{s}_oof.npy").exists()]
        if not todo: continue
        dirs = [T.RUNS / f"{n}_f{f}" for f in range(T.N_FOLDS)]
        if not all((d / "done.json").exists() for d in dirs): T.log(f"{n}: belum 5 fold, dilewati"); continue
        nets = []
        for d in dirs:
            m = T.make_model({**T.DEFAULTS, **json.loads((d / "cfg.json").read_text())}, nc)
            m.load_state_dict(torch.load(d / "model.pt", map_location="cpu")); nets.append(m.to(T.DEV).to(memory_format=torch.channels_last).eval())
        for s in todo:
            va, te = C[s]; oof = np.zeros((len(y), nc), np.float32)
            for f, net in enumerate(nets): oof[folds == f] = T.predict(net, np.asarray(va[folds == f]), "flip", bs=128)
            pt = np.mean([T.predict(net, np.asarray(te), "flip", bs=128) for net in nets], 0)
            np.save(PROBS / f"{n}_{s}_oof.npy", oof); np.save(PROBS / f"{n}_{s}_test.npy", pt)
            T.log(f"{n}@{s}: OOF F1 {f1(y, oof):.4f}")
        del nets; torch.cuda.empty_cache()


def search(names, tag, sift, maxn):
    train, sub, nc, _, _ = T.load_meta(); y = train.label.values; prior = np.bincount(y, minlength=nc) / len(y)
    O, P = {}, {}
    for p in sorted(PROBS.glob("*_oof.npy")):
        n, s = p.name[:-8].rsplit("_", 1)
        if n in names: O[f"{n}@{s}"] = np.load(p); P[f"{n}@{s}"] = np.load(PROBS / f"{n}_{s}_test.npy")
    sc = lambda ks: f1(y, balance(np.mean([O[k] for k in ks], 0), prior))
    single = pd.DataFrame([dict(run=k.split("@")[0], res=int(k.split("@")[1]), oof=f1(y, O[k]), oof_balance=sc([k])) for k in O])
    print(single.pivot(index="run", columns="res", values="oof").round(4).to_string())
    print("dengan balance:"); print(single.pivot(index="run", columns="res", values="oof_balance").round(4).to_string())
    ens = [max(O, key=lambda k: sc([k]))]; best = sc(ens); print(f"greedy: {ens[0]} {best:.4f}")
    while len(ens) < maxn:
        k = max((k for k in O if k not in ens), key=lambda k: sc(ens + [k])); s = sc(ens + [k])
        if s < best + 0.0005: break
        ens.append(k); best = s; print(f"greedy: + {k} {best:.4f}")
    pt = balance(np.mean([P[k] for k in ens], 0), prior); lab = pt.argmax(1); note = ""
    sf = T.RUNS / "sift_test.csv"
    if sift and sf.exists():
        st = pd.read_csv(sf); m = (st.inlier >= sift).values & (st.label >= 0).values
        note = f" | SIFT inlier>={sift}: {int((lab[m] != st.label.values[m]).sum())} label diganti"; lab[m] = st.label.values[m]
    s = pd.DataFrame({"id": sub.id, "label": lab.astype(int)})
    assert len(s) == len(sub) and (s.id.values == sub.id.values).all() and s.label.between(0, nc - 1).all()
    path = T.RUNS / f"{tag}_submission.csv"; s.to_csv(path, index=False)
    ref = pd.read_csv(T.RUNS / "manip_best_sift_submission.csv").label.values  # LB 0.83262
    print(f"ensemble: {' + '.join(ens)} | OOF+balance {best:.4f}{note}")
    print(f"{path.name} sha={hashlib.sha256(path.read_bytes()).hexdigest()[:16]} | sama dengan manip_best_sift (LB 0.83262): {np.mean(lab == ref):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["probs", "search"]); ap.add_argument("names", nargs="+")
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 288, 320, 352])
    ap.add_argument("--tag", default="best"); ap.add_argument("--sift", type=int, default=0); ap.add_argument("--maxn", type=int, default=5)
    a = ap.parse_args()
    probs(a.names, a.sizes) if a.cmd == "probs" else search(a.names, a.tag, a.sift, a.maxn)
