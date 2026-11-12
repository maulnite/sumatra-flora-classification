#!/usr/bin/env python
"""Cocokkan test ke train pakai SIFT (lanjutan manipulasi_gambar.ipynb Bagian 4), versi lebih ketat:
- RootSIFT + lebih banyak keypoint (600, sisi panjang 512)
- presisi diukur dengan query train yang DIKORUPSI + diputar (corrupt_v5, mirip test), dicocokkan ke train LAIN, bukan train bersih
Output: runs/sift_test.csv (id, idx, good, inlier, label), runs/sift_trainq.csv, dan tabel presisi/cakupan per ambang inlier.

  python sift_match.py
"""
import time
import multiprocessing as mp
import cv2, numpy as np, pandas as pd, torch
import dq, train as T

SIDE, NF, RATIO = 512, 600, 0.75


def feat(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY); s = SIDE / max(g.shape)
    if s < 1: g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    kp, d = cv2.SIFT_create(nfeatures=NF).detectAndCompute(g, None)
    if d is None: return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7))  # RootSIFT: norma L2 = 1
    return np.array([k.pt for k in kp], np.float32), d.astype(np.float32)


def feat_clean(p): return feat(dq.read_rgb(p))


def feat_corrupt(args):  # train dikorupsi + diputar seperti test; seed per gambar
    p, seed = args
    rng = np.random.RandomState(seed + 7_000_003)
    img = T.rot_img(dq.read_rgb(p), rng.uniform(-180, 180), rng.rand() < 0.5)
    np.random.seed(seed); dq.CORRUPTERS["corrupt_v5"].set_random_seed(seed)
    return feat(dq.CORRUPTERS["corrupt_v5"](image=img)["image"])


def main():
    train, sub, nc, trp, tep = T.load_meta(); y = train.label.values
    grp = pd.read_csv(T.CACHE / "scene_folds.csv").group_id.values
    t0 = time.time()
    with mp.get_context("fork").Pool(48) as pool:  # fork sebelum CUDA dipakai
        KTR = pool.map(feat_clean, trp, chunksize=8); KTE = pool.map(feat_clean, tep, chunksize=8)
        KQ = pool.map(feat_corrupt, [(p, k) for k, p in enumerate(trp)], chunksize=8)
    T.log(f"SIFT {time.time() - t0:.0f}s | keypoint rata-rata train {np.mean([len(k[0]) for k in KTR]):.0f}, test {np.mean([len(k[0]) for k in KTE]):.0f}, train korup {np.mean([len(k[0]) for k in KQ]):.0f}")
    own = np.concatenate([np.full(len(k[0]), i) for i, k in enumerate(KTR)])
    kidx = np.concatenate([np.arange(len(k[0])) for k in KTR])
    DB = torch.from_numpy(np.concatenate([k[1] for k in KTR])).to(T.DEV)

    def match(kq, exclude=-1):
        pts, D = kq
        if len(D) < 8: return -1, 0, 0
        sim, j = (torch.from_numpy(D).to(T.DEV) @ DB.T).topk(8)
        d, j = (2 - 2 * sim).clamp_min(0).sqrt().cpu().numpy(), j.cpu().numpy(); o = own[j]
        gq, gt = [], []
        for r in range(len(D)):
            v = np.flatnonzero(o[r] != exclude)
            if len(v) >= 2 and d[r, v[0]] < RATIO * d[r, v[1]]: gq.append(r); gt.append(j[r, v[0]])
        if len(gq) < 4: return -1, len(gq), 0
        gq, gt = np.array(gq), np.array(gt); img = own[gt]; best, best_in = -1, 0
        for c in np.bincount(img).argsort()[::-1][:3]:
            s = img == c
            if s.sum() < 4: continue
            _, inl = cv2.estimateAffinePartial2D(pts[gq[s]], KTR[c][0][kidx[gt[s]]], method=cv2.RANSAC, ransacReprojThreshold=8.0)
            n = int(inl.sum()) if inl is not None else 0
            if n > best_in: best, best_in = int(c), n
        return best, len(gq), best_in

    t0 = time.time()
    MQ = pd.DataFrame([match(KQ[i], exclude=i) for i in range(len(KQ))], columns=["idx", "good", "inlier"])
    ME = pd.DataFrame([match(k) for k in KTE], columns=["idx", "good", "inlier"])
    T.log(f"matching {time.time() - t0:.0f}s")
    MQ["label"] = np.where(MQ.idx >= 0, y[MQ.idx.clip(0)], -1); MQ["benar"] = MQ.label == y
    MQ["grup_sama"] = (MQ.idx >= 0) & (grp[MQ.idx.clip(0)] == grp)
    ME["label"] = np.where(ME.idx >= 0, y[ME.idx.clip(0)], -1); ME.insert(0, "id", sub.id)
    ref = pd.read_csv(T.RUNS / "ens320_balance_submission.csv").label.values  # model saja (LB 0.82164)
    rows = []
    for t in [3, 4, 5, 6, 8, 10, 12, 20, 30]:
        q, e = MQ.inlier >= t, ME.inlier >= t
        rows.append(dict(min_inlier=t, trainq_cakupan=q.mean(), trainq_n=int(q.sum()), trainq_presisi=MQ.benar[q].mean(), trainq_grup_sama=MQ.grup_sama[q].mean(),
                         test_cakupan=e.mean(), test_n=int(e.sum()), test_setuju_model=(ME.label[e] == ref[e]).mean(), test_diganti=int((ME.label[e] != ref[e]).sum())))
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
    ME.to_csv(T.RUNS / "sift_test.csv", index=False); MQ.to_csv(T.RUNS / "sift_trainq.csv", index=False)
    T.log("runs/sift_test.csv & runs/sift_trainq.csv ditulis")


if __name__ == "__main__":
    main()
