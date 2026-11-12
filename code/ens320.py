import json, hashlib, itertools
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
import train as T
train, sub, NC, TRP, TEP = T.load_meta()
y = train.label.values; folds = T.load_folds("scene")
prior = np.bincount(y, minlength=NC) / len(y)
VAL = np.load(T.CACHE / "corr4r_320.npy", mmap_mode="r"); TEST = np.load(T.CACHE / "test_320.npy", mmap_mode="r")
f1 = lambda p: f1_score(y, p.argmax(1), average="macro")
def balance(p, target, iters=100):
    w = np.ones(p.shape[1])
    for _ in range(iters):
        q = p * w; q /= q.sum(1, keepdims=True); w *= target / q.mean(0)
    q = p * w; return q / q.sum(1, keepdims=True)
def load(name, f):
    d = T.RUNS / f"{name}_f{f}"; m = T.make_model(json.loads((d / "cfg.json").read_text()), NC)
    m.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    return m.to(T.DEV).to(memory_format=torch.channels_last).eval()
OOF, PT = {}, {}
for name in ["R_r18d", "R_e600", "R_regnety_008"]:
    oof = np.zeros((len(y), NC), np.float32); pts = []
    for f in range(5):
        net = load(name, f)
        oof[folds == f] = T.predict(net, np.asarray(VAL[folds == f]), "flip")
        pts.append(T.predict(net, np.asarray(TEST), "flip")); del net
    OOF[name], PT[name] = oof, np.mean(pts, 0)
    print(f"{name} @320 OOF F1 = {f1(oof):.4f}", flush=True)
names = list(OOF); rows = []
for k in range(1, 4):
    for c in itertools.combinations(names, k):
        p = np.mean([OOF[n] for n in c], 0)
        rows.append(dict(ensemble=" + ".join(c), oof=f1(p), oof_balance=f1(balance(p, prior))))
tab = pd.DataFrame(rows).sort_values("oof_balance", ascending=False)
print(tab.round(4).to_string(index=False), flush=True)
best = tab.iloc[0]; combo = best.ensemble.split(" + ")
pt = np.mean([PT[n] for n in combo], 0); pt_bal = balance(pt, prior)
for tag, p in [("ens320", pt), ("ens320_balance", pt_bal)]:
    s = pd.DataFrame({"id": sub.id, "label": p.argmax(1).astype(int)})
    assert len(s) == len(sub) and (s.id.values == sub.id.values).all() and s.label.between(0, NC - 1).all()
    path = T.RUNS / f"{tag}_submission.csv"; s.to_csv(path, index=False)
    print(tag, path.name, "sha=" + hashlib.sha256(path.read_bytes()).hexdigest()[:16], flush=True)
print("best by OOF+balance:", best.ensemble, round(best.oof_balance, 4), flush=True)
