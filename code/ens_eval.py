import itertools, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
D = "dateraquest-2-0-machine-learning-competition"
train = pd.read_csv(f"{D}/train.csv"); sub = pd.read_csv(f"{D}/sample_submission.csv")
y = train.label.values; folds = np.load("cache/scene_folds.npy")
names = ["R", "R_r18d", "R_e600", "R_r50", "R_r34", "R_cbam_gem"]
oof = {n: np.load(f"runs/{n}_oof.npy") for n in names}
pt = {n: np.load(f"runs/{n}_ptest.npy").mean(0) for n in names}
f1 = lambda p: f1_score(y, p.argmax(1), average="macro")
rows = []
for k in range(1, 5):
    for combo in itertools.combinations(names, k):
        p = np.mean([oof[n] for n in combo], 0)
        rows.append((f1(p), " + ".join(combo)))
rows.sort(reverse=True)
print("TOP 12 (OOF macro F1, scene split)")
for s, c in rows[:12]:
    print(f"{s:.4f}  {c}")
best = rows[0][1].split(" + ")
pte = np.mean([pt[n] for n in best], 0)
out = pd.DataFrame({"id": sub.id, "label": pte.argmax(1).astype(int)})
assert len(out) == len(sub) and (out.id.values == sub.id.values).all() and out.label.between(0, 29).all()
path = "runs/ens_best_submission.csv"; out.to_csv(path, index=False)
print("best:", " + ".join(best), "| OOF", round(rows[0][0], 4))
print("wrote", path, "sha=" + hashlib.sha256(open(path, "rb").read()).hexdigest()[:16])
for n in names:
    print(n, "agree with ens:", round((pd.read_csv(f"runs/{n}_submission.csv").label.values == out.label.values).mean(), 4))
