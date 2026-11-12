import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import train as T
train, sub, nc, _, _ = T.load_meta(); y = train.label.values
prior = np.bincount(y, minlength=nc) / len(y)
def balance(p, target, iters=100):
    w = np.ones(p.shape[1])
    for _ in range(iters):
        q = p * w; q /= q.sum(1, keepdims=True); w *= target / q.mean(0)
    q = p * w; return q / q.sum(1, keepdims=True)
P = np.mean([np.load(T.RUNS/"probs/R_r18d_i288_e600_352_oof.npy"), np.load(T.RUNS/"probs/R_r34d_e600_320_oof.npy")], 0)
pred = balance(P, prior).argmax(1)
mq = pd.read_csv(T.RUNS/"sift_trainq.csv")
f1 = lambda p: f1_score(y, p, average="macro")
print("OOF macro F1 pre-SIFT :", round(f1(pred), 4), "acc", round((pred==y).mean(), 4))
rows = []
for t in [3, 4, 5, 6, 8, 10, 12, 20, 30]:
    m = (mq.inlier >= t).values & (mq.label >= 0).values
    post = pred.copy(); post[m] = mq.label.values[m]
    rows.append(dict(min_inlier=t, n_diganti=int((pred[m] != post[m]).sum()), n_query=int(m.sum()),
                     f1_post=round(f1(post),4), delta=round(f1(post)-f1(pred),4), acc_post=round((post==y).mean(),4)))
print(pd.DataFrame(rows).to_string(index=False))
m = (mq.inlier >= 5).values & (mq.label >= 0).values
print("subset inlier>=5: pre-SIFT akurasi", round((pred[m]==y[m]).mean(),4), "| SIFT akurasi", round((mq.label.values[m]==y[m]).mean(),4), "| n", int(m.sum()))
