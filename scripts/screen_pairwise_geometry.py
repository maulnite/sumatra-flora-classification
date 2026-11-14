"""Cheap fold-safe diagnostic of per-image RootSIFT matching after neural retrieval.

Reuses the analytic 448px/240-keypoint gallery, extracts queries from the exact
primary 352px validation cache, and never accesses test labels or runs training.
This is a diagnostic for a richer geometric matcher, not an ASIFT reproduction.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import yaml

try:
    from . import reproduce_checkpoint_inference as replay
except ImportError:
    import reproduce_checkpoint_inference as replay

ROOT = Path(__file__).resolve().parents[1]


def gallery_indices(folds, fold):
    source, query = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if not len(source) or not len(query):
        raise ValueError("Empty training gallery or held-out set")
    return source, query


def shortlist(arrays, source, k):
    results = [set() for _ in range(len(arrays[0]["query"]))]
    for data in arrays:
        for view in ("clean", "corrupt"):
            assert len(data[view]) == len(source)
            values = data["query"] @ data[view].T
            near = np.argpartition(-values, k-1, axis=1)[:, :k]
            for target, row in zip(results, near):
                target.update(source[row].tolist())
    return [np.array(sorted(x), dtype=np.int64) for x in results]


def rootsift(rgb, cfg):
    detector = cv2.SIFT_create(nfeatures=cfg["query_keypoints"], contrastThreshold=cfg["query_contrast_threshold"],
                               edgeThreshold=cfg["query_edge_threshold"])
    keypoints, desc = detector.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    if desc is None:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
    desc = np.sqrt(desc / np.maximum(desc.sum(1, keepdims=True), 1e-9)).astype(np.float32)
    return np.array([p.pt for p in keypoints], np.float32), desc


def unique_correspondences(left, right, pixels):
    seen_left, seen_right, selected = set(), set(), []
    for index, (a, b) in enumerate(zip(left, right)):
        ka, kb = tuple(np.floor(a/pixels).astype(int)), tuple(np.floor(b/pixels).astype(int))
        if ka not in seen_left and kb not in seen_right:
            seen_left.add(ka); seen_right.add(kb); selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def verify_pair(qp, qd, rp, rd, query_shape, reference_shape, cfg, seed):
    empty = dict(inliers=0, matches=0, ratio=0., query_area=0., reference_area=0., median_error=None, accepted=False)
    if len(qd) < 4 or len(rd) < 4:
        return empty
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(qd, rd, k=2)
    reverse = {m.queryIdx:m.trainIdx for m in bf.match(rd, qd)}
    pairs = [(a.queryIdx, a.trainIdx, a.distance/max(b.distance, 1e-9)) for a,b in pairs
             if a.distance < cfg["ratio"]*b.distance and reverse.get(a.trainIdx) == a.queryIdx]
    if len(pairs) < 4:
        return {**empty, "matches":len(pairs)}
    pairs.sort(key=lambda x:x[2])
    q = qp[[x[0] for x in pairs]]; r = rp[[x[1] for x in pairs]]
    unique = unique_correspondences(q, r, cfg["deduplicate_pixels"])
    q, r = q[unique], r[unique]
    if len(q) < 4:
        return {**empty, "matches":len(q)}
    cv2.setRNGSeed(seed)
    transform, mask = cv2.findHomography(q, r, cv2.RANSAC, cfg["ransac_pixels"], maxIters=cfg["ransac_iterations"], confidence=cfg["ransac_confidence"])
    if transform is None or mask is None or not np.isfinite(transform).all():
        return {**empty, "matches":len(q)}
    selected = mask.ravel().astype(bool); qi, ri = q[selected], r[selected]
    n = len(qi)
    if n < 4:
        return {**empty, "matches":len(q), "inliers":n}
    predicted = cv2.perspectiveTransform(qi[:,None,:], transform)[:,0]
    error = float(np.median(np.linalg.norm(predicted-ri, axis=1)))
    qa = float(cv2.contourArea(cv2.convexHull(qi)) / np.prod(query_shape[:2]))
    ra = float(cv2.contourArea(cv2.convexHull(ri)) / np.prod(reference_shape[:2]))
    ratio = n/len(q)
    accepted = (n>=cfg["minimum_inliers"] and ratio>=cfg["minimum_inlier_ratio"] and
                min(qa,ra)>=cfg["minimum_hull_fraction_both_images"] and error<=cfg["maximum_median_forward_reprojection_error"])
    return dict(inliers=n, matches=len(q), ratio=ratio, query_area=qa, reference_area=ra, median_error=error, accepted=bool(accepted))


def choose_match(matches, labels, margin):
    ranked = sorted((x for x in matches if x["accepted"]), key=lambda x:(-x["inliers"],x["median_error"],x["reference_index"]))
    if not ranked:
        return None
    winner = ranked[0]; label = int(labels[winner["reference_index"]])
    other = max((m["inliers"] for m in ranked[1:] if labels[m["reference_index"]] != label), default=0)
    if winner["inliers"] < margin * max(1, other):
        return None
    return dict(**winner, label=label, second_class_inliers=other)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/LOCAL-PROP-005-DIAG.yaml")
    args=parser.parse_args(); cfg=yaml.safe_load(args.config.read_text())
    out=ROOT/cfg["output_root"]; out.mkdir(parents=True,exist_ok=True)
    if (out/"result.json").exists():
        raise ValueError("Completed diagnostic exists; refusing duplicate execution")
    cv2.setNumThreads(1); started=time.monotonic()
    team=Path(cfg["team_root"]); frame=pd.read_csv(team/"cache/scene_folds.csv")
    official=pd.read_csv(ROOT/cfg["dataset_root"]/"train.csv")
    assert np.array_equal(frame.id,official.id) and np.array_equal(frame.label,official.label)
    source,queries=gallery_indices(frame.fold.to_numpy(),cfg["fold"])
    assert not set(frame.iloc[source].group_id)&set(frame.iloc[queries].group_id)
    analytic=ROOT/cfg["analytic_root"]
    feat=dict(np.load(analytic/"features.npz")); meta=pd.read_csv(analytic/"image_quality.csv").iloc[:len(frame)]
    assert np.array_equal(meta.id,frame.id) and (meta.split=="train").all()
    paths=[args.config,Path(__file__),analytic/"features.npz",analytic/"image_quality.csv",team/"cache/scene_folds.csv"]
    features=[]; probabilities=[]
    for run in cfg["backbones"]:
        p=ROOT/cfg["feature_root"]/f"{run}_f{cfg['fold']}.npz"
        features.append(dict(np.load(p))); paths.append(p)
        p=ROOT/cfg["reference_root"]/"predictions"/f"{run}_f{cfg['fold']}"/"validation.npy"
        probabilities.append(np.load(p)); paths.append(p)
    image_path=team/f"cache/corr4r_{cfg['query_size']}.npy"; paths.append(image_path)
    pinned={str(p.resolve()):replay.digest(p) for p in paths}
    replay.write_json(out/"preflight.json",dict(at=replay.now(),input_sha256=pinned,source_rows=len(source),query_rows=len(queries),config=cfg))
    candidates=shortlist(features,source,cfg["neural_neighbors_per_backbone_per_view"])
    assert all(set(c).issubset(set(source)) for c in candidates)
    labels=frame.label.to_numpy(); images=np.load(image_path,mmap_mode="r")
    offsets,desc,points=feat["offsets"],feat["descriptors"],feat["points"]
    replay.log(f"Screening fold {cfg['fold']}: {len(queries)} queries against {len(source)} references; avg {np.mean([len(c) for c in candidates]):.1f} candidate images")
    def one(item):
        row, index=item; image=np.array(images[index]); qp,qd=rootsift(image,cfg); matches=[]
        for reference in candidates[row]:
            begin,end=offsets[reference:reference+2]
            shape=(int(meta.iloc[reference].sift_height),int(meta.iloc[reference].sift_width))
            seed=int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{index}:{reference}".encode()).digest()[:4],"little")%(2**31-1)
            m=verify_pair(qp,qd,points[begin:end],desc[begin:end],image.shape,shape,cfg["matching"],seed)
            matches.append(dict(reference_index=int(reference),**m))
        best=choose_match(matches,labels,cfg["matching"]["minimum_winner_vs_other_class_inlier_ratio"])
        return dict(index=int(index),query_keypoints=len(qp),candidate_count=len(candidates[row]),
                    candidate_contains_true_class=bool((labels[candidates[row]]==labels[index]).any()),match=best)
    rows=[]
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        for row in pool.map(one,enumerate(queries)):
            rows.append(row)
            if len(rows)%50==0 or len(rows)==len(queries):
                replay.log(f"Queries {len(rows)}/{len(queries)}")
                replay.write_json(out/"status.json",dict(state="matching",at=replay.now(),completed=len(rows),total=len(queries)))
    reference=replay.load_reference(ROOT/cfg["reference_root"]/"reference_sources")
    prior=np.bincount(labels,minlength=30)/len(labels); y=labels[queries]
    base=reference.balance(np.mean(probabilities,axis=0),prior).argmax(1); predicted=base.copy()
    accepted=np.array([x["match"] is not None for x in rows])
    for i,row in enumerate(rows):
        if row["match"] is not None:predicted[i]=row["match"]["label"]
    f1=lambda pred:float(f1_score(y,pred,average="macro",labels=list(range(30)),zero_division=0))
    for p,expected in pinned.items():
        assert replay.digest(p)==expected,f"Input changed during diagnostic: {p}"
    result=dict(experiment_id=cfg["experiment_id"],at=replay.now(),fold=cfg["fold"],validation_rows=len(y),
                duration_seconds=time.monotonic()-started,baseline_primary=f1(base),candidate_primary=f1(predicted),primary_delta=f1(predicted)-f1(base),
                accepted_matches=int(accepted.sum()),accepted_precision=float((predicted[accepted]==y[accepted]).mean()) if accepted.any() else None,
                labels_changed=int((base!=predicted).sum()),fixed_errors=int(((base!=y)&(predicted==y)).sum()),
                broken_correct=int(((base==y)&(predicted!=y)).sum()),
                neural_shortlist_true_class_coverage=float(np.mean([x["candidate_contains_true_class"] for x in rows])),
                all_input_hashes_unchanged=True,training_performed=False,asift_performed=False,rift_performed=False,submission_performed=False,
                direct_promotion_allowed=False)
    replay.write_json(out/"matches.json",rows); replay.write_json(out/"result.json",result)
    replay.write_json(out/"status.json",dict(state="completed",at=replay.now()))
    replay.log(json.dumps(result))


if __name__=="__main__":
    main()
