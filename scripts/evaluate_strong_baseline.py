"""Reproduce EXP-024 and evaluate one frozen blend on genuinely held-out IDs.

Team files are read only. All new artifacts stay in LOCAL-PROP-002.
No training, weight search, upload, or test-score-based model selection.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, confusion_matrix

try:
    from . import run_local_experiment as common
except ImportError:
    import run_local_experiment as common

ROOT = common.ROOT


def balance(prob, prior, iterations=100):
    """Identical to team evalres.py; labels are not an input."""
    weights = np.ones(prob.shape[1])
    for _ in range(iterations):
        q = prob * weights
        q /= q.sum(1, keepdims=True)
        weights *= prior / q.mean(0)
    q = prob * weights
    return q / q.sum(1, keepdims=True)


def blend(baseline, extra, weight):
    if baseline.shape != extra.shape or not 0 <= weight <= 1:
        raise ValueError("Probability shapes or blend weight differ")
    for p in (baseline, extra):
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
            raise ValueError("Invalid probability matrix")
    return (1 - weight) * baseline + weight * extra


def shared_indices(train, folds, local):
    for metadata in (folds, local):
        if not np.array_equal(train.id, metadata.id) or not np.array_equal(train.label, metadata.label):
            raise ValueError("Official train ID/label order mismatch")
    if (folds.groupby("group_id").fold.nunique() > 1).any():
        raise ValueError("Team scene group spans multiple folds")
    seen = set(folds.loc[local.role == "train", "group_id"])
    return np.flatnonzero((local.role == "validation") & ~folds.group_id.isin(seen))


def sift_labels(labels, matches, minimum, query_indices=None, folds=None):
    output = np.array(labels, copy=True)
    if query_indices is None:
        subset = matches
        valid = (subset.inlier.to_numpy() >= minimum) & subset.label.between(0, 29).to_numpy()
    else:
        subset = matches.iloc[query_indices]
        reference = subset.idx.to_numpy(dtype=int)
        valid = ((subset.inlier.to_numpy() >= minimum) & subset.label.between(0, 29).to_numpy()
                 & (reference >= 0) & (reference < len(folds)))
        # Rejected same-fold top matches cannot be replaced by an unavailable next match.
        safe_ref = reference.clip(0, len(folds) - 1)
        valid &= folds[safe_ref] != folds[np.asarray(query_indices)]
    output[valid] = subset.label.to_numpy(dtype=int)[valid]
    return output, valid


def metric(labels, prediction):
    return float(f1_score(labels, prediction, labels=list(range(30)), average="macro", zero_division=0))


def snapshot(source, destination):
    digest = common.sha(source)
    if destination.exists():
        if common.sha(destination) != digest:
            raise ValueError(f"Frozen input changed: {source}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    if common.sha(destination) != digest or common.sha(source) != digest:
        raise ValueError(f"Input changed while taking snapshot: {source}")
    return digest


def serialize_submission(ids, labels):
    return pd.DataFrame({"id": ids, "label": np.asarray(labels, dtype=int)}).to_csv(index=False).encode()


def prepare(cfg):
    team, out = Path(cfg["team_root"]), ROOT / cfg["output_root"]
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    paths = ["cache/scene_folds.csv", "cache/scene_folds.npy", "runs/sift_test.csv", "runs/sift_trainq.csv",
             "runs/best_sift_submission.csv", "dateraquest-2-0-machine-learning-competition/train.csv",
             "dateraquest-2-0-machine-learning-competition/sample_submission.csv"]
    paths += [f"runs/probs/{member}_{kind}.npy" for member in cfg["baseline"]["members"] for kind in ("oof", "test")]
    hashes = {p: snapshot(team / p, inputs / p) for p in paths}
    train = pd.read_csv(ROOT / "data/raw/train.csv")
    sub = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    source_train = pd.read_csv(inputs / "dateraquest-2-0-machine-learning-competition/train.csv")
    source_sub = pd.read_csv(inputs / "dateraquest-2-0-machine-learning-competition/sample_submission.csv")
    assert train.equals(source_train) and sub.equals(source_sub)
    assert common.sha(ROOT / cfg["candidate"]["training_manifest"]) == cfg["candidate"]["training_manifest_sha256"]
    assert common.sha(ROOT / cfg["candidate"]["model"]) == cfg["candidate"]["model_sha256"]
    folds = pd.read_csv(inputs / "cache/scene_folds.csv")
    local = pd.read_csv(ROOT / cfg["candidate"]["training_manifest"])
    index = shared_indices(train, folds, local)
    assert len(index) == cfg["validation"]["expected_rows"]
    assert train.iloc[index].label.nunique() == cfg["validation"]["expected_classes"]
    fold = np.load(inputs / "cache/scene_folds.npy")
    assert np.array_equal(fold, folds.fold)
    assert np.bincount(fold[index], minlength=5).tolist() == cfg["validation"]["expected_fold_counts"]
    mean = {kind: np.mean([np.load(inputs / f"runs/probs/{name}_{kind}.npy") for name in cfg["baseline"]["members"]], 0)
            for kind in ("oof", "test")}
    prior = np.bincount(train.label, minlength=30) / len(train)
    matches = pd.read_csv(inputs / "runs/sift_test.csv")
    assert np.array_equal(matches.id, sub.id)
    raw_pred = balance(mean["test"], prior).argmax(1)
    labels, mask = sift_labels(raw_pred, matches, cfg["baseline"]["sift_test_minimum_inliers"])
    content = serialize_submission(sub.id, labels)
    baseline_sha = hashlib.sha256(content).hexdigest()
    assert baseline_sha == cfg["baseline"]["submission_sha256"]
    assert content == (inputs / "runs/best_sift_submission.csv").read_bytes()
    (out / "baseline_reproduced_submission.csv").write_bytes(content)
    selected = folds.iloc[index].copy()
    selected.insert(0, "official_row_index", index)
    selected.to_csv(out / "evaluation_manifest.csv", index=False)
    group_counts = folds.groupby("group_id").size()
    evidence = dict(baseline="EXP-024", public_macro_f1=cfg["baseline"]["verified_public_macro_f1"],
            baseline_csv_reproduced_byte_identically=True, baseline_sha256=baseline_sha,
            full_raw_neural_oof_macro_f1=metric(train.label, mean["oof"].argmax(1)),
            full_balanced_neural_oof_macro_f1=metric(train.label, balance(mean["oof"], prior).argmax(1)),
            sift_test_labels_changed=int((raw_pred != labels).sum()), actual_group_ids=len(group_counts),
            actual_non_singleton_groups=int((group_counts > 1).sum()), fold_sizes=np.bincount(fold).tolist(),
            evaluated_rows=len(index), evaluated_classes=int(selected.label.nunique()),
            excluded_old_validation_rows=int((local.role == "validation").sum()) - len(index),
            input_sha256=hashes, manifest_sha256=common.sha(out / "evaluation_manifest.csv"))
    common.write_json(out / "baseline_verified.json", evidence)
    common.log(f"Verified strongest baseline; full balanced OOF={evidence['full_balanced_neural_oof_macro_f1']:.6f}, shared holdout={len(index)}")
    return train, sub, folds, index, mean, prior, inputs


def infer_extra(cfg, index):
    team, out = Path(cfg["team_root"]), ROOT / cfg["output_root"]
    paths = {"validation": team / "cache/corr4r_224.npy", "test": team / "cache/test_224.npy"}
    identity = dict(model_sha256=common.sha(ROOT / cfg["candidate"]["model"]),
                    cache_sha256={k: common.sha(v) for k, v in paths.items()},
                    manifest_sha256=common.sha(out / "evaluation_manifest.csv"),
                    inference_source_sha256=common.sha(Path(common.__file__)), tta="flip", input_size=224)
    destination = out / "extra_probabilities.npz"
    if destination.exists():
        assert json.loads((out / "inference_identity.json").read_text()) == identity
        saved = np.load(destination)
        return saved["validation"], saved["test"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(cfg["execution"]["gpu_memory_fraction_cap"])
    model = common.make_model().to("cuda").to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(ROOT / cfg["candidate"]["model"], map_location="cuda", weights_only=False))
    va = np.load(paths["validation"], mmap_mode="r")
    test = np.load(paths["test"], mmap_mode="r")
    pv = common.predict(model, va[index], "cuda")
    pt = common.predict(model, test, "cuda")
    assert pv.shape == (len(index), 30) and pt.shape == (2010, 30)
    assert np.isfinite(pv).all() and np.isfinite(pt).all()
    # The frozen cache files belong to another contributor: never update them.
    assert identity["cache_sha256"] == {k: common.sha(v) for k, v in paths.items()}
    np.savez(destination, validation=pv, test=pt, official_indices=index)
    common.write_json(out / "inference_identity.json", identity)
    common.log("Blur branch re-evaluated on the team's frozen views")
    return pv, pt


def evaluate(cfg, train, sub, folds, index, mean, prior, inputs, pv, pt):
    out = ROOT / cfg["output_root"]
    labels = train.label.to_numpy()[index]
    base = mean["oof"][index]
    weight = cfg["candidate"]["blend_weight"]
    probs = dict(control=balance(base, prior), candidate=balance(blend(base, pv, weight), prior))
    matches = pd.read_csv(inputs / "runs/sift_trainq.csv")
    assert len(matches) == len(train)
    reference = matches.idx.to_numpy().astype(int)
    valid_ref = (reference >= 0) & (reference < len(train))
    assert np.array_equal(matches.label.to_numpy()[valid_ref], train.label.to_numpy()[reference[valid_ref]])
    predicted = {arm: p.argmax(1) for arm, p in probs.items()}
    secondary = {arm: sift_labels(p, matches, 5, index, folds.fold.to_numpy())[0] for arm, p in predicted.items()}
    scores = {arm: dict(primary=metric(labels, predicted[arm]), fold_filtered_sift=metric(labels, secondary[arm])) for arm in probs}
    delta = {kind: scores["candidate"][kind] - scores["control"][kind] for kind in ("primary", "fold_filtered_sift")}
    groups = folds.iloc[index].group_id.to_numpy()
    assert (folds.iloc[index].groupby("group_id").label.nunique() == 1).all()
    within_class = [[np.flatnonzero(groups == group) for group in np.unique(groups[labels == c])] for c in range(30)]
    rng = np.random.RandomState(cfg["validation"]["bootstrap"]["seed"])
    bootstrap = []
    for _ in range(cfg["validation"]["bootstrap"]["replicates"]):
        selected = np.concatenate([np.concatenate([g[i] for i in rng.randint(len(g), size=len(g))]) for g in within_class])
        bootstrap.append(metric(labels[selected], predicted["candidate"][selected]) - metric(labels[selected], predicted["control"][selected]))
    interval = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    criteria = cfg["screen_success_criteria"]
    checks = dict(primary_gain=delta["primary"] >= criteria["minimum_primary_macro_f1_gain"],
                  fold_filtered_sift_non_decreasing=delta["fold_filtered_sift"] >= criteria["minimum_secondary_macro_f1_gain"],
                  bootstrap_lower_bound=interval[0] >= criteria["minimum_primary_bootstrap_lower_bound"])
    result = dict(at=common.now(), proposal_id=cfg["proposal_id"], baseline="EXP-024", baseline_public_macro_f1=.87012,
                  protocol=cfg["validation"]["protocol"], validation_rows=len(index), blend_weight=weight,
                  scores=scores, delta=delta, primary_bootstrap_delta_95_percent=interval,
                  extra_model_alone_macro_f1=metric(labels, pv.argmax(1)),
                  checks=checks, eligible_for_candidate_csv=all(checks.values()), submission_attempted=False,
                  per_class_f1={arm:f1_score(labels,predicted[arm],labels=list(range(30)),average=None,zero_division=0).tolist() for arm in predicted},
                  limitations=cfg["validation"]["limitations"])
    np.savez(out / "validation_predictions.npz", labels=labels, official_indices=index, **probs)
    np.savez(out / "confusion_matrices.npz", **{arm:confusion_matrix(labels,pred,labels=list(range(30))) for arm,pred in predicted.items()})
    candidate_test = balance(blend(mean["test"], pt, weight), prior)
    baseline_test = balance(mean["test"], prior)
    sift = pd.read_csv(inputs / "runs/sift_test.csv")
    final, _ = sift_labels(candidate_test.argmax(1), sift, 5)
    baseline_final, _ = sift_labels(baseline_test.argmax(1), sift, 5)
    result["test_prediction_labels_changed_from_baseline"] = int((final != baseline_final).sum())
    result["test_neural_labels_changed_from_baseline"] = int((candidate_test.argmax(1) != baseline_test.argmax(1)).sum())
    np.save(out / "candidate_test_probabilities.npy", candidate_test)
    if result["eligible_for_candidate_csv"]:
        path = out / "LOCAL-PROP-002_submission.csv"
        path.write_bytes(serialize_submission(sub.id, final))
        errors = common.validate(path, ROOT / "data/raw/sample_submission.csv")
        assert not errors, errors
        result["candidate_csv_sha256"] = common.sha(path)
    common.write_json(out / "result.json", result)
    common.write_json(out / "status.json", dict(status="completed_csv_ready" if all(checks.values()) else "completed_gate_failed", at=common.now(), checks=checks))
    common.log(json.dumps({k: result[k] for k in ("scores", "delta", "primary_bootstrap_delta_95_percent", "checks", "test_prediction_labels_changed_from_baseline")}))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/LOCAL-PROP-002.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(config=cfg, source_sha256=common.sha(Path(__file__)), common_source_sha256=common.sha(Path(common.__file__)))
        existing = out / "specification.json"
        if existing.exists() and json.loads(existing.read_text()) != identity:
            raise ValueError("Frozen specification changed; use a new experiment output root")
        common.write_json(existing, identity)
        common.write_json(out / "status.json", dict(status="verifying_strong_baseline", at=common.now()))
        train, sub, folds, index, mean, prior, inputs = prepare(cfg)
        common.write_json(out / "status.json", dict(status="predicting_extra_branch", at=common.now()))
        pv, pt = infer_extra(cfg, index)
        evaluate(cfg, train, sub, folds, index, mean, prior, inputs, pv, pt)


if __name__ == "__main__":
    main()
