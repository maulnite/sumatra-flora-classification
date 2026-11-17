"""Read-only replay of Nafis's best local CSV and one fixed PVT probability blend.

Implements the inspected infer2.py math independently. It does not import teammate
modules, alter teammate files, tune public weights, or submit predictions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import reproduce_checkpoint_inference as replay
from .evaluate_strong_baseline import serialize_submission
from .validate_submission import validate

ROOT = Path(__file__).resolve().parents[1]


def fuse(first, second, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Invalid fixed weight")
    replay.compare_probabilities(first, second, 0, 0)
    if weight == 0: return first.copy()
    if weight == 1: return second.copy()
    result = np.exp((1-weight)*np.log(np.maximum(first, 1e-6))+weight*np.log(np.maximum(second, 1e-6)))
    return result/result.sum(1, keepdims=True)


def evidence(matrix, old):
    result = matrix.copy()
    for row, (label, inliers) in enumerate(zip(old.label, old.inlier)):
        if inliers >= 5:
            if not 0 <= label < 30: raise ValueError("Invalid SIFT label")
            result[row, label] = max(result[row, label], 4.)
    return result


def infer(probability, prior, matrix, iterations):
    value = probability.copy()
    for _ in range(iterations):
        value = value*(prior/np.maximum(value.mean(0), 1e-9))[None]
        value /= value.sum(1, keepdims=True)
    return (np.log(np.maximum(value, 1e-9))+matrix).argmax(1)


def prepare(cfg, config):
    if cfg["automatic_upload"]: raise ValueError("Preparation must not upload")
    out = ROOT / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    source = Path(cfg["nafis_root"])
    inputs = {source / cfg["nafis_probability_file"]: cfg["nafis_probability_sha256"],
        source / "evidence2.npy": cfg["nafis_evidence_sha256"],
        source / "infer2.py": cfg["nafis_inference_source_sha256"],
        source / cfg["nafis_submission_file"]: cfg["nafis_submission_sha256"],
        Path(cfg["old_sift"]): cfg["old_sift_sha256"],
        ROOT / cfg["pvt_probability_file"]: cfg["pvt_probability_sha256"]}
    for path, expected in inputs.items():
        if replay.digest(path) != expected: raise ValueError(f"Changed source: {path}")
    sample = pd.read_csv(ROOT / cfg["dataset_root"] / "sample_submission.csv")
    assert sample.id.tolist() == [f"test_{i:04d}.jpg" for i in range(1, 2011)]
    train = pd.read_csv(ROOT / cfg["dataset_root"] / "train.csv")
    prior = np.bincount(train.label, minlength=30)/len(train)
    old = pd.read_csv(cfg["old_sift"]); assert np.array_equal(old.id, sample.id)
    matrix = evidence(np.load(source / "evidence2.npy"), old)
    baseline = np.load(source / cfg["nafis_probability_file"])
    pvt = np.load(ROOT / cfg["pvt_probability_file"])
    assert matrix.shape == baseline.shape == pvt.shape == (2010, 30)
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
    baseline_labels = infer(baseline, prior, matrix, cfg["balance_iterations"])
    original = serialize_submission(sample.id, baseline_labels)
    assert original == (source / cfg["nafis_submission_file"]).read_bytes()
    (out / "baseline_reproduced.csv").write_bytes(original)
    probability = fuse(baseline, pvt, cfg["pvt_weight"])
    labels = infer(probability, prior, matrix, cfg["balance_iterations"])
    candidate = out / "NAFIS90_PVTB3_10_candidate.csv"
    candidate.write_bytes(serialize_submission(sample.id, labels))
    errors = validate(candidate, ROOT / cfg["dataset_root"] / "sample_submission.csv")
    assert not errors, errors
    np.save(out / "candidate_raw_test.npy", probability)
    changed = labels != baseline_labels
    pd.DataFrame(dict(id=sample.id[changed], baseline=baseline_labels[changed], candidate=labels[changed])).to_csv(out / "changed_predictions.csv", index=False)
    for path, expected in inputs.items():
        if replay.digest(path) != expected: raise ValueError(f"Input changed during preparation: {path}")
    summary = dict(at=replay.now(), experiment_id=cfg["experiment_id"], baseline_public=cfg["nafis_public_score"],
        baseline_submission_ref=cfg["nafis_submission_ref"], baseline_csv_byte_identical=True,
        baseline_csv_sha256=replay.digest(out / "baseline_reproduced.csv"), pvt_weight=cfg["pvt_weight"],
        blend=cfg["blend"], fixed_weight_not_optimized=True, candidate_file=str(candidate.relative_to(ROOT)),
        candidate_sha256=replay.digest(candidate), changed_labels=int(changed.sum()),
        changed_strong_evidence_rows=int((changed & (matrix.max(1) >= 4)).sum()), valid_csv=True,
        independent_local_comparison_available=False, verified_improvement=False, submission_uploaded=False,
        note="Full-train Nafis predictions cannot establish an honest fold2 blend gain; one public probe remains a team decision",
        source_files_unchanged=True, input_sha256={str(path): value for path, value in inputs.items()},
        config_sha256=replay.digest(config), runner_sha256=replay.digest(Path(__file__)))
    replay.write_json(out / "preparation.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/DEADLINE-NAFIS-PVT.yaml")
    args = parser.parse_args(); prepare(yaml.safe_load(args.config.read_text()), args.config)
