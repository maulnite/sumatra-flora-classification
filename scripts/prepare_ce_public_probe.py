"""Infer CE A and replace one member of the reproduced best-team ensemble.

Preparation only: uploading is a separate, explicitly authorized Kaggle CLI action.
The training and reference artifacts are read only. No weight search is performed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml

from . import adaptive_self_training as common
from . import reproduce_checkpoint_inference as replay
from . import supervised_contrastive as supcon
from .evaluate_strong_baseline import serialize_submission, sift_labels
from .validate_submission import validate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/LOCAL-PROP-006-A-PUBLIC.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(cfg, args.config, out)


def prepare(cfg, config_path, out):
    start = time.monotonic()
    training_path = ROOT / cfg["training_config"]
    training = yaml.safe_load(training_path.read_text())
    train_out = ROOT / training["output_root"]
    team = Path(training["team_root"])
    reference = ROOT / training["reference_root"]
    checkpoint = ROOT / cfg["checkpoint"]
    assert replay.digest(checkpoint) == cfg["checkpoint_sha256"]
    assert replay.digest(train_out / "spec.json") == cfg["training_spec_sha256"]
    assert cfg["selected_arm"] == "A" and cfg["replacement_fold"] == training["fold"] == 2
    assert cfg["folds"] == [0, 1, 2, 3, 4] and cfg["family_weights"] == [.5, .5]
    assert cfg["inference_size"] == training["validation"]["student_inference_size"] == 352
    assert cfg["tta"] == "flip" and not cfg["automatic_upload"]
    replay.log("Verifying immutable training inputs, code, environment and checkpoint")
    supcon.verify(training, training_path, train_out)
    original = json.loads((ROOT / training["reference_summary"]).read_text())
    pinned = original["provenance"]["input_sha256"]
    hashes = {}
    for rel in ("cache/test_352.npy", "runs/sift_test.csv", "runs/best_sift_submission.csv",
                "runs/probs/R_r18d_i288_e600_352_test.npy", "runs/probs/R_r34d_e600_320_test.npy"):
        path = team / rel
        hashes[str(path)] = replay.digest(path)
        assert hashes[str(path)] == pinned[rel], f"Reference input changed: {path}"
    frame, sample, _, vi = common.metadata(training)
    prior = np.bincount(frame.label, minlength=30) / len(frame)
    matches = pd.read_csv(team / "runs/sift_test.csv")
    assert np.array_equal(matches.id, sample.id)

    # Rebuild the strongest reference from individual saved checkpoint forwards.
    families = {}
    for run, size in (("R_r18d_i288_e600", 352), ("R_r34d_e600", 320)):
        arrays = []
        for fold in cfg["folds"]:
            path = reference / "predictions" / f"{run}_f{fold}" / "test.npy"
            p = np.load(path)
            replay.compare_probabilities(p, p, 0, 0)
            assert p.shape == (2010, 30)
            hashes[str(path)] = replay.digest(path)
            arrays.append(p)
        families[run] = arrays
        expected = np.load(team / f"runs/probs/{run}_{size}_test.npy")
        assert np.array_equal(np.mean(arrays, axis=0), expected)

    ref = common.setup(training)
    baseline_prob = np.mean([np.mean(p, axis=0) for p in families.values()], axis=0)
    baseline_labels, _ = sift_labels(ref.balance(baseline_prob, prior, cfg["balance_iterations"]).argmax(1),
                                    matches, cfg["sift_minimum_inliers"])
    baseline_csv = serialize_submission(sample.id, baseline_labels)
    assert baseline_csv == (team / "runs/best_sift_submission.csv").read_bytes()
    (out / "baseline_reproduced.csv").write_bytes(baseline_csv)
    assert replay.digest(out / "baseline_reproduced.csv") == cfg["baseline_submission_sha256"]
    replay.log("Ten-member reference ensemble and best submission reproduced exactly")

    model = ref.make_model(dict(arch="resnet18d", attn="none", pool="avg"), 30)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to("cuda").to(memory_format=torch.channels_last).eval()
    replay.log("Replaying final A validation forward pass at the original batch size 32")
    validation = common.predict(model, np.load(train_out / "cache/primary_352.npy", mmap_mode="r"), training, ref)
    with np.load(train_out / "A/epoch_600_student.npz") as stored:
        check = replay.compare_probabilities(validation, stored["primary"], 1e-5, 1e-4)
    assert check["within_tolerance"] and check["predicted_labels_different"] == 0
    common.save_array(out / "A_validation.npy", validation)
    replay.log("Inferring all 2010 official test images with A@352 and horizontal-flip TTA")
    test = np.load(team / "cache/test_352.npy", mmap_mode="r")
    assert test.shape == (2010, 352, 352, 3) and test.dtype == np.uint8
    probability = common.predict(model, test, training, ref)
    replay.compare_probabilities(probability, probability, 0, 0)
    common.save_array(out / "A_test.npy", probability)
    families["R_r18d_i288_e600"][cfg["replacement_fold"]] = probability
    combined = np.mean([np.mean(p, axis=0) for p in families.values()], axis=0)
    balanced = ref.balance(combined, prior, cfg["balance_iterations"])
    neural_labels = balanced.argmax(1)
    labels, sift_mask = sift_labels(neural_labels, matches, cfg["sift_minimum_inliers"])
    common.save_array(out / "ensemble_raw_test.npy", combined)
    common.save_array(out / "ensemble_balanced_test.npy", balanced)
    path = out / "LOCAL-PROP-006-A_f2replace_submission.csv"
    path.write_bytes(serialize_submission(sample.id, labels))
    errors = validate(path, ROOT / training["dataset_root"] / "sample_submission.csv")
    assert not errors, errors
    changed = labels != baseline_labels
    assert changed.any(), "Candidate CSV is identical to baseline; do not consume a submission"
    pd.DataFrame(dict(id=sample.id[changed], baseline=baseline_labels[changed], candidate=labels[changed])).to_csv(
        out / "changed_predictions.csv", index=False)
    assert replay.digest(checkpoint) == cfg["checkpoint_sha256"]
    for path_in, expected in hashes.items():
        assert replay.digest(path_in) == expected, f"Input changed during inference: {path_in}"
    summary = dict(experiment_id=cfg["experiment_id"], at=replay.now(), duration_seconds=time.monotonic()-start,
                   authorization=cfg["authorization"], selected_arm="A", selected_checkpoint=str(checkpoint.relative_to(ROOT)),
                   checkpoint_sha256=cfg["checkpoint_sha256"], training_spec_sha256=cfg["training_spec_sha256"],
                   config_sha256=replay.digest(config_path), runner_sha256=replay.digest(Path(__file__)),
                   composition=cfg["composition"], new_checkpoint_total_weight=.1, original_members_retained=9,
                   family_weights=cfg["family_weights"], balance_iterations=cfg["balance_iterations"],
                   sift_minimum_inliers=cfg["sift_minimum_inliers"],
                   baseline_csv_identical=True, baseline_public_score=cfg["baseline_public_score"],
                   validation_replay=check, single_student_primary_f1=common.score(frame.iloc[vi].label, validation),
                   local_validation_scope="A plus matching R34d fold2 only; no full replacement-ensemble OOF score",
                   full_ensemble_oof_available=False, rows=len(sample), csv_format_passed=True,
                   labels_changed_vs_baseline=int(changed.sum()), sift_accepted_rows=int(sift_mask.sum()),
                   sift_labels_changed=int((labels != neural_labels).sum()), submission_file=str(path.relative_to(ROOT)),
                   submission_sha256=replay.digest(path), probability_sha256=replay.digest(out / "A_test.npy"),
                   input_sha256=hashes, all_inputs_unchanged=True, kaggle_upload_performed=False)
    replay.write_json(out / "preparation.json", summary)
    replay.log(json.dumps({k: summary[k] for k in ("duration_seconds", "labels_changed_vs_baseline", "submission_sha256", "csv_format_passed")}))


if __name__ == "__main__":
    main()
