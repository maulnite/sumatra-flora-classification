"""Prepare one fixed PVT-B3 member replacement after the preregistered local gate.

Upload is a separate Kaggle CLI operation. All training inputs are read-only;
no public weight search or additional training takes place in this script.
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

from . import train_pvt_idmm as pvt
from .evaluate_strong_baseline import serialize_submission, sift_labels
from .validate_submission import validate

common, replay, ROOT = pvt.common, pvt.replay, pvt.ROOT
FAMILIES = ("R_r18d_i288_e600", "R_r34d_e600")


def replace_member(families, replacement, fold):
    """Retain nine members and the reference float32 family reduction order."""
    if len(families) != 2 or any(len(family) != 5 for family in families) or fold not in range(5):
        raise ValueError("Expected two ordered five-fold families")
    for probability in [replacement] + [value for family in families for value in family]:
        replay.compare_probabilities(probability, replacement, 0, 0)
    copies = [list(family) for family in families]
    copies[0][fold] = replacement
    return np.mean([np.mean(family, axis=0) for family in copies], axis=0)


def completed_candidate(cfg, training, train_out, checkpoint):
    complete = json.loads((train_out / cfg["architecture"] / "supervised/complete.json").read_text())
    if complete["spec_sha256"] != cfg["training_spec_sha256"] or complete["completed_epochs"] != 800:
        raise ValueError("Candidate must be the frozen final epoch 800")
    if cfg["selected_arm"] != "candidate" or training["fold"] != cfg["replacement_fold"]:
        raise ValueError("Wrong arm or source fold")
    gate = pvt.screening(complete["final_metrics"], training, ROOT / training["reference_experiment_root"])
    if gate != complete["screening"] or not gate["passed"]:
        raise ValueError("Candidate did not pass the preregistered local gate")
    if complete["artifacts"].get(str(checkpoint.resolve())) != cfg["checkpoint_sha256"]:
        raise ValueError("Checkpoint is not the completed candidate")
    return complete


def prepare(cfg, config, out):
    started = time.monotonic()
    training_path = ROOT / cfg["training_config"]
    training = yaml.safe_load(training_path.read_text()); train_out = ROOT / training["output_root"]
    checkpoint = ROOT / cfg["checkpoint"]; team = Path(training["team_root"])
    reference = ROOT / training["reference_root"]
    if cfg["automatic_upload"] or cfg["folds"] != list(range(5)) or cfg["family_weights"] != [.5, .5]:
        raise ValueError("Unsupported fixed-probe configuration")
    if (cfg["inference_size"], cfg["inference_batch_size"], cfg["tta"]) != (
            training["validation"]["student_inference_size"], training["validation"]["batch_size"], "flip"):
        raise ValueError("Inference must match saved validation")
    assert replay.digest(checkpoint) == cfg["checkpoint_sha256"]
    assert replay.digest(train_out / "spec.json") == cfg["training_spec_sha256"]
    replay.log("Verifying final PVT, frozen source protocol and advancement gate")
    pvt.verify(training, training_path, train_out)
    complete = completed_candidate(cfg, training, train_out, checkpoint)
    old = yaml.safe_load((ROOT / training["reference_training_config"]).read_text())
    original = json.loads((ROOT / old["reference_summary"]).read_text())
    expected = original["provenance"]["input_sha256"]
    manifest = ROOT / cfg["baseline_member_manifest"]
    assert replay.digest(manifest) == cfg["baseline_member_manifest_sha256"]
    member_hashes = json.loads(manifest.read_text())["preparation"]["input_sha256"]
    hashes = {str(manifest.resolve()): cfg["baseline_member_manifest_sha256"], str(checkpoint.resolve()): cfg["checkpoint_sha256"]}
    for rel in ("cache/test_352.npy", "runs/sift_test.csv", "runs/best_sift_submission.csv",
                "runs/probs/R_r18d_i288_e600_352_test.npy", "runs/probs/R_r34d_e600_320_test.npy"):
        path = team / rel; hashes[str(path)] = replay.digest(path)
        assert hashes[str(path)] == expected[rel], f"Changed reference: {path}"
    frame, sample, _, vi = common.metadata(training)
    prior = np.bincount(frame.label, minlength=30)/len(frame)
    matches = pd.read_csv(team / "runs/sift_test.csv"); assert np.array_equal(matches.id, sample.id)
    families = []
    for run, size in zip(FAMILIES, (352, 320)):
        arrays = []
        for fold in cfg["folds"]:
            path = reference / "predictions" / f"{run}_f{fold}" / "test.npy"
            hashes[str(path)] = replay.digest(path); assert hashes[str(path)] == member_hashes[str(path)]
            probability = np.load(path); assert probability.shape == (2010, 30)
            replay.compare_probabilities(probability, probability, 0, 0); arrays.append(probability)
        assert np.array_equal(np.mean(arrays, axis=0), np.load(team / f"runs/probs/{run}_{size}_test.npy"))
        families.append(arrays)
    ref = common.setup(training)
    baseline = np.mean([np.mean(family, axis=0) for family in families], axis=0)
    baseline_labels, _ = sift_labels(ref.balance(baseline, prior, cfg["balance_iterations"]).argmax(1), matches, cfg["sift_minimum_inliers"])
    content = serialize_submission(sample.id, baseline_labels)
    assert content == (team / "runs/best_sift_submission.csv").read_bytes()
    (out / "baseline_reproduced.csv").write_bytes(content)
    assert replay.digest(out / "baseline_reproduced.csv") == cfg["baseline_submission_sha256"]
    replay.log("Strongest ten-member baseline and CSV reproduced exactly; replaying PVT validation")
    model = pvt.make_model(training, cfg["architecture"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.cuda().to(memory_format=torch.channels_last).eval()
    validation_path = ROOT / training["reference_experiment_root"] / "cache/primary_352.npy"
    validation = common.predict(model, np.load(validation_path, mmap_mode="r"), training, ref)
    stored_path = train_out / cfg["architecture"] / "supervised/candidate/epoch_800_student.npz"
    stored = dict(np.load(stored_path))
    replay_check = replay.compare_probabilities(validation, stored["primary"], 1e-5, 1e-4)
    assert replay_check["within_tolerance"] and replay_check["predicted_labels_different"] == 0
    common.save_array(out / "B3_validation.npy", validation)
    replay.log("Inferring PVT-B3 on 2010 official test images, 352px flip TTA")
    test = np.load(team / "cache/test_352.npy", mmap_mode="r")
    assert test.shape == (2010, 352, 352, 3) and test.dtype == np.uint8
    probability = common.predict(model, test, training, ref)
    replay.compare_probabilities(probability, probability, 0, 0)
    common.save_array(out / "B3_test.npy", probability)
    combined = replace_member(families, probability, cfg["replacement_fold"])
    balanced = ref.balance(combined, prior, cfg["balance_iterations"])
    neural_labels = balanced.argmax(1)
    labels, mask = sift_labels(neural_labels, matches, cfg["sift_minimum_inliers"])
    common.save_array(out / "ensemble_raw_test.npy", combined)
    common.save_array(out / "ensemble_balanced_test.npy", balanced)
    path = out / "LOCAL-PROP-008-B3_f2replace_submission.csv"
    path.write_bytes(serialize_submission(sample.id, labels))
    issues = validate(path, ROOT / training["dataset_root"] / "sample_submission.csv"); assert not issues, issues
    changed = labels != baseline_labels; assert changed.any(), "Identical candidate; do not consume a submission"
    pd.DataFrame(dict(id=sample.id[changed], baseline=baseline_labels[changed], candidate=labels[changed])).to_csv(out / "changed_predictions.csv", index=False)
    for name, sha in hashes.items():
        assert replay.digest(name) == sha, f"Changed input during inference: {name}"
    summary = dict(experiment_id=cfg["experiment_id"], at=replay.now(), duration_seconds=time.monotonic()-started,
        selected_checkpoint=cfg["checkpoint"], checkpoint_sha256=cfg["checkpoint_sha256"],
        training_spec_sha256=cfg["training_spec_sha256"], config_sha256=replay.digest(config), runner_sha256=replay.digest(Path(__file__)),
        authorization=cfg["authorization"], architecture=cfg["architecture"], selection=cfg["selection"],
        composition=cfg["composition"], new_checkpoint_total_weight=.1, original_members_retained=9,
        balance_iterations=cfg["balance_iterations"], sift_minimum_inliers=cfg["sift_minimum_inliers"],
        local_gate=complete["screening"], local_pair_metrics=complete["final_metrics"]["candidate"],
        validation_replay=replay_check, baseline_csv_identical=True, baseline_public_score=cfg["baseline_public_score"],
        full_ensemble_oof_available=False, local_validation_scope="PVT plus matching R34d fold2 only; no full replacement-ensemble OOF",
        rows=len(sample), csv_format_passed=True, labels_changed_vs_baseline=int(changed.sum()), sift_accepted_rows=int(mask.sum()),
        sift_labels_changed=int((labels != neural_labels).sum()), submission_file=str(path.relative_to(ROOT)),
        submission_sha256=replay.digest(path), probability_sha256=replay.digest(out / "B3_test.npy"),
        input_sha256=hashes, all_inputs_unchanged=True, kaggle_upload_performed=False)
    replay.write_json(out / "preparation.json", summary)
    replay.log(json.dumps({key: summary[key] for key in ("duration_seconds", "labels_changed_vs_baseline", "submission_sha256")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/LOCAL-PROP-008-B3-PUBLIC.yaml")
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    with (out / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(cfg, args.config, out)


if __name__ == "__main__":
    main()
