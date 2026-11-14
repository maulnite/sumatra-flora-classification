"""Replay all ten best-team checkpoints and compare against saved predictions.

Use the documented team Python, whose versions are checked against W&B records.
Only selected inference functions are loaded from a frozen copy of team source.
Team inputs remain read-only; no training or remote submission is performed.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import timm
import yaml

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("torch", "torchvision", "timm", "numpy", "pandas", "opencv-python-headless", "albumentations", "scikit-learn")
_WORKER = None


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"{now()} {message}", flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def compare_probabilities(actual, expected, atol, rtol):
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError("Probability shape/order mismatch")
    for array in (actual, expected):
        if not np.isfinite(array).all() or np.any(array < 0) or not np.allclose(array.sum(1), 1, atol=1e-5):
            raise ValueError("Invalid probability values")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return dict(shape=list(actual.shape), array_exactly_equal=bool(np.array_equal(actual, expected)),
                within_tolerance=bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
                max_absolute_difference=float(difference.max()), mean_absolute_difference=float(difference.mean()),
                row_max_difference_p99=float(np.percentile(difference.max(1), 99)),
                predicted_labels_different=int((actual.argmax(1) != expected.argmax(1)).sum()))


def load_reference(source):
    """Avoid importing train.py's training/W&B/launcher side effects."""
    spec = importlib.util.spec_from_file_location("reproduction_reference_dq", source / "dq.py")
    dq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dq)
    selected = {"rot_img", "make_model", "norm_stats", "norm", "predict"}
    tree = ast.parse((source / "train.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    assert {n.name for n in nodes} == selected
    namespace = dict(cv2=dq.cv2, np=np, torch=torch, timm=timm, functools=functools, dq=dq, DEV="cuda")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source / "train.py"), "exec"), namespace)
    tree = ast.parse((source / "evalres.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("val_rot_size", "balance")]
    namespace["T"] = SimpleNamespace(rot_img=namespace["rot_img"])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source / "evalres.py"), "exec"), namespace)
    return SimpleNamespace(**namespace)


def freeze_file(path, destination):
    before = digest(path)
    if destination.exists():
        if digest(destination) != before:
            raise ValueError(f"Frozen reference changed: {path}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    if digest(destination) != before or digest(path) != before:
        raise ValueError(f"Reference changed during snapshot: {path}")
    return before


def preflight(cfg, out):
    team = Path(cfg["team_root"])
    source = out / "reference_sources"
    hashes = {}
    for name in ("train.py", "dq.py", "evalres.py"):
        hashes[name] = freeze_file(team / name, source / name)
    versions = {p: importlib.metadata.version(p) for p in VERSIONS}
    train = pd.read_csv(ROOT / cfg["dataset_root"] / "train.csv")
    test = pd.read_csv(ROOT / cfg["dataset_root"] / "sample_submission.csv")
    data = team / "dateraquest-2-0-machine-learning-competition"
    assert train.equals(pd.read_csv(data / "train.csv")) and test.equals(pd.read_csv(data / "sample_submission.csv"))
    folds = np.load(team / "cache/scene_folds.npy")
    metadata = pd.read_csv(team / "cache/scene_folds.csv")
    assert np.array_equal(metadata.id, train.id) and np.array_equal(metadata.label, train.label)
    assert np.array_equal(folds, metadata.fold)
    assert np.bincount(folds).tolist() == cfg["expected_fold_sizes"]
    assert (metadata.groupby("group_id").fold.nunique() == 1).all()
    checks = []
    for member in cfg["members"]:
        run = member["run"]
        for fold in cfg["folds"]:
            directory = team / "runs" / f"{run}_f{fold}"
            c = json.loads((directory / "cfg.json").read_text())
            assert c["arch"] == member["architecture"] and c["img"] == member["training_size"]
            assert c["epochs"] == 600 and c["tta"] == cfg["tta"] and c["attn"] == "none" and c["pool"] == "avg"
            assert (directory / "done.json").exists()
            for name in ("model.pt", "cfg.json"):
                path = directory / name
                hashes[str(path.relative_to(team))] = digest(path)
            records = sorted((directory / "wandb").glob("run-*/files/requirements.txt"))
            if not records:
                raise ValueError(f"Original environment record missing: {directory}")
            original = dict(line.split("==", 1) for line in records[-1].read_text().splitlines() if "==" in line)
            original = {p: original[p] for p in VERSIONS}
            if original != versions:
                raise ValueError(f"Environment differs from original run: {directory}")
            checks.append(dict(run=run, fold=fold, versions_match_original=True))
        size = member["inference_size"]
        for rel in (f"cache/corr4r_{size}.npy", f"cache/test_{size}.npy", f"runs/probs/{run}_{size}_oof.npy", f"runs/probs/{run}_{size}_test.npy"):
            hashes[rel] = digest(team / rel)
    for rel in ("cache/scene_folds.csv", "cache/scene_folds.npy", "runs/sift_test.csv", "runs/best_sift_submission.csv"):
        hashes[rel] = digest(team / rel)
    assert hashes["runs/best_sift_submission.csv"] == cfg["expected_submission_sha256"]
    evidence = dict(at=now(), python=sys.executable, versions=versions, environment_checks=checks,
                    input_sha256=hashes, train_rows=len(train), test_rows=len(test),
                    cuda_flags=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
                        cudnn_deterministic=torch.backends.cudnn.deterministic,
                        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32))
    for key, value in evidence["cuda_flags"].items():
        assert value == cfg[key], f"Unexpected backend flag {key}"
    write_json(out / "preflight.json", evidence)
    return train, test, folds, source, evidence


def run_checkpoints(cfg, out, folds, reference):
    team = Path(cfg["team_root"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(cfg["gpu_memory_fraction"])
    atol, rtol = (cfg["probability_tolerance"][k] for k in ("absolute", "relative"))
    results = {}
    for member in cfg["members"]:
        name, size = member["run"], member["inference_size"]
        validation = np.load(team / f"cache/corr4r_{size}.npy", mmap_mode="r")
        test = np.load(team / f"cache/test_{size}.npy", mmap_mode="r")
        expected_oof = np.load(team / f"runs/probs/{name}_{size}_oof.npy")
        all_oof = np.zeros_like(expected_oof)
        test_folds = []
        for fold in cfg["folds"]:
            destination = out / "predictions" / f"{name}_f{fold}"
            destination.mkdir(parents=True, exist_ok=True)
            paths = (destination / "validation.npy", destination / "test.npy")
            indices = np.flatnonzero(folds == fold)
            start = time.monotonic()
            if not all(p.exists() for p in paths):
                original = team / "runs" / f"{name}_f{fold}"
                config = json.loads((original / "cfg.json").read_text())
                model = reference.make_model(config, 30)
                state = torch.load(original / "model.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                model = model.to("cuda").to(memory_format=torch.channels_last).eval()
                # Actual checkpoint forward passes. Reference .npy is only used for comparison.
                val_prob = reference.predict(model, np.asarray(validation[indices]), "flip", bs=cfg["batch_size"])
                test_prob = reference.predict(model, np.asarray(test), "flip", bs=cfg["batch_size"])
                for path, array in zip(paths, (val_prob, test_prob)):
                    tmp = path.with_suffix(".tmp.npy")
                    np.save(tmp, array)
                    tmp.replace(path)
                del model, state
                torch.cuda.empty_cache()
            else:
                val_prob, test_prob = (np.load(p) for p in paths)
            all_oof[indices] = val_prob
            test_folds.append(test_prob)
            check = compare_probabilities(val_prob, expected_oof[indices], atol, rtol)
            write_json(destination / "comparison.json", dict(run=name, fold=fold, seconds=time.monotonic() - start, oof=check))
            write_json(out / "status.json", dict(status="checkpoint_inference", run=name, fold=fold, at=now(), oof=check))
            log(f"{name} fold {fold}: max difference={check['max_absolute_difference']:.9g}, changed OOF labels={check['predicted_labels_different']}, {time.monotonic()-start:.1f}s")
        # Match the team's original float32 reduction and fold order exactly.
        mean_test = np.mean(test_folds, 0)
        np.save(out / f"{name}_{size}_oof.npy", all_oof)
        np.save(out / f"{name}_{size}_test.npy", mean_test)
        results[name] = dict(oof=compare_probabilities(all_oof, expected_oof, atol, rtol),
                test=compare_probabilities(mean_test, np.load(team / f"runs/probs/{name}_{size}_test.npy"), atol, rtol))
        write_json(out / "probability_comparison.json", results)
    return results


def init_preprocessing(source, cfg):
    global _WORKER
    reference = load_reference(Path(source))
    reference.cv2.setNumThreads(1)
    torch.set_num_threads(1)
    sizes = sorted({m["inference_size"] for m in cfg["members"]})
    caches = {(split, size): np.load(Path(cfg["team_root"]) / "cache" /
                (f"corr4r_{size}.npy" if split == "train" else f"test_{size}.npy"), mmap_mode="r")
              for split in ("train", "test") for size in sizes}
    _WORKER = reference, cfg, sizes, caches


def replay_image(job):
    reference, cfg, sizes, caches = _WORKER
    split, index, identity = job
    project_path = ROOT / cfg["dataset_root"] / split / identity
    original_path = Path(cfg["team_root"]) / "dateraquest-2-0-machine-learning-competition" / split / identity
    result = dict(split=split, index=index, id=identity, raw_file_identical=digest(project_path) == digest(original_path), views={})
    for size in sizes:
        actual = (reference.val_rot_size((str(project_path), index, size)) if split == "train" else
                  reference.dq.squash(reference.dq.read_rgb(project_path), size))
        expected = caches[split, size][index]
        exact = np.array_equal(actual, expected)
        diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16)) if not exact else None
        result["views"][str(size)] = dict(exact=bool(exact), maximum_pixel_difference=int(diff.max()) if diff is not None else 0,
                                         mean_pixel_difference=float(diff.mean()) if diff is not None else 0.)
    return result


def replay_preprocessing(cfg, out, source, train, test):
    jobs = [(split, i, row.id) for split, frame in (("train", train), ("test", test)) for i, row in enumerate(frame.itertuples())]
    records = []
    with ProcessPoolExecutor(max_workers=cfg["preprocessing_workers"], mp_context=mp.get_context("spawn"),
                             initializer=init_preprocessing, initargs=(str(source), cfg)) as pool:
        for record in pool.map(replay_image, jobs, chunksize=4):
            records.append(record)
            if len(records) % 500 == 0:
                write_json(out / "status.json", dict(status="preprocessing_replay", images_done=len(records), images_total=len(jobs), at=now()))
                log(f"Raw-image preprocessing verified {len(records)}/{len(jobs)}")
    sizes = sorted({m["inference_size"] for m in cfg["members"]})
    summary = dict(images=len(records), raw_files_different=sum(not r["raw_file_identical"] for r in records),
                   views={f"{split}_{size}": dict(rows=sum(r["split"] == split for r in records),
                       unequal_images=sum(not r["views"][str(size)]["exact"] for r in records if r["split"] == split),
                       maximum_pixel_difference=max(r["views"][str(size)]["maximum_pixel_difference"] for r in records if r["split"] == split))
                   for split in ("train", "test") for size in sizes})
    write_json(out / "preprocessing_comparison.json", summary)
    write_json(out / "preprocessing_image_results.json", records)
    return summary


def assemble_final(cfg, out, train, test, reference):
    team = Path(cfg["team_root"])
    mean = {kind: np.mean([np.load(out / f"{m['run']}_{m['inference_size']}_{kind}.npy") for m in cfg["members"]], 0) for kind in ("oof", "test")}
    prior = np.bincount(train.label, minlength=30) / len(train)
    ptest = reference.balance(mean["test"], prior, cfg["prior_balance_iterations"])
    labels = ptest.argmax(1)
    matches = pd.read_csv(team / "runs/sift_test.csv")
    assert np.array_equal(matches.id, test.id)
    mask = (matches.inlier >= cfg["sift_minimum_inliers"]) & (matches.label >= 0)
    replaced = int((labels[mask] != matches.label.to_numpy()[mask]).sum())
    labels[mask] = matches.label.to_numpy()[mask]
    path = out / "recomputed_best_sift_submission.csv"
    pd.DataFrame(dict(id=test.id, label=labels.astype(int))).to_csv(path, index=False)
    from scripts.validate_submission import validate
    assert not validate(path, ROOT / cfg["dataset_root"] / "sample_submission.csv")
    score = lambda p: float(f1_score(train.label, p.argmax(1), labels=list(range(30)), average="macro", zero_division=0))
    return dict(raw_neural_oof_macro_f1=score(mean["oof"]),
                balanced_neural_oof_macro_f1=score(reference.balance(mean["oof"], prior, cfg["prior_balance_iterations"])),
                sift_replacements=replaced, csv_sha256=digest(path),
                csv_identical_to_reference=path.read_bytes() == (team / "runs/best_sift_submission.csv").read_bytes(),
                csv_labels_different=int((labels != pd.read_csv(team / "runs/best_sift_submission.csv").label.to_numpy()).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/BASELINE-INFERENCE-REPRO-001.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(config=cfg, runner_sha256=digest(Path(__file__)))
    existing = out / "specification.json"
    if existing.exists() and json.loads(existing.read_text()) != identity:
        raise ValueError("Frozen reproduction specification changed")
    with (out / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(existing, identity)
        write_json(out / "status.json", dict(status="preflight", pid=__import__('os').getpid(), at=now()))
        started = time.monotonic()
        train, test, folds, source, evidence = preflight(cfg, out)
        log("Original environment, checkpoints, data order, and frozen sources verified")
        reference = load_reference(source)
        comparisons = run_checkpoints(cfg, out, folds, reference)
        final = assemble_final(cfg, out, train, test, reference)
        log(f"Recomputed checkpoint CSV: {final}")
        preprocessing = replay_preprocessing(cfg, out, source, train, test) if cfg["preprocessing_replay"] else None
        for rel, original_hash in evidence["input_sha256"].items():
            assert digest(Path(cfg["team_root"]) / rel) == original_hash, f"Team input changed during run: {rel}"
        exact = all(c[k]["array_exactly_equal"] for c in comparisons.values() for k in ("oof", "test"))
        close = all(c[k]["within_tolerance"] for c in comparisons.values() for k in ("oof", "test"))
        pixels_exact = preprocessing is not None and preprocessing["raw_files_different"] == 0 and all(v["unequal_images"] == 0 for v in preprocessing["views"].values())
        passed = close and final["csv_identical_to_reference"] and pixels_exact
        result = dict(reproduction_id=cfg["reproduction_id"], at=now(), duration_seconds=time.monotonic() - started,
                      checkpoints_executed=10, versions=evidence["versions"], probabilities=comparisons,
                      all_probability_arrays_exact=exact, all_probability_arrays_within_tolerance=close,
                      final=final, preprocessing=preprocessing, all_requested_checks_passed=passed,
                      training_performed=False, kaggle_upload_performed=False, sift_matching_recomputed=False,
                      source_inputs_unchanged=True)
        write_json(out / "result.json", result)
        write_json(out / "status.json", dict(status="completed_passed" if passed else "completed_with_differences", at=now(),
                                             all_requested_checks_passed=passed))
        log(json.dumps(result))


if __name__ == "__main__":
    # Imports from the repository work both for the script and spawned workers.
    sys.path.insert(0, str(ROOT))
    main()
