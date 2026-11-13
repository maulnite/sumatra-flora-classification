"""Reproducible LOCAL-PROP-001 A/B runner, with conditional Kaggle submission.

Run with .venv/bin/python. Upstream augmentation policy is vendored in local_dq.
The local holdout is a separate protocol from the team's scene237 OOF.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import multiprocessing as mp

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
import timm
from timm.data import Mixup
from timm.utils import ModelEmaV3
import yaml
from sklearn.metrics import confusion_matrix, f1_score

try:
    from . import local_dq as dq
    from .validate_submission import validate
except ImportError:
    import local_dq as dq
    from validate_submission import validate

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("v4rot", "clean", "v4rot_blur_06", "v4rot_blur_10")
COMPETITION = "dateraquest-2-0-machine-learning-competition"


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"{now()} {message}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def parse_cli_json(output):
    """The Kaggle CLI prints its version warning to stdout before the JSON."""
    lines = output.splitlines()
    while lines and (not lines[0].strip() or lines[0].startswith("Warning:")):
        lines.pop(0)
    return json.loads("\n".join(lines))


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256(json.dumps(values).encode()).digest()[:4], "little")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    manifest = ROOT / cfg["split"]["manifest"]
    if sha(manifest) != cfg["split"]["manifest_sha256"]:
        raise ValueError("Frozen manifest hash differs")
    rows = read_csv(manifest)
    official = read_csv(ROOT / cfg["dataset_root"] / "train.csv")
    if len(rows) != len(official) or any(a["id"] != b["id"] or a["label"] != b["label"] for a, b in zip(rows, official)):
        raise ValueError("Manifest does not align with official metadata")
    for role, count in (("train", 2378), ("validation", 634), ("purged_scene_neighbor", 2)):
        subset = [r for r in rows if r["role"] == role]
        assert len(subset) == count
        if role != "purged_scene_neighbor":
            assert {int(r["label"]) for r in subset} == set(range(30))
    train, val = ([r for r in rows if r["role"] == role] for role in ("train", "validation"))
    for group in cfg["split"]["require_disjoint_groups"]:
        assert not ({r[group] for r in train} & {r[group] for r in val})
    return cfg, train, val


def fingerprint(cfg):
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "timm", "numpy", "albumentations", "albucore", "opencv-python-headless", "scikit-learn")}
    return dict(config=cfg, source_sha256={p: sha(ROOT / "scripts" / p) for p in
                ("run_local_experiment.py", "local_dq.py")}, versions=versions,
                python=sys.version, interpreter=sys.executable)


def rotate(img, angle, reflect):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101 if reflect else cv2.BORDER_CONSTANT, borderValue=0)


def gaussian(img, sigma):
    return cv2.GaussianBlur(img, (7, 7), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def post_blur(img, seed, probability):
    rng = np.random.RandomState(seed)
    apply = rng.rand() < probability
    sigma = rng.uniform(.4, 1.)
    return gaussian(img, sigma) if apply else img


def validation_item(args):
    path, index = args
    cv2.setNumThreads(1)
    img = dq.read_rgb(path)
    clean = dq.squash(img)
    rng = np.random.RandomState(index + 1_000_003)
    rot = rotate(img, rng.uniform(-180, 180), rng.rand() < .5)
    # The vendored Lambda functions use global NumPy; Compose has a separate RNG.
    np.random.seed(index)
    dq.corrupt_v4.set_random_seed(index)
    v4 = dq.squash(dq.corrupt_v4(image=rot)["image"])
    return v4, clean, gaussian(v4, .6), gaussian(v4, 1.)


def prepare_cache(cfg, val):
    out = ROOT / cfg["output_root"] / "cache"
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(protocol=cfg["validation"]["protocol"], manifest=cfg["split"]["manifest_sha256"],
                     augmentation=sha(ROOT / "scripts/local_dq.py"),
                     versions=fingerprint(cfg)["versions"], ids=[r["id"] for r in val])
    meta_path = out / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["signature"] != signature:
            raise ValueError("Cache provenance differs; investigate rather than replacing frozen views")
        for name, digest in meta["sha256"].items():
            assert sha(out / name) == digest, name
        log("Frozen validation cache verified")
        return out
    arrays = {name: np.lib.format.open_memmap(out / f"{name}.tmp.npy", mode="w+", dtype=np.uint8,
                                            shape=(len(val), 224, 224, 3)) for name in CONDITIONS}
    args = [(str(ROOT / cfg["dataset_root"] / "train" / r["id"]), int(r["row_index"])) for r in val]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        for i, result in enumerate(pool.map(validation_item, args, chunksize=4)):
            for name, image in zip(CONDITIONS, result):
                arrays[name][i] = image
            if (i + 1) % 100 == 0:
                log(f"Validation cache {i + 1}/{len(val)}")
    for name, arr in arrays.items():
        arr.flush()
    arrays.clear()
    for name in CONDITIONS:
        (out / f"{name}.tmp.npy").replace(out / f"{name}.npy")
    write_json(meta_path, dict(signature=signature, created_at=now(),
                             sha256={f"{n}.npy": sha(out / f"{n}.npy") for n in CONDITIONS}))
    log("Four validation conditions frozen")
    return out


class TrainingViews(Dataset):
    def __init__(self, rows, data, probability, seed=42):
        self.rows, self.data, self.probability, self.seed = rows, Path(data), probability, seed

    def __len__(self):
        return 2 * len(self.rows)

    def __getitem__(self, key):
        index, epoch = key
        corrupt = index >= len(self.rows)
        row = self.rows[index % len(self.rows)]
        identity = (self.seed, epoch, row["id"], "corrupt" if corrupt else "clean")
        base_seed = stable_seed(*identity, "base")
        np.random.seed(base_seed)
        random.seed(base_seed)
        img = dq.read_rgb(self.data / "train" / row["id"])
        rng = np.random.RandomState(stable_seed(*identity, "rotation"))
        if rng.rand() < .9:
            img = rotate(img, rng.uniform(-180, 180), rng.rand() < .5)
        transform = dq.corrupt_v4 if corrupt else dq.light_color
        transform.set_random_seed(base_seed)
        img = transform(image=img)["image"]
        dq.rrc.set_random_seed(stable_seed(*identity, "crop"))
        img = dq.rrc(image=img)["image"]
        if corrupt:
            img = post_blur(img, stable_seed(*identity, "post-blur"), self.probability)
        return np.ascontiguousarray(img.transpose(2, 0, 1)), int(row["label"])


class EpochSampler(Sampler):
    def __init__(self, count, seed=42):
        self.count, self.seed, self.epoch = count, seed, 1

    def __len__(self):
        return self.count

    def __iter__(self):
        generator = torch.Generator().manual_seed(stable_seed(self.seed, self.epoch, "sampler"))
        return iter((i, self.epoch) for i in torch.randperm(2 * self.count, generator=generator)[:self.count].tolist())


def collate(batch):
    return np.stack([x for x, _ in batch]), np.array([y for _, y in batch], dtype=np.int64)


def worker_init(_):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def make_model():
    return timm.create_model("resnet18d", pretrained=False, num_classes=30, drop_rate=.2)


def normalize(x, device):
    x = torch.as_tensor(np.ascontiguousarray(x), device=device).float()
    mean = torch.as_tensor(dq.MEAN, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(dq.STD, device=device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()


@torch.no_grad()
def predict(model, array, device="cuda", batch_size=128):
    model.eval()
    result = []
    for offset in range(0, len(array), batch_size):
        x = normalize(array[offset:offset + batch_size].transpose(0, 3, 1, 2), device)
        with autocast(device):
            prob = (model(x).float().softmax(1) + model(x.flip(3)).float().softmax(1)) / 2
        result.append(prob.cpu().numpy())
    return np.concatenate(result)


def metric(labels, prob):
    return float(f1_score(labels, prob.argmax(1), labels=list(range(30)), average="macro", zero_division=0))


def save_checkpoint(path, state):
    tmp = Path(path).with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def smoke(cfg, train, val):
    torch.set_num_threads(4)
    seed_all(cfg["seed"])
    a = TrainingViews(train, ROOT / cfg["dataset_root"], 0.)
    b = TrainingViews(train, ROOT / cfg["dataset_root"], .25)
    for k in (0, 1, len(train), len(train) + 1):
        x, label = a[k, 1]
        repeated, _ = a[k, 1]
        assert np.array_equal(x, repeated)
        if k < len(train):
            assert np.array_equal(x, b[k, 1][0]), "Clean arm views differ"
        assert x.shape == (3, 224, 224) and x.dtype == np.uint8
    model = make_model()
    ema = ModelEmaV3(model, decay=.999, foreach=False)
    batch, labels = collate([a[0, 1], a[len(train), 1]])
    mix = Mixup(mixup_alpha=.2, cutmix_alpha=1., prob=.5, switch_prob=.5, label_smoothing=.1, num_classes=30)
    xb, yt = mix(normalize(batch, "cpu"), torch.tensor(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.05)
    loss = F.cross_entropy(model(xb), yt)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step()
    ema.update(model)
    out = ROOT / cfg["output_root"] / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / "roundtrip.pt", dict(model=model.state_dict(), ema=ema.state_dict(), rng=rng_state()))
    state = torch.load(out / "roundtrip.pt", map_location="cpu", weights_only=False)
    replica = ModelEmaV3(make_model(), decay=.999, foreach=False)
    replica.load_state_dict(state["ema"])
    ema.module.eval()
    replica.module.eval()
    with torch.no_grad():
        assert torch.equal(ema.module(xb), replica.module(xb))
    restore_rng(state["rng"])
    expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng(state["rng"])
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])
    write_json(out / "result.json", dict(passed=True, device="cpu", loss=float(loss), validation_rows=len(val),
               checked=["deterministic views", "clean A/B equality", "finite backward", "EMA save/load", "RNG roundtrip"], at=now()))
    log(f"CPU smoke passed, loss={loss.item():.4f}")


def gpu_smoke(cfg, train):
    seed_all(cfg["seed"])
    dataset = TrainingViews(train, ROOT / cfg["dataset_root"], 0.)
    batch, labels = collate([dataset[i, 1] for i in range(4)])
    model = make_model().to("cuda").to(memory_format=torch.channels_last)
    ema = ModelEmaV3(model, decay=.999, foreach=False)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.05)
    with autocast("cuda"):
        loss = F.cross_entropy(model(normalize(batch, "cuda")), torch.as_tensor(labels, device="cuda"))
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    ema.update(model)
    torch.cuda.synchronize()
    write_json(ROOT / cfg["output_root"] / "smoke/gpu_result.json", dict(passed=True, at=now(),
               device=torch.cuda.get_device_name(), precision="bf16", loss=float(loss),
               allocated_memory_bytes=torch.cuda.max_memory_allocated()))
    del model, ema, opt
    torch.cuda.empty_cache()
    log("GPU bf16 forward/backward/AdamW/EMA smoke passed")


def other_gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip() and line.split(",")[0].strip() != str(os.getpid())]


def wait_for_gpu(out, allow_sharing=False):
    if allow_sharing:
        return
    idle = 0
    while True:
        others = other_gpu_processes()
        if not others:
            idle += 1
            if idle >= 3:
                return
        else:
            idle = 0
        write_json(out / "status.json", dict(status="waiting_for_gpu", at=now(), other_gpu_processes=others,
                                             allow_sharing=allow_sharing))
        log(f"Waiting for GPU: {others or 'confirming idle'}")
        time.sleep(20)


def train_arm(cfg, rows, val, cache, arm, microbatch=64, device="cuda"):
    out = ROOT / cfg["output_root"] / cfg["arms"][arm]["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(cfg)
    identity["microbatch"] = microbatch
    identity["cache_metadata_sha256"] = sha(cache / "metadata.json")
    saved_fp = out / "fingerprint.json"
    if saved_fp.exists() and json.loads(saved_fp.read_text()) != identity:
        raise ValueError(f"Run provenance changed in {out}; refusing mismatched resume")
    if (out / "done.json").exists():
        log(f"{arm} already complete")
        return
    write_json(saved_fp, identity)
    seed_all(cfg["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = make_model()
    initial_hash = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    write_json(out / "initialization.json", dict(seed=cfg["seed"], sha256=initial_hash))
    model = model.to(device).to(memory_format=torch.channels_last)
    ema = ModelEmaV3(model, decay=.999, foreach=False)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.05)
    dataset = TrainingViews(rows, ROOT / cfg["dataset_root"], cfg["arms"][arm]["blur"]["probability_on_corrupt_view"], cfg["seed"])
    sampler = EpochSampler(len(rows), cfg["seed"])
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, drop_last=True, num_workers=4,
                        persistent_workers=True, collate_fn=collate, worker_init_fn=worker_init,
                        prefetch_factor=2, multiprocessing_context="spawn",
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    epochs = cfg["common"]["training"]["epochs"]
    warm, total = 3 * len(loader), epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else
                         .5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    mixer = Mixup(mixup_alpha=.2, cutmix_alpha=1., prob=.5, switch_prob=.5, label_smoothing=.1, num_classes=30)
    start, history = 1, []
    ckpt = out / "checkpoint.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        start, history = state["epoch"] + 1, state["history"]
        log(f"{arm}: resume epoch {start}")
    arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in CONDITIONS}
    labels = np.array([int(r["label"]) for r in val])
    start_time = time.monotonic()
    for epoch in range(start, epochs + 1):
        sampler.epoch = epoch
        model.train()
        loss_sum = 0.
        ep_start = time.monotonic()
        for images, y in loader:
            x = normalize(images, device)
            target = torch.as_tensor(y, device=device)
            x, target = mixer(x, target)
            opt.zero_grad(set_to_none=True)
            loss_value = 0.
            for offset in range(0, len(x), microbatch):
                fraction = len(x[offset:offset + microbatch]) / len(x)
                with autocast(device):
                    loss = F.cross_entropy(model(x[offset:offset + microbatch]), target[offset:offset + microbatch]) * fraction
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in {arm}, epoch {epoch}")
                loss.backward()
                loss_value += loss.item()
            opt.step()
            scheduler.step()
            ema.update(model)
            loss_sum += loss_value
        record = dict(epoch=epoch, loss=loss_sum / len(loader), seconds=time.monotonic() - ep_start)
        if epoch % 25 == 0 or epoch == epochs:
            for condition in (CONDITIONS if epoch == epochs else ("v4rot", "clean")):
                prob = predict(ema.module, arrays[condition], device)
                record[f"f1_{condition}"] = metric(labels, prob)
        history.append(record)
        write_json(out / "history.json", history)
        write_json(ROOT / cfg["output_root"] / "status.json", dict(status="training", arm=arm, at=now(), **record))
        log(f"{arm} {record}")
        if epoch % 10 == 0 or epoch == epochs:
            save_checkpoint(ckpt, dict(epoch=epoch, model=model.state_dict(), ema=ema.state_dict(),
                       optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(), history=history))
        if epoch == 10:
            write_json(out / "benchmark.json", dict(first_10_epochs_seconds=time.monotonic() - start_time,
                       estimated_total_minutes=(time.monotonic() - start_time) / 10 * epochs / 60))
    probs = {name: predict(ema.module, arrays[name], device) for name in CONDITIONS}
    np.savez(out / "validation_probabilities.npz", **probs, ids=np.array([r["id"] for r in val]), labels=labels)
    save_checkpoint(out / "model.pt", ema.module.state_dict())
    write_json(out / "done.json", dict(epoch=epochs, checkpoint="final EMA", at=now(),
               metrics={name: metric(labels, p) for name, p in probs.items()}, initialization_sha256=initial_hash,
               checkpoint_sha256=sha(out / "model.pt")))
    del loader, model, ema, opt
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def decision(metrics_a, metrics_b, criteria):
    delta = {key: metrics_b[key] - metrics_a[key] for key in CONDITIONS}
    blur_delta = (delta["v4rot_blur_06"] + delta["v4rot_blur_10"]) / 2
    checks = dict(primary_macro_f1_increases=delta["v4rot"] > 0,
                  blur_gain_reaches_threshold=blur_delta >= criteria["minimum_blur_metric_gain_B_minus_A"],
                  primary_drop_within_limit=delta["v4rot"] >= -criteria["maximum_v4rot_macro_f1_drop"],
                  clean_drop_within_limit=delta["clean"] >= -criteria["maximum_clean_macro_f1_drop"])
    return dict(delta=delta, mean_blur_delta=blur_delta, checks=checks, eligible_for_submission=all(checks.values()))


def compare(cfg, val):
    root = ROOT / cfg["output_root"]
    records, predictions = {}, {}
    for arm, arm_cfg in cfg["arms"].items():
        out = root / arm_cfg["run_id"]
        records[arm] = json.loads((out / "done.json").read_text())
        p = np.load(out / "validation_probabilities.npz")
        assert p["ids"].tolist() == [r["id"] for r in val]
        assert p["labels"].tolist() == [int(r["label"]) for r in val]
        predictions[arm] = {key: p[key] for key in CONDITIONS}
    assert records["control"]["initialization_sha256"] == records["post_resize_blur"]["initialization_sha256"]
    result = decision(records["control"]["metrics"], records["post_resize_blur"]["metrics"], cfg["screen_success_criteria"])
    result.update(at=now(), protocol=cfg["validation"]["protocol"], records=records,
                  comparison_scope="one local holdout, not team OOF or leaderboard gain")
    labels = np.array([int(r["label"]) for r in val])
    group_ids = np.array([int(r["expanded_group"]) for r in val])
    grouped = [[np.flatnonzero(group_ids == g) for g in np.unique(group_ids[labels == c])] for c in range(30)]
    rng = np.random.RandomState(43)
    deltas = []
    for _ in range(2000):
        selected = np.concatenate([np.concatenate([groups[k] for k in rng.randint(len(groups), size=len(groups))]) for groups in grouped])
        scores = {}
        for arm, probs in predictions.items():
            scores[arm] = {name: metric(labels[selected], p[selected]) for name, p in probs.items()}
        deltas.append([scores["post_resize_blur"][name] - scores["control"][name] for name in CONDITIONS])
    intervals = np.percentile(deltas, [2.5, 97.5], axis=0)
    result["paired_group_bootstrap_delta_95_percent"] = {name: intervals[:, i].tolist() for i, name in enumerate(CONDITIONS)}
    for arm, probs in predictions.items():
        result[arm + "_per_class_f1"] = {name: f1_score(labels, p.argmax(1), labels=list(range(30)), average=None, zero_division=0).tolist() for name, p in probs.items()}
        np.savez(root / f"{arm}_confusion_matrices.npz", **{name: confusion_matrix(labels, p.argmax(1), labels=list(range(30))) for name, p in probs.items()})
    write_json(root / "comparison.json", result)
    log(f"A/B result: {json.dumps({k: result[k] for k in ('delta', 'mean_blur_delta', 'checks', 'eligible_for_submission')})}")
    return result


def submission_authorization(result, override_reason=None):
    """Keep the scientific gate result intact when the user requests a submission."""
    if override_reason is not None:
        if not isinstance(override_reason, str) or not override_reason.strip():
            raise ValueError("An explicit user override requires a nonempty authorization reason")
        return dict(mode="explicit_user_override", reason=override_reason.strip(),
                    original_gate_passed=bool(result["eligible_for_submission"]))
    if not result["eligible_for_submission"]:
        raise ValueError("Candidate did not meet the predeclared submission gate")
    return dict(mode="conditional_gate_passed", original_gate_passed=True)


def infer(cfg, device="cuda", override_reason=None):
    root = ROOT / cfg["output_root"]
    result = json.loads((root / "comparison.json").read_text())
    authorization = submission_authorization(result, override_reason)
    source = root / cfg["arms"]["post_resize_blur"]["run_id"] / "model.pt"
    assert sha(source) == result["records"]["post_resize_blur"]["checkpoint_sha256"]
    model = make_model().to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(source, map_location=device, weights_only=False))
    template = ROOT / cfg["dataset_root"] / "sample_submission.csv"
    rows = read_csv(template)
    arrays = np.stack([dq.squash(dq.read_rgb(ROOT / cfg["dataset_root"] / "test" / r["id"])) for r in rows])
    probs = predict(model, arrays, device)
    assert probs.shape == (2010, 30) and np.isfinite(probs).all()
    np.save(root / "test_probabilities.npy", probs)
    path = root / "LOCAL-PROP-001-B_submission.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows(dict(id=r["id"], label=int(label)) for r, label in zip(rows, probs.argmax(1)))
    errors = validate(path, template)
    if errors:
        raise ValueError(errors)
    write_json(root / "submission_manifest.json", dict(at=now(), file=str(path), sha256=sha(path),
               model_sha256=sha(source), comparison_sha256=sha(root / "comparison.json"), rows=2010,
               training_rows=2378, trained_on_all_official_train=False, ensemble=False,
               source="candidate local-holdout final-epoch-300 EMA", pretrained=False,
               authorization=authorization, inference_fingerprint=fingerprint(cfg)))
    log(f"Validated prediction CSV: {path}")
    return path


def submit(cfg, override_reason=None):
    root = ROOT / cfg["output_root"]
    policy = cfg.get("submission", {})
    assert policy.get("conditional_submission_authorized") is True, "Missing explicit authorization record"
    result = json.loads((root / "comparison.json").read_text())
    authorization = submission_authorization(result, override_reason)
    ledger = root / "submission_attempt.json"
    if ledger.exists():
        raise RuntimeError("Submission attempt already recorded. Reconcile remote state before retrying")
    manifest = json.loads((root / "submission_manifest.json").read_text())
    assert manifest["authorization"] == authorization, "Inference and submission authorization differ"
    assert manifest["comparison_sha256"] == sha(root / "comparison.json")
    path = Path(manifest["file"])
    assert sha(path) == manifest["sha256"]
    assert not validate(path, ROOT / cfg["dataset_root"] / "sample_submission.csv")
    cmd = ["kaggle-orie", "competitions", "submissions", COMPETITION, "--page-size", "200", "--format", "json"]
    history = subprocess.run(cmd, capture_output=True, text=True, check=True)
    (root / "submissions_before.raw.txt").write_text(history.stdout)
    submissions = parse_cli_json(history.stdout)
    if isinstance(submissions, dict):
        submissions = submissions["submissions"]
    assert all("date" in r for r in submissions), "Unknown submission timestamp schema"
    write_json(root / "submissions_before.json", submissions)
    today = datetime.now(timezone.utc).date().isoformat()
    used = sum(str(r.get("date", "")).startswith(today) for r in submissions)
    if used >= 3:
        write_json(root / "status.json", dict(status="submission_waiting_daily_quota", used_today=used, at=now()))
        raise RuntimeError("Three team submissions already recorded today; wait for quota reset")
    # No retries: a timeout can follow a successful remote upload.
    gate_note = "gate-user-override" if authorization["mode"] == "explicit_user_override" else "gate-pass"
    message = f"LOCAL-PROP-001-B scratch r18d224 R-v4 postresize-blur p.25 s.4-1 ep300 EMA flip localholdout634 {gate_note} sha={manifest['sha256'][:16]}"
    attempt = dict(at=now(), status="attempting", file_sha256=manifest["sha256"], message=message,
                   authorization=authorization)
    write_json(ledger, attempt)
    completed = subprocess.run(["kaggle-orie", "competitions", "submit", COMPETITION, "-f", str(path), "-m", message], capture_output=True, text=True)
    attempt.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
                   status="submitted" if completed.returncode == 0 else "requires_remote_reconciliation")
    write_json(ledger, attempt)
    log(f"Kaggle submission status: {attempt['status']}")
    if completed.returncode:
        raise RuntimeError("Kaggle submission failed or is ambiguous; inspect recorded response without blind retry")
    for _ in range(20):
        latest = subprocess.run(cmd, capture_output=True, text=True, check=True)
        rows = parse_cli_json(latest.stdout)
        if isinstance(rows, dict):
            rows = rows["submissions"]
        matches = [r for r in rows if r.get("description") == message]
        if matches:
            write_json(root / "submission_result.json", matches[0])
            if "COMPLETE" in matches[0].get("status", "") or "ERROR" in matches[0].get("status", ""):
                log(f"Kaggle result: {matches[0]}")
                break
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prep", "smoke", "train", "compare", "infer", "submit", "auto"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs/LOCAL-PROP-001.yaml")
    ap.add_argument("--arm", choices=["control", "post_resize_blur"], default="control")
    ap.add_argument("--allow-gpu-sharing", action="store_true")
    ap.add_argument("--microbatch", type=int, choices=[32, 64], default=64)
    ap.add_argument("--submission-override-reason", default=None,
                    help="Record an explicit user request to submit despite the original validation gate (infer/submit only)")
    args = ap.parse_args()
    if args.submission_override_reason is not None and args.command not in ("infer", "submit"):
        ap.error("Submission overrides are allowed only for explicit infer/submit commands")
    cfg, rows, val = load_config(args.config)
    root = ROOT / cfg["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    # Every mutating operation uses one project-specific lock.
    with (root / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(root / "runner.json", dict(pid=os.getpid(), command=args.command, at=now(), allow_gpu_sharing=args.allow_gpu_sharing))
        if args.command in ("prep", "auto"):
            cache = prepare_cache(cfg, val)
        else:
            cache = root / "cache"
        if args.command in ("smoke", "auto"):
            smoke(cfg, rows, val)
        if args.command in ("train", "auto", "infer"):
            wait_for_gpu(root, args.allow_gpu_sharing)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in the selected project environment")
            if args.allow_gpu_sharing:
                torch.cuda.set_per_process_memory_fraction(.20)
            if args.command == "auto":
                gpu_smoke(cfg, rows)
        if args.command == "train":
            train_arm(cfg, rows, val, cache, args.arm, args.microbatch)
        if args.command == "auto":
            for arm in ("control", "post_resize_blur"):
                train_arm(cfg, rows, val, cache, arm, args.microbatch)
        if args.command in ("compare", "auto"):
            result = compare(cfg, val)
            if not result["eligible_for_submission"]:
                write_json(root / "status.json", dict(status="complete_no_submission_gate_failed", at=now(), checks=result["checks"]))
                return
        if args.command in ("infer", "auto"):
            infer(cfg, override_reason=args.submission_override_reason)
        if args.command in ("submit", "auto"):
            submit(cfg, override_reason=args.submission_override_reason)
            write_json(root / "status.json", dict(status="complete_submitted", at=now()))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
