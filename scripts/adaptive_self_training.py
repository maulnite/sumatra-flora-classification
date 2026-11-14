"""Matched fold-0 adaptive self-training pilot, initialized from team EMA weights.

An AdaMatch-inspired adaptation, with EMA weak-view pseudo-labels, clipped
distribution alignment, relative thresholding and source logit interpolation.
The source-only control has identical inputs and BatchNorm forward passes.
Run --prepare, then --smoke, then use the detached launcher. No upload code.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import gc

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from timm.utils import ModelEmaV3
import yaml

try:
    from . import reproduce_checkpoint_inference as reference_utils
except ImportError:
    import reproduce_checkpoint_inference as reference_utils

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("primary", "clean", "blur06", "blur10")
write_json, log, digest, now = (getattr(reference_utils, x) for x in ("write_json", "log", "digest", "now"))


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256(json.dumps(values).encode()).digest()[:4], "little")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(value, temporary); temporary.replace(path)


def save_npz(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays); temporary.replace(path)


def source_indices(fold_metadata, fold):
    selected = np.flatnonzero(fold_metadata.fold.to_numpy() != fold)
    heldout = np.flatnonzero(fold_metadata.fold.to_numpy() == fold)
    if set(fold_metadata.iloc[selected].group_id) & set(fold_metadata.iloc[heldout].group_id):
        raise ValueError("A held-out group is present in source training")
    return selected, heldout


class AdaptiveLabels:
    """Statistics depend only on source model outputs and unlabeled target outputs."""
    def __init__(self, cfg):
        self.cfg, self.source, self.target, self.confidence = cfg, None, None, None

    @torch.no_grad()
    def select(self, source_probability, target_probability, epoch, anchor_labels=None, anchor_agreement=None):
        ps, pt = source_probability.detach().float(), target_probability.detach().float()
        source, target = ps.mean(0), pt.mean(0)
        confidence = float(ps.max(1).values.mean())
        decay = self.cfg["statistics_ema_decay"]
        if self.source is None:
            self.source, self.target, self.confidence = source, target, confidence
        else:
            self.source = decay * self.source.to(ps.device) + (1 - decay) * source
            self.target = decay * self.target.to(ps.device) + (1 - decay) * target
            self.confidence = decay * self.confidence + (1 - decay) * confidence
        lower, upper = self.cfg["distribution_alignment_ratio_clip"]
        ratio = (self.source / self.target.clamp_min(1e-8)).clamp(lower, upper)
        aligned = pt * ratio
        aligned /= aligned.sum(1, keepdim=True)
        values, labels = aligned.max(1)
        threshold = self.cfg["relative_threshold_multiplier"] * self.confidence
        mask = values >= threshold
        if epoch <= self.cfg["initial_two_backbone_agreement_epochs"]:
            if anchor_labels is None or anchor_agreement is None:
                raise ValueError("Initial curriculum requires fold-safe anchor predictions")
            mask &= anchor_agreement & (labels == anchor_labels)
        return labels.detach(), mask.detach(), aligned.detach(), float(threshold)

    def state_dict(self):
        return dict(source=self.source.detach().cpu() if self.source is not None else None,
                    target=self.target.detach().cpu() if self.target is not None else None,
                    confidence=self.confidence)

    def load_state_dict(self, state):
        self.source, self.target, self.confidence = state["source"], state["target"], state["confidence"]


def masked_target_loss(logits, labels, selected):
    # Average over the entire target batch. Accepting fewer examples reduces the loss weight.
    losses = F.cross_entropy(logits.float(), labels.detach(), reduction="none")
    return (losses * selected.detach().to(losses.dtype)).mean()


def target_weight(cfg, epoch, microbatch, microbatches, enabled):
    if not enabled:
        return 0.
    progress = min(1., (epoch - 1 + microbatch / microbatches) / cfg["target_loss_warmup_epochs"])
    return cfg["target_loss_maximum_weight"] * .5 * (1 - math.cos(math.pi * progress))


def target_views(image, seed, cfg):
    rng = np.random.RandomState(seed)
    weak = image.copy()
    if rng.rand() < cfg["target_weak_horizontal_flip_probability"]:
        weak = weak[:, ::-1].copy()
    shift = cfg["target_weak_shift_pixels"]
    dx, dy = rng.randint(-shift, shift + 1, 2)
    weak = cv2.warpAffine(weak, np.float32([[1, 0, dx], [0, 1, dy]]), (weak.shape[1], weak.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    strong = weak.copy()
    if rng.rand() < cfg["target_strong_color_probability"]:
        amount = cfg["target_strong_brightness_contrast"]
        strong = np.clip(strong.astype(np.float32) * rng.uniform(1 - amount, 1 + amount) +
                         rng.uniform(-amount, amount) * 255, 0, 255).astype(np.uint8)
    if rng.rand() < cfg["target_strong_cutout_probability"]:
        lo, hi = cfg["target_strong_cutout_side_fraction"]
        side = max(1, round(min(strong.shape[:2]) * rng.uniform(lo, hi)))
        x, y = rng.randint(0, strong.shape[1]), rng.randint(0, strong.shape[0])
        strong[max(0, y-side//2):min(strong.shape[0], y+side//2), max(0, x-side//2):min(strong.shape[1], x+side//2)] = 0
    return np.ascontiguousarray(weak.transpose(2, 0, 1)), np.ascontiguousarray(strong.transpose(2, 0, 1))


class PairedBatches(Dataset):
    def __init__(self, cfg, metadata, source, target_count):
        self.cfg, self.metadata, self.indices, self.target_count = cfg, metadata, np.asarray(source), target_count
        if (metadata.iloc[self.indices].fold == cfg["fold"]).any():
            raise ValueError("Held-out images cannot enter the adaptation loader")
        self.runtime, self.target_cache, self.rrc = None, None, None
        self.source_batch = cfg["training"]["source_microbatch"]
        self.steps = len(source) // cfg["training"]["effective_source_batch"]
        self.microbatches = self.steps * cfg["training"]["gradient_accumulation"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(runtime=None, target_cache=None, rrc=None)
        return state

    def __len__(self):
        return self.microbatches

    def __getitem__(self, key):
        epoch, microbatch = key
        if self.runtime is None:
            self.runtime = reference_utils.load_reference(ROOT / self.cfg["reference_root"] / "reference_sources")
            self.target_cache = np.load(ROOT / self.cfg["output_root"] / "cache/target_288.npy", mmap_mode="r")
            size = self.cfg["training"]["image_size"]
            self.rrc = A.Compose([A.RandomResizedCrop(size=(size, size), scale=(.25, 1.), ratio=(.5, 2.)), A.HorizontalFlip()])
            cv2.setNumThreads(1)
        cfg, ref = self.cfg, self.runtime
        order = np.random.RandomState(stable_seed(cfg["seed"], epoch, "source-order")).permutation(self.indices)
        indices = order[microbatch*self.source_batch:(microbatch+1)*self.source_batch]
        sw, ss, labels = [], [], []
        for index in indices:
            row = self.metadata.iloc[index]
            seed = stable_seed(cfg["seed"], epoch, row.id)
            rng = np.random.RandomState(seed)
            image = ref.dq.read_rgb(ROOT / cfg["dataset_root"] / "train" / row.id)
            if rng.rand() < cfg["training"]["source_rotation_probability"]:
                angle = cfg["training"]["source_rotation_degrees"]
                image = ref.rot_img(image, rng.uniform(-angle, angle), rng.rand() < .5)
            views = []
            for kind, transform in (("weak", ref.dq.light_color), ("strong", ref.dq.corrupt_v4)):
                local_seed = stable_seed(seed, kind)
                np.random.seed(local_seed); random.seed(local_seed); transform.set_random_seed(local_seed)
                augmented = transform(image=image.copy())["image"]
                self.rrc.set_random_seed(stable_seed(seed, "crop"))
                augmented = self.rrc(image=augmented)["image"]
                views.append(np.ascontiguousarray(augmented.transpose(2, 0, 1)))
            sw.append(views[0]); ss.append(views[1]); labels.append(row.label)
        rng = np.random.RandomState(stable_seed(cfg["seed"], epoch, microbatch, "target-indices"))
        target_indices = rng.randint(self.target_count, size=cfg["training"]["target_microbatch"])
        tw, ts = [], []
        for position, index in enumerate(target_indices):
            weak, strong = target_views(self.target_cache[index], stable_seed(cfg["seed"], epoch, microbatch, position, int(index)), cfg["training"])
            tw.append(weak); ts.append(strong)
        return dict(source_weak=np.stack(sw), source_strong=np.stack(ss), labels=np.asarray(labels, dtype=np.int64),
                    target_weak=np.stack(tw), target_strong=np.stack(ts), target_indices=target_indices)


class EpochKeys(Sampler):
    def __init__(self, count):
        self.count, self.epoch = count, 1

    def __len__(self):
        return self.count

    def __iter__(self):
        return iter((self.epoch, i) for i in range(self.count))


def identity(batch):
    return batch


def init_worker(_):
    torch.set_num_threads(1); cv2.setNumThreads(1)


def normalize(images, ref):
    x = torch.as_tensor(np.ascontiguousarray(images), device="cuda").float()
    mean, std = ref.norm_stats()
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def forward_loss(model, teacher, statistics, batch, anchors, cfg, epoch, microbatch, microbatches, enabled, ref):
    n, nt = len(batch["labels"]), len(batch["target_indices"])
    x = normalize(np.concatenate([batch[k] for k in ("source_weak", "source_strong", "target_weak", "target_strong")]), ref)
    y = torch.as_tensor(batch["labels"], device="cuda")
    target_indices = batch["target_indices"]
    teacher.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        weak_logits = teacher(torch.cat([x[:n], x[2*n:2*n+nt]]))
        weak_probability = weak_logits.float().softmax(1)
        pseudo, mask, _, threshold = statistics.select(weak_probability[:n], weak_probability[n:], epoch,
                torch.as_tensor(anchors["labels"][target_indices], device="cuda"),
                torch.as_tensor(anchors["agreement"][target_indices], device="cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        source_logits = model(x[:2*n]).float()
        joint_logits = model(x).float()
        generator = torch.Generator(device="cuda").manual_seed(stable_seed(cfg["seed"], epoch, microbatch, "logit-interpolation"))
        mix = torch.rand(source_logits.shape, generator=generator, device="cuda")
        logits = mix * source_logits + (1 - mix) * joint_logits[:2*n]
        source_loss = .5 * (F.cross_entropy(logits[:n], y, label_smoothing=cfg["training"]["label_smoothing"]) +
                            F.cross_entropy(logits[n:], y, label_smoothing=cfg["training"]["label_smoothing"]))
        target_loss = masked_target_loss(joint_logits[2*n+nt:], pseudo, mask)
        weight = target_weight(cfg["adaptive"], epoch, microbatch, microbatches, enabled)
        loss = source_loss + weight * target_loss
    return loss, dict(source_loss=float(source_loss.detach()), target_loss=float(target_loss.detach()),
                      target_weight=weight, selected=int(mask.sum()), target_count=nt, threshold=threshold,
                      selected_classes=np.bincount(pseudo[mask].cpu().numpy(), minlength=30).tolist())


def make_model(cfg, ref, frozen=False):
    run = cfg["frozen_run"] if frozen else cfg["student_run"]
    path = Path(cfg["team_root"]) / "runs" / f"{run}_f{cfg['fold']}"
    model = ref.make_model(json.loads((path / "cfg.json").read_text()), 30)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True), strict=True)
    return model.to("cuda").to(memory_format=torch.channels_last)


def optimizer_and_scheduler(model, cfg, steps):
    c = cfg["training"]
    opt = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"], foreach=False)
    total, warm = c["epochs"] * steps, c["warmup_epochs"] * steps
    floor = c["minimum_learning_rate"] / c["learning_rate"]
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1) / max(1, warm) if s < warm else
                           floor + (1 - floor) * .5 * (1 + math.cos(math.pi * min(1, (s-warm)/max(1, total-warm)))))
    return opt, schedule


def ema_model(model, cfg):
    return ModelEmaV3(model, decay=cfg["training"]["ema_decay"], use_warmup=False, foreach=False)


@torch.inference_mode()
def predict(model, array, cfg, ref):
    model.eval(); parts = []
    size = cfg["validation"]["batch_size"]
    for start in range(0, len(array), size):
        x = normalize(np.array(array[start:start+size]).transpose(0, 3, 1, 2), ref)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = (model(x).float().softmax(1) + model(x.flip(3)).float().softmax(1)) / 2
        parts.append(p.cpu().numpy())
    return np.concatenate(parts)


def score(y, p):
    return float(f1_score(y, p.argmax(1), labels=list(range(30)), average="macro", zero_division=0))


def evaluate(model, cfg, ref, y, prior, out, frozen):
    student = {k: predict(model, np.load(out / f"cache/{k}_352.npy", mmap_mode="r"), cfg, ref) for k in CONDITIONS}
    p = {k: (student[k] + frozen[k]) / 2 for k in CONDITIONS}
    result = dict(primary=score(y, ref.balance(p["primary"], prior, cfg["validation"]["balance_iterations"])),
                  raw_primary=score(y, p["primary"]), clean=score(y, p["clean"]),
                  mean_blur=float(np.mean([score(y, p[k]) for k in ("blur06", "blur10")])),
                  single_student_primary=score(y, student["primary"]),
                  per_class_f1=f1_score(y, ref.balance(p["primary"], prior).argmax(1), labels=list(range(30)), average=None, zero_division=0).tolist())
    return result, student, p


def code_identity(config):
    paths = [config, ROOT / "scripts/adaptive_self_training.py",
             ROOT / "scripts/launch_adaptive_self_training.py", ROOT / "scripts/reproduce_checkpoint_inference.py"]
    return {str(p.resolve()): digest(p) for p in paths}


def setup(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    torch.set_num_threads(cfg["execution"]["cpu_threads"])
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(cfg["execution"]["gpu_memory_fraction"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(cfg["seed"])
    return reference_utils.load_reference(ROOT / cfg["reference_root"] / "reference_sources")


def metadata(cfg):
    team, data = Path(cfg["team_root"]), ROOT / cfg["dataset_root"]
    train, test = pd.read_csv(data / "train.csv"), pd.read_csv(data / "sample_submission.csv")
    frame = pd.read_csv(team / "cache/scene_folds.csv")
    assert np.array_equal(frame.id, train.id) and np.array_equal(frame.label, train.label)
    assert np.array_equal(frame.fold, np.load(team / "cache/scene_folds.npy"))
    si, vi = source_indices(frame, cfg["fold"])
    assert len(si) == cfg["validation"]["expected_train_rows"]
    assert len(vi) == cfg["validation"]["expected_validation_rows"]
    assert train.id.is_unique and test.id.is_unique and set(train.id).isdisjoint(test.id)
    return frame, test, si, vi


def save_array(path, array):
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, array); temporary.replace(path)


def prepare(cfg, config, out, ref):
    if (out / "spec.json").exists():
        verify_spec(cfg, config, out)
        log("Existing prepared inputs and immutable specification verified")
        return
    frame, test, si, vi = metadata(cfg)
    team, data = Path(cfg["team_root"]), ROOT / cfg["dataset_root"]
    original = json.loads((ROOT / cfg["reference_summary"]).read_text())
    versions = {p: importlib.metadata.version(p) for p in reference_utils.VERSIONS}
    assert versions == original["versions"], "Environment differs from reproduced reference"
    expected = original["provenance"]["input_sha256"]
    inputs = {}
    def pin(path, expected_hash=None):
        value = digest(path)
        if expected_hash is not None and value != expected_hash:
            raise ValueError(f"Reference input changed: {path}")
        inputs[str(path.resolve())] = value
    for name in ("train.py", "dq.py", "evalres.py"):
        pin(ROOT / cfg["reference_root"] / "reference_sources" / name, expected[name])
    for run in (cfg["student_run"], cfg["frozen_run"]):
        for name in ("cfg.json", "model.pt"):
            rel = f"runs/{run}_f{cfg['fold']}/{name}"
            pin(team / rel, expected[rel])
        for name in ("validation.npy", "test.npy"):
            pin(ROOT / cfg["reference_root"] / "predictions" / f"{run}_f{cfg['fold']}" / name)
    for rel in ("cache/scene_folds.csv", "cache/scene_folds.npy", "cache/corr4r_352.npy", "cache/corr4r_320.npy"):
        pin(team / rel, expected[rel])
    for name in ("train.csv", "sample_submission.csv", "class_mapping.csv"):
        pin(data / name)
    log("Pinning raw image hashes and constructing fold-isolated evaluation caches")
    paths = [data / "train" / x for x in frame.id] + [data / "test" / x for x in test.id]
    with ThreadPoolExecutor(max_workers=cfg["execution"]["cpu_threads"]) as pool:
        for path, value in zip(paths, pool.map(digest, paths)):
            inputs[str(path.resolve())] = value
    cache = out / "cache"; cache.mkdir(exist_ok=True)
    def read_resize(pair):
        path, size = pair
        return ref.dq.squash(ref.dq.read_rgb(path), size)
    with ThreadPoolExecutor(max_workers=cfg["execution"]["cpu_threads"]) as pool:
        target = np.stack(list(pool.map(read_resize, [(data / "test" / x, 288) for x in test.id])))
        save_array(cache / "target_288.npy", target); del target
        for size in (352, 320):
            primary = np.array(np.load(team / f"cache/corr4r_{size}.npy", mmap_mode="r")[vi])
            clean = np.stack(list(pool.map(read_resize, [(data / "train" / x, size) for x in frame.iloc[vi].id])))
            save_array(cache / f"primary_{size}.npy", primary)
            save_array(cache / f"clean_{size}.npy", clean)
            for name, sigma in (("blur06", .6), ("blur10", 1.)):
                blurred = np.stack([cv2.GaussianBlur(x, (7, 7), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101) for x in primary])
                save_array(cache / f"{name}_{size}.npy", blurred)
    # Only fold-0 teachers are allowed here. All-fold test averages would leak
    # the held-out validation labels indirectly through their training weights.
    p18, p34 = [np.load(ROOT / cfg["reference_root"] / "predictions" / f"{run}_f{cfg['fold']}" / "test.npy")
                for run in (cfg["student_run"], cfg["frozen_run"])]
    assert p18.shape == p34.shape == (len(test), 30)
    save_npz(cache / "anchors.npz", labels=((p18+p34)/2).argmax(1), agreement=p18.argmax(1)==p34.argmax(1))
    manifest = frame.copy(); manifest["role"] = np.where(frame.fold == cfg["fold"], "validation_only", "source_training")
    manifest.to_csv(out / "split_manifest.csv", index=False)
    test[["id"]].to_csv(out / "target_manifest.csv", index=False)
    log("Evaluating fixed R34d and initial R18d with the pilot's inference batch size")
    frozen_model = make_model(cfg, ref, frozen=True)
    frozen = {k: predict(frozen_model, np.load(cache / f"{k}_320.npy", mmap_mode="r"), cfg, ref) for k in CONDITIONS}
    save_npz(out / "frozen_r34d_probabilities.npz", **frozen)
    del frozen_model; gc.collect(); torch.cuda.empty_cache()
    model = make_model(cfg, ref)
    y = frame.iloc[vi].label.to_numpy(); prior = np.bincount(frame.label, minlength=30) / len(frame)
    baseline, student, ensemble = evaluate(model, cfg, ref, y, prior, out, frozen)
    comparisons = {}
    for run, actual in ((cfg["student_run"], student["primary"]), (cfg["frozen_run"], frozen["primary"])):
        saved = np.load(ROOT / cfg["reference_root"] / "predictions" / f"{run}_f{cfg['fold']}" / "validation.npy")
        comparisons[run] = reference_utils.compare_probabilities(actual, saved, atol=.01, rtol=.01)
        if comparisons[run]["max_absolute_difference"] > .01:
            raise ValueError("Initial inference drift exceeds 0.01 absolute probability")
    baseline.update(inference_batch_comparison=comparisons, at=now())
    write_json(out / "initial_baseline.json", baseline)
    save_npz(out / "initial_student.npz", **student); save_npz(out / "initial_ensemble.npz", **ensemble)
    del model; gc.collect(); torch.cuda.empty_cache()
    for path in sorted(cache.glob("*")) + [out / x for x in ("split_manifest.csv", "target_manifest.csv", "frozen_r34d_probabilities.npz",
                                                                           "initial_baseline.json", "initial_student.npz", "initial_ensemble.npz")]:
        pin(path)
    spec = dict(experiment_id=cfg["experiment_id"], created_at=now(), code_sha256=code_identity(config), input_sha256=inputs,
                versions=versions, source_rows=len(si), validation_rows=len(vi), target_rows=len(test),
                source_rows_per_epoch=(len(si)//64)*64, optimizer_updates_per_arm=(len(si)//64)*cfg["training"]["epochs"],
                teacher_folds=[cfg["fold"]], graph=False, training_performed=False,
                initial_baseline=baseline, backend=dict(cudnn_benchmark=False, matmul_allow_tf32=False))
    write_json(out / "spec.json", spec)
    log(f"Prepared: source={len(si)}, validation={len(vi)}, target={len(test)}, initial primary={baseline['primary']:.6f}")


def verify_spec(cfg, config, out):
    spec = json.loads((out / "spec.json").read_text())
    if spec["code_sha256"] != code_identity(config):
        raise ValueError("Code/config changed after preparation; create a new experiment output before training")
    if spec["versions"] != {p: importlib.metadata.version(p) for p in reference_utils.VERSIONS}:
        raise ValueError("Python environment changed")
    for path, expected in spec["input_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Pinned input changed: {path}")
    return spec


def checkpoint_state(model, ema, opt, schedule, statistics, epoch, history, spec_sha):
    return dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(), scheduler=schedule.state_dict(),
                statistics=statistics.state_dict(), epoch=epoch, history=history, rng=rng_state(), spec_sha256=spec_sha)


def load_training_checkpoint(path, model, ema, opt, schedule, statistics, spec_sha):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["spec_sha256"] != spec_sha:
        raise ValueError("Checkpoint belongs to a different immutable specification")
    model.load_state_dict(checkpoint["model"], strict=True); ema.load_state_dict(checkpoint["ema"], strict=True)
    opt.load_state_dict(checkpoint["optimizer"]); schedule.load_state_dict(checkpoint["scheduler"])
    statistics.load_state_dict(checkpoint["statistics"]); restore_rng(checkpoint["rng"])
    return checkpoint["epoch"], checkpoint["history"]


def smoke(cfg, config, out, ref):
    verify_spec(cfg, config, out)
    frame, test, si, _ = metadata(cfg)
    seed_all(cfg["seed"])
    dataset = PairedBatches(cfg, frame, si, len(test))
    batch = dataset[(21, 0)]; repeated = dataset[(21, 0)]
    assert all(np.array_equal(batch[k], repeated[k]) for k in batch), "View generation is not deterministic"
    assert batch["source_weak"].shape == (8, 3, 288, 288) and batch["target_weak"].shape == (24, 3, 288, 288)
    model = make_model(cfg, ref); model.train(); ema = ema_model(model, cfg)
    opt, schedule = optimizer_and_scheduler(model, cfg, dataset.steps)
    statistics = AdaptiveLabels(cfg["adaptive"]); anchors = dict(np.load(out / "cache/anchors.npz"))
    peak_before = torch.cuda.max_memory_allocated(); started = time.monotonic(); totals=[]
    opt.zero_grad(set_to_none=True)
    for i in range(cfg["training"]["gradient_accumulation"]):
        loss, values = forward_loss(model, ema.module, statistics, dataset[(21, i)], anchors, cfg, 21, i, len(dataset), True, ref)
        if not torch.isfinite(loss):
            raise ValueError("Smoke loss is non-finite")
        (loss / cfg["training"]["gradient_accumulation"]).backward(); totals.append(values)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"], error_if_nonfinite=True)
    assert all(p.grad is None for p in ema.module.parameters())
    opt.step(); schedule.step(); ema.update(model)
    spec_sha = digest(out / "spec.json")
    path = out / "smoke_checkpoint.pt"
    save_checkpoint(path, checkpoint_state(model, ema, opt, schedule, statistics, 0, [], spec_sha))
    expected = torch.load(path, map_location="cpu", weights_only=False)
    seed_all(999)
    epoch, history = load_training_checkpoint(path, model, ema, opt, schedule, statistics, spec_sha)
    assert epoch == 0 and history == []
    assert all(torch.equal(value.cpu(), expected["model"][key]) for key, value in model.state_dict().items())
    assert all(torch.equal(value.cpu(), expected["ema"][key]) for key, value in ema.state_dict().items())
    assert torch.equal(torch.get_rng_state(), expected["rng"]["torch"])
    result = dict(passed=True, at=now(), spec_sha256=spec_sha, full_optimizer_steps=1, microbatches=len(totals),
                  target_loss_exercised=all(x["target_weight"] == 1 for x in totals),
                  selected_examples=sum(x["selected"] for x in totals), target_examples=sum(x["target_count"] for x in totals),
                  gradient_norm=float(norm), duration_seconds=time.monotonic()-started,
                  gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  gpu_preexisting_peak_gib=peak_before/2**30, checkpoint_roundtrip_passed=True)
    assert result["selected_examples"] > 0 and result["target_loss_exercised"]
    write_json(out / "smoke_result.json", result); log(f"Smoke passed: {json.dumps(result)}")


def training_arm(arm, cfg, out, ref, frame, test, si, vi, frozen):
    directory = out / arm; directory.mkdir(exist_ok=True)
    if (directory / "done.json").exists():
        return json.loads((directory / "done.json").read_text())
    seed_all(cfg["seed"])
    dataset = PairedBatches(cfg, frame, si, len(test)); keys = EpochKeys(len(dataset))
    model = make_model(cfg, ref); ema = ema_model(model, cfg)
    opt, schedule = optimizer_and_scheduler(model, cfg, dataset.steps)
    statistics = AdaptiveLabels(cfg["adaptive"]); anchors = dict(np.load(out / "cache/anchors.npz"))
    checkpoint, spec_sha = directory / "checkpoint.pt", digest(out / "spec.json")
    history, completed = [], 0
    if checkpoint.exists():
        completed, history = load_training_checkpoint(checkpoint, model, ema, opt, schedule, statistics, spec_sha)
    else:
        save_checkpoint(checkpoint, checkpoint_state(model, ema, opt, schedule, statistics, 0, history, spec_sha))
    loader = DataLoader(dataset, batch_size=None, sampler=keys, collate_fn=identity, num_workers=cfg["execution"]["workers"],
                        worker_init_fn=init_worker, persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn",
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    y = frame.iloc[vi].label.to_numpy(); prior = np.bincount(frame.label, minlength=30) / len(frame)
    accumulation, epochs = cfg["training"]["gradient_accumulation"], cfg["training"]["epochs"]
    for epoch in range(completed+1, epochs+1):
        keys.epoch = epoch; model.train(); started = time.monotonic()
        totals = dict(source_loss=0., target_loss=0., selected=0, target_count=0, threshold=0., target_weight=0.)
        classes = np.zeros(30, dtype=np.int64); opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            loss, values = forward_loss(model, ema.module, statistics, batch, anchors, cfg, epoch, i, len(dataset),
                                        cfg["arms"][arm]["target_loss_enabled"], ref)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss: arm={arm} epoch={epoch} microbatch={i}")
            (loss/accumulation).backward()
            for k in totals:
                totals[k] += values[k]
            classes += values["selected_classes"]
            if (i+1) % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"], error_if_nonfinite=True)
                opt.step(); schedule.step(); ema.update(model); opt.zero_grad(set_to_none=True)
                step = (i+1)//accumulation
                write_json(out / "status.json", dict(state="training", at=now(), pid=os.getpid(), arm=arm, epoch=epoch,
                        optimizer_step_in_epoch=step, optimizer_steps_per_epoch=dataset.steps, total_epochs=epochs,
                        completed_updates=(epoch-1)*dataset.steps+step, source_loss=values["source_loss"], target_loss=values["target_loss"],
                        target_weight=values["target_weight"], acceptance_this_microbatch=values["selected"]/values["target_count"],
                        seconds_in_epoch=time.monotonic()-started,
                        estimated_seconds_remaining_this_arm=float(np.median([h["seconds"] for h in history[-3:]])*(epochs-epoch+1)) if len(history)>=3 else None))
        record = dict(epoch=epoch, seconds=time.monotonic()-started, learning_rate=schedule.get_last_lr()[0],
                      selected_fraction=totals["selected"]/totals["target_count"], selected_classes=int((classes>0).sum()),
                      pseudo_label_histogram=classes.tolist(),
                      **{k: totals[k]/len(dataset) for k in ("source_loss", "target_loss", "threshold", "target_weight")})
        if epoch % cfg["validation"]["log_every_epochs"] == 0 or epoch == epochs:
            state = rng_state()
            record["validation"], student, ensemble = evaluate(ema.module, cfg, ref, y, prior, out, frozen)
            save_npz(directory / f"epoch_{epoch:03d}_student.npz", **student)
            save_npz(directory / f"epoch_{epoch:03d}_ensemble.npz", **ensemble)
            restore_rng(state)
        history.append(record)
        save_checkpoint(checkpoint, checkpoint_state(model, ema, opt, schedule, statistics, epoch, history, spec_sha))
        write_json(directory / "history.json", history)
        log(f"arm={arm} epoch={epoch}/{epochs} source_loss={record['source_loss']:.5f} target_loss={record['target_loss']:.5f} "
            f"accepted={record['selected_fraction']:.3f} seconds={record['seconds']:.1f}" +
            (f" primary={record['validation']['primary']:.6f}" if "validation" in record else ""))
    save_checkpoint(directory / "model.pt", ema.module.state_dict())
    done = dict(arm=arm, at=now(), final_epoch=epochs, spec_sha256=spec_sha, metrics=history[-1]["validation"],
                optimizer_updates=epochs*dataset.steps, training_seconds=sum(h["seconds"] for h in history),
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(directory / "done.json", done)
    # Deleting the loader shuts down only the workers created by this arm.
    del loader, model, ema, opt, schedule, dataset
    gc.collect(); torch.cuda.empty_cache()
    return done


def finish(cfg, out, ref, frame, vi, results):
    from scripts.pilot_retrieval_graph import bootstrap
    initial = json.loads((out / "initial_baseline.json").read_text())
    a, b = results["A"]["metrics"], results["B"]["metrics"]
    c = cfg["validation"]
    gates = dict(primary_vs_initial=b["primary"]-initial["primary"] >= c["minimum_B_gain_vs_initial_baseline"],
                 primary_vs_control=b["primary"]-a["primary"] >= c["minimum_B_gain_vs_A"],
                 clean_vs_initial=b["clean"]-initial["clean"] >= c["minimum_clean_raw_ensemble_delta_vs_baseline_and_A"],
                 clean_vs_control=b["clean"]-a["clean"] >= c["minimum_clean_raw_ensemble_delta_vs_baseline_and_A"],
                 blur_vs_initial=b["mean_blur"]-initial["mean_blur"] >= c["minimum_mean_blur_raw_ensemble_delta_vs_baseline_and_A"],
                 blur_vs_control=b["mean_blur"]-a["mean_blur"] >= c["minimum_mean_blur_raw_ensemble_delta_vs_baseline_and_A"])
    prior = np.bincount(frame.label, minlength=30) / len(frame); y = frame.iloc[vi].label.to_numpy()
    final_name = f"epoch_{cfg['training']['epochs']:03d}_ensemble.npz"
    b_labels = ref.balance(np.load(out / "B" / final_name)["primary"], prior).argmax(1)
    intervals = {}
    for label, path in (("initial", out / "initial_ensemble.npz"), ("control", out / "A" / final_name)):
        base_labels = ref.balance(np.load(path)["primary"], prior).argmax(1)
        intervals[label] = bootstrap(y, base_labels, b_labels, frame.iloc[vi].reset_index(drop=True), cfg)
    result = dict(experiment_id=cfg["experiment_id"], at=now(), final_epoch_only=True, fold=cfg["fold"],
                  baseline=initial, arms=results, gates=gates, screening_passed=all(gates.values()),
                  primary_gain_vs_initial=b["primary"]-initial["primary"], primary_gain_vs_control=b["primary"]-a["primary"],
                  paired_group_bootstrap_95_percent=intervals, additional_folds_started=False, kaggle_upload_performed=False)
    write_json(out / "result.json", result)
    write_json(out / "status.json", dict(state="completed", at=now(), screening_passed=result["screening_passed"]))
    log(f"Completed A/B screening: passed={result['screening_passed']}, gain_vs_initial={result['primary_gain_vs_initial']:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/LOCAL-PROP-004-SSL.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "smoke", "run"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    with (out / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            ref = setup(cfg)
            if args.prepare:
                prepare(cfg, args.config, out, ref)
            elif args.smoke:
                smoke(cfg, args.config, out, ref)
            else:
                verify_spec(cfg, args.config, out)
                check = json.loads((out / "smoke_result.json").read_text())
                assert check["passed"] and check["spec_sha256"] == digest(out / "spec.json")
                frame, test, si, vi = metadata(cfg)
                frozen = dict(np.load(out / "frozen_r34d_probabilities.npz"))
                results = {arm: training_arm(arm, cfg, out, ref, frame, test, si, vi, frozen) for arm in cfg["execution"]["arms_sequential"]}
                finish(cfg, out, ref, frame, vi, results)
        except torch.cuda.OutOfMemoryError:
            write_json(out / "status.json", dict(state="waiting_after_cuda_oom", at=now(), pid=os.getpid(),
                                                resume="last_complete_epoch", batch_sizes_unchanged=True))
            traceback.print_exc()
            raise SystemExit(cfg["execution"]["oom_exit_code"])
        except Exception as error:
            write_json(out / "status.json", dict(state="failed", at=now(), pid=os.getpid(), error=str(error)))
            raise


if __name__ == "__main__":
    # Also permits the shared bootstrap helper to be imported when run as a file.
    sys.path.insert(0, str(ROOT))
    main()
