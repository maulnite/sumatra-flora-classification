"""Source-only PVTv2 IDMM and matched supervised controls, initialized from scratch.

Independent implementation of ECCV 2022 IDMM. Two global and four local crops
predict compact training-photo identities with CutMix and smoothed CE. Species
labels are used only in the subsequent paired supervised stage. No test image,
external weight or teacher weight initializes either PVT. See experiment notes
for explicit differences from the paper recipe.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import timm
from timm.models import pvt_v2
import yaml

try:
    from . import supervised_contrastive as reference
except ImportError:
    import supervised_contrastive as reference

common, replay = reference.common, reference.replay
ROOT = Path(__file__).resolve().parents[1]


def code_identity(config):
    paths = [config, Path(__file__), ROOT / "scripts/launch_pvt_idmm.py",
             Path(reference.__file__), Path(common.__file__), Path(replay.__file__),
             Path(inspect.getfile(pvt_v2))]
    return {str(p.resolve()): replay.digest(p) for p in paths}


def validate_reference(cfg):
    upstream = ROOT / cfg["reference_experiment_root"]
    path = ROOT / cfg["reference_training_config"]
    old = yaml.safe_load(path.read_text())
    if replay.digest(upstream / "spec.json") != cfg["reference_spec_sha256"]:
        raise ValueError("Reference specification changed")
    for key in ("fold", "seed", "dataset_root", "team_root", "reference_root"):
        if cfg[key] != old[key]:
            raise ValueError(f"Reference split differs: {key}")
    if cfg["training"]["pretrained"] or cfg["execution"]["target_images_used"]:
        raise ValueError("Only official source images and scratch weights are allowed")
    reference.verify(old, path, upstream)
    return upstream


def make_model(cfg, name, classes=30):
    if cfg["training"]["pretrained"]:
        raise ValueError("External initialization is forbidden")
    return timm.create_model(name, pretrained=False, num_classes=classes,
                             drop_rate=cfg["training"]["drop_rate"],
                             drop_path_rate=cfg["training"]["drop_path_rate"])


def transfer_backbone(model, state):
    """Preserve the original species head; discard the instance classifier."""
    missing, unexpected = model.load_state_dict({k: v for k, v in state.items() if not k.startswith("head.")}, strict=False)
    if set(missing) != {"head.weight", "head.bias"} or unexpected:
        raise ValueError(f"Invalid IDMM backbone: missing={missing}, unexpected={unexpected}")


def source_order(source, seed, epoch):
    return np.random.RandomState(common.stable_seed(seed, epoch, "uniform-source-order")).permutation(source)


class SourceBatches(Dataset):
    def __init__(self, cfg, frame, source, phase):
        self.cfg, self.frame, self.source, self.phase = cfg, frame, np.asarray(source), phase
        if phase not in ("idmm", "supervised") or (frame.iloc[self.source].fold == cfg["fold"]).any():
            raise ValueError("Invalid phase or held-out photograph in training")
        self.ids = {int(index): instance for instance, index in enumerate(self.source)}
        self.steps = math.ceil(len(source) / cfg["training"]["source_batch"])
        self.runtime = self.transforms = None

    def __len__(self):
        return self.steps

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(runtime=None, transforms=None)
        return state

    def initialize(self):
        self.runtime = replay.load_reference(ROOT / self.cfg["reference_root"] / "reference_sources")
        cv2.setNumThreads(1)
        if self.phase == "idmm":
            c = self.cfg["idmm"]
            self.transforms = [A.Compose([
                A.RandomResizedCrop(size=(c["image_size"],)*2, scale=tuple(scale), ratio=tuple(c["crop_ratio"])),
                A.HorizontalFlip(p=.5),
                A.ColorJitter(*c["color_jitter"], p=c["color_probability"]),
                A.ToGray(p=c["grayscale_probability"]),
                A.GaussianBlur(blur_limit=(3, 13), sigma_limit=tuple(c["gaussian_sigma"]), p=c["gaussian_blur_probability"]),
            ]) for scale in c["crop_scales"]]
        else:
            c = self.cfg["training"]
            self.transforms = [A.Compose([A.RandomResizedCrop(size=(c["image_size"],)*2,
                               scale=tuple(c["crop_scale"]), ratio=tuple(c["crop_ratio"])), A.HorizontalFlip(p=.5)])]

    def __getitem__(self, key):
        epoch, step = key
        if self.runtime is None:
            self.initialize()
        c, batch = self.cfg, self.cfg["training"]["source_batch"]
        indices = source_order(self.source, c["seed"], epoch)[step*batch:(step+1)*batch]
        views = [[] for _ in range(6 if self.phase == "idmm" else 1)]
        for index in indices:
            row = self.frame.iloc[index]
            seed = common.stable_seed(c["seed"], self.phase, epoch, int(index))
            image = self.runtime.dq.read_rgb(ROOT / c["dataset_root"] / "train" / row.id)
            if self.phase == "idmm":
                for view, result in enumerate(views):
                    transform = self.transforms[0 if view < 2 else 1]
                    transform.set_random_seed(common.stable_seed(seed, view))
                    result.append(np.ascontiguousarray(transform(image=image)["image"].transpose(2, 0, 1)))
            else:
                rng = np.random.RandomState(seed)
                if rng.rand() < c["training"]["source_rotation_probability"]:
                    angle = c["training"]["source_rotation_degrees"]
                    image = self.runtime.rot_img(image, rng.uniform(-angle, angle), rng.rand() < .5)
                transform = (self.runtime.dq.corrupt_v4 if rng.rand() < c["training"]["strong_view_probability"]
                             else self.runtime.dq.light_color)
                common.random.seed(seed); np.random.seed(seed); transform.set_random_seed(seed)
                image = transform(image=image)["image"]
                self.transforms[0].set_random_seed(common.stable_seed(seed, "crop"))
                image = self.transforms[0](image=image)["image"]
                views[0].append(np.ascontiguousarray(image.transpose(2, 0, 1)))
        labels = (np.array([self.ids[int(i)] for i in indices], dtype=np.int64) if self.phase == "idmm"
                  else self.frame.iloc[indices].label.to_numpy(dtype=np.int64))
        # View-major concatenation matches the two independent crop-group losses.
        return dict(images=np.concatenate([np.stack(view) for view in views]), labels=labels, indices=indices)


def mix_batch(images, labels, seed, kind, alpha):
    """Mix across the complete effective group BEFORE splitting into microbatches."""
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(labels))
    lam = float(rng.beta(alpha, alpha))
    if kind == "cutmix":
        height, width = images.shape[-2:]
        cut = math.sqrt(1 - lam)
        cy, cx = rng.randint(height), rng.randint(width)
        hh, ww = int(height*cut), int(width*cut)
        y1, y2 = max(0, cy-hh//2), min(height, cy+hh//2)
        x1, x2 = max(0, cx-ww//2), min(width, cx+ww//2)
        mixed = images.copy()
        mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        lam = 1 - ((y2-y1)*(x2-x1) / (height*width))
    elif kind == "mixup":
        mixed = lam*images.astype(np.float32) + (1-lam)*images[permutation].astype(np.float32)
    else:
        raise ValueError(kind)
    return mixed, labels, labels[permutation], lam


def mixed_loss(logits, first, second, lam, smoothing):
    return lam*F.cross_entropy(logits.float(), first, label_smoothing=smoothing) + (1-lam)*F.cross_entropy(logits.float(), second, label_smoothing=smoothing)


def groups_for_update(batch, cfg, phase, epoch, step):
    n = len(batch["labels"])
    groups = [(batch["images"][:2*n], np.tile(batch["labels"], 2)),
              (batch["images"][2*n:], np.tile(batch["labels"], 4))] if phase == "idmm" else [(batch["images"], batch["labels"])]
    result = []
    for group, (images, labels) in enumerate(groups):
        seed = common.stable_seed(cfg["seed"], phase, epoch, step, group, "mixing")
        kind = "cutmix" if phase == "idmm" or np.random.RandomState(seed).rand() < cfg["training"]["cutmix_probability"] else "mixup"
        result.append(mix_batch(images, labels, seed, kind, cfg["training"]["mix_alpha"]))
    return result


def update(state, groups, cfg, phase, epoch, step, ref):
    common.seed_all(common.stable_seed(cfg["seed"], phase, epoch, step, "paired-dropout"))
    model = state["model"]; model.train(); state["opt"].zero_grad(set_to_none=True)
    micro = cfg["idmm" if phase == "idmm" else "training"]["microbatch"]
    losses = []
    for images, first, second, lam in groups:
        total = 0.
        for start in range(0, len(images), micro):
            end = min(len(images), start+micro)
            x = common.normalize(images[start:end], ref)
            a = torch.as_tensor(first[start:end], device="cuda")
            b = torch.as_tensor(second[start:end], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = mixed_loss(logits, a, b, lam, cfg["training"]["label_smoothing"])*(end-start)/len(images)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward(); total += float(loss.detach())
        losses.append(total)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"], error_if_nonfinite=True)
    state["opt"].step(); state["schedule"].step()
    if state["ema"] is not None:
        state["ema"].update(model)
    return dict(loss=sum(losses), group_losses=losses, gradient_norm=float(norm))


def make_states(cfg, out, name, phase, steps, instances, idmm_state=None):
    initial = torch.load(out / name / "initial_random_weights.pt", map_location="cpu", weights_only=True)
    states = {}
    for arm in (["idmm"] if phase == "idmm" else ["control", "candidate"]):
        common.seed_all(common.stable_seed(cfg["seed"], name, "initialization"))
        model = make_model(cfg, name); model.load_state_dict(initial, strict=True)
        if phase == "idmm":
            model.reset_classifier(instances)
            torch.nn.init.trunc_normal_(model.head.weight, std=.02); torch.nn.init.zeros_(model.head.bias)
        elif arm == "candidate":
            transfer_backbone(model, idmm_state if idmm_state is not None else
                              torch.load(out / name / "idmm/model.pt", map_location="cpu", weights_only=True))
        model = model.cuda().to(memory_format=torch.channels_last)
        resolved = copy.deepcopy(cfg)
        resolved["training"].update(learning_rate=cfg["backbones"][name]["learning_rate"],
                                    epochs=cfg["idmm" if phase == "idmm" else "training"]["epochs"])
        opt, schedule = common.optimizer_and_scheduler(model, resolved, steps)
        states[arm] = dict(model=model, opt=opt, schedule=schedule,
                           ema=common.ema_model(model, cfg) if phase == "supervised" else None)
    return states


def pack_checkpoint(states, epoch, out, name, phase, history):
    return dict(spec_sha256=replay.digest(out / "spec.json"), architecture=name, phase=phase,
                epoch=epoch, history=history, rng=common.rng_state(), arms={arm: dict(
                model=s["model"].state_dict(), optimizer=s["opt"].state_dict(), scheduler=s["schedule"].state_dict(),
                ema=s["ema"].state_dict() if s["ema"] is not None else None) for arm, s in states.items()})


def restore(states, path, out, name, phase):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint["spec_sha256"], checkpoint["architecture"], checkpoint["phase"]) != (replay.digest(out / "spec.json"), name, phase):
        raise ValueError("Checkpoint belongs to a different frozen experiment/phase")
    if set(states) != set(checkpoint["arms"]):
        raise ValueError("Checkpoint arm mismatch")
    for arm, state in states.items():
        saved = checkpoint["arms"][arm]
        state["model"].load_state_dict(saved["model"], strict=True)
        state["opt"].load_state_dict(saved["optimizer"]); state["schedule"].load_state_dict(saved["scheduler"])
        if state["ema"] is not None:
            state["ema"].load_state_dict(saved["ema"])
    common.restore_rng(checkpoint["rng"])
    return checkpoint["epoch"], checkpoint["history"]


def prepare(cfg, config, out):
    if (out / "spec.json").exists():
        verify(cfg, config, out); return
    upstream = validate_reference(cfg)
    frame, _, si, vi = common.metadata(cfg)
    inputs, models = {}, {}
    paths = [ROOT / cfg["reference_training_config"], upstream / "spec.json", upstream / "frozen_r34d_probabilities.npz",
             upstream / "A/history.json", upstream / "A/epoch_600_ensemble.npz", upstream / "team_baseline.json"]
    paths += [upstream / f"cache/{condition}_352.npy" for condition in common.CONDITIONS]
    for name in cfg["backbones"]:
        directory = out / name; directory.mkdir(exist_ok=True)
        common.seed_all(common.stable_seed(cfg["seed"], name, "initialization"))
        model = make_model(cfg, name)
        path = directory / "initial_random_weights.pt"
        common.save_checkpoint(path, model.state_dict()); paths.append(path)
        models[name] = dict(parameters_30_classes=sum(p.numel() for p in model.parameters()), pretrained=False)
        del model
    manifest = frame.iloc[si].copy(); manifest["instance_id"] = np.arange(len(si))
    manifest.to_csv(out / "source_instances.csv", index=False); paths.append(out / "source_instances.csv")
    for path in paths:
        inputs[str(path.resolve())] = replay.digest(path)
    steps = math.ceil(len(si)/cfg["training"]["source_batch"])
    replay.write_json(out / "spec.json", dict(experiment_id=cfg["experiment_id"], at=replay.now(),
        code_sha256=code_identity(config), input_sha256=inputs,
        versions={p: importlib.metadata.version(p) for p in replay.VERSIONS}, models=models,
        source_rows=len(si), validation_rows=len(vi), steps_per_epoch=steps,
        idmm_updates_per_backbone=steps*cfg["idmm"]["epochs"], supervised_updates_per_arm=steps*cfg["training"]["epochs"],
        source_protocol="fold-2 isolated uniform photographs; compact IDs exclude validation; no test inputs",
        paired_species_heads_identical=True, candidate_total_compute_matched_to_control=False,
        optimizer_settings_matched_across_sizes=False, target_images_used=False, pretrained_weights_used=False,
        validation_prior="legacy all-train frequencies used for evaluation only; raw and standalone also reported"))


def verify(cfg, config, out):
    upstream = validate_reference(cfg)
    spec = json.loads((out / "spec.json").read_text())
    if spec["code_sha256"] != code_identity(config):
        raise ValueError("Prepared code/config changed")
    if spec["versions"] != {p: importlib.metadata.version(p) for p in replay.VERSIONS}:
        raise ValueError("Environment changed")
    for path, expected in spec["input_sha256"].items():
        if replay.digest(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    return upstream


def smoke(cfg, config, out, ref):
    upstream = verify(cfg, config, out)
    frame, _, si, _ = common.metadata(cfg)
    results = {}; started = time.monotonic()
    for name in cfg["backbones"]:
        result = {}
        for phase in ("idmm", "supervised"):
            data = SourceBatches(cfg, frame, si, phase)
            batch = data[(1, 0)]; repeat = data[(1, 0)]
            assert all(np.array_equal(batch[k], repeat[k]) for k in batch)
            assert len(batch["labels"]) == cfg["training"]["source_batch"]
            states = make_states(cfg, out, name, phase, data.steps, len(si),
                                 idmm_state=ssl_state if phase == "supervised" else None)
            if phase == "supervised":
                assert torch.equal(states["control"]["model"].head.weight, states["candidate"]["model"].head.weight)
            groups = groups_for_update(batch, cfg, phase, 1, 0)
            torch.cuda.synchronize(); before = time.monotonic()
            for arm, state in states.items():
                old = state["model"].head.weight.detach().clone()
                result[f"{phase}_{arm}"] = update(state, groups, cfg, phase, 1, 0, ref)
                assert not torch.equal(old, state["model"].head.weight)
                if phase == "supervised":
                    probability = common.predict(state["ema"].module, np.load(upstream / "cache/primary_352.npy", mmap_mode="r")[:2], cfg, ref)
                    assert probability.shape == (2, 30) and np.isfinite(probability).all()
                    np.testing.assert_allclose(probability.sum(1), 1, atol=1e-6)
            torch.cuda.synchronize(); result[f"{phase}_seconds"] = time.monotonic()-before
            if phase == "idmm":
                ssl_state = {k: v.detach().cpu().clone() for k, v in states["idmm"]["model"].state_dict().items()}
            path = out / "smoke_checkpoint.pt"
            common.save_checkpoint(path, pack_checkpoint(states, 1, out, name, phase, []))
            expected = {arm: s["model"].head.weight.detach().cpu().clone() for arm, s in states.items()}
            for state in states.values():
                with torch.no_grad():
                    state["model"].head.weight.zero_()
            epoch, _ = restore(states, path, out, name, phase)
            assert epoch == 1 and all(torch.equal(expected[arm], s["model"].head.weight.cpu()) for arm, s in states.items())
            path.unlink(); del states, batch, repeat, groups; gc.collect(); torch.cuda.empty_cache()
        results[name] = result
    replay.write_json(out / "smoke_result.json", dict(passed=True, at=replay.now(), spec_sha256=replay.digest(out / "spec.json"),
        real_source_batch=cfg["training"]["source_batch"], idmm_views=6, results=results,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, elapsed_seconds=time.monotonic()-started))
    replay.log(f"Real-source GPU smoke passed: {results}")


def screening(metrics, cfg, upstream):
    """Development-fold screening only; does not establish leaderboard gain."""
    candidate, control = metrics["candidate"], metrics["control"]
    history = json.loads((upstream / "A/history.json").read_text())
    baseline = history[-1]["validation"]
    checks = dict(primary_vs_CE_A=candidate["primary"] >= baseline["primary"]+cfg["validation"]["minimum_gain_vs_CE_A_pair"],
                  primary_vs_control=candidate["primary"] >= control["primary"]+cfg["validation"]["minimum_gain_vs_control"],
                  clean_vs_CE_A=candidate["clean"] >= baseline["clean"]+cfg["validation"]["minimum_clean_delta_vs_CE_A"],
                  blur_vs_CE_A=candidate["mean_blur"] >= baseline["mean_blur"]+cfg["validation"]["minimum_blur_delta_vs_CE_A"])
    return dict(passed=all(checks.values()), checks=checks, CE_A=baseline,
                primary_delta_vs_control=candidate["primary"]-control["primary"],
                primary_delta_vs_CE_A=candidate["primary"]-baseline["primary"])


def evaluate(states, cfg, ref, upstream, directory, epoch, frame, vi, history):
    prior = np.bincount(frame.label, minlength=30)/len(frame)
    frozen = dict(np.load(upstream / "frozen_r34d_probabilities.npz"))
    metrics = {}
    for arm, state in states.items():
        model = state["ema"].module
        value, student, ensemble = common.evaluate(model, cfg, ref, frame.iloc[vi].label.to_numpy(), prior, upstream, frozen)
        metrics[arm] = value
        dest = directory / arm; dest.mkdir(exist_ok=True)
        common.save_npz(dest / f"epoch_{epoch}_student.npz", **student)
        common.save_npz(dest / f"epoch_{epoch}_ensemble.npz", **ensemble)
    record = dict(epoch=epoch, at=replay.now(), metrics=metrics)
    history.append(record); replay.write_json(directory / "history.json", history)
    replay.log(f"{directory.parent.name} supervised epoch={epoch}, metrics={metrics}")


def run_phase(cfg, config, out, ref, name, phase, upstream, frame, si, vi):
    directory = out / name / phase; directory.mkdir(exist_ok=True)
    if (directory / "complete.json").exists():
        saved = json.loads((directory / "complete.json").read_text())
        assert saved["spec_sha256"] == replay.digest(out / "spec.json")
        for path, digest in saved["artifacts"].items():
            assert replay.digest(path) == digest
        return saved
    data = SourceBatches(cfg, frame, si, phase)
    keys = common.EpochKeys(data.steps)
    loader = DataLoader(data, batch_size=None, sampler=keys, collate_fn=common.identity,
                        num_workers=cfg["execution"]["workers"], worker_init_fn=common.init_worker,
                        persistent_workers=True, prefetch_factor=cfg["execution"]["prefetch_factor"],
                        multiprocessing_context="spawn")
    states = make_states(cfg, out, name, phase, data.steps, len(si))
    checkpoint = directory / "checkpoint.pt"
    completed, history = restore(states, checkpoint, out, name, phase) if checkpoint.exists() else (0, [])
    epochs = cfg["idmm" if phase == "idmm" else "training"]["epochs"]
    started = time.monotonic()
    replay.log(f"Start {name}/{phase}: resume={completed}, epochs={epochs}, updates/epoch={data.steps}")
    for epoch in range(completed+1, epochs+1):
        keys.epoch = epoch; losses = {arm: [] for arm in states}
        for step, batch in enumerate(loader):
            groups = groups_for_update(batch, cfg, phase, epoch, step)
            for arm, state in states.items():
                losses[arm].append(update(state, groups, cfg, phase, epoch, step, ref)["loss"])
            replay.write_json(out / "status.json", dict(at=replay.now(), architecture=name, phase=phase,
                epoch=epoch, epochs=epochs, step=step+1, steps=data.steps,
                checkpoint_epoch=completed, phase_elapsed_seconds=time.monotonic()-started,
                current_loss={arm: float(np.mean(value)) for arm, value in losses.items()}))
        if phase == "supervised" and (epoch % cfg["validation"]["log_every_epochs"] == 0 or epoch == epochs):
            evaluate(states, cfg, ref, upstream, directory, epoch, frame, vi, history)
        if epoch % cfg["execution"]["checkpoint_every_epochs"] == 0 or epoch == epochs:
            common.save_checkpoint(checkpoint, pack_checkpoint(states, epoch, out, name, phase, history))
            completed = epoch
            replay.log(f"{name}/{phase} epoch={epoch}/{epochs} loss={ {a: round(float(np.mean(v)),5) for a,v in losses.items()} }")
    artifacts = {}
    for arm, state in states.items():
        dest = directory if phase == "idmm" else directory / arm
        dest.mkdir(exist_ok=True)
        path = dest / "model.pt"
        common.save_checkpoint(path, state["model"].state_dict() if phase == "idmm" else state["ema"].module.state_dict())
        artifacts[str(path.resolve())] = replay.digest(path)
    result = dict(spec_sha256=replay.digest(out / "spec.json"), at=replay.now(), architecture=name, phase=phase,
        completed_epochs=epochs, optimizer_updates_per_arm=epochs*data.steps, artifacts=artifacts,
        final_metrics=history[-1]["metrics"] if history else None)
    if phase == "supervised":
        result["screening"] = screening(result["final_metrics"], cfg, upstream)
    replay.write_json(directory / "complete.json", result)
    del states, loader; gc.collect(); torch.cuda.empty_cache()
    return result


def run(cfg, config, out, ref):
    upstream = verify(cfg, config, out)
    smoke_result = json.loads((out / "smoke_result.json").read_text())
    assert smoke_result["passed"] and smoke_result["spec_sha256"] == replay.digest(out / "spec.json")
    frame, _, si, vi = common.metadata(cfg)
    results = []
    for name in cfg["backbones"]:
        for phase in ("idmm", "supervised"):
            results.append(run_phase(cfg, config, out, ref, name, phase, upstream, frame, si, vi))
    replay.write_json(out / "result.json", dict(at=replay.now(), results=results,
        screening_passed=any(x.get("screening", {}).get("passed", False) for x in results),
        submission_created=False, additional_folds_started=False, note="Final fold-2 metrics require paired gate review; no automatic upload"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/LOCAL-PROP-008-PVT-IDMM.yaml")
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "smoke", "run"):
        action.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["output_root"]; out.mkdir(exist_ok=True, parents=True)
    with (out / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.prepare:
                torch.set_num_threads(cfg["execution"]["cpu_threads"])
                prepare(cfg, args.config, out)
            else:
                ref = common.setup(cfg)
                (smoke if args.smoke else run)(cfg, args.config, out, ref)
        except torch.cuda.OutOfMemoryError:
            traceback.print_exc(); replay.log("CUDA OOM; supervisor will retry unchanged from the last complete checkpoint")
            sys.exit(cfg["execution"]["oom_exit_code"])


if __name__ == "__main__":
    main()
