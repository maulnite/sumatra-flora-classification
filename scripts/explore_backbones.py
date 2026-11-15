"""Scratch EfficientNetV2 / ConvNeXt comparison on the frozen CE-A source protocol.

Both backbones see each identical batch in turn. Optimizer settings are explicit
per architecture; this is a practical recipe comparison, not a pure architecture
ablation. No test images, pretrained weights, or upload operations are used.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from timm.models import convnext, efficientnet
import yaml

try:
    from . import supervised_contrastive as source
except ImportError:
    import supervised_contrastive as source

common, replay = source.common, source.replay
ROOT = Path(__file__).resolve().parents[1]


def code_identity(config):
    paths = [config, Path(__file__), ROOT / "scripts/launch_backbone_exploration.py",
             Path(source.__file__), Path(common.__file__), Path(replay.__file__),
             Path(inspect.getfile(convnext)), Path(inspect.getfile(efficientnet))]
    return {str(p.resolve()): replay.digest(p) for p in paths}


def validate_protocol(cfg):
    reference_path = ROOT / cfg["reference_training_config"]
    reference = yaml.safe_load(reference_path.read_text())
    upstream = ROOT / cfg["reference_experiment_root"]
    if replay.digest(upstream / "spec.json") != cfg["reference_spec_sha256"]:
        raise ValueError("Reference training specification changed")
    for key in ("fold", "seed", "dataset_root", "team_root", "reference_root"):
        if cfg[key] != reference[key]:
            raise ValueError(f"Unmatched source protocol: {key}")
    for key in ("image_size", "epochs", "classes_per_batch", "photographs_per_class", "source_batch",
                "views_per_image", "minimum_learning_rate", "weight_decay", "label_smoothing",
                "gradient_clip_norm", "source_rotation_probability", "source_rotation_degrees",
                "batch_mixup_cutmix_probability", "ema_decay"):
        if cfg["training"][key] != reference["training"][key]:
            raise ValueError(f"Unmatched CE-A training setting: {key}")
    for key in ("expected_train_rows", "expected_validation_rows", "student_inference_size",
                "frozen_inference_size", "batch_size", "tta", "balance_iterations"):
        if cfg["validation"][key] != reference["validation"][key]:
            raise ValueError(f"Unmatched validation setting: {key}")
    if cfg["training"]["pretrained"] or cfg["execution"]["target_images_used"]:
        raise ValueError("This protocol permits scratch source-only training")
    source.verify(reference, reference_path, upstream)
    return upstream


def make_model(cfg, name):
    if cfg["training"]["pretrained"]:
        raise ValueError("Pretrained weights are forbidden")
    model = cfg["backbones"][name]
    return timm.create_model(model["architecture"], pretrained=False, num_classes=30,
                             drop_rate=model["drop_rate"], drop_path_rate=model["drop_path_rate"])


def optimizer_config(cfg, name):
    resolved = copy.deepcopy(cfg)
    for key in ("learning_rate", "warmup_epochs"):
        resolved["training"][key] = cfg["backbones"][name][key]
    return resolved


def verify(cfg, config, out):
    upstream = validate_protocol(cfg)
    spec = json.loads((out / "spec.json").read_text())
    if spec["code_sha256"] != code_identity(config):
        raise ValueError("Prepared runner, model implementation or config changed")
    if spec["versions"] != {p: importlib.metadata.version(p) for p in replay.VERSIONS}:
        raise ValueError("Environment changed")
    for path, expected in spec["input_sha256"].items():
        if replay.digest(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    return upstream


def prepare(cfg, config, out):
    if (out / "spec.json").exists():
        verify(cfg, config, out)
        return
    upstream = validate_protocol(cfg)
    frame, _, source_indices, validation_indices = common.metadata(cfg)
    inputs = {}
    paths = [ROOT / cfg["reference_training_config"], upstream / "spec.json",
             upstream / "frozen_r34d_probabilities.npz", upstream / "team_baseline.json",
             upstream / "team_ensemble.npz", upstream / "A/epoch_600_ensemble.npz",
             upstream / "A/history.json", upstream / "split_manifest.csv"]
    paths += [upstream / f"cache/{condition}_352.npy" for condition in common.CONDITIONS]
    models = {}
    for name in cfg["backbones"]:
        common.seed_all(cfg["seed"])
        model = make_model(cfg, name)
        directory = out / name
        directory.mkdir(exist_ok=True)
        path = directory / "initial_random_weights.pt"
        common.save_checkpoint(path, model.state_dict())
        paths.append(path)
        models[name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                            features=model.num_features, pretrained=False,
                            learning_rate=cfg["backbones"][name]["learning_rate"],
                            warmup_epochs=cfg["backbones"][name]["warmup_epochs"])
        del model
    for path in paths:
        inputs[str(path.resolve())] = replay.digest(path)
    steps = len(source_indices) // cfg["training"]["source_batch"]
    replay.write_json(out / "spec.json", dict(experiment_id=cfg["experiment_id"], at=replay.now(),
        code_sha256=code_identity(config), input_sha256=inputs,
        versions={p: importlib.metadata.version(p) for p in replay.VERSIONS},
        models=models, source_rows=len(source_indices), validation_rows=len(validation_indices),
        steps_per_epoch=steps, updates_per_backbone=steps*cfg["training"]["epochs"],
        source_protocol="LOCAL-PROP-006 CE A: identical deterministic draws and paired views",
        optimizer_settings_matched_across_architectures=False,
        target_images_used=False, pretrained_weights_used=False, full_oof_available=False))
    replay.log(f"Prepared {models}; source={len(source_indices)}, held-out={len(validation_indices)}")


def make_states(cfg, out, steps):
    states = {}
    for name in cfg["backbones"]:
        model = make_model(cfg, name)
        model.load_state_dict(torch.load(out / name / "initial_random_weights.pt", map_location="cpu", weights_only=True), strict=True)
        model = model.to("cuda").to(memory_format=torch.channels_last)
        opt, schedule = common.optimizer_and_scheduler(model, optimizer_config(cfg, name), steps)
        states[name] = dict(model=model, ema=common.ema_model(model, cfg), opt=opt, schedule=schedule, history=[])
    return states


def update(name, state, x, labels, cfg, epoch, step):
    seed = common.stable_seed(cfg["seed"], epoch, step, name, "backbone-dropout")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = state["model"]
    model.train()
    state["opt"].zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.float(), labels, label_smoothing=cfg["training"]["label_smoothing"])
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite CE: {name}")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"], error_if_nonfinite=True)
    state["opt"].step()
    state["schedule"].step()
    state["ema"].update(model)
    return dict(cross_entropy=float(loss.detach()), gradient_norm=float(norm))


def smoke(cfg, config, out, ref):
    verify(cfg, config, out)
    frame, _, si, _ = common.metadata(cfg)
    data = source.SourceBatches(cfg, frame, si)
    batch, repeat = data[(1, 0)], data[(1, 0)]
    assert all(np.array_equal(batch[k], repeat[k]) for k in batch)
    assert batch["images"].shape == (120, 3, 288, 288)
    assert np.bincount(batch["labels"], minlength=30).tolist() == [4]*30
    assert (frame.iloc[batch["indices"]].fold != cfg["fold"]).all()
    states = make_states(cfg, out, data.steps)
    x = common.normalize(batch["images"], ref)
    labels = torch.as_tensor(batch["labels"], device="cuda")
    elapsed, metrics = {}, {}
    for name, state in states.items():
        before = next(state["model"].parameters()).detach().clone()
        torch.cuda.synchronize()
        start = time.monotonic()
        metrics[name] = update(name, state, x, labels, cfg, 1, 0)
        torch.cuda.synchronize()
        elapsed[name] = time.monotonic()-start
        assert not torch.equal(before, next(state["model"].parameters()).detach())
        assert state["schedule"].last_epoch == 1
        assert {int(v["step"]) for v in state["opt"].state.values()} == {1}
        assert all(p.grad is None for p in state["ema"].module.parameters())
        replay.log(f"Full 120-view GPU update passed: {name}, {elapsed[name]:.2f}s, {metrics[name]}")
    path = out / "smoke_checkpoint.pt"
    common.save_checkpoint(path, source.paired_state(states, 0, out))
    assert source.restore_paired(states, path, out) == 0
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for name, state in states.items():
        for key in ("model", "ema"):
            assert all(torch.equal(p.cpu(), saved["arms"][name][key][k]) for k, p in state[key].state_dict().items())
        assert {int(v["step"]) for v in state["opt"].state.values()} == {1}
    replay.write_json(out / "smoke_result.json", dict(passed=True, at=replay.now(),
        spec_sha256=replay.digest(out / "spec.json"), full_source_views=120, source_only=True,
        optimizer_updates_per_backbone=1, checkpoint_roundtrip_passed=True, metrics=metrics,
        cold_update_seconds=elapsed, gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


def finish(cfg, out, upstream, ref, frame, vi, states):
    from scripts.pilot_retrieval_graph import bootstrap
    c = cfg["validation"]
    baseline = json.loads((upstream / "team_baseline.json").read_text())
    control = json.loads((upstream / "A/history.json").read_text())[-1]["validation"]
    y = frame.iloc[vi].label.to_numpy()
    prior = np.bincount(frame.label, minlength=30)/len(frame)
    comparisons = {}
    for name, state in states.items():
        final = state["history"][-1]["validation"]
        gates = dict(primary_vs_team=final["primary"]-baseline["primary"] >= c["minimum_gain_vs_team_pair"],
                     primary_vs_CE_A=final["primary"]-control["primary"] >= c["minimum_gain_vs_CE_A_pair"])
        for metric, key in (("clean", "minimum_clean_delta_vs_both"), ("mean_blur", "minimum_blur_delta_vs_both")):
            for label, previous in (("team", baseline), ("CE_A", control)):
                gates[f"{metric}_vs_{label}"] = final[metric]-previous[metric] >= c[key]
        prediction = ref.balance(np.load(out / name / "epoch_600_ensemble.npz")["primary"], prior).argmax(1)
        intervals = {}
        for label, path in (("team", upstream / "team_ensemble.npz"), ("CE_A", upstream / "A/epoch_600_ensemble.npz")):
            old = ref.balance(np.load(path)["primary"], prior).argmax(1)
            intervals[label] = bootstrap(y, old, prediction, frame.iloc[vi].reset_index(drop=True), cfg)
        comparisons[name] = dict(metrics=final, gates=gates, screening_passed=all(gates.values()),
                                  primary_gain_vs_team=final["primary"]-baseline["primary"],
                                  primary_gain_vs_CE_A=final["primary"]-control["primary"],
                                  conditional_group_bootstrap_95_percent=intervals)
    result = dict(experiment_id=cfg["experiment_id"], at=replay.now(), fold=cfg["fold"],
                  team_pair=baseline, CE_A_pair=control, backbones=comparisons,
                  screening_passed=any(v["screening_passed"] for v in comparisons.values()),
                  full_oof_available=False, additional_folds_started=False, kaggle_upload_performed=False)
    replay.write_json(out / "result.json", result)
    replay.write_json(out / "status.json", dict(state="completed", at=replay.now(), screening_passed=result["screening_passed"]))


def train(cfg, config, out, ref):
    upstream = verify(cfg, config, out)
    smoke_result = json.loads((out / "smoke_result.json").read_text())
    assert smoke_result["passed"] and smoke_result["spec_sha256"] == replay.digest(out / "spec.json")
    frame, _, si, vi = common.metadata(cfg)
    data = source.SourceBatches(cfg, frame, si)
    keys = common.EpochKeys(len(data))
    states = make_states(cfg, out, data.steps)
    checkpoint = out / "paired_checkpoint.pt"
    completed = 0
    if checkpoint.exists():
        completed = source.restore_paired(states, checkpoint, out)
        replay.log(f"Resumed both backbones after epoch {completed}")
    else:
        common.save_checkpoint(checkpoint, source.paired_state(states, 0, out))
    loader = DataLoader(data, batch_size=None, sampler=keys, collate_fn=common.identity,
        num_workers=cfg["execution"]["workers"], persistent_workers=True, prefetch_factor=2,
        multiprocessing_context="spawn", worker_init_fn=common.init_worker,
        generator=torch.Generator().manual_seed(cfg["seed"]))
    frozen = dict(np.load(upstream / "frozen_r34d_probabilities.npz"))
    y, prior = frame.iloc[vi].label.to_numpy(), np.bincount(frame.label, minlength=30)/len(frame)
    epochs = cfg["training"]["epochs"]
    for epoch in range(completed+1, epochs+1):
        keys.epoch = epoch
        started = time.monotonic()
        totals = {name: 0. for name in states}
        for step, batch in enumerate(loader):
            x = common.normalize(batch["images"], ref)
            labels = torch.as_tensor(batch["labels"], device="cuda")
            metrics = {name: update(name, state, x, labels, cfg, epoch, step) for name, state in states.items()}
            for name in states:
                totals[name] += metrics[name]["cross_entropy"]
            history = next(iter(states.values()))["history"]
            eta = float(np.median([h["paired_epoch_seconds"] for h in history[-3:]])*(epochs-epoch+1)) if len(history) >= 3 else None
            replay.write_json(out / "status.json", dict(state="training", at=replay.now(), pid=os.getpid(), epoch=epoch,
                total_epochs=epochs, optimizer_step_in_epoch=step+1, optimizer_steps_per_epoch=data.steps,
                completed_updates_per_backbone=(epoch-1)*data.steps+step+1, metrics=metrics,
                seconds_in_epoch=time.monotonic()-started, estimated_seconds_remaining=eta))
        seconds = time.monotonic()-started
        for name, state in states.items():
            record = dict(epoch=epoch, paired_epoch_seconds=seconds,
                          cross_entropy=totals[name]/data.steps, learning_rate=state["schedule"].get_last_lr()[0])
            if epoch % cfg["validation"]["log_every_epochs"] == 0 or epoch == epochs:
                rng = common.rng_state()
                record["validation"], student, ensemble = common.evaluate(state["ema"].module, cfg, ref, y, prior, upstream, frozen)
                common.save_npz(out / name / f"epoch_{epoch:03d}_student.npz", **student)
                common.save_npz(out / name / f"epoch_{epoch:03d}_ensemble.npz", **ensemble)
                common.save_checkpoint(out / name / "model.pt", state["ema"].module.state_dict())
                common.restore_rng(rng)
            state["history"].append(record)
        common.save_checkpoint(checkpoint, source.paired_state(states, epoch, out))
        for name, state in states.items():
            replay.write_json(out / name / "history.json", state["history"])
        replay.log(f"epoch={epoch}/{epochs} paired_seconds={seconds:.1f} " + " ".join(
            f"{name}:CE={state['history'][-1]['cross_entropy']:.4f}" +
            (f",primary={state['history'][-1]['validation']['primary']:.6f}" if "validation" in state["history"][-1] else "")
            for name, state in states.items()))
    # Verify frozen inputs again before declaring final results.
    verify(cfg, config, out)
    for name, state in states.items():
        common.save_checkpoint(out / name / "model.pt", state["ema"].module.state_dict())
        replay.write_json(out / name / "done.json", dict(at=replay.now(), epochs=epochs,
            optimizer_updates=epochs*data.steps, model_sha256=replay.digest(out / name / "model.pt"),
            metrics=state["history"][-1]["validation"], spec_sha256=replay.digest(out / "spec.json")))
    finish(cfg, out, upstream, ref, frame, vi, states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/LOCAL-PROP-007-BACKBONES.yaml")
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "smoke", "run"):
        modes.add_argument("--"+name, action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.prepare:
                prepare(cfg, args.config, out)
            else:
                ref = common.setup(cfg)
                if args.smoke:
                    smoke(cfg, args.config, out, ref)
                else:
                    train(cfg, args.config, out, ref)
        except torch.cuda.OutOfMemoryError:
            replay.write_json(out / "status.json", dict(state="waiting_after_cuda_oom", at=replay.now(),
                resume="last_complete_paired_epoch", full_batch_unchanged=True))
            traceback.print_exc()
            raise SystemExit(cfg["execution"]["oom_exit_code"])
        except Exception as error:
            replay.write_json(out / "status.json", dict(state="failed", at=replay.now(), error=str(error)))
            raise


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
