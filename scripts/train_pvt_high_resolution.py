"""Continue our own scratch B3 for 100 source-only epochs at 448px.

The held-out fold, corruption views and 352px evaluation stay fixed. This is a
paper-inspired resolution stage, not an exact reproduction. No test image enters
training, and no upload operation is performed. Epoch 100 EMA is preregistered.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import gc
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

try:
    from . import train_pvt_idmm as pvt
except ImportError:
    import train_pvt_idmm as pvt

common, replay, ROOT = pvt.common, pvt.replay, pvt.ROOT
PHASE = "source_supervised_448"


def resolve(meta):
    base = yaml.safe_load((ROOT / meta["base_config"]).read_text())
    cfg = copy.deepcopy(base)
    cfg["training"].update(meta["training_overrides"])
    cfg["validation"].update(meta["validation_overrides"])
    cfg.update(experiment_id=meta["experiment_id"], output_root=meta["output_root"], execution=meta["execution"])
    if cfg["training"]["pretrained"] or cfg["execution"]["target_images_used"]:
        raise ValueError("Only our own source-trained weights and source photographs are allowed")
    if meta["source_epoch_offset"] < 800 or (cfg["training"]["image_size"], cfg["training"]["epochs"]) != (448, 100):
        raise ValueError("Expected the frozen 100-epoch 448px continuation")
    if cfg["validation"]["student_inference_size"] != base["validation"]["student_inference_size"]:
        raise ValueError("Primary evaluation resolution must remain matched")
    return base, cfg


def identity(config, base_out):
    paths = list(json.loads((base_out / "spec.json").read_text())["code_sha256"])
    paths += [str(config.resolve()), str(Path(__file__).resolve()), str(ROOT / "scripts/launch_pvt_high_resolution.py")]
    return {str(Path(path).resolve()): replay.digest(path) for path in paths}


def upstream(meta, base):
    base_out = ROOT / base["output_root"]
    if replay.digest(base_out / "spec.json") != meta["source_spec_sha256"]:
        raise ValueError("Source experiment changed")
    pvt.verify(base, ROOT / meta["base_config"], base_out)
    checkpoint = ROOT / meta["source_checkpoint"]
    complete = json.loads((base_out / meta["architecture"] / "supervised/complete.json").read_text())
    if (replay.digest(checkpoint) != meta["source_checkpoint_sha256"] or
            complete["artifacts"].get(str(checkpoint.resolve())) != meta["source_checkpoint_sha256"]):
        raise ValueError("Expected the completed source-only B3 candidate")
    return base_out


def prepare(meta, config, out):
    base, cfg = resolve(meta); source = upstream(meta, base)
    if (out / "spec.json").exists():
        verify(meta, config, out); return
    frame, _, si, vi = common.metadata(cfg)
    directory = source / meta["architecture"] / "supervised/candidate"
    history = json.loads((source / meta["architecture"] / "supervised/history.json").read_text())
    baseline = history[-1]["metrics"]["candidate"]
    paths = [ROOT / meta["source_checkpoint"], source / "spec.json", directory / "epoch_800_student.npz",
             directory / "epoch_800_ensemble.npz", source / meta["architecture"] / "supervised/complete.json"]
    inputs = {str(path.resolve()): replay.digest(path) for path in paths}
    replay.write_json(out / "spec.json", dict(at=replay.now(), experiment_id=meta["experiment_id"],
        code_sha256=identity(config, source), input_sha256=inputs, source_spec_sha256=meta["source_spec_sha256"],
        source_rows=len(si), validation_rows=len(vi), fold=cfg["fold"], source_epoch_offset=meta["source_epoch_offset"],
        baseline=baseline, steps_per_epoch=10, planned_updates=1000,
        final_selection="epoch_100_EMA", target_images_used=False, inference_resolution=352,
        stop_at_utc=meta["execution"]["stop_at_utc"]))


def verify(meta, config, out):
    base, _ = resolve(meta); source = upstream(meta, base)
    spec = json.loads((out / "spec.json").read_text())
    if spec["code_sha256"] != identity(config, source):
        raise ValueError("Frozen code/config changed")
    for path, expected in spec["input_sha256"].items():
        if replay.digest(path) != expected:
            raise ValueError(f"Input changed: {path}")
    return source, spec


def make_states(meta, cfg):
    model = pvt.make_model(cfg, meta["architecture"])
    model.load_state_dict(torch.load(ROOT / meta["source_checkpoint"], map_location="cpu", weights_only=True), strict=True)
    model = model.cuda().to(memory_format=torch.channels_last)
    opt, schedule = common.optimizer_and_scheduler(model, cfg, 10)
    return dict(candidate=dict(model=model, opt=opt, schedule=schedule, ema=common.ema_model(model, cfg)))


def screening(final, baseline, gate):
    checks = dict(primary=final["primary"] >= baseline["primary"]+gate["minimum_primary_gain_vs_B3"],
                  clean=final["clean"] >= baseline["clean"]+gate["minimum_clean_delta_vs_B3"],
                  mean_blur=final["mean_blur"] >= baseline["mean_blur"]+gate["minimum_blur_delta_vs_B3"])
    return dict(passed=all(checks.values()), checks=checks,
                primary_delta=final["primary"]-baseline["primary"])


def smoke(meta, config, out, cfg, ref):
    source, _ = verify(meta, config, out)
    frame, _, si, _ = common.metadata(cfg)
    data = pvt.SourceBatches(cfg, frame, si, "supervised")
    batch = data[(meta["source_epoch_offset"]+1, 0)]
    assert batch["images"].shape == (256, 3, 448, 448)
    assert (frame.iloc[batch["indices"]].fold != cfg["fold"]).all()
    states = make_states(meta, cfg); state = states["candidate"]
    old = state["model"].head.weight.detach().cpu().clone()
    validation_root = ROOT / cfg["reference_experiment_root"]
    probability = common.predict(state["model"], np.load(validation_root / "cache/primary_352.npy", mmap_mode="r"), cfg, ref)
    stored = np.load(source / meta["architecture"] / "supervised/candidate/epoch_800_student.npz")["primary"]
    replay_check = replay.compare_probabilities(probability, stored, 1e-5, 1e-4)
    assert replay_check["within_tolerance"] and replay_check["predicted_labels_different"] == 0
    epoch = meta["source_epoch_offset"]+1
    groups = pvt.groups_for_update(batch, cfg, "supervised", epoch, 0)
    started = time.monotonic()
    metrics = pvt.update(state, groups, cfg, "supervised", epoch, 0, ref)
    assert not torch.equal(old, state["model"].head.weight.cpu())
    assert state["schedule"].last_epoch == 1
    path = out / "smoke_checkpoint.pt"
    common.save_checkpoint(path, pvt.pack_checkpoint(states, 0, out, meta["architecture"], PHASE, []))
    assert pvt.restore(states, path, out, meta["architecture"], PHASE)[0] == 0
    path.unlink()
    replay.write_json(out / "smoke_result.json", dict(at=replay.now(), passed=True,
        spec_sha256=replay.digest(out / "spec.json"), source_validation_replay=replay_check,
        real_source_photographs=256, image_size=448, metrics=metrics,
        update_and_checkpoint_seconds=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


def run(meta, config, out, cfg, ref):
    _, spec = verify(meta, config, out)
    smoke_result = json.loads((out / "smoke_result.json").read_text())
    assert smoke_result["passed"] and smoke_result["spec_sha256"] == replay.digest(out / "spec.json")
    frame, _, si, vi = common.metadata(cfg)
    data = pvt.SourceBatches(cfg, frame, si, "supervised"); assert data.steps == 10
    keys = common.EpochKeys(data.steps)
    loader = DataLoader(data, batch_size=None, sampler=keys, collate_fn=common.identity,
        num_workers=cfg["execution"]["workers"], worker_init_fn=common.init_worker,
        persistent_workers=True, prefetch_factor=1, multiprocessing_context="spawn")
    states = make_states(meta, cfg); checkpoint = out / "checkpoint.pt"
    completed, history = pvt.restore(states, checkpoint, out, meta["architecture"], PHASE) if checkpoint.exists() else (0, [])
    stop = datetime.fromisoformat(cfg["execution"]["stop_at_utc"])
    started = time.monotonic()
    replay.log(f"448px source continuation: resume={completed}/100, fixed 1000 updates, stop={stop}")
    for epoch in range(completed+1, 101):
        if datetime.now(timezone.utc) >= stop:
            common.save_checkpoint(checkpoint, pvt.pack_checkpoint(states, epoch-1, out, meta["architecture"], PHASE, history))
            replay.write_json(out / "deadline_stop.json", dict(at=replay.now(), completed_epochs=epoch-1,
                              reason="Reserved deadline buffer; final epoch100 gate not evaluated"))
            raise SystemExit(2)
        keys.epoch = meta["source_epoch_offset"]+epoch; losses = []
        for step, batch in enumerate(loader):
            groups = pvt.groups_for_update(batch, cfg, "supervised", keys.epoch, step)
            value = pvt.update(states["candidate"], groups, cfg, "supervised", keys.epoch, step, ref); losses.append(value["loss"])
            replay.write_json(out / "status.json", dict(at=replay.now(), epoch=epoch, epochs=100,
                step=step+1, steps=10, checkpoint_epoch=completed, phase_elapsed_seconds=time.monotonic()-started,
                loss=float(np.mean(losses)), image_size=448, source_only=True))
        if epoch % cfg["validation"]["log_every_epochs"] == 0:
            pvt.evaluate(states, cfg, ref, ROOT / cfg["reference_experiment_root"], out, epoch, frame, vi, history)
        if epoch % 10 == 0:
            common.save_checkpoint(checkpoint, pvt.pack_checkpoint(states, epoch, out, meta["architecture"], PHASE, history)); completed=epoch
            replay.log(f"448px epoch={epoch}/100 loss={float(np.mean(losses)):.5f}")
    model_path = out / "candidate/model.pt"
    common.save_checkpoint(model_path, states["candidate"]["ema"].module.state_dict())
    final = history[-1]["metrics"]["candidate"]
    replay.write_json(out / "result.json", dict(at=replay.now(), experiment_id=meta["experiment_id"],
        completed_epochs=100, updates=1000, metrics=final, baseline=spec["baseline"],
        gate=screening(final, spec["baseline"], meta["gate"]),
        model_sha256=replay.digest(model_path), spec_sha256=replay.digest(out / "spec.json"),
        screening_passed=screening(final, spec["baseline"], meta["gate"])["passed"], submission_uploaded=False))
    replay.write_json(out / "status.json", dict(at=replay.now(), state="completed", epochs=100, metrics=final))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/LOCAL-PROP-009-PVT448.yaml")
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "smoke", "run"): action.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args(); meta = yaml.safe_load(args.config.read_text()); _, cfg = resolve(meta)
    out = ROOT / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    with (out / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.prepare: prepare(meta, args.config, out)
            else:
                ref = common.setup(cfg)
                (smoke if args.smoke else run)(meta, args.config, out, cfg, ref)
        except torch.cuda.OutOfMemoryError:
            traceback.print_exc(); sys.exit(cfg["execution"]["oom_exit_code"])


if __name__ == "__main__": main()
