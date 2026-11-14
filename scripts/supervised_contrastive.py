"""Paired scratch CE versus CE+SupCon on official source labels only.

Both arms process each identical batch sequentially, sharing the data loader and
using the same dropout seed. A single atomic paired checkpoint keeps their epoch
counts aligned after OOM recovery. True class positives span distinct scenes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gc
import importlib.metadata
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

try:
    from . import adaptive_self_training as common
except ImportError:
    import adaptive_self_training as common

ROOT=Path(__file__).resolve().parents[1]
replay=common.reference_utils


def supcon_loss(embedding, labels, temperature):
    """Mean log-probability of every non-self same-class positive, full batch."""
    if embedding.ndim!=2 or len(embedding)!=len(labels) or temperature<=0:
        raise ValueError("Invalid contrastive comparison set")
    z=F.normalize(embedding.float(),dim=1)
    diagonal=torch.eye(len(z),device=z.device,dtype=torch.bool)
    positive=(labels[:,None]==labels[None,:]) & ~diagonal
    counts=positive.sum(1)
    if (counts==0).any():
        raise ValueError("Every contrastive anchor must have a non-self positive")
    logits=z@z.T/temperature
    logits=logits-logits.max(1,keepdim=True).values.detach()
    denominator=torch.logsumexp(logits.masked_fill(diagonal,-torch.inf),dim=1,keepdim=True)
    log_probability=logits-denominator
    return -((log_probability*positive).sum(1)/counts).mean()


def contrastive_weight(cfg,epoch,step,steps,enabled):
    progress=min(1.,(epoch-1+step/steps)/cfg["warmup_epochs"])
    return cfg["maximum_weight"]*.5*(1-math.cos(math.pi*progress)) if enabled else 0.


def batch_indices(frame,source,cfg,epoch,step):
    rng=np.random.RandomState(common.stable_seed(cfg["seed"],epoch,step,"class-balanced-source"))
    selected=[]
    for label in range(cfg["training"]["classes_per_batch"]):
        candidates=source[frame.iloc[source].label.to_numpy()==label]
        first=int(rng.choice(candidates))
        others=candidates[frame.iloc[candidates].group_id.to_numpy()!=frame.iloc[first].group_id]
        if not len(others):
            raise ValueError(f"Class {label} has fewer than two scene groups")
        selected.extend([first,int(rng.choice(others))])
    return rng.permutation(selected)


class SourceBatches(Dataset):
    def __init__(self,cfg,frame,source):
        self.cfg,self.frame,self.source=cfg,frame,np.asarray(source)
        if (frame.iloc[self.source].fold==cfg["fold"]).any():
            raise ValueError("Validation images in source sampler")
        self.steps=len(source)//cfg["training"]["source_batch"]
        self.ref,self.crop=None,None
        assert cfg["training"]["classes_per_batch"]==30 and cfg["training"]["photographs_per_class"]==2
        assert cfg["training"]["source_batch"]==60
        batch_indices(frame,self.source,cfg,1,0)

    def __len__(self):return self.steps

    def __getstate__(self):
        state=self.__dict__.copy();state.update(ref=None,crop=None);return state

    def __getitem__(self,key):
        epoch,step=key
        if self.ref is None:
            self.ref=replay.load_reference(ROOT/self.cfg["reference_root"]/"reference_sources")
            size=self.cfg["training"]["image_size"]
            self.crop=A.Compose([A.RandomResizedCrop(size=(size,size),scale=(.25,1.),ratio=(.5,2.)),A.HorizontalFlip()])
            cv2.setNumThreads(1)
        c=self.cfg;indices=batch_indices(self.frame,self.source,c,epoch,step)
        weak,strong=[],[]
        for position,index in enumerate(indices):
            row=self.frame.iloc[index];seed=common.stable_seed(c["seed"],epoch,step,position,int(index))
            rng=np.random.RandomState(seed)
            image=self.ref.dq.read_rgb(ROOT/c["dataset_root"]/"train"/row.id)
            if rng.rand()<c["training"]["source_rotation_probability"]:
                angle=c["training"]["source_rotation_degrees"]
                image=self.ref.rot_img(image,rng.uniform(-angle,angle),rng.rand()<.5)
            for name,transform,output in (("weak",self.ref.dq.light_color,weak),("strong",self.ref.dq.corrupt_v4,strong)):
                local=common.stable_seed(seed,name)
                common.random.seed(local);np.random.seed(local);transform.set_random_seed(local)
                view=transform(image=image.copy())["image"]
                self.crop.set_random_seed(common.stable_seed(seed,"paired-crop"))
                view=self.crop(image=view)["image"]
                output.append(np.ascontiguousarray(view.transpose(2,0,1)))
        return dict(images=np.concatenate([np.stack(weak),np.stack(strong)]),
                    labels=np.tile(self.frame.iloc[indices].label.to_numpy(dtype=np.int64),2),indices=indices)


class SupConNet(nn.Module):
    def __init__(self,ref,cfg):
        super().__init__()
        self.encoder=ref.make_model(dict(arch="resnet18d",attn="none",pool="avg"),30)
        c=cfg["contrastive"];n=self.encoder.num_features
        self.projection=nn.Sequential(nn.Linear(n,c["projection_hidden"]),nn.ReLU(),nn.Linear(c["projection_hidden"],c["projection_dimension"]))

    def forward(self,x):
        feature=self.encoder.forward_head(self.encoder.forward_features(x),pre_logits=True)
        return self.encoder.get_classifier()(feature),self.projection(feature)


def identity(config):
    paths=[config,Path(__file__),ROOT/"scripts/launch_supervised_contrastive.py",Path(common.__file__),Path(replay.__file__)]
    return {str(p.resolve()):replay.digest(p) for p in paths}


def verify(cfg,config,out):
    spec=json.loads((out/"spec.json").read_text())
    if spec["code_sha256"]!=identity(config):raise ValueError("Prepared code/config changed")
    versions={p:importlib.metadata.version(p) for p in replay.VERSIONS}
    if versions!=spec["versions"]:raise ValueError("Environment changed")
    for path,expected in spec["input_sha256"].items():
        if replay.digest(path)!=expected:raise ValueError(f"Pinned input changed: {path}")
    return spec


def prepare(cfg,config,out,ref):
    if (out/"spec.json").exists():verify(cfg,config,out);return
    frame,_,source,vi=common.metadata(cfg)
    original=json.loads((ROOT/cfg["reference_summary"]).read_text());expected=original["provenance"]["input_sha256"]
    versions={p:importlib.metadata.version(p) for p in replay.VERSIONS};assert versions==original["versions"]
    inputs={}
    def pin(path,expected_hash=None):
        value=replay.digest(path)
        if expected_hash is not None and value!=expected_hash:raise ValueError(f"Reference changed: {path}")
        inputs[str(path.resolve())]=value
    team=Path(cfg["team_root"]);data=ROOT/cfg["dataset_root"]
    for name in ("train.py","dq.py","evalres.py"):
        pin(ROOT/cfg["reference_root"]/"reference_sources"/name,expected[name])
    for run in (cfg["student_run"],cfg["frozen_run"]):
        for name in ("model.pt","cfg.json"):
            rel=f"runs/{run}_f{cfg['fold']}/{name}";pin(team/rel,expected[rel])
    for rel in ("cache/scene_folds.csv","cache/scene_folds.npy","cache/corr4r_352.npy","cache/corr4r_320.npy"):
        pin(team/rel,expected[rel])
    pin(data/"train.csv");pin(data/"class_mapping.csv");pin(data/"sample_submission.csv")
    replay.log("Pinning official source files; building fold-2 evaluation caches")
    paths=[data/"train"/x for x in frame.id]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path,value in zip(paths,pool.map(replay.digest,paths)):inputs[str(path.resolve())]=value
    cache=out/"cache";cache.mkdir(exist_ok=True)
    for size in (352,320):
        primary=np.array(np.load(team/f"cache/corr4r_{size}.npy",mmap_mode="r")[vi])
        with ThreadPoolExecutor(max_workers=4) as pool:
            clean=np.stack(list(pool.map(lambda name:ref.dq.squash(ref.dq.read_rgb(data/"train"/name),size),frame.iloc[vi].id)))
        common.save_array(cache/f"primary_{size}.npy",primary);common.save_array(cache/f"clean_{size}.npy",clean)
        for name,sigma in (("blur06",.6),("blur10",1.)):
            common.save_array(cache/f"{name}_{size}.npy",np.stack([cv2.GaussianBlur(x,(7,7),sigmaX=sigma,borderType=cv2.BORDER_REFLECT_101) for x in primary]))
    manifest=frame.copy();manifest["role"]=np.where(frame.fold==cfg["fold"],"validation_only","source_training")
    manifest.to_csv(out/"split_manifest.csv",index=False)
    model=common.make_model(cfg,ref,frozen=True)
    frozen={k:common.predict(model,np.load(cache/f"{k}_320.npy",mmap_mode="r"),cfg,ref) for k in common.CONDITIONS}
    common.save_npz(out/"frozen_r34d_probabilities.npz",**frozen)
    del model;gc.collect();torch.cuda.empty_cache()
    model=common.make_model(cfg,ref)
    prior=np.bincount(frame.label,minlength=30)/len(frame);y=frame.iloc[vi].label.to_numpy()
    baseline,student,ensemble=common.evaluate(model,cfg,ref,y,prior,out,frozen)
    common.save_npz(out/"team_student.npz",**student);common.save_npz(out/"team_ensemble.npz",**ensemble)
    replay.write_json(out/"team_baseline.json",baseline)
    del model;gc.collect();torch.cuda.empty_cache()
    common.seed_all(cfg["seed"]);random_model=SupConNet(ref,cfg)
    common.save_checkpoint(out/"initial_random_weights.pt",random_model.state_dict());del random_model
    for path in sorted(cache.glob("*"))+[out/x for x in ("split_manifest.csv","frozen_r34d_probabilities.npz","team_baseline.json","team_student.npz","team_ensemble.npz","initial_random_weights.pt")]:pin(path)
    steps=len(source)//cfg["training"]["source_batch"]
    replay.write_json(out/"spec.json",dict(experiment_id=cfg["experiment_id"],at=replay.now(),code_sha256=identity(config),input_sha256=inputs,versions=versions,
              source_rows=len(source),validation_rows=len(vi),steps_per_epoch=steps,optimizer_updates_per_arm=steps*cfg["training"]["epochs"],
              true_contrastive_comparison_set=120,test_images_used=False,initialization="shared_random_state_dict",team_baseline=baseline))
    replay.log(f"Prepared: source={len(source)}, validation={len(vi)}, team primary={baseline['primary']:.6f}, updates/arm={steps*cfg['training']['epochs']}")


def make_states(cfg,out,ref,steps):
    states={}
    initial=torch.load(out/"initial_random_weights.pt",map_location="cpu",weights_only=True)
    for arm in ("A","B"):
        common.seed_all(cfg["seed"])
        model=SupConNet(ref,cfg);model.load_state_dict(initial,strict=True)
        model=model.to("cuda").to(memory_format=torch.channels_last)
        opt,schedule=common.optimizer_and_scheduler(model,cfg,steps)
        states[arm]=dict(model=model,ema=common.ema_model(model,cfg),opt=opt,schedule=schedule,history=[])
    return states


def paired_state(states,epoch,out):
    return dict(epoch=epoch,spec_sha256=replay.digest(out/"spec.json"),rng=common.rng_state(),arms={arm:dict(model=s["model"].state_dict(),
                ema=s["ema"].state_dict(),optimizer=s["opt"].state_dict(),scheduler=s["schedule"].state_dict(),history=s["history"]) for arm,s in states.items()})


def restore_paired(states,path,out):
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    assert checkpoint["spec_sha256"]==replay.digest(out/"spec.json")
    for arm,s in states.items():
        c=checkpoint["arms"][arm]
        s["model"].load_state_dict(c["model"],strict=True);s["ema"].load_state_dict(c["ema"],strict=True)
        s["opt"].load_state_dict(c["optimizer"]);s["schedule"].load_state_dict(c["scheduler"]);s["history"]=c["history"]
    common.restore_rng(checkpoint["rng"])
    return checkpoint["epoch"]


def update(arm,state,x,labels,cfg,epoch,step,steps):
    # A/B receive the same dropout realization. This also makes restart draws
    # independent of worker prefetch and diagnostic evaluation order.
    seed=common.stable_seed(cfg["seed"],epoch,step,"paired-model-dropout")
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=state["model"];model.train();state["opt"].zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):logits,embedding=model(x)
    ce=F.cross_entropy(logits.float(),labels,label_smoothing=cfg["training"]["label_smoothing"])
    contrastive=supcon_loss(embedding,labels,cfg["contrastive"]["temperature"])
    weight=contrastive_weight(cfg["contrastive"],epoch,step,steps,cfg["arms"][arm]["contrastive_loss_enabled"])
    loss=ce+weight*contrastive
    if not torch.isfinite(loss):raise ValueError("Non-finite training loss")
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["gradient_clip_norm"],error_if_nonfinite=True)
    state["opt"].step();state["schedule"].step();state["ema"].update(model)
    return dict(cross_entropy=float(ce.detach()),supcon_loss=float(contrastive.detach()),supcon_weight=weight,gradient_norm=float(norm))


def smoke(cfg,config,out,ref):
    verify(cfg,config,out);frame,_,source,_=common.metadata(cfg)
    data=SourceBatches(cfg,frame,source);batch=data[(21,0)];again=data[(21,0)]
    assert all(np.array_equal(batch[k],again[k]) for k in batch)
    assert batch["images"].shape==(120,3,288,288) and np.bincount(batch["labels"]).tolist()==[4]*30
    states=make_states(cfg,out,ref,data.steps)
    assert all(torch.equal(p,states["B"]["model"].state_dict()[k]) for k,p in states["A"]["model"].state_dict().items())
    x=common.normalize(batch["images"],ref);labels=torch.as_tensor(batch["labels"],device="cuda")
    started=time.monotonic();metrics={a:update(a,s,x,labels,cfg,21,0,data.steps) for a,s in states.items()}
    assert metrics["B"]["supcon_weight"]==1 and metrics["A"]["supcon_weight"]==0
    assert metrics["A"]["cross_entropy"]==metrics["B"]["cross_entropy"]
    assert metrics["A"]["supcon_loss"]==metrics["B"]["supcon_loss"]
    # Different gradients must actually change the encoder, rather than only the projection head.
    difference=max(float((p-states["B"]["model"].encoder.state_dict()[k]).abs().max()) for k,p in states["A"]["model"].encoder.state_dict().items() if p.is_floating_point())
    assert difference>0
    path=out/"smoke_checkpoint.pt";common.save_checkpoint(path,paired_state(states,0,out))
    assert restore_paired(states,path,out)==0
    expected=torch.load(path,map_location="cpu",weights_only=False)
    for arm,s in states.items():
        assert all(torch.equal(p.cpu(),expected["arms"][arm]["ema"][k]) for k,p in s["ema"].state_dict().items())
        assert all(torch.equal(p.cpu(),expected["arms"][arm]["model"][k]) for k,p in s["model"].state_dict().items())
        assert {int(v["step"]) for v in s["opt"].state.values()}=={1}
        assert s["schedule"].last_epoch==1
        assert all(p.grad is None for p in s["ema"].module.parameters())
    replay.write_json(out/"smoke_result.json",dict(passed=True,at=replay.now(),spec_sha256=replay.digest(out/"spec.json"),
                true_contrastive_batch=120,paired_updates=1,metrics=metrics,encoder_max_parameter_difference=difference,
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,duration_seconds=time.monotonic()-started,checkpoint_roundtrip_passed=True))
    replay.log(f"Paired GPU smoke passed: {json.dumps(metrics)}")


def finish(cfg,out,ref,frame,vi,states):
    from scripts.pilot_retrieval_graph import bootstrap
    c=cfg["validation"];base=json.loads((out/"team_baseline.json").read_text());a=states["A"]["history"][-1]["validation"];b=states["B"]["history"][-1]["validation"]
    gates=dict(primary_vs_initial=b["primary"]-base["primary"]>=c["minimum_B_gain_vs_initial_baseline"],primary_vs_control=b["primary"]-a["primary"]>=c["minimum_B_gain_vs_A"])
    for metric,key in (("clean","minimum_clean_raw_ensemble_delta_vs_baseline_and_A"),("mean_blur","minimum_mean_blur_raw_ensemble_delta_vs_baseline_and_A")):
        for name,comparison in (("initial",base),("control",a)):gates[f"{metric}_vs_{name}"]=b[metric]-comparison[metric]>=c[key]
    y=frame.iloc[vi].label.to_numpy();prior=np.bincount(frame.label,minlength=30)/len(frame)
    final=f"epoch_{cfg['training']['epochs']:03d}_ensemble.npz";prediction=ref.balance(np.load(out/"B"/final)["primary"],prior).argmax(1)
    intervals={}
    for name,path in (("initial",out/"team_ensemble.npz"),("control",out/"A"/final)):
        labels=ref.balance(np.load(path)["primary"],prior).argmax(1)
        intervals[name]=bootstrap(y,labels,prediction,frame.iloc[vi].reset_index(drop=True),cfg)
    result=dict(experiment_id=cfg["experiment_id"],at=replay.now(),fold=cfg["fold"],baseline=base,arms=dict(A=a,B=b),gates=gates,
                screening_passed=all(gates.values()),primary_gain_vs_initial=b["primary"]-base["primary"],primary_gain_vs_control=b["primary"]-a["primary"],
                paired_group_bootstrap_95_percent=intervals,additional_folds_started=False,kaggle_upload_performed=False)
    replay.write_json(out/"result.json",result);replay.write_json(out/"status.json",dict(state="completed",at=replay.now(),screening_passed=result["screening_passed"]))


def train(cfg,config,out,ref):
    verify(cfg,config,out);smoke_result=json.loads((out/"smoke_result.json").read_text())
    assert smoke_result["passed"] and smoke_result["spec_sha256"]==replay.digest(out/"spec.json")
    frame,_,source,vi=common.metadata(cfg);data=SourceBatches(cfg,frame,source);keys=common.EpochKeys(len(data))
    states=make_states(cfg,out,ref,data.steps);checkpoint=out/"paired_checkpoint.pt";completed=0
    if checkpoint.exists():completed=restore_paired(states,checkpoint,out)
    else:common.save_checkpoint(checkpoint,paired_state(states,0,out))
    for arm in states:(out/arm).mkdir(exist_ok=True)
    loader=DataLoader(data,batch_size=None,sampler=keys,collate_fn=common.identity,num_workers=cfg["execution"]["workers"],
                      persistent_workers=True,prefetch_factor=2,multiprocessing_context="spawn",worker_init_fn=common.init_worker,
                      generator=torch.Generator().manual_seed(cfg["seed"]))
    frozen=dict(np.load(out/"frozen_r34d_probabilities.npz"));y=frame.iloc[vi].label.to_numpy();prior=np.bincount(frame.label,minlength=30)/len(frame)
    epochs=cfg["training"]["epochs"]
    for epoch in range(completed+1,epochs+1):
        keys.epoch=epoch;started=time.monotonic();totals={a:dict(cross_entropy=0.,supcon_loss=0.,supcon_weight=0.) for a in states}
        for step,batch in enumerate(loader):
            x=common.normalize(batch["images"],ref);labels=torch.as_tensor(batch["labels"],device="cuda")
            metrics={a:update(a,s,x,labels,cfg,epoch,step,data.steps) for a,s in states.items()}
            for arm in states:
                for k in totals[arm]:totals[arm][k]+=metrics[arm][k]
            history=states["A"]["history"]
            replay.write_json(out/"status.json",dict(state="training",at=replay.now(),pid=os.getpid(),epoch=epoch,total_epochs=epochs,
                  optimizer_step_in_epoch=step+1,optimizer_steps_per_epoch=data.steps,completed_updates_per_arm=(epoch-1)*data.steps+step+1,
                  schedule="A_then_B_each_shared_batch",metrics=metrics,seconds_in_epoch=time.monotonic()-started,
                  estimated_seconds_remaining=float(np.median([h["paired_epoch_seconds"] for h in history[-3:]])*(epochs-epoch+1)) if len(history)>=3 else None))
        seconds=time.monotonic()-started
        for arm,s in states.items():
            record=dict(epoch=epoch,paired_epoch_seconds=seconds,learning_rate=s["schedule"].get_last_lr()[0],**{k:v/data.steps for k,v in totals[arm].items()})
            if epoch%cfg["validation"]["log_every_epochs"]==0 or epoch==epochs:
                rng=common.rng_state()
                record["validation"],student,ensemble=common.evaluate(s["ema"].module.encoder,cfg,ref,y,prior,out,frozen)
                common.save_npz(out/arm/f"epoch_{epoch:03d}_student.npz",**student);common.save_npz(out/arm/f"epoch_{epoch:03d}_ensemble.npz",**ensemble)
                common.restore_rng(rng)
            s["history"].append(record)
        common.save_checkpoint(checkpoint,paired_state(states,epoch,out))
        for arm,s in states.items():replay.write_json(out/arm/"history.json",s["history"])
        replay.log(f"epoch={epoch}/{epochs} paired_seconds={seconds:.1f} "+" ".join(f"{arm}:CE={s['history'][-1]['cross_entropy']:.4f},SupCon={s['history'][-1]['supcon_loss']:.4f}"+
                    (f",primary={s['history'][-1]['validation']['primary']:.6f}" if 'validation' in s['history'][-1] else '') for arm,s in states.items()))
    for arm,s in states.items():
        common.save_checkpoint(out/arm/"model.pt",s["ema"].module.encoder.state_dict())
        common.save_checkpoint(out/arm/"projection.pt",s["ema"].module.projection.state_dict())
        replay.write_json(out/arm/"done.json",dict(arm=arm,at=replay.now(),epochs=epochs,optimizer_updates=epochs*data.steps,
                    spec_sha256=replay.digest(out/"spec.json"),metrics=s["history"][-1]["validation"],gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    finish(cfg,out,ref,frame,vi,states)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,default=ROOT/"configs/LOCAL-PROP-006-SUPCON.yaml")
    modes=parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare","smoke","run"):modes.add_argument("--"+name,action="store_true")
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text());out=ROOT/cfg["output_root"];out.mkdir(parents=True,exist_ok=True)
    with (out/"runner.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            ref=common.setup(cfg)
            if args.prepare:prepare(cfg,args.config,out,ref)
            elif args.smoke:smoke(cfg,args.config,out,ref)
            else:train(cfg,args.config,out,ref)
        except torch.cuda.OutOfMemoryError:
            replay.write_json(out/"status.json",dict(state="waiting_after_cuda_oom",at=replay.now(),resume="last_complete_paired_epoch",batch_sizes_unchanged=True))
            traceback.print_exc();raise SystemExit(cfg["execution"]["oom_exit_code"])
        except Exception as error:
            replay.write_json(out/"status.json",dict(state="failed",at=replay.now(),error=str(error)));raise


if __name__=="__main__":
    sys.path.insert(0,str(ROOT));main()
