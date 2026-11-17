"""Research-only BioCLIP2.5 probe and partial fine-tuning; never a competition artifact.

Only official training photographs enter local optimization. External pretrained
weights make this experiment ineligible under the audited competition rules.
No test inference, submission generation, network upload or text encoder is used.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import gc
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback
import warnings
import cv2
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml
try:
    from . import train_pvt_idmm as source
except ImportError:
    import train_pvt_idmm as source
common, replay, ROOT = source.common, source.replay, source.ROOT
CONDITIONS = ('primary','clean','blur06','blur10')
VERSIONS = (*replay.VERSIONS,'open_clip_torch','ftfy','huggingface_hub','safetensors')


def validate_config(cfg):
    if not cfg['research_only'] or cfg['competition_eligible'] or cfg['automatic_submission']:
        raise ValueError('Pretrained experiment must stay research-only without submission')
    if not cfg['experiment_id'].startswith('RESEARCH-') or not cfg['output_root'].startswith('outputs/RESEARCH-'):
        raise ValueError('Keep pretrained artifacts separate from scratch competition experiments')
    if cfg['execution']['target_images_used'] or not cfg['pretrained']['visual_encoder_only']:
        raise ValueError('Only source photographs and the image encoder are supported')
    if not cfg['training']['pretrained'] or cfg['validation']['tta'] != 'flip':
        raise ValueError('Expected explicitly declared pretrained flip-TTA experiment')
    if cfg['training']['image_size'] != cfg['validation']['image_size']:
        raise ValueError('Training and evaluation resolution must match')


def identity(config):
    from open_clip.model import VisionTransformer
    files=[config,Path(__file__),ROOT/'scripts/launch_bioclip_research.py',
           Path(source.__file__),Path(common.__file__),Path(replay.__file__),Path(inspect.getfile(VisionTransformer))]
    return {str(p.resolve()):replay.digest(p) for p in files}


def verify(cfg,config,out):
    validate_config(cfg); spec=json.loads((out/'spec.json').read_text())
    assert spec['code_sha256']==identity(config),'Prepared code/config changed'
    assert spec['versions']=={x:importlib.metadata.version(x) for x in VERSIONS},'Environment changed'
    for path,expected in spec['input_sha256'].items():
        if replay.digest(path)!=expected:raise ValueError(f'Input changed: {path}')
    return spec


def receipt(cfg):
    r=json.loads((ROOT/cfg['pretrained']['weights_receipt']).read_text())
    assert r['state']=='completed' and r['revision']==cfg['pretrained']['revision']
    assert r['repo']==cfg['pretrained']['repository']
    assert r['sha256']['open_clip_model.safetensors']==cfg['pretrained']['checkpoint_sha256']
    return r


def prepare(cfg,config,out):
    validate_config(cfg)
    if (out/'spec.json').exists():verify(cfg,config,out);return
    weights=receipt(cfg); frame,_,si,vi=common.metadata(cfg)
    upstream=ROOT/cfg['reference_experiment_root']; old=json.loads((upstream/'spec.json').read_text())['input_sha256']
    inputs={}
    def pin(path,expected=None):
        path=Path(path); value=replay.digest(path)
        if expected is not None and value!=expected:raise ValueError(f'Input differs from scene-fold reference: {path}')
        inputs[str(path.resolve())]=value
    for path,expected in weights['sha256'].items():pin(weights['paths'][path],expected)
    pin(ROOT/cfg['pretrained']['weights_receipt'])
    for name in ('train.csv','class_mapping.csv'):pin(ROOT/cfg['dataset_root']/name)
    for name in ('cache/scene_folds.csv','cache/scene_folds.npy'):
        path=Path(cfg['team_root'])/name;pin(path,old[str(path.resolve())])
    for name in ('train.py','dq.py','evalres.py'):pin(ROOT/cfg['reference_root']/'reference_sources'/name)
    paths=[ROOT/cfg['dataset_root']/'train'/row.id for row in frame.itertuples()]
    with ThreadPoolExecutor(max_workers=cfg['execution']['cpu_threads']) as pool:
        for path,value in zip(paths,pool.map(replay.digest,paths)):
            assert old[str(path.resolve())]==value;inputs[str(path.resolve())]=value
    ref=replay.load_reference(ROOT/cfg['reference_root']/'reference_sources'); size=cfg['training']['image_size']
    cache=out/'cache';cache.mkdir(exist_ok=True)
    replay.log('Preparing source-only 224px clean cache and fixed held-out evaluation views')
    def read(index):return cv2.resize(ref.dq.read_rgb(paths[index]),(size,size),interpolation=cv2.INTER_CUBIC)
    with ThreadPoolExecutor(max_workers=cfg['execution']['cpu_threads']) as pool:
        common.save_array(cache/'source_clean.npy',np.stack(list(pool.map(read,si))))
    for condition in CONDITIONS:
        path=upstream/f'cache/{condition}_352.npy';pin(path,old[str(path.resolve())])
        original=np.load(path,mmap_mode='r')
        common.save_array(cache/f'{condition}.npy',np.stack([cv2.resize(x,(size,size),interpolation=cv2.INTER_CUBIC) for x in original]))
    frozen=upstream/'frozen_r34d_probabilities.npz';pin(frozen,old[str(frozen.resolve())])
    manifest=frame.copy();manifest['role']=np.where(frame.fold==cfg['fold'],'validation_only','source_training')
    manifest.to_csv(out/'split_manifest.csv',index=False)
    for path in [out/'split_manifest.csv',*sorted(cache.glob('*.npy'))]:pin(path)
    replay.write_json(out/'spec.json',dict(at=replay.now(),experiment_id=cfg['experiment_id'],research_only=True,
        competition_eligible=False,external_pretraining=True,external_pretraining_overlap_with_local_images_unknown=True,
        code_sha256=identity(config),versions={x:importlib.metadata.version(x) for x in VERSIONS},input_sha256=inputs,
        source_rows=len(si),validation_rows=len(vi),fold=cfg['fold'],seed=cfg['seed'],steps_per_epoch=math.ceil(len(si)/cfg['training']['source_batch']),
        target_images_used=False,main_metric='standalone raw primary macro F1',selection=cfg['training']['checkpoint_selection'],
        validation_resolution=size,validation_views='same frozen 352px views resized bicubic to model-native224; not identical resolution to scratch comparison'))


def mark_trainable(model,last):
    for p in model.parameters():p.requires_grad_(False)
    blocks=model.visual.transformer.resblocks
    if not 0<last<=len(blocks):raise ValueError('Invalid number of trainable blocks')
    for block in blocks[-last:]:
        for p in block.parameters():p.requires_grad_(True)
    for p in model.visual.ln_post.parameters():p.requires_grad_(True)
    model.visual.proj.requires_grad_(True)
    for p in model.head.parameters():p.requires_grad_(True)
    return {k:p for k,p in model.named_parameters() if p.requires_grad}


class BioClassifier(nn.Module):
    def __init__(self,visual,dimension,classes=30):
        super().__init__();self.visual=visual;self.head=nn.Linear(dimension,classes)
    def features(self,x):return F.normalize(self.visual(x).float(),dim=-1)
    def forward(self,x):return self.head(self.features(x))


def make_model(cfg):
    from open_clip.model import _build_vision_tower
    from safetensors import safe_open
    r=receipt(cfg); raw=json.loads(Path(r['paths']['open_clip_config.json']).read_text())
    architecture=raw['model_cfg']; assert architecture['vision_cfg']['layers']==32
    visual=_build_vision_tower(architecture['embed_dim'],architecture['vision_cfg'])
    with safe_open(r['paths']['open_clip_model.safetensors'],framework='pt',device='cpu') as tensors:
        weights={k.removeprefix('visual.'):tensors.get_tensor(k) for k in tensors.keys() if k.startswith('visual.')}
    visual.load_state_dict(weights,strict=True);del weights
    model=BioClassifier(visual,architecture['embed_dim']);mark_trainable(model,cfg['training']['trainable_last_blocks'])
    preprocessing=raw['preprocess_cfg'];model.register_buffer('input_mean',torch.tensor(preprocessing['mean'])[None,:,None,None]*255,persistent=False)
    model.register_buffer('input_std',torch.tensor(preprocessing['std'])[None,:,None,None]*255,persistent=False)
    return model.cuda().to(memory_format=torch.channels_last)


def normalize(model,images):
    x=torch.as_tensor(np.ascontiguousarray(images),device='cuda').float().to(memory_format=torch.channels_last)
    return (x-model.input_mean)/model.input_std


def setup(cfg):
    torch.set_num_threads(cfg['execution']['cpu_threads']);cv2.setNumThreads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(cfg['execution']['gpu_memory_fraction'])
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;common.seed_all(cfg['seed'])


@torch.no_grad()
def extract(model,array,cfg,out,name):
    path=out/f'features_{name}.npz'; spec_sha=replay.digest(out/'spec.json')
    if path.exists():
        with np.load(path) as saved:
            assert saved['spec_sha256'].item()==spec_sha
            features=saved['features']
        assert features.shape==(len(array),2,model.head.in_features) and np.isfinite(features).all()
        return features
    model.eval();parts=[];batch=cfg['validation']['batch_size']
    for start in range(0,len(array),batch):
        x=normalize(model,np.asarray(array[start:start+batch]).transpose(0,3,1,2))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f=torch.stack([model.features(x),model.features(x.flip(3))],dim=1)
        parts.append(f.cpu().numpy())
        replay.write_json(out/'status.json',dict(at=replay.now(),state='frozen_feature_extraction',condition=name,rows=min(start+batch,len(array)),total=len(array)))
    result=np.concatenate(parts);common.save_npz(path,features=result,spec_sha256=np.array(spec_sha));return result


def probe_probabilities(model,features):
    weights=model.head.weight.detach().cpu().numpy();bias=model.head.bias.detach().cpu().numpy()
    scores=features@weights.T+bias;scores-=scores.max(-1,keepdims=True)
    p=np.exp(scores);p/=p.sum(-1,keepdims=True);return p.mean(1)


def metrics(probabilities,y,frozen):
    result={name:common.score(y,p) for name,p in probabilities.items()}
    result['mean_blur']=(result['blur06']+result['blur10'])/2
    result['matching_pair_raw_primary']=common.score(y,(probabilities['primary']+frozen['primary'])/2)
    return result


def baseline(model,cfg,out,frame,si,vi):
    path=out/'linear_probe.json'
    if path.exists():
        state=torch.load(out/'linear_probe_head.pt',map_location='cpu',weights_only=True);model.head.load_state_dict(state,strict=True)
        return json.loads(path.read_text())
    started=time.monotonic();cache=out/'cache';feature=extract(model,np.load(cache/'source_clean.npy',mmap_mode='r'),cfg,out,'source')
    replay.log('Fitting fixed-C multinomial classifier on frozen pretrained source features')
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        clf=LogisticRegression(C=cfg['linear_probe']['C'],max_iter=cfg['linear_probe']['max_iter'],random_state=cfg['seed'],tol=1e-6)
        clf.fit(feature.reshape(-1,feature.shape[-1]),np.repeat(frame.iloc[si].label.to_numpy(),2))
    assert np.array_equal(clf.classes_,np.arange(30))
    model.head.load_state_dict({'weight':torch.from_numpy(clf.coef_.astype(np.float32)),'bias':torch.from_numpy(clf.intercept_.astype(np.float32))},strict=True)
    common.save_checkpoint(out/'linear_probe_head.pt',model.head.state_dict());probabilities={}
    for name in CONDITIONS:
        f=extract(model,np.load(cache/f'{name}.npy',mmap_mode='r'),cfg,out,name)
        probabilities[name]=probe_probabilities(model,f)
    common.save_npz(out/'linear_probe_probabilities.npz',**probabilities)
    frozen=dict(np.load(ROOT/cfg['reference_experiment_root']/'frozen_r34d_probabilities.npz'))
    result=dict(at=replay.now(),duration_seconds=time.monotonic()-started,metrics=metrics(probabilities,frame.iloc[vi].label.to_numpy(),frozen),
                local_source_images=len(si),feature_training_rows=len(si)*2,pretrained_backbone_frozen=True,classifier_only=True,competition_eligible=False,
                spec_sha256=replay.digest(out/'spec.json'),head_sha256=replay.digest(out/'linear_probe_head.pt'))
    replay.write_json(path,result);replay.log(f'Frozen BioCLIP linear probe: {result["metrics"]}');return result


def optimizer(model,cfg,steps):
    c=cfg['training'];groups=[]
    for head in (False,True):
        for decay in (False,True):
            params=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith('head.')==head and (p.ndim>1)==decay]
            if params:groups.append(dict(params=params,lr=c['head_learning_rate'] if head else c['backbone_learning_rate'],weight_decay=c['weight_decay'] if decay else 0))
    opt=torch.optim.AdamW(groups);total=steps*c['epochs'];warm=steps*c['warmup_epochs']
    def rate(step):
        if step<warm:return (step+1)/warm
        progress=min(1.,(step-warm)/max(1,total-warm))
        return c['minimum_lr_ratio']+(1-c['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*progress))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,rate)


def delta(model):return {k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}


def apply_delta(model,saved):
    parameters={k:p for k,p in model.named_parameters() if p.requires_grad}
    if set(parameters)!=set(saved):raise ValueError('Trainable checkpoint keys changed')
    with torch.no_grad():
        for name,param in parameters.items():param.copy_(saved[name].to(param.device))


@contextmanager
def ema_weights(model,ema):
    saved=delta(model);apply_delta(model,ema)
    try:yield
    finally:apply_delta(model,saved)


def update(model,opt,schedule,ema,batch,cfg,epoch,step):
    common.seed_all(common.stable_seed(cfg['seed'],epoch,step,'bioclip-ft'));model.train();opt.zero_grad(set_to_none=True)
    images,labels=batch['images'],batch['labels'];c=cfg['training']
    seed=common.stable_seed(cfg['seed'],epoch,step,'bioclip-mixup')
    if np.random.RandomState(seed).rand()<c['mixup_probability']:
        images,first,second,lam=source.mix_batch(images,labels,seed,'mixup',c['mixup_alpha'])
    else:first,second,lam=labels,labels,1.
    total=0.;start_time=time.monotonic()
    for start in range(0,len(labels),c['microbatch']):
        end=min(len(labels),start+c['microbatch']);x=normalize(model,images[start:end]);a=torch.as_tensor(first[start:end],device='cuda');b=torch.as_tensor(second[start:end],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
        loss=source.mixed_loss(logits,a,b,lam,c['label_smoothing'])*(end-start)/len(labels)
        if not torch.isfinite(loss):raise ValueError('Nonfinite fine-tuning loss')
        loss.backward();total+=float(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['gradient_clip_norm'],error_if_nonfinite=True)
    opt.step();schedule.step()
    with torch.no_grad():
        for name,param in model.named_parameters():
            if name in ema:ema[name].lerp_(param.detach(),1-c['ema_decay'])
    return dict(loss=total,gradient_norm=float(norm),seconds=time.monotonic()-start_time)


@torch.no_grad()
def predict(model,array,cfg):
    model.eval();parts=[];batch=cfg['validation']['batch_size']
    for start in range(0,len(array),batch):
        x=normalize(model,np.asarray(array[start:start+batch]).transpose(0,3,1,2))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            p=(model(x).float().softmax(1)+model(x.flip(3)).float().softmax(1))/2
        parts.append(p.cpu().numpy())
    result=np.concatenate(parts);replay.compare_probabilities(result,result,0,0);return result


def smoke(cfg,config,out):
    verify(cfg,config,out);setup(cfg);model=make_model(cfg);frame,_,si,_=common.metadata(cfg)
    data=source.SourceBatches(cfg,frame,si,'supervised');batch=data[(1,0)];assert len(batch['labels'])==32
    opt,schedule=optimizer(model,cfg,len(data));trainable=mark_trainable(model,cfg['training']['trainable_last_blocks'])
    ema={k:p.detach().clone() for k,p in trainable.items()};frozen=model.visual.conv1.weight.detach().clone();key=next(k for k in trainable if '.resblocks.' in k)
    before=trainable[key].detach().clone();loss=update(model,opt,schedule,ema,batch,cfg,1,0)
    assert torch.equal(frozen,model.visual.conv1.weight) and model.visual.conv1.weight.grad is None
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert not torch.equal(before,trainable[key]) and schedule.last_epoch==1
    checkpoint=out/'smoke_delta.pt';common.save_checkpoint(checkpoint,delta(model));saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert all(torch.equal(saved[k],p.detach().cpu()) for k,p in trainable.items());checkpoint.unlink()
    replay.write_json(out/'smoke_result.json',dict(at=replay.now(),passed=True,spec_sha256=replay.digest(out/'spec.json'),metrics=loss,
        trainable_parameters=sum(p.numel() for p in trainable.values()),visual_parameters=sum(p.numel() for p in model.visual.parameters()),
        frozen_parameters_unchanged=True,trainable_transformer_block_changed=True,checkpoint_roundtrip_exact=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,projected_training_compute_hours=loss['seconds']*len(data)*cfg['training']['epochs']/3600,
        projection_excludes_feature_extraction_evaluation_checkpoint_overhead=True,competition_eligible=False))


def run(cfg,config,out):
    spec=verify(cfg,config,out);smoke_result=json.loads((out/'smoke_result.json').read_text())
    assert smoke_result['passed'] and smoke_result['spec_sha256']==replay.digest(out/'spec.json')
    setup(cfg);model=make_model(cfg);frame,_,si,vi=common.metadata(cfg);probe=baseline(model,cfg,out,frame,si,vi)
    data=source.SourceBatches(cfg,frame,si,'supervised');keys=common.EpochKeys(len(data))
    loader=DataLoader(data,batch_size=None,sampler=keys,collate_fn=common.identity,num_workers=cfg['execution']['workers'],worker_init_fn=common.init_worker,persistent_workers=True,prefetch_factor=1,multiprocessing_context='spawn')
    trainable=mark_trainable(model,cfg['training']['trainable_last_blocks']);opt,schedule=optimizer(model,cfg,len(data));ema={k:p.detach().clone() for k,p in trainable.items()}
    completed=0;history=[];checkpoint=out/'checkpoint.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['spec_sha256']==replay.digest(out/'spec.json')
        apply_delta(model,saved['model']);opt.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['scheduler'])
        ema={k:v.cuda() for k,v in saved['ema'].items()};completed=saved['epoch'];history=saved['history'];common.restore_rng(saved['rng'])
    frozen=dict(np.load(ROOT/cfg['reference_experiment_root']/'frozen_r34d_probabilities.npz'));started=time.monotonic()
    for epoch in range(completed+1,cfg['training']['epochs']+1):
        keys.epoch=epoch;losses=[]
        for step,batch in enumerate(loader):
            value=update(model,opt,schedule,ema,batch,cfg,epoch,step);losses.append(value['loss'])
            replay.write_json(out/'status.json',dict(at=replay.now(),state='partial_fine_tuning',epoch=epoch,epochs=cfg['training']['epochs'],step=step+1,steps=len(data),loss=float(np.mean(losses)),trainable_last_blocks=cfg['training']['trainable_last_blocks'],elapsed_seconds=time.monotonic()-started,competition_eligible=False))
        if epoch%cfg['validation']['log_every_epochs']==0 or epoch==cfg['training']['epochs']:
            with ema_weights(model,ema):probabilities={name:predict(model,np.load(out/f'cache/{name}.npy',mmap_mode='r'),cfg) for name in CONDITIONS}
            value=metrics(probabilities,frame.iloc[vi].label.to_numpy(),frozen);history.append(dict(epoch=epoch,at=replay.now(),metrics=value))
            common.save_npz(out/f'epoch_{epoch}_probabilities.npz',**probabilities);replay.write_json(out/'history.json',history);replay.log(f'Epoch {epoch}: {value}')
        common.save_checkpoint(checkpoint,dict(spec_sha256=replay.digest(out/'spec.json'),epoch=epoch,model=delta(model),ema={k:v.detach().cpu() for k,v in ema.items()},optimizer=opt.state_dict(),scheduler=schedule.state_dict(),history=history,rng=common.rng_state()))
        replay.log(f'Fine-tuning epoch {epoch}/{cfg["training"]["epochs"]}, loss={np.mean(losses):.5f}')
    common.save_checkpoint(out/'final_ema_delta.pt',{k:v.detach().cpu() for k,v in ema.items()})
    verify(cfg,config,out);final=history[-1]['metrics']
    replay.write_json(out/'result.json',dict(at=replay.now(),experiment_id=cfg['experiment_id'],research_only=True,competition_eligible=False,completed_epochs=cfg['training']['epochs'],updates=schedule.last_epoch,selection='final_epoch_30_EMA',linear_probe=probe['metrics'],fine_tuned=final,primary_delta_vs_probe=final['primary']-probe['metrics']['primary'],spec_sha256=replay.digest(out/'spec.json'),final_ema_delta_sha256=replay.digest(out/'final_ema_delta.pt'),pretrained_sha256=cfg['pretrained']['checkpoint_sha256'],target_images_used=False,kaggle_upload_performed=False))
    replay.write_json(out/'status.json',dict(at=replay.now(),state='completed',metrics=final,research_only=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'configs/RESEARCH-010-BIOCLIP25-FT.yaml')
    group=parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare','smoke','run'):group.add_argument('--'+action,action='store_true')
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text());out=ROOT/cfg['output_root'];out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:(prepare if args.prepare else smoke if args.smoke else run)(cfg,args.config,out)
        except torch.cuda.OutOfMemoryError:
            traceback.print_exc();sys.exit(cfg['execution']['oom_exit_code'])

if __name__=='__main__':main()
