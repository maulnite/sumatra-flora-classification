"""Infer final PVT448 and compare CSVs without training or Kaggle upload.

Reproduce our previous public-best CSV first. Change only its PVT checkpoint,
keeping nine reference members, reduction order, prior balancing and SIFT fixed.
The resulting file has no measured public score. Teammate sources are read-only.
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
from . import train_pvt_high_resolution as high
from .prepare_pvt_public_probe import FAMILIES, replace_member
from .evaluate_strong_baseline import serialize_submission, sift_labels
from .validate_submission import validate

ROOT, common, replay, pvt = high.ROOT, high.common, high.replay, high.pvt


def compare(first, second, sample, mapping, out, prefix):
    assert first.id.equals(sample.id) and second.id.equals(sample.id)
    paired = pd.DataFrame(dict(id=sample.id, ours_label=first.label, reference_label=second.label))
    paired['ours_species'] = paired.ours_label.map(mapping)
    paired['reference_species'] = paired.reference_label.map(mapping)
    paired['different'] = paired.ours_label != paired.reference_label
    paired.to_csv(out/f'{prefix}_all.csv', index=False)
    changed = paired[paired.different]
    changed.to_csv(out/f'{prefix}_differences.csv', index=False)
    flows = changed.groupby(['ours_label','ours_species','reference_label','reference_species']).size().rename('count').reset_index()
    flows = flows.sort_values(['count','ours_label','reference_label'], ascending=[False,True,True])
    flows.to_csv(out/f'{prefix}_transitions.csv', index=False)
    return dict(rows=len(paired), same=int((~paired.different).sum()), different=int(paired.different.sum()),
                agreement_percent=float((~paired.different).mean()*100), top_transitions=flows.head(10).to_dict(orient='records'))


def prepare(cfg, config, out):
    started = time.monotonic(); hashes = {}
    def pin(path, expected=None):
        path = Path(path); value = replay.digest(path)
        if expected is not None and value != expected: raise ValueError(f'Changed input: {path}')
        hashes[str(path.resolve())] = value
        return path
    if cfg['automatic_upload']: raise ValueError('Comparison must not upload')
    meta_path = pin(ROOT/cfg['training_config']); meta = yaml.safe_load(meta_path.read_text())
    _, training = high.resolve(meta); train_out = ROOT/training['output_root']
    _, spec = high.verify(meta, meta_path, train_out)
    result = json.loads(pin(train_out/'result.json').read_text())
    checkpoint = pin(ROOT/cfg['checkpoint'],cfg['checkpoint_sha256'])
    assert result['model_sha256'] == cfg['checkpoint_sha256'] and result['completed_epochs'] == 100
    assert result['spec_sha256'] == replay.digest(train_out/'spec.json')
    assert high.screening(result['metrics'],result['baseline'],meta['gate']) == result['gate'] and result['gate']['passed']
    assert (cfg['inference_size'],cfg['inference_batch_size'],cfg['tta']) == (352,16,'flip')
    assert training['validation']['student_inference_size'] == 352 and training['validation']['batch_size'] == 16
    old = yaml.safe_load(pin(ROOT/cfg['previous_public_config']).read_text())
    old_receipt = json.loads(pin(ROOT/cfg['previous_public_receipt']).read_text())['preparation']
    old_csv = pin(ROOT/cfg['previous_public_csv'], cfg['previous_public_csv_sha256'])
    nafis_csv = pin(cfg['nafis_csv'],cfg['nafis_csv_sha256'])
    test_path = pin(cfg['test_cache'],cfg['test_cache_sha256'])
    manifest = pin(ROOT/old['baseline_member_manifest'],old['baseline_member_manifest_sha256'])
    members = json.loads(manifest.read_text())['preparation']['input_sha256']
    frame,sample,_,_ = common.metadata(training); prior = np.bincount(frame.label,minlength=30)/len(frame)
    assert cfg['replacement_fold'] == old['replacement_fold'] == training['fold'] == 2
    families = []
    for run in FAMILIES:
        paths = [ROOT/training['reference_root']/'predictions'/f'{run}_f{fold}'/'test.npy' for fold in old['folds']]
        families.append([np.load(pin(path,members[str(path.resolve())])) for path in paths])
    old_probability_path = ROOT/'outputs/LOCAL-PROP-008-B3-PUBLIC/B3_test.npy'
    old_probability = np.load(pin(old_probability_path,old_receipt['probability_sha256']))
    sift_path = Path(training['team_root'])/'runs/sift_test.csv'
    matches = pd.read_csv(pin(sift_path,old_receipt['input_sha256'][str(sift_path)])); assert matches.id.equals(sample.id)
    ref = replay.load_reference(ROOT/training['reference_root']/'reference_sources')
    def labels(probability):
        combined = replace_member(families,probability,cfg['replacement_fold'])
        balanced = ref.balance(combined,prior,cfg['balance_iterations'])
        predicted,_ = sift_labels(balanced.argmax(1),matches,cfg['sift_minimum_inliers'])
        return predicted, combined
    old_labels,_ = labels(old_probability)
    assert serialize_submission(sample.id,old_labels) == old_csv.read_bytes()
    replay.log('Previous public-best CSV reproduced byte-for-byte; replaying final PVT448 validation')
    ref = common.setup(training)
    model = pvt.make_model(training,meta['architecture'])
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model = model.cuda().to(memory_format=torch.channels_last).eval()
    val_path = pin(ROOT/training['reference_experiment_root']/'cache/primary_352.npy')
    validation = common.predict(model,np.load(val_path,mmap_mode='r'),training,ref)
    stored_path = pin(train_out/'candidate/epoch_100_student.npz')
    check = replay.compare_probabilities(validation,np.load(stored_path)['primary'],1e-5,1e-4)
    assert check['within_tolerance'] and check['predicted_labels_different'] == 0
    common.save_array(out/'PVT448_validation.npy',validation)
    replay.log('Inferring 2010 test images at matched 352px, batch16, flip TTA')
    test = np.load(test_path,mmap_mode='r'); assert test.shape == (2010,352,352,3) and test.dtype == np.uint8
    probability = common.predict(model,test,training,ref)
    replay.compare_probabilities(probability,probability,0,0)
    common.save_array(out/'PVT448_test.npy',probability)
    candidate_labels,combined = labels(probability)
    common.save_array(out/'ensemble_raw_test.npy',combined)
    candidate = out/'LOCAL-PROP-009-PVT448_f2replace_candidate.csv'
    candidate.write_bytes(serialize_submission(sample.id,candidate_labels))
    for path in (candidate,old_csv,nafis_csv): assert not validate(path,ROOT/training['dataset_root']/'sample_submission.csv')
    mapping = pd.read_csv(ROOT/training['dataset_root']/'class_mapping.csv').set_index('label').species
    fresh,previous,nafis = [pd.read_csv(path) for path in (candidate,old_csv,nafis_csv)]
    comparisons = dict(previous_public_vs_nafis=compare(previous,nafis,sample,mapping,out,'public_best_vs_nafis'),
                       PVT448_vs_nafis=compare(fresh,nafis,sample,mapping,out,'PVT448_vs_nafis'),
                       PVT448_vs_previous_public=compare(fresh,previous,sample,mapping,out,'PVT448_vs_previous_public'))
    for path,expected in hashes.items(): assert replay.digest(path) == expected, f'Input changed during inference: {path}'
    summary = dict(at=replay.now(),experiment_id=cfg['experiment_id'],duration_seconds=time.monotonic()-started,
                   checkpoint_sha256=cfg['checkpoint_sha256'],selection='preregistered_final_epoch_100_EMA',inference_size=352,tta='flip',
                   previous_public_csv_byte_identical=True,validation_replay=check,rows=len(sample),comparisons=comparisons,
                   new_csv=str(candidate.relative_to(ROOT)),new_csv_sha256=replay.digest(candidate),
                   new_probability_sha256=replay.digest(out/'PVT448_test.npy'),same_nine_members_and_postprocessing=True,
                   new_model_total_weight=.1,new_candidate_public_score=None,nafis_public=.90780,previous_public=.87197,
                   new_checkpoint_local_pair_primary=result['metrics']['primary'],accuracy_of_disagreements_unknown=True,
                   training_performed=False,kaggle_upload_performed=False,all_source_files_unchanged=True,
                   input_sha256=hashes,config_sha256=replay.digest(config),runner_sha256=replay.digest(Path(__file__)))
    replay.write_json(out/'comparison.json',summary)
    replay.log(json.dumps({k:dict(same=v['same'],different=v['different'],agreement_percent=v['agreement_percent']) for k,v in comparisons.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/LOCAL-PROP-009-CSV-COMPARE.yaml')
    args=parser.parse_args(); cfg=yaml.safe_load(args.config.read_text()); out=ROOT/cfg['output_root']; out.mkdir(parents=True,exist_ok=True)
    with (out/'comparison.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(cfg,args.config,out)

if __name__ == '__main__': main()
