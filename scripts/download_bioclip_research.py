"""Download and verify pinned public BioCLIP weights for the separate research run."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    temporary=path.with_suffix('.tmp.json');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/RESEARCH-010-BIOCLIP25-FT.yaml')
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text());p=cfg['pretrained']
    if not cfg['research_only'] or cfg['competition_eligible'] or cfg['automatic_submission']:
        raise ValueError('Pinned pretrained download is only configured for the separate research experiment')
    out=ROOT/cfg['output_root'];out.mkdir(parents=True,exist_ok=True);receipt=ROOT/p['weights_receipt']
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        assert saved['revision']==p['revision'] and saved['repo']==p['repository']
        assert saved['sha256']['open_clip_model.safetensors']==p['checkpoint_sha256']
        for name,path in saved['paths'].items():assert digest(path)==saved['sha256'][name]
        print('Pinned weights already verified; existing receipt unchanged.');return
    os.environ.setdefault('HF_HUB_DISABLE_XET','1');os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
    from huggingface_hub import hf_hub_download
    for attempt in range(1,4):
        try:
            paths={name:hf_hub_download(repo_id=p['repository'],revision=p['revision'],filename=name,cache_dir=out/'pretrained_cache',token=False)
                   for name in ['open_clip_config.json','open_clip_model.safetensors']}
            hashes={name:digest(path) for name,path in paths.items()}
            assert hashes['open_clip_model.safetensors']==p['checkpoint_sha256']
            value=dict(at=datetime.now(timezone.utc).isoformat(),state='completed',repo=p['repository'],revision=p['revision'],paths=paths,sha256=hashes,
                       checkpoint_bytes=Path(paths['open_clip_model.safetensors']).stat().st_size,external_pretrained=True,competition_eligible=False)
            write(receipt,value);write(out/'download_status.json',value);print('Pinned weights downloaded and verified.');return
        except Exception:
            if attempt==3:raise
            time.sleep(15)

if __name__=='__main__':main()
