import copy
from pathlib import Path
import unittest
import torch
from torch import nn
import yaml
from scripts import train_bioclip_research as r

CFG=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/RESEARCH-010-BIOCLIP25-FT.yaml').read_text())

class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__();self.conv1=nn.Linear(4,4)
        self.transformer=nn.Module();self.transformer.resblocks=nn.ModuleList([nn.Linear(4,4) for _ in range(8)])
        self.ln_post=nn.LayerNorm(4);self.proj=nn.Parameter(torch.randn(4,3))
    def forward(self,x):
        x=self.conv1(x)
        for block in self.transformer.resblocks:x=x+.1*torch.tanh(block(x))
        return self.ln_post(x)@self.proj

class ResearchTests(unittest.TestCase):
    def test_scope_rejects_competition_upload_and_target_training(self):
        r.validate_config(CFG)
        for where,key,value in [(None,'competition_eligible',True),(None,'automatic_submission',True),
                                (None,'output_root','outputs/LOCAL-PROP-001'),('execution','target_images_used',True)]:
            cfg=copy.deepcopy(CFG)
            if where:cfg[where][key]=value
            else:cfg[key]=value
            with self.assertRaises(ValueError):r.validate_config(cfg)

    def test_only_selected_blocks_and_classifier_update(self):
        torch.manual_seed(44);model=r.BioClassifier(TinyVisual(),3);selected=r.mark_trainable(model,2)
        self.assertFalse(any(name.startswith('visual.transformer.resblocks.5.') for name in selected))
        self.assertTrue(any(name.startswith('visual.transformer.resblocks.6.') for name in selected))
        before={name:p.detach().clone() for name,p in model.named_parameters()}
        opt=torch.optim.AdamW(selected.values(),lr=.01)
        torch.nn.functional.cross_entropy(model(torch.randn(8,4)),torch.arange(8)%4).backward();opt.step()
        for name,p in model.named_parameters():
            if not p.requires_grad:self.assertTrue(torch.equal(before[name],p),name);self.assertIsNone(p.grad)
        self.assertTrue(any(not torch.equal(before[name],p) for name,p in selected.items() if '.resblocks.' in name))

    def test_delta_and_ema_restore_even_when_evaluation_fails(self):
        model=r.BioClassifier(TinyVisual(),3);r.mark_trainable(model,2)
        original=r.delta(model);ema={name:p+1 for name,p in original.items()}
        with self.assertRaisesRegex(RuntimeError,'evaluation'):
            with r.ema_weights(model,ema):
                self.assertTrue(all(torch.equal(r.delta(model)[name],value) for name,value in ema.items()))
                raise RuntimeError('evaluation')
        self.assertTrue(all(torch.equal(r.delta(model)[name],value) for name,value in original.items()))
        with self.assertRaises(ValueError):r.apply_delta(model,{})

if __name__=='__main__':unittest.main()
