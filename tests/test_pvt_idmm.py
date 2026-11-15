"""Behavioral checks for fold isolation, mixing, accumulation and head transfer."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from scripts import train_pvt_idmm as p

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/LOCAL-PROP-008-PVT-IDMM.yaml").read_text())


class Identity:
    def set_random_seed(self, seed):
        pass

    def __call__(self, image):
        return {"image": image}


class Tiny(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.body = torch.nn.Linear(5, 7)
        self.head = torch.nn.Linear(7, classes)

    def forward(self, x):
        return self.head(self.body(x).tanh())


class TestPvtIdmm(unittest.TestCase):
    def test_source_epoch_includes_every_photo_once_and_final_partial(self):
        source = np.arange(2411)
        order = p.source_order(source, 44, 1)
        self.assertEqual(sorted(order), source.tolist())
        self.assertTrue(np.array_equal(order, p.source_order(source, 44, 1)))
        self.assertFalse(np.array_equal(order, p.source_order(source, 44, 2)))
        batches = [order[i:i+256] for i in range(0, len(order), 256)]
        self.assertEqual(len(batches), 10)
        self.assertEqual(len(batches[-1]), 107)

    def test_heldout_excluded_and_ssl_labels_independent_of_species(self):
        cfg = copy.deepcopy(CFG); cfg["training"]["source_batch"] = 2
        frame = pd.DataFrame(dict(id=["a", "b", "c", "d"], fold=[0, 2, 1, 4], label=[3, 7, 9, 12]))
        with self.assertRaises(ValueError):
            p.SourceBatches(cfg, frame, np.arange(4), "idmm")
        a = p.SourceBatches(cfg, frame, np.array([0, 2, 3]), "idmm")
        b = p.SourceBatches(cfg, frame.assign(label=[29]*4), np.array([0, 2, 3]), "idmm")
        for data in (a, b):
            data.runtime = SimpleNamespace(dq=SimpleNamespace(read_rgb=lambda path: np.full((8, 8, 3), ord(path.name), np.uint8)))
            data.transforms = [Identity(), Identity()]
        first, second = a[(1, 0)], b[(1, 0)]
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(set(a.ids.values()), {0, 1, 2})
        self.assertNotIn(1, a.ids)
        self.assertEqual(first["images"].shape, (12, 3, 8, 8))
        np.testing.assert_array_equal(first["images"][:2], first["images"][2:4])

    def test_cutmix_lambda_uses_actual_clipped_area_and_complete_group(self):
        images = np.stack([np.full((3, 13, 17), i, np.uint8) for i in range(9)])
        labels = np.arange(9)
        for seed in range(12):
            mixed, a, b, lam = p.mix_batch(images, labels, seed, "cutmix", .5)
            np.testing.assert_array_equal(a, labels)
            self.assertEqual(sorted(b), labels.tolist())
            for i in np.flatnonzero(b != a):
                fraction = np.mean(mixed[i, 0] != images[i, 0])
                self.assertAlmostEqual(1-lam, fraction, places=12)
            repeat = p.mix_batch(images, labels, seed, "cutmix", .5)
            np.testing.assert_array_equal(mixed, repeat[0])

    def test_mixup_and_smoothed_loss_match_explicit_soft_targets(self):
        x = np.arange(48, dtype=np.uint8).reshape(4, 3, 2, 2)
        mixed, a, b, lam = p.mix_batch(x, np.arange(4), 37, "mixup", .5)
        np.testing.assert_allclose(mixed, lam*x+(1-lam)*x[b], rtol=1e-6)
        logits = torch.randn(4, 4)
        target = lam*F.one_hot(torch.tensor(a), 4)+(1-lam)*F.one_hot(torch.tensor(b), 4)
        target = .9*target + .1/4
        expected = -(target*logits.log_softmax(1)).sum(1).mean()
        actual = p.mixed_loss(logits, torch.tensor(a), torch.tensor(b), lam, .1)
        torch.testing.assert_close(actual, expected)

    def test_uneven_accumulation_matches_full_two_group_gradient(self):
        torch.manual_seed(9)
        full, micro = Tiny(11), Tiny(11)
        micro.load_state_dict(full.state_dict())
        groups = [(torch.randn(n, 5), torch.randint(11, (n,)), torch.randint(11, (n,))) for n in (10, 20)]
        full_loss = sum(p.mixed_loss(full(x), a, b, .37, .1) for x, a, b in groups)
        full_loss.backward()
        partial_loss = 0
        for x, a, b in groups:
            for start in range(0, len(x), 6):
                stop = min(start+6, len(x))
                value = p.mixed_loss(micro(x[start:stop]), a[start:stop], b[start:stop], .37, .1)*(stop-start)/len(x)
                partial_loss += value.detach(); value.backward()
        torch.testing.assert_close(full_loss, partial_loss)
        for a, b in zip(full.parameters(), micro.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-5)

    def test_instance_head_is_discarded_and_species_head_preserved(self):
        source, target = Tiny(2411), Tiny(30)
        head = {k: v.clone() for k, v in target.head.state_dict().items()}
        p.transfer_backbone(target, source.state_dict())
        for k, v in head.items():
            torch.testing.assert_close(target.head.state_dict()[k], v)
        torch.testing.assert_close(target.body.weight, source.body.weight)
        invalid = source.state_dict(); del invalid["body.weight"]
        with self.assertRaises(ValueError):
            p.transfer_backbone(target, invalid)

    def test_factory_explicitly_forbids_external_weights(self):
        with patch.object(p.timm, "create_model") as factory:
            p.make_model(CFG, "pvt_v2_b3", 2411)
            self.assertIs(factory.call_args.kwargs["pretrained"], False)
            self.assertEqual(factory.call_args.kwargs["num_classes"], 2411)
        cfg = copy.deepcopy(CFG); cfg["training"]["pretrained"] = True
        with self.assertRaises(ValueError):
            p.make_model(cfg, "pvt_v2_b3")

    def test_checkpoint_rejects_other_spec_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out / "spec.json").write_text(json.dumps({"test": 1}))
            path = out / "checkpoint.pt"
            torch.save(dict(spec_sha256="different", architecture="pvt_v2_b3", phase="idmm"), path)
            with self.assertRaises(ValueError):
                p.restore({}, path, out, "pvt_v2_b3", "idmm")

    def test_screening_requires_both_primary_gains_and_clean_blur_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out / "A").mkdir()
            baseline = dict(primary=.83, clean=.87, mean_blur=.62)
            (out / "A/history.json").write_text(json.dumps([dict(epoch=800, validation=baseline)]))
            metrics = dict(control=dict(primary=.84), candidate=dict(primary=.85, clean=.87, mean_blur=.64))
            self.assertTrue(p.screening(metrics, CFG, out)["passed"])
            for key, value in (("primary", .841), ("clean", .85), ("mean_blur", .60)):
                bad = copy.deepcopy(metrics); bad["candidate"][key] = value
                self.assertFalse(p.screening(bad, CFG, out)["passed"])

    def test_scheduler_has_exact_stage_update_budget_and_warmup(self):
        cfg = copy.deepcopy(CFG); cfg["training"]["learning_rate"] = .0005
        model = Tiny(30)
        optimizer, schedule = p.common.optimizer_and_scheduler(model, cfg, 10)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], .0005/100)
        for _ in range(100):
            optimizer.step(); schedule.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], .0005)
        for _ in range(7900):
            optimizer.step(); schedule.step()
        self.assertEqual(schedule.last_epoch, 8000)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], .000001)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
