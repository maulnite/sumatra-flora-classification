"""Fold safety, pseudo-label gradients, confidence gates and stateless data draws."""
import copy
import unittest
from pathlib import Path
import tempfile

from scripts import adaptive_self_training as ssl


class AdaptiveSelfTrainingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ssl.yaml.safe_load((ssl.ROOT / "configs/LOCAL-PROP-004-SSL.yaml").read_text())

    def test_heldout_group_cannot_leak_into_source(self):
        frame = ssl.pd.DataFrame(dict(id=["a", "b"], label=[0, 0], fold=[0, 1], group_id=[5, 5]))
        with self.assertRaises(ValueError):
            ssl.source_indices(frame, 0)
        with self.assertRaises(ValueError):
            ssl.PairedBatches(self.cfg, frame, [0], 1)
        frame.group_id = [4, 5]
        source, validation = ssl.source_indices(frame, 0)
        self.assertEqual(source.tolist(), [1]); self.assertEqual(validation.tolist(), [0])

    def test_confidence_is_relative_and_early_anchor_is_required(self):
        t = ssl.torch
        cfg = copy.deepcopy(self.cfg["adaptive"])
        # Disable marginal changes in this numerical fixture to isolate gates.
        cfg["distribution_alignment_ratio_clip"] = [1., 1.]
        stats = ssl.AdaptiveLabels(cfg)
        source = t.tensor([[.8, .2], [.2, .8]], requires_grad=True)
        target = t.tensor([[.75, .25], [.3, .7]], requires_grad=True)
        labels, mask, aligned, threshold = stats.select(source, target, 1, t.tensor([0, 1]), t.tensor([True, True]))
        self.assertAlmostEqual(threshold, .72, places=6)
        self.assertEqual(mask.tolist(), [True, False])
        self.assertFalse(aligned.requires_grad)
        _, mask, _, _ = stats.select(source, target, 1, t.tensor([1, 1]), t.tensor([True, True]))
        self.assertEqual(mask.tolist(), [False, False])
        _, mask, _, _ = stats.select(source, target, 11)
        self.assertEqual(mask.tolist(), [True, False])

    def test_alignment_normalizes_and_checkpoint_statistics_roundtrip(self):
        t = ssl.torch
        stats = ssl.AdaptiveLabels(self.cfg["adaptive"])
        source, target = t.tensor([[.9, .1], [.8, .2]]), t.tensor([[.3, .7], [.2, .8]])
        _, _, aligned, _ = stats.select(source, target, 11)
        self.assertTrue(t.allclose(aligned.sum(1), t.ones(2)))
        self.assertGreater(float(aligned[0, 0]), float(target[0, 0]))
        other = ssl.AdaptiveLabels(self.cfg["adaptive"]); other.load_state_dict(stats.state_dict())
        left, right = stats.select(source, target, 12), other.select(source, target, 12)
        self.assertTrue(all(t.equal(a, b) for a, b in zip(left[:3], right[:3])))

    def test_loss_averages_all_examples_and_empty_mask_has_zero_gradient(self):
        t = ssl.torch
        logits = t.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        labels = t.tensor([0, 1]); mask = t.tensor([True, False])
        loss = ssl.masked_target_loss(logits, labels, mask)
        self.assertTrue(t.allclose(loss, ssl.F.cross_entropy(logits[:1], labels[:1])/2))
        loss.backward(); self.assertTrue(t.equal(logits.grad[1], t.zeros(2)))
        logits.grad.zero_()
        empty = ssl.masked_target_loss(logits, labels, t.tensor([False, False])); empty.backward()
        self.assertEqual(float(empty.detach()), 0.); self.assertTrue(t.equal(logits.grad, t.zeros_like(logits)))

    def test_augmentation_repeatability_and_control_has_no_target_loss(self):
        np = ssl.np
        image = np.random.RandomState(3).randint(256, size=(32, 32, 3), dtype=np.uint8)
        original = image.copy()
        a, b = ssl.target_views(image, 123, self.cfg["training"]), ssl.target_views(image, 123, self.cfg["training"])
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(ssl.target_weight(self.cfg["adaptive"], 100, 0, 296, False), 0.)
        self.assertEqual(ssl.target_weight(self.cfg["adaptive"], 1, 0, 296, True), 0.)
        self.assertEqual(ssl.target_weight(self.cfg["adaptive"], 21, 0, 296, True), 1.)

    def test_checkpoint_roundtrip_restores_all_cpu_random_streams(self):
        ssl.seed_all(44); state = ssl.rng_state()
        expected = (ssl.random.random(), ssl.np.random.rand(), ssl.torch.rand(3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"; ssl.save_checkpoint(path, state)
            restored = ssl.torch.load(path, weights_only=False)
        ssl.restore_rng(restored)
        actual = (ssl.random.random(), ssl.np.random.rand(), ssl.torch.rand(3))
        self.assertEqual(expected[:2], actual[:2]); self.assertTrue(ssl.torch.equal(expected[2], actual[2]))


if __name__ == "__main__":
    unittest.main()
