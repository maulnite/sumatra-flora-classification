"""Checks of A/B isolation and the conditional remote-write gate."""
import unittest
from unittest.mock import patch

try:
    from scripts import run_local_experiment as run
except ModuleNotFoundError:
    run = None


@unittest.skipIf(run is None, "Use the project's .venv for training dependencies")
class LocalExperimentTests(unittest.TestCase):
    def test_kaggle_version_warning_does_not_hide_submission_history(self):
        payload = 'Warning: version update available\n[{"date":"2026-11-13","ref":12}]\n'
        self.assertEqual(run.parse_cli_json(payload), [{"date": "2026-11-13", "ref": 12}])
        with self.assertRaises(ValueError):
            run.parse_cli_json('Unexpected API failure')

    def test_additional_blur_preserves_global_rng_and_clean_views(self):
        np = run.np
        image = np.random.RandomState(8).randint(0, 256, (256, 256, 3), dtype=np.uint8)
        rows = [{"id": "fake.jpg", "label": "0"}]
        a = run.TrainingViews(rows, ".", 0.)
        b = run.TrainingViews(rows, ".", 1.)
        with patch.object(run.dq, "read_rgb", return_value=image):
            clean_a = a[0, 2][0]
            clean_b = b[0, 2][0]
            self.assertTrue(np.array_equal(clean_a, clean_b))
            corrupted_a = a[1, 2][0]
            state_a = np.random.get_state()
            corrupted_b = b[1, 2][0]
            state_b = np.random.get_state()
            expected = run.post_blur(corrupted_a.transpose(1, 2, 0),
                       run.stable_seed(42, 2, "fake.jpg", "corrupt", "post-blur"), 1.)
            self.assertTrue(np.array_equal(expected.transpose(2, 0, 1), corrupted_b))
            self.assertTrue(np.array_equal(state_a[1], state_b[1]))
            self.assertEqual(state_a[2:], state_b[2:])
            self.assertTrue(np.array_equal(corrupted_a, a[1, 2][0]))

    def test_sampler_no_replacement_and_repeatable_epoch_order(self):
        a, b = run.EpochSampler(20), run.EpochSampler(20)
        for epoch in (1, 2, 300):
            a.epoch = b.epoch = epoch
            self.assertEqual(list(a), list(b))
            self.assertEqual(len({i for i, ep in a}), 20)
            self.assertTrue(all(ep == epoch for i, ep in a))

    def test_submission_requires_primary_gain_and_robustness_guard(self):
        criteria = dict(minimum_blur_metric_gain_B_minus_A=.02,
                        maximum_v4rot_macro_f1_drop=.005, maximum_clean_macro_f1_drop=.01)
        baseline = dict(v4rot=.7, clean=.8, v4rot_blur_06=.6, v4rot_blur_10=.5)
        candidate = dict(v4rot=.71, clean=.8, v4rot_blur_06=.63, v4rot_blur_10=.53)
        self.assertTrue(run.decision(baseline, candidate, criteria)["eligible_for_submission"])
        for key, bad in (("v4rot", .7), ("clean", .78), ("v4rot_blur_10", .49)):
            with self.subTest(key=key):
                self.assertFalse(run.decision(baseline, {**candidate, key: bad}, criteria)["eligible_for_submission"])


if __name__ == "__main__":
    unittest.main()
