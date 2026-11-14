"""Comparison checks for checkpoint-level reproduction, without GPU access."""
import unittest

try:
    from scripts import reproduce_checkpoint_inference as replay
except ModuleNotFoundError:
    replay = None


@unittest.skipIf(replay is None, "Reproduction dependencies unavailable")
class ReproductionComparisonTests(unittest.TestCase):
    def test_exact_arrays_and_small_rounding_are_distinguished(self):
        np = replay.np
        expected = np.array([[.8, .2], [.1, .9]], dtype=np.float32)
        exact = replay.compare_probabilities(expected.copy(), expected, 1e-5, 1e-4)
        self.assertTrue(exact['array_exactly_equal'])
        actual = expected.copy()
        actual[0] += np.array([1e-6, -1e-6], dtype=np.float32)
        small = replay.compare_probabilities(actual, expected, 1e-5, 1e-4)
        self.assertFalse(small['array_exactly_equal'])
        self.assertTrue(small['within_tolerance'])
        self.assertEqual(small['predicted_labels_different'], 0)

    def test_prediction_changes_are_not_hidden_by_metric_only_comparison(self):
        np = replay.np
        expected = np.array([[.8,.2],[.1,.9]],dtype=np.float32)
        changed = expected[::-1].copy()
        result = replay.compare_probabilities(changed,expected,1e-5,1e-4)
        self.assertFalse(result['within_tolerance'])
        self.assertEqual(result['predicted_labels_different'],2)
        with self.assertRaises(ValueError):
            replay.compare_probabilities(changed[:1],expected,1e-5,1e-4)
        changed[0,0] = np.nan
        with self.assertRaises(ValueError):
            replay.compare_probabilities(changed,expected,1e-5,1e-4)


if __name__=='__main__':
    unittest.main()
