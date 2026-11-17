import unittest
import numpy as np
import pandas as pd
from scripts.prepare_deadline_blend import evidence, fuse


class BlendTests(unittest.TestCase):
    def test_fixed_geometric_endpoints_and_normalization_without_mutation(self):
        first = np.array([[.8, .2], [.4, .6]], dtype=np.float32)
        second = first[:, ::-1].copy(); original = first.copy()
        np.testing.assert_array_equal(fuse(first, second, 0), first)
        np.testing.assert_array_equal(fuse(first, second, 1), second)
        np.testing.assert_allclose(fuse(first, second, .1).sum(1), 1, atol=1e-6)
        np.testing.assert_array_equal(first, original)
        with self.assertRaises(ValueError): fuse(first, second, 1.1)

    def test_old_sift_only_sets_minimum_evidence_for_accepted_rows(self):
        matrix = np.zeros((3, 30), dtype=np.float32); matrix[0, 2] = 12
        old = pd.DataFrame(dict(label=[2, 4, 5], inlier=[5, 4, 10]))
        actual = evidence(matrix, old)
        self.assertEqual(actual[0, 2], 12)
        self.assertEqual(actual[1].max(), 0)
        self.assertEqual(actual[2, 5], 4)
        self.assertEqual(matrix[2, 5], 0)


if __name__ == "__main__": unittest.main()
