"""Protect probability weights and final-checkpoint advancement for public probes."""
import unittest
import numpy as np
from scripts.prepare_pvt_public_probe import replace_member


class PublicProbeTests(unittest.TestCase):
    def test_exact_member_replacement_keeps_nine_others_and_does_not_mutate(self):
        rng = np.random.RandomState(44)
        values = [rng.dirichlet([1, 1, 1], 17).astype(np.float32) for _ in range(11)]
        families = [values[:5], values[5:10]]
        originals = [[v.copy() for v in family] for family in families]
        baseline = np.mean([np.mean(f, axis=0) for f in families], axis=0)
        same = replace_member(families, families[0][2], 2)
        np.testing.assert_array_equal(same, baseline)
        actual = replace_member(families, values[10], 2)
        expected = baseline + .1*(values[10]-families[0][2])
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        for family, original in zip(families, originals):
            for a, b in zip(family, original): np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(actual.sum(1), 1, atol=1e-6)

    def test_invalid_probability_or_family_is_rejected(self):
        a = np.full((17, 3), 1/3, dtype=np.float32)
        families = [[a]*5, [a]*5]
        for replacement in (a*2, a[:16], np.full_like(a, np.nan)):
            with self.assertRaises(ValueError): replace_member(families, replacement, 2)
        for fold in (-1, 5):
            with self.assertRaises(ValueError): replace_member(families, a, fold)
        with self.assertRaises(ValueError): replace_member([[a]*4, [a]*5], a, 2)


if __name__ == "__main__":
    unittest.main()
