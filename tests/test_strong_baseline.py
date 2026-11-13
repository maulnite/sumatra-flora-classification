"""Risk checks for strong-baseline alignment and leakage filtering."""
import unittest

try:
    from scripts import evaluate_strong_baseline as strong
except ModuleNotFoundError:
    strong = None


@unittest.skipIf(strong is None, 'Use the project .venv')
class StrongBaselineTests(unittest.TestCase):
    def test_training_scene_neighbors_are_excluded(self):
        pd = strong.pd
        train = pd.DataFrame(dict(id=['a','b','c','d'], label=[0,0,1,1]))
        folds = train.assign(group_id=[0,0,1,2],fold=[0,0,1,2])
        local = train.assign(role=['train','validation','validation','purged_scene_neighbor'])
        self.assertEqual(strong.shared_indices(train, folds, local).tolist(), [2])
        with self.assertRaises(ValueError):
            strong.shared_indices(train, folds.iloc[::-1], local)

    def test_sift_rejects_same_fold_reference_even_with_many_inliers(self):
        np,pd = strong.np,strong.pd
        matches=pd.DataFrame(dict(idx=[1,2,-1],inlier=[100,20,0],label=[1,2,-1]))
        labels=np.array([9,9,9])
        result,valid=strong.sift_labels(labels,matches,5,np.array([0,1,2]),np.array([0,0,1]))
        self.assertEqual(result.tolist(),[9,2,9])
        self.assertEqual(valid.tolist(),[False,True,False])

    def test_zero_weight_recovers_baseline_and_blending_keeps_rows_normalized(self):
        np=strong.np
        a=np.array([[.8,.2],[.3,.7]],dtype=np.float32)
        b=np.array([[.2,.8],[.6,.4]],dtype=np.float32)
        self.assertTrue(np.array_equal(strong.blend(a,b,0),a))
        self.assertTrue(np.allclose(strong.blend(a,b,.1).sum(1),1))
        with self.assertRaises(ValueError):
            strong.blend(a,b*2,.1)


if __name__=='__main__':
    unittest.main()
