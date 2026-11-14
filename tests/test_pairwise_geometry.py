"""Geometric verification and fold-safe reference identity checks."""
import unittest

from scripts import screen_pairwise_geometry as geometry


class PairwiseGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg=geometry.yaml.safe_load((geometry.ROOT/'configs/LOCAL-PROP-005-DIAG.yaml').read_text())

    def test_shortlist_preserves_reference_identity_and_excludes_heldout(self):
        np=geometry.np
        source,queries=geometry.gallery_indices(np.array([1,0,1,2]),1)
        self.assertEqual(source.tolist(),[1,3]);self.assertEqual(queries.tolist(),[0,2])
        features=dict(clean=np.eye(2),corrupt=np.eye(2),query=np.eye(2))
        candidates=geometry.shortlist([features],source,1)
        self.assertEqual([x.tolist() for x in candidates],[[1],[3]])

    def test_duplicate_points_cannot_inflate_correspondence_count(self):
        np=geometry.np
        p=np.array([[2.,2.],[2.1,2.1],[8.,8.],[15.,15.]],dtype=np.float32)
        q=np.array([[3.,3.],[3.1,3.1],[3.,3.],[16.,16.]],dtype=np.float32)
        self.assertEqual(geometry.unique_correspondences(p,q,2).tolist(),[0,3])

    def test_geometric_positive_and_collinear_negative(self):
        np=geometry.np
        p=np.array([(x,y) for x in [20,50,80,110] for y in [20,50,80,110]],dtype=np.float32)
        q=p@np.array([[.9,.1],[-.1,.9]],np.float32)+5
        d=np.random.RandomState(44).normal(size=(16,128)).astype(np.float32)
        d/=np.linalg.norm(d,axis=1,keepdims=True)
        good=geometry.verify_pair(p,d,q,d,(150,150),(150,150),self.cfg['matching'],44)
        self.assertTrue(good['accepted']);self.assertEqual(good['inliers'],16)
        line=np.column_stack((np.arange(16),np.arange(16))).astype(np.float32)
        bad=geometry.verify_pair(line,d,line,d,(150,150),(150,150),self.cfg['matching'],44)
        self.assertFalse(bad['accepted'])

    def test_same_class_runner_up_does_not_hide_other_class_conflict(self):
        np=geometry.np
        def item(index,inliers):return dict(reference_index=index,inliers=inliers,median_error=.1,accepted=True)
        labels=np.array([0,0,1])
        self.assertIsNone(geometry.choose_match([item(0,15),item(1,14),item(2,12)],labels,1.5))
        result=geometry.choose_match([item(0,18),item(1,17),item(2,10)],labels,1.5)
        self.assertEqual(result['label'],0);self.assertEqual(result['second_class_inliers'],10)


if __name__=='__main__':unittest.main()
