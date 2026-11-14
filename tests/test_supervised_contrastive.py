"""Full-batch positive masks, class-group sampling and paired control gradients."""
import unittest

from scripts import supervised_contrastive as sup


class SupervisedContrastiveTests(unittest.TestCase):
    def setUp(self):
        self.cfg=sup.yaml.safe_load((sup.ROOT/'configs/LOCAL-PROP-006-SUPCON.yaml').read_text())

    def test_cross_image_same_class_positives_reduce_loss(self):
        t=sup.torch
        labels=t.tensor([0,0,1,1])
        aligned=t.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],requires_grad=True)
        mixed=t.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],requires_grad=True)
        low=sup.supcon_loss(aligned,labels,.1);high=sup.supcon_loss(mixed,labels,.1)
        self.assertLess(float(low.detach()),float(high.detach()))
        low.backward();self.assertTrue(t.isfinite(aligned.grad).all())
        order=t.tensor([2,0,3,1])
        self.assertTrue(t.allclose(high,sup.supcon_loss(mixed[order],labels[order],.1)))

    def test_each_anchor_needs_nonself_positive(self):
        t=sup.torch
        with self.assertRaises(ValueError):sup.supcon_loss(t.eye(3),t.arange(3),.1)
        with self.assertRaises(ValueError):sup.supcon_loss(t.eye(3),t.tensor([0,0,0]),0.)

    def test_zero_contrastive_weight_preserves_ce_gradient(self):
        t=sup.torch;t.manual_seed(44)
        x=t.randn(4,5,requires_grad=True);labels=t.tensor([0,0,1,1])
        ce=sup.F.cross_entropy(x[:,:2],labels)
        first=t.autograd.grad(ce,x,retain_graph=True)[0]
        second=t.autograd.grad(ce+0*sup.supcon_loss(x,labels,.1),x)[0]
        self.assertTrue(t.equal(first,second))

    def test_class_balanced_draws_span_scenes_and_never_select_holdout(self):
        rows=[]
        for c in range(30):
            for photo in range(4):rows.append(dict(label=c,fold=2 if photo==3 else 0,group_id=c*10+photo))
        frame=sup.pd.DataFrame(rows);source=sup.np.flatnonzero(frame.fold.to_numpy()!=2)
        left=sup.batch_indices(frame,source,self.cfg,1,0);right=sup.batch_indices(frame,source,self.cfg,1,0)
        self.assertTrue(sup.np.array_equal(left,right));self.assertEqual(len(set(left)),60)
        selected=frame.iloc[left]
        self.assertEqual(selected.groupby('label').size().tolist(),[2]*30)
        self.assertEqual(selected.groupby('label').group_id.nunique().tolist(),[2]*30)
        self.assertTrue((selected.fold!=2).all())
        with self.assertRaises(ValueError):sup.SourceBatches(self.cfg,frame,sup.np.arange(len(frame)))


if __name__=='__main__':unittest.main()
