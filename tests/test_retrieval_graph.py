"""Graph pilot checks: gallery identity, clamped sources, query isolation."""
import unittest

try:
    from scripts import pilot_retrieval_graph as pilot
except ModuleNotFoundError:
    pilot = None


@unittest.skipIf(pilot is None, "Optional graph dependencies unavailable")
class RetrievalGraphTests(unittest.TestCase):
    def test_two_views_do_not_count_as_two_retrieval_identities(self):
        np = pilot.np
        gallery = pilot.unit(np.array([[1.,0.],[.99,.01],[.8,.2],[0.,1.]]))
        indices, _ = pilot.distinct_neighbors(np.array([[1.,0.]],dtype=np.float32), gallery, np.array([0,0,1,2]), 2)
        self.assertEqual(indices.tolist(), [[0,2]])

    def test_source_labels_stay_clamped_and_isolated_target_keeps_unary(self):
        np = pilot.np
        graph = pilot.sparse.csr_matrix(np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]]))
        unary = np.array([[.2,.8],[.3,.7],[.6,.4]])
        result, _ = pilot.diffuse(graph, unary, np.array([0]), dict(iterations=100,tolerance=1e-8,diffusion_alpha=.8))
        self.assertTrue(np.array_equal(result[0], [1.,0.]))
        self.assertTrue(np.array_equal(result[2], unary[2]))
        self.assertGreater(result[1,0],.8)

    def test_independent_query_order_and_other_queries_cannot_change_readout(self):
        np = pilot.np
        embedding = pilot.unit(np.array([[1.,0.],[0.,1.],[-1.,0.]]))
        query = pilot.unit(np.array([[.9,.1],[.1,.9]]))
        values=np.array([[1.,0.],[0.,1.],[1.,0.]])
        unary=np.array([[.4,.6],[.6,.4]])
        cfg=dict(neighbors=2,independent_query_minimum_similarity=0., independent_query_reciprocal_neighbors=True,
                 final_neural_graph_weight=.5,edge_temperature=.1)
        all_result,_=pilot.independent_readout(query,embedding,values,unary,np.zeros(3),cfg)
        one_result,_=pilot.independent_readout(query[:1],embedding,values,unary[:1],np.zeros(3),cfg)
        self.assertTrue(np.array_equal(all_result[:1],one_result))
        self.assertTrue(np.allclose(all_result.sum(1),1))


if __name__ == "__main__":
    unittest.main()
