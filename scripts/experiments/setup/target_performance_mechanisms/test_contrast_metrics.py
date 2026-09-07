import unittest
import numpy as np
from .contrast_metrics import ranking_metrics,contrast_parts,contrast_swaps


class ContrastMetricsTest(unittest.TestCase):
    def test_positive_episode_scaling_changes_pool_not_within_or_decisions(self):
        y=np.array([0,1,0,1]);ids=np.array([0,0,1,1]);mapping=np.tile([0,1],(4,1))
        base=np.array([1.,2.,10.,20.]);changed=base*np.array([10.,10.,1.,1.])
        a,_=ranking_metrics(base,y,mapping,ids,True);b,_=ranking_metrics(changed,y,mapping,ids,True)
        self.assertEqual(a['within_episode_auc'],1.);self.assertEqual(b['within_episode_auc'],1.)
        self.assertNotEqual(a['pooled_margin_auc'],b['pooled_margin_auc'])
        self.assertTrue(np.array_equal(base>0,changed>0))
        self.assertAlmostEqual(a['pooled_margin_auc'],a['within_pair_fraction']*a['within_pair_weighted_auc']+(1-a['within_pair_fraction'])*a['cross_episode_pair_auc'])

    def test_mapping_and_undefined_episode(self):
        y=np.array([0,1,0]);m=np.array([-1.,1.,1.]);ids=np.array([0,0,1]);maps=np.array([[1,0],[1,0],[0,1]])
        a,rows=ranking_metrics(m,y,maps,ids,True)
        self.assertEqual(a['defined_episodes'],1);self.assertEqual(a['undefined_episodes'],1)
        self.assertEqual(rows[0]['auc'],1.);self.assertIsNone(rows[1]['auc'])

    def test_final_contrast_and_paired_swaps(self):
        rng=np.random.default_rng(17);q=rng.normal(size=(4,3));a=rng.normal(size=(1,2,3)).repeat(4,0);b=rng.normal(size=(1,2,3)).repeat(4,0)
        x=contrast_parts(q,a,7.);z=contrast_parts(q,b,7.);scores=contrast_swaps(x,z,7.)
        norm=lambda t:t/np.linalg.norm(t,axis=-1,keepdims=True)
        exact=7.*np.sum(norm(q)*(norm(a)[:,1]-norm(a)[:,0]),axis=1)
        np.testing.assert_allclose(scores['intact'],exact,atol=1e-14)
        np.testing.assert_allclose(scores['orientation_only']/scores['removed'],x['strength']/z['strength'])
        np.testing.assert_allclose(scores['strength_only']/scores['intact'],z['strength']/x['strength'])
        np.testing.assert_allclose(x['strength']**2,2.-2.*x['class_cosine'],atol=1e-14)


if __name__=='__main__':unittest.main()
