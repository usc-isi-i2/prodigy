import unittest
import numpy as np
import torch
from core import build_cache,neighbor_sets,Model,child_weights,CN_CAP,RES_CAP

class TestCore(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.edges=np.array([[0,2],[1,2],[0,3],[1,4],[3,4],[1,5]])
        self.pairs=np.array([[0,1],[1,0],[0,0],[0,6],[3,5]])
        self.cache=build_cache(self.edges,self.pairs,7)

    def tensors(self):
        return {k:torch.as_tensor(v,dtype=torch.float32 if k in ('scale','candidate_scale','child_scale') else torch.long) for k,v in self.cache.items()}

    def test_membership_and_self_exclusion(self):
        adj=neighbor_sets(self.edges,7)
        for i,(u,v) in enumerate(self.pairs):
            real=self.cache['cn'][i];real=real[real<7]
            self.assertEqual(set(real),set() if u==v else adj[u]&adj[v])
            if u==v:
                self.assertEqual(float(self.cache['candidate_scale'][i].sum()),0)
                continue
            for side,(a,b) in enumerate(((u,v),(v,u))):
                sl=slice(side*RES_CAP,(side+1)*RES_CAP)
                candidates=self.cache['candidates'][i,sl];chosen=candidates[candidates<7]
                self.assertEqual(set(chosen),adj[a]-adj[b]-{u,v})
                for slot in range(side*RES_CAP,side*RES_CAP+len(chosen)):
                    edge=self.cache['child_pairs'][self.cache['child_index'][i,slot]]
                    self.assertNotEqual(edge[0],edge[1]);self.assertNotIn(int(edge[1]),adj[int(edge[0])])
                    self.assertEqual(set(edge),{int(b),int(self.cache['candidates'][i,slot])})

    def test_pooling_symmetry_zero_completion_and_gradients(self):
        torch.manual_seed(1);m=Model();c=self.tensors();emb=torch.randn(7,256,requires_grad=True)
        projected=torch.cat([m.project(emb),torch.zeros(1,64)])
        z=(emb,projected);ids=torch.arange(5)
        observed=m.score_rows(z,c,ids);weights=child_weights(m,z,c)
        self.assertFalse(weights.requires_grad)
        completed=m.score_rows(z,c,ids,weights)
        torch.testing.assert_close(observed,m.score_rows(z,c,ids,torch.zeros_like(weights)),rtol=0,atol=0)
        torch.testing.assert_close(observed[0],observed[1],rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(completed[0],completed[1],rtol=1e-6,atol=1e-6)
        self.assertEqual(float(completed[2]),-1e9)
        completed[[0,3,4]].sum().backward();self.assertTrue(torch.isfinite(emb.grad).all())
        self.assertGreater(float(m.project[0].weight.grad.abs().sum()),0)

    def test_caps_preserve_population_mass(self):
        edges=np.array([[endpoint,node] for endpoint in [0,1] for node in range(2,50)]+[[0,node] for node in range(50,90)])
        c=build_cache(edges,np.array([[0,1],[1,0]]),91)
        self.assertEqual(int((c['cn'][0]<91).sum()),CN_CAP)
        self.assertAlmostEqual(float(c['scale'][0])*CN_CAP,48)
        self.assertAlmostEqual(float(c['candidate_scale'][0].sum()),40)
        np.testing.assert_array_equal(c['cn'][0],c['cn'][1])
        np.testing.assert_array_equal(c['candidates'][0,:RES_CAP],c['candidates'][1,RES_CAP:])

    def test_encoder_initialization_matches_control(self):
        from core import j
        torch.manual_seed(2);a=j.Model('joint');torch.manual_seed(2);b=Model()
        for k,v in a.encoder.state_dict().items():torch.testing.assert_close(v,b.encoder.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in b.parameters()),525633)

if __name__=='__main__':unittest.main()
