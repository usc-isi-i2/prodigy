import random
import unittest
import numpy as np
import torch
from torch_geometric.data import Data
from experiments.sampler import NeighborSampler
from experiments.nm_campaign import CampaignNeighborTask, prepare_model_dataset, materialize
from data.dataset import SubgraphDataset

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        edges=[]
        for source in range(3):
            for a in range(40):
                for b in range(a+1,40):
                    edges.append((source*40+a, source*40+b))
        graph=Data(x=torch.randn(120,768),edge_index=torch.tensor(edges).T,graph_id=torch.arange(3).repeat_interleave(40),num_nodes=120)
        graph.source_graph_names=['a','b','twibot20']
        cls.sampler=NeighborSampler(graph,2,hop_sizes=[9,9],limit=101,walk_hops=1)
        cls.dataset=SubgraphDataset(graph,cls.sampler,bidirectional=False)
        cls.dataset.source_node_pools={i:torch.arange(i*40,(i+1)*40) for i in range(3)}

    def task(self, flags):
        return CampaignNeighborTask(self.sampler,120,'inout',strata=[np.arange(40),np.arange(40,80)],
            confine_to_single_stratum=True,stratum_weighting='balanced',filter_min_degree=True,
            flags=flags,adaptive_weights=torch.ones(16))

    def test_every_sampler_excludes_holdout(self):
        for flags in ['', 'proportional','blocked','cross_graph','degree_balanced','uniform_positive','degree_hard','region_adaptive','coverage_cycle']:
            task=self.task(flags);rng=random.Random(7)
            for _ in range(10):
                ep=task.sample(10,7,3,4,rng)
                self.assertEqual(len(ep),10)
                for center,members in ep.items():
                    self.assertLess(center,80)
                    self.assertEqual(len(set(members)),7)
                    self.assertTrue(all(n<80 for n in members))
                    self.assertTrue(all(n//40==center//40 for n in members))

    def test_mixed_draw_uses_both_sources(self):
        ep=self.task('cross_graph').sample(30,7,3,4,random.Random(3))
        self.assertEqual({c//40 for c in ep},{0,1})

    def test_block_length(self):
        task=self.task('blocked');rng=random.Random(9)
        seen=[next(iter(task.sample(3,7,3,4,rng)))//40 for _ in range(130)]
        self.assertEqual(seen[:64],[0]*64);self.assertEqual(seen[64:128],[1]*64)

    def test_holdout_guard(self):
        with self.assertRaises(ValueError):
            prepare_model_dataset(self.dataset,dict(neighbor_sampling_source_subset='a,twibot20',campaign_holdout='twibot20'))

if __name__=='__main__': unittest.main()
