import contextlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from mixture_scaling import interleaved_mlp_pairs as m
from mixture_scaling.mixed_batch_pilot import pair_rows

class MixedTests(unittest.TestCase):
    def test_three_diagnostic_pairs(self):
        rows=pair_rows();self.assertEqual(len(rows),3)
        self.assertTrue(all('election2020' not in r[1:] for r in rows))

    def test_both_sources_same_weights_one_step_and_exact_half_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            graphs=[]
            def load(*args):
                g=dict(x=torch.randn(4,1536),positive=torch.tensor([[0,1],[1,2]]),validation=torch.tensor([[0],[1]]),sampler=m.lp.ExactNonedges(4,torch.tensor([1,6])),receipt={})
                graphs.append(g);return g
            @contextlib.contextmanager
            def tracking(*args):yield SimpleNamespace(summary={}),lambda *args:None
            seen=[];weights=[];original=m.score
            def scoring(model,x,pairs):
                seen.append((next(i for i,g in enumerate(graphs) if g['x'] is x),pairs.shape[1]))
                weights.append(model.network[0].weight.detach().clone())
                return original(model,x,pairs)
            args=SimpleNamespace(root=tmp,seed=0,max_steps=2,validation_interval=2,log_interval=2,patience=3,batch_schedule='mixed')
            with patch.object(m,'load_graph',side_effect=load),patch.object(m,'tracked_run',side_effect=tracking),patch.object(m,'score',side_effect=scoring),patch.object(m,'validate',return_value=dict(bce=.5,roc_auc=.7)):
                m.train_one(('mixed','A','B'),{'protocol':{}},args,torch.device('cpu'))
            self.assertEqual(seen,[(0,3072),(1,3072),(0,3072),(1,3072)])
            self.assertTrue(torch.equal(weights[0],weights[1]));self.assertTrue(torch.equal(weights[2],weights[3]));self.assertFalse(torch.equal(weights[0],weights[2]))
            p=Path(tmp)/'node_neighbors/lp/mixed';ck=torch.load(p/'best.pt',weights_only=False)
            for state in ck['optimizer']['state'].values():self.assertEqual(float(state['step']),2.)
            self.assertEqual(ck['metadata']['run_protocol']['schedule'],'balanced_mixed_batches')

if __name__=='__main__':unittest.main()
