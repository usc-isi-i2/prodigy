import contextlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from mixture_scaling import interleaved_mlp_pairs as m

class InterleavedTests(unittest.TestCase):
    def test_pair_plan_excludes_election_and_reverse_duplicates(self):
        rows=m.pair_rows();self.assertEqual(len(rows),28);self.assertEqual(len(m.SOURCES),8)
        self.assertEqual(len({frozenset((a,b)) for _,a,b in rows}),28)
        self.assertTrue(all(a!=b and 'election2020' not in (a,b) for _,a,b in rows))

    def test_alternation_shared_optimizer_and_joint_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded=[]
            def load(source,*args):
                g=dict(x=torch.randn(4,1536),positive=torch.tensor([[0,1],[1,2]]),
                       validation=torch.tensor([[0],[1]]),sampler=m.lp.ExactNonedges(4,torch.tensor([1,6])),receipt={})
                loaded.append(g);return g
            logged=[]
            @contextlib.contextmanager
            def tracking(*args):yield SimpleNamespace(summary={}),lambda step,data:logged.append((step,data))
            reports=[dict(bce=v,roc_auc=.8) for v in [.2,.8,.3,.4,.1,.9]]
            args=SimpleNamespace(root=tmp,seed=0,max_steps=6,validation_interval=2,patience=3,log_interval=2)
            original=m.score;seen=[]
            def scoring(model,x,pairs):
                seen.append(next(i for i,g in enumerate(loaded) if x is g['x']))
                return original(model,x,pairs)
            with patch.object(m,'load_graph',side_effect=load),patch.object(m,'tracked_run',side_effect=tracking),patch.object(m,'validate',side_effect=reports),patch.object(m,'score',side_effect=scoring):
                m.train_one(('pair','A','B'),{'protocol':{}},args,torch.device('cpu'))
            self.assertEqual(seen,[0,1,0,1,0,1])
            p=Path(tmp)/'node_neighbors/lp/pair';s=json.loads((p/'summary.json').read_text())
            self.assertEqual(s['updates_per_source'],{'A':3,'B':3})
            self.assertEqual(s['best_step'],4);self.assertAlmostEqual(s['best_validation_bce'],.35)
            self.assertEqual(s['stop_reason'],'safety_cap');self.assertFalse(s['converged'])
            ck=torch.load(p/'best.pt',weights_only=False)
            for state in ck['optimizer']['state'].values():self.assertEqual(float(state['step']),4)
            self.assertEqual(len([v for _,v in logged if 'validation/loss_A' in v]),3)
            args.max_steps=8
            with self.assertRaises(ValueError):m.train_one(('pair','A','B'),{'protocol':{}},args,torch.device('cpu'))

if __name__=='__main__':unittest.main()
