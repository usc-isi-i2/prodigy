import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from mixture_scaling import mini_lp as lp
from mixture_scaling.bias_lp_matrix import BiasMLP
from mixture_scaling.sequential_mlp_pairs import pair_rows

class SequentialTests(unittest.TestCase):
    def test_all_ordered_pairs(self):
        rows=pair_rows()
        self.assertEqual(len(rows),72)
        self.assertEqual(len({r[0] for r in rows}),72)
        pairs={(a,b) for _,a,b in rows}
        self.assertTrue(all(a!=b and (b,a) in pairs for a,b in pairs))

    def test_continuation_loads_bias_and_weights_and_records_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'first.pt';model=BiasMLP('node_neighbors')
            with torch.no_grad():model.decoder_bias.fill_(-3.25)
            torch.save(dict(model=model.state_dict(),step=4000,metadata=dict(sources=['A'])),cp)
            edges=torch.tensor([[0,1],[1,2]])
            data=dict(x=torch.randn(4,1536),known_keys=torch.tensor([1,6]),receipt={})
            data.update({k:edges for k in ['train','context','supervision','validation','test']})
            args=SimpleNamespace(root=tmp,run_id='A_then_B',view='node_neighbors',warm_start=str(cp),seed=0,max_steps=1,validation_interval=1,patience=3,log_interval=1)
            captured=[]
            def make_model(view,device):
                m=BiasMLP(view);captured.append(m);return m
            def scoring(m,x,pairs):
                self.assertTrue(torch.equal(m.network[0].weight,model.network[0].weight))
                self.assertEqual(float(m.decoder_bias),-3.25)
                return lp_original_score(m,x,pairs)+m.decoder_bias
            lp_original_score=lp.score
            @contextlib.contextmanager
            def tracking(*a):yield SimpleNamespace(summary={}),lambda *a:None
            with patch.object(lp,'prepare',return_value=data),patch.object(lp,'view_features',return_value=data['x']),patch.object(lp,'model_for',side_effect=make_model),patch.object(lp,'score',side_effect=scoring),patch.object(lp,'validate',return_value={'bce':.5,'roc_auc':.6}),patch.object(lp,'tracked_run',side_effect=tracking):
                lp.train('B',{'protocol':{}},args,torch.device('cpu'))
            result=torch.load(Path(tmp)/'node_neighbors/lp/A_then_B/best.pt',weights_only=False)
            self.assertEqual(result['metadata']['sources'],['A','B'])
            self.assertEqual(result['metadata']['warm_start']['step'],4000)
            self.assertEqual(result['metadata']['stage'],2)
            self.assertNotEqual(float(result['model']['decoder_bias']),-3.25)

if __name__=='__main__':unittest.main()
