import contextlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from mixture_scaling import interleaved_mlp_pairs as m
from mixture_scaling.ranking_preservation import rows

class RankingTests(unittest.TestCase):
 def test_rank_loss_zero_at_teacher_shift_invariant_and_no_teacher_gradient(self):
  teacher=torch.tensor([2.,-1.,0.,1.,3.,-2.],requires_grad=True)
  student=teacher.detach().clone().requires_grad_()
  loss=m.ranking_distillation(student,teacher,1);self.assertAlmostEqual(float(loss),0.,places=6)
  loss.backward();self.assertTrue(torch.allclose(student.grad,torch.zeros_like(student),atol=1e-7));self.assertIsNone(teacher.grad)
  changed=(-teacher.detach()).requires_grad_();loss=m.ranking_distillation(changed,teacher,1)
  self.assertGreater(float(loss),0);self.assertAlmostEqual(float(loss),float(m.ranking_distillation(changed+5,teacher,1)),places=5)
  loss.backward();self.assertGreater(float(changed.grad.abs().sum()),0)
 def test_plan_has_matched_controls(self):
  self.assertEqual(len(rows()),4)
  for a,b in {(a,b) for _,a,b,w in rows()}:self.assertEqual({w for _,x,y,w in rows() if (x,y)==(a,b)},{0,1})
 def test_warm_start_preserves_optimizer_and_step_zero_candidate(self):
  for weight in (0,1):
   with tempfile.TemporaryDirectory() as tmp:
    model=m.BiasMLP('node_neighbors');opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=1e-5)
    for param in model.parameters():param.grad=torch.ones_like(param)
    opt.step();cp=Path(tmp)/'source.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),metadata=dict(sources=['A'],seed=0)),cp)
    def load(*args):return dict(x=torch.randn(4,1536),positive=torch.tensor([[0,1],[1,2]]),validation=torch.tensor([[0],[1]]),sampler=m.lp.ExactNonedges(4,torch.tensor([1,6])),receipt={})
    logged=[]
    @contextlib.contextmanager
    def tracking(*args):yield SimpleNamespace(summary={}),lambda step,data:logged.append((step,data))
    args=SimpleNamespace(root=tmp,seed=0,max_steps=4,validation_interval=2,patience=3,log_interval=2,warm_start=str(cp),rank_weight=weight)
    reports=[dict(bce=v,roc_auc=.8) for v in [.2,.2,.3,.3,.4,.4]]
    with patch.object(m,'load_graph',side_effect=load),patch.object(m,'tracked_run',side_effect=tracking),patch.object(m,'validate',side_effect=reports):m.train_one(('pair','A','B'),{'protocol':{}},args,torch.device('cpu'))
    p=Path(tmp)/'node_neighbors/lp/pair';best=torch.load(p/'best.pt',weights_only=False);last=torch.load(p/'latest.pt',weights_only=False)
    self.assertEqual(best['step'],0)
    for name,tensor in model.state_dict().items():self.assertTrue(torch.equal(tensor,best['model'][name]))
    for state in last['optimizer']['state'].values():self.assertEqual(float(state['step']),5.)
    self.assertEqual(json.loads((p/'summary.json').read_text())['updates_per_source'],{'A':2,'B':2})
    self.assertEqual(any('train/ranking_kl' in x for _,x in logged),weight==1)
if __name__=='__main__':unittest.main()
