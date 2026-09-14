import contextlib,copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from mixture_scaling import async_convergence as m

def graph(n_positive=7):
    return dict(x=torch.randn(7,1536,generator=torch.Generator().manual_seed(42)),positive=torch.tensor([[0,1,2],[1,2,3]]).repeat(1,(n_positive+2)//3)[:,:n_positive],
                validation=torch.tensor([[0,1],[1,2]]),sampler=m.base.lp.ExactNonedges(7,torch.tensor([1,9,17])),receipt={})

class AsyncTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_exact_replay_crosses_partial_batch_and_shuffle(self):
    graphs=[graph(2051),graph(1030)];m.initialize_sampling(graphs,0,torch.device('cpu'))
    model=m.base.BiasMLP('node_neighbors');opt=torch.optim.AdamW(model.parameters(),lr=.0005)
    runtime=m.empty_runtime()
    def step():
        i=runtime['logical_step']%2;pairs,y,n=m.next_batch(graphs[i])
        opt.zero_grad();loss=m.task_loss(m.base.score(model,graphs[i]['x'],pairs),y);loss.backward();opt.step()
        runtime['logical_step']+=1
        return pairs.clone(),y.clone(),n
    step();step()
    ck=m.snapshot(model,opt,graphs,runtime,{})
    frozen=copy.deepcopy(ck)
    first=[step() for _ in range(6)]
    final=m.snapshot(model,opt,graphs,runtime,{})
    runtime=m.restore(ck,model,opt,graphs)
    second=[step() for _ in range(6)]
    self.assertTrue(any(batch[2]<1024 for batch in first))
    for a,b in zip(first,second):
        self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[1],b[1]))
    for k,t in final['model'].items():self.assertTrue(torch.equal(t,model.state_dict()[k]))
    for k,state in final['optimizer']['state'].items():
        for key,value in state.items():self.assertTrue(torch.equal(value,opt.state_dict()['state'][k][key]))
    for k,t in ck['model'].items():self.assertTrue(torch.equal(t,frozen['model'][k]))

 def test_kd_replaces_hard_loss_and_teacher_detached(self):
    logits=torch.tensor([.2,-.3],requires_grad=True);teacher=torch.tensor([1.,2.],requires_grad=True)
    loss=m.task_loss(logits,torch.tensor([0.,1.]),teacher)
    other=m.task_loss(logits,torch.tensor([1.,0.]),teacher)
    self.assertTrue(torch.equal(loss,other));loss.backward()
    self.assertTrue(torch.allclose(logits.grad,(logits.detach().sigmoid()-teacher.detach().sigmoid())/2))
    self.assertIsNone(teacher.grad)
    self.assertEqual(float(m.task_loss(teacher.detach(),torch.zeros(2),teacher)-m.task_loss(teacher.detach(),torch.ones(2),teacher)),0)

 def test_measurement_does_not_advance_sampler(self):
    graphs=[graph(),graph()];m.initialize_sampling(graphs,0,torch.device('cpu'))
    model=m.base.BiasMLP('node_neighbors');opt=torch.optim.AdamW(model.parameters())
    before=m.snapshot(model,opt,graphs,m.empty_runtime(),{})
    probes=[m.fixed_probe(g,8+i) for i,g in enumerate(graphs)]
    validation=[m.diagnostic.validation_pairs(g) for g in graphs]
    m.measure(model,graphs,validation,probes)
    after=m.snapshot(model,opt,graphs,m.empty_runtime(),{})
    self.assertTrue(model.training)
    self.assertTrue(torch.equal(before['rng']['cpu'],after['rng']['cpu']))
    for a,b in zip(before['samplers'],after['samplers']):
        self.assertEqual(a['offset'],b['offset'])
        for key in ('order','generator','order_generator'):self.assertTrue(torch.equal(a[key],b[key]))

 def test_highest_auc_saved_even_without_patience_reset(self):
    t=dict(best_auc=.8,best_path='old',reference=.8,stale=0)
    m.update_tracker(t,.80001,'new',.0001)
    self.assertEqual(t['best_path'],'new');self.assertEqual(t['stale'],1)

 def test_two_rewinds_and_sibling_matched_export(self):
    with tempfile.TemporaryDirectory() as tmp:
        args=SimpleNamespace(root=tmp,arm='async_kd',config='unused',seed=0,learning_rate=.0005,
           max_steps=14,validation_interval=2,patience=2,min_delta=.0001,minimum_steps=0,log_interval=2)
        seq=iter([(.5,.5),(.6,.7),(.61,.69),(.62,.68),(.65,.7),(.64,.7),(.63,.7),(.65,.7)])
        def measure(*_):
            a,b=next(seq)
            return dict(validation=[dict(auc=a,bce=1-a),dict(auc=b,bce=1-b)],training_probe=[dict(bce=.2,auc=.8)]*2)
        @contextlib.contextmanager
        def track(*_):yield SimpleNamespace(summary={}),lambda *args:None
        with patch.object(m.base,'preflight'),patch.object(m,'load_config',return_value={}),patch.object(m.base,'load_graph',side_effect=lambda *_:graph()),patch.object(m,'measure',side_effect=measure),patch.object(m,'tracked_run',side_effect=track):
            m.train(args,torch.device('cpu'))
        run=Path(tmp)/'node_neighbors/lp/async_kd'
        summary=json.loads((run/'summary.json').read_text())
        self.assertEqual(summary['physical_steps'],12);self.assertEqual(summary['logical_step'],4)
        self.assertEqual(summary['stop_reason'],'all_tasks_converged')
        self.assertEqual([e['source'] for e in summary['events']],[m.SOURCES[1],m.SOURCES[0]])
        self.assertEqual(summary['surviving_counts'][1]['kd_updates'],1)
        self.assertEqual(summary['physical_counts'][1]['kd_updates'],3)
        self.assertEqual(summary['teacher_forward_batches'],3)
        endpoint=torch.load(run/'endpoint.pt',weights_only=False)
        best=torch.load(run/'checkpoints/physical_000008_logical_000004.pt',weights_only=False)
        for key,value in endpoint['model'].items():self.assertTrue(torch.equal(value,best['model'][key]))
        # The raw fork retains stale=0 for unfinished A, before applying the event.
        fork=torch.load(Path(tmp)/'first_rewind.pt',weights_only=False)
        self.assertEqual(fork['runtime']['trackers'][0]['stale'],0)
        for arm in ('rewind_bce','extended_bce'):
            args.arm=arm
            constant=lambda *_:dict(validation=[dict(auc=.6,bce=.4)]*2,training_probe=[dict(auc=.6,bce=.4)]*2)
            with patch.object(m.base,'preflight'),patch.object(m,'load_config',return_value={}),patch.object(m.base,'load_graph',side_effect=lambda *_:graph()),patch.object(m,'measure',side_effect=constant),patch.object(m,'tracked_run',side_effect=track):
                m.train(args,torch.device('cpu'))
        m.prepare_eval(args)
        manifest=json.loads((Path(tmp)/'evaluation_manifest.json').read_text())
        self.assertEqual(len(manifest['models']),5)
        self.assertTrue(manifest['replay_check']['within_1e_6'])
        for arm in ('rewind_bce_matched','extended_bce_matched'):
            cp=torch.load(Path(tmp)/'node_neighbors/lp'/arm/'best.pt',weights_only=False)
            self.assertEqual(cp['step'],4)
            self.assertEqual(cp['runtime']['counts'][0]['positive_examples'],endpoint['runtime']['counts'][0]['positive_examples'])

if __name__=='__main__':unittest.main()
