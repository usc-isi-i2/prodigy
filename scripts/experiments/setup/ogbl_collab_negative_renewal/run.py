"""Frozen negative-renewal experiment; no test reader or scorer."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ogbl_collab_compact_joint'))
import run as joint
from features import build
ROOT=Path('/dataMeR1/phil/gfm/ogbl_collab_compact_joint')

def cfg():
    return dict(name='negative_renewal_v1',revision=joint.revision(),arms=['fixed','renew'],seeds=[0,1,2],steps=2000,interval=50,train_year=2017,validation_year=2018,renewal='initial archived fresh pool, then independent pool every50 updates; seed 2026091400+pool_index; target2017 positives and validation negatives excluded; no test exclusions',negative_count=100000,probe='pool40 masked against union of optimized pools, size recorded; zero training-pair overlap',selection='earliest maximum standalone validation Hits@50',advance='all3 paired gains positive and mean gain>=0.005; then earlier chronological replication',invariants='BCE, Adam.001 wd.0001 clip5;2048positive+1024uniform+1024hard, top2048 mined every10; original mean/std/calibration; matched positive/uniform/index-slot RNG, hard identities necessarily differ',prior_test_exploration_disclosed=True,test_access=False)

def tracking(out,name):
    import wandb
    return wandb.init(project='ogbl-collab-compact-joint',group='negative_renewal_v1',name=name,mode='offline',dir=str(out),config=cfg())

def prepare(a):
    a.out.mkdir(exist_ok=False,parents=True); joint.save(a.out/'protocol.json',cfg())
    run=tracking(a.out,'prepare'); started=time.monotonic()
    meta=joint.read(ROOT/'candidate_fresh_v1/prepared.json')
    for path,sha in meta['files'].items():
        if 'train2018' not in path:
            assert joint.shared.sha256_file(Path(path))==sha
    import subprocess
    assert subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()==joint.gate.UPSTREAM
    assert (a.upstream/'aa_dc.py').read_bytes()==subprocess.check_output(['git','-C',str(a.upstream),'show','HEAD:aa_dc.py'])
    tr=np.load(ROOT/'candidate_fresh_v1/train2017.npz'); va=np.load(ROOT/'joint_v1/assessment/year2018.npz')
    import pandas as pd
    raw=a.dataset/'raw'
    graph=dict(num_nodes=len(np.load(ROOT/'joint_v1/node_features.npy')),node_feat=pd.read_csv(raw/'node-feat.csv.gz',header=None).values,edge_index=pd.read_csv(raw/'edge.csv.gz',header=None).values.T,edge_year=pd.read_csv(raw/'edge_year.csv.gz',header=None).values,edge_weight=pd.read_csv(raw/'edge_weight.csv.gz',header=None).values)
    t=torch.load(a.dataset/'split/time/train.pt',weights_only=False)
    split={'train':{k:np.asarray(v) for k,v in t.items()}}
    aa=joint.module(a.upstream/'aa_dc.py','renew_aa')
    calibration=joint.read(ROOT/'joint_v1/prepared.json')['calibration']
    np.testing.assert_array_equal(joint.graph_edges(graph,split,2017),tr['graph_edges'])
    score=build(aa,joint,graph,split,2017,calibration)
    replay=score(tr['neg']); np.testing.assert_array_equal(replay,tr['nsymmetric'])
    joint.save(a.out/'feature_parity.json',dict(exact=True,pairs=100000,elapsed=time.monotonic()-started))
    blocked=joint.gate.keys(va['neg'],graph['num_nodes']); records=[]
    used=[joint.gate.keys(tr['neg'],graph['num_nodes'])]
    for index in range(1,41): #40 is independent probe, never optimized
        tick=time.monotonic(); neg=joint.gate.negatives(tr['pos'],graph['num_nodes'],2026091400+index,count=100100)
        neg=neg[~np.isin(joint.gate.keys(neg,graph['num_nodes']),blocked)][:100000]
        if index==40:
            neg=neg[~np.isin(joint.gate.keys(neg,graph['num_nodes']),np.concatenate(used))]
        assert len(neg)>=99900 if index==40 else len(neg)==100000
        if index<40: used.append(joint.gate.keys(neg,graph['num_nodes']))
        feat=score(neg)
        path=a.out/f'pool{index}.npz'; np.savez_compressed(path,neg=neg,nsymmetric=feat)
        row=dict(index=index,count=len(neg),sha256=joint.shared.sha256_file(path),validation_overlap=0,positive_overlap=int(np.intersect1d(joint.gate.keys(neg,graph['num_nodes']),joint.gate.keys(tr['pos'],graph['num_nodes'])).size),seconds=time.monotonic()-tick)
        assert row['positive_overlap']==0
        records.append(row); joint.save(a.out/'pools.json',records); print(json.dumps(row),flush=True)
    joint.save(a.out/'prepared.json',dict(complete=True,records=records,seconds=time.monotonic()-started)); run.finish()

def train(a):
    assert joint.read(a.out/'protocol.json')==cfg()
    out=a.out/f'{a.arm}{a.seed}'; out.mkdir(exist_ok=False)
    run=tracking(out,out.name); torch.set_num_threads(8); torch.cuda.set_device(0)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); generator=torch.Generator().manual_seed(a.seed)
    tr=np.load(ROOT/'candidate_fresh_v1/train2017.npz'); va=np.load(ROOT/'joint_v1/assessment/year2018.npz'); ms=np.load(ROOT/'joint_v1/standardization.npz')
    x=torch.as_tensor(np.load(ROOT/'joint_v1/node_features.npy'),device='cuda:0')
    tp=joint.panel_tensors(tr,ms['mean'],ms['std'],'cuda:0'); vp=joint.panel_tensors(va,ms['mean'],ms['std'],'cuda:0')
    adj=joint.adjacency(tr['graph_edges'],len(x),'cuda:0'); vadj=joint.adjacency(va['graph_edges'],len(x),'cuda:0')
    records=joint.read(a.out/'prepared.json')['records']
    def pool(index):
        path=a.out/f'pool{index}.npz'
        assert joint.shared.sha256_file(path)==records[index-1]['sha256']
        return np.load(path)
    probe=pool(40)
    def negative(data):
        return (torch.as_tensor(data['neg'],device='cuda:0'),torch.as_tensor((data['nsymmetric']-ms['mean'])/ms['std'],device='cuda:0'))
    pp={'p':tp['p'],'n':negative(probe)}
    model=joint.Model('joint').to('cuda:0'); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    history=[]; best=None; started=time.monotonic(); stream=hashlib.sha256()
    for step in range(1,2001):
        if a.arm=='renew' and step>1 and (step-1)%50==0:
            tp['n']=negative(pool((step-1)//50))
        if (step-1)%10==0:
            with torch.no_grad():
                model.eval(); z=model.encode(x,adj); hard=joint.scores(model,z,tp['n']).topk(2048).indices.cpu(); del z
        pi,ni,raw=joint.sampled_indices(generator,len(tp['p'][0]),len(tp['n'][0]),hard); stream.update(raw)
        model.train(); opt.zero_grad(set_to_none=True); z=model.encode(x,adj); vals=[]
        for side,index in [('p',pi),('n',ni)]:
            index=index.to('cuda:0'); e,f=tp[side]; vals.append(model.score(z,e[index],f[index]))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(torch.cat(vals),torch.cat([torch.ones_like(vals[0]),torch.zeros_like(vals[1])]))
        assert torch.isfinite(loss); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.); assert torch.isfinite(grad); opt.step(); del z,vals
        if step%50==0:
            scores=joint.evaluate(model,x,vadj,vp); h=joint.gate.hits(scores['p'],scores['n'])
            with torch.no_grad():
                model.eval(); z=model.encode(x,adj); p=joint.scores(model,z,tp['p']).cpu().numpy(); n=joint.scores(model,z,tp['n']).cpu().numpy(); q=joint.scores(model,z,pp['n']).cpu().numpy(); del z
            row=dict(step=step,hits=h,loss=float(loss),seen_hits=joint.gate.hits(p,n),probe_hits=joint.gate.hits(p,q),seen_cutoff=float(np.partition(n,-50)[-50]),probe_cutoff=float(np.partition(q,-50)[-50]),seconds=time.monotonic()-started)
            if best is None or h>best['hits']:
                best=row.copy(); torch.save(dict(model=model.state_dict(),step=step,seed=a.seed),out/'best.pt'); np.savez_compressed(out/'best_scores.npz',**scores)
            history.append(row); joint.save(out/'history.json',history); run.log(row); print(json.dumps(dict(arm=a.arm,seed=a.seed,**row)),flush=True)
    torch.save(dict(model=model.state_dict(),step=2000),out/'final.pt')
    from ogb.linkproppred import Evaluator
    saved=np.load(out/'best_scores.npz'); ev=Evaluator(name='ogbl-collab'); ev.K=50
    assert best['hits']==ev.eval(dict(y_pred_pos=saved['p'],y_pred_neg=saved['n']))['hits@50']
    state=torch.load(out/'best.pt',map_location='cuda:0',weights_only=False); model.load_state_dict(state['model'])
    replay=joint.evaluate(model,x,vadj,vp)
    for side in ('p','n'): np.testing.assert_array_equal(replay[side],saved[side])
    result=dict(exact_checkpoint_replay=True,score_sha256=joint.shared.sha256_file(out/'best_scores.npz'),total_inference_scalars=496448,complete=True,best=best,seed=a.seed,arm=a.arm,revision=joint.revision(),sample_stream_sha256=stream.hexdigest(),history_sha256=joint.shared.sha256_file(out/'history.json'),checkpoint_sha256=joint.shared.sha256_file(out/'best.pt'),seconds=time.monotonic()-started,wandb_dir=run.dir,test_scored=False,neural_parameters=sum(p.numel() for p in model.parameters()))
    joint.save(out/'results.json',result); run.finish()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['prepare','train','dry-run']); p.add_argument('--out',type=Path,default=ROOT/'negative_renewal_v1'); p.add_argument('--dataset',type=Path,default=Path('/dataMeR1/phil/data/ogb/ogbl_collab')); p.add_argument('--upstream',type=Path,default=Path('/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204/upstream')); p.add_argument('--seed',type=int,default=0); p.add_argument('--arm',choices=['fixed','renew'],default='fixed'); a=p.parse_args()
    if a.phase=='dry-run': print(json.dumps(cfg(),indent=2))
    else: {'prepare':prepare,'train':train}[a.phase](a)
