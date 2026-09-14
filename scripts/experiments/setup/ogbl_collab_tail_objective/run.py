"""Frozen fixed-pool BCE versus BCE+extreme-tail ranking; validation only."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ogbl_collab_compact_joint'))
import run as j
from fusion import hits


def contract():
    return dict(name='h2_tail_v1',revision=j.revision(),arms=['bce','tail'],seeds=[0,1,2],steps=2000,interval=50,train_year=2017,validation_year=2018,selection='earliest maximum standalone validation Hits@50',objective='balanced sampled BCE + 1*mean softplus(1 + n - p), Cartesian product of same2048 broad positives with top50 training negatives; top50 refreshed every10 updates',invariants='fixed100k negative panel,2048pos/1024uniform/1024mined,top2048 refresh10,Adam.001 wd.0001 clip5,496385 neural weights',advance='all3 paired selected differences positive and mean>=0.005; otherwise stop',test_access=False,prior_test_exploration_disclosed=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--original',type=Path,required=True); p.add_argument('--arm',choices=['bce','tail'],default='tail'); p.add_argument('--seed',type=int,choices=range(3),default=0); p.add_argument('--device',default='cuda:1'); p.add_argument('--dry-run',action='store_true'); a=p.parse_args(); cfg=contract()
    if a.dry_run: print(json.dumps(cfg,indent=2)); return
    assert a.device=='cuda:1'
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=a.out/'protocol.json'
    if manifest.exists(): assert j.read(manifest)==cfg
    else: j.save(manifest,cfg)
    out=a.out/f'{a.arm}{a.seed}'; out.mkdir(exist_ok=False)
    pm=j.read(a.source/'prepared.json'); om=j.read(a.original/'prepared.json'); files={}
    for path in [a.source/'train2017.npz',a.original/'node_features.npy',a.original/'standardization.npz',a.original/'assessment/year2018.npz']:
        expected=pm['files'][str(path)]; assert j.shared.sha256_file(path)==expected; files[str(path)]=expected
    pp=a.original/'year2017.npz'; assert j.shared.sha256_file(pp)==om['files']['year2017.npz']; files[str(pp)]=om['files']['year2017.npz']
    tr=np.load(a.source/'train2017.npz'); va=np.load(a.original/'assessment/year2018.npz'); old=np.load(pp); ms=np.load(a.original/'standardization.npz')
    torch.set_num_threads(8); torch.cuda.set_device(a.device); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); generator=torch.Generator().manual_seed(a.seed)
    x=torch.as_tensor(np.load(a.original/'node_features.npy'),device=a.device); keys=j.gate.keys
    assert not np.intersect1d(keys(tr['neg'],len(x)),keys(va['neg'],len(x))).size
    assert not np.intersect1d(keys(tr['neg'],len(x)),keys(old['neg'],len(x))).size
    assert not np.intersect1d(keys(tr['pos'],len(x)),keys(old['neg'],len(x))).size
    np.testing.assert_array_equal(tr['graph_edges'],old['graph_edges'])
    probe=dict(neg=old['neg'],nsymmetric=old['nsymmetric'],pos=tr['pos'],psymmetric=tr['psymmetric'])
    tp=j.panel_tensors(tr,ms['mean'],ms['std'],a.device); vp=j.panel_tensors(va,ms['mean'],ms['std'],a.device); up=j.panel_tensors(probe,ms['mean'],ms['std'],a.device)
    adj=j.adjacency(tr['graph_edges'],len(x),a.device); vadj=j.adjacency(va['graph_edges'],len(x),a.device)
    model=j.Model('joint').to(a.device); optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    import wandb
    cfg=dict(cfg,arm=a.arm,seed=a.seed,device=a.device,files=files)
    j.save(out/'config.json',cfg); w=wandb.init(project='ogbl-collab-compact-joint',group='h2-tail-v1',name=out.name,mode='offline',dir=str(out),config=cfg)
    history=[]; best=None; stream=hashlib.sha256(); started=time.monotonic()
    for step in range(1,2001):
        if (step-1)%10==0:
            with torch.no_grad():
                model.eval(); z=model.encode(x,adj); ns=j.scores(model,z,tp['n']); hard=ns.topk(2048).indices.cpu(); tail=ns.topk(50).indices; del z,ns
        pi,ni,raw=j.sampled_indices(generator,len(tp['p'][0]),len(tp['n'][0]),hard); stream.update(raw)
        model.train(); optimizer.zero_grad(set_to_none=True); z=model.encode(x,adj)
        values=[]
        for side,index in [('p',pi),('n',ni)]:
            index=index.to(a.device); edges,features=tp[side]; values.append(model.score(z,edges[index],features[index]))
        bce=F.binary_cross_entropy_with_logits(torch.cat(values),torch.cat([torch.ones_like(values[0]),torch.zeros_like(values[1])]))
        if a.arm=='tail':
            edges,features=tp['n']; tv=model.score(z,edges[tail],features[tail]); ranking=F.softplus(1+tv[None,:]-values[0][:,None]).mean(); loss=bce+ranking
        else: ranking=bce.detach()*0; loss=bce
        assert torch.isfinite(loss); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.); assert torch.isfinite(grad); optimizer.step(); del z,values
        if step%50==0:
            sc=j.evaluate(model,x,vadj,vp); vh=hits(**sc)
            seen=j.evaluate(model,x,adj,tp); unseen=j.evaluate(model,x,adj,up)
            row=dict(step=step,loss=float(loss),bce=float(bce),ranking=float(ranking),validation_hits=vh,seen_hits=hits(**seen),probe_hits=hits(**unseen),seen_cutoff=float(np.partition(seen['n'],-50)[-50]),probe_cutoff=float(np.partition(unseen['n'],-50)[-50]),elapsed_seconds=time.monotonic()-started)
            if best is None or vh>best['validation_hits']:
                best=row.copy(); torch.save(dict(model=model.state_dict(),step=step,seed=a.seed,revision=cfg['revision']),out/'best.pt'); np.savez_compressed(out/'best_scores.npz',**sc)
            history.append(row); j.save(out/'history.json',history); w.log(row); print(json.dumps(dict(arm=a.arm,seed=a.seed,**row)),flush=True)
    torch.save(dict(model=model.state_dict(),step=2000,seed=a.seed,revision=cfg['revision']),out/'final.pt')
    state=torch.load(out/'best.pt',map_location=a.device,weights_only=False); model.load_state_dict(state['model']); replay=j.evaluate(model,x,vadj,vp); saved=np.load(out/'best_scores.npz')
    for side in ['p','n']: np.testing.assert_array_equal(replay[side],saved[side])
    from ogb.linkproppred import Evaluator
    e=Evaluator(name='ogbl-collab'); e.K=50; assert hits(**replay)==e.eval(dict(y_pred_pos=replay['p'],y_pred_neg=replay['n']))['hits@50']
    result=dict(complete=True,arm=a.arm,seed=a.seed,best=best,revision=cfg['revision'],test_access=False,stream_sha256=stream.hexdigest(),neural_parameters=sum(v.numel() for v in model.parameters()),total_inference_scalars_conservative=496448,wandb_directory=w.dir,elapsed_seconds=time.monotonic()-started,max_gpu_bytes=torch.cuda.max_memory_allocated(),files={n:j.shared.sha256_file(out/n) for n in ['history.json','best.pt','best_scores.npz','final.pt','config.json']},selected_replay_exact=True,official_evaluator_parity=True)
    j.save(out/'results.json',result); w.finish()
if __name__=='__main__': main()
