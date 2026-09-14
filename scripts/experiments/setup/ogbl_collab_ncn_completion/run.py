"""Frozen validation-only joint / observed-NCN / completed-NCN comparison."""
import argparse,hashlib,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core import j,Model,tensor_cache,child_weights,all_scores
from fusion import hits


def contract():
    return dict(name='ncn_completion_v1',revision=j.revision(),arms=['control','observed','completion'],seeds=[0,1,2],steps=2000,interval=50,train_year=2017,validation_year=2018,selection='earliest maximum standalone validation Hits@50',neural_parameters=dict(control=496385,observed=525633,completion=525633),conservative_inference_scalars=dict(control=496448,observed=525697,completion=525697),objective='balanced BCE; unchanged positive/uniform-negative/mining slot streams',budget='2048 positive +1024 uniform negative+1024 top2048-mined; refresh10; Adam .001 wd .0001 clip5',representation='same initialized 256-wide SAGE encoder; new heads use product/absolute difference and 64-dimensional neighbor pooling, no handcrafted14; baseline retains handcrafted14',completion='one-step shared observed decoder; sigmoid(logit+log(.1)); detached weights refreshed every10 training updates and freshly on every validation evaluation; cap8 one-sided neighbors per endpoint, cap32 common neighbors, fixed label-independent priority and full-count/sample-count expansion',advance='for each new arm all3 paired selected gains positive and mean>=.005; if either passes, recommend historical replication; higher mean wins with observed tie preference; no test access',test_access=False,prior_test_exploration_disclosed=True,interpretation='adaptive validation research; NCN-inspired compact adaptation, not paper reproduction or capacity-only ablation')


@torch.no_grad()
def evaluate(model,arm,x,adj,panel,cache):
    if arm=='control': return j.evaluate(model,x,adj,panel)
    model.eval();z=model.encode(x,adj)
    weights=child_weights(model,z,cache) if arm=='completion' else None
    count=len(panel['p'][0]);total=len(cache['pairs'])
    return dict(p=all_scores(model,z,cache,0,count,weights).cpu().numpy(),n=all_scores(model,z,cache,count,total,weights).cpu().numpy())


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=['control','observed','completion'],default='control');p.add_argument('--seed',type=int,choices=range(3),default=0);p.add_argument('--device',default='cuda:1');p.add_argument('--dry-run',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args();cfg=contract()
    if a.dry_run: print(cfg);return
    assert a.device in ['cuda:0','cuda:1','cuda:2','cuda:3']
    torch.set_num_threads(8);torch.cuda.set_device(a.device)
    a.out.mkdir(parents=True,exist_ok=True);out=a.out/f'{a.arm}{a.seed}';out.mkdir(exist_ok=False)
    pm=j.read(a.cache/'prepared.json');assert pm['complete']
    for name,digest in pm['files'].items():assert j.shared.sha256_file(Path(name))==digest
    for c in pm['caches']:assert j.shared.sha256_file(a.cache/f'{c["name"]}.npz')==c['sha256']
    tr=np.load(a.source/'train2017.npz');va=np.load(a.original/'assessment/year2018.npz');ms=np.load(a.original/'standardization.npz')
    rawx=np.load(a.original/'node_features.npy');n=len(rawx)
    assert not np.intersect1d(j.gate.keys(tr['neg'],n),j.gate.keys(va['neg'],n)).size
    tc=vc=None
    if a.arm!='control':
        tcn=np.load(a.cache/'train.npz');vcn=np.load(a.cache/'validation.npz')
        np.testing.assert_array_equal(tcn['pairs'],np.concatenate([tr['pos'],tr['neg']]))
        np.testing.assert_array_equal(vcn['pairs'],np.concatenate([va['pos'],va['neg']]))
        tc=tensor_cache(tcn,a.device);vc=tensor_cache(vcn,a.device)
    x=torch.as_tensor(rawx,device=a.device);tp=j.panel_tensors(tr,ms['mean'],ms['std'],a.device);vp=j.panel_tensors(va,ms['mean'],ms['std'],a.device)
    adj=j.adjacency(tr['graph_edges'],n,a.device);vadj=j.adjacency(va['graph_edges'],n,a.device)
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);generator=torch.Generator().manual_seed(a.seed)
    model=(j.Model('joint') if a.arm=='control' else Model()).to(a.device)
    assert sum(v.numel() for v in model.parameters())==cfg['neural_parameters'][a.arm]
    encoder_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.encoder.state_dict().values())).hexdigest()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    cfg=dict(cfg,arm=a.arm,seed=a.seed,device=a.device,smoke=a.smoke,source_files=pm['files'],cache_manifest_sha256=j.shared.sha256_file(a.cache/'prepared.json'),initial_encoder_sha256=encoder_hash)
    j.save(out/'config.json',cfg)
    import wandb
    w=wandb.init(project='ogbl-collab-compact-joint',group='ncn-completion-v1'+('-smoke' if a.smoke else ''),name=out.name,mode='offline',dir=str(out),config=cfg)
    history=[];best=None;stream=hashlib.sha256();started=time.monotonic();weights=None;count=len(tp['p'][0])
    for step in range(1,(10 if a.smoke else 2000)+1):
        if (step-1)%10==0:
            with torch.no_grad():
                model.eval();z=model.encode(x,adj)
                if a.arm=='control': ns=j.scores(model,z,tp['n'])
                else:
                    weights=child_weights(model,z,tc) if a.arm=='completion' else None
                    ns=all_scores(model,z,tc,count,len(tc['pairs']),weights)
                hard=ns.topk(2048).indices.cpu();del z,ns
        pi,ni,raw=j.sampled_indices(generator,count,len(tp['n'][0]),hard);stream.update(raw)
        model.train();optimizer.zero_grad(set_to_none=True);z=model.encode(x,adj);values=[]
        for side,index in [('p',pi),('n',ni)]:
            index=index.to(a.device)
            if a.arm=='control':
                edges,features=tp[side];values.append(model.score(z,edges[index],features[index]))
            else:values.append(model.score_rows(z,tc,index+(0 if side=='p' else count),weights))
        loss=F.binary_cross_entropy_with_logits(torch.cat(values),torch.cat([torch.ones_like(values[0]),torch.zeros_like(values[1])]))
        assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(grad);optimizer.step();del z,values
        if step%50==0:
            sc=evaluate(model,a.arm,x,vadj,vp,vc);vh=hits(**sc)
            row=dict(step=step,loss=float(loss),validation_hits=vh,elapsed_seconds=time.monotonic()-started)
            if best is None or vh>best['validation_hits']:
                best=row.copy();torch.save(dict(model=model.state_dict(),step=step,seed=a.seed,arm=a.arm,revision=j.revision()),out/'best.pt');np.savez_compressed(out/'best_scores.npz',**sc)
            history.append(row);j.save(out/'history.json',history);w.log(row);print(dict(arm=a.arm,seed=a.seed,**row),flush=True)
    if a.smoke:
        row=dict(smoke_only=True,steps=10,seconds=time.monotonic()-started,max_gpu_bytes=torch.cuda.max_memory_allocated(),final_loss=float(loss),neural_parameters=sum(v.numel() for v in model.parameters()),initial_encoder_sha256=encoder_hash)
        j.save(out/'smoke.json',row);print(row,flush=True);w.finish();return
    torch.save(dict(model=model.state_dict(),step=2000,seed=a.seed,arm=a.arm,revision=j.revision()),out/'final.pt')
    state=torch.load(out/'best.pt',map_location=a.device,weights_only=False);model.load_state_dict(state['model']);replay=evaluate(model,a.arm,x,vadj,vp,vc);saved=np.load(out/'best_scores.npz')
    for side in ['p','n']:np.testing.assert_array_equal(replay[side],saved[side])
    from ogb.linkproppred import Evaluator
    e=Evaluator(name='ogbl-collab');e.K=50;assert hits(**replay)==e.eval(dict(y_pred_pos=replay['p'],y_pred_neg=replay['n']))['hits@50']
    result=dict(complete=True,arm=a.arm,seed=a.seed,best=best,revision=j.revision(),test_access=False,stream_sha256=stream.hexdigest(),neural_parameters=sum(v.numel() for v in model.parameters()),total_inference_scalars_conservative=cfg['conservative_inference_scalars'][a.arm],wandb_directory=w.dir,elapsed_seconds=time.monotonic()-started,max_gpu_bytes=torch.cuda.max_memory_allocated(),initial_encoder_sha256=encoder_hash,files={name:j.shared.sha256_file(out/name) for name in ['config.json','history.json','best.pt','best_scores.npz','final.pt']},selected_replay_exact=True,official_evaluator_parity=True)
    j.save(out/'results.json',result);w.finish()
if __name__=='__main__':main()
