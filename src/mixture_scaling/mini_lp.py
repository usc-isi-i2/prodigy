"""GPU-resident LP matrices with exact nonedges and fixed train-only context."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .config import load_config
from .lattice import SOURCE_ORDER
from .model import NodeMLP
from .mini_transfer import identity
from .node_only_transfer import save_checkpoint, seed_everything
from .ladder_tracking import tracked_run
from .evaluation_artifacts import atomic_json, save_scores
from .binary_metrics import binary_report, evaluation_report
from .evaluate_lattice import load_pair_module


def edge_partition(edges,n,seed):
    u,v=edges.cpu().numpy(); keep=u!=v
    keys=np.unique(np.minimum(u[keep],v[keep])*n+np.maximum(u[keep],v[keep]))
    order=np.random.default_rng(seed).permutation(len(keys)); k=max(1,round(.15*len(keys)))
    groups=[keys[order[2*k:]],keys[order[k:2*k]],keys[order[:k]]]
    return torch.from_numpy(keys),[torch.from_numpy(np.stack((g//n,g%n))) for g in groups]


def prepare(source,config,root,seed):
    path=config['graphs'][source]['path']; expected=dict(graph=identity(path),seed=seed,split='undirected unique nonself 70/15/15 v1')
    cache=Path(root)/'_cache'; cache.mkdir(parents=True,exist_ok=True); file=cache/(source+'.pt')
    with file.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if file.exists():
            data=torch.load(file,map_location='cpu',weights_only=False)
            if data['receipt']!=expected: raise ValueError('graph/split cache identity mismatch')
            return data
        raw=torch.load(path,map_location='cpu',weights_only=False)
        x=raw['x'].float(); edges=raw['edge_index']; keys,groups=edge_partition(edges,len(x),seed)
        if not torch.isfinite(x).all() or not (x!=0).any(1).all(): raise ValueError('nonzero finite features required')
        data=dict(x=x,known_keys=keys,train=groups[0],validation=groups[1],test=groups[2],receipt=expected)
        temp=file.with_suffix('.tmp');torch.save(data,temp);temp.replace(file)
        return data


def fixed_neighbors(edges,n,fanout,seed):
    from scipy.sparse import csr_matrix
    u,v=edges.numpy(); a=csr_matrix((np.ones(2*len(u),dtype=np.bool_),
        (np.concatenate((u,v)),np.concatenate((v,u)))),shape=(n,n))
    a.sum_duplicates(); a.sort_indices(); rng=np.random.default_rng(seed)
    ids=np.full((n,fanout),-1,dtype=np.int64)
    for node in range(n):
        neighbors=a.indices[a.indptr[node]:a.indptr[node+1]]
        chosen=neighbors if len(neighbors)<=fanout else rng.choice(neighbors,fanout,replace=False)
        ids[node,:len(chosen)]=chosen
    return torch.from_numpy(ids)


@torch.no_grad()
def view_features(source,data,root,view,seed,device):
    x=data['x'].to(device)
    if view=='node': return x
    file=Path(root)/'_cache'/(source+'_neighbors10.pt')
    if file.exists():
        cached=torch.load(file,map_location='cpu',weights_only=False)
        if cached['receipt']!=data['receipt']: raise ValueError('context identity mismatch')
        means=cached['means'].to(device)
    else:
        ids=fixed_neighbors(data['train'],len(x),10,seed)
        means=torch.empty_like(x)
        for start in range(0,len(x),2048):
            batch=ids[start:start+2048].to(device); mask=batch>=0
            means[start:start+len(batch)]=(x[batch.clamp_min(0)]*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        temp=file.with_suffix('.tmp');torch.save(dict(means=means.cpu(),neighbor_ids=ids,receipt=data['receipt']),temp);temp.replace(file)
    return torch.cat((x,means),1)


class ExactNonedges:
    def __init__(self,n,keys): self.n=n;self.keys=keys
    def forbidden(self,pairs):
        lo=torch.minimum(pairs[0],pairs[1]);hi=torch.maximum(pairs[0],pairs[1]);keys=lo*self.n+hi
        positions=torch.searchsorted(self.keys,keys).clamp_max(len(self.keys)-1)
        return (lo==hi)|(self.keys[positions]==keys)
    def sample(self,count,generator):
        pairs=torch.randint(self.n,(2,count),generator=generator,device=self.keys.device)
        bad=self.forbidden(pairs)
        while bool(bad.any()):
            pairs[:,bad]=torch.randint(self.n,(2,int(bad.sum())),generator=generator,device=self.keys.device)
            bad=self.forbidden(pairs)
        return pairs


def pair_batch(positive,sampler,generator):
    negative=sampler.sample(positive.shape[1]*5,generator)
    pairs=torch.cat((positive,negative),1)
    labels=torch.cat((torch.ones(positive.shape[1],device=pairs.device),torch.zeros(negative.shape[1],device=pairs.device)))
    return pairs,labels


def score(model,x,pairs):
    n=pairs.shape[1]; z=model(x[pairs.flatten()]);return (z[:n]*z[n:]).sum(1)


@torch.no_grad()
def validate(model,x,positive,sampler,seed):
    model.eval(); generator=torch.Generator(device=x.device).manual_seed(seed+32452843)
    logits=[];labels=[]
    for start in range(0,min(positive.shape[1],20*1024),1024):
        pairs,y=pair_batch(positive[:,start:start+1024],sampler,generator)
        logits.append(score(model,x,pairs));labels.append(y)
    report=binary_report(torch.cat(labels).cpu().numpy(),torch.cat(logits).cpu().numpy())
    model.train(); return report


def model_for(view,device): return NodeMLP(768 if view=='node' else 1536,256,256,0).to(device)


def train(source,config,args,device):
    root=Path(args.root);run_id='ss_'+source;run_dir=root/args.view/'lp'/run_id
    if (run_dir/'summary.json').exists():return
    if run_dir.exists():raise FileExistsError(f'partial run: {run_dir}')
    started=time.monotonic();data=prepare(source,config,root,args.seed)
    x=view_features(source,data,root,args.view,args.seed,device)
    positive=data['train'].to(device);val=data['validation'].to(device)
    sampler=ExactNonedges(len(x),data['known_keys'].to(device))
    seed_everything(args.seed);model=model_for(args.view,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=1e-5)
    generator=torch.Generator(device=device).manual_seed(args.seed+49979687)
    order_generator=torch.Generator(device=device).manual_seed(args.seed+7919)
    protocol=dict(config['protocol'],input_dim=x.shape[1],max_steps=args.max_steps,
        validation_interval=args.validation_interval,patience_evaluations=args.patience,fanout=10 if args.view=='node_neighbors' else 0)
    metadata=dict(run_id=run_id,sources=[source],objective='lp',architecture='node_mlp',view=args.view,
        protocol=protocol,seed=args.seed,graph_split_receipt=data['receipt'],
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        training_negative_sampling='uniform ordered pairs rejecting self-loops and every known edge in either direction',
        context='fixed up-to-10 distinct neighbors from train edges only; mean; zero for isolates' if args.view=='node_neighbors' else 'none',
        checkpoint_selection='source validation BCE only; distinct from final-test edges',
        stopping=dict(patience=args.patience,validation_interval=args.validation_interval,min_delta=1e-4,minimum_steps=2500,safety_cap=args.max_steps))
    run_dir.mkdir(parents=True);atomic_json(run_dir/'metadata.json',metadata)
    offset=positive.shape[1];best=reference=float('inf');stale=best_step=0;reason='safety_cap'
    total=torch.zeros((),device=device);window=0
    print(json.dumps(dict(starting=source,view=args.view,load_seconds=time.monotonic()-started)),flush=True)
    with tracked_run(run_dir,metadata,args) as (run,log):
        for step in range(1,args.max_steps+1):
            if offset>=positive.shape[1]:
                order=torch.randperm(positive.shape[1],generator=order_generator,device=device);offset=0
            pos=positive[:,order[offset:offset+1024]];offset+=pos.shape[1]
            pairs,y=pair_batch(pos,sampler,generator)
            optimizer.zero_grad(set_to_none=True);logits=score(model,x,pairs)
            loss=F.binary_cross_entropy_with_logits(logits,y);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=loss.detach();window+=1
            if step%args.log_interval==0:
                value=float(total/window)
                if not np.isfinite(value):raise FloatingPointError('nonfinite loss')
                log(step,{'train/loss':value,'train/elapsed_seconds':time.monotonic()-started});total.zero_();window=0
            if step%args.validation_interval==0 or step==args.max_steps:
                report=validate(model,x,val,sampler,args.seed);value=report['bce']
                if value<best:
                    best=value;best_step=step;save_checkpoint(run_dir/'best.pt',model,optimizer,step,metadata)
                if value<reference-1e-4:reference=value;stale=0
                elif step>=2500:stale+=1
                log(step,{'validation/loss':value,'validation/auc':report['roc_auc'],'validation/stale_checks':stale})
                atomic_json(run_dir/'validation'/f'step_{step}.json',report)
                save_checkpoint(run_dir/'latest.pt',model,optimizer,step,metadata)
                print(json.dumps(dict(source=source,view=args.view,step=step,validation=value,stale=stale,elapsed_seconds=time.monotonic()-started)),flush=True)
                if stale>=args.patience:reason='validation_plateau';break
        save_checkpoint(run_dir/'checkpoints'/f'step_{step}.pt',model,optimizer,step,metadata)
        summary=dict(metadata,status='complete',final_step=step,best_step=best_step,best_validation_bce=best,
            stop_reason=reason,converged=reason=='validation_plateau',elapsed_seconds=time.monotonic()-started)
        run.summary.update(dict(final_step=step,best_step=best_step,stop_reason=reason))
    atomic_json(run_dir/'summary.json',summary)


def worker(config,args,device):
    root=Path(args.root)/args.view;claims=root/'_claims';claims.mkdir(parents=True,exist_ok=True)
    for source in args.sources:
        with (claims/(source+'.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            train(source,config,args,device)
        torch.cuda.empty_cache()


@torch.no_grad()
def eval_targets(config,args,device):
    root=Path(args.root);pair=load_pair_module(Path(args.prodigy_root))
    for target in args.sources[args.worker_index::args.workers]:
        data=prepare(target,config,root,args.seed);n=len(data['x'])
        background=pair.Adjacency.from_edge_index(data['train'].numpy(),n)
        forbidden_holdout=pair.Adjacency.from_edge_index(torch.cat((data['validation'],data['test']),1).numpy(),n)
        rng=np.random.default_rng(args.seed)
        pairs=pair.build_pair_set(background,forbidden_holdout,'degree_matched',rng,
            max_positives=2000,holdout_edge_index=data['test'].numpy())
        valmask=pair.split_val_mask(pairs,rng);nodes=pairs.nodes()
        if pair.leakage_check(background,pairs):raise ValueError('test positive in train topology')
        x=view_features(target,data,root,args.view,args.seed,device)
        for source in args.sources:
            output=root/args.view/'results'/f'ss_{source}__to__{target}.json'
            if output.exists():continue
            cp=root/args.view/'lp'/('ss_'+source)/'best.pt'
            checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
            model=model_for(args.view,device).eval();model.load_state_dict(checkpoint['model'])
            table=torch.cat([model(x[nodes[start:start+4096]]) for start in range(0,len(nodes),4096)]).cpu().numpy()
            embeddings=pair.NodeEmbeddings(table,nodes,n)
            logits=pair.pair_scores(embeddings,pairs,'dot');cosine=pair.pair_scores(embeddings,pairs,'cosine')
            report=evaluation_report(pairs.label,logits,valmask,cosine)
            output.parent.mkdir(parents=True,exist_ok=True)
            save_scores(output.with_suffix('.scores.npz'),u=pairs.u,v=pairs.v,labels=pairs.label,dot_logits=logits,cosine_scores=cosine,validation_mask=valmask)
            atomic_json(output,dict(status='complete',source=source,target=target,view=args.view,
                checkpoint_step=checkpoint['step'],checkpoint_identity=identity(cp),graph_split_receipt=data['receipt'],
                metrics=report,score_kind='raw dot product',negative_kind='degree_matched; all known edges excluded',
                split='final heldout positives distinct from source early-stopping validation; 30% for calibration, 70% final metrics',
                holdout_leakage_edges=0))
        print(json.dumps(dict(evaluated_target=target,view=args.view)),flush=True)
        del x;torch.cuda.empty_cache()


def aggregate(args):
    root=Path(args.root)/args.view;rows=[]
    for source in args.sources:
        for target in args.sources:
            p=json.loads((root/'results'/f'ss_{source}__to__{target}.json').read_text());m=p['metrics']['test']
            rows.append(dict(source=source,target=target,auc=m['roc_auc'],bce=m['bce'],average_precision=m['average_precision'],checkpoint_step=p['checkpoint_step']))
    with (root/'results'/'matrix.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(root/'COMPLETE.json',dict(status='complete',cells=len(rows)))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','eval','aggregate'])
    p.add_argument('--root',required=True);p.add_argument('--config',default='configs/nonzero_mini_transfer.yaml')
    p.add_argument('--view',choices=['node','node_neighbors'],required=True)
    p.add_argument('--device',type=int,choices=range(4),default=0)
    p.add_argument('--worker-index',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--sources',nargs='+',choices=SOURCE_ORDER,default=list(SOURCE_ORDER))
    p.add_argument('--seed',type=int,default=0);p.add_argument('--max-steps',type=int,default=100000)
    p.add_argument('--validation-interval',type=int,default=2000);p.add_argument('--patience',type=int,default=3)
    p.add_argument('--log-interval',type=int,default=100)
    p.add_argument('--wandb-mode',default='offline');p.add_argument('--wandb-project',default='nonzero-mini-transfer')
    p.add_argument('--wandb-group',default=None)
    p.add_argument('--prodigy-root',default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot')
    args=p.parse_args()
    if args.phase=='aggregate':aggregate(args);return
    torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
    config=load_config(args.config);device=torch.device(f'cuda:{args.device}')
    (worker if args.phase=='train' else eval_targets)(config,args,device)
if __name__=='__main__':main()
