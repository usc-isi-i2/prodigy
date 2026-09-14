"""Paired counterfactual AdamW steps; checkpoint files are read only."""
import argparse,csv,json
from pathlib import Path
import torch
import torch.nn.functional as F
from . import gradient_diagnostic as d
from . import interleaved_mlp_pairs as base
from .ranking_preservation import CASES,SOURCE_ROOT
from .config import load_config
from .evaluation_artifacts import atomic_json
from .mini_transfer import identity

def corrected(a,b):
    # Symmetric two-task projection against ORIGINAL gradients.
    dot=sum((x.double()*y.double()).sum() for x,y in zip(a,b))
    aa=sum(x.double().square().sum() for x in a)
    bb=sum(x.double().square().sum() for x in b)
    if dot>=0:return [(x+y)/2 for x,y in zip(a,b)]
    return [(x-y*(dot/bb.clamp_min(1e-30)).to(x.dtype)+y-x*(dot/aa.clamp_min(1e-30)).to(x.dtype))/2 for x,y in zip(a,b)]

def norm(xs):return sum(x.double().square().sum() for x in xs).sqrt().item()

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,choices=(0,1),required=True);p.add_argument('--device',type=int,choices=range(4),required=True);p.add_argument('--output',required=True);p.add_argument('--trials',type=int,default=8);p.add_argument('--config',default='configs/nonzero_mini_transfer.yaml');args=p.parse_args()
    if args.trials<1:p.error('positive trials required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(f'cuda:{args.device}');a,b=CASES[args.case]
    graphs=[base.load_graph(s,load_config(args.config),0,device) for s in (a,b)]
    parts=[d.validation_pairs(g) for g in graphs]
    run=d.PILOT/f'warm_{a}__and__{b}__rank0'
    states=[('initial',SOURCE_ROOT/f'ss_{a}'/'best.pt'),('best',run/'best.pt'),('final',run/'latest.pt')]
    rows=[];receipts=[]
    for state,path in states:
        ck=torch.load(path,map_location='cpu',weights_only=False);model,_=d.snapshot_model(ck,device)
        before=[d.evaluate(model,g,ps)[0] for g,ps in zip(graphs,parts)]
        receipts.append(dict(state=state,checkpoint=identity(path),checkpoint_step=ck['step']))
        for trial in range(args.trials):
            batches=[];grads=[];train_before=[]
            for i,g in enumerate(graphs):
                generator=torch.Generator(device=device).manual_seed(900001+1009*trial+31*i)
                order=torch.randperm(g['positive'].shape[1],generator=generator,device=device)[:512]
                pairs,y=base.lp.pair_batch(g['positive'][:,order],g['sampler'],generator);batches.append((pairs,y))
                loss=F.binary_cross_entropy_with_logits(base.score(model,g['x'],pairs),y)
                train_before.append(float(loss.detach()));grads.append(d.gradients(model,loss))
            geo=d.geometry(*grads)
            actions={'mixed':[(x+y)/2 for x,y in zip(*grads)],'corrected':corrected(*grads)}
            deltas={}
            for action,gs in actions.items():
                disposable,opt=d.snapshot_model(ck,device)
                for param,g in zip(disposable.parameters(),gs):param.grad=g.clone()
                torch.nn.utils.clip_grad_norm_(disposable.parameters(),1.);opt.step()
                deltas[action]=[p.detach().clone()-q.detach() for p,q in zip(disposable.parameters(),model.parameters())]
                del disposable,opt
            ratio=norm(deltas['corrected'])/max(norm(deltas['mixed']),1e-30)
            deltas['mixed_norm_matched']=[x*ratio for x in deltas['mixed']]
            for action,delta in deltas.items():
                disposable,_=d.snapshot_model(ck,device)
                with torch.no_grad():
                    for p,q,change in zip(disposable.parameters(),model.parameters(),delta):p.copy_(q+change)
                for i,(g,ps) in enumerate(zip(graphs,parts)):
                    after,_=d.evaluate(disposable,g,ps)
                    pairs,y=batches[i]
                    with torch.no_grad():train_after=float(F.binary_cross_entropy_with_logits(base.score(disposable,g['x'],pairs),y))
                    rows.append(dict(state=state,trial=trial,action=action,source=(a,b)[i],gradient_cosine=geo['cosine'],conflict=geo['dot']<0,update_norm=norm(delta),matched_scale=ratio,train_bce_delta=train_after-train_before[i],bce_before=before[i]['bce'],bce_delta=after['bce']-before[i]['bce'],auc_before=before[i]['auc'],auc_delta_pp=100*(after['auc']-before[i]['auc'])))
                del disposable
            print(json.dumps(dict(case=args.case,state=state,trial=trial)),flush=True)
        del model;torch.cuda.empty_cache()
    with (out/'updates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    atomic_json(out/'COMPLETE.json',dict(status='complete',checkpoints=receipts,trials=args.trials,sources=[a,b],protocol='512 positives per source; mean loss; symmetric raw-gradient projection before clipping and AdamW; ordinary actual parameter delta rescaled to corrected delta norm; saved optimizer restored for every action; fixed source validation; no test data'))
if __name__=='__main__':main()
