"""Disposable one-step diagnostics; no training or checkpoint mutation."""
import argparse,copy,csv,json
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from . import interleaved_mlp_pairs as base
from .ranking_preservation import CASES,SOURCE_ROOT
from .config import load_config
from .mini_transfer import identity
from .evaluation_artifacts import atomic_json

PILOT=Path('/dataMeR1/phil/gfm/mixture-scaling/state/ranking_preservation_s0/node_neighbors/lp')

def geometry(a,b):
    x=torch.cat([v.detach().reshape(-1).double() for v in a]);y=torch.cat([v.detach().reshape(-1).double() for v in b])
    nx=float(x.norm());ny=float(y.norm());dot=float(x@y)
    return dict(norm_left=nx,norm_right=ny,dot=dot,cosine=dot/(nx*ny) if nx*ny>1e-30 else None)

def groups(model):
    names=[n for n,_ in model.named_parameters()]
    return {'all':list(range(len(names))),'first_layer':[i for i,n in enumerate(names) if n.startswith('network.0')],
            'second_layer':[i for i,n in enumerate(names) if n.startswith('network.3')],'decoder_bias':[i for i,n in enumerate(names) if n=='decoder_bias']}

def snapshot_model(ck,device):
    model=base.BiasMLP('node_neighbors').to(device);model.load_state_dict(ck['model']);model.eval()
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=1e-5)
    opt.load_state_dict(copy.deepcopy(ck['optimizer']))
    return model,opt

def gradients(model,loss):
    return [g.detach().clone() for g in torch.autograd.grad(loss,tuple(model.parameters()))]

def validation_pairs(g):
    generator=torch.Generator(device=g['x'].device).manual_seed(32452843)
    positive=g['validation'];parts=[]
    for start in range(0,min(positive.shape[1],20*1024),1024):parts.append(base.lp.pair_batch(positive[:,start:start+1024],g['sampler'],generator))
    return parts

def evaluate(model,g,parts,with_gradient=False):
    scores=[];labels=[];total=sum(len(y) for _,y in parts);loss_value=0.
    gs=[torch.zeros_like(p) for p in model.parameters()] if with_gradient else None
    for pairs,y in parts:
        with torch.set_grad_enabled(with_gradient):
            logits=base.score(model,g['x'],pairs);loss=F.binary_cross_entropy_with_logits(logits,y)
            loss_value+=float(loss.detach())*len(y)/total
            if with_gradient:
                grad=gradients(model,loss)
                for dest,src in zip(gs,grad):dest.add_(src,alpha=len(y)/total)
        scores.append(logits.detach().cpu().numpy());labels.append(y.cpu().numpy())
    return dict(bce=loss_value,auc=float(roc_auc_score(np.concatenate(labels),np.concatenate(scores)))),gs

def training_batch(g,trial,source_index):
    generator=torch.Generator(device=g['x'].device).manual_seed(900001+1009*trial+31*source_index)
    order=torch.randperm(g['positive'].shape[1],generator=generator,device=g['x'].device)[:1024]
    return base.lp.pair_batch(g['positive'][:,order],g['sampler'],generator)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,choices=(0,1),required=True);p.add_argument('--device',type=int,choices=range(4),required=True);p.add_argument('--output',required=True);p.add_argument('--trials',type=int,default=8);p.add_argument('--config',default='configs/nonzero_mini_transfer.yaml');args=p.parse_args()
    if args.trials<1:p.error('positive trials required')
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(f'cuda:{args.device}');a,b=CASES[args.case];config=load_config(args.config)
    graphs=[base.load_graph(s,config,0,device) for s in (a,b)];parts=[validation_pairs(g) for g in graphs]
    source_cp=SOURCE_ROOT/f'ss_{a}'/'best.pt';teacher_ck=torch.load(source_cp,map_location='cpu',weights_only=False)
    teacher,_=snapshot_model(teacher_ck,device);teacher.requires_grad_(False)
    states=[('initial',source_cp,0,0)]
    for w in (0,1):
        run=PILOT/f'warm_{a}__and__{b}__rank{w}';summary=json.loads((run/'summary.json').read_text())
        states.extend([(f'rank{w}_best',run/'best.pt',w,summary['best_step']),(f'rank{w}_final',run/'latest.pt',w,summary['final_step'])])
    grow=[];urows=[];receipts=[]
    for state,path,weight,step in states:
        ck=torch.load(path,map_location='cpu',weights_only=False);model,_=snapshot_model(ck,device);group=groups(model)
        reports=[];vg=[]
        for g,ps in zip(graphs,parts):
            report,grad=evaluate(model,g,ps,True);reports.append(report);vg.append(grad)
        receipts.append(dict(state=state,checkpoint=identity(path),continuation_step=step,checkpoint_step=ck['step'],validation=reports))
        for trial in range(args.trials):
            tg=[];losses=[];batch=[]
            for i,g in enumerate(graphs):
                pairs,y=training_batch(g,trial,i);batch.append((pairs,y))
                loss=F.binary_cross_entropy_with_logits(base.score(model,g['x'],pairs),y);losses.append(float(loss.detach()));tg.append(gradients(model,loss))
            vectors={'train_A':tg[0],'train_B':tg[1],'val_A':vg[0],'val_B':vg[1]}
            actions={'A_task':tg[0],'B_task':tg[1]}
            if weight:
                pairs,y=batch[0];logits=base.score(model,graphs[0]['x'],pairs)
                with torch.no_grad():teacher_logits=base.score(teacher,graphs[0]['x'],pairs)
                effective=F.binary_cross_entropy_with_logits(logits,y)+base.ranking_distillation(logits,teacher_logits,int(y.sum()))
                actions['A_with_rank']=gradients(model,effective);vectors['train_A_with_rank']=actions['A_with_rank']
            for left,right in [('train_A','train_B'),('train_A','val_A'),('train_A','val_B'),('train_B','val_A'),('train_B','val_B')]+([('train_A_with_rank','val_A'),('train_A_with_rank','val_B')] if weight else []):
                for gn,inds in group.items():grow.append(dict(state=state,step=step,trial=trial,group=gn,left=left,right=right,**geometry([vectors[left][i] for i in inds],[vectors[right][i] for i in inds])))
            if trial==0:actions['momentum_only']=[torch.zeros_like(p) for p in model.parameters()]
            for action,grad in actions.items():
                disposable,opt=snapshot_model(ck,device)
                for param,g in zip(disposable.parameters(),grad):param.grad=g.clone()
                preclip=float(torch.nn.utils.clip_grad_norm_(disposable.parameters(),1.));opt.step()
                delta=[p.detach()-q.detach() for p,q in zip(disposable.parameters(),model.parameters())]
                for i,(graph,ps) in enumerate(zip(graphs,parts)):
                    after,_=evaluate(disposable,graph,ps)
                    geo=geometry(vg[i],delta)
                    urows.append(dict(state=state,step=step,trial=trial,action=action,validation_source='A' if i==0 else 'B',
                        bce_before=reports[i]['bce'],bce_after=after['bce'],bce_delta=after['bce']-reports[i]['bce'],
                        auc_before=reports[i]['auc'],auc_after=after['auc'],auc_delta_pp=100*(after['auc']-reports[i]['auc']),
                        first_order_bce_delta=geo['dot'],update_norm=geo['norm_right'],validation_gradient_update_cosine=geo['cosine'],preclip_norm=preclip))
                del disposable,opt
        print(json.dumps(dict(case=args.case,state=state,step=step,trials=args.trials,validation=reports)),flush=True)
        del model;torch.cuda.empty_cache()
    for name,rows in [('gradients',grow),('updates',urows)]:
        with (out/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    atomic_json(out/'COMPLETE.json',dict(status='complete',case=args.case,source_A=a,source_B=b,trials=args.trials,checkpoints=receipts,
        protocol='fixed source validation seed0; 8 fresh training-batch probes per state; every AdamW action starts from identical saved weights and optimizer; no checkpoint writes'))
if __name__=='__main__':main()
