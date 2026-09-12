"""Equal-update interleaved LP pairs, jointly selected on source validation."""
import argparse
import copy
import csv
import fcntl
import itertools
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from . import mini_lp as lp
from .bias_lp_matrix import BiasMLP, REFERENCE, PAIRS, evaluate
from .binary_metrics import binary_report
from .config import load_config
from .evaluation_artifacts import atomic_json
from .lattice import SOURCE_ORDER
from .mini_transfer import identity
from .node_only_transfer import save_checkpoint, seed_everything
from .ladder_tracking import tracked_run

SOURCES=tuple(s for s in SOURCE_ORDER if s!='election2020')


def pair_rows():
    return [(f'interleave_{a}__and__{b}',a,b) for a,b in itertools.combinations(SOURCES,2)]


def preflight(config):
    return {s:dict(graph=identity(config['graphs'][s]['path']),
                   split=identity(REFERENCE/'_cache'/f'{s}.pt'),
                   context=identity(REFERENCE/'_cache'/f'{s}_neighbors10_disjoint50.pt'),
                   evaluation_pairs=identity(PAIRS/f'ss_{SOURCE_ORDER[0]}__to__{s}.scores.npz')) for s in SOURCES}


def score(model,x,pairs):
    return lp.score(model,x,pairs)+model.decoder_bias


@torch.no_grad()
def validate(model,graph,seed):
    model.eval();x=graph['x'];generator=torch.Generator(device=x.device).manual_seed(seed+32452843)
    positive=graph['validation'];logits=[];labels=[]
    for start in range(0,min(positive.shape[1],20*1024),1024):
        pairs,y=lp.pair_batch(positive[:,start:start+1024],graph['sampler'],generator)
        logits.append(score(model,x,pairs));labels.append(y)
    result=binary_report(torch.cat(labels).cpu().numpy(),torch.cat(logits).cpu().numpy())
    model.train();return result


def load_graph(source,config,seed,device):
    data=lp.prepare(source,config,REFERENCE,seed)
    return dict(x=lp.view_features(source,data,REFERENCE,'node_neighbors',seed,device),
                positive=data['supervision'].to(device),validation=data['validation'].to(device),
                sampler=lp.ExactNonedges(len(data['x']),data['known_keys'].to(device)),receipt=data['receipt'])


def ranking_distillation(student,teacher,n_positive):
    # Each positive is compared with five sampled nonedges; preserve teacher preferences.
    def differences(scores):
        return scores[:n_positive,None]-scores[n_positive:].reshape(n_positive,5)
    target=torch.sigmoid(differences(teacher.detach()))
    logits=differences(student)
    entropy=F.binary_cross_entropy_with_logits(differences(teacher.detach()),target)
    return F.binary_cross_entropy_with_logits(logits,target)-entropy


def train_one(row,config,args,device):
    run_id,a,b=row;run_dir=Path(args.root)/'node_neighbors/lp'/run_id
    expected=dict(sources=[a,b],seed=args.seed,schedule='alternating_equal_updates',initialization='fresh',
                  max_steps=args.max_steps,validation_interval=args.validation_interval,patience=args.patience,
                  min_delta=1e-4,minimum_steps=2500)
    warm_start=getattr(args,'warm_start',None);rank_weight=getattr(args,'rank_weight',0.)
    if warm_start:
        expected.update(initialization='singleton_checkpoint',checkpoint=identity(warm_start),rank_weight=rank_weight,optimizer_policy='preserve')
    if (run_dir/'summary.json').exists():
        summary=json.loads((run_dir/'summary.json').read_text())
        if summary['run_protocol']!=expected:raise ValueError(f'protocol mismatch: {run_dir}')
        return
    if run_dir.exists():raise FileExistsError(f'partial run: {run_dir}')
    started=time.monotonic();graphs=[load_graph(s,config,args.seed,device) for s in (a,b)]
    seed_everything(args.seed);model=BiasMLP('node_neighbors').to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=1e-5)
    teacher=None
    if warm_start:
        checkpoint=torch.load(warm_start,map_location='cpu',weights_only=False)
        if checkpoint['metadata']['sources']!=[a] or checkpoint['metadata']['seed']!=args.seed:
            raise ValueError('initial checkpoint must match first source and seed')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        if rank_weight:
            teacher=copy.deepcopy(model).eval().requires_grad_(False)
    for g in graphs:
        g.update(generator=torch.Generator(device=device).manual_seed(args.seed+49979687),
                 order_generator=torch.Generator(device=device).manual_seed(args.seed+7919),offset=g['positive'].shape[1])
    metadata=dict(run_id=run_id,sources=[a,b],seed=args.seed,objective='lp',architecture='node_mlp',view='node_neighbors',
        run_protocol=expected,protocol=dict(config['protocol'],input_dim=1536,hidden_dim=256,output_dim=256,dropout=0.,
            fanout=10,max_steps=args.max_steps,validation_interval=args.validation_interval,learning_rate=.0005),
        graph_split_receipts={s:g['receipt'] for s,g in zip((a,b),graphs)},
        checkpoint_selection='minimum equal-weight mean of A and B source validation BCE',
        optimizer_policy='one shared AdamW, no resets',context='fixed up-to-10 neighbor mean, context disjoint from supervision',
        training_negative_sampling='1:5 uniform independent endpoints excluding every known edge and self-loops')
    run_dir.mkdir(parents=True);atomic_json(run_dir/'metadata.json',metadata)
    totals=torch.zeros(2,device=device);counts=[0,0];updates=[0,0]
    best=reference=float('inf');best_step=stale=0;reason='safety_cap'
    print(json.dumps(dict(starting=run_id)),flush=True)
    with tracked_run(run_dir,metadata,args) as (run,log):
        if warm_start:
            initial=[validate(model,g,args.seed) for g in graphs]
            best=reference=sum(r['bce'] for r in initial)/2
            save_checkpoint(run_dir/'best.pt',model,optimizer,0,metadata)
            atomic_json(run_dir/'validation'/'step_0.json',dict(mean_bce=best,per_source=dict(zip((a,b),initial))))
            log(0,{'validation/loss':best,'validation/loss_A':initial[0]['bce'],'validation/loss_B':initial[1]['bce']})
        rank_total=0.;rank_count=0
        for step in range(1,args.max_steps+1):
            index=(step-1)%2;g=graphs[index];positive=g['positive']
            if g['offset']>=positive.shape[1]:
                g['order']=torch.randperm(positive.shape[1],generator=g['order_generator'],device=device);g['offset']=0
            pos=positive[:,g['order'][g['offset']:g['offset']+1024]];g['offset']+=pos.shape[1]
            pairs,y=lp.pair_batch(pos,g['sampler'],g['generator'])
            optimizer.zero_grad(set_to_none=True);logits=score(model,g['x'],pairs)
            task_loss=F.binary_cross_entropy_with_logits(logits,y);loss=task_loss
            if teacher is not None and index==0:
                with torch.no_grad():teacher_scores=score(teacher,g['x'],pairs)
                penalty=ranking_distillation(logits,teacher_scores,pos.shape[1])
                loss=loss+rank_weight*penalty
                rank_total+=float(penalty.detach());rank_count+=1
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            totals[index]+=task_loss.detach();counts[index]+=1;updates[index]+=1
            if step%args.log_interval==0 or step==args.max_steps:
                values=[float(totals[i]/counts[i]) for i in (0,1)]
                if not all(torch.isfinite(torch.tensor(values))):raise FloatingPointError('nonfinite loss')
                log(step,{'train/loss':sum(values)/2,'train/loss_A':values[0],'train/loss_B':values[1],
                          'train/updates_A':updates[0],'train/updates_B':updates[1],'train/elapsed_seconds':time.monotonic()-started})
                if rank_count:log(step,{'train/ranking_kl':rank_total/rank_count,'train/rank_weight':rank_weight})
                rank_total=0.;rank_count=0
                totals.zero_();counts=[0,0]
            if step%args.validation_interval==0 or step==args.max_steps:
                reports=[validate(model,g,args.seed) for g in graphs];value=sum(r['bce'] for r in reports)/2
                if not torch.isfinite(torch.tensor(value)):raise FloatingPointError('nonfinite validation')
                if value<best:
                    best=value;best_step=step;save_checkpoint(run_dir/'best.pt',model,optimizer,step,metadata)
                if value<reference-1e-4:reference=value;stale=0
                elif step>=2500:stale+=1
                log(step,{'validation/loss':value,'validation/loss_A':reports[0]['bce'],'validation/loss_B':reports[1]['bce'],
                          'validation/auc_A':reports[0]['roc_auc'],'validation/auc_B':reports[1]['roc_auc'],'validation/stale_checks':stale})
                atomic_json(run_dir/'validation'/f'step_{step}.json',dict(mean_bce=value,per_source=dict(zip((a,b),reports))))
                save_checkpoint(run_dir/'latest.pt',model,optimizer,step,metadata)
                print(json.dumps(dict(run_id=run_id,step=step,validation=value,validation_A=reports[0]['bce'],validation_B=reports[1]['bce'],stale=stale,elapsed_seconds=time.monotonic()-started)),flush=True)
                if stale>=args.patience:reason='validation_plateau';break
        save_checkpoint(run_dir/'checkpoints'/f'step_{step}.pt',model,optimizer,step,metadata)
        summary=dict(metadata,status='complete',final_step=step,best_step=best_step,best_validation_bce=best,
                     updates_per_source=dict(zip((a,b),updates)),stop_reason=reason,converged=reason=='validation_plateau',elapsed_seconds=time.monotonic()-started)
        run.summary.update(dict(final_step=step,best_step=best_step,stop_reason=reason))
    atomic_json(run_dir/'summary.json',summary)


def train(args,device):
    config=load_config(args.config);preflight(config)
    claims=Path(args.root)/'_claims';claims.mkdir(parents=True,exist_ok=True)
    for row in pair_rows():
        with (claims/(row[0]+'.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            train_one(row,config,args,device)
        torch.cuda.empty_cache()


def aggregate(args):
    root=Path(args.root);rows=[]
    for run_id,a,b in pair_rows():
        for target in SOURCES:
            d=json.loads((root/'results/node_neighbors_bias'/f'{run_id}__to__{target}.json').read_text())
            rows.append(dict(first=a,second=b,target=target,checkpoint_step=d['checkpoint_step'],**d['metrics']['test']))
    with (root/'results/matrix.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    atomic_json(root/'results/COMPLETE.json',dict(status='complete',interleaved_models=len(pair_rows()),evaluation_cells=len(rows)))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','train','eval','aggregate']);p.add_argument('--root',required=True)
    p.add_argument('--config',default='configs/nonzero_mini_transfer.yaml');p.add_argument('--device',type=int,choices=range(4),default=0);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=100000);p.add_argument('--validation-interval',type=int,default=2000);p.add_argument('--patience',type=int,default=3);p.add_argument('--log-interval',type=int,default=100)
    p.add_argument('--wandb-mode',default='offline');p.add_argument('--wandb-project',default='nonzero-mini-transfer');p.add_argument('--wandb-group',default='interleaved-mlp-pairs')
    p.add_argument('--prodigy-root',default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot');args=p.parse_args()
    if any(v<=0 or v%2 for v in (args.max_steps,args.validation_interval,args.log_interval)) or args.patience<1:
        p.error('step counts/intervals must be positive and even; patience must be positive')
    args.views=('node_neighbors',);args.targets=SOURCES;args.model_rows=[(r,a+'+'+b) for r,a,b in pair_rows()]
    if args.phase=='plan':print(json.dumps(dict(pairs=pair_rows(),inputs=preflight(load_config(args.config))),indent=2));return
    if args.phase=='aggregate':aggregate(args);return
    torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
    (train if args.phase=='train' else evaluate)(args,torch.device(f'cuda:{args.device}'))

if __name__=='__main__':main()
