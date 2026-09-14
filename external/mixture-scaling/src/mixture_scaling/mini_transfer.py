"""GPU-resident feature reconstruction matrix on verified nonzero graph views."""
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
from .model import MaskedFeatureMLP
from .node_only_transfer import save_checkpoint, seed_everything
from .ladder_tracking import tracked_run
from .evaluation_artifacts import atomic_json
from .lattice import SOURCE_ORDER


def identity(path):
    p = Path(path).resolve(); s = p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def partitions(n, seed=0):
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 104729))
    k = max(1, round(n * .15))
    return order[2*k:], order[k:2*k], order[:k]


def losses(model, x, generator):
    mask = torch.rand(x.shape, device=x.device, generator=generator) < .5
    mask[:, 0] |= ~mask.any(1)
    prediction = model(torch.where(mask, model.mask_token, x))
    target = x.masked_fill(~mask, 0)
    prediction = prediction.masked_fill(~mask, 0)
    cosine_error = (1 - F.cosine_similarity(prediction, target, dim=-1)).clamp_min(0)
    mse = ((prediction - target).square().sum(1) / mask.sum(1))
    return cosine_error.square(), mse, cosine_error


def load_x(path, device):
    raw = torch.load(path, map_location='cpu', weights_only=False)
    x = raw['x'] if isinstance(raw, dict) else raw.x
    if x.ndim != 2 or x.shape[1] != 768 or not torch.isfinite(x).all() or not (x != 0).any(1).all():
        raise ValueError('expected finite nonzero [N,768] features')
    return x.float().to(device)


def model_for(config, device):
    p = config['protocol']
    return MaskedFeatureMLP(768, p['hidden_dim'], p['output_dim'], 0).to(device)


@torch.no_grad()
def evaluate(model, x, nodes, seed, replicates=1, batch_size=4096):
    model.eval(); reports=[]
    for rep in range(replicates):
        generator = torch.Generator(device=x.device).manual_seed(seed + rep)
        sums = torch.zeros(3, device=x.device)
        for start in range(0, len(nodes), batch_size):
            values = losses(model, x[nodes[start:start+batch_size]], generator)
            sums += torch.stack([v.sum() for v in values])
        reports.append((sums / len(nodes)).cpu().tolist())
    model.train()
    values = np.asarray(reports)
    return dict(scaled_cosine_error_mean=float(values[:,0].mean()),
                scaled_cosine_error_std=float(values[:,0].std()),
                masked_mse_mean=float(values[:,1].mean()), cosine_error_mean=float(values[:,2].mean()),
                replicate_metrics=reports, n_nodes=len(nodes), mask_replicates=replicates)


def train(source, config, args, device):
    root=Path(args.root); run_id='ss_'+source; run_dir=root/'fp'/run_id
    if (run_dir/'summary.json').exists(): return
    if run_dir.exists(): raise FileExistsError(f'partial run: {run_dir}')
    path=config['graphs'][source]['path']; graph_id=identity(path)
    started=time.monotonic(); x=load_x(path,device)
    train_ids,val_ids,test_ids=partitions(len(x),args.seed)
    val_ids=val_ids[:20*1024].to(device); train_ids=train_ids.to(device)
    seed_everything(args.seed); model=model_for(config,device)
    p=config['protocol']; optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    generator=torch.Generator(device=device).manual_seed(args.seed+49979687)
    order_generator=torch.Generator(device=device).manual_seed(args.seed+7919)
    run_dir.mkdir(parents=True)
    metadata=dict(run_id=run_id,sources=[source],objective='fp',architecture='node_mlp',
        input_view='center_node_features_only',seed=args.seed,protocol=p,graph_identity=graph_id,
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        split='seeded disjoint node partitions: 70% train, 15% source validation, 15% final test',
        node_counts=dict(train=len(train_ids),validation=len(val_ids),test=len(test_ids)),
        objective_description='50% coordinate mask, squared cosine error on masked coordinates',
        checkpoint_selection='source validation only',mask_rng='seeded CUDA generator',
        precision='float32; TF32 disabled',stopping=dict(validation_interval=args.validation_interval,
        patience=args.patience,min_delta=1e-4,minimum_steps=2500,safety_cap=args.max_steps))
    atomic_json(run_dir/'metadata.json',metadata)
    best=float('inf'); reference=float('inf'); stale=0; best_step=0; offset=len(train_ids)
    loss_sum=torch.zeros((),device=device); window=0; reason='safety_cap'
    print(json.dumps(dict(starting=source,load_seconds=time.monotonic()-started)),flush=True)
    with tracked_run(run_dir,metadata,args) as (run,log):
        for step in range(1,args.max_steps+1):
            if offset>=len(train_ids):
                order=train_ids[torch.randperm(len(train_ids),generator=order_generator,device=device)]; offset=0
            ids=order[offset:offset+1024]; offset+=len(ids)
            optimizer.zero_grad(set_to_none=True)
            loss=losses(model,x[ids],generator)[0].mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); loss_sum+=loss.detach(); window+=1
            if step%args.log_interval==0:
                value=float(loss_sum/window)
                if not np.isfinite(value): raise FloatingPointError('non-finite training loss')
                log(step,{'train/loss':value,'train/elapsed_seconds':time.monotonic()-started})
                loss_sum.zero_(); window=0
            if step%args.validation_interval==0 or step==args.max_steps:
                report=evaluate(model,x,val_ids,args.seed+32452843,batch_size=1024)
                value=report['scaled_cosine_error_mean']
                if not np.isfinite(value): raise FloatingPointError('non-finite validation')
                if value < best:
                    best=value; best_step=step
                    save_checkpoint(run_dir/'best.pt',model,optimizer,step,metadata)
                if value < reference-1e-4: reference=value; stale=0
                elif step>=2500: stale+=1
                log(step,{'validation/loss':value,'validation/stale_checks':stale})
                print(json.dumps(dict(source=source,step=step,validation=value,stale=stale,
                    elapsed_seconds=time.monotonic()-started)),flush=True)
                save_checkpoint(run_dir/'latest.pt',model,optimizer,step,metadata)
                if stale>=args.patience: reason='validation_plateau'; break
        save_checkpoint(run_dir/'checkpoints'/f'step_{step}.pt',model,optimizer,step,metadata)
        summary=dict(metadata,status='complete',final_step=step,best_step=best_step,
            best_validation_loss=best,converged=reason=='validation_plateau',stop_reason=reason,
            elapsed_seconds=time.monotonic()-started)
        run.summary.update(dict(final_step=step,best_step=best_step,stop_reason=reason))
    atomic_json(run_dir/'summary.json',summary)
    print(json.dumps(dict(completed=source,**{k:summary[k] for k in ('final_step','best_step','elapsed_seconds')})),flush=True)


def worker(config,args,device):
    root=Path(args.root); claims=root/'_claims'; claims.mkdir(parents=True,exist_ok=True)
    for source in args.sources:
        with (claims/(source+'.lock')).open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: continue
            train(source,config,args,device)
        torch.cuda.empty_cache()


def eval_targets(config,args,device):
    root=Path(args.root)
    for target in SOURCE_ORDER[args.worker_index::args.workers]:
        path=config['graphs'][target]['path']; x=load_x(path,device)
        nodes=partitions(len(x),args.seed)[2].to(device)
        for source in SOURCE_ORDER:
            output=root/'results'/'fp'/f'ss_{source}__to__{target}.json'
            if output.exists(): continue
            checkpoint_path=root/'fp'/('ss_'+source)/'best.pt'
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            model=model_for(config,device); model.load_state_dict(checkpoint['pretrain_model'])
            report=evaluate(model,x,nodes,args.seed+80021,replicates=10)
            atomic_json(output,dict(status='complete',source=source,target=target,**report,
                checkpoint_step=checkpoint['step'],checkpoint_identity=identity(checkpoint_path),
                graph_identity=identity(path),split='final held-out 15% nodes; never used for checkpoint selection'))
        print(json.dumps(dict(evaluated_target=target)),flush=True)
        del x; torch.cuda.empty_cache()


def aggregate(args):
    root=Path(args.root); rows=[]
    for source in SOURCE_ORDER:
        for target in SOURCE_ORDER:
            rows.append(json.loads((root/'results'/'fp'/f'ss_{source}__to__{target}.json').read_text()))
    fields=['source','target','scaled_cosine_error_mean','masked_mse_mean','cosine_error_mean','checkpoint_step','n_nodes']
    with (root/'results'/'feature_reconstruction_matrix.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    atomic_json(root/'FR_COMPLETE.json',dict(status='complete',cells=len(rows)))


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['train','eval','aggregate'])
    p.add_argument('--config',default='configs/nonzero_mini_transfer.yaml'); p.add_argument('--root',required=True)
    p.add_argument('--device',type=int,choices=range(4),default=0)
    p.add_argument('--worker-index',type=int,default=0); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--sources',nargs='+',choices=SOURCE_ORDER,default=list(SOURCE_ORDER))
    p.add_argument('--seed',type=int,default=0); p.add_argument('--max-steps',type=int,default=100000)
    p.add_argument('--validation-interval',type=int,default=2000); p.add_argument('--patience',type=int,default=3)
    p.add_argument('--log-interval',type=int,default=100)
    p.add_argument('--wandb-mode',default='offline'); p.add_argument('--wandb-project',default='nonzero-mini-transfer')
    p.add_argument('--wandb-group',default=None); args=p.parse_args()
    if args.phase=='aggregate': aggregate(args); return
    torch.set_num_threads(4); torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(f'cuda:{args.device}'); config=load_config(args.config)
    config['protocol'].update(max_steps=args.max_steps, validation_interval=args.validation_interval, patience_evaluations=args.patience, minimum_steps=2500, node_test_fraction=.15, min_delta=1e-4, checkpoint_steps='best/latest each validation and terminal')
    (worker if args.phase=='train' else eval_targets)(config,args,device)

if __name__=='__main__': main()
