"""Matched warm-start interleaving with/without source-A ranking distillation."""
import argparse,csv,fcntl,json,math
from pathlib import Path
import torch
from . import interleaved_mlp_pairs as base
from .config import load_config
from .evaluation_artifacts import atomic_json
from .mini_transfer import identity

CASES=(('ukr_rus_twitter','facebook_page_reference'),('ukr_rus_suspended','covid_political'))
SOURCE_ROOT=Path('/dataMeR1/phil/gfm/mixture-scaling-bias-lp/state/bias_s0/node_neighbors/lp')

def rows():
    return [(f'warm_{a}__and__{b}__rank{weight}',a,b,weight) for a,b in CASES for weight in (0,1)]

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','train','eval','aggregate']);p.add_argument('--root',required=True)
    p.add_argument('--config',default='configs/nonzero_mini_transfer.yaml');p.add_argument('--device',type=int,choices=range(4),default=0);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=100000);p.add_argument('--validation-interval',type=int,default=2000);p.add_argument('--patience',type=int,default=3);p.add_argument('--log-interval',type=int,default=100)
    p.add_argument('--wandb-mode',default='offline');p.add_argument('--wandb-project',default='nonzero-mini-transfer');p.add_argument('--wandb-group',default='ranking-preservation')
    p.add_argument('--continuation-lr',type=float,default=None,help='override LR after restoring AdamW state')
    p.add_argument('--prodigy-root',default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot');args=p.parse_args()
    if any(v<=0 or v%2 for v in (args.max_steps,args.validation_interval,args.log_interval)) or args.patience<1:p.error('positive even step counts and positive patience required')
    if args.continuation_lr is not None and (not math.isfinite(args.continuation_lr) or args.continuation_lr<=0):p.error('continuation LR must be finite and positive')
    root=Path(args.root)
    if args.phase=='plan':
        print(json.dumps(dict(continuation_lr=args.continuation_lr,inputs=base.preflight(load_config(args.config)),runs=[dict(run_id=r,source_A=a,source_B=b,weight=w,checkpoint=identity(SOURCE_ROOT/f'ss_{a}'/'best.pt')) for r,a,b,w in rows()]),indent=2));return
    if args.phase=='aggregate':
        results=[]
        for run_id,a,b,w in rows():
            for target in base.SOURCES:
                d=json.loads((root/'results/node_neighbors_bias'/f'{run_id}__to__{target}.json').read_text())
                results.append(dict(first=a,second=b,rank_weight=w,target=target,checkpoint_step=d['checkpoint_step'],**d['metrics']['test']))
        with (root/'results/matrix.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=results[0]);writer.writeheader();writer.writerows(results)
        atomic_json(root/'results/COMPLETE.json',dict(status='complete',models=4,evaluation_cells=len(results)));return
    torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False;device=torch.device(f'cuda:{args.device}')
    if args.phase=='eval':
        args.views=('node_neighbors',);args.targets=base.SOURCES;args.model_rows=[(r,f'{a}+{b}:rank{w}') for r,a,b,w in rows()];base.evaluate(args,device);return
    config=load_config(args.config);base.preflight(config);claims=root/'_claims';claims.mkdir(parents=True,exist_ok=True)
    for run_id,a,b,w in rows():
        with (claims/(run_id+'.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            args.warm_start=str(SOURCE_ROOT/f'ss_{a}'/'best.pt');args.rank_weight=w
            base.train_one((run_id,a,b),config,args,device)
        torch.cuda.empty_cache()
if __name__=='__main__':main()
