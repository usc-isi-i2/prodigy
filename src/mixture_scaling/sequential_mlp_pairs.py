"""Ordered two-source continuation of the gallery's neighbor LP checkpoints."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import torch
from . import mini_lp as lp
from .bias_lp_matrix import BiasMLP, REFERENCE, evaluate
from .config import load_config
from .evaluation_artifacts import atomic_json
from .lattice import SOURCE_ORDER
from .mini_transfer import identity


def pair_rows():
    return [(f'seq_{a}__then__{b}', a, b) for a in SOURCE_ORDER for b in SOURCE_ORDER if a != b]


def preflight(args):
    rows=[]
    for run_id,a,b in pair_rows():
        cp=Path(args.source_root)/'node_neighbors/lp'/f'ss_{a}'/'best.pt'
        summary=json.loads((cp.parent/'summary.json').read_text())
        if summary['status']!='complete' or summary['seed']!=args.seed:
            raise ValueError(f'invalid first stage: {cp}')
        rows.append(dict(run_id=run_id,first=a,second=b,checkpoint=identity(cp),first_stage_converged=summary['converged']))
    return rows


def train(args,device):
    rows=preflight(args);config=load_config(args.config)
    config['protocol']['decoder']=dict(kind='dot_plus_learned_bias',transfer='frozen')
    prepare,features,score=lp.prepare,lp.view_features,lp.score
    lp.prepare=lambda source,config,root,seed:prepare(source,config,REFERENCE,seed)
    lp.view_features=lambda source,data,root,view,seed,device:features(source,data,REFERENCE,view,seed,device)
    lp.model_for=lambda view,device:BiasMLP(view).to(device)
    lp.score=lambda model,x,pairs:score(model,x,pairs)+model.decoder_bias
    claims=Path(args.root)/'_claims';claims.mkdir(parents=True,exist_ok=True)
    for row in rows:
        args.run_id=row['run_id'];args.warm_start=row['checkpoint']['path']
        with (claims/(args.run_id+'.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            run=Path(args.root)/'node_neighbors/lp'/args.run_id
            if (run/'summary.json').exists():
                old=json.loads((run/'summary.json').read_text())
                if old['warm_start']['checkpoint']!=row['checkpoint'] or old['seed']!=args.seed:
                    raise ValueError(f'completed run provenance mismatch: {run}')
                continue
            lp.train(row['second'],config,args,device)
        torch.cuda.empty_cache()


def aggregate(args):
    root=Path(args.root);rows=[]
    for run_id,a,b in pair_rows():
        for target in SOURCE_ORDER:
            d=json.loads((root/'results/node_neighbors_bias'/f'{run_id}__to__{target}.json').read_text())
            rows.append(dict(first=a,second=b,target=target,checkpoint_step=d['checkpoint_step'],**d['metrics']['test']))
    with (root/'results/matrix.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    atomic_json(root/'results/COMPLETE.json',dict(status='complete',sequential_models=72,evaluation_cells=len(rows)))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','train','eval','aggregate'])
    p.add_argument('--root',required=True);p.add_argument('--source-root',default='/dataMeR1/phil/gfm/mixture-scaling-bias-lp/state/bias_s0')
    p.add_argument('--config',default='configs/nonzero_mini_transfer.yaml');p.add_argument('--device',type=int,choices=range(4),default=0);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=100000);p.add_argument('--validation-interval',type=int,default=2000);p.add_argument('--patience',type=int,default=3);p.add_argument('--log-interval',type=int,default=100)
    p.add_argument('--wandb-mode',default='offline');p.add_argument('--wandb-project',default='nonzero-mini-transfer');p.add_argument('--wandb-group',default='sequential-mlp-pairs')
    p.add_argument('--prodigy-root',default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot')
    args=p.parse_args();args.view='node_neighbors';args.views=('node_neighbors',)
    args.model_rows=[(run_id,a+'->'+b) for run_id,a,b in pair_rows()]
    if args.phase=='plan':print(json.dumps(preflight(args),indent=2));return
    if args.phase=='aggregate':aggregate(args);return
    torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
    (train if args.phase=='train' else evaluate)(args,torch.device(f'cuda:{args.device}'))

if __name__=='__main__':main()
