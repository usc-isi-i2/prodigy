"""Read-only source-validation checks for the fresh interleaved trajectory audit."""
import argparse,json
from pathlib import Path
import torch
from . import interleaved_mlp_pairs as base
from . import gradient_diagnostic as diagnostic
from .ranking_preservation import SOURCE_ROOT
from .config import load_config
from .mini_transfer import identity
from .evaluation_artifacts import atomic_json

PAIRS=(('ukr_rus_twitter','facebook_page_reference'),('covid19_twitter','midterm'),('covid_political','ukr_rus_suspended'))
JOINT=Path('/dataMeR1/phil/gfm/mixture-scaling/state/interleaved_mlp_pairs_noelec_s0/node_neighbors/lp')

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,choices=range(3),required=True);p.add_argument('--device',type=int,choices=range(4),required=True);p.add_argument('--output',required=True);p.add_argument('--config',default='configs/nonzero_mini_transfer.yaml');args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(f'cuda:{args.device}');sources=PAIRS[args.case]
    graphs=[base.load_graph(s,load_config(args.config),0,device) for s in sources]
    pairs=[diagnostic.validation_pairs(g) for g in graphs]
    run=JOINT/f'interleave_{sources[0]}__and__{sources[1]}'
    paths=[(f'single_{s}',SOURCE_ROOT/f'ss_{s}'/'best.pt') for s in sources]+[('joint_best',run/'best.pt'),('joint_final',run/'latest.pt')]
    rows=[]
    for name,path in paths:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        model,_=diagnostic.snapshot_model(ck,device)
        for s,g,ps in zip(sources,graphs,pairs):
            report,_=diagnostic.evaluate(model,g,ps)
            rows.append(dict(model=name,source=s,step=ck['step'],checkpoint=identity(path),**report))
        print(json.dumps(dict(completed=name)),flush=True)
        del model
    atomic_json(out,dict(case=args.case,sources=sources,rows=rows,protocol='unchanged singleton and fresh joint checkpoints; original fixed seed-0 source-validation pairs; no test evaluation'))
if __name__=='__main__':main()
