"""Bias-only LP reruns with frozen-bias, identical-pair transfer evaluation."""
import argparse,csv,fcntl,json,math
from pathlib import Path
import numpy as np
import torch
from . import mini_lp as lp
from .model import NodeMLP
from .config import load_config
from .lattice import SOURCE_ORDER
from .mini_transfer import identity
from .binary_metrics import binary_report,evaluation_report
from .evaluation_artifacts import atomic_json,save_scores
from .evaluate_lattice import load_pair_module
REFERENCE=Path('/dataMeR1/phil/gfm/mixture-scaling-disjoint-neighbor/state/disjoint_context_s0')
PAIRS=Path('/dataMeR1/phil/gfm/mixture-scaling-uniform-eval/state/uniform_eval_s0_v2/node_neighbors_disjoint_uniform')
VIEWS=('node','node_neighbors')
class BiasMLP(NodeMLP):
    def __init__(self,view):
        super().__init__(768 if view=='node' else 1536,256,256,0)
        self.decoder_bias=torch.nn.Parameter(torch.tensor(-math.log(5.)))

def training(args,device):
    config=load_config(args.config)
    config['protocol']['decoder']=dict(kind='dot_plus_learned_bias',initial_bias=-math.log(5.),scale=1.,transfer='freeze source bias; no target adaptation')
    original_prepare=lp.prepare;original_features=lp.view_features;original_score=lp.score
    lp.prepare=lambda source,config,root,seed:original_prepare(source,config,REFERENCE,seed)
    lp.view_features=lambda source,data,root,view,seed,device:original_features(source,data,REFERENCE,view,seed,device)
    lp.model_for=lambda view,device:BiasMLP(view).to(device)
    lp.score=lambda model,x,pairs:original_score(model,x,pairs)+model.decoder_bias
    claims=Path(args.root)/'_claims';claims.mkdir(parents=True,exist_ok=True)
    for source in SOURCE_ORDER:
        for view in VIEWS:
            with (claims/f'{view}_{source}.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                args.view=view;lp.train(source,config,args,device)
            torch.cuda.empty_cache()

@torch.no_grad()
def evaluate(args,device):
    pair_module=load_pair_module(Path(args.prodigy_root));root=Path(args.root)
    for target in SOURCE_ORDER[args.device::4]:
        data=torch.load(REFERENCE/'_cache'/f'{target}.pt',map_location='cpu',weights_only=False)
        ref=PAIRS/f'ss_{SOURCE_ORDER[0]}__to__{target}.scores.npz';a=np.load(ref)
        pairs=pair_module.PairSet(u=a['u'],v=a['v'],label=a['labels'],negative_kind='uniform_both_endpoints_exact_nonedge');nodes=pairs.nodes()
        val=a['validation_mask'];balanced=a['balanced_test_indices'];n=len(data['x'])
        assert not lp.ExactNonedges(n,data['known_keys']).forbidden(torch.from_numpy(np.stack((pairs.u[pairs.label==0],pairs.v[pairs.label==0])))).any()
        for view in getattr(args,'views',VIEWS):
            x=lp.view_features(target,data,REFERENCE,view,args.seed,device)
            for run_id,source in getattr(args,'model_rows',[(f'ss_{s}',s) for s in SOURCE_ORDER]):
                cp=root/view/'lp'/run_id/'best.pt';dest=root/'results'/(view+'_bias')/f'{run_id}__to__{target}.json'
                provenance=dict(checkpoint=identity(cp),pair_reference=identity(ref),split=data['receipt'],decoder='dot_plus_learned_bias',bias_policy='frozen source checkpoint bias')
                if dest.exists():
                    assert json.loads(dest.read_text())['provenance']==provenance
                    continue
                ck=torch.load(cp,map_location='cpu',weights_only=False);model=BiasMLP(view).to(device).eval();model.load_state_dict(ck['model'])
                z=torch.cat([model(x[nodes[i:i+4096]]) for i in range(0,len(nodes),4096)]).cpu().numpy()
                emb=pair_module.NodeEmbeddings(z,nodes,n);dot=pair_module.pair_scores(emb,pairs,'dot');cos=pair_module.pair_scores(emb,pairs,'cosine');bias=float(model.decoder_bias);logits=dot+bias
                report=evaluation_report(pairs.label,logits,val,cos);report['balanced_test']=binary_report(pairs.label[balanced],logits[balanced]);report['baselines']['training_prior_1_6']=binary_report(pairs.label[~val],np.full(sum(~val),-math.log(5.)))
                dest.parent.mkdir(parents=True,exist_ok=True)
                save_scores(dest.with_suffix('.scores.npz'),u=pairs.u,v=pairs.v,labels=pairs.label,dot_logits=logits,raw_dot_scores=dot,cosine_scores=cos,validation_mask=val,balanced_test_indices=balanced)
                atomic_json(dest,dict(status='complete',source=source,target=target,view=view,checkpoint_step=ck['step'],decoder_bias=bias,metrics=report,provenance=provenance,score_kind='dot_plus_frozen_source_bias',negative_ratio=5,known_edges_in_negatives=0))
            del x;torch.cuda.empty_cache()
        print(json.dumps(dict(completed_target=target)),flush=True)

def aggregate(args):
    root=Path(args.root);hist={}
    for view in VIEWS:
        stage=view+'_bias';rows=[];summaries=[];hist[stage]={}
        for source in SOURCE_ORDER:
            run=root/view/'lp'/f'ss_{source}';summary=json.loads((run/'summary.json').read_text());summaries.append(summary)
            hist[stage]['ss_'+source]=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
            for target in SOURCE_ORDER:
                d=json.loads((root/'results'/stage/f'ss_{source}__to__{target}.json').read_text());m=d['metrics']
                rows.append(dict(source=source,target=target,auc=m['test']['roc_auc'],bce=m['test']['bce'],balanced_bce=m['balanced_test']['bce'],calibrated_bce=m['calibrated_test']['bce'],average_precision=m['test']['average_precision'],checkpoint_step=d['checkpoint_step'],decoder_bias=d['decoder_bias']))
        with (root/'results'/stage/'matrix.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        atomic_json(root/'results'/stage/'summaries.json',summaries)
    atomic_json(root/'results'/'bias_curves.json',hist);atomic_json(root/'results'/'COMPLETE.json',dict(status='complete',trained_models=18,cells=162))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','eval','aggregate']);p.add_argument('--root',required=True);p.add_argument('--device',type=int,choices=range(4),default=0);p.add_argument('--config',default='configs/nonzero_mini_transfer.yaml');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=100000);p.add_argument('--validation-interval',type=int,default=2000);p.add_argument('--patience',type=int,default=3);p.add_argument('--log-interval',type=int,default=100)
    p.add_argument('--wandb-mode',default='offline');p.add_argument('--wandb-project',default='nonzero-mini-transfer');p.add_argument('--wandb-group',default='bias-lp-matrices');p.add_argument('--prodigy-root',default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot')
    args=p.parse_args()
    if args.phase=='aggregate':aggregate(args);return
    torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
    (training if args.phase=='train' else evaluate)(args,torch.device(f'cuda:{args.device}'))
if __name__=='__main__':main()
