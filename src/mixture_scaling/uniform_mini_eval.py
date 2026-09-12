"""Reevaluate frozen mini LP models using the training negative distribution."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import torch
from .mini_lp import ExactNonedges,model_for,view_features
from .mini_transfer import identity
from .lattice import SOURCE_ORDER
from .evaluate_lattice import load_pair_module
from .binary_metrics import evaluation_report,binary_report
from .evaluation_artifacts import atomic_json,save_scores


def uniform_pairs(positive_u,positive_v,positive_val,n,known_keys,seed,ratio=5):
    sampler=ExactNonedges(n,known_keys.cpu())
    generator=torch.Generator().manual_seed(seed+67867967)
    negative=sampler.sample(len(positive_u)*ratio,generator).numpy()
    rng=np.random.default_rng(seed+86028121)
    count=negative.shape[1];negative_val=np.zeros(count,dtype=bool)
    negative_val[rng.permutation(count)[:int(np.sum(positive_val))*ratio]]=True
    u=np.concatenate((positive_u,negative[0]));v=np.concatenate((positive_v,negative[1]))
    y=np.concatenate((np.ones(len(positive_u),dtype=np.int64),np.zeros(count,dtype=np.int64)))
    val=np.concatenate((positive_val,negative_val))
    assert not sampler.forbidden(torch.from_numpy(negative)).any()
    return u,v,y,val


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    device=torch.device(f'cuda:{args.device}');torch.backends.cuda.matmul.allow_tf32=False
    pair_module=load_pair_module(Path(args.prodigy_root));reference=Path(args.reference_root)
    output=Path(args.output_root)
    for target in SOURCE_ORDER[args.worker_index::args.workers]:
        data=torch.load(reference/'_cache'/f'{target}.pt',map_location='cpu',weights_only=False)
        n=len(data['x']);ref_scores=reference/'node_neighbors/results'/f'ss_{SOURCE_ORDER[0]}__to__{target}.scores.npz'
        original=np.load(ref_scores);pos=original['labels']==1
        u,v,y,val=uniform_pairs(original['u'][pos],original['v'][pos],original['validation_mask'][pos],n,data['known_keys'],args.seed)
        pairs=pair_module.PairSet(u=u,v=v,label=y,negative_kind='uniform_both_endpoints_exact_nonedge')
        nodes=pairs.nodes()
        # A deterministic 1:1 subset for class-balance comparison, without refitting calibration.
        balanced=np.flatnonzero((~val)&(y==1))
        balanced=np.concatenate((balanced,np.flatnonzero((~val)&(y==0))[:len(balanced)]))
        assert len(balanced)==2800 and sum((~val)&(y==0))==7000
        for view,root,stage in [('node',Path(args.node_root),'node_uniform'),('node_neighbors',reference,'node_neighbors_disjoint_uniform')]:
            x=view_features(target,data,reference,view,args.seed,device)
            for source in SOURCE_ORDER:
                dest=output/stage/f'ss_{source}__to__{target}.json'
                cp=root/view/'lp'/f'ss_{source}'/'best.pt'
                provenance=dict(checkpoint=identity(cp),reference_pairs=identity(ref_scores),graph_split=data['receipt'],
                                negative_kind='uniform_both_endpoints_exact_nonedge',negative_ratio=5,seed=args.seed)
                if dest.exists():
                    old=json.loads(dest.read_text())
                    if old['provenance']!=provenance:raise ValueError('existing evaluation provenance mismatch')
                    continue
                checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
                model=model_for(view,device).eval();model.load_state_dict(checkpoint['model'])
                z=torch.cat([model(x[nodes[start:start+4096]]) for start in range(0,len(nodes),4096)]).cpu().numpy()
                embeddings=pair_module.NodeEmbeddings(z,nodes,n)
                logits=pair_module.pair_scores(embeddings,pairs,'dot');cosine=pair_module.pair_scores(embeddings,pairs,'cosine')
                report=evaluation_report(y,logits,val,cosine)
                report['balanced_test']=binary_report(y[balanced],logits[balanced])
                prior_logit=-np.log(5.)
                report['baselines']['training_prior_1_6']=binary_report(y[~val],np.full(sum(~val),prior_logit))
                dest.parent.mkdir(parents=True,exist_ok=True)
                save_scores(dest.with_suffix('.scores.npz'),u=u,v=v,labels=y,dot_logits=logits,cosine_scores=cosine,validation_mask=val,balanced_test_indices=balanced)
                atomic_json(dest,dict(status='complete',source=source,target=target,view=view,checkpoint_step=checkpoint['step'],
                    metrics=report,provenance=provenance,negative_kind='uniform_both_endpoints_exact_nonedge',negative_ratio=5,
                    positive_pairs_and_positive_validation_mask_unchanged=True,known_edges_in_negatives=0,
                    notes='Frozen checkpoint. Uniform independent endpoints, all known undirected edges/self-loops rejected. Test 1400 positive + 7000 negative; calibration 600 + 3000.'))
            del x;torch.cuda.empty_cache()
        print(json.dumps(dict(completed_target=target)),flush=True)


def aggregate(args):
    root=Path(args.output_root)
    for stage in ['node_uniform','node_neighbors_disjoint_uniform']:
        rows=[]
        for source in SOURCE_ORDER:
            for target in SOURCE_ORDER:
                p=json.loads((root/stage/f'ss_{source}__to__{target}.json').read_text());m=p['metrics']
                rows.append(dict(source=source,target=target,auc=m['test']['roc_auc'],bce=m['test']['bce'],
                    balanced_bce=m['balanced_test']['bce'],calibrated_bce=m['calibrated_test']['bce'],
                    average_precision=m['test']['average_precision'],checkpoint_step=p['checkpoint_step']))
        with (root/stage/'matrix.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(root/'COMPLETE.json',dict(status='complete',cells=162,negative_kind='uniform_both_endpoints_exact_nonedge',negative_ratio=5))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['eval','aggregate']);p.add_argument('--output-root',required=True)
    p.add_argument('--node-root',default='/dataMeR1/phil/gfm/mixture-scaling-mini-lp/state/lp_nonzero_walk1_s0')
    p.add_argument('--reference-root',default='/dataMeR1/phil/gfm/mixture-scaling-disjoint-neighbor/state/disjoint_context_s0')
    p.add_argument('--prodigy-root',default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot')
    p.add_argument('--device',type=int,choices=range(4),default=0);p.add_argument('--worker-index',type=int,default=0)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=0);args=p.parse_args()
    (evaluate if args.phase=='eval' else aggregate)(args)
if __name__=='__main__':main()
