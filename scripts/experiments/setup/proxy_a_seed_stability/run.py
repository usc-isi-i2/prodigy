"""Three independent raw-feature proxy-A repeats; no GNN training."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/experiments/analysis/graphs/structure/graph_divergence'))
sys.path.insert(0, str(ROOT / 'scripts/experiments/analysis/graphs/structure_features/path_feature_coupling'))
from compute_graph_divergence import load_graph, sample_feature_rows, proxy_a_distance
from analyze_neighbor_augmented_features import sample_nonmissing_node_ids


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/dataMeR1/phil/data'))
    p.add_argument('--samples', type=int, default=4000)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    historical = json.loads((ROOT / 'scripts/experiments/analysis/graphs/structure/graph_divergence/data/graph_divergence_data.json').read_text())
    names = historical['graphs']
    plan = dict(seeds=[0,1,2], samples=args.samples, graphs=names,
                policies=['legacy_sorted', 'uniform'], pair_fits=3*2*36,
                feature_space='raw, no PCA or normalization',
                classifier='LogisticRegression C=1 max_iter=1000; stratified 70/30 split',
                note='Joint node-sampling and split variability; seeds do not replay historical split RNG. Each unordered pair fitted once.',
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    print(json.dumps(plan),flush=True)
    if args.dry_run: return
    import torch
    from sklearn.exceptions import ConvergenceWarning
    torch.set_num_threads(4)
    warnings.filterwarnings('error', category=ConvergenceWarning)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'protocol.json').write_text(json.dumps(plan,indent=2))
    samples = {}; audit=[]
    for gi,name in enumerate(names):
        path=args.data_root / historical['meta']['graph_paths'][name]
        graph=load_graph(path); x=graph['x']; n=len(x)
        for seed in plan['seeds']:
            graph_seed=int(np.random.SeedSequence([seed,gi,771]).generate_state(1)[0])
            for policy in plan['policies']:
                rng=np.random.default_rng(graph_seed)
                if policy=='legacy_sorted': rows,ids,_=sample_feature_rows(x,n,args.samples,rng)
                else: ids,rows=sample_nonmissing_node_ids(x,n,args.samples,rng)
                assert len(rows)==args.samples and len(np.unique(ids))==args.samples and np.isfinite(rows).all()
                samples[seed,policy,name]=rows
                np.savez_compressed(args.out/f'{name}_{policy}_s{seed}.npz',ids=ids,features=rows)
                audit.append(dict(graph=name,seed=seed,policy=policy,nodes=n,feature_shape=list(x.shape),artifact_bytes=path.stat().st_size,artifact_mtime_ns=path.stat().st_mtime_ns,mean_relative_id=float(np.mean(ids/n)),feature_sha256=hashlib.sha256(rows.tobytes()).hexdigest()))
        del x,graph;gc.collect()
        print('sampled',name,flush=True)
    (args.out/'sampling_audit.json').write_text(json.dumps(audit,indent=2))
    results=[]
    for seed in plan['seeds']:
        for policy in plan['policies']:
            matrix=np.zeros((len(names),len(names)))
            for i in range(len(names)):
                for j in range(i+1,len(names)):
                    rng=np.random.default_rng(np.random.SeedSequence([seed,i,j,991]))
                    distance=proxy_a_distance(samples[seed,policy,names[i]],samples[seed,policy,names[j]],rng)
                    assert distance is not None and np.isfinite(distance)
                    matrix[i,j]=matrix[j,i]=distance
            results.append(dict(seed=seed,policy=policy,proxy_a_distance=matrix.tolist()))
            (args.out/'results.json').write_text(json.dumps(dict(graphs=names,runs=results),indent=2))
            print('completed',seed,policy,flush=True)
    (args.out/'DONE').write_text('216 pair fits completed without convergence warnings\n')

if __name__=='__main__': main()
