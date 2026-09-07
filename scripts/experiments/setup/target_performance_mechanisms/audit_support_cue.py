"""Input-only agreement of original bot supports with the receive-edge cue."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .replay import batch_hash
from .role_context import query_mask


def describe(support_y, support_cue, query_y, query_cue):
    sy, sc, qy, qc = [np.asarray(x) for x in (support_y,support_cue,query_y,query_cue)]
    if any(x.ndim!=1 or not np.isin(x,[0,1]).all() for x in (sy,sc,qy,qc)):
        raise ValueError('Binary vectors required')
    if sy.shape!=sc.shape or qy.shape!=qc.shape or set(sy)!={0,1} or set(qy)!={0,1}:
        raise ValueError('Aligned binary classes required')
    agreement=float(np.mean(sy==sc))
    orientation=1 if agreement>.5 else -1 if agreement<.5 else 0
    scores=orientation*qc
    differences=scores[qy==1,None]-scores[qy==0][None,:]
    return dict(support_best_polarity_agreement=max(agreement,1-agreement),
        support_exception_rate=min(agreement,1-agreement),
        support_polarity=orientation, support_cue_positive_fraction=float(sc.mean()),
        query_cue_auc=float(((differences>0)+.5*(differences==0)).mean()),
        query_cue_positive_fraction=float(qc.mean()))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError('New output and hidden GPUs required')
    torch.set_num_threads(2)
    all_rows=[]
    for stream in ('original','fresh'):
        root=args.reference_root/stream/'twibot20'
        cache=json.loads((root/'cache.json').read_text())
        if len(cache['batch_sha256'])!=32: raise ValueError('Incomplete bank')
        for bi,h in enumerate(cache['batch_sha256']):
            b=torch.load(root/'batches'/f'batch_{bi:03d}.pt',map_location='cpu',weights_only=False)
            if batch_hash(b)!=h: raise ValueError('Input changed')
            g=b[0]; q=query_mask(b); y=b[2].argmax(1).numpy()
            cue=(torch.bincount(g.edge_index[1],minlength=len(g.x))[g.ptr[:-1]]>0).numpy()
            for ep in range(4):
                selected=(g.task_id_per_sample==ep).numpy()
                support=selected & ~q.numpy(); query=selected & q.numpy()
                all_rows.append(dict(stream=stream,episode=bi*4+ep,
                    **describe(y[support],cue[support],y[query],cue[query])))
    summaries={}
    for stream in ('original','fresh'):
        rows=[r for r in all_rows if r['stream']==stream]
        exceptions=np.array([r['support_exception_rate'] for r in rows])
        summaries[stream]=dict(episodes=len(rows),
            mean_exception_rate=float(exceptions.mean()),
            exception_quantiles=np.quantile(exceptions,[0,.25,.5,.75,1]).tolist(),
            no_exceptions=int((exceptions==0).sum()),
            tied_polarity=int((exceptions==.5).sum()),
            at_least_20_percent_exceptions=int((exceptions>=.2).sum()),
            mean_query_cue_auc=float(np.mean([r['query_cue_auc'] for r in rows])))
    args.output.mkdir(parents=True)
    output=dict(summaries=summaries,rows=all_rows,new_model_forwards=0,
        interpretation='Cue-label exceptions, not evidence of corrupted bot labels. Polarity chosen using supports only.')
    (args.output/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(summaries),flush=True)


if __name__=='__main__': main()
