"""Does early orientation dependence identify later native ranking loss?

Saved predictions only. Retrospective, conditional diagnostic; not mediation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from .orientation_failure_risk import risk_summary


def main():
    import torch
    from .analyze_mixture_predictions import input_labels
    from .probe_cached_inputs import standardized_probe
    from .replay import batch_hash
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--reversal-root',type=Path,required=True)
    p.add_argument('--sources',required=True,help='Explicit discovery or validation sources; no default sweep')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError('New output and hidden GPUs required')
    torch.set_num_threads(2)
    done=json.loads((args.reversal_root/'DONE.json').read_text())
    if done['rows']!=432 or len(done['receipts'])!=36: raise ValueError('Completed reversal panel required')
    source_list=args.sources.split(',')
    if len(set(source_list))!=len(source_list): raise ValueError('Duplicate requested sources')
    metrics=json.loads((args.reversal_root/'results.json').read_text())
    rows=[]
    for stream in ('original','fresh'):
        ref=args.reference_root/stream/'twibot20'
        labels=input_labels(ref,'twibot20')
        y=labels['local_y'].numpy(); episodes=labels['episode_ids'].numpy()
        degrees=[]; cues=[]
        for bi,h in enumerate(labels['cache']['batch_sha256']):
            batch=torch.load(ref/'batches'/f'batch_{bi:03d}.pt',map_location='cpu',weights_only=False)
            if batch_hash(batch)!=h: raise ValueError('Input changed')
            g=batch[0]; q=batch[5].reshape(-1,2)[:,0].bool()
            deg=torch.bincount(g.edge_index[1],minlength=len(g.x))[g.ptr[:-1]]
            cue=standardized_probe(deg.float().log1p()[:,None],batch)
            degrees.append(deg[q]); cues.append(cue[:,1]-cue[:,0])
        degree=torch.cat(degrees).numpy(); cue=torch.cat(cues).numpy()
        for source in source_list:
            records=[r for r in metrics if r['stream']==stream and r['source']==source and
                     r['condition']=='intact' and r['decoder']=='full_model']
            if len(records)!=2 or {r['step'] for r in records}!={100,2500}:
                raise ValueError('Unique source endpoints required')
            logits={}
            for record in records:
                parts={c:[] for c in ('intact','support','query','both')}
                directory=args.reversal_root/stream/record['model_id']
                for bi,h in enumerate(labels['cache']['batch_sha256']):
                    item=torch.load(directory/f'batch_{bi:03d}.pt',map_location='cpu',weights_only=False)
                    if item['batch']!=bi or item['batch_sha256']!=h: raise ValueError('Prediction input mismatch')
                    for c in parts: parts[c].append(item['conditions'][c]['logits']['full_model'])
                logits[record['step']]={c:torch.cat(v) for c,v in parts.items()}
            probabilities={step:logits[step]['intact'].softmax(1)[:,1].numpy() for step in (100,2500)}
            margins={c:(v[:,1].double()-v[:,0].double()).numpy() for c,v in logits[100].items()}
            early_auc=[]; late_auc=[]; arrays={k:[] for k in ('margin','interaction','loss','degree_group','weights')}
            for ep in np.unique(episodes):
                indices=np.flatnonzero(episodes==ep)
                pos=indices[y[indices]==1]; neg=indices[y[indices]==0]
                def diff(x): return x[pos,None]-x[neg][None,:]
                ea=diff(probabilities[100]); la=diff(probabilities[2500])
                early_auc.append(float(((ea>0)+.5*(ea==0)).mean()))
                late_auc.append(float(((la>0)+.5*(la==0)).mean()))
                selected=ea>0
                d={c:diff(v) for c,v in margins.items()}
                interaction=(d['intact']+d['both']-d['support']-d['query'])/2
                equal=diff(degree)==0; cd=diff(cue)
                group=np.full(equal.shape,'cue_tied',dtype='<U16')
                group[equal]='equal'; group[~equal & (cd>0)]='cue_correct'; group[~equal & (cd<0)]='cue_wrong'
                values=dict(margin=d['intact'],interaction=interaction,
                    loss=1-((la>0)+.5*(la==0)),degree_group=group,
                    weights=np.full(ea.shape,1/(len(np.unique(episodes))*ea.size)))
                for key,value in values.items(): arrays[key].extend(value[selected].tolist())
            for step,got in ((100,np.mean(early_auc)),(2500,np.mean(late_auc))):
                expected=next(r['mean_episode_auc'] for r in records if r['step']==step)
                if abs(got-expected)>1e-12: raise ValueError('Native episode AUC not reproduced')
            result=risk_summary(**{k:np.asarray(v) for k,v in arrays.items()})
            row=dict(stream=stream,source=source,eligible_pairs=len(arrays['loss']),
                early_auc=float(np.mean(early_auc)),late_auc=float(np.mean(late_auc)),
                eligible_mass=float(np.sum(arrays['weights'])),analysis=result)
            rows.append(row)
    args.output.mkdir(parents=True)
    output=dict(rows=rows,new_model_forwards=0,new_training=False,
        interpretation='Retrospective early-correct pair risk, not causal mediation or independent pair replication',
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (args.output/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    for r in rows:
        print(json.dumps({k:v for k,v in r.items() if k!='analysis'}|{'contrasts':{
            key:{k:v for k,v in val.items() if k!='strata'} for key,val in r['analysis'].items()}}),flush=True)


if __name__=='__main__': main()
