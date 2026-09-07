"""Saved Hong Kong bot-ranking change by zero/nonzero query feature pairs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .analyze_mixture_predictions import input_labels
from .replay import batch_hash
from .role_context import query_mask


def accounting(y, zero, early, late):
    y,zero,early,late=map(np.asarray,(y,zero,early,late))
    if any(x.ndim!=1 or x.shape!=y.shape for x in (zero,early,late)) or set(y)!={0,1}:
        raise ValueError('Aligned binary episode required')
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    missing=zero[pos,None]|zero[neg][None,:]
    credit=[]
    for score in (early,late):
        d=score[pos,None]-score[neg][None,:]
        credit.append((d>0)+.5*(d==0))
    a,b=credit
    result=[]
    for name,mask in (('both_nonzero',~missing),('at_least_one_zero',missing)):
        result.append(dict(group=name,mass=float(mask.mean()),
            early=float(a[mask].sum()/a.size),late=float(b[mask].sum()/a.size),
            change=float((b[mask]-a[mask]).sum()/a.size)))
    assert abs(sum(r['change'] for r in result)-float((b-a).mean()))<1e-12
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or torch.cuda.is_available(): raise ValueError('New output and CPU required')
    torch.set_num_threads(2)
    rows=[]; population={}
    for stream in ('original','fresh'):
        root=args.reference_root/stream/'twibot20'
        labels=input_labels(root,'twibot20')
        zero=[]; support_fractions=[]
        for bi,h in enumerate(labels['cache']['batch_sha256']):
            batch=torch.load(root/'batches'/f'batch_{bi:03d}.pt',map_location='cpu',weights_only=False)
            if batch_hash(batch)!=h: raise ValueError('Cache changed')
            q=query_mask(batch); g=batch[0]
            z=(g.x[g.ptr[:-1]]==0).all(1)
            zero.append(z[q])
            for ep in range(4):
                support_fractions.append(float(z[(g.task_id_per_sample==ep)&~q].float().mean()))
        zero=torch.cat(zero).numpy()
        population[stream]=dict(query_zero_fraction=float(zero.mean()),
            mean_support_zero_fraction=float(np.mean(support_fractions)),
            episodes_without_zero_supports=int(np.sum(np.array(support_fractions)==0)))
        exports={}
        for step in (100,2500):
            path=root/f'corrected_ss_cp_hk_step{step}__baseline.pt'
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if len(saved)!=32: raise ValueError('Incomplete predictions')
            for bi,item in enumerate(saved):
                if item['batch']!=bi or item['batch_sha256']!=labels['cache']['batch_sha256'][bi]:
                    raise ValueError('Prediction identities changed')
            exports[step]={decoder:torch.cat([s['logits'][decoder] for s in saved]).softmax(1)[:,1].numpy()
                           for decoder in ('full_model','S0_pool/ridge')}
        y=labels['local_y'].numpy(); episodes=labels['episode_ids'].numpy()
        for decoder in exports[100]:
            parts=[]
            for ep in np.unique(episodes):
                mask=episodes==ep
                parts.extend(accounting(y[mask],zero[mask],exports[100][decoder][mask],exports[2500][decoder][mask]))
            for group in ('both_nonzero','at_least_one_zero'):
                selected=[r for r in parts if r['group']==group]
                r=dict(stream=stream,decoder=decoder,group=group,
                       **{k:float(np.mean([v[k] for v in selected])) for k in ('mass','early','late','change')})
                r['conditional_early']=r['early']/r['mass'] if r['mass'] else None
                r['conditional_late']=r['late']/r['mass'] if r['mass'] else None
                rows.append(r)
    args.output.mkdir(parents=True)
    result=dict(rows=rows,population=population,new_model_forwards=0,
        scope='Original HK bot episodes; query-pair accounting only, supports unchanged; equal episode weights')
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
