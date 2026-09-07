"""One final-decoder orientation/strength analysis on matched saved episodes."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import evaluate_logits
from .contrast_metrics import contrast_parts,contrast_swaps,ranking_metrics
from .message_scale import forward_scaled
from .replay import batch_hash
from .role_context import query_mask
from .run_episode_cardinality import make_model
from .verify_member_training import model_digest

TARGETS=('covid_political','facebook_page_reference','twibot20')


def final_forward(model,batch,**options):
    saved=[]
    handle=model.final_label_mlp.register_forward_hook(lambda m,a,v:saved.append(v.detach().clone()))
    try:logits,trace=forward_scaled(model,batch,**options)
    finally:handle.remove()
    if len(saved)!=1 or batch[2].shape[1]!=2:raise ValueError('one final label projection and binary decoder required')
    q=query_mask(batch);x=trace['final_input'];n=len(x)
    left,right=(v.reshape(-1,2)[q] for v in batch[3])
    expected=torch.where(q)[0][:,None].expand(-1,2)
    if not torch.equal(left,expected) or torch.any(right<n):raise ValueError('decoder edge ordering differs')
    labels=saved[0][right-n]
    parts=contrast_parts(x[q].numpy(),labels.numpy(),float(model.logit_scale.exp()))
    error=float(np.max(np.abs(parts['margin']-(logits[:,1].double()-logits[:,0].double()).numpy())))
    if error>1e-4:raise ValueError(f'final contrast does not reproduce margins: {error}')
    return logits,trace,parts,error


def summarize_contrasts(intact,removed,labels,tau):
    args=(labels['local_y'].numpy(),labels['mapping'].numpy(),labels['episode_ids'].numpy(),labels['use_global'])
    margins=contrast_swaps(intact,removed,tau);rows={};episodes={}
    for condition,margin in margins.items():rows[condition],episodes[condition]=ranking_metrics(margin,*args)
    # Positive episode-constant strength cannot alter within-episode ranking.
    for a,b in [('orientation_only','removed'),('strength_only','intact'),('unit_strength_intact','intact'),('unit_strength_removed','removed')]:
        for x,y in zip(episodes[a],episodes[b]):
            if x['auc']!=y['auc']:raise ValueError(f'positive-scaling ranking invariant failed: {a}, {b}')
    if not np.array_equal(margins['strength_only']>0,margins['intact']>0):raise ValueError('positive scaling changed binary decisions')
    ids=labels['episode_ids'].numpy();geometry=[]
    for i,e in enumerate(np.unique(ids)):
        mask=ids==e
        for parts in (intact,removed):
            if not np.allclose(parts['strength'][mask],parts['strength'][mask][0],rtol=0,atol=1e-12):raise ValueError('contrast strength differs inside an episode')
            if not np.allclose(parts['orientation'][mask],parts['orientation'][mask][0],rtol=0,atol=1e-12):raise ValueError('contrast direction differs inside an episode')
        j=np.where(mask)[0][0]
        geometry.append({'episode':int(e),'intact_strength':float(intact['strength'][j]),'removed_strength':float(removed['strength'][j]),
            'strength_ratio':float(removed['strength'][j]/intact['strength'][j]),
            'contrast_orientation_cosine':float(np.dot(intact['orientation'][j],removed['orientation'][j])),
            'intact_class_cosine':float(intact['class_cosine'][j]),'removed_class_cosine':float(removed['class_cosine'][j]),
            'intact_episode_auc':episodes['intact'][i]['auc'],'removed_episode_auc':episodes['removed'][i]['auc']})
    return rows,geometry,margins


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--scale-root',type=Path,required=True)
    p.add_argument('--dose-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-batches',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.output.exists() or torch.cuda.is_available() or not 1<=args.threads<=4 or not 1<=args.max_batches<=32:raise ValueError('new output and bounded CPU replay required')
    torch.set_num_threads(args.threads)
    read=lambda p:json.loads(p.read_text())
    scale=read(args.scale_root/'protocol.json');topology=read(Path(scale['reference_replay'])/'protocol.json')
    if read(args.scale_root/'DONE.json').get('cells')!=900:raise ValueError('complete scale reference required')
    contracts=read(args.dose_root/'checkpoint_inventory.json')
    selected=[r for r in contracts if r['step']==2500 or (r['step']==100 and r['source']=='cp_hk')]
    plan=[(t,s,r) for t in TARGETS for s in ('original','fresh') for r in selected if r['step']==2500 or t=='facebook_page_reference']
    if len(plan)!=42:raise ValueError('expected 36 final and six early-page paired cells')
    protocol={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'paired_cells':42,'score_conditions':6,
        'targets':list(TARGETS),'batches':args.max_batches,'threads':args.threads,'smoke_only':args.max_batches!=32,
        'scale_root':str(args.scale_root),'dose_root':str(args.dose_root),'new_training':False,
        'design':'Intact versus zero support messages only. Six final checkpoints on CP/FB/BOT; Hong Kong step100 on FB connects the observed training-stage reversal. Fixed final queries; normalize final label vectors, form d=l1-l0, swap d orientation and strength at the score margin. Native logits must match prior saved values.',
        'interpretation':'Final score counterfactuals, not target-fitted classifiers or necessarily realizable latent label pairs. Positive contrast scaling preserves episode ranks and binary decisions. Means across training seeds and episode streams; no IID inference.'}
    if args.dry_run:print(json.dumps(protocol,indent=2));return
    args.output.mkdir(parents=True);write_json(args.output/'protocol.json',protocol)
    rows=[];geometry=[];receipts=[];start=time.monotonic()
    with torch.no_grad():
        for target,stream,rec in plan:
            mid=rec['model_id'];step=rec['step'];common={k:rec[k] for k in ('model_id','source','seed','step')};common.update(target=target,stream=stream)
            directory=Path(topology['cache_roots'][f'{target}/{stream}'])/target
            ep=read(directory.parent/'protocol.json')
            if step==2500:path=args.scale_root/'predictions'/target/stream/f'{mid}.pt';names=('scale_1','scale_0')
            else:path=args.dose_root/'predictions'/target/stream/f'{mid}__step{step}.pt';names=((0,0),(100,0))
            old=torch.load(path,map_location='cpu',weights_only=False);labels=dict(old['labels'])
            count=sum(labels['batch_counts'][:args.max_batches])
            for name in ('local_y','mapping','episode_ids'):labels[name]=labels[name][:count]
            state=torch.load(rec['checkpoint'],map_location='cpu',weights_only=True)['model']
            if model_digest(state)!=rec['weights_sha256']:raise ValueError('checkpoint changed')
            first=torch.load(directory/'batches/batch_000.pt',map_location='cpu',weights_only=False)
            model=make_model(ep,target,labels['cache']['graph_path'],first[0].x.shape[1],state)
            tau=float(model.logit_scale.exp());collected=[[],[]];logits=[[],[]];errors=[];checks=0
            for bi in range(args.max_batches):
                batch=torch.load(directory/'batches'/f'batch_{bi:03d}.pt',map_location='cpu',weights_only=False)
                if batch_hash(batch)!=labels['cache']['batch_sha256'][bi]:raise ValueError('cached inputs changed')
                a,ta,pa,ea=final_forward(model,batch,alpha=1.)
                b,tb,pb,eb=final_forward(model,batch,alpha=0.)
                for stage in ('U1_pre_meta','M2_post_meta','final_input'):
                    torch.testing.assert_close(ta[stage][query_mask(batch)],tb[stage][query_mask(batch)],rtol=0,atol=0);checks+=1
                for i,(value,parts) in enumerate(((a,pa),(b,pb))):logits[i].append(value);collected[i].append(parts)
                errors.extend([ea,eb])
            combined=[{k:np.concatenate([v[k] for v in items]) for k in items[0]} for items in collected]
            native={}
            for i,name in enumerate(names):
                logits[i]=torch.cat(logits[i]);torch.testing.assert_close(logits[i],old['logits'][name][:count],rtol=0,atol=0)
                native['intact' if i==0 else 'removed']=evaluate_logits(logits[i],labels)
            scores,geo,margins=summarize_contrasts(*combined,labels,tau)
            for condition,values in scores.items():rows.append({**common,'condition':condition,**values,**{'native_'+k:v for k,v in native.get(condition,{}).items()}})
            geometry.extend({**common,**g} for g in geo)
            if model_digest(model.state_dict())!=rec['weights_sha256']:raise ValueError('model changed')
            receipts.append({**common,'weights_sha256':rec['weights_sha256'],'episode_fingerprint':labels['cache']['episode_fingerprint'],
                'exact_query_checks':checks,'endpoint_logits_bit_exact':True,'max_decoder_margin_error':max(errors),'tau':tau})
            private=args.output/'predictions'/target/stream;private.mkdir(parents=True,exist_ok=True)
            torch.save({'margins':margins,'labels':labels},private/f'{mid}__step{step}.pt')
            for name,values in [('metrics',rows),('geometry',geometry),('receipts',receipts)]:write_json(args.output/f'{name}.json',values)
            print(json.dumps({**common,'paired_cells_done':len(receipts),'elapsed_seconds':time.monotonic()-start}),flush=True)
    write_json(args.output/'DONE.json',{'paired_cells':len(receipts),'score_cells':len(rows),'smoke_only':args.max_batches!=32,
        'all_endpoint_logits_bit_exact':True,'query_checks':sum(r['exact_query_checks'] for r in receipts),
        'max_decoder_margin_error':max(r['max_decoder_margin_error'] for r in receipts),'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':main()
