"""Test the reviewer-proposed message-scale/normalization explanation directly."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
import torch_geometric

from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import input_labels,evaluate_logits,METRICS
from .message_content import aggregation_audit
from .message_scale import SCALES,forward_scaled,radial_swap,summarize_trace
from .replay import batch_hash
from .role_context import query_mask
from .run_episode_cardinality import make_model
from .verify_member_training import model_digest


def conditions():
    return ([{'name':f'scale_{a:g}','alpha':a} for a in SCALES]
            +[{'name':f'bias_{a:g}','alpha':a,'bias_only':True} for a in (.01,1.)]
            +[{'name':f'no_bn_{a:g}','alpha':a,'bypass_bn':True} for a in (0.,1.)]
            +[{'name':f'{stage}_{kind}','stage':stage,'kind':kind}
              for stage in ('post_relu','pre_meta') for kind in ('norm_only','direction_only')])


def merge_geometry(accum,key,row):
    if key not in accum:accum[key]=dict(row);return
    out=accum[key]
    for name,value in row.items():
        if name in ('stage','cohort'):continue
        if name=='count':out[name]+=value
        elif name.endswith('_min'):out[name]=min(out.get(name,value),value)
        elif name.endswith('_max') or name.startswith('max_'):out[name]=max(out.get(name,value),value)
        else:out[name]=out.get(name,0)+value


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference-replay',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--targets',nargs='+')
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--max-batches',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.output.exists() or torch.cuda.is_available() or not 1<=args.threads<=4 or not 1<=args.max_batches<=32:
        raise ValueError('new output, hidden GPUs and 1-4 CPU threads required')
    torch.set_num_threads(args.threads)
    read=lambda f:json.loads(f.read_text())
    prior=read(args.reference_replay/'protocol.json');done=read(args.reference_replay/'DONE.json')
    if done['smoke_only'] or not done['baseline_metrics_match'] or not done['all_weights_unchanged']:
        raise ValueError('completed historical topology reference required')
    targets=args.targets or prior['targets']
    if len(set(targets))!=len(targets) or not set(targets)<=set(prior['targets']):raise ValueError('invalid target set')
    if len(set(args.seeds))!=len(args.seeds) or not set(args.seeds)<={0,1,2}:raise ValueError('invalid seed set')
    arms=sorted([a for a in read(Path(prior['arms'])) if a['policy']=='lowest_sorted' and a['seed'] in args.seeds],key=lambda a:(a['source'],a['seed']))
    if len(arms)!=2*len(args.seeds) or {(a['source'],a['seed']) for a in arms}!={(s,n) for s in ('cp_hk','ukr_rus') for n in args.seeds}:
        raise ValueError('incomplete paired source/seed grid')
    reference={(r['target'],r['stream'],r['model_id'],r['support_condition']):r
               for r in read(args.reference_replay/'metrics.json')
               if r['query_condition']=='intact' and r['draw']==0 and r['support_condition'] in ('intact','removed')}
    old_receipts={(r['target'],r['stream'],r['model_id']):r for r in read(args.reference_replay/'receipts.json')}
    panel=conditions();expected=len(panel)*len(arms)*len(targets)*2
    partial=args.max_batches!=32 or set(targets)!=set(prior['targets']) or set(args.seeds)!={0,1,2}
    protocol={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'reference_replay':str(args.reference_replay),'targets':targets,'seeds':args.seeds,
        'models':[a['model_id'] for a in arms],'conditions':panel,'batches':args.max_batches,
        'expected_cells':expected,'partial_replay':partial,'new_training':False,
        'environment':{'torch':torch.__version__,'torch_geometric':torch_geometric.__version__,'threads':args.threads,'device':'cpu'},
        'scope':'Support-only interventions on the original six final checkpoints and fixed inputs. Exploratory response to reviewer, not independent confirmation or an outcome-selected remedy.',
        'scale':'Multiply projected W x + b before aggregation and message MLP; MLP(0), self projections and saved frozen BN remain at alpha=0.',
        'norm_swaps':'Paired intact/zero per-vector norm/direction swaps after node ReLU before pooling, and before the metagraph. Zero directions remain zero; unassignable nonzero norms are counted.',
        'bn_control':'Bypass only background BatchNorm on support nodes, retaining its saved tensors unchanged. Not retraining or normalization-free inference elsewhere.',
        'geometry':'Occurrence-weighted stage norms/directions, connected/isolated support receivers and query controls. No IID claims. No labels used for interventions.'}
    if args.dry_run:print(json.dumps(protocol,indent=2));return
    args.output.mkdir(parents=True);write_json(args.output/'protocol.json',protocol)
    started=time.monotonic();rows=[];receipts=[]
    with torch.no_grad():
        for target in targets:
            for stream in ('original','fresh'):
                root=Path(prior['cache_roots'][f'{target}/{stream}']);directory=root/target
                encoder_protocol=read(root/'protocol.json');labels=input_labels(directory,target)
                if encoder_protocol['eval_episode_seed_offset']!=(0 if stream=='original' else 100003):raise ValueError('wrong stream')
                first=torch.load(directory/'batches/batch_000.pt',map_location='cpu',weights_only=False)
                geometry=[]
                for arm in arms:
                    mid=arm['model_id'];common={'target':target,'stream':stream,'model_id':mid,'source':arm['source'],'seed':arm['seed']}
                    old=old_receipts[target,stream,mid]
                    if old['checkpoint']!=arm['checkpoint'] or old['weights_sha256']!=arm['final_sha256'] or old['episode_fingerprint']!=labels['cache']['episode_fingerprint']:
                        raise ValueError('checkpoint or cached input identity differs')
                    state=torch.load(arm['checkpoint'],map_location='cpu',weights_only=True)['model']
                    if model_digest(state)!=arm['final_sha256']:raise ValueError('changed checkpoint')
                    model=make_model(encoder_protocol,target,labels['cache']['graph_path'],first[0].x.shape[1],state)
                    aggregation=aggregation_audit(model)
                    if aggregation[0]['aggregation_module']!='SumAggregation':raise ValueError('reference aggregation changed')
                    predictions={c['name']:[] for c in panel};stats={};swaps={};query_checks=0
                    for bi in range(args.max_batches):
                        batch=torch.load(directory/'batches'/f'batch_{bi:03d}.pt',map_location='cpu',weights_only=False)
                        fingerprint=batch_hash(batch)
                        if fingerprint!=labels['cache']['batch_sha256'][bi]:raise ValueError('cached input changed')
                        q=query_mask(batch)
                        baseline,intact=forward_scaled(model,batch,alpha=1.)
                        removed,zero=forward_scaled(model,batch,alpha=0.)
                        for condition in panel:
                            name=condition['name']
                            if name=='scale_1':prediction,trace=baseline,intact
                            elif name=='scale_0':prediction,trace=removed,zero
                            elif 'stage' not in condition:
                                prediction,trace=forward_scaled(model,batch,**{k:v for k,v in condition.items() if k!='name'})
                            else:
                                stage=condition['stage'];key='post_relu' if stage=='post_relu' else 'U1_pre_meta'
                                mask=~q if stage=='pre_meta' else ~q[batch[0].batch]
                                direction,norm=(intact[key],zero[key]) if condition['kind']=='norm_only' else (zero[key],intact[key])
                                replacement=direction.clone();replacement[mask],audit=radial_swap(direction[mask],norm[mask])
                                for k,v in audit.items():swaps[name,k]=swaps.get((name,k),0)+v
                                prediction,trace=forward_scaled(model,batch,replace_stage=stage,replacement=replacement)
                                torch.testing.assert_close(trace[key][mask],replacement[mask],rtol=0,atol=0)
                            for stage in ('U1_pre_meta','M2_post_meta'):
                                torch.testing.assert_close(trace[stage][q],intact[stage][q],rtol=0,atol=0);query_checks+=1
                            predictions[name].append(prediction)
                            for row in summarize_trace(batch,trace,intact,zero):merge_geometry(stats,(name,row['stage'],row['cohort']),row)
                        if batch_hash(batch)!=fingerprint or model_digest(model.state_dict())!=arm['final_sha256'] or aggregation_audit(model)!=aggregation:
                            raise ValueError('model, operator or input changed')
                        if bi%8==7:print(json.dumps({**common,'batches_done':bi+1,'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
                    current={name:torch.cat(parts) for name,parts in predictions.items()}
                    eval_labels=dict(labels)
                    count=sum(labels['batch_counts'][:args.max_batches])
                    for key in ('local_y','mapping','episode_ids'):eval_labels[key]=labels[key][:count]
                    previous=torch.load(args.reference_replay/'predictions'/target/stream/f'{mid}.pt',map_location='cpu',weights_only=False)['logits']
                    max_error=0.
                    for name,condition in (('scale_1','intact'),('scale_0','removed')):
                        old_prediction=previous['intact',condition,0][:count]
                        torch.testing.assert_close(current[name],old_prediction,rtol=0,atol=1e-5)
                        max_error=max(max_error,float((current[name]-old_prediction).abs().max()))
                    for name,prediction in current.items():
                        scores=evaluate_logits(prediction,eval_labels)
                        if args.max_batches==32 and name in ('scale_1','scale_0'):
                            ref=reference[target,stream,mid,'intact' if name=='scale_1' else 'removed']
                            if max(abs(scores[k]-ref[k]) for k in METRICS)>1e-6:raise ValueError('historical endpoint metric mismatch')
                        rows.append({**common,'condition':name,**scores})
                    for (name,stage,cohort),values in stats.items():geometry.append({**common,'condition':name,**values})
                    out=args.output/'predictions'/target/stream;out.mkdir(parents=True,exist_ok=True)
                    torch.save({'logits':current,'labels':eval_labels},out/f'{mid}.pt')
                    receipts.append({**common,'batches':args.max_batches,'query_checks':query_checks,'query_occurrences':count,
                        'weights_sha256':arm['final_sha256'],'checkpoint':arm['checkpoint'],'episode_fingerprint':old['episode_fingerprint'],
                        'input_weights_and_operator_unchanged':True,'historical_endpoint_max_logit_error':max_error,
                        'historical_endpoint_metrics_match':args.max_batches==32,'aggregation':aggregation,
                        'radial_swaps':{name:{k:swaps[name,k] for k in ('vectors','zero_direction','nonzero_norm_unassignable')} for name in current if name.startswith(('post_relu','pre_meta'))}})
                    write_json(args.output/'metrics.json',rows);write_json(args.output/'receipts.json',receipts)
                    print(json.dumps({**common,'cells_done':len(rows),'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
                dest=args.output/'geometry';dest.mkdir(exist_ok=True)
                write_json(dest/f'{target}__{stream}.json',geometry)
    if len(rows)!=expected or len(receipts)!=len(arms)*len(targets)*2:raise ValueError('incomplete grid')
    write_json(args.output/'DONE.json',{'cells':len(rows),'model_target_streams':len(receipts),
        'partial_replay':partial,'all_inputs_weights_and_operators_unchanged':True,
        'historical_endpoint_metrics_match':args.max_batches==32,'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':main()
