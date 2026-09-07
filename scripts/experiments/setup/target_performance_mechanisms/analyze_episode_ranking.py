"""Read existing scale and training-step logits; no model forward or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import evaluate_logits,METRICS
from .contrast_metrics import ranking_metrics


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--scale-root',type=Path,required=True)
    p.add_argument('--dose-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or torch.cuda.is_available():raise ValueError('new output and hidden GPUs required')
    torch.set_num_threads(2);args.output.mkdir(parents=True)
    write_json(args.output/'protocol.json',{'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'All existing 900 scale cells, and intact/removed cells at all five saved steps; no new model forwards.',
        'metrics':'Native pooled probability AUC reproduced first; stable margin-space pooled, macro within-episode, pair-weighted within-episode and cross-episode AUC. Undefined episodes retained and counted.',
        'independence':'Repeated accounts and episode streams are not independent seeds; no IID confidence intervals.',
        'scale_root':str(args.scale_root),'dose_root':str(args.dose_root)})
    rows=[];inventory=[];start=time.monotonic()
    for mode,root in [('scale',args.scale_root),('training_step',args.dose_root)]:
        done=json.loads((root/'DONE.json').read_text())
        if mode=='scale' and (done.get('cells')!=900 or done.get('partial_replay')):raise ValueError('incomplete scale replay')
        if mode=='training_step' and (done.get('cells')!=1140 or done.get('smoke_only')):raise ValueError('incomplete dose replay')
        refs=json.loads((root/'metrics.json').read_text())
        index={(r['target'],r['stream'],r['model_id'],r['condition'] if mode=='scale' else (r['step'],r['suppression_percent'],r['draw'])):r for r in refs}
        paths=sorted((root/'predictions').glob('*/*/*.pt'))
        if len(paths)!=(60 if mode=='scale' else 300):raise ValueError('incomplete saved prediction files')
        for path in paths:
            record=torch.load(path,map_location='cpu',weights_only=False);labels=record['labels']
            target,stream=path.parts[-3:-1];stem=path.stem
            mid,step=(stem,None) if mode=='scale' else (stem.rsplit('__step',1)[0],int(stem.rsplit('__step',1)[1]))
            for condition,logits in record['logits'].items():
                if mode=='training_step' and condition not in ((0,0),(100,0)):continue
                key=(target,stream,mid,condition if mode=='scale' else (step,*condition))
                ref=index[key];original=evaluate_logits(logits,labels)
                error=max(abs(original[k]-ref[k]) for k in METRICS)
                if error>1e-6:raise ValueError(f'production metric mismatch: {key}')
                m=(logits[:,1].double()-logits[:,0].double()).numpy()
                stats,_=ranking_metrics(m,labels['local_y'].numpy(),labels['mapping'].numpy(),labels['episode_ids'].numpy(),labels['use_global'])
                rows.append({'mode':mode,'target':target,'stream':stream,'model_id':mid,'source':ref['source'],'seed':ref['seed'],
                    'step':step if step is not None else 2500,'condition':condition if mode=='scale' else ('intact' if condition[0]==0 else 'removed'),
                    'native_pooled_auc':original['roc_auc'],'native_nll':original['nll'],'metric_reproduction_error':error,**stats})
            inventory.append({'mode':mode,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        write_json(args.output/'metrics.json',rows);write_json(args.output/'inventory.json',inventory)
        print(json.dumps({'mode':mode,'cells':len(rows),'elapsed_seconds':time.monotonic()-start}),flush=True)
    if len(rows)!=1500:raise ValueError('incomplete ranking analysis')
    write_json(args.output/'DONE.json',{'cells':len(rows),'prediction_files':len(inventory),'new_model_forwards':0,
        'max_native_metric_error':max(r['metric_reproduction_error'] for r in rows),
        'max_probability_margin_auc_difference':max(abs(r['native_pooled_auc']-r['pooled_margin_auc']) for r in rows),
        'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':main()
