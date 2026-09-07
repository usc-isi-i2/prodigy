"""Audit frozen interventions and actual validation selection/stopping records."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[7]
sys.path.insert(0,str(ROOT))
from scripts.experiments.setup.nm_interventions_overnight.plan import ARMS,HOLDOUT,ORDER
from scripts.experiments.analysis.transfer.ablations.prodigy_nm.nm_interventions_overnight.collect import audit_exposure


def audit(records,combined_flags=None,require_complete=True):
    definitions={a:[f for f in spec[0].split(',') if f] for a,spec in ARMS.items()}
    if combined_flags:definitions['combined']=sorted(combined_flags)
    expected={f'nmi_{a}_r{r}_s0' for a in definitions for r in range(1,9)}
    ids=[r['model_id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate training model records')
    if set(ids)-expected:raise ValueError('Unexpected training models')
    missing=sorted(expected-set(ids))
    if missing and require_complete:raise ValueError(f'Missing required training models: {missing}')
    fingerprints={};checked=[]
    for record in records:
        model=record['model_id'];match=re.fullmatch(r'nmi_(.+)_r([1-8])_s0',model)
        arm,rung=match[1],int(match[2]);sources=list(ORDER[:rung]);flags=sorted(definitions[arm])
        p=record['params'];s=record['selection'];protocol=record['validation_protocol'];history=record['validation_history']
        def need(condition,message):
            if not condition:raise ValueError(f'{model}: {message}')
        need(p['prefix']==model,'effective config model ID')
        need(sorted(filter(None,p['campaign_flags'].split(',')))==flags and s['flags']==flags,'intervention flags')
        need(p['neighbor_sampling_source_subset'].split(',')==sources and s['sources']==sources,'training sources')
        fixed=dict(seed=0,batch_size=1,epochs=1,campaign_protocol=True,campaign_eval_interval=2000,
            campaign_val_per_source=16,campaign_min_delta=.001,early_stopping_patience=2,
            campaign_holdout=HOLDOUT,neighbor_matching_edge_split=True,
            eval_test_before_train=False,eval_after_train=False,task_name='neighbor_matching',
            layers='S,U,M',gnn_type='sage',n_way=30,n_shots=3,n_query=4,
            learning_rate=.001,weight_decay=.001,workers=4,
            original_features=True,structural_features='none')
        need(all(p.get(k)==v for k,v in fixed.items()),'fixed training protocol')
        cap=1250*rung if arm=='budget' else 10000
        need(p['dataset_len_cap']==cap and p['emb_dim']==(512 if 'wide' in flags else 256),'episode cap or capacity')
        need(s['status']=='complete' and 0<s['training_steps']<=cap,'completion or episode limit')
        need(protocol['sources']==sources and protocol['excluded']==HOLDOUT,'validation source separation')
        need(protocol['episodes_per_source']==16 and protocol['interval']==2000 and
             protocol['patience']==2 and protocol['min_delta']==.001,'validation/stopping protocol')
        need(set(protocol['fingerprints'])==set(sources),'validation fingerprint panel')
        steps=[h['step'] for h in history]
        need(steps==sorted(set(steps)) and steps[-1]==s['training_steps'],'validation step history')
        need(steps==list(range(2000,s['training_steps']+1,2000))+
             ([s['training_steps']] if s['training_steps']%2000 else []),'validation cadence')
        meaningful_best=-math.inf;stale=0
        for i,h in enumerate(history):
            need(set(h['per_source'])==set(sources),'validation includes an inactive/missing source')
            values=[]
            for source,row in h['per_source'].items():
                need(row['episodes']==16 and row['fingerprint']==protocol['fingerprints'][source],'validation episodes')
                need(math.isfinite(row['roc_auc']) and 0<=row['roc_auc']<=1,'invalid validation AUC')
                fp=row['fingerprint']
                need(source not in fingerprints or fingerprints[source]==fp,'cross-model validation episode mismatch')
                fingerprints[source]=fp;values.append(row['roc_auc'])
            score=h['macro_roc_auc']
            need(abs(score-sum(values)/len(values))<1e-12,'source-macro validation aggregation')
            if score>meaningful_best+.001:meaningful_best,stale=score,0
            else:stale+=1
            need(i==len(history)-1 or stale<2,'continued after the stopping criterion')
        best=max(history,key=lambda h:h['macro_roc_auc'])
        need(s['best_step']==best['step'] and s['best_val']==best['macro_roc_auc'],'selected checkpoint is not earliest maximum')
        need(s['checkpoint'].endswith(f"state_dict_{best['step']}.ckpt"),'checkpoint step label')
        need((s['stop_reason']=='validation_plateau' and stale>=2) or
             (s['stop_reason']=='cap' and s['training_steps']==cap and stale<2),'stop reason disagrees with history')
        audit_exposure(s['exposure'],s['training_steps'],sources,flags)
        checked.append(dict(model_id=model,status='passed',training_steps=s['training_steps'],
                            selected_step=s['best_step'],validation_checks=len(history),stop_reason=s['stop_reason']))
    return dict(status='passed' if not missing else 'incomplete',expected_models=len(expected),
                checked_models=len(checked),missing_models=missing,validation_fingerprints=fingerprints,models=checked)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('records',type=Path);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--combined-selection',type=Path);a.add_argument('--allow-partial',action='store_true')
    args=a.parse_args();flags=json.loads(args.combined_selection.read_text())['flags'] if args.combined_selection else None
    result=audit(json.loads(args.records.read_text()),flags,not args.allow_partial)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f"{result['status']}: {result['checked_models']}/{result['expected_models']} training models audited")
