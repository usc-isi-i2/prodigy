"""Validate the complete scale/swap replay and join existing source evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY=['target','stream','model_id']
METRICS=['roc_auc','accuracy','f1','nll']
DESCRIPTORS=['support_norm_mean','support_norm_cv','support_unit_dispersion','support_class_cosine']


def pair_scores(cells):
    if cells.duplicated(KEY+['condition']).any():raise ValueError('duplicate condition')
    baseline=cells[cells.condition=='scale_1'][KEY+METRICS]
    paired=cells.merge(baseline,on=KEY,suffixes=('','_intact'),validate='many_to_one')
    if len(paired)!=len(cells):raise ValueError('missing intact baseline')
    for metric in METRICS:paired['delta_'+metric]=paired[metric]-paired[metric+'_intact']
    return paired


def geometry_means(rows):
    table=pd.DataFrame(rows)
    if table.duplicated(KEY+['condition','stage','cohort']).any():raise ValueError('duplicate geometry')
    active=table['count']>0
    for name in ['norm','cos_intact','cos_zero','l2_intact','l2_zero','intact_norm','zero_norm']:
        table[name+'_mean']=table[name+'_sum']/table['count'].where(active)
    table['norm_sd']=np.sqrt(np.maximum(0,table.norm_sq_sum/table['count'].where(active)-table.norm_mean**2))
    table['norm_ratio_intact']=table.norm_sum/table.intact_norm_sum.replace(0,np.nan)
    table['direction_change_intact']=1-table.cos_intact_mean
    table['gate_change_fraction']=table.positive_gate_changes_from_intact/table.coordinates.replace(0,np.nan)
    queries=table[active & table.cohort.str.startswith('query')]
    if not (queries.max_abs_intact==0).all():raise ValueError('query geometry changed')
    return table


def source_evidence(data):
    cells=pd.DataFrame(json.loads((data/'role_context_replay/metrics.json').read_text()))
    if len(cells)!=720 or cells.duplicated(KEY+['variant']).any():raise ValueError('bad historical grid')
    grid=cells.pivot(index=['source','target','stream'],columns='variant',values='roc_auc').reset_index()
    grid['support_delta_points']=100*(grid.edges_support-grid.baseline)
    grid['query_delta_points']=100*(grid.edges_query-grid.baseline)
    ep=pd.DataFrame(json.loads((data/'cached_support_geometry_20260907/episodes.json').read_text()))
    if len(ep)!=11520:raise ValueError('missing cached geometry')
    g=ep.groupby(['source','target','stream'])[DESCRIPTORS].mean().reset_index()
    joined=grid.merge(g,on=['source','target','stream'],validate='one_to_one')
    sensitivity=pd.DataFrame(json.loads((data/'cached_support_geometry_20260907/sensitivity.json').read_text())['rows'])
    joined=joined.merge(sensitivity[['source','target','stream','mean_probability_total_variation']],
                        on=['source','target','stream'],validate='one_to_one')
    if len(joined)!=90:raise ValueError('incomplete nine-source map')
    correlations=[]
    for target,tgroup in joined.groupby('target'):
        for stream in ['original','fresh','stream_mean']:
            group=tgroup[tgroup.stream==stream] if stream!='stream_mean' else tgroup.groupby('source').mean(numeric_only=True).reset_index()
            for scope in ['all_nine','excluding_hong_kong']:
                s=group if scope=='all_nine' else group[group.source!='cp_hk']
                for feature in DESCRIPTORS+['mean_probability_total_variation']:
                    correlations.append({'target':target,'stream':stream,'scope':scope,'sources':len(s),'descriptor':feature,
                        'spearman':s[feature].corr(s.support_delta_points,method='spearman'),
                        'status':'exploratory; all outcomes already observed; not independent confirmation'})
    return joined,pd.DataFrame(correlations)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path(__file__).parent/'data')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--complete-targets',nargs='+',help='Explicit complete blocks of a still-running full replay; never partial seeds, streams or conditions.')
    args=p.parse_args()
    read=lambda n:json.loads((args.input/f'{n}.json').read_text())
    protocol,receipts=map(read,['protocol','receipts'])
    done=read('DONE') if (args.input/'DONE.json').exists() else None
    if protocol['partial_replay']:raise ValueError('original full replay required')
    if args.complete_targets:
        targets=args.complete_targets
        if len(set(targets))!=len(targets) or not set(targets)<=set(protocol['targets']):raise ValueError('invalid block selection')
    else:
        targets=protocol['targets']
        if not done or done['partial_replay'] or not done['all_inputs_weights_and_operators_unchanged'] or not done['historical_endpoint_metrics_match'] or done['cells']!=900:
            raise ValueError('complete and verified full replay required')
    cells=pd.DataFrame(read('metrics'))
    cells=cells[cells.target.isin(targets)].copy()
    receipts=[r for r in receipts if r['target'] in targets]
    panel={(t,s,m) for t in targets for s in ('original','fresh') for m in protocol['models']}
    expected={(*k,c['name']) for k in panel for c in protocol['conditions']}
    if set(map(tuple,cells[KEY+['condition']].to_numpy()))!=expected or len(cells)!=len(targets)*180:
        raise ValueError('incomplete metric panel')
    if len(receipts)!=len(panel) or {(r['target'],r['stream'],r['model_id']) for r in receipts}!=panel:
        raise ValueError('incomplete receipts')
    for r in receipts:
        if r['batches']!=32 or r['query_checks']!=960 or not r['input_weights_and_operator_unchanged'] or not r['historical_endpoint_metrics_match'] or r['historical_endpoint_max_logit_error']>1e-5:
            raise ValueError('failed replay receipt')
    paths=[args.input/'geometry'/f'{target}__{stream}.json' for target in sorted(targets) for stream in ('original','fresh')]
    if not all(path.exists() for path in paths):raise ValueError('incomplete stage geometry')
    geometry=geometry_means([r for path in paths for r in json.loads(path.read_text())])
    if len(geometry)!=len(cells)*31 or not (geometry.groupby(KEY+['condition']).size()==31).all():raise ValueError('incomplete stage/cohort panel')
    if not np.isfinite(cells[METRICS].to_numpy()).all():raise ValueError('nonfinite metrics')
    paired=pair_scores(cells)
    summary=paired.groupby(['target','source','condition']).agg(
        cells=('roc_auc','size'),auc_mean=('roc_auc','mean'),delta_auc_mean=('delta_roc_auc','mean'),
        delta_auc_min=('delta_roc_auc','min'),delta_auc_max=('delta_roc_auc','max'),
        delta_auc_positive=('delta_roc_auc',lambda x:int((x>0).sum())),delta_nll_mean=('delta_nll','mean')).reset_index()
    historical,correlations=source_evidence(args.data)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,table in [('cells',paired),('source_summary',summary),('geometry',geometry),
                       ('nine_source_map',historical),('geometry_associations',correlations)]:
        table.to_csv(args.output/f'{name}.csv',index=False)
    files=[args.input/f'{n}.json' for n in ('protocol','metrics','receipts')]+paths
    if done:files.append(args.input/'DONE.json')
    validation={'cells':len(cells),'geometry_cells':len(geometry),'query_checks':sum(r['query_checks'] for r in receipts),
        'completed_targets':targets,'completed_block_snapshot':args.complete_targets is not None,
        'full_run_complete':done is not None and done.get('cells')==900,
        'historical_endpoint_max_logit_error':max(r['historical_endpoint_max_logit_error'] for r in receipts),
        'unassignable_radial_norms':sum(v['nonzero_norm_unassignable'] for r in receipts for v in r['radial_swaps'].values()),
        'runtime_revision':protocol['revision'],'all_queries_unchanged':True,
        'sha256':{str(path.relative_to(args.input)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        'uncertainty':'Means/ranges over 3 initialization seeds x 2 episode streams, not confidence intervals. Historical map: one distinct seed-0 checkpoint per source. Associations exploratory.'}
    (args.output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k!='sha256'},indent=2))
    print(summary[summary.target=='covid_political'].to_string(index=False))


if __name__=='__main__':main()
