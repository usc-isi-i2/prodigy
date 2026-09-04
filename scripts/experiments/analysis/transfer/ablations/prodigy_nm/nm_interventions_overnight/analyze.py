"""Plot frozen NM results; never use this output for combination selection."""
from pathlib import Path
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[6]
sys.path.insert(0,str(ROOT))
from scripts.experiments.setup.nm_interventions_overnight.plan import ARMS, TARGETS, HOLDOUT


def verdict(delta):
    if pd.isna(delta):return 'incomplete'
    return 'improved' if delta>0.001 else 'degraded' if delta < -0.001 else 'inconclusive'


def resource_summary():
    path=HERE/'data/resources.csv'
    if not path.exists():return ''
    try:resources=pd.read_csv(path)
    except pd.errors.EmptyDataError:return ''
    records=json.loads((HERE/'data/model_records.json').read_text())
    late_gain={}
    for row in records:
        history=row['validation_history']
        late_gain[row['model_id']]=(row['selection']['stop_reason']=='cap' and len(history)>1 and
            history[-1]['macro_roc_auc']-history[-2]['macro_roc_auc']>row['params']['campaign_min_delta'])
    resources['arm']=resources.model_id.str.extract(r'^nmi_(.+)_r\d+_s\d+$')[0]
    resources['cap_with_last_check_gain']=resources.model_id.map(late_gain)
    resources['plateau']=resources.stop_reason.eq('validation_plateau')
    resources['cap']=resources.stop_reason.eq('cap')
    resources['seconds_per_1000_episodes']=1000*resources.loop_seconds/resources.training_steps
    resources['peak_tensor_mib']=resources.peak_allocated_bytes/2**20
    summary=resources.groupby('arm',sort=False,as_index=False).agg(
        trained_models=('model_id','count'),model_parameters=('model_parameters','max'),
        mean_episodes=('training_steps','mean'),plateau_stops=('plateau','sum'),
        cap_stops=('cap','sum'),cap_with_last_check_gain=('cap_with_last_check_gain','sum'),
        mean_seconds_per_1000_episodes=('seconds_per_1000_episodes','mean'),
        peak_tensor_mib=('peak_tensor_mib','max'))
    summary.to_csv(HERE/'data/resource_summary.csv',index=False)
    return ('\n\nTraining cost and stopping evidence for completed models:\n\n'+
        summary.round(2).to_markdown(index=False)+'\n\n'
        'Parameter counts include the registered frozen label table; resources.csv separately records '
        'optimizer parameter slots and the auxiliary head. Timing comes from concurrent runs, excludes '
        'initial validation-cache construction from the loop timer, and is not an isolated speed benchmark. '
        'Peak tensor memory excludes CUDA context overhead. A cap stop is not evidence of convergence; '
        'cap_with_last_check_gain counts capped runs whose final validation increment still exceeded 0.001. '
        'Effect verdicts apply to this bounded training protocol.\n')


def main():
    path=HERE/'data/nm_results.csv'
    try:frame=pd.read_csv(path)
    except (pd.errors.EmptyDataError,FileNotFoundError):
        frame=pd.DataFrame(columns=['model_id','target','sources','roc_auc'])
    if frame.duplicated(['model_id','target']).any():raise ValueError('Duplicate result cells')
    parts=frame.model_id.str.extract(r'^nmi_(.+)_r(\d+)_s(\d+)$')
    if parts.isna().any().any():raise ValueError('Unrecognized model IDs')
    if len(parts) and not (parts[2]=='0').all():raise ValueError('Separate training seeds before aggregation')
    frame['arm']=parts[0];frame['rung']=parts[1].astype(int)
    frame['included']=[t in s.split(',') for t,s in zip(frame.target,frame.sources)]
    frame['included']=frame.included.astype(bool)
    frame['roc_auc']=pd.to_numeric(frame.roc_auc)
    base=frame[frame.arm=='baseline'][['rung','target','roc_auc']].rename(columns={'roc_auc':'baseline_auc'})
    paired=frame.merge(base,on=['rung','target'],how='left',validate='many_to_one')
    paired['delta']=paired.roc_auc-paired.baseline_auc
    expected={(r,t) for r in range(1,9) for t in TARGETS}
    rows=[]
    for arm in list(ARMS)+(['combined'] if 'combined' in set(paired.arm) else []):
        group=paired[paired.arm==arm];endpoint=group[group.rung==8]
        inc=endpoint[endpoint.included];hold=endpoint[endpoint.target==HOLDOUT]
        observed=set(zip(group.rung,group.target))
        if observed-expected:raise ValueError(f'Unexpected evaluation cells for {arm}')
        complete=observed==expected and group.baseline_auc.notna().all()
        delta=float(inc.delta.mean()) if len(inc)==8 and inc.delta.notna().all() else None
        unseen=float(hold.delta.iloc[0]) if len(hold)==1 else None
        overall=float(endpoint.delta.mean()) if len(endpoint)==len(TARGETS) and endpoint.delta.notna().all() else None
        rows.append(dict(arm=arm,status=verdict(delta) if complete else 'incomplete',
            cells=len(group),expected_cells=len(expected),
            endpoint_included_status=verdict(delta),endpoint_included_delta=delta,
            endpoint_unseen_status=verdict(unseen),endpoint_unseen_delta=unseen,
            endpoint_all_targets_delta=overall,
            all_rung_included_delta=float(group[group.included].delta.mean()) if complete else None))
    summary=pd.DataFrame(rows);summary.to_csv(HERE/'data/arm_summary.csv',index=False)
    endpoint=summary[summary.arm!='baseline'].copy()
    if endpoint.endpoint_included_delta.notna().any():
        fig,axes=plt.subplots(1,2,figsize=(12,8),sharey=True)
        for ax,column,title in zip(axes,['endpoint_included_delta','endpoint_unseen_delta'],
                ['Eight included sources (macro mean)','Unseen TwiBot-20']):
            values=pd.to_numeric(endpoint[column],errors='coerce')
            ax.scatter(values,range(len(endpoint)),s=45)
            ax.axvline(0,color='black',lw=1);ax.axvspan(-.001,.001,color='grey',alpha=.15)
            ax.set(xlabel='NM ROC-AUC difference from baseline',title=title,
                   yticks=range(len(endpoint)),yticklabels=endpoint.arm)
            ax.grid(axis='x',alpha=.2)
        axes[0].invert_yaxis();fig.suptitle('Eight-source endpoint, seed 0; missing results have no point')
        fig.tight_layout();(HERE/'figures').mkdir(exist_ok=True)
        fig.savefig(HERE/'figures/endpoint_deltas.png',dpi=170)
        fig.savefig(HERE/'figures/endpoint_deltas.pdf');plt.close(fig)
    role=frame.copy()
    role['role']=['included' if inc else 'unseen' if t==HOLDOUT else 'not_yet_included'
                  for inc,t in zip(role.included,role.target)]
    role_summary=role.groupby(['arm','rung','role'],as_index=False).agg(roc_auc=('roc_auc','mean'),
        target_count=('target','nunique'))
    role_summary['expected_targets']=[r if role_name=='included' else 1 if role_name=='unseen' else 8-r
        for r,role_name in zip(role_summary.rung,role_summary.role)]
    role_summary['complete_panel']=role_summary.target_count==role_summary.expected_targets
    role_summary.loc[~role_summary.complete_panel,'roc_auc']=float('nan')
    all_targets=role.groupby(['arm','rung'],as_index=False).agg(roc_auc=('roc_auc','mean'),
        target_count=('target','nunique'))
    # A fixed-panel mean is defined only when every target has a result.
    all_targets=all_targets[all_targets.target_count==len(TARGETS)].assign(
        role='all_targets',expected_targets=len(TARGETS),complete_panel=True)
    pd.concat([role_summary,all_targets],ignore_index=True).to_csv(HERE/'data/role_summary.csv',index=False)
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    comparison=''
    if 'combined' in set(summary.arm):
        singles=summary[(~summary.arm.isin(['baseline','combined','budget'])) & summary.endpoint_included_delta.notna()]
        combined=summary[summary.arm=='combined'].iloc[0]
        if len(singles) and pd.notna(combined.endpoint_included_delta):
            best=singles.loc[singles.endpoint_included_delta.idxmax()]
            margin=combined.endpoint_included_delta-best.endpoint_included_delta
            pd.DataFrame([dict(best_single_arm=best.arm,
                combined_minus_best_single_included=margin,
                verdict=verdict(margin),comparison='descriptive test comparison; not selection')]).to_csv(
                    HERE/'data/combined_comparison.csv',index=False)
            comparison=(f'\nCombined included-source endpoint delta versus the strongest observed individual '
                f'endpoint ({best.arm}): {margin:+.6f} ({verdict(margin)}). This is a descriptive test '
                'comparison; test outcomes did not choose the recipe.\n')
    for name,title in [('all_targets','Fixed nine-graph evaluation panel'),('included','Included training sources'),('unseen','Unseen TwiBot-20'),
                       ('not_yet_included','Sources outside the current training rung')]:
        if name=='all_targets':
            subset=role.merge(all_targets[['arm','rung']],on=['arm','rung'],how='inner')
        else:
            complete_pairs=role_summary[(role_summary.role==name)&role_summary.complete_panel][['arm','rung']]
            subset=role[role.role==name].merge(complete_pairs,on=['arm','rung'],how='inner')
        if subset.empty:continue
        fig,ax=plt.subplots(figsize=(12,7))
        for arm,group in subset.groupby('arm',sort=False):
            curve=group.groupby('rung').roc_auc.mean()
            ax.plot(curve.index,curve.values,marker='o',label=arm,linewidth=3 if arm=='baseline' else 1.4,alpha=1 if arm=='baseline' else .8)
        ax.set(xlabel='Number of training source graphs',ylabel='NM ROC-AUC',title=title,xticks=range(1,9))
        ax.grid(alpha=.2);ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
        fig.tight_layout();fig.savefig(figures/f'{name}_ladder.png',dpi=170);fig.savefig(figures/f'{name}_ladder.pdf');plt.close(fig)
    (HERE/'FINDINGS.md').write_text('# Source-held-out NM intervention campaign\n\n'
        'Seed 0 exploratory results. All checkpoints selected using active training-source validation only; TwiBot-20 excluded from selection.\n\n'
        'Overall arm status requires all 8 rungs × 9 targets and paired baselines. Endpoint columns describe only the eight-source endpoint and may be available before the full campaign is complete. Effects use a ±0.001 practical threshold, not statistical significance. Baseline is the reference; its zero delta is not an intervention finding.\n\n'+
        summary.to_markdown(index=False)+comparison+resource_summary()+'\n\nThe all-target curve uses the same nine graphs at every rung and requires a complete target panel. Included-source and not-yet-included-source averages change graph membership across rungs; use the fixed-panel and unseen-graph curves to avoid that composition confound. All panels remain separate. No CLS or LP runs are included. Plateau/cap metadata and exact configurations are retained in data/model_records.json.\n')
    print(summary.to_string(index=False))

if __name__=='__main__':main()
