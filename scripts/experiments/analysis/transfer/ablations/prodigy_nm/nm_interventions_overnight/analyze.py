"""Plot frozen NM results; never use this output for combination selection."""
from pathlib import Path
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
        rows.append(dict(arm=arm,status=verdict(delta) if complete else 'incomplete',
            cells=len(group),expected_cells=len(expected),
            endpoint_included_status=verdict(delta),endpoint_included_delta=delta,
            endpoint_unseen_status=verdict(unseen),endpoint_unseen_delta=unseen,
            all_rung_included_delta=float(group[group.included].delta.mean()) if complete else None))
    summary=pd.DataFrame(rows);summary.to_csv(HERE/'data/arm_summary.csv',index=False)
    role=frame.copy()
    role['role']=['included' if inc else 'unseen' if t==HOLDOUT else 'not_yet_included'
                  for inc,t in zip(role.included,role.target)]
    role.groupby(['arm','rung','role'],as_index=False).agg(roc_auc=('roc_auc','mean'),
        target_count=('target','nunique')).to_csv(HERE/'data/role_summary.csv',index=False)
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    for name,title in [('included','Included training sources'),('unseen','Unseen TwiBot-20'),
                       ('not_yet_included','Sources outside the current training rung')]:
        subset=role[role.role==name]
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
        summary.to_markdown(index=False)+'\n\nIncluded-source, unseen-graph, and not-yet-included-source panels are separate. No CLS or LP runs are included. Plateau/cap metadata and exact configurations are retained in data/model_records.json.\n')
    print(summary.to_string(index=False))

if __name__=='__main__':main()
