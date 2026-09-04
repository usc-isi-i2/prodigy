"""Plot frozen NM results; do not use this output for combination selection."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent


def main():
    frame=pd.read_csv(HERE/'data/nm_results.csv')
    parts=frame.model_id.str.extract(r'^nmi_(.+)_r(\d+)_s(\d+)$')
    frame['arm']=parts[0];frame['rung']=parts[1].astype(int)
    frame['included']=[t in s.split(',') for t,s in zip(frame.target,frame.sources)]
    base=frame[frame.arm=='baseline'][['rung','target','roc_auc']].rename(columns={'roc_auc':'baseline_auc'})
    paired=frame.merge(base,on=['rung','target'],how='left',validate='many_to_one')
    paired['delta']=paired.roc_auc-paired.baseline_auc
    rows=[]
    for arm,group in paired.groupby('arm',sort=False):
        endpoint=group[group.rung==8]
        inc=endpoint[endpoint.included]
        hold=endpoint[endpoint.target=='twibot20']
        complete=len(group)==72 and group.baseline_auc.notna().all()
        delta=float(inc.delta.mean()) if len(inc)==8 else None
        status='incomplete' if not complete else ('improved' if delta>0.001 else 'degraded' if delta < -0.001 else 'inconclusive')
        rows.append(dict(arm=arm,status=status,cells=len(group),endpoint_included_delta=delta,
            endpoint_unseen_delta=float(hold.delta.iloc[0]) if len(hold)==1 else None,
            all_rung_included_delta=float(group[group.included].delta.mean())))
    summary=pd.DataFrame(rows);summary.to_csv(HERE/'data/arm_summary.csv',index=False)
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    for name,subset,title in [('included',frame[frame.included],'Included training sources'),('unseen',frame[frame.target=='twibot20'],'Unseen TwiBot-20')]:
        fig,ax=plt.subplots(figsize=(12,7))
        for arm,group in subset.groupby('arm',sort=False):
            curve=group.groupby('rung').roc_auc.mean()
            ax.plot(curve.index,curve.values,marker='o',label=arm,linewidth=3 if arm=='baseline' else 1.4,alpha=1 if arm=='baseline' else .8)
        ax.set(xlabel='Number of training source graphs',ylabel='NM ROC-AUC',title=title,xticks=range(1,9))
        ax.grid(alpha=.2);ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
        fig.tight_layout();fig.savefig(figures/f'{name}_ladder.png',dpi=170);fig.savefig(figures/f'{name}_ladder.pdf');plt.close(fig)
    (HERE/'FINDINGS.md').write_text('# Source-held-out NM intervention campaign\n\n'
        'Seed 0 exploratory results. All checkpoints selected using active training-source validation only; TwiBot-20 excluded from selection.\n\n'
        'Status uses full 8-rung × 9-target coverage and the eight-source included-target mean delta, with ±0.001 practical threshold. Unseen-graph effects are separate.\n\n'+
        summary.to_markdown(index=False)+'\n\nNo CLS or LP runs are included. Plateau/cap metadata and exact configurations are retained in data/model_records.json.\n')
    print(summary.to_string(index=False))

if __name__=='__main__':main()
