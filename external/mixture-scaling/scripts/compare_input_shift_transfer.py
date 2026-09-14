"""Descriptive comparison with historical transfer results; no model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(args):
    d = json.loads(Path(args.audit).read_text())
    joint, single = pd.read_csv(args.interleaved), pd.read_csv(args.single)
    out = Path(args.output)
    rows = []
    for (a,b), group in joint.groupby(['first','second']):
        targets = sorted(set(d['graphs'])-{a,b})
        values = []
        for source in [a,b]:
            cells = single[(single.source==source)&single.target.isin(targets)]
            assert len(cells)==6 and cells.target.nunique()==6
            values.append(cells.auc.mean())
        cells = group[group.target.isin(targets)]
        assert len(cells)==6 and cells.target.nunique()==6
        r = dict(first=a,second=b,joint_transfer_auc=cells.roc_auc.mean(),
                 best_constituent_transfer_auc=max(values),
                 transfer_delta_pp=100*(cells.roc_auc.mean()-max(values)))
        for v,t in d['distances'].items():
            match = [p for p in t['pairs'] if {p['first'],p['second']}=={a,b}]
            assert len(match)==1
            r[v+'_shift']=match[0]['normalized_sliced_w1']
        rows.append(r)
    frame=pd.DataFrame(rows)
    (out/'data').mkdir(parents=True,exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    frame.to_csv(out/'data/shift_vs_historical_transfer.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    correlations={}
    for ax,v,title in zip(axes,['node','node_neighbors'],['Raw node embedding','Node + mean neighbors']):
        x,y=frame[v+'_shift'],frame.transfer_delta_pp
        rho=x.rank().corr(y.rank())
        correlations[v]=rho
        ax.scatter(x,y,c=np.where(y>0,'#17856a','#ca543b'),s=45,alpha=.85)
        ax.margins(y=.18)
        ax.axhline(0,color='gray',lw=1)
        ax.set_xlabel('Normalized sliced Wasserstein distance')
        ax.set_ylabel('Joint − best constituent transfer AUC (pp)')
        ax.set_title(f'{title}\nDescriptive Spearman ρ = {rho:.2f}')
        for a,b,label in [('ukr_rus_twitter','facebook_page_reference','Ukraine–Facebook'),
                          ('covid_political','ukr_rus_suspended','Political–Suspended')]:
            r=frame[((frame['first']==a)&(frame.second==b))|((frame['first']==b)&(frame.second==a))].iloc[0]
            ax.annotate(label,(r[v+'_shift'],r.transfer_delta_pp),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
    fig.suptitle('Input shift versus existing interleaving outcomes\n28 pairs share sources; different pairs have different six-target sets. No inferential p-values.',fontsize=12)
    fig.savefig(out/'figures/shift_vs_transfer.png',dpi=170)
    (out/'data/historical_comparison.json').write_text(json.dumps(dict(
        input_files=dict(interleaved=str(Path(args.interleaved).resolve()),single=str(Path(args.single).resolve())),
        pairs=len(frame),positive_delta=int((frame.transfer_delta_pp>0).sum()),spearman=correlations),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--audit',required=True)
    p.add_argument('--interleaved',required=True)
    p.add_argument('--single',required=True)
    p.add_argument('--output',required=True)
    run(p.parse_args())
