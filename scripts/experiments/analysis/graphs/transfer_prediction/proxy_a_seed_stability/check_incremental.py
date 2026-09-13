"""Proxy-A beyond donor/target fixed effects; graph-held-out relative prediction.

No diagonal cells. New-target prediction removes the held-out graph in both
roles from fitting. Its unknown intercept is removed only for scoring centered
AUC; absolute AUC is deliberately not claimed. No cell-IID significance tests.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
OUT=BASE/'incremental';(OUT/'data').mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
d=json.loads((BASE/'sample_40000/data/results.json').read_text());names=d['graphs'];n=len(names)
xs=np.array([r['proxy_a_distance'] for r in sorted(d['runs'],key=lambda r:r['seed']) if r['policy']=='uniform'])
x=xs.mean(0)
f=pd.read_csv(BASE.parent/'similarity_vs_transfer_v2/data/final_core_auc/transfer_matrix_three_seed_mean_long.csv')
y=f[f.metric=='roc_auc_ovr_macro'].pivot(index='train',columns='test',values='value').reindex(index=names,columns=names).to_numpy()
assert xs.shape==(3,n,n) and np.isfinite(y).all()
i,j=np.where(~np.eye(n,dtype=bool));D=np.column_stack([np.eye(n)[i],np.eye(n)[j]])
v=y[i,j]

def residual(a,design):return a-design@np.linalg.lstsq(design,a,rcond=None)[0]

def estimate(mat):
 r_x=residual(mat[i,j],D);r_y=residual(v,D)
 beta=float(r_x@r_y/(r_x@r_x))
 return dict(partial_pearson=float(pearsonr(r_x,r_y).statistic),residual_spearman=float(spearmanr(r_x,r_y).statistic),slope_auc_per_proxy_a=beta,partial_r2=float(pearsonr(r_x,r_y).statistic**2))
summary={'all_cells':estimate(x),'estimator_seeds':[estimate(m) for m in xs]}
rx=residual(x[i,j],D);ry=residual(v,D)
pd.DataFrame(dict(source=np.array(names)[i],target=np.array(names)[j],proxy_a=x[i,j],auc=v,proxy_a_residual=rx,auc_residual=ry)).to_csv(OUT/'data/residuals.csv',index=False)
folds=[];predictions=[]
for g in range(n):
 train=(i!=g)&(j!=g);test=j==g
 # Full dummy columns retained: unseen graph coefficients become zero under
 # minimum-norm solve. Their arbitrary intercept is irrelevant to centered scores.
 b0=np.linalg.lstsq(D[train],v[train],rcond=None)[0]
 full=np.column_stack([D,x[i,j]])
 b1=np.linalg.lstsq(full[train],v[train],rcond=None)[0]
 actual=v[test];p0=D[test]@b0;p1=full[test]@b1
 tx=residual(x[i[train],j[train]],D[train]);ty=residual(v[train],D[train])
 row=dict(target=names[g],fit_partial_r=float(pearsonr(tx,ty).statistic),fit_beta=float(b1[-1]))
 for name,pred in [('source_only',p0),('source_plus_proxy',p1)]:
  errors=(pred-pred.mean())-(actual-actual.mean())
  row[name+'_centered_rmse']=float(np.sqrt(np.mean(errors**2)))
  row[name+'_rho']=float(spearmanr(pred,actual).statistic)
  row[name+'_regret']=float(actual.max()-actual[np.argmax(pred)])
 folds.append(row)
 for source,a,b,c in zip(np.array(names)[i[test]],actual,p0,p1):predictions.append(dict(target=names[g],source=source,auc=a,source_only=b,source_plus_proxy=c))
folds=pd.DataFrame(folds);folds.to_csv(OUT/'data/graph_holdout_folds.csv',index=False)
pd.DataFrame(predictions).to_csv(OUT/'data/graph_holdout_predictions.csv',index=False)
summary['holdout_means']={c:float(folds[c].mean()) for c in folds if c!='target'}
summary['holdout_improved_rmse_targets']=int((folds.source_plus_proxy_centered_rmse<folds.source_only_centered_rmse).sum())
summary['leave_graph_out_partial_r_range']=[float(folds.fit_partial_r.min()),float(folds.fit_partial_r.max())]
# Exploratory influence diagnostic, retaining the full-data result above.
e=names.index('election2020');c=names.index('covid_political')
keep=~(((i==e)&(j==c))|((i==c)&(j==e)))
summary['without_election_political_pair_partial_r']=float(pearsonr(residual(x[i[keep],j[keep]],D[keep]),residual(v[keep],D[keep])).statistic)
(OUT/'data/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(8,6));ax.scatter(rx,ry,color='#517d99',s=40,alpha=.8)
for k in np.flatnonzero(~keep):
 ax.annotate('Election → Political' if i[k]==e else 'Political → Election',(rx[k],ry[k]),xytext=(10,-8),textcoords='offset points',fontsize=9)
a=np.array([rx.min(),rx.max()]);ax.plot(a,summary['all_cells']['slope_auc_per_proxy_a']*a,color='black')
ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6)
ax.set(xlabel='Proxy-A residual after source + target effects',ylabel='NM AUC residual after source + target effects',title=f"Compatibility beyond general donor strength: partial r = {summary['all_cells']['partial_pearson']:.3f}")
fig.text(.5,.02,'72 foreign transfer cells; 40k uniform proxy-A averaged over 3 estimator seeds.\nNM AUC averaged over 3 training seeds. Descriptive fit; cells are dependent.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,1])
for ext in ['png','pdf']:fig.savefig(OUT/f'figures/residual_scatter.{ext}',dpi=170)
print(json.dumps(summary,indent=2));print(folds.to_string(index=False))
