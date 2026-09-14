from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
s=pd.read_csv(ROOT/'data/single_source.tsv',sep='\t');l=pd.read_csv(ROOT/'data/ladder.csv')
order=l.sort_values('n_sources').iloc[-1]['sources'].split(',')
targets=list(s.target.drop_duplicates());short={'ukr_rus_twitter':'UKR','covid19_twitter':'COVID','midterm':'Midterm','covid_political':'CovPol','election2020':'Election','ukr_rus_suspended':'Suspended','twibot20':'TwiBot','cp_hk_twitter':'HK','facebook_page_reference':'Facebook'}
S=s.pivot(index='source',columns='target',values='value').loc[order,targets].to_numpy()
L=l.pivot(index='n_sources',columns='target',values='roc_auc').loc[range(1,10),targets].to_numpy()
mean=np.cumsum(S,axis=0)/np.arange(1,10)[:,None];best=np.maximum.accumulate(S,axis=0)
def corr(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel()
 return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else None
def summarize(n):
 actual=L[1:n]-L[0];dm=np.diff(mean[:n],axis=0);dl=np.diff(L[:n],axis=0)
 result={'cells':int(actual.size),'delta_correlation':corr(dm,dl),'delta_sign_agreement':float(((dm>0)==(dl>0)).mean()),'actual_positive_delta_fraction':float((dl>0).mean()),'delta_target_centered_correlation':corr(dm-dm.mean(0),dl-dl.mean(0)), 'delta_rung_centered_correlation':corr(dm-dm.mean(1,keepdims=True),dl-dl.mean(1,keepdims=True))}
 for name,P in [('prefix_mean',mean),('prefix_best',best),('flat',np.broadcast_to(S[0],S.shape))]:
  gain=P[1:n]-P[0];err=actual-gain
  result[name]={'gain_correlation':corr(actual,gain),'rmse_auc_points':float(np.sqrt(np.mean(err**2))*100),'mae_auc_points':float(np.mean(abs(err))*100),'skill_vs_flat':float(1-np.sum(err**2)/np.sum(actual**2))}
 result['best_increment_correlation']=corr(np.diff(best[:n],axis=0),dl)
 mask=np.array([[t not in order[:i+1] for t in targets] for i in range(1,n)])
 residual=actual-(best[1:n]-best[0])
 result['still_unseen_targets']={'cells':int(mask.sum()),'best_rmse_auc_points':float(np.sqrt(np.mean(residual[mask]**2))*100),'flat_rmse_auc_points':float(np.sqrt(np.mean(actual[mask]**2))*100),'skill_vs_flat':float(1-np.sum(residual[mask]**2)/np.sum(actual[mask]**2))}
 result['pooled_level_correlation']=corr(mean[:n],L[:n])
 return result
summary={'full_historical':summarize(9),'rungs_1_to_5_no_suspended_training':summarize(5),'specialist_checkpoint_steps':s.groupby('source').checkpoint_step.first().to_dict()}
rows=[]
for i in range(1,9):
 for j,t in enumerate(targets):rows.append({'rung':i+1,'added':order[i],'target':t,'observed_delta_pp':100*(L[i,j]-L[i-1,j]),'mean_predicted_delta_pp':100*(mean[i,j]-mean[i-1,j]),'actual_auc':L[i,j],'specialist_auc':S[i,j],'best_available_specialist_auc':best[i,j]})
pd.DataFrame(rows).to_csv(ROOT/'data/comparison.csv',index=False)
final=pd.DataFrame({'target':[short[t] for t in targets],'r1':L[0]*100,'r9':L[-1]*100,'actual_gain':(L[-1]-L[0])*100,'mean_predicted_gain':(mean[-1]-mean[0])*100,'best_specialist':best[-1]*100,'r9_minus_best_specialist':(L[-1]-best[-1])*100})
final.to_csv(ROOT/'data/final_comparison.csv',index=False)
(ROOT/'data'/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(2,3,figsize=(13,7.5),sharex=True)
for j,(ax,t) in enumerate(zip(axes.ravel(),targets)):
 ax.axvspan(5.5,9.3,color='#fff1d6',zorder=0)
 ax.plot(range(1,10),L[:,j]*100,'o-',label='Observed ladder',color='#202020',lw=2)
 ax.plot(range(1,10),(L[0,j]+mean[:,j]-mean[0,j])*100,'s--',label='Mean specialist gain',color='#2879bd',ms=4)
 ax.plot(range(1,10),(L[0,j]+best[:,j]-best[0,j])*100,':',label='Best specialist gain',color='#ab488c',lw=2)
 ax.set_title(short[t]);ax.set_ylabel('Cosine ROC-AUC (%)');ax.grid(alpha=.2);ax.set_xticks(range(1,10));ax.set_xlim(.7,9.3)
 ax.set_xlabel('Rung')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,.955),frameon=False)
fig.suptitle('Does single-source transfer predict the MLP ladder?',fontsize=16,y=1)
fig.text(.5,.015,'Predictions anchored to observed rung 1; no fitting. Shading: rungs trained with corrupted Suspended data.\nSame cosine-AUC protocol, six shared targets; specialist and ladder training budgets differ.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.065,1,.92));fig.savefig(ROOT/'figures'/'comparison.png',dpi=180);fig.savefig(ROOT/'figures'/'comparison.pdf');plt.close(fig)
print(json.dumps(summary,indent=2));print(final.round(2).to_string(index=False))
