from pathlib import Path
import json,os
import numpy as np,pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,roc_curve,average_precision_score
P=Path(os.environ.get('PAIR_ERROR_OUTPUT','/tmp/mlp-error-analysis'));P.mkdir(parents=True,exist_ok=True);base=Path(os.environ.get('PAIR_SCORE_ROOT',str(P/'dataMeR1/phil/gfm')));C=Path(os.environ.get('PAIR_COVARIATES',str(P/'covariates')))
roots={'single':base/'mixture-scaling-bias-lp/state/bias_s0/results/node_neighbors_bias','seq':base/'mixture-scaling/state/sequential_mlp_pairs_adam_s0/results/node_neighbors_bias','inter':base/'mixture-scaling/state/interleaved_mlp_pairs_noelec_s0/results/node_neighbors_bias'}
sources=['ukr_rus_twitter','covid19_twitter','midterm','covid_political','ukr_rus_suspended','twibot20','cp_hk_twitter','facebook_page_reference']
pairs=[('Ukraine + Facebook','ukr_rus_twitter','facebook_page_reference'),('COVID political + suspended','covid_political','ukr_rus_suspended')]
def load(kind,run,target):
 p=roots[kind]/(run+'__to__'+target);return dict(np.load(str(p)+'.scores.npz')),json.loads(Path(str(p)+'.json').read_text())
def credit(y,s):
 result=np.zeros(len(y));neg=np.sort(s[y==0]);pos=np.sort(s[y==1]);v=s[y==1]
 result[y==1]=(np.searchsorted(neg,v,'left')+np.searchsorted(neg,v,'right'))/(2*len(neg));v=s[y==0]
 result[y==0]=1-(np.searchsorted(pos,v,'left')+np.searchsorted(pos,v,'right'))/(2*len(pos))
 assert np.allclose(result[y==1].mean(),roc_auc_score(y,s))
 return result
def ranking_turnover(y,a,b):
 pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);gain=loss=0.
 for inds in np.array_split(pos,14):
  aa=(a[inds,None]>a[neg]).astype(float)+.5*(a[inds,None]==a[neg])
  bb=(b[inds,None]>b[neg]).astype(float)+.5*(b[inds,None]==b[neg]);delta=bb-aa
  gain+=np.maximum(delta,0).sum();loss+=np.maximum(-delta,0).sum()
 n=len(pos)*len(neg);assert np.isclose((gain-loss)/n,roc_auc_score(y,b)-roc_auc_score(y,a))
 return gain/n,loss/n
metrics=[];transitions=[];strata=[];examples=[];selectors=[]
for pair,a,b in pairs:
 for target in sources:
  if target in (a,b):continue
  models={k:load(kind,run,target) for k,kind,run in [('A','single','ss_'+a),('B','single','ss_'+b),('A→B','seq',f'seq_{a}__then__{b}'),('B→A','seq',f'seq_{b}__then__{a}'),('Interleaved','inter',f'interleave_{a}__and__{b}') ]}
  ref=models['A'][0];val=ref['validation_mask'].astype(bool);test=~val;y=ref['labels'][test].astype(int)
  for arr,report in models.values():
   for key in ['u','v','labels','validation_mask']:assert np.array_equal(arr[key],ref[key])
   assert np.isclose(roc_auc_score(y,arr['dot_logits'][test]),report['metrics']['test']['roc_auc'])
  best=max(['A','B'],key=lambda k:models[k][1]['metrics']['validation']['roc_auc'])
  testbest=max(['A','B'],key=lambda k:models[k][1]['metrics']['test']['roc_auc'])
  selectors.append(dict(pair=pair,target=target,validation_selected=best,test_best=testbest))
  bas,brep=models[best];bs=bas['dot_logits'][test];bd=brep['metrics']['decisions'];bp=bs>=bd['threshold'] if bd['threshold'] is not None else np.zeros(len(y),bool)
  bc=credit(y,bs);bl=np.logaddexp(0,bs)-y*bs
  probs=[]
  for k in ['A','B']:
   arr,r=models[k];cal=r['metrics']['decisions']['calibration'];probs.append(expit(cal['slope']*arr['dot_logits']+cal['intercept']))
  ensemble=np.mean(probs,axis=0);fpr,tpr,thresholds=roc_curve(ref['labels'][val],ensemble[val],drop_intermediate=False);eth=thresholds[np.argmax(tpr-fpr)]
  for name,(arr,r) in models.items():
   s=arr['dot_logits'][test];dec=r['metrics']['decisions'];pred=s>=dec['threshold'] if dec['threshold'] is not None else np.zeros(len(y),bool)
   cal=dec['calibration'];cal_s=cal['slope']*s+cal['intercept'];loss=np.logaddexp(0,s)-y*s
   metrics.append(dict(pair=pair,target=target,model=name,auc=roc_auc_score(y,s),ap=average_precision_score(y,s),bce=loss.mean(),positive_bce=loss[y==1].mean(),negative_bce=loss[y==0].mean(),calibrated_bce=(np.logaddexp(0,cal_s)-y*cal_s).mean(),balanced_accuracy=.5*((pred[y==1]==1).mean()+(pred[y==0]==0).mean()),baseline=best,baseline_auc=roc_auc_score(y,bs),test_best_single=max(models[k][1]['metrics']['test']['roc_auc'] for k in ['A','B']),cosine_auc=roc_auc_score(y,arr['cosine_scores'][test])))
   if name in ['A','B']:continue
   cr=credit(y,s);gain,damage=ranking_turnover(y,bs,s)
   for label in [0,1]:
    mask=y==label;correct=pred==y;basecorrect=bp==y
    transitions.append(dict(pair=pair,target=target,model=name,label=label,n=int(mask.sum()),lost=int((mask&basecorrect&~correct).sum()),gained=int((mask&~basecorrect&correct).sum()),both_wrong=int((mask&~basecorrect&~correct).sum()),both_right=int((mask&basecorrect&correct).sum()),rank_recovered=gain,rank_damaged=damage))
   covpath=C/f'{target}.npz'
   if covpath.exists():
    cov=dict(np.load(covpath));assert all(np.array_equal(cov[k],ref[k]) for k in ['u','v','labels','validation_mask'])
    for feature in ['context_min_degree','min_sampled_neighbors','node_cosine','neighbor_cosine','node_mean_norm']:
     values=cov[feature]
     for label in [0,1]:
      mask=y==label
      if feature=='min_sampled_neighbors':bins=np.digitize(values[test],[1,5,10]);names=['0','1–4','5–9','10']
      else:
       cuts=np.quantile(values[val&(ref['labels']==label)],[.25,.5,.75]);bins=np.digitize(values[test],cuts);names=['Q1','Q2','Q3','Q4']
      for j,binname in enumerate(names):
       sel=mask&(bins==j)
       if sel.sum()<10:continue
       strata.append(dict(pair=pair,target=target,model=name,feature=feature,label=label,bin=binname,n=int(sel.sum()),rank_delta=(cr-bc)[sel].mean(),bce_delta=(loss-bl)[sel].mean(),lost=((bp==y)&(pred!=y))[sel].mean(),gained=((bp!=y)&(pred==y))[sel].mean()))
    worst=np.argsort(cr-bc)[:10]
    for idx in worst:examples.append(dict(pair=pair,target=target,model=name,u=int(ref['u'][test][idx]),v=int(ref['v'][test][idx]),label=int(y[idx]),ranking_delta=float(cr[idx]-bc[idx]),baseline_logit=float(bs[idx]),pair_logit=float(s[idx]),node_cosine=float(cov['node_cosine'][test][idx]),min_context_degree=int(cov['context_min_degree'][test][idx])))
  es=logit(np.clip(ensemble[test],1e-12,1-1e-12));pred=ensemble[test]>=eth
  metrics.append(dict(pair=pair,target=target,model='Ensemble',auc=roc_auc_score(y,es),bce=(np.logaddexp(0,es)-y*es).mean(),calibrated_bce=(np.logaddexp(0,es)-y*es).mean(),balanced_accuracy=.5*(pred[y==1].mean()+(~pred[y==0]).mean()),baseline=best,baseline_auc=roc_auc_score(y,bs)))
for name,data in [('metrics',metrics),('transitions',transitions),('strata',strata),('examples',examples),('selectors',selectors)]:pd.DataFrame(data).to_csv(P/(name+'.csv'),index=False)
d=pd.DataFrame(metrics);d['auc_delta']=100*(d.auc-d.baseline_auc)
print(d.groupby(['pair','model'])[['auc','auc_delta','bce','positive_bce','negative_bce','calibrated_bce','balanced_accuracy','cosine_auc']].mean().round(5).to_string());print(pd.DataFrame(selectors).to_string(index=False))
t=pd.DataFrame(transitions);t['lost_rate']=t.lost/t.n;t['gained_rate']=t.gained/t.n
print(t.groupby(['pair','model'])[['lost_rate','gained_rate','rank_recovered','rank_damaged']].mean().round(5).to_string())

if strata:
 z=pd.DataFrame(strata)
 z=z[z.model=='Interleaved']
 q=z.groupby(['pair','feature','label','bin']).agg(rank_delta=('rank_delta','mean'),bce_delta=('bce_delta','mean'),lost=('lost','mean'),gained=('gained','mean'),n=('n','sum'),targets=('target','nunique')).reset_index()
 q.to_csv(P/'aggregate_strata.csv',index=False)
 print('AGGREGATE_STRATA')
 print(q.round(5).to_csv(index=False))
