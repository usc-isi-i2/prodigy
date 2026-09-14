from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
p=Path('/tmp/mlp-error-analysis');root=p/'dataMeR1/phil/gfm';rows=[]
selectors=pd.read_csv(p/'selectors.csv')
for pair,a,b in [('Ukraine + Facebook','ukr_rus_twitter','facebook_page_reference'),('COVID political + suspended','covid_political','ukr_rus_suspended')]:
 for _,r in selectors[selectors.pair==pair].iterrows():
  t=r.target;source=a if r.validation_selected=='A' else b
  base=dict(np.load(root/f'mixture-scaling-bias-lp/state/bias_s0/results/node_neighbors_bias/ss_{source}__to__{t}.scores.npz'))
  inter=dict(np.load(root/f'mixture-scaling/state/interleaved_mlp_pairs_noelec_s0/results/node_neighbors_bias/interleave_{a}__and__{b}__to__{t}.scores.npz'))
  mask=(~base['validation_mask'])&(abs(base['cosine_scores'])>1e-7)&(abs(inter['cosine_scores'])>1e-7)
  y=base['labels'][mask];bc=base['cosine_scores'][mask];ic=inter['cosine_scores'][mask]
  bn=base['raw_dot_scores'][mask]/bc;inn=inter['raw_dot_scores'][mask]/ic
  assert np.all(bn>=0) and np.all(inn>=0)
  for name,s in [('Baseline dot',bc*bn),('Interleaved dot',ic*inn),('Baseline magnitudes + interleaved angles',ic*bn),('Interleaved magnitudes + baseline angles',bc*inn),('Baseline cosine',bc),('Interleaved cosine',ic)]:
   rows.append(dict(pair=pair,target=t,score=name,auc=roc_auc_score(y,s),n=len(y),excluded=int((~base['validation_mask']).sum()-len(y))))
d=pd.DataFrame(rows);d.to_csv(p/'norm_swap.csv',index=False);print(d.groupby(['pair','score'])[['auc','excluded']].mean().round(5).to_string())
