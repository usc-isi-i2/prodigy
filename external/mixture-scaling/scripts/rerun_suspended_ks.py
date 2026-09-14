import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import ks_2samp
from mixture_scaling.evaluate_lattice import unwrap
root=Path('/dataMeR1/phil/gfm/mixture-scaling/results');old=root/'feature_dimension_ks_s2026';out=root/'feature_dimension_ks_suspended_v2_s2026';out.mkdir(parents=True,exist_ok=False)
artifact=root/'suspended_csv_repair_20260911_v2/graphs/retweet_graph_suspended.pt'
torch.set_num_threads(4);g=unwrap(artifact);n=len(g.x);ids=np.random.default_rng(2026).choice(n,min(n,20000),replace=False);x=g.x[torch.from_numpy(ids)].float().numpy().copy();assert x.shape[1]==768 and np.isfinite(x).all();del g
r=json.loads((old/'results.json').read_text());r['previous_suspended_metadata']=r['metadata']['ukr_rus_suspended'];r['metadata']['ukr_rus_suspended']={'path':str(artifact),'n_nodes':n,'sample_size':len(x),'all_zero_rows':int(np.all(x==0,axis=1).sum()),'all_zero_fraction':float(np.all(x==0,axis=1).mean())};r['rerun_note']='Only eight Suspended pairs replaced; all other graph samples and results unchanged. Repaired artifact changes graph population, so not a matched-node comparison.'
archive=np.load(old/'sampled_features.npz');before=[]
for row in r['pairs']:
 if 'ukr_rus_suspended' not in [row['a'],row['b']]:continue
 before.append(json.loads(json.dumps(row)));other=row['b'] if row['a']=='ukr_rus_suspended' else row['a'];y=archive[other]
 for mode in ['all','nonzero']:
  a=x;b=y;count=r['sample_size_'+mode]
  if mode=='nonzero':a=a[np.any(a!=0,axis=1)];b=b[np.any(b!=0,axis=1)]
  assert min(len(a),len(b))>=count
  a=a[:count];b=b[:count];d=[float(ks_2samp(a[:,k],b[:,k],method='asymp').statistic) for k in range(768)]
  row[mode]={'ks':d,'mean':float(np.mean(d)),'median':float(np.median(d)),'p90':float(np.quantile(d,.9)),'max':max(d),'top_dimension':int(np.argmax(d))}
 print(other,'old',before[-1]['nonzero']['mean'],'new',row['nonzero']['mean'],flush=True)
r['previous_suspended_pairs']=before;(out/'results.json').write_text(json.dumps(r));print('DONE',r['metadata']['ukr_rus_suspended'],flush=True)
