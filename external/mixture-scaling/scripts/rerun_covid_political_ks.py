import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.stats import ks_2samp
from mixture_scaling.evaluate_lattice import unwrap
root=Path('/dataMeR1/phil/gfm/mixture-scaling/results');old=root/'feature_dimension_ks_s2026';out=root/'feature_dimension_ks_political_updated_s2026';out.mkdir(parents=True,exist_ok=False)
artifact=Path('/dataMeR1/phil/data/covid_political/graphs/retweet_graph.pt')
assert hashlib.sha256(artifact.read_bytes()).hexdigest()=='755219daede78f913ab897fec51c98f261cff797a9869d80fb7ed35ce0b4d516'
torch.set_num_threads(4);g=unwrap(artifact);n=len(g.x);ids=np.random.default_rng(2026).choice(n,min(n,20000),replace=False);x=g.x[torch.from_numpy(ids)].float().numpy().copy();assert x.shape[1]==768 and np.isfinite(x).all();del g
r=json.loads((root/'feature_dimension_ks_suspended_v2_s2026/results.json').read_text());r['previous_covid_political_metadata']=r['metadata']['covid_political'];r['metadata']['covid_political']={'path':str(artifact),'n_nodes':n,'sample_size':len(x),'all_zero_rows':int(np.all(x==0,axis=1).sum()),'all_zero_fraction':float(np.all(x==0,axis=1).mean())};r['rerun_note']='Eight COVID Political pairs updated from canonical rebuilt raw-bio features; corrected Suspended comparisons retained. Same seed and cached samples for other graphs.'
archive=np.load(old/'sampled_features.npz');before=[]
for row in r['pairs']:
 if 'covid_political' not in [row['a'],row['b']]:continue
 before.append(json.loads(json.dumps(row)));other=row['b'] if row['a']=='covid_political' else row['a'];y=archive[other]
 if other=='ukr_rus_suspended':
  sg=unwrap(root/'suspended_csv_repair_20260911_v2/graphs/retweet_graph_suspended.pt');si=np.random.default_rng(2026).choice(len(sg.x),20000,replace=False);y=sg.x[torch.from_numpy(si)].float().numpy().copy();del sg
 for mode in ['all','nonzero']:
  a=x;b=y;count=r['sample_size_'+mode]
  if mode=='nonzero':a=a[np.any(a!=0,axis=1)];b=b[np.any(b!=0,axis=1)]
  assert min(len(a),len(b))>=count
  a=a[:count];b=b[:count];d=[float(ks_2samp(a[:,k],b[:,k],method='asymp').statistic) for k in range(768)]
  row[mode]={'ks':d,'mean':float(np.mean(d)),'median':float(np.median(d)),'p90':float(np.quantile(d,.9)),'max':max(d),'top_dimension':int(np.argmax(d))}
 print(other,'old',before[-1]['nonzero']['mean'],'new',row['nonzero']['mean'],flush=True)
r['previous_covid_political_pairs']=before;(out/'results.json').write_text(json.dumps(r));print('DONE',r['metadata']['covid_political'],flush=True)
