"""Pairwise KS distances for each raw feature coordinate, with zero-row sensitivity."""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import ks_2samp
from mixture_scaling.config import load_config
from mixture_scaling.evaluate_lattice import unwrap

torch.set_num_threads(4)
out=Path('/dataMeR1/phil/gfm/mixture-scaling/results/feature_dimension_ks_s2026');out.mkdir(parents=True,exist_ok=False)
cfg=load_config('configs/node_only_transfer.yaml');samples={};meta={};started=time.monotonic()
for name,g in cfg['graphs'].items():
 graph=unwrap(g['path']);n=len(graph.x);ids=np.random.default_rng(2026).choice(n,min(n,20000),replace=False);x=graph.x[torch.from_numpy(ids)].float().numpy().copy();del graph
 assert x.shape[1]==768 and np.isfinite(x).all()
 samples[name]=x;zero=np.all(x==0,axis=1);meta[name]={'path':g['path'],'n_nodes':n,'sample_size':len(x),'all_zero_rows':int(zero.sum()),'all_zero_fraction':float(zero.mean())};print('sampled',name,meta[name],flush=True)
np.savez_compressed(out/'sampled_features.npz',**samples)
names=list(samples);n_all=min(10000,min(map(len,samples.values())));n_nonzero=min(10000,min(int(np.any(x!=0,axis=1).sum()) for x in samples.values()));assert n_nonzero>=100
result={'seed':2026,'graphs':names,'metadata':meta,'sample_size_all':n_all,'sample_size_nonzero':n_nonzero,'feature_dimensions':768,'method':'Two-sided empirical KS statistic, per raw coordinate; same sampled rows reused for every pair; symmetric matrices; no normalization. Nonzero excludes exactly all-zero feature rows, not inferred missing text.','pairs':[]}
for i,a in enumerate(names):
 for j in range(i+1,len(names)):
  b=names[j];row={'a':a,'b':b}
  for mode,n in [('all',n_all),('nonzero',n_nonzero)]:
   x=samples[a];y=samples[b]
   if mode=='nonzero':x=x[np.any(x!=0,axis=1)];y=y[np.any(y!=0,axis=1)]
   x=x[:n];y=y[:n]
   d=[float(ks_2samp(x[:,k],y[:,k],method='asymp').statistic) for k in range(768)]
   row[mode]={'ks':d,'mean':float(np.mean(d)),'median':float(np.median(d)),'p90':float(np.quantile(d,.9)),'max':max(d),'top_dimension':int(np.argmax(d))}
  result['pairs'].append(row);print('pair',a,b,row['nonzero']['mean'],flush=True)
result['elapsed_seconds']=time.monotonic()-started
(out/'results.json').write_text(json.dumps(result));print('DONE',result['elapsed_seconds'],flush=True)
