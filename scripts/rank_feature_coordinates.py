"""Coordinatewise empirical KS across fixed nonzero-node samples."""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import ks_2samp
from mixture_scaling.config import load_config
from mixture_scaling.evaluate_lattice import unwrap

torch.set_num_threads(4)
root=Path('/dataMeR1/phil/gfm/mixture-scaling/results');out=root/'coordinate_ks_nonzero_corrected_s2026';out.mkdir(parents=True,exist_ok=False)
config=load_config('configs/node_only_transfer.yaml');samples={};meta={};start=time.monotonic()
for name,entry in config['graphs'].items():
 path=entry['path']
 if name=='ukr_rus_suspended':path=str(root/'suspended_csv_repair_20260911_v2/graphs/retweet_graph_suspended.pt')
 g=unwrap(path);rng=np.random.default_rng(2026);ids=rng.choice(len(g.x),min(len(g.x),30000),replace=False);x=g.x[torch.from_numpy(ids)].float().numpy().copy();del g
 x=x[np.any(x!=0,axis=1)];assert len(x)>=20000 and np.isfinite(x).all();samples[name]=x[:20000];meta[name]={'path':path,'candidate_nodes':len(ids),'nonzero_candidates':len(x),'used':20000}
 print('sampled',name,flush=True)
np.savez_compressed(out/'samples.npz',**samples)
names=list(samples);pairs=[];dist=[]
for i,a in enumerate(names):
 for b in names[i+1:]:
  vals=ks_2samp(samples[a],samples[b],axis=0,method='asymp').statistic
  assert vals.shape==(768,);pairs.append([a,b]);dist.append(vals.tolist())
 print('compared',a,round(time.monotonic()-start,1),flush=True)
d=np.array(dist);mask=np.array(['facebook_page_reference' not in p for p in pairs]);allmean=d.mean(0);twittermean=d[mask].mean(0)
top=np.argsort(-allmean,kind='stable')[:5];topt=np.argsort(-twittermean,kind='stable')[:5];chosen=sorted(set(top.tolist()+topt.tolist()))
result={'seed':2026,'sampling':meta,'pairs':pairs,'ks':dist,'mean_all':allmean.tolist(),'mean_without_facebook':twittermean.tolist(),'top5_all':top.tolist(),'top5_without_facebook':topt.tolist(),'dimension_indexing':'zero-based','seconds':time.monotonic()-start,'plot_samples':{n:{str(k):x[:,k].tolist() for k in chosen} for n,x in samples.items()}}
(out/'results.json').write_text(json.dumps(result));print('TOP_ALL',top.tolist(),allmean[top].tolist(),'TOP_NO_FB',topt.tolist(),twittermean[topt].tolist(),flush=True)
