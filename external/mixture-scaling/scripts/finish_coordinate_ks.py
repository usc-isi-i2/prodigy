import json,time
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
out=Path('/dataMeR1/phil/gfm/mixture-scaling/results/coordinate_ks_nonzero_corrected_s2026');start=time.monotonic()
samples=dict(np.load(out/'samples.npz'));names=list(samples);sortedx={k:np.sort(v,axis=0) for k,v in samples.items()};pairs=[];dist=[]
for i,a in enumerate(names):
 for b in names[i+1:]:
  vals=[]
  for k in range(768):
   aa=sortedx[a][:,k];bb=sortedx[b][:,k];support=np.union1d(aa,bb);v=np.max(np.abs(np.searchsorted(aa,support,side='right')/len(aa)-np.searchsorted(bb,support,side='right')/len(bb)));vals.append(float(v))
  assert np.isclose(vals[0],ks_2samp(samples[a][:,0],samples[b][:,0],method='asymp').statistic)
  pairs.append([a,b]);dist.append(vals)
 print('compared',a,round(time.monotonic()-start,1),flush=True)
d=np.array(dist);mask=np.array(['facebook_page_reference' not in p for p in pairs]);allmean=d.mean(0);twittermean=d[mask].mean(0);top=np.argsort(-allmean,kind='stable')[:5];topt=np.argsort(-twittermean,kind='stable')[:5];chosen=sorted(set(top.tolist()+topt.tolist()))
r={'seed':2026,'nodes_per_graph':20000,'zero_vectors_excluded':True,'suspended':'corrected candidate','pairs':pairs,'ks':dist,'mean_all':allmean.tolist(),'mean_without_facebook':twittermean.tolist(),'top5_all':top.tolist(),'top5_without_facebook':topt.tolist(),'dimension_indexing':'zero-based','seconds':time.monotonic()-start,'plot_samples':{n:{str(k):x[:,k].tolist() for k in chosen} for n,x in samples.items()}}
(out/'results_direct.json').write_text(json.dumps(r));print('TOP',top.tolist(),allmean[top].tolist(),'NOFB',topt.tolist(),twittermean[topt].tolist(),flush=True)
