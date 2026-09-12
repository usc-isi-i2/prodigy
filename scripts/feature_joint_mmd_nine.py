"""Exact unbiased multiscale RBF MMD on fixed nonzero raw bio features."""
import json, time, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist, pdist
import torch
from mixture_scaling.evaluate_lattice import unwrap

root=Path('/dataMeR1/phil/gfm/mixture-scaling/results')
out=root/'feature_joint_mmd_nine_s2026';out.mkdir(exist_ok=True)
started=time.monotonic();torch.set_num_threads(4)
r=json.loads((root/'feature_dimension_ks_political_updated_s2026/results.json').read_text())
names=list(r['graphs'])
archive=np.load(root/'feature_dimension_ks_s2026/sampled_features.npz')
samples={n:archive[n] for n in names}
p=root/'suspended_csv_repair_20260911_v2/graphs/retweet_graph_suspended.pt'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='18baf3fa4e49b767eb1a0f7b670a8aa39ac16e2de0a991ca75a90685496e1cd4'
g=unwrap(p);idx=np.random.default_rng(2026).choice(len(g.x),20000,replace=False)
samples['ukr_rus_suspended']=g.x[torch.from_numpy(idx)].float().numpy().copy();del g
p=Path('/dataMeR1/phil/data/covid_political/graphs/retweet_graph.pt')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='755219daede78f913ab897fec51c98f261cff797a9869d80fb7ed35ce0b4d516'
g=unwrap(p);idx=np.random.default_rng(2026).choice(len(g.x),20000,replace=False)
samples['covid_political']=g.x[torch.from_numpy(idx)].float().numpy().copy();del g
samples={n:x[np.any(x!=0,axis=1)].astype(np.float64) for n,x in samples.items()}
assert all(np.isfinite(x).all() and len(x)>=4000 for x in samples.values())
rng=np.random.default_rng(2026)
pooled=np.concatenate([x[rng.choice(len(x),300,replace=False)] for x in samples.values()])
median=json.loads((root/'feature_joint_mmd_seven_s2026/results.json').read_text())['median_squared_distance']
factors=np.array([.5,1.,2.]);denominators=2*median*factors**2

def kernel_means(x,y,self_kernel=False):
 d=cdist(x,y,'sqeuclidean');v=[]
 for den in denominators:
  k=np.exp(-d/den)
  if self_kernel:np.fill_diagonal(k,0)
  v.append(float(k.sum()/(len(x)*(len(x)-1) if self_kernel else len(x)*len(y))))
 return np.array(v)

results={'graphs':names,'n_per_graph':2000,'repeats':3,'seed':2026,'dimensions':768,'median_squared_distance':median,'sigma_factors':factors.tolist(),'method':'Unbiased MMD squared, RBF exp(-||x-y||^2/(2 sigma^2)), sigma = pooled median Euclidean distance times factor. Mean of three bandwidth kernels; raw features, exact zero rows removed. Same global bandwidths across all pairs. Three repeated subsets from fixed cached samples; repeat spread is not a population confidence interval. Negative estimates allowed. Diagonal is independent same-graph split baseline, not self-comparison.','provenance':'Original feature KS sampled_features.npz, except repaired Suspended sampled with seed 2026 from verified v2 graph. COVID Political resampled from hash-verified repaired canonical graph; Election retains original cleaned-profile cached features. Bandwidths fixed to the seven-graph analysis; repeated subsets resampled for nine graphs.','replicate_matrices':[],'bandwidth_matrices':[]}
for rep in range(3):
 sets={};within={}
 for name,x in samples.items():
  ids=rng.choice(len(x),4000,replace=False);a,b=x[ids[:2000]],x[ids[2000:]]
  sets[name]=(a,b);within[name]=(kernel_means(a,a,True),kernel_means(b,b,True))
 matrix=np.zeros((3,len(names),len(names)))
 for i,a in enumerate(names):
  matrix[:,i,i]=within[a][0]+within[a][1]-2*kernel_means(*sets[a])
  for j in range(i+1,len(names)):
   b=names[j];v=within[a][0]+within[b][0]-2*kernel_means(sets[a][0],sets[b][0])
   matrix[:,i,j]=matrix[:,j,i]=v
 results['bandwidth_matrices'].append(matrix.tolist());results['replicate_matrices'].append(matrix.mean(axis=0).tolist())
 print('completed repeat',rep+1, 'seconds',time.monotonic()-started,flush=True)
a=np.array(results['replicate_matrices']);results['mean']=a.mean(axis=0).tolist();results['std']=a.std(axis=0,ddof=1).tolist();results['elapsed_seconds']=time.monotonic()-started
(out/'results.json').write_text(json.dumps(results,indent=2));print('DONE',results['elapsed_seconds'],flush=True)
