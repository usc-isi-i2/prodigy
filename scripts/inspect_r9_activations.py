"""Read-only sampled-node activation diagnostics for the selected rung-9 model."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mixture_scaling.config import load_config
from mixture_scaling.evaluate_lattice import unwrap
from mixture_scaling.model import NodeMLP

torch.set_num_threads(4)
root=Path('/dataMeR1/phil/gfm/mixture-scaling')
c=torch.load(root/'state/node_mlp_ladder_s0_convergence_fast/lp/ladder_r9/best.pt',map_location='cpu',weights_only=False)
m=NodeMLP(768,256,256,0);m.load_state_dict(c['model']);m.eval()
config=load_config('configs/node_only_transfer.yaml')

def stats(x):
 x=x.double(); centered=x-x.mean(0); eigen=torch.linalg.eigvalsh(centered.T@centered/(len(x)-1)).clamp_min(0).flip(0); total=eigen.sum();p=eigen/total
 norms=x.norm(dim=1); u=F.normalize(x,dim=1); cos=(u[:len(u)//2]*u[len(u)//2:2*(len(u)//2)]).sum(1)
 return {'variance_spectrum':p.tolist(),'top1_variance':p[0].item(),'top10_variance':p[:10].sum().item(),'effective_rank':torch.exp(-(p[p>0]*p[p>0].log()).sum()).item(),'dims90':int((p.cumsum(0)<.9).sum()+1),'norm_quantiles':torch.quantile(norms,torch.tensor([0.,.5,1.],dtype=torch.double)).tolist(),'cosine_quantiles':torch.quantile(cos,torch.tensor([.05,.5,.95],dtype=torch.double)).tolist(),'mean_energy_fraction':(x.mean(0).square().sum()/x.square().sum(1).mean()).item(),'dead_dimensions':int((x.abs().max(0).values<1e-10).sum())}

result={'checkpoint_step':c['step'],'sampling':'4096 uniform nodes without replacement per graph (or all if fewer), seed 2026; not edge-weighted or held-out-only','graphs':{}}
with torch.no_grad():
 for name,entry in config['graphs'].items():
  g=unwrap(entry['path']); n=g.x.shape[0];ids=np.random.default_rng(2026).choice(n,min(n,4096),replace=False);x=g.x[torch.from_numpy(ids)].float().clone();del g
  pre=m.network[0](x);h=F.relu(pre);z=m.network[3](h)
  result['graphs'][name]={'sample_size':len(x),'input':stats(x),'hidden':stats(h),'output':stats(z),'hidden_zero_fraction':float((h==0).float().mean())}
  print(name,json.dumps({k:v for k,v in result['graphs'][name]['output'].items() if k!='variance_spectrum'}),flush=True)
Path('activation_results.json').write_text(json.dumps(result,indent=2))
