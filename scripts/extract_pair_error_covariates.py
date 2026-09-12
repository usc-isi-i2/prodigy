"""Read-only CPU extraction of covariates for existing evaluation pairs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch

SOURCES=('ukr_rus_twitter','covid19_twitter','midterm','covid_political','ukr_rus_suspended','twibot20','cp_hk_twitter','facebook_page_reference')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
 out=Path(args.output)
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);torch.set_num_threads(4)
 cache=Path('/dataMeR1/phil/gfm/mixture-scaling-disjoint-neighbor/state/disjoint_context_s0/_cache')
 pairs=Path('/dataMeR1/phil/gfm/mixture-scaling-uniform-eval/state/uniform_eval_s0_v2/node_neighbors_disjoint_uniform')
 for source in SOURCES:
  d=torch.load(cache/f'{source}.pt',map_location='cpu',weights_only=False)
  c=torch.load(cache/f'{source}_neighbors10_disjoint50.pt',map_location='cpu',weights_only=False)
  a=np.load(pairs/f'ss_ukr_rus_twitter__to__{source}.scores.npz');u=a['u'];v=a['v'];n=len(d['x'])
  context=d['context'].numpy();degree=np.bincount(context.ravel(),minlength=n)
  known=d['known_keys'].numpy();full_degree=np.bincount(np.concatenate((known//n,known%n)),minlength=n)
  features={k:a[k] for k in ('u','v','labels','validation_mask')}
  features.update(context_min_degree=np.minimum(degree[u],degree[v]),context_max_degree=np.maximum(degree[u],degree[v]),
                  full_min_degree=np.minimum(full_degree[u],full_degree[v]),full_max_degree=np.maximum(full_degree[u],full_degree[v]),
                  min_sampled_neighbors=np.minimum((c['neighbor_ids'][u]>=0).sum(1).numpy(),(c['neighbor_ids'][v]>=0).sum(1).numpy()))
  for name,x in [('node',d['x']),('neighbor',c['means'])]:
   cos=[];norm=[]
   for start in range(0,len(u),1024):
    xu=x[u[start:start+1024]].float();xv=x[v[start:start+1024]].float()
    cos.extend(torch.nn.functional.cosine_similarity(xu,xv,dim=1).tolist())
    norm.extend(((xu.norm(dim=1)+xv.norm(dim=1))/2).tolist())
   features[name+'_cosine']=np.array(cos);features[name+'_mean_norm']=np.array(norm)
  np.savez_compressed(out/f'{source}.npz',**features)
  (out/f'{source}.json').write_text(json.dumps(dict(source=source,split_receipt=d['receipt'],context_definition='unique undirected context edges only',full_degree_definition='all known undirected edges, descriptive only',pair_reference=str(pairs/f'ss_ukr_rus_twitter__to__{source}.scores.npz')),indent=2))
  print(source,flush=True);del d,c,features
if __name__=='__main__':main()
