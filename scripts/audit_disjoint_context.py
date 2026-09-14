"""Read-only audit of actual cached edge partitions and sampled contexts."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--original-root',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(4);reports=[]
for file in sorted((args.root/'_cache').glob('*.pt')):
 if '_neighbors10_' in file.name:continue
 data=torch.load(file,map_location='cpu',weights_only=False)
 old=torch.load(args.original_root/'_cache'/file.name,map_location='cpu',weights_only=False)
 n=len(data['x']);encode=lambda e:np.minimum(e[0].numpy(),e[1].numpy())*n+np.maximum(e[0].numpy(),e[1].numpy())
 for k in ['train','validation','test','known_keys']:assert torch.equal(data[k],old[k]),(file.name,k)
 del old
 groups={k:encode(data[k]) for k in ['context','supervision','validation','test']}
 joined=np.concatenate(list(groups.values()))
 assert np.array_equal(np.sort(joined),data['known_keys'].numpy()),file.name
 sample=torch.load(file.with_name(file.stem+'_neighbors10_disjoint50.pt'),map_location='cpu',weights_only=False)
 ids=sample['neighbor_ids'].numpy();u=np.repeat(np.arange(n,dtype=np.int64),10);v=ids.reshape(-1);keep=v>=0;u=u[keep];v=v[keep]
 keys=np.minimum(u,v)*n+np.maximum(u,v);context=np.sort(groups['context'])
 index=np.searchsorted(context,keys).clip(0,len(context)-1)
 assert np.array_equal(context[index],keys),file.name
 reports.append(dict(source=file.stem,context_edges=len(groups['context']),supervised_edges=len(groups['supervision']),
                     sampled_neighbor_links=len(keys),supervision_context_overlap=0,heldout_context_overlap=0,original_splits_unchanged=True))
 print(json.dumps(reports[-1]),flush=True)
 del data,sample
assert len(reports)==9
(args.root/'disjoint_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
