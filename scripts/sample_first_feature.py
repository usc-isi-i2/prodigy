import json
from pathlib import Path
import numpy as np
import torch
from mixture_scaling.config import load_config
from mixture_scaling.evaluate_lattice import unwrap

torch.set_num_threads(4)
r={}
for name,entry in load_config('configs/node_only_transfer.yaml')['graphs'].items():
 g=unwrap(entry['path']);n=g.x.shape[0];idx=np.random.default_rng(2026).choice(n,min(n,100000),replace=False)
 sample=g.x[torch.from_numpy(idx)].float(); keep=sample.ne(0).any(dim=1); values=sample[keep,0].numpy().copy(); removed=int((~keep).sum());del sample,g
 assert np.isfinite(values).all()
 r[name]={'n_nodes':n,'sample_size_before':len(idx),'removed_all_zero':removed,'sample_size':len(values),'values':values.tolist()}
 print(name,len(values),flush=True)
p=Path('/dataMeR1/phil/gfm/mixture-scaling/results/first_feature_hist_nonzero_s2026');p.mkdir(parents=True,exist_ok=True)
(p/'samples.json').write_text(json.dumps(r))
