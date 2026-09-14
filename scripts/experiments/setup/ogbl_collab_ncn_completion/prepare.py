"""Build label-independent neighborhood caches from admitted pre-target graphs."""
import argparse,time
from pathlib import Path
import numpy as np
from core import j,build_cache
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
manifest=j.read(a.source/'prepared.json');files={}
for path in [a.source/'train2017.npz',a.original/'assessment/year2018.npz',a.original/'node_features.npy',a.original/'standardization.npz']:
    digest=j.shared.sha256_file(path);assert digest==manifest['files'][str(path)];files[str(path)]=digest
n=len(np.load(a.original/'node_features.npy',mmap_mode='r'))
import wandb
w=wandb.init(project='ogbl-collab-compact-joint',group='ncn-completion-v1',name='prepare',mode='offline',dir=str(a.out),config=dict(revision=j.revision(),files=files,test_access=False))
started=time.monotonic();rows=[]
for name,path in [('train',a.source/'train2017.npz'),('validation',a.original/'assessment/year2018.npz')]:
    t=time.monotonic();panel=np.load(path);pairs=np.concatenate([panel['pos'],panel['neg']])
    cache=build_cache(panel['graph_edges'],pairs,n)
    out=a.out/f'{name}.npz';np.savez_compressed(out,**cache,npositive=len(panel['pos']))
    row=dict(name=name,targets=len(pairs),children=len(cache['child_pairs']),nonempty_common=int((cache['scale']>0).sum()),candidate_count=int((cache['candidate_scale']>0).sum()),seconds=time.monotonic()-t,sha256=j.shared.sha256_file(out))
    rows.append(row);w.log(row);print(row,flush=True)
j.save(a.out/'prepared.json',dict(complete=True,revision=j.revision(),files=files,caches=rows,seconds=time.monotonic()-started,test_access=False,wandb_directory=w.dir));w.finish()
