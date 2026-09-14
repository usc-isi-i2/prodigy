"""Compact NCN-inspired pooling and one-step, shared-weight completion."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ogbl_collab_compact_joint'))
import run as j
CN_CAP=32
RES_CAP=8
PRIOR=.1


def neighbor_sets(edges,n):
    adj=[set() for _ in range(n)]
    for u,v in edges:
        if u!=v: adj[int(u)].add(int(v));adj[int(v)].add(int(u))
    return adj


def common_table(pairs,adj,priority):
    n=len(adj); nodes=np.full((len(pairs),CN_CAP),n,np.int32);scale=np.zeros(len(pairs),np.float32)
    for i,(u,v) in enumerate(pairs):
        values=adj[int(u)]&adj[int(v)]
        if u==v: values=set()
        selected=sorted(values,key=lambda x:int(priority[x]))[:CN_CAP]
        if selected: nodes[i,:len(selected)]=selected;scale[i]=len(values)/len(selected)
    return nodes,scale


def build_cache(edges,pairs,n):
    adj=neighbor_sets(edges,n)
    priority=np.random.default_rng(314159).permutation(n)
    cn,scale=common_table(pairs,adj,priority)
    candidates=np.full((len(pairs),2*RES_CAP),n,np.int32)
    cs=np.zeros_like(candidates,dtype=np.float32)
    missing=np.zeros((len(pairs),2*RES_CAP,2),np.int32)
    for i,(u,v) in enumerate(pairs):
        u,v=int(u),int(v)
        if u==v: continue
        for side,(a,b) in enumerate(((u,v),(v,u))):
            values=adj[a]-adj[b]-{u,v}
            chosen=sorted(values,key=lambda x:int(priority[x]))[:RES_CAP]
            if chosen:
                sl=slice(side*RES_CAP,side*RES_CAP+len(chosen))
                candidates[i,sl]=chosen;cs[i,sl]=len(values)/len(chosen)
                missing[i,sl,0]=b;missing[i,sl,1]=chosen
    flat=np.sort(missing.reshape(-1,2),axis=1)
    children,inverse=np.unique(flat,axis=0,return_inverse=True)
    child_cn,child_scale=common_table(children,adj,priority)
    return dict(pairs=np.asarray(pairs,dtype=np.int32),cn=cn,scale=scale,candidates=candidates,
                candidate_scale=cs,child_index=inverse.reshape(candidates.shape).astype(np.int32),
                child_pairs=children,child_cn=child_cn,child_scale=child_scale)


def tensor_cache(cache,device):
    return {k:torch.as_tensor(cache[k],dtype=torch.float32 if k in ('scale','candidate_scale','child_scale') else torch.long,device=device) for k in cache.files if k not in ('npositive',)}


class Model(j.Model):
    def __init__(self):
        super().__init__('joint') # Preserve the paired encoder's initialization.
        self.project=nn.Sequential(nn.Linear(256,64),nn.ReLU())
        self.decoder=nn.Sequential(nn.Linear(576,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
        assert sum(x.numel() for x in self.parameters())==525633

    def encode(self,x,adj):
        z=self.encoder(x,adj)
        projected=self.project(z)
        projected=torch.cat([projected,torch.zeros((1,64),device=z.device,dtype=z.dtype)])
        return z,projected

    def score_rows(self,z,cache,indices,weights=None,child=False):
        emb,projected=z
        prefix='child_' if child else ''
        edges=cache[prefix+'pairs'][indices]
        cn=cache[prefix+'cn'][indices]
        pooled=projected[cn].sum(1)*cache[prefix+'scale'][indices,None]
        if weights is not None:
            assert not child
            w=weights[cache['child_index'][indices]]*cache['candidate_scale'][indices]
            pooled=pooled+(projected[cache['candidates'][indices]]*w[:,:,None]).sum(1)
        left,right=emb[edges[:,0]],emb[edges[:,1]]
        result=self.decoder(torch.cat([left*right,(left-right).abs(),pooled],dim=1)).flatten()
        return result.masked_fill(edges[:,0]==edges[:,1],-1e9)


@torch.no_grad()
def child_weights(model,z,cache,batch=2048):
    values=[]
    for start in range(0,len(cache['child_pairs']),batch):
        ids=torch.arange(start,min(start+batch,len(cache['child_pairs'])),device=z[0].device)
        logits=model.score_rows(z,cache,ids,child=True)
        values.append(torch.sigmoid(logits+np.log(PRIOR)))
    return torch.cat(values)


@torch.no_grad()
def all_scores(model,z,cache,start,stop,weights=None,batch=2048):
    values=[]
    for offset in range(start,stop,batch):
        ids=torch.arange(offset,min(offset+batch,stop),device=z[0].device)
        values.append(model.score_rows(z,cache,ids,weights))
    return torch.cat(values)
