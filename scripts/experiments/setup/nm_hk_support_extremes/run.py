"""Select nearest/farthest/diverse/random HK supports from a fixed candidate bank."""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import struct
import sys
import time
import zipfile
os.environ.setdefault('WANDB_MODE','disabled')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data,Batch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/experiments/setup/final_core')]
from experiments.sampler import NeighborSampler
from data.dataset import SubgraphDataset,SeededNodeIndex
from scripts.experiments.setup.nm_hk_mechanism.run import build_model,heads,metrics
from scripts.experiments.setup.nm_support_resampling.run import encode,meta,digest_state
from scripts.experiments.setup.nm_support_geometry.run import digest
from evaluate_fixed_grid import git_commit


def tensor_ref(storage,offset,size,stride,*unused):
    return dict(storage=storage,offset=offset,size=size,stride=stride)


class MetadataOnly(pickle.Unpickler):
    """Read trusted torch-save metadata without materializing tensor storages."""
    def persistent_load(self,pid):
        assert pid[0]=='storage'
        return dict(type=pid[1].__name__,key=pid[2],count=pid[4])
    def find_class(self,module,name):
        if module=='torch._utils' and name in ['_rebuild_tensor','_rebuild_tensor_v2']:
            return tensor_ref
        return super().find_class(module,name)


def extract_views(path,offset,count):
    # Torch's old runtime lacks mmap loading. Map only uncompressed int64 edge
    # storages from the standard zip archive; never read the feature storage.
    out={}
    with zipfile.ZipFile(path) as z:
        meta_name=next(n for n in z.namelist() if n.endswith('/data.pkl'))
        prefix=meta_name[:-len('data.pkl')]
        with z.open(meta_name) as f: raw=MetadataOnly(f).load()
        for name,key in [('train','edge_index_views'),('validation','target_edge_index_views'),('test','target_edge_index_views')]:
            ref=raw[key]['static_'+name];st=ref['storage']
            assert st['type']=='LongStorage' and ref['size'][0]==2
            entry=z.getinfo(prefix+'data/'+st['key']);assert entry.compress_type==zipfile.ZIP_STORED
            with path.open('rb') as f:
                f.seek(entry.header_offset+26);fn,extra=struct.unpack('<HH',f.read(4))
            start=entry.header_offset+30+fn+extra
            storage=np.memmap(path,dtype='<i8',mode='r',offset=start,shape=(st['count'],))
            edges=np.ndarray(shape=ref['size'],dtype='<i8',buffer=storage,
                offset=ref['offset']*8,strides=tuple(s*8 for s in ref['stride']))
            keep=(edges[0]>=offset)&(edges[0]<offset+count)
            selected=np.array(edges[:,keep],copy=True)-offset
            assert ((selected>=0)&(selected<count)).all()
            out[name]=torch.from_numpy(selected)
            print('extracted',name,selected.shape,flush=True)
    return out


def edgekeys(e,n,undirected=False):
    e=e.numpy()
    a,b=(np.minimum(e[0],e[1]),np.maximum(e[0],e[1])) if undirected else (e[0],e[1])
    return np.unique(a*n+b)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('/dataMeR1/phil/gfm/error_audit/nm_hk_mechanism_20260908'))
    p.add_argument('--inputs',type=Path,default=Path('/dataMeR1/phil/gfm/error_audit/nm_complete_inputs_20260908_v2'))
    p.add_argument('--audit',type=Path,default=Path('/dataMeR1/phil/gfm/error_audit/nm_canonical_split_20260908'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--candidate-cap',type=int,default=32)
    p.add_argument('--seed',type=int,default=90837)
    a=p.parse_args();assert a.candidate_cap>=3;a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(a.seed)
    started=time.time()
    receipt=json.loads((a.cache/'receipt.json').read_text());assert receipt['complete']
    cachepath=a.cache/'embeddings_and_logits_private.pt';assert digest(cachepath)==receipt['cache_sha256']
    cache=torch.load(cachepath,map_location='cpu',weights_only=False);cases=cache['cases'];assert len(cases)==200
    inputs=json.loads((a.inputs/'receipt.json').read_text());offset=inputs['targets']['cp_hk']['source_node_offset']
    raw=torch.load(receipt['standalone_features'],map_location='cpu',weights_only=False)
    features=raw['x'];n=len(features);assert n==333800
    views=extract_views(Path(inputs['feature_artifact']),offset,n)
    original_edges=edgekeys(raw['edge_index'],n)
    assert np.array_equal(original_edges,np.unique(np.concatenate([edgekeys(e,n) for e in views.values()])))
    uk={k:edgekeys(e,n,True) for k,e in views.items()}
    assert all(len(np.intersect1d(uk[x],uk[y],assume_unique=True))==0 for x,y in [('train','validation'),('train','test'),('validation','test')])
    torch.save(dict(views=views,offset=offset,feature_path=receipt['standalone_features'],source_artifact=inputs['feature_artifact']),a.out/'hk_canonical_views_private.pt')
    del raw
    sampler=NeighborSampler(Data(edge_index=views['train'],num_nodes=n),num_hops=2,hop_sizes=[9,9],limit=101,walk_hops=1)
    dataset=SubgraphDataset(Data(x=features,edge_index=views['train'],num_nodes=n),sampler,bidirectional=False)
    # This checkpoint has use_edge_features=False; GraphSAGE ignores edge attrs.
    t=views['test'].numpy();row=np.r_[t[0],t[1]];col=np.r_[t[1],t[0]]
    order=np.argsort(row,kind='stable');row=row[order];col=col[order]
    ptr=np.r_[0,np.cumsum(np.bincount(row,minlength=n))]
    params=json.loads((a.audit/'effective_config.json').read_text());params['device']=a.device
    model=build_model(params,cache['checkpoint'],a.device);before=digest_state(model)
    assert before==cache['model_state_sha256']
    plans=torch.load(a.audit/'cp_hk'/'test_plan_private.pt',map_location='cpu',weights_only=False)
    banks={};all_ids=set()
    for c in cases:
        plan=plans[c['batch']][0][c['ep']]
        queryids={int(v)-offset for members in plan.values() for v in members[3:]}
        anchor=int(c['anchor']);banned=queryids|{anchor}|{int(c['true_support_'+str(j)]) for j in range(3)}
        pool=np.array(sorted(set(col[ptr[anchor]:ptr[anchor+1]])-banned),dtype=np.int64)
        assert len(pool)==c['alternative_pool_size'] and len(pool)>=3,(c['case'],len(pool),c['alternative_pool_size'])
        for trial in c['trials']:assert set(int(v)-offset for v in trial['alternative']).issubset(set(pool))
        rng=np.random.default_rng(a.seed+1009*c['case'])
        bank=np.sort(rng.choice(pool,min(len(pool),a.candidate_cap),replace=False))
        banks[c['case']]=bank;all_ids.update(bank.tolist())
    ids=sorted(all_ids);print('unique candidate centers',len(ids),flush=True)
    graphs=[]
    for node in ids:
        seed=int((a.seed*1000003+node)%2147483646)+1
        graphs.append(dataset[SeededNodeIndex(node,seed)])
    with torch.inference_mode():
        encoded=[]
        for k in range(0,len(graphs),128):encoded.append(encode(model,Batch.from_data_list(graphs[k:k+128]),a.device).cpu())
        encoded=torch.cat(encoded);lookup={v:i for i,v in enumerate(ids)}
        selections=[];rows=[];all_logits=[];prob_error=0.
        old=pd.read_csv(a.cache/'head_comparison_private.csv');base=old[old.condition=='original'].set_index('case')
        previous_any=old[old.condition!='original'].groupby('case').native_correct.max()
        for c in cases:
            args=[v.to(a.device) for v in cache['episodes'][c['batch'],c['ep']]]
            original=meta(model,*args)[c['local']]
            assert int(original.argmax()==c['truth'])==int(base.loc[c['case'],'native_correct'])
            err=abs(float(original.softmax(0)[c['truth']])-base.loc[c['case'],'native_probability']);assert err<1e-5
            prob_error=max(prob_error,err)
            indices=np.array([lookup[int(v)] for v in banks[c['case']]])
            z=F.normalize(encoded[indices],dim=1);q=F.normalize(args[0][c['local']].cpu(),dim=0)
            sims=(z@q).numpy();order=np.argsort(sims,kind='stable')
            near=order[-3:];far=order[:3]
            diversity=[int(order[-1])]
            while len(diversity)<3:
                distance=1-(z@z[diversity].T).max(dim=1).values.numpy();distance[diversity]=-np.inf
                diversity.append(int(distance.argmax()))
            rng=np.random.default_rng(a.seed+2003*c['case'])
            methods=[('nearest',0,near),('farthest',0,far),('diverse',0,np.array(diversity))]
            methods += [('random',draw,rng.choice(len(indices),3,replace=False)) for draw in range(5)]
            for method,draw,chosen in methods:
                # Canonical ID order removes any role-order difference for identical triples.
                chosen=np.sort(chosen);ix=indices[chosen];centers=[ids[v] for v in ix]
                assert len(set(centers))==3
                changed=args[0].clone();slots=[c['truth']*7+j for j in range(3)]
                changed[slots]=encoded[ix].to(a.device)
                keep=torch.ones(210,dtype=torch.bool,device=a.device);keep[slots]=False
                assert torch.equal(changed[keep],args[0][keep])
                logits=meta(model,changed,*args[1:])[c['local']]
                proto,avg=heads(changed,c['local'],model)
                entry=dict(case=c['case'],cohort=c['cohort'],method=method,draw=draw,
                    previously_persistent=bool(c['cohort']=='native_failed' and previous_any.loc[c['case']]==0),
                    pool_size=c['alternative_pool_size'],bank_size=len(indices),
                    selected_mean_cosine=float(sims[chosen].mean()),
                    original_mean_cosine=float((F.normalize(args[0][slots],dim=1)@F.normalize(args[0][c['local']],dim=0)).mean()),
                    baseline_correct=int(c['hk_correct']))
                for label,values in [('native',logits),('prototype',proto),('mean_cosine',avg)]:entry.update(metrics(values,c['truth'],label))
                rows.append(entry);all_logits.append(torch.stack([logits,proto,avg]).cpu())
                selections.append(dict(case=c['case'],method=method,draw=draw,centers=centers,embedding_indices=ix.tolist()))
        assert digest_state(model)==before
    df=pd.DataFrame(rows);assert len(df)==1600
    assert df[df.method=='nearest'].previously_persistent.sum()==71
    # Selection must bracket all random/diverse averages on the exact candidate bank.
    for case,s in df.groupby('case'):
        low=float(s.loc[s.method=='farthest','selected_mean_cosine'].iloc[0]);high=float(s.loc[s.method=='nearest','selected_mean_cosine'].iloc[0])
        assert s.selected_mean_cosine.between(low-1e-6,high+1e-6).all()
    df.to_csv(a.out/'selection_results_private.csv',index=False)
    torch.save(dict(graphs=graphs,global_candidate_ids=[v+offset for v in ids],
        source_local_candidate_ids=ids,embeddings=encoded,banks=banks,selections=selections,
        logits=torch.stack(all_logits),logit_row_keys=df[['case','method','draw']].to_dict('records'),
        head_order=['native','prototype','mean_cosine']),a.out/'candidate_bank_private.pt')
    report=dict(complete=True,revision=git_commit(),seconds=time.time()-started,seed=a.seed,
        candidate_cap=a.candidate_cap,unique_candidate_nodes=len(ids),cases=200,rows=len(df),
        canonical_view_edges={k:int(v.shape[1]) for k,v in views.items()},
        source_full_edge_union_matches=True,undirected_splits_disjoint=True,
        all_200_pool_sizes_match_previous_run=True,all_previous_alternative_ids_in_pools=True,
        max_baseline_probability_error=prob_error,model_state_sha256=before,model_unchanged=True,
        candidate_bank_sha256=digest(a.out/'candidate_bank_private.pt'),
        canonical_views_sha256=digest(a.out/'hk_canonical_views_private.pt'),
        csv_sha256=digest(a.out/'selection_results_private.csv'),
        prior_cache_sha256=receipt['cache_sha256'],feature_artifact=receipt['standalone_features'],
        candidate_context_policy='one fixed private-seeded static_train context per candidate identity',
        selection_policy='nearest/farthest by mean cosine to frozen query; diverse starts nearest then farthest-first; five uniform random triples on same capped bank')
    (a.out/'receipt.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',report['seconds'],flush=True)


if __name__=='__main__':main()
