"""Reconstruct/hash complete canonical NM inputs; CPU feature/overlap diagnostics."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('WANDB_MODE', 'disabled')
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/experiments/setup/final_core')]
from evaluate_fixed_grid import (load_dataset, TrainerFS, seed_everything,
    reset_fixed_eval_rng, make_frozen_loader, iter_batch_tensors, atomic_json, git_commit)


def update_hash(h, batch):
    for t in iter_batch_tensors(batch):
        a = t.detach().cpu().contiguous().numpy()
        h.update(str((a.dtype, a.shape)).encode())
        h.update(a.tobytes())


def restore(packed, features):
    batch = packed['batch']
    ids = batch[0].global_node_ids
    x = features[ids.clamp_min(0)].clone()
    x[ids < 0] = 0
    batch[0].x = x
    return batch


def pack(batch, features, path):
    # Save all model inputs except x, whose exact rows are recoverable from the
    # immutable backing graph. Keep PyG boundaries and metadata without conversion.
    graph = batch[0].clone()
    expected = batch[0].x
    del graph.x
    packed = {'schema': 1, 'batch': (graph, *batch[1:]),
              'feature_lookup': 'backing_graph.x[global_node_ids], -1 is zero supernode'}
    torch.save(packed, path)
    recovered = restore(torch.load(path, map_location='cpu', weights_only=False), features)
    assert torch.equal(expected, recovered[0].x)
    return recovered


def normalized(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0), norm[:,0] > 0


def episode_metrics(graph, ep, bi, rows, offset):
    start = ep*210
    ids = graph.global_node_ids.numpy()
    ptr = graph.ptr.numpy()
    x = graph.x.numpy()
    sets, contexts, centers, means, context_means, zero_rates, sizes, edges = [], [], [], [], [], [], [], []
    for slot in range(start, start+210):
        a,b = int(ptr[slot]), int(ptr[slot+1])
        real = ids[a:b] >= 0
        nodeids = ids[a:b][real]
        values = x[a:b][real]
        assert len(nodeids) and len(np.unique(nodeids)) == len(nodeids)
        assert int(nodeids[0]) == int(graph.center_node_idx[slot])
        sets.append(set(nodeids.tolist()))
        contexts.append(set(nodeids[1:].tolist()))
        centers.append(values[0])
        means.append(values.mean(axis=0))
        context_means.append(values[1:].mean(axis=0) if len(values)>1 else np.zeros_like(values[0]))
        sizes.append(len(values))
        zero_rates.append(float((np.linalg.norm(values, axis=1)==0).mean()))
    # Edges are local to subgraphs within the PyG batch. Count actual input edges.
    edge_owners = np.searchsorted(ptr[1:], graph.edge_index[0].numpy(), side='right')
    edge_counts = np.bincount(edge_owners, minlength=len(ptr)-1)
    qi = np.array([i for i in range(210) if i%7>=3])
    si = np.array([i for i in range(210) if i%7<3])
    truth = qi//7
    support_unions = [set().union(*(sets[c*7+j] for j in range(3))) for c in range(30)]
    # All query-vs-class comparisons, including competing candidates.
    jac = np.zeros((120,30))
    shared = np.zeros((120,30))
    center_present = np.zeros((120,30), dtype=bool)
    for qidx,slot in enumerate(qi):
        for c,union in enumerate(support_unions):
            n = len(sets[slot] & union)
            shared[qidx,c] = n
            jac[qidx,c] = n/len(sets[slot] | union)
            center_present[qidx,c] = int(ids[ptr[start+slot]]) in union
    values = {'center':np.array(centers), 'full_mean':np.array(means),
              'context_mean':np.array(context_means)}
    # A diagnostic summary, not the architecture's literal flattened input.
    cn,_ = normalized(values['center']); nn,_ = normalized(values['context_mean'])
    values['center_context_concat'] = np.concatenate([cn,nn],axis=1)
    descriptors = {}
    dispersion = {}
    for name,vectors in values.items():
        z,valid = normalized(vectors)
        support = z[si].reshape(30,3,-1)
        scores = (z[qi] @ support.mean(axis=1).T)
        class_valid = valid[si].reshape(30,3).all(axis=1)
        scores[:,~class_valid] = np.nan
        scores[~valid[qi],:] = np.nan
        descriptors[name] = scores
        dispersion[name] = (np.sum(support[:,0]*support[:,1],axis=1)+
            np.sum(support[:,0]*support[:,2],axis=1)+np.sum(support[:,1]*support[:,2],axis=1))/3
        dispersion[name][~class_valid] = np.nan
    descriptors['node_jaccard'] = jac
    out = []
    for ix,slot in enumerate(qi):
        sample = start+int(slot)
        key = (f'{bi}:{ep}',sample)
        old = rows.loc[key]
        assert int(old['query'])+offset == int(ids[ptr[sample]])
        assert int(old.anchor)+offset == int(graph.task_label_map[ep,truth[ix]])
        for j in range(3):
            assert int(old['true_support_'+str(j)])+offset == int(ids[ptr[start+int(truth[ix])*7+j]])
        r = {'episode':key[0], 'sample':sample, 'query':int(old['query']),
             'anchor':int(old.anchor), 'ukr_correct':int(old.ukr_correct),
             'hk_correct':int(old.hk_correct), 'query_degree':int(old.query_degree),
             'query_nodes':sizes[slot], 'query_edges':int(edge_counts[sample]),
             'query_zero_fraction':zero_rates[slot],
             'true_support_nodes_mean':float(np.mean([sizes[truth[ix]*7+j] for j in range(3)])),
             'true_shared_nodes':shared[ix,truth[ix]],
             'query_center_in_true_support':bool(center_present[ix,truth[ix]]),
             'query_center_in_any_rival_support':bool(np.delete(center_present[ix],truth[ix]).any())}
        for name,scores in descriptors.items():
            scores = scores[ix]
            true = scores[truth[ix]]
            rival = np.delete(scores,truth[ix])
            best = np.nanmax(rival) if np.isfinite(rival).any() else np.nan
            r[name+'_true'] = true
            r[name+'_margin'] = true-best
            r[name+'_all_valid'] = bool(np.isfinite(scores).all())
            r[name+'_strict_win'] = bool(true>best+1e-7) if np.isfinite(true+best) else None
            if name in dispersion:
                r[name+'_true_support_pair_cosine'] = dispersion[name][truth[ix]]
        out.append(r)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-root', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_canonical_split_20260908'))
    p.add_argument('--bio-root', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_canonical_split_bios_20260908'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    protocol = json.loads((args.audit_root/'protocol_summary.json').read_text())
    params = json.loads((args.audit_root/'effective_config.json').read_text())
    params.update(device='cpu', log_dir=str(args.out/'logs'), state_dir=str(args.out/'state'), override_log=True)
    seed_everything(params)
    dataset = load_dataset(params)
    trainer = TrainerFS(dataset, params)
    receipt = dict(revision=git_commit(), feature_artifact=str(Path(params['root'])/params['graph_filename']),
        feature_shape=list(dataset.graph.x.shape), no_model_forwards=True, targets={})
    started = time.time()
    for target, name in [('ukr_rus','ukr_rus_twitter'),('cp_hk','cp_hk_twitter')]:
        dest = args.out/target; dest.mkdir()
        info = protocol['targets'][target]
        source = args.bio_root/name/'paired_cluster_queries_private.tsv'
        rows = pd.read_csv(source,sep='\t'); rows=rows[rows.split=='test'].set_index(['episode','sample'])
        assert len(rows)==61440 and rows.index.is_unique
        reset_fixed_eval_rng(target)
        trainer.parameter.update(neighbor_sampling_source_subset=target,eval_only_split='test')
        loader = trainer._build_dataloaders(dataset,trainer.dataset_name)[3]
        plans = torch.load(args.audit_root/target/'test_plan_private.pt',map_location='cpu',weights_only=False)
        reset_fixed_eval_rng(target)
        h = hashlib.sha256(); restored = hashlib.sha256(); allrows=[]; files=[]
        for bi,batch in enumerate(make_frozen_loader(dataset,loader.collate_fn,plans)):
            update_hash(h,batch)
            path = dest/f'batch_{bi:03d}_private.pt'
            recovered = pack(batch,dataset.graph.x,path)
            update_hash(restored,recovered)
            for ep in range(32):
                allrows.extend(episode_metrics(batch[0],ep,bi,rows,info['source_node_offset']))
            files.append(dict(file=path.name,bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            del recovered
            print(target,bi,'rows',len(allrows),flush=True)
        expected = info['splits']['test']['all_input_tensors_sha256']
        assert h.hexdigest()==restored.hexdigest()==expected,(target,h.hexdigest(),restored.hexdigest(),expected)
        frame = pd.DataFrame(allrows)
        assert len(frame)==61440 and not frame.duplicated(['episode','sample']).any()
        frame.to_csv(dest/'query_geometry_private.csv',index=False)
        receipt['targets'][target]=dict(rows=len(frame),original_hash=h.hexdigest(),
            reloaded_hash=restored.hexdigest(),expected_hash=expected,files=files,
            input_table_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_node_offset=info['source_node_offset'])
        atomic_json(args.out/'receipt.json',receipt)
        del batch,allrows,frame;gc.collect()
    receipt.update(complete=True,seconds=time.time()-started)
    atomic_json(args.out/'receipt.json',receipt)
    print('COMPLETE',receipt['seconds'],flush=True)


if __name__=='__main__': main()
