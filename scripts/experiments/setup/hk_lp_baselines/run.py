#!/usr/bin/env python3
"""Small HK direct-LP pilot on one frozen set of held-out endpoint pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('WANDB_MODE', 'disabled')
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/experiments/setup/final_core')]
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv
from scripts.experiments.setup.hk_lp_baselines.protocol import (
    edge_keys, hash_arrays, make_pairs, measure, evaluate_score,
)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


class LPEncoder(nn.Module):
    def __init__(self, kind, input_dim=768, hidden=256):
        super().__init__()
        self.kind = kind
        if kind == 'mlp':
            self.first = nn.Linear(input_dim, hidden)
        else:
            self.first = SAGEConv(input_dim, hidden, aggr='mean')
        self.second = nn.Linear(hidden, hidden)
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(10.))))
        self.bias = nn.Parameter(torch.tensor(-5.))

    def forward(self, x, mean):
        if self.kind == 'mlp':
            h = self.first(x)
        else:
            h = self.first.lin_l(mean) + self.first.lin_r(x)
        return F.normalize(self.second(F.relu(h)), dim=-1)


def self_test():
    """Pin the precomputed-mean optimization against real PyG message passing."""
    torch.manual_seed(19)
    x = torch.randn(7, 5)
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    sums = torch.zeros_like(x).index_add_(0, edges[1], x[edges[0]])
    deg = torch.bincount(edges[1], minlength=len(x)).clamp_min(1)
    mean = sums / deg[:, None]
    for kind in ('mlp', 'graphsage'):
        model = LPEncoder(kind, 5, 8)
        actual = model(x, mean)
        if kind == 'graphsage':
            expected = F.normalize(model.second(F.relu(model.first(x, edges))), dim=-1)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        score = (actual[[0, 1]] * actual[[2, 3]]).sum(-1)
        score.sum().backward()
        assert model.second.weight.grad.abs().sum() > 0
        assert torch.isfinite(actual).all()
        assert abs(float((actual[0] * actual[2]).sum() - (actual[0] * actual[3]).sum())) > 1e-6
    print('SELF_TEST_PASS: PyG parity, isolates, finite gradients, both endpoints', flush=True)


def train_encoder(kind, x, mean, pairs, batch_stream, args, nodes):
    from scripts.experiments.setup.nm_support_resampling.run import digest_state
    torch.manual_seed(args.seed)
    model = LPEncoder(kind).to(args.device)
    initial_hash = digest_state(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    u = torch.tensor(pairs['u'], device=args.device)
    v = torch.tensor(pairs['v'], device=args.device)
    y = torch.tensor(pairs['y'], dtype=torch.float32, device=args.device)
    t0 = time.monotonic()
    history = []
    model.train()
    for step, batch in enumerate(batch_stream, 1):
        ix = torch.tensor(np.r_[batch, batch + args.train_positives], device=args.device)
        endpoints, inverse = torch.unique(torch.cat((u[ix], v[ix])), return_inverse=True)
        z = model(x[endpoints], mean[endpoints])
        a, b = inverse.chunk(2)
        score = (z[a] * z[b]).sum(-1)
        logits = model.log_scale.exp().clamp(max=100) * score + model.bias
        loss = F.binary_cross_entropy_with_logits(logits, y[ix])
        if not torch.isfinite(loss):
            raise RuntimeError(f'{kind}: nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if step == 1 or step % 250 == 0 or step == args.steps:
            print(json.dumps(dict(phase='train', model=kind, step=step,
                loss=history[-1], seconds=round(time.monotonic()-t0, 2))), flush=True)
    model.eval()
    final_hash = digest_state(model)
    assert initial_hash != final_hash
    torch.save(dict(model=model.cpu().state_dict(), kind=kind, steps=args.steps,
                    initial_hash=initial_hash, final_hash=final_hash), args.out/f'{kind}.pt')
    model.to(args.device)
    with torch.no_grad():
        embeddings = torch.cat([model(x[ids], mean[ids]).cpu()
            for ids in torch.tensor(nodes, device=args.device).split(8192)]).numpy()
    elapsed = time.monotonic() - t0
    return embeddings, dict(seconds=elapsed, initial_hash=initial_hash,
        final_hash=final_hash, first_100_mean_loss=float(np.mean(history[:100])),
        last_100_mean_loss=float(np.mean(history[-100:]))), history


def prodigy_embeddings(args, x_cpu, train_edges, nodes):
    from torch_geometric.data import Data, Batch
    from data.dataset import SubgraphDataset, SeededNodeIndex
    from experiments.sampler import NeighborSampler
    from scripts.experiments.setup.nm_hk_mechanism.run import build_model
    from scripts.experiments.setup.nm_support_resampling.run import digest_state
    from scripts.eval.pair_link_eval import embed_nodes
    import yaml
    params = yaml.safe_load(args.config.read_text())['params']
    assert params['edge_view'] == 'static_train' and params['neighbor_matching_edge_split']
    assert params['neighbor_sampling_source_subset'] == 'cp_hk'
    assert params['task_name'] == 'neighbor_matching'
    params['device'] = args.device
    assert params['n_hop'] == 2
    model = build_model(params, args.checkpoint, args.device)
    before = digest_state(model)
    assert before == 'a15e3f23cb3c7fdfd36a385e76bf366f57865d4d6bc6f2638002aac9c9d18755'
    graph = Data(x=x_cpu, edge_index=train_edges, num_nodes=len(x_cpu))
    sampler = NeighborSampler(graph, num_hops=2, hop_sizes=[9, 9], limit=101, walk_hops=1)
    dataset = SubgraphDataset(graph, sampler, bidirectional=False)
    context_dir = args.out / 'prodigy_contexts'
    context_dir.mkdir()
    output = []
    for start in range(0, len(nodes), 128):
        chunk = nodes[start:start+128]
        graphs = [dataset[SeededNodeIndex(int(n), int((args.context_seed + int(n)*1000003) % (2**63-1)))]
                  for n in chunk]
        batch = Batch.from_data_list(graphs)
        z = embed_nodes(model, dataset, chunk, args.device, batch_size=128, cached_batches=[batch])
        assert np.isfinite(z).all()
        output.append(z)
        del batch.x
        torch.save(batch, context_dir/f'batch_{start:06d}.pt')
        if start == 0 or (start//128) % 10 == 0:
            print(json.dumps(dict(phase='prodigy_encode', complete=start+len(chunk), total=len(nodes))), flush=True)
    assert before == digest_state(model)
    z = np.concatenate(output)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return z, dict(model_hash=before, frozen_state_unchanged=True, context_nodes=len(nodes))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--steps', type=int, default=2500)
    p.add_argument('--train-positives', type=int, default=100000)
    p.add_argument('--eval-positives', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pair-seed', type=int, default=90841)
    p.add_argument('--context-seed', type=int, default=90843)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--features', type=Path, default=Path('/dataMeR1/phil/data/cp_hk_twitter/graphs/retweet_graph.pt'))
    p.add_argument('--views', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_hk_support_extremes_20260908/hk_canonical_views_private.pt'))
    p.add_argument('--config', type=Path, default=Path('/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-final-core/files/wandb/run-20260807_133711-3pn08pge/files/effective_config.yaml'))
    p.add_argument('--checkpoint', type=Path, default=Path('/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-final-core/files/state/final_core/finalcore_ss_cp_hk_s0_20260807/checkpoint/state_dict_2500.ckpt'))
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test()
        return
    if args.out is None:
        p.error('--out is required')
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        learning_rate=.002, weight_decay=.001, positive_batch_size=256,
        batch_seed=90842, initial_scale=10., initial_bias=-5.,
        status='pilot' if args.steps == 2500 and args.eval_positives == 2000 and args.train_positives == 100000 else 'smoke',
        neighborhood='train edges only; baseline full undirected one-hop; PRODIGY directed sampled two-hop [9,9]',
        description='README.md in the recorded code revision is part of this protocol')
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    assert args.device.startswith('cuda:') and int(args.device.split(':')[1]) in range(4)
    assert torch.cuda.is_available()
    self_test()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'protocol.json', protocol)
    started = time.monotonic()
    phases = {}
    hashes = dict(views=file_hash(args.views), checkpoint=file_hash(args.checkpoint), features=file_hash(args.features),
                  checkpoint_training_config=file_hash(args.config))
    (args.out/'checkpoint_training_config.yaml').write_text(args.config.read_text())
    assert hashes['views'] == 'b46d5953aff4ba7602fa4e573f4fc1e2d1efa365a4966a40c90780b7bbd43c40'
    assert hashes['checkpoint'] == 'c664ff20d14d4fbba03ccdc9a459d67f243195e1cdbf74d3a135931bd6d2cf0d'
    from scripts.experiments.setup.nm_source_stage_audit.run import map_features
    from scripts.eval.pair_link_eval import Adjacency, PairSet, heuristic_scores
    views = torch.load(args.views, map_location='cpu', weights_only=False)['views']
    x_cpu = torch.from_numpy(np.array(map_features(args.features), copy=True))
    n, d = x_cpu.shape
    assert (n, d) == (333800, 768)
    expected_counts = dict(train=829082, validation=177494, test=177803)
    assert {k: v.shape[1] for k, v in views.items()} == expected_counts
    keys = {k: edge_keys(v.numpy(), n) for k, v in views.items()}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert len(np.intersect1d(keys[a], keys[b])) == 0
    known = np.concatenate(list(keys.values()))
    assert len(known) == len(np.unique(known))
    adj = Adjacency.from_edge_index(views['train'].numpy(), n)
    pairs, pair_info, used = {}, {}, set()
    rng = np.random.default_rng(args.pair_seed)
    for split, count in [('validation', args.eval_positives), ('test', args.eval_positives), ('train', args.train_positives)]:
        pairs[split], pair_info[split] = make_pairs(keys[split], known, adj.degree, count, rng, used)
        np.savez_compressed(args.out/f'{split}_pairs.npz', **pairs[split])
    write_json(args.out/'pair_receipt.json', dict(input_hashes=hashes, splits=pair_info))
    phases['load_hash_pairs_seconds'] = time.monotonic()-started
    print(json.dumps(dict(phase='pairs_ready', seconds=phases['load_hash_pairs_seconds'], splits=pair_info)), flush=True)
    t0 = time.monotonic()
    x = x_cpu.to(args.device)
    csr = torch.sparse_csr_tensor(torch.tensor(adj.indptr, device=args.device),
        torch.tensor(adj.indices.astype(np.int64), device=args.device),
        torch.ones(len(adj.indices), device=args.device), size=(n, n))
    mean = torch.sparse.mm(csr, x) / torch.tensor(adj.degree.clip(min=1), device=args.device)[:, None]
    assert mean.dtype == x.dtype and torch.isfinite(mean).all()
    nodes = np.unique(np.concatenate([pairs[s][k] for s in ('validation', 'test') for k in ('u', 'v')]))
    lookup = np.full(n, -1, dtype=np.int64)
    lookup[nodes] = np.arange(len(nodes))
    embeddings = {}
    with torch.no_grad():
        center_z = F.normalize(x[nodes], dim=1)
        mean_z = F.normalize(mean[nodes], dim=1)
        embeddings['raw_cosine'] = center_z.cpu().numpy()
        embeddings['raw_plus_neighbor_mean'] = F.normalize(torch.cat((center_z, mean_z), dim=1), dim=1).cpu().numpy()
    phases['neighbor_precompute_seconds'] = time.monotonic()-t0
    stream = np.random.default_rng(90842).integers(args.train_positives, size=(args.steps, 256))
    np.save(args.out/'training_batch_indices.npy', stream)
    train_info = {}
    for kind in ('mlp', 'graphsage'):
        embeddings[kind], train_info[kind], loss = train_encoder(kind, x, mean, pairs['train'], stream, args, nodes)
        np.save(args.out/f'{kind}_loss.npy', np.asarray(loss))
    del x, mean, csr
    torch.cuda.empty_cache()
    t0 = time.monotonic()
    embeddings['prodigy_nm'], prodigy_info = prodigy_embeddings(args, x_cpu, views['train'], nodes)
    phases['prodigy_sample_encode_cache_seconds'] = time.monotonic()-t0
    np.savez_compressed(args.out/'endpoint_embeddings.npz', node_ids=nodes, **embeddings)
    t0 = time.monotonic()
    scores = {s: {} for s in ('validation', 'test')}
    metrics, controls = {}, {}
    permutation = np.random.default_rng(90844).permutation(len(pairs['test']['v']))
    for name, z in embeddings.items():
        for split in scores:
            a, b = lookup[pairs[split]['u']], lookup[pairs[split]['v']]
            assert (a >= 0).all() and (b >= 0).all()
            scores[split][name] = (z[a] * z[b]).sum(-1)
        a, b = lookup[pairs['test']['u']], lookup[pairs['test']['v']][permutation]
        controls[name] = measure(pairs['test']['y'], (z[a] * z[b]).sum(-1))
    for name in ('common_neighbors', 'adamic_adar', 'jaccard', 'preferential_attachment'):
        for split in scores:
            pair = pairs[split]
            scores[split][name] = heuristic_scores(name, PairSet(pair['u'], pair['v'], pair['y'], 'degree_matched'), adj)
    raw_zero = np.linalg.norm(x_cpu.numpy(), axis=1) == 0
    strata = {}
    for split in scores:
        pair = pairs[split]
        np.savez_compressed(args.out/f'{split}_scores.npz', **pair, **scores[split])
        mask = raw_zero[pair['u']] | raw_zero[pair['v']]
        strata[split] = dict(any_zero_feature_pairs=int(mask.sum()), total=len(mask), subsets={})
        for label, select in [('both_features_nonzero', ~mask), ('any_zero_feature', mask)]:
            record = dict(n=int(select.sum()), positives=int(pair['y'][select].sum()), models={})
            if len(np.unique(pair['y'][select])) == 2:
                for name in scores[split]:
                    sign = 1 if measure(pairs['validation']['y'], scores['validation'][name])['auc'] >= .5 else -1
                    record['models'][name] = measure(pair['y'][select], sign*scores[split][name][select])
            strata[split]['subsets'][label] = record
    for name in scores['test']:
        metrics[name] = evaluate_score(pairs['validation'], pairs['test'], scores['validation'][name], scores['test'][name])
    phases['score_write_seconds'] = time.monotonic()-t0
    phases['total_seconds'] = time.monotonic()-started
    result = dict(complete=True, protocol=protocol, input_hashes=hashes, pair_info=pair_info,
        metrics=metrics, training=train_info, prodigy=prodigy_info, phases=phases,
        endpoint_permutation_controls_raw_orientation=controls, feature_strata=strata,
        batch_stream_hash=hash_arrays(stream), endpoint_embedding_hashes={k: hash_arrays(v) for k,v in embeddings.items()},
        software=dict(torch=torch.__version__, numpy=np.__version__))
    write_json(args.out/'results.json', result)
    print(json.dumps(dict(complete=True, metrics=metrics, phases=phases)), flush=True)


if __name__ == '__main__':
    main()
