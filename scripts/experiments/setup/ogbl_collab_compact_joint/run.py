#!/usr/bin/env python3
"""Bounded compact joint-vs-structure experiment; official test is never scored."""
import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


gate = module(HERE.parent / 'ogbl_collab_temporal_gate/run.py', 'temporal_gate')
shared = module(HERE.parent / 'ogbl_collab_mlp_lp/run.py', 'collab_shared')
sage = module(HERE.parent / 'ogbl_collab_sage_lp/run_ogb_style.py', 'collab_sage')
ARMS = ('structure', 'joint')
COUNTS = {'structure': 36_865, 'joint': 496_385}
SEEDS = (0, 1, 2)
STEPS = 2000
INTERVAL = 50


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def revision():
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def numeric_count(value):
    if isinstance(value, dict):
        return sum(numeric_count(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(numeric_count(v) for v in value)
    return int(isinstance(value, (int, float)) and not isinstance(value, bool))


def contract():
    return dict(name='compact_joint_v1', revision=revision(), arms=list(ARMS), seeds=list(SEEDS),
                parameters=COUNTS, steps=STEPS, selection_interval=INTERVAL,
                years={'calibration': 2015, 'train': 2016, 'selection': 2017, 'assessment': 2018},
                optimizer={'name': 'Adam', 'lr': .001, 'weight_decay': .0001, 'clip_norm': 5.},
                sampling={'positive': 2048, 'uniform_negative': 1024, 'hard_negative': 1024,
                          'hard_pool': 2048, 'refresh_steps': 10},
                node_input='supplied 128-dimensional author features; no learned node IDs',
                encoder='three mean SAGE layers 128->256->256->256; unique unweighted historical graph',
                decoder='[product,absolute difference,symmetric14]->256->128->1; structure uses14 only',
                structural_features='mean of forward/reverse 14 transformed features; calibration frozen2015',
                selection='2017 full-panel Hits@50; earliest maximum; no baseline fallback',
                objective='balanced BCE with model-specific hard-negative mining',
                advancement_rule='all3 paired joint-control gains positive AND joint mean at least1 percentage point above AA-DC',
                expected_cells=[f'{arm}_seed{seed}' for seed in SEEDS for arm in ARMS],
                test_scored=False, assessment_previously_inspected=True,
                qualification='static node features are not historical snapshots; no ensemble; not a capacity-only ablation')


class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.encoder = sage.SAGE(128) if arm == 'joint' else None
        self.decoder = nn.Sequential(nn.Linear(526 if self.encoder is not None else 14, 256),
                                     nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        assert sum(p.numel() for p in self.parameters()) == COUNTS[arm] < 1_000_000

    def encode(self, x, adj):
        return self.encoder(x, adj) if self.encoder is not None else None

    def score(self, z, edges, features):
        if z is not None:
            left, right = z[edges[:, 0]], z[edges[:, 1]]
            features = torch.cat([left * right, (left - right).abs(), features], dim=1)
        score = self.decoder(features).flatten()
        return score.masked_fill(edges[:, 0] == edges[:, 1], -1e9)


def graph_edges(graph, split, year):
    edges, _, _, _ = gate.historical_inputs(graph, split, year)
    return np.unique(np.sort(edges, axis=1), axis=0)


def load_data(args):
    graph, split, evaluator, version = shared.load_official_dataset(args.dataset_root)
    identity = shared.audit_dataset(graph, split, version)
    assert identity['split_fingerprint'] == gate.FINGERPRINT
    del split['test']  # Metadata fingerprint only; no test scorer exists here.
    upstream = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    assert upstream == gate.UPSTREAM
    source = args.upstream / 'aa_dc.py'
    assert source.read_bytes() == subprocess.check_output(['git', '-C', str(args.upstream), 'show', 'HEAD:aa_dc.py'])
    return graph, split, evaluator, module(source, 'aa'), identity


def tracking(out, name, config):
    import wandb
    run = wandb.init(project='ogbl-collab-compact-joint', group='compact_joint_v1', name=name,
                     mode='offline', dir=str(out), config=config)
    print(json.dumps({'event': 'wandb', 'directory': run.dir, 'name': name}), flush=True)
    return run


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = contract()
    save(args.out / 'protocol.json', cfg)
    run = tracking(args.out, 'prepare', cfg)
    graph, split, _, aa, identity = load_data(args)
    np.save(args.out / 'node_features.npy', np.asarray(graph['node_feat'], dtype=np.float32))
    metadata, calibration = [], None
    for year in (2015, 2016, 2017):
        panel, meta, calibration = gate.make_year(aa, shared, graph, split, year, calibration,
                                                   warm_only=True, symmetric_features=True)
        panel['graph_edges'] = graph_edges(graph, split, year)
        reference = args.panel_reference / f'year{year}.npz'
        with np.load(reference) as old:
            for key in ('pos', 'neg', 'pfeatures', 'nfeatures', 'bp', 'bn'):
                assert np.array_equal(panel[key], old[key]), f'Legacy panel mismatch {year}/{key}'
        for side in ('p', 'n'):
            assert panel[side+'symmetric'].shape[1] == 14 and np.isfinite(panel[side+'symmetric']).all()
        meta['legacy_panel_exact_match'] = True
        meta['legacy_panel_sha256'] = shared.sha256_file(reference)
        meta['symmetric_features_fingerprint'] = shared.fingerprint(panel['psymmetric'], panel['nsymmetric'])
        np.savez_compressed(args.out / f'year{year}.npz', **panel)
        metadata.append(meta)
        if year == 2016:
            fit = np.concatenate([panel['psymmetric'], panel['nsymmetric']])
            np.savez(args.out / 'standardization.npz', mean=fit.mean(0), std=np.maximum(fit.std(0), 1e-5))
        run.log({'year': year, 'feature_seconds': meta['elapsed_seconds']})
    files = ['node_features.npy', 'standardization.npz'] + [f'year{y}.npz' for y in (2015, 2016, 2017)]
    save(args.out / 'prepared.json', dict(complete=True, revision=revision(), calibration=calibration,
         protocol_sha256=shared.sha256_file(args.out / 'protocol.json'), metadata=metadata,
         dataset_fingerprint=identity['split_fingerprint'], upstream_revision=gate.UPSTREAM,
         fitted_auxiliary_scalars=28+numeric_count(calibration),
         files={name: shared.sha256_file(args.out / name) for name in files}, wandb_directory=run.dir))
    run.finish()


def verify_prepared(args):
    cfg, prepared = read(args.out / 'protocol.json'), read(args.out / 'prepared.json')
    assert cfg == contract(), 'Protocol or revision mismatch: use the producing revision'
    assert prepared['complete'] and prepared['revision'] == revision()
    assert prepared['protocol_sha256'] == shared.sha256_file(args.out / 'protocol.json')
    for name, sha in prepared['files'].items():
        assert shared.sha256_file(args.out / name) == sha, name
    return cfg, prepared


def panel_tensors(panel, mean, std, device):
    return {side: (torch.as_tensor(panel['pos' if side == 'p' else 'neg'], dtype=torch.long, device=device),
                   torch.as_tensor((panel[side+'symmetric'] - mean) / std, device=device)) for side in ('p', 'n')}


def adjacency(edges, n, device):
    from torch_sparse import SparseTensor
    both = np.concatenate([edges, edges[:, ::-1]])
    return SparseTensor(row=torch.as_tensor(both[:, 1].copy(), device=device),
                        col=torch.as_tensor(both[:, 0].copy(), device=device),
                        sparse_sizes=(n, n)).coalesce()


@torch.no_grad()
def scores(model, z, data, batch=16384):
    return torch.cat([model.score(z, data[0][i:i+batch], data[1][i:i+batch])
                      for i in range(0, len(data[0]), batch)])


@torch.no_grad()
def evaluate(model, x, adj, panel):
    model.eval()
    z = model.encode(x, adj)
    return {side: scores(model, z, data).cpu().numpy() for side, data in panel.items()}


def sampled_indices(generator, positive_count, negative_count, hard):
    p = torch.randint(positive_count, (2048,), generator=generator)
    uniform = torch.randint(negative_count, (1024,), generator=generator)
    slots = torch.randint(2048, (1024,), generator=generator)
    negative = torch.cat([uniform, hard.cpu()[slots]])
    return p, negative, torch.cat([p, uniform, slots]).numpy().tobytes()


def train(args):
    cfg, prepared = verify_prepared(args)
    assert args.device.startswith('cuda:') and int(args.device.split(':')[1]) in range(4)
    profile = args.phase == 'profile'
    out = args.out / ('profile' if profile else f'{args.arm}_seed{args.seed}')
    out.mkdir(exist_ok=False)
    steps = 50 if profile else STEPS
    cfg = dict(cfg, arm=args.arm, seed=args.seed, device=args.device, smoke_only=profile, executed_budget=steps)
    cfg['gpu_name'] = torch.cuda.get_device_name(args.device)
    save(out / 'config.json', cfg)
    run = tracking(out, out.name, cfg)
    started = time.monotonic()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    meanstd = np.load(args.out / 'standardization.npz')
    tr = np.load(args.out / 'year2016.npz')
    train_panel = panel_tensors(tr, meanstd['mean'], meanstd['std'], args.device)
    x = torch.as_tensor(np.load(args.out / 'node_features.npy'), device=args.device)
    adj = adjacency(tr['graph_edges'], len(x), args.device) if args.arm == 'joint' else None
    validation, validation_adj = None, None
    if not profile:
        va = np.load(args.out / 'year2017.npz')
        validation = panel_tensors(va, meanstd['mean'], meanstd['std'], args.device)
        validation_adj = adjacency(va['graph_edges'], len(x), args.device) if args.arm == 'joint' else None
    model = Model(args.arm).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    stream = hashlib.sha256()
    best, best_step, history, step_seconds = -np.inf, None, [], []
    torch.cuda.reset_peak_memory_stats()
    startup_seconds = time.monotonic() - started
    for step in range(1, steps+1):
        torch.cuda.synchronize()
        tick = time.monotonic()
        if (step-1) % 10 == 0:
            with torch.no_grad():
                model.eval()
                z = model.encode(x, adj)
                hard = scores(model, z, train_panel['n']).topk(2048).indices.cpu()
                del z
        p, n, stream_bytes = sampled_indices(generator, len(train_panel['p'][0]), len(train_panel['n'][0]), hard)
        stream.update(stream_bytes)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z = model.encode(x, adj)
        values = []
        for side, index in [('p', p), ('n', n)]:
            index = index.to(args.device)
            edges, features = train_panel[side]
            values.append(model.score(z, edges[index], features[index]))
        logits = torch.cat(values)
        labels = torch.cat([torch.ones_like(values[0]), torch.zeros_like(values[1])])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 5.)
        assert torch.isfinite(gradient)
        optimizer.step()
        del z, values, logits
        torch.cuda.synchronize()
        step_seconds.append(time.monotonic() - tick)
        if step % 10 == 0:
            run.log({'step': step, 'train/loss': float(loss), 'train/gradient_norm': float(gradient),
                     'system/step_seconds': step_seconds[-1], 'system/peak_gpu_bytes': torch.cuda.max_memory_allocated()})
        if not profile and step % INTERVAL == 0:
            prediction = evaluate(model, x, validation_adj, validation)
            value = gate.hits(prediction['p'], prediction['n'])
            if value > best:
                best, best_step = value, step
                torch.save({'model': model.state_dict(), 'arm': args.arm, 'seed': args.seed,
                            'step': step, 'revision': revision()}, out / 'best.pt')
            row = dict(step=step, train_loss=float(loss), selection_hits_at_50=value,
                       elapsed_seconds=time.monotonic()-started)
            history.append(row)
            save(out / 'history.json', history)
            run.log({'step': step, 'selection/hits_at_50': value, 'selection/best': best})
            print(json.dumps(row | {'arm': args.arm, 'seed': args.seed}), flush=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': steps, 'arm': args.arm, 'seed': args.seed, 'revision': revision()}, out / 'last.pt')
    receipt = dict(complete=True, smoke_only=profile, arm=args.arm, seed=args.seed, revision=revision(),
                   parameters=sum(p.numel() for p in model.parameters()), steps_run=steps,
                   total_inference_learned_scalars=COUNTS[args.arm]+prepared['fitted_auxiliary_scalars'],
                   startup_seconds=startup_seconds, elapsed_seconds=time.monotonic()-started,
                   steady_step_seconds=float(np.mean(step_seconds[10:])),
                   peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                   host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   uniform_draw_stream_sha256=stream.hexdigest(), wandb_directory=run.dir,
                   protocol_sha256=shared.sha256_file(args.out / 'protocol.json'),
                   prepared_sha256=shared.sha256_file(args.out / 'prepared.json'),
                   last_checkpoint_sha256=shared.sha256_file(out / 'last.pt'), test_scored=False)
    if not profile:
        receipt.update(selected_step=best_step, selection_hits_at_50=best,
                       checkpoint_sha256=shared.sha256_file(out / 'best.pt'), history=history)
    save(out / ('profile.json' if profile else 'selection.json'), receipt)
    run.summary.update({k:v for k,v in receipt.items() if k != 'history'})
    run.finish()
    print(json.dumps({k:v for k,v in receipt.items() if k != 'history'}), flush=True)


def frozen_selections(args):
    cfg, prepared = verify_prepared(args)
    selections = []
    for name in cfg['expected_cells']:
        path = args.out / name
        s = read(path / 'selection.json')
        assert s['complete'] and not s['smoke_only'] and s['steps_run'] == STEPS
        assert s['revision'] == revision() and s['parameters'] == COUNTS[s['arm']] < 1_000_000
        assert s['total_inference_learned_scalars'] == COUNTS[s['arm']]+prepared['fitted_auxiliary_scalars'] < 1_000_000
        assert name == f"{s['arm']}_seed{s['seed']}"
        assert s['protocol_sha256'] == shared.sha256_file(args.out / 'protocol.json')
        assert s['prepared_sha256'] == shared.sha256_file(args.out / 'prepared.json')
        assert shared.sha256_file(path / 'best.pt') == s['checkpoint_sha256']
        assert shared.sha256_file(path / 'last.pt') == s['last_checkpoint_sha256']
        assert [h['step'] for h in s['history']] == list(range(INTERVAL, STEPS+1, INTERVAL))
        winner = max(s['history'], key=lambda h: h['selection_hits_at_50'])
        assert winner['step'] == s['selected_step'] and winner['selection_hits_at_50'] == s['selection_hits_at_50']
        selections.append(s)
    for seed in SEEDS:
        pair = [s for s in selections if s['seed'] == seed]
        assert pair[0]['uniform_draw_stream_sha256'] == pair[1]['uniform_draw_stream_sha256']
    return selections


def hitmask(pos, neg):
    return pos > np.partition(neg, -50)[-50]


def comparison(pos, neg, bp, bn, panel):
    current, base = hitmask(pos, neg), hitmask(bp, bn)
    repeat = panel['pfeatures'][:, 8] > 0
    return dict(hits_at_50=float(current.mean()), recovered=int((current & ~base).sum()),
                lost=int((~current & base).sum()), net_hits=int(current.sum())-int(base.sum()),
                repeat_net_hits=int(current[repeat].sum())-int(base[repeat].sum()),
                new_pair_net_hits=int(current[~repeat].sum())-int(base[~repeat].sum()),
                zero_base_hits=int(current[panel['bp'] == 0].sum()),
                negative50=float(np.partition(neg, -50)[-50]))


def assess(args):
    selections = frozen_selections(args)
    prepared = read(args.out / 'prepared.json')
    out = args.out / 'assessment'
    out.mkdir(exist_ok=False)
    save(out / 'selection_frozen.json', dict(selections=selections, assessment_scored=False, revision=revision()))
    run = tracking(out, 'assessment', contract())
    graph, split, evaluator, aa, _ = load_data(args)
    panel, meta, _ = gate.make_year(aa, shared, graph, split, 2018, prepared['calibration'],
                                    warm_only=True, symmetric_features=True)
    panel['graph_edges'] = graph_edges(graph, split, 2018)
    np.savez_compressed(out / 'year2018.npz', **panel)
    std = np.load(args.out / 'standardization.npz')
    tensors = panel_tensors(panel, std['mean'], std['std'], args.device)
    x = torch.as_tensor(graph['node_feat'], dtype=torch.float32, device=args.device)
    adj = adjacency(panel['graph_edges'], len(x), args.device)
    rows, arrays = [], {}
    for s in selections:
        name = f"{s['arm']}_seed{s['seed']}"
        model = Model(s['arm']).to(args.device)
        state = torch.load(args.out / name / 'best.pt', map_location=args.device, weights_only=False)
        assert state['step'] == s['selected_step'] and state['revision'] == revision()
        model.load_state_dict(state['model'], strict=True)
        values = evaluate(model, x, adj, tensors)
        evaluator.K = 50
        metric = evaluator.eval({'y_pred_pos': values['p'], 'y_pred_neg': values['n']})['hits@50']
        row = comparison(values['p'], values['n'], panel['official_bp'], panel['official_bn'], panel)
        assert row['hits_at_50'] == metric
        for side in ('p', 'n'):
            edges = panel['pos' if side == 'p' else 'neg']
            assert np.isfinite(values[side]).all()
            assert (values[side][edges[:, 0] == edges[:, 1]] == -1e9).all()
            arrays[name+'_'+side] = values[side]
        rows.append(dict(arm=s['arm'], seed=s['seed'], selected_step=s['selected_step'], **row))
        del model
    np.savez_compressed(out / 'scores.npz', **arrays)
    summary = {}
    for arm in ARMS:
        values = [r['hits_at_50'] for r in rows if r['arm'] == arm]
        summary[arm] = dict(mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)))
    result = dict(complete=True, revision=revision(), test_scored=False, rows=rows, summary=summary,
                  official_aadc_hits_at_50=gate.hits(panel['official_bp'], panel['official_bn']),
                  metadata=meta, selection_frozen_sha256=shared.sha256_file(out / 'selection_frozen.json'),
                  score_sha256=shared.sha256_file(out / 'scores.npz'),
                  panel_sha256=shared.sha256_file(out / 'year2018.npz'), wandb_directory=run.dir)
    paired = [next(r['hits_at_50'] for r in rows if r['arm']=='joint' and r['seed']==seed) -
              next(r['hits_at_50'] for r in rows if r['arm']=='structure' and r['seed']==seed) for seed in SEEDS]
    result['paired_joint_minus_structure'] = paired
    result['advancement_rule_passed'] = bool(min(paired)>0 and
        summary['joint']['mean'] >= result['official_aadc_hits_at_50']+.01)
    save(out / 'results.json', result)
    run.summary.update({'complete': True, 'joint_mean': summary['joint']['mean'],
                        'structure_mean': summary['structure']['mean']})
    run.finish()
    print(json.dumps(result), flush=True)


def audit(args):
    selections = frozen_selections(args)
    out = args.out / 'assessment'
    result = read(out / 'results.json')
    assert read(out / 'selection_frozen.json')['selections'] == selections
    for filename, key in [('selection_frozen.json', 'selection_frozen_sha256'),
                          ('scores.npz', 'score_sha256'), ('year2018.npz', 'panel_sha256')]:
        assert shared.sha256_file(out / filename) == result[key]
    panel, values = np.load(out / 'year2018.npz'), np.load(out / 'scores.npz')
    assert len(result['rows']) == 6
    assert {f"{r['arm']}_seed{r['seed']}" for r in result['rows']} == set(contract()['expected_cells'])
    for row in result['rows']:
        name = f"{row['arm']}_seed{row['seed']}"
        actual = comparison(values[name+'_p'], values[name+'_n'], panel['official_bp'], panel['official_bn'], panel)
        assert all(row[key] == value for key, value in actual.items())
    for arm in ARMS:
        v = [r['hits_at_50'] for r in result['rows'] if r['arm'] == arm]
        assert result['summary'][arm] == {'mean': float(np.mean(v)), 'sample_sd': float(np.std(v, ddof=1))}
    paired = [next(r['hits_at_50'] for r in result['rows'] if r['arm']=='joint' and r['seed']==seed) -
              next(r['hits_at_50'] for r in result['rows'] if r['arm']=='structure' and r['seed']==seed) for seed in SEEDS]
    assert paired == result['paired_joint_minus_structure']
    assert result['advancement_rule_passed'] == bool(min(paired)>0 and
        result['summary']['joint']['mean'] >= result['official_aadc_hits_at_50']+.01)
    receipt = dict(complete=True, six_cells_verified=True, checkpoint_hashes_and_selection_verified=True,
                   paired_uniform_streams_verified=True, metrics_recomputed=True, test_scored=False,
                   results_sha256=shared.sha256_file(out / 'results.json'), revision=revision())
    save(out / 'audit.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'profile', 'train', 'assess', 'audit'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=Path('/dataMeR1/phil/data/ogb'))
    p.add_argument('--upstream', type=Path, default=Path('/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204/upstream'))
    p.add_argument('--panel-reference', type=Path, default=Path('/dataMeR1/phil/gfm/ogbl_collab_temporal_gate/warm_v1'))
    p.add_argument('--arm', choices=ARMS, default='joint')
    p.add_argument('--seed', type=int, choices=SEEDS, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.dry_run:
        print(json.dumps(contract() | {'phase': args.phase, 'out': str(args.out),
                                     'arm': args.arm, 'seed': args.seed, 'device': args.device}, indent=2))
        return
    if args.phase in ('profile', 'train', 'assess'):
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            p.error('Unset CUDA_VISIBLE_DEVICES; device IDs must refer directly to owned physical GPUs')
        if args.device not in [f'cuda:{i}' for i in range(4)]:
            p.error('Only owned GPUs cuda:0 through cuda:3 are permitted')
        torch.cuda.set_device(args.device)
    torch.set_num_threads(args.threads)
    {'prepare': prepare, 'profile': train, 'train': train, 'assess': assess, 'audit': audit}[args.phase](args)


if __name__ == '__main__':
    main()
