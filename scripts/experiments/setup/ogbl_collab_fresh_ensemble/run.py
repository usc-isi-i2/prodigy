"""Frozen, archive-only two-expert experiment; select never opens test artifacts."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from ogb.linkproppred import Evaluator
import wandb


def read(p):
    return json.loads(p.read_text())


def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def metric(scores):
    p, n = scores['p'], scores['n']
    assert p.ndim == n.ndim == 1 and len(n) == 100000
    assert np.isfinite(p).all() and np.isfinite(n).all()
    cutoff = float(np.partition(n, -50)[-50])
    value = float(np.mean(p > cutoff))
    official = Evaluator(name='ogbl-collab').eval({'y_pred_pos':p, 'y_pred_neg':n})['hits@50']
    assert value == official
    return dict(hits=value, cutoff=cutoff, positive_hits=int(np.sum(p > cutoff)))


def average(a, b):
    return {side: (a[side].astype(np.float64) + b[side].astype(np.float64))/2 for side in ('p','n')}


def scores(path, expected, panel):
    assert sha(path) == expected, str(path)
    with np.load(path) as f:
        result = {s: f[s] for s in ('p','n')}
    for side, edges in [('p',panel['pos']), ('n',panel['neg'])]:
        assert len(result[side]) == len(edges)
        assert np.isfinite(result[side]).all()
        assert np.all(result[side][edges[:,0] == edges[:,1]] == -1e9)
    return result


def checkpoint(path, expected, seed, step):
    assert sha(path) == expected
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['seed'] == seed and state['step'] == step
    count = sum(v.numel() for v in state['model'].values())
    assert count == 496385
    return dict(path=str(path), sha256=expected, seed=seed, step=step, state_scalars=count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['select','test'])
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--validation-panel', type=Path, required=True)
    p.add_argument('--test-panel', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    revision = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    contract = dict(name='fresh_ensemble_v1', revision=revision, pairs=[[0,1],[0,2],[1,2]],
        rule='equal arithmetic mean of raw joint logits in float64; no normalization or AA fusion',
        selection='maximum validation Hits@50, lexicographically first pair breaks ties; lowest seed breaks standalone ties',
        checkpoints='reuse existing per-seed fusion-selected checkpoints; no new checkpoint search',
        gate='pair validation Hits strictly exceeds best standalone at those checkpoints',
        stop='failed gate: no test access; passed gate: one frozen pair test, no extension',
        target=0.7129, inference_scalar_bound=992897,
        counting='2*(496385 neural +63 conservative original auxiliary scalars)+1 fixed averaging coefficient',
        qualification='Post-hoc development; repeated prior test exploration and invalid test-supervised/negative-exposed diagnostics disclosed. Leaderboard acceptance unconfirmed.',
        source=str(a.source), validation_panel=str(a.validation_panel))
    if a.dry_run:
        print(json.dumps(contract, indent=2)); return
    torch.set_num_threads(2)
    start = time.monotonic()
    if a.phase == 'select':
        a.out.mkdir(parents=True, exist_ok=False)
        save(a.out/'protocol.json',contract)
    else:
        assert read(a.out/'protocol.json') == contract
        assert not (a.out/'test_started.json').exists()
    run = wandb.init(project='ogbl-collab-compact', group='fresh_ensemble_v1', name=a.phase,
                     mode='offline', dir=str(a.out), config=contract)
    if a.phase == 'select':
        prepared = read(a.source/'prepared.json')
        assert sha(a.validation_panel) == prepared['files'][str(a.validation_panel)]
        panel = np.load(a.validation_panel)
        frozen = read(a.source/'frozen.json')
        raw, rows, checkpoints, identities = {}, [], [], {}
        for seed in range(3):
            root = a.source/f'select{seed}'
            r = read(root/'results.json')
            assert r == frozen['selections'][seed] and r['exact_selected_replay'] and r['complete']
            assert not r['test_scored'] and r['seed'] == seed
            assert sha(root/'history.json') == r['history_sha256']
            history = read(root/'history.json')
            winner = max((dict(step=h['step'], **g) for h in history for g in h['grid']), key=lambda g:g['hits'])
            assert all(r['best'][k] == v for k,v in winner.items())
            checkpoints.append(checkpoint(root/'best.pt',r['best_checkpoint_sha256'],seed,r['best']['step']))
            raw[seed] = scores(root/'best_scores.npz',r['best_score_sha256'],panel)
            m = metric(raw[seed])
            assert m['hits'] == next(h['joint_hits'] for h in history if h['step']==r['best']['step'])
            rows.append(dict(seed=seed, **m))
            identities[str(seed)] = r['best_score_sha256']
        pairs = [dict(pair=list(pair), **metric(average(raw[pair[0]],raw[pair[1]]))) for pair in itertools.combinations(range(3),2)]
        best = max(pairs, key=lambda r:r['hits'])
        control = max(rows, key=lambda r:r['hits'])
        result = dict(complete=True, pair=best, standalone_control=control, standalone_rows=rows,pair_rows=pairs,
            gate_passed=best['hits']>control['hits'], test_opened=False, checkpoints=checkpoints,
            validation_panel_sha256=sha(a.validation_panel), validation_score_hashes=identities,
            source_frozen_sha256=sha(a.source/'frozen.json'), protocol_sha256=sha(a.out/'protocol.json'),
            elapsed_seconds=time.monotonic()-start, wandb_directory=run.dir)
        save(a.out/'selection.json',result)
        run.log({'validation/ensemble':best['hits'], 'validation/control':control['hits'], 'gate_passed':result['gate_passed']})
    else:
        selected=read(a.out/'selection.json')
        assert selected['gate_passed'], 'Validation gate failed: test access prohibited'
        assert selected['source_frozen_sha256']==sha(a.source/'frozen.json')
        assert selected['protocol_sha256']==sha(a.out/'protocol.json')
        assert a.test_panel is not None
        save(a.out/'test_started.json',dict(selection_sha256=sha(a.out/'selection.json'), revision=revision))
        panel=np.load(a.test_panel)
        raw=[]; counts=[]
        frozen=read(a.source/'frozen.json')
        for seed in selected['pair']['pair']:
            root=a.source/f'refit{seed}'; r=read(root/'results.json')
            assert r['complete'] and r['frozen_sha256']==selected['source_frozen_sha256']
            assert r['frozen_settings']==frozen['selections'][seed]['best']
            assert r['test_panel_sha256']==sha(a.test_panel) and r['test_negative_training_overlap']==0
            counts.append(checkpoint(root/'final.pt',r['final_sha256'],seed,r['steps']))
            raw.append(scores(root/'test_scores.npz',r['test_score_sha256'],panel))
        combined=average(*raw); m=metric(combined)
        np.savez_compressed(a.out/'test_scores.npz',**combined)
        result=dict(complete=True,selected_pair=selected['pair']['pair'],test=m,
            exceeds_target=m['hits']>contract['target'],inference_scalar_bound=contract['inference_scalar_bound'],
            checkpoints=counts, selection_sha256=sha(a.out/'selection.json'), test_panel_sha256=sha(a.test_panel),
            test_score_sha256=sha(a.out/'test_scores.npz'),qualification=contract['qualification'],
            elapsed_seconds=time.monotonic()-start,wandb_directory=run.dir)
        save(a.out/'results.json',result); run.log({'test/hits':m['hits']})
    run.finish(); print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
