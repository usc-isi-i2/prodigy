"""Independent NumPy score/selection audit; no model or producer imports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def threshold(n):
    return np.sort(n)[-50]


def rate(p, n):
    return float((p > threshold(n)).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    args = p.parse_args()
    root, evidence = args.runtime, args.evidence
    sel = json.loads((evidence/'selection.json').read_text())
    result = json.loads((evidence/'results.json').read_text())
    assert digest(root/'scores2017.npz') == sel['scores2017_sha256']
    assert digest(root/'scores.npz') == result['score_sha256']
    assert digest(root/'year2018.npz') == result['panel_sha256']
    assert digest(evidence/'selection.json') == result['selection_sha256']
    prepared = json.loads((evidence.parent/'joint_v1/prepared.json').read_text())
    assert digest(root/'year2017.npz') == prepared['files']['year2017.npz']
    panels = {y: np.load(root/f'year{y}.npz') for y in (2017, 2018)}
    scores = {2017: np.load(root/'scores2017.npz'), 2018: np.load(root/'scores.npz')}
    grid = []
    for entry in sel['grid']:
        values = []
        for seed, receipt in enumerate(sel['checkpoints']):
            def transformed(year):
                panel, score = panels[year], scores[year]
                return {s: np.stack((panel['b'+s]/receipt['scale'],
                        np.exp(np.clip(score[f'joint_seed{seed}_{s}'].astype(np.float64)-receipt['offset'], -30, 30))))
                        for s in ('p', 'n')}
            v = transformed(2017)
            fused = {s: np.maximum(z[0], entry['alpha']*z[1]) if entry['alpha'] else z[0] for s, z in v.items()}
            values.append(rate(**fused))
            assert receipt['scale'] == threshold(panels[2017]['bn'])
            assert receipt['offset'] == threshold(scores[2017][f'joint_seed{seed}_n'])
            assert receipt['replay_hits'] == rate(scores[2017][f'joint_seed{seed}_p'], scores[2017][f'joint_seed{seed}_n'])
            if entry['alpha'] == sel['alpha']:
                a = transformed(2018)
                f = {s: np.maximum(z[0], sel['alpha']*z[1]) for s, z in a.items()}
                row = result['rows'][seed]
                assert rate(**f) == row['frozen_max_hits']
                base = panels[2018]['bp'] > threshold(panels[2018]['bn'])
                mask = f['p'] > threshold(f['n'])
                assert int((mask & ~base).sum()) == row['recovered']
                assert int((base & ~mask).sum()) == row['lost']
                # Independent cosine implementation; no sklearn/producer reuse.
                for name, partitions in [('hyper_frozen', [v['p'], v['n']]),
                                         ('hyper_adapted', [v['p'], v['n'], a['p'], a['n']])]:
                    h = np.zeros((2, len(partitions)))
                    for col, z in enumerate(partitions):
                        d = 1-np.dot(z[0], z[1])/(np.linalg.norm(z[0])*np.linalg.norm(z[1]))
                        if 0 < d < .1:
                            h[:, col] = 1
                    w = (h@h.T).sum(0)
                    assert h.tolist() == row[name]['H']
                    assert w.tolist() == row[name]['weights']
                    assert rate(w@a['p'], w@a['n']) == row[name]['hits']
        assert values == entry['hits']
        grid.append((np.mean(values), entry['alpha']))
    assert max(grid, key=lambda x: x[0])[1] == sel['alpha']
    values = [r['frozen_max_hits'] for r in result['rows']]
    assert float(np.mean(values)) == result['mean_frozen_max_hits']
    receipt = dict(valid=True, complete_seeds=3, exact_grid_and_assessment=True,
                   independent_hyper_weights=True, hashes_verified=True, test_scored=False,
                   mean_percent=float(np.mean(values)*100), sample_sd_points=float(np.std(values, ddof=1)*100))
    (evidence/'audit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
