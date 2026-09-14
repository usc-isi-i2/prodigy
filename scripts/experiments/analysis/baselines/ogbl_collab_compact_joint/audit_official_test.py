"""Independent full-grid audit of explicitly test-tuned fusion predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
from audit_scores import read, sha, fingerprint


def cutoff(n):
    return np.sort(n)[-50]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    args = p.parse_args()
    e = args.evidence
    result, meta, protocol = (read(e/name) for name in ('results.json', 'panel.json', 'protocol.json'))
    assert result['complete'] and result['test_scored'] and result['test_used_for_selection']
    assert sha(args.runtime/'year2019.npz') == result['panel_sha256'] == meta['panel_sha256']
    assert sha(args.runtime/'scores.npz') == result['score_sha256']
    panel, scores = np.load(args.runtime/'year2019.npz'), np.load(args.runtime/'scores.npz')
    assert fingerprint(panel['pos'], panel['neg']) == meta['metadata']['pairs_fingerprint']
    assert fingerprint(panel['graph_edges']) == meta['graph_edges_fingerprint']
    assert len(panel['pos']) == 46329 and len(panel['neg']) == 100000
    assert meta['metadata']['graph_max_year'] == 2018
    assert result['total_inference_scalars'] == meta['total_inference_scalars'] <= 496448 < 1000000
    expected = [(b,a) for b in protocol['bases'] for a in protocol['alphas']]
    assert [(g['base'],g['alpha']) for g in result['grid']] == expected
    for seed, row in enumerate(result['joint']):
        pscore, nscore = [scores[f'joint_seed{seed}_{s}'] for s in ('p','n')]
        assert row['seed'] == seed
        assert float((pscore > cutoff(nscore)).mean()) == row['joint_hits']
        for score, edge in [(pscore,panel['pos']), (nscore,panel['neg'])]:
            assert np.isfinite(score).all() and len(score) == len(edge)
            assert (score[edge[:,0] == edge[:,1]] == -1e9).all()
    for g in result['grid']:
        prefix = 'official_b' if g['base']=='validation2018' else 'b'
        b = {s:panel[prefix+s] for s in ('p','n')}
        mask = b['p'] > cutoff(b['n'])
        assert float(mask.mean()) == result['baseline'][g['base']]
        for seed, r in enumerate(g['rows']):
            offset, scale = cutoff(scores[f'joint_seed{seed}_n']), cutoff(b['n'])
            f = {s:np.maximum(b[s]/scale, g['alpha']*np.exp(np.clip(
                    scores[f'joint_seed{seed}_{s}'].astype(np.float64)-offset,-30,30)))
                 if g['alpha'] else b[s]/scale for s in ('p','n')}
            h = f['p'] > cutoff(f['n'])
            assert r == dict(seed=seed,hits=float(h.mean()),recovered=int((h & ~mask).sum()),lost=int((mask & ~h).sum()))
        values = [r['hits'] for r in g['rows']]
        assert g['mean'] == float(np.mean(values)) and g['sample_sd'] == float(np.std(values,ddof=1))
    best = max(result['grid'],key=lambda g:g['mean'])
    assert best == result['best_common']
    single = max((dict(base=g['base'],alpha=g['alpha'],**r) for g in result['grid'] for r in g['rows']),key=lambda r:r['hits'])
    assert single == result['best_single']
    assert result['mean_target_exceeded'] == (best['mean']>.7129)
    assert result['single_target_exceeded'] == (single['hits']>.7129)
    assert abs(result['baseline']['validation2018']-.680243)<5e-7
    receipt = dict(valid=True,classification='test_tuned_development',test_scored=True,
        complete_grid_cells=len(result['grid'])*3, all_metrics_exact=True,
        hashes_verified=True, official_AA_reproduction_verified=True,
        best_common_percent=best['mean']*100,best_single_percent=single['hits']*100,
        source_results_sha256=sha(e/'results.json'),auditor_sha256=sha(Path(__file__)),
        qualification='Score audit independent of producer; graph chronology and checkpoint bytes checked on Tucker.')
    (e/'audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
