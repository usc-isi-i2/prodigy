"""Independent NumPy audit of saved assessment scores; no model or test access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        value = np.ascontiguousarray(a)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    results = read(args.evidence / 'results.json')
    frozen = read(args.evidence / 'selection_frozen.json')
    assert results['complete'] and results['test_scored'] is False
    assert sha(args.runtime / 'scores.npz') == results['score_sha256']
    assert sha(args.runtime / 'year2018.npz') == results['panel_sha256']
    assert sha(args.evidence / 'selection_frozen.json') == results['selection_frozen_sha256']
    panel = np.load(args.runtime / 'year2018.npz')
    predictions = np.load(args.runtime / 'scores.npz')
    assert len(panel['pos']) == 60084 and len(panel['neg']) == 100000
    assert fingerprint(panel['pos'], panel['neg']) == results['metadata']['pairs_fingerprint']
    assert fingerprint(panel['pfeatures'], panel['nfeatures']) == results['metadata']['features_fingerprint']
    threshold = np.sort(panel['official_bn'])[-50]
    baseline = panel['official_bp'] > threshold
    assert baseline.mean() == results['official_aadc_hits_at_50']
    repeat = panel['pfeatures'][:, 8] > 0
    expected = {(a, s) for a in ('structure', 'joint') for s in (0, 1, 2)}
    assert {(r['arm'], r['seed']) for r in results['rows']} == expected and len(results['rows']) == 6
    checked = []
    for row in results['rows']:
        name = f"{row['arm']}_seed{row['seed']}"
        positive, negative = predictions[name+'_p'], predictions[name+'_n']
        for score, edges in [(positive, panel['pos']), (negative, panel['neg'])]:
            assert len(score) == len(edges) and np.isfinite(score).all()
            assert np.all(score[edges[:, 0] == edges[:, 1]] == -1e9)
        cutoff = np.sort(negative)[-50]
        hits = positive > cutoff
        assert hits.mean() == row['hits_at_50'] and cutoff == row['negative50']
        assert np.count_nonzero(hits & ~baseline) == row['recovered']
        assert np.count_nonzero(~hits & baseline) == row['lost']
        assert int(hits.sum())-int(baseline.sum()) == row['net_hits']
        assert int(hits[repeat].sum())-int(baseline[repeat].sum()) == row['repeat_net_hits']
        assert int(hits[~repeat].sum())-int(baseline[~repeat].sum()) == row['new_pair_net_hits']
        assert np.count_nonzero(hits[panel['bp'] == 0]) == row['zero_base_hits']
        selection = next(s for s in frozen['selections'] if s['arm']==row['arm'] and s['seed']==row['seed'])
        assert selection['total_inference_learned_scalars'] < 1_000_000
        earliest = min(selection['history'], key=lambda h: (-h['selection_hits_at_50'], h['step']))
        assert earliest['step'] == row['selected_step'] == selection['selected_step']
        checked.append(name)
    for arm in ('structure', 'joint'):
        values = [r['hits_at_50'] for r in results['rows'] if r['arm'] == arm]
        assert results['summary'][arm] == {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1))}
    paired = [next(r['hits_at_50'] for r in results['rows'] if r['arm']=='joint' and r['seed']==s) -
              next(r['hits_at_50'] for r in results['rows'] if r['arm']=='structure' and r['seed']==s) for s in (0, 1, 2)]
    passed = min(paired)>0 and results['summary']['joint']['mean'] >= baseline.mean()+.01
    assert paired == results['paired_joint_minus_structure']
    assert bool(passed) == results['advancement_rule_passed']
    output = dict(complete=True, expected_cells=6, checked=checked, test_scored=False,
                  source_revision=results['revision'], source_results_sha256=sha(args.evidence / 'results.json'),
                  auditor_sha256=sha(Path(__file__)), raw_score_hashes_verified=True,
                  official_pairs_and_original_features_verified=True, all_metrics_recomputed=True,
                  self_pair_rule_verified=True, selection_rule_verified=True,
                  advancement_rule_passed=bool(passed),
                  qualification='Checkpoint-byte validation is in the separate Tucker audit; this independently audits scores.')
    args.out.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
