"""Summarize the frozen NM K/V experiment; retain selected-case denominators."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(root):
    receipt = json.loads((root / 'receipt.json').read_text())
    path = root / 'results_private.csv'
    assert receipt['complete'] and receipt['unchanged_model']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['csv_sha256']
    d = pd.read_csv(path)
    assert not d.duplicated(['case', 'method', 'draw', 'condition']).any()
    expected = {'original', 'full', 'keys', 'values', 'joint'}
    if receipt.get('value_factorial'):
        expected |= {'value_norms', 'value_directions'}
    assert set(d.condition) == expected
    for column in ['native_correct', 'original_correct', 'persistent']:
        assert d[column].isin([0, 1, False, True]).all()
        d[column] = d[column].astype(bool)
    assert np.isfinite(d.select_dtypes('number')).all().all()
    out = dict(receipt=receipt, methods={})
    for method, m in d.groupby('method'):
        baseline = m[m.condition.eq('original')].set_index(['case', 'draw'])
        full = m[m.condition.eq('full')].set_index(['case', 'draw'])
        joint = m[m.condition.eq('joint')].set_index(['case', 'draw'])
        assert len(baseline) == len(full) == len(joint)
        assert full.native_correct.equals(joint.native_correct)
        assert np.allclose(full.native_probability, joint.native_probability, atol=1e-5, rtol=0)
        values = {}
        for cohort, mask in [('all', np.ones(len(m), dtype=bool)),
                              ('native_failed', m.cohort.eq('native_failed')),
                              ('native_correct', m.cohort.eq('native_correct')),
                              ('persistent', m.persistent)]:
            frame = m[mask]
            section = {}
            for condition, s in frame.groupby('condition'):
                s = s.set_index(['case', 'draw'])
                a, b = baseline.loc[s.index], full.loc[s.index]
                delta = s.native_nll - a.native_nll
                section[condition] = dict(
                    cases=int(s.index.get_level_values('case').nunique()), draws=len(s),
                    correct=int(s.native_correct.sum()), accuracy=float(s.native_correct.mean()),
                    mean_nll=float(s.native_nll.mean()), mean_nll_change=float(delta.mean()),
                    full_outcome_agreement=int(s.native_correct.eq(b.native_correct).sum()),
                    mean_abs_nll_difference_from_full=float((s.native_nll - b.native_nll).abs().mean()),
                    mean_true_positive_attention=float(s.true_positive_attention.mean()),
                    mean_true_negative_attention=float(s.true_negative_attention.mean()),
                    mean_true_label_norm=float(s.true_label_norm.mean()),
                    true_score_terms={column.removeprefix('true_score_'): float(s[column].mean())
                        for column in s if column.startswith('true_score_')},
                )
                flipped = a.native_correct.ne(b.native_correct)
                section[condition]['full_flip_cases'] = int(flipped.sum())
                section[condition]['reproduces_full_flips'] = int(s.loc[flipped, 'native_correct'].eq(b.loc[flipped, 'native_correct']).sum())
                if 'support_value_norm_mean' in s:
                    section[condition]['mean_support_value_norm'] = float(s.support_value_norm_mean.mean())
            values[cohort] = section
        out['methods'][method] = values
    out['parity_summary'] = {
        'checks': len(receipt['parity']),
        'prediction_differences': sum(v['prediction_difference'] for v in receipt['parity']),
        'correctness_differences': sum(v['correctness_difference'] for v in receipt['parity']),
        'max_probability_error': max(v['probability_error'] for v in receipt['parity']),
        'max_winner_shortfall': max(v['winner_shortfall'] for v in receipt['parity']),
    }
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.input_root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for method, m in result['methods'].items():
        for cohort in ['native_failed', 'native_correct', 'persistent']:
            print(method, cohort, {k: (v['correct'], v['draws'], round(v['mean_nll'], 4), round(v['mean_true_positive_attention'], 4)) for k, v in m[cohort].items()})
