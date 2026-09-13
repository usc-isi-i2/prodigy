"""Validate and summarize full-canonical two-source NM stage accounting.

Only aggregates are written to the repository. Private rows and model-input
caches remain in the supplied external input directories. This is descriptive
stage accounting, not an estimate of causal mediation or information loss.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cross(a, b):
    return {f'{i}_{j}': int(((a == i) & (b == j)).sum())
            for i in [0, 1] for j in [0, 1]}


def summary(d):
    out = dict(rows=len(d), queries=int(d['query'].nunique()))
    for head in ['encoded', 'prototype', 'native']:
        out[head] = dict(accuracy=float(d[head+'_correct'].mean()),
            node_weighted_accuracy=float(d.groupby('query')[head+'_correct'].mean().mean()),
            mean_rank=float(d[head+'_rank'].mean()),
            median_rank=float(d[head+'_rank'].median()),
            mean_margin=float(d[head+'_margin'].mean()))
    out['encoded_to_native'] = cross(d.encoded_correct, d.native_correct)
    out['prototype_to_native'] = cross(d.prototype_correct, d.native_correct)
    out['native_nll'] = float(d.native_nll.mean())
    out['replay_native_accuracy'] = float(d.replay_native_correct.mean())
    out['replay_correctness_differences'] = int(d.native_correct.ne(d.replay_native_correct).sum())
    out['native_errors'] = int((~d.native_correct).sum())
    out['encoded_correct_fraction_of_native_errors'] = (
        out['encoded_to_native']['1_0']/out['native_errors'] if out['native_errors'] else None)
    return out


def aggregate(root, geometry_root, reference_root):
    receipt = json.loads((root/'receipt.json').read_text())
    geometry_receipt = json.loads((geometry_root/'receipt.json').read_text())
    geometry_aggregate = json.loads((Path(__file__).parent/'data/canonical_split/nm_complete_input_geometry.json').read_text())
    assert receipt['complete'] and receipt['model_states_unchanged']
    provenance = json.loads((Path(__file__).parent/'data/canonical_split/checkpoint_provenance.json').read_text())
    for short, name in [('ukr', 'ukraine'), ('hk', 'hongkong')]:
        assert receipt['checkpoints'][short]['sha256'] == provenance['checkpoints'][name]['sha256']
    result = dict(protocol='canonical_nm_source_stage_accounting_v1', receipt=receipt,
        geometry_receipt_sha256=digest(geometry_root/'receipt.json'), targets={})
    for target, native, foreign in [('ukr_rus', 'ukr', 'hk'), ('cp_hk', 'hk', 'ukr')]:
        info = receipt['targets'][target]
        path = root/target/'stage_predictions_private.csv'
        assert digest(path) == info['csv_sha256']
        d = pd.read_csv(path)
        assert len(d) == 122880 and not d.duplicated(['episode', 'sample', 'model']).any()
        assert set(d.model) == {'ukr', 'hk'}
        for head in ['encoded', 'prototype', 'native']:
            assert d[head+'_correct'].isin([0, 1, False, True]).all()
            d[head+'_correct'] = d[head+'_correct'].astype(bool)
            assert d[head+'_rank'].between(1, 30).all()
        assert np.isfinite(d.select_dtypes('number')).all().all()
        gpath = geometry_root/target/'query_geometry_private.csv'
        # The runtime's input_table_sha256 names its INPUT prediction TSV,
        # whereas the later aggregate records the OUTPUT geometry CSV hash.
        assert digest(gpath) == geometry_aggregate['targets'][target]['input_sha256']
        assert geometry_receipt['targets'][target]['input_table_sha256'] == info['reference_sha256']
        assert info['full_input_sha256'] == geometry_receipt['targets'][target]['expected_hash']
        g = pd.read_csv(gpath)
        assert len(g) == 61440 and not g.duplicated(['episode', 'sample']).any()
        reference = pd.read_csv(reference_root/(native+'.tsv'), sep='\t')
        assert digest(reference_root/(native+'.tsv')) == info['reference_sha256']
        reference = reference[reference.split == 'test']
        assert len(reference) == 61440 and not reference.duplicated(['episode', 'sample']).any()
        g = g.merge(reference[['episode', 'sample', 'query', 'anchor']],
            on=['episode', 'sample'], validate='one_to_one', suffixes=('', '_ref'))
        assert g['query'].equals(g.query_ref) and g.anchor.equals(g.anchor_ref)
        g['frequency'] = g.groupby('query')['query'].transform('size')
        g['ambiguity'] = g.groupby(['episode', 'query']).anchor.transform('nunique') > 1
        g['degree_group'] = pd.cut(g.query_degree, [-1, 9, 99, 999, 9999, 99999, np.inf],
            labels=['0–9', '10–99', '100–999', '1k–9,999', '10k–99,999', '100k+'])
        g['frequency_group'] = pd.cut(g.frequency, [0, 1, 4, 19, np.inf], labels=['1', '2–4', '5–19', '20+'])
        g['raw_group'] = np.where(~g.full_mean_all_valid, 'undefined',
            np.where(g.full_mean_margin > 1e-7, 'true_closest',
            np.where(g.full_mean_margin < -1e-7, 'rival_closer', 'tied')))
        frames = {}
        out = dict(native=native, models={}, raw_groups=g.raw_group.value_counts().to_dict(),
            geometry_sha256=digest(gpath), full_population_queries=int(g['query'].nunique()))
        for model, s in d.groupby('model'):
            s = s.merge(g, on=['episode', 'sample', 'query', 'anchor'], validate='one_to_one')
            assert len(s) == 61440 and s.episode.nunique() == 512
            assert s.groupby('episode').size().eq(120).all()
            assert s.groupby(['episode', 'anchor']).size().eq(4).all()
            assert s.native_correct.eq(s[model+'_correct'].astype(bool)).all()
            assert s.native_correct.mean() == info['native_accuracies'][model]
            assert int(s.native_correct.ne(s.replay_native_correct).sum()) == info['correctness_mismatches'][model]
            assert int(s.native_prediction.ne(s.replay_native_prediction).sum()) == info['prediction_mismatches'][model]
            out['models'][model] = dict(all=summary(s),
                raw_valid=summary(s[s.full_mean_all_valid]),
                single_anchor=summary(s[~s.ambiguity]),
                raw_groups={str(k): summary(v) for k, v in s.groupby('raw_group', observed=True)},
                degree_groups={str(k): summary(v) for k, v in s.groupby('degree_group', observed=True)},
                frequency_groups={str(k): summary(v) for k, v in s.groupby('frequency_group', observed=True)})
            # These correlations compare the SAME rows, not different feature-availability populations.
            valid = s[s.full_mean_all_valid]
            out['models'][model]['raw_encoded_spearman'] = float(valid.full_mean_margin.corr(valid.encoded_margin, method='spearman'))
            frames[model] = s
        keys = ['episode', 'sample', 'query', 'anchor']
        metrics = [h+'_'+m for h in ['encoded', 'prototype', 'native'] for m in ['correct', 'margin', 'rank']]
        paired = frames[native][keys+metrics+['raw_group', 'full_mean_all_valid', 'ambiguity']].merge(
            frames[foreign][keys+metrics], on=keys, validate='one_to_one', suffixes=('_native_source', '_foreign_source'))
        def paired_summary(s):
            x = dict(rows=len(s), queries=int(s['query'].nunique()))
            for head in ['encoded', 'prototype', 'native']:
                a, b = s[head+'_correct_native_source'], s[head+'_correct_foreign_source']
                x[head] = dict(native_source_accuracy=float(a.mean()), foreign_source_accuracy=float(b.mean()),
                    native_minus_foreign=float(a.mean()-b.mean()), paired_outcomes=cross(a, b))
            return x
        out['paired'] = paired_summary(paired)
        out['paired_raw_valid'] = paired_summary(paired[paired.full_mean_all_valid])
        out['paired_single_anchor'] = paired_summary(paired[~paired.ambiguity])
        out['paired_by_raw'] = {str(k): paired_summary(v) for k, v in paired.groupby('raw_group')}
        out['paired_by_encoded_outcome'] = {}
        for a in [False, True]:
            for b in [False, True]:
                s = paired[(paired.encoded_correct_native_source == a) & (paired.encoded_correct_foreign_source == b)]
                out['paired_by_encoded_outcome'][f'{int(a)}_{int(b)}'] = paired_summary(s)
        final_native = paired.native_correct_native_source
        final_foreign = paired.native_correct_foreign_source
        out['final_disagreements'] = {}
        for name, mask in [('native_only', final_native & ~final_foreign),
                           ('foreign_only', ~final_native & final_foreign),
                           ('both_wrong', ~final_native & ~final_foreign),
                           ('both_correct', final_native & final_foreign)]:
            s = paired[mask]
            out['final_disagreements'][name] = dict(rows=len(s),
                encoded_paired_outcomes=cross(s.encoded_correct_native_source, s.encoded_correct_foreign_source),
                raw_groups=s.raw_group.value_counts().to_dict())
        result['targets'][target] = out
    return result


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    colors = {'ukr': '#236494', 'hk': '#b94a32'}
    for ax, (target, title) in zip(axes, [('ukr_rus', 'Ukraine target'), ('cp_hk', 'Hong Kong target')]):
        data = result['targets'][target]
        for model, name in [('ukr', 'Ukraine-trained'), ('hk', 'HK-trained')]:
            values = [100*data['models'][model]['all'][h]['accuracy'] for h in ['encoded', 'prototype', 'native']]
            x = np.arange(3)+(-.16 if model == 'ukr' else .16)
            ax.bar(x, values, width=.30, color=colors[model], label=name)
            for xx, yy in zip(x, values):
                ax.text(xx, yy+.6, f'{yy:.1f}', ha='center', fontsize=10)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(3), ['Encoded\nmean cosine', 'Encoded\nprototype', 'Final\nmetagraph'])
        ax.set_ylim(0, 65)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Accuracy (%)')
    axes[1].legend(frameon=False, loc='upper left')
    fig.suptitle('Same canonical inputs: source differences before and after the metagraph', fontsize=14)
    fig.text(.5, .015, '61,440 query occurrences per target · fixed heads · seed-0 checkpoints · descriptive comparison', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .95])
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--geometry-root', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure', type=Path)
    a = p.parse_args()
    result = aggregate(a.input_root, a.geometry_root, a.reference_root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if a.figure:
        a.figure.parent.mkdir(parents=True, exist_ok=True)
        plot(result, a.figure)
    for target, values in result['targets'].items():
        print(target, json.dumps(values['paired'], indent=2))
